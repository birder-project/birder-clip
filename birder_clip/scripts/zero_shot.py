import argparse
import logging
import time
from pathlib import Path
from typing import Any
from typing import Optional
from typing import get_args

import numpy as np
import numpy.typing as npt
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from birder.common import cli
from birder.common import lib as birder_lib
from birder.common.fs_ops import read_class_file
from birder.conf import settings as birder_settings
from birder.data.datasets.directory import ImageLoaderName
from birder.data.datasets.directory import get_image_loader
from birder.data.datasets.directory import make_image_dataset
from birder.data.transforms.classification import inference_preset
from torch.utils.data import DataLoader

from birder_clip.common import fs_ops
from birder_clip.common import lib
from birder_clip.conf import settings  # pylint: disable=unused-import  # noqa: F401
from birder_clip.inference.zero_shot import build_class_text_embeddings
from birder_clip.inference.zero_shot_templates import TEMPLATE_SETS
from birder_clip.model_registry import Task
from birder_clip.model_registry import registry
from birder_clip.tokenizers import get_tokenizer

logger = logging.getLogger(__name__)


def _validate_label_names(label_names: list[str]) -> None:
    collisions = sorted(set(label_names).intersection({"sample", "label"}))
    if len(collisions) > 0:
        collision_str = ", ".join(f"'{name}'" for name in collisions)
        raise ValueError(
            f"Class names cannot use reserved metadata column names: {collision_str}. "
            "Rename the colliding class labels."
        )


def _metadata_pa_fields(columns: list[str]) -> list[pa.Field]:
    return [pa.field(name, pa.int64()) if name == "label" else pa.field(name, pa.string()) for name in columns]


def _init_parquet_writer(metadata_columns: list[str], label_names: list[str], path: Path) -> pq.ParquetWriter:
    schema = pa.schema(_metadata_pa_fields(metadata_columns) + [pa.field(name, pa.float32()) for name in label_names])
    return pq.ParquetWriter(path, schema)


def save_logits_parquet(
    writer: pq.ParquetWriter,
    sample_paths: list[str],
    label_names: list[str],
    logits: npt.NDArray[np.float32],
    labels: Optional[npt.NDArray[np.int64]] = None,
) -> None:
    logger.info(f"Writing logits at {writer.where}")
    data = {"sample": pa.array(sample_paths, type=pa.string())}
    if labels is not None:
        data["label"] = pa.array(labels, type=pa.int64())

    for i, label_name in enumerate(label_names):
        data[label_name] = pa.array(logits[:, i], type=pa.float32())

    table = pa.Table.from_pydict(data, schema=writer.schema)
    writer.write_table(table)


def save_output_parquet(
    writer: pq.ParquetWriter,
    sample_paths: list[str],
    label_names: list[str],
    outs: npt.NDArray[np.float32],
    labels: Optional[npt.NDArray[np.int64]] = None,
) -> None:
    logger.info(f"Writing outputs at {writer.where}")
    data = {"sample": pa.array(sample_paths, type=pa.string())}
    if labels is not None:
        data["label"] = pa.array(labels, type=pa.int64())

    data["prediction"] = pa.array(np.array(label_names)[outs.argmax(axis=1)], type=pa.string())
    for i, label_name in enumerate(label_names):
        data[label_name] = pa.array(outs[:, i], type=pa.float32())

    table = pa.Table.from_pydict(data, schema=writer.schema)
    writer.write_table(table)


def _save_dataframe(path: Path, df: pl.DataFrame, append: bool, data_type: str) -> None:
    if append is False:
        logger.info(f"Saving {data_type} at {path}")
        df.write_csv(path)
    else:
        logger.info(f"Adding {data_type} to {path}")
        with open(path, "a", encoding="utf-8") as handle:
            df.write_csv(handle, include_header=False)


def save_logits(
    path: Path,
    sample_paths: list[str],
    label_names: list[str],
    logits: npt.NDArray[np.float32],
    append: bool,
    labels: Optional[npt.NDArray[np.int64]] = None,
) -> None:
    data: dict[str, Any] = {"sample": sample_paths}
    if labels is not None:
        data["label"] = labels

    data.update({name: logits[:, i] for i, name in enumerate(label_names)})
    logits_df = pl.DataFrame(data)
    logits_df = logits_df.sort("sample", descending=False)
    _save_dataframe(path, logits_df, append, data_type="logits")


def save_output(
    path: Path,
    sample_paths: list[str],
    label_names: list[str],
    outs: npt.NDArray[np.float32],
    append: bool,
    labels: Optional[npt.NDArray[np.int64]] = None,
) -> None:
    data: dict[str, Any] = {"sample": sample_paths}
    if labels is not None:
        data["label"] = labels

    data["prediction"] = np.array(label_names)[outs.argmax(axis=1)]
    data.update({name: outs[:, i] for i, name in enumerate(label_names)})
    output_df = pl.DataFrame(data)
    output_df = output_df.sort("sample", descending=False)
    _save_dataframe(path, output_df, append, data_type="output")


def get_class_to_idx(args: argparse.Namespace) -> dict[str, int]:
    if args.class_file is not None:
        return read_class_file(args.class_file)

    return {class_name: idx for idx, class_name in enumerate(args.classes)}


def get_templates(args: argparse.Namespace) -> list[str]:
    if args.template is not None:
        return args.template  # type: ignore[no-any-return]
    if args.template_file is not None:
        with open(args.template_file, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if len(line.strip()) > 0]

    return list(TEMPLATE_SETS[args.template_set])


def predict(args: argparse.Namespace) -> None:
    if args.gpu is True:
        device = torch.device("cuda")
    elif args.mps is True:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

    logger.info(f"Using device {device}")

    model_dtype: torch.dtype = getattr(torch, args.model_dtype)
    if args.amp_dtype is None:
        amp_dtype = torch.get_autocast_dtype(device.type)
        logger.debug(f"AMP: {args.amp}, AMP dtype: {amp_dtype}")
    else:
        amp_dtype = getattr(torch, args.amp_dtype)

    if args.fast_matmul is True or args.amp is True:
        torch.set_float32_matmul_precision("high")

    net, model_info = fs_ops.load_model(
        device,
        args.network,
        config=args.model_config,
        tag=args.tag,
        image_encoder=args.image_encoder,
        text_encoder=args.text_encoder,
        embed_dim=args.embed_dim,
        tokenizer=args.tokenizer,
        image_encoder_config=args.image_encoder_config,
        text_encoder_config=args.text_encoder_config,
        epoch=args.epoch,
        new_size=args.size,
        inference=True,
        dtype=model_dtype,
    )

    if args.tokenizer is not None:
        tokenizer_name = args.tokenizer
    else:
        tokenizer_name = net.tokenizer_name

    tokenizer = get_tokenizer(tokenizer_name)

    logger.debug(f"Model loaded with signature: {model_info.signature}")
    logger.debug(f"RGB stats: {model_info.rgb_stats}")
    logger.debug(f"Using tokenizer: {tokenizer_name}")

    if args.compile is True:
        net = torch.compile(net, mode=args.compile_mode)
        net.encode_image = torch.compile(net.encode_image, mode=args.compile_mode)
        net.encode_text = torch.compile(net.encode_text, mode=args.compile_mode)
        net.forward_logits = torch.compile(net.forward_logits, mode=args.compile_mode)

    class_to_idx = get_class_to_idx(args)
    class_names = list(class_to_idx.keys())
    templates = get_templates(args)
    if args.size is None:
        args.size = lib.get_size_from_signature(model_info.signature)
        logger.debug(f"Using size={args.size}")

    input_channels = lib.get_channels_from_signature(model_info.signature)
    inference_transform = inference_preset(args.size, model_info.rgb_stats, args.center_crop, args.simple_crop)
    loader = get_image_loader(args.img_loader, input_channels)
    dataset = make_image_dataset(args.data_path, class_to_idx, transforms=inference_transform, loader=loader)
    num_samples = len(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    text_embeddings = build_class_text_embeddings(
        net,
        tokenizer,
        class_names,
        templates,
        device=device,
        batch_size=args.text_batch_size,
        amp=args.amp,
        amp_dtype=amp_dtype,
    )

    top_k = min(args.top_k, len(class_names))
    network_name = lib.get_image_text_network_name(
        args.network,
        tag=args.tag,
        image_encoder=args.image_encoder,
        text_encoder=args.text_encoder,
        embed_dim=args.embed_dim,
        tokenizer=args.tokenizer,
    )
    epoch_str = ""
    if args.epoch is not None:
        epoch_str = f"_e{args.epoch}"

    base_output_path = (
        f"{network_name}_{len(class_to_idx)}{epoch_str}_{args.size[0]}px_crop{args.center_crop}_{num_samples}"
    )
    if args.simple_crop is True:
        base_output_path = f"{base_output_path}_sc"
    if args.model_dtype != "float32":
        base_output_path = f"{base_output_path}_{args.model_dtype}"
    if args.prefix is not None:
        base_output_path = f"{args.prefix}_{base_output_path}"
    if args.suffix is not None:
        base_output_path = f"{base_output_path}_{args.suffix}"

    logits_path = birder_settings.RESULTS_DIR.joinpath(f"{base_output_path}_logits.{args.output_format}")
    output_path = birder_settings.RESULTS_DIR.joinpath(f"{base_output_path}_output.{args.output_format}")
    if args.save_logits is True or args.save_output is True:
        _validate_label_names(class_names)

    logits_writer: Optional[pq.ParquetWriter] = None
    output_writer: Optional[pq.ParquetWriter] = None
    metadata_columns = ["sample"]
    if args.save_labels is True:
        metadata_columns.append("label")

    append = False
    tic = time.time()
    with torch.inference_mode():
        for paths, images, labels in dataloader:
            images = images.to(device, dtype=model_dtype, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=args.amp, dtype=amp_dtype):
                image_embeddings = net.encode_image(images, normalize=True)
                logits = net.forward_logits(image_embeddings, text_embeddings)
                prob = F.softmax(logits, dim=-1)

            sample_paths = list(paths)
            labels_to_save = labels.numpy() if args.save_labels is True else None
            if args.save_logits is True:
                logits_np = logits.float().cpu().numpy()
                if args.output_format == "parquet":
                    if logits_writer is None:
                        logits_writer = _init_parquet_writer(metadata_columns, class_names, logits_path)

                    save_logits_parquet(logits_writer, sample_paths, class_names, logits_np, labels=labels_to_save)
                else:
                    save_logits(logits_path, sample_paths, class_names, logits_np, append=append, labels=labels_to_save)

            elif args.save_output is True:
                prob_np = prob.float().cpu().numpy()
                if args.output_format == "parquet":
                    if output_writer is None:
                        output_writer = _init_parquet_writer(
                            metadata_columns + ["prediction"], class_names, output_path
                        )

                    save_output_parquet(output_writer, sample_paths, class_names, prob_np, labels=labels_to_save)
                else:
                    save_output(output_path, sample_paths, class_names, prob_np, append=append, labels=labels_to_save)

            values, indices = prob.topk(top_k, dim=-1)

            for path, sample_values, sample_indices in zip(paths, values, indices):
                predictions = [
                    f"{class_names[idx.item()]}: {value.item():.4f}"
                    for value, idx in zip(sample_values, sample_indices)
                ]
                logger.info(f"{path}: {', '.join(predictions)}")

            append = True

    if logits_writer is not None:
        logits_writer.close()
    if output_writer is not None:
        output_writer.close()

    toc = time.time()
    rate = num_samples / (toc - tic)
    logger.info(f"{birder_lib.format_duration(toc-tic)} to classify {num_samples:,} samples ({rate:.2f} samples/sec)")


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Run zero-shot image classification",
        epilog=(
            "Usage example:\n"
            "python -m birder_clip.scripts.zero_shot --network openai_clip_vit_l14 --classes eagle hawk falcon --gpu "
            "--template-set identity data/validation_il-common_packed\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    parser.add_argument("-n", "--network", type=str, help="the image-text network to use")
    parser.add_argument("--image-encoder", type=str, help="the image encoder to use")
    parser.add_argument("--text-encoder", type=str, help="the text encoder to use")
    parser.add_argument("--embed-dim", type=int, metavar="N", help="shared image-text embedding dimension")
    parser.add_argument("--tokenizer", type=str, help="the tokenizer to use")
    parser.add_argument(
        "--image-encoder-config",
        action=cli.FlexibleDictAction,
        help="override the image encoder configuration, accepts key-value pairs or JSON",
    )
    parser.add_argument(
        "--text-encoder-config",
        action=cli.FlexibleDictAction,
        help="override the text encoder configuration, accepts key-value pairs or JSON",
    )
    parser.add_argument("-e", "--epoch", type=int, metavar="N", help="model checkpoint to load")
    parser.add_argument("-t", "--tag", type=str, help="model tag")
    parser.add_argument("--class-file", type=str, help="Birder class file, one class per line")
    parser.add_argument("--classes", type=str, nargs="*", help="class names to use for zero-shot classification")
    parser.add_argument(
        "--model-config",
        action=cli.FlexibleDictAction,
        help="override the model default configuration, accepts key-value pairs or JSON",
    )
    parser.add_argument("--compile", default=False, action="store_true", help="enable compilation")
    parser.add_argument(
        "--compile-mode", type=str, choices=list(torch._inductor.list_mode_options().keys()), help="torch.compile mode"
    )
    parser.add_argument(
        "--template-set",
        choices=sorted(TEMPLATE_SETS.keys()),
        default="default",
        help="built-in zero-shot template set",
    )
    parser.add_argument("--template", action="append", help="additional prompt template")
    parser.add_argument("--template-file", type=str, help="file with one prompt template per line")
    parser.add_argument(
        "--top-k", type=int, default=birder_settings.TOP_K, metavar="N", help="number of predictions to print"
    )
    parser.add_argument("--batch-size", type=int, default=32, metavar="N", help="the batch size")
    parser.add_argument("--text-batch-size", type=int, metavar="N", help="class batch size for text encoding")
    parser.add_argument(
        "--size", type=int, nargs="+", metavar=("H", "W"), help="image size for inference (defaults to model signature)"
    )
    parser.add_argument(
        "--img-loader",
        type=str,
        choices=get_args(ImageLoaderName),
        default="tv",
        help="backend to load and decode images",
    )
    parser.add_argument("-j", "--num-workers", type=int, default=8, metavar="N", help="number of preprocessing workers")
    parser.add_argument(
        "--prefetch-factor", type=int, metavar="N", help="number of batches loaded in advance by each worker"
    )
    parser.add_argument("--center-crop", type=float, default=1.0, help="center crop ratio to use during inference")
    parser.add_argument(
        "--simple-crop",
        default=False,
        action="store_true",
        help="use a simple crop that preserves aspect ratio but may trim parts of the image",
    )
    parser.add_argument("--save-output", default=False, action="store_true", help="save raw output")
    parser.add_argument("--save-logits", default=False, action="store_true", help="save raw model logits")
    parser.add_argument(
        "--save-labels",
        default=False,
        action="store_true",
        help="include the label column in raw outputs (logits, output)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["csv", "parquet"],
        default="csv",
        help="output format for raw data (logits, output)",
    )
    parser.add_argument("--prefix", type=str, help="add prefix to output file")
    parser.add_argument("--suffix", type=str, help="add suffix to output file")
    parser.add_argument(
        "--model-dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="model dtype to use",
    )
    parser.add_argument(
        "--amp", default=False, action="store_true", help="use torch.amp.autocast for mixed precision inference"
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        choices=["float16", "bfloat16"],
        help="whether to use float16 or bfloat16 for mixed precision",
    )
    parser.add_argument(
        "--fast-matmul", default=False, action="store_true", help="use fast matrix multiplication (affects precision)"
    )
    parser.add_argument("--gpu", default=False, action="store_true", help="use gpu")
    parser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use")
    parser.add_argument("--mps", default=False, action="store_true", help="use mps (Metal Performance Shaders) device")
    parser.add_argument("data_path", nargs="*", help="data files path (directories and files)")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.size = cli.parse_size(args.size)
    if args.network is None:
        raise cli.ValidationError("--network is required")
    if registry.exists(args.network, task=Task.IMAGE_TEXT) is False:
        raise cli.ValidationError(f"--network {args.network} not supported, see list-models tool for available options")
    if args.compile_mode is not None and args.compile is False:
        raise cli.ValidationError("--compile-mode requires --compile")
    if args.class_file is None and args.classes is None:
        raise cli.ValidationError("--class-file or --classes is required")
    if args.class_file is not None and args.classes is not None:
        raise cli.ValidationError("--class-file and --classes cannot be used together")
    if args.classes is not None and len(args.classes) == 0:
        raise cli.ValidationError("--classes requires at least one class name")
    if args.template is not None and args.template_file is not None:
        raise cli.ValidationError("--template and --template-file cannot be used together")
    if args.template_set != "default" and args.template is not None:
        raise cli.ValidationError("--template-set and --template cannot be used together")
    if args.template_set != "default" and args.template_file is not None:
        raise cli.ValidationError("--template-set and --template-file cannot be used together")
    if args.amp is True and args.model_dtype != "float32":
        raise cli.ValidationError("--amp can only be used with --model-dtype float32")
    if args.save_logits is True and args.save_output is True:
        raise cli.ValidationError("--save-logits cannot be used with --save-output")
    if len(args.data_path) == 0:
        raise cli.ValidationError("data_path is required")
    if args.center_crop > 1 or args.center_crop <= 0.0:
        raise cli.ValidationError(f"--center-crop must be in range of (0, 1.0], got {args.center_crop}")


def args_from_dict(**kwargs: Any) -> argparse.Namespace:
    parser = get_args_parser()
    parser.set_defaults(**kwargs)
    args = parser.parse_args([])
    validate_args(args)

    return args


def main() -> None:
    parser = get_args_parser()
    args = parser.parse_args()
    validate_args(args)
    predict(args)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
