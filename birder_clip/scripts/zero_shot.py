import argparse
import logging
import time
from typing import Any
from typing import Optional
from typing import get_args

import numpy as np
import numpy.typing as npt
import polars as pl
import pyarrow.parquet as pq
import torch
from birder.common import cli
from birder.common import lib as birder_lib
from birder.common.fs_ops import read_class_file
from birder.conf import settings as birder_settings
from birder.data.dataloader.webdataset import make_wds_loader
from birder.data.datasets.directory import ImageLoaderName
from birder.data.datasets.directory import class_to_idx_from_paths
from birder.data.datasets.directory import get_image_loader
from birder.data.datasets.directory import make_image_dataset
from birder.data.datasets.webdataset import make_wds_dataset
from birder.data.datasets.webdataset import prepare_wds_args
from birder.data.datasets.webdataset import wds_args_from_info
from birder.data.transforms.classification import inference_preset
from birder.results.classification import Results
from birder.scripts.predict import _init_parquet_writer
from birder.scripts.predict import _sanitize_results_labels
from birder.scripts.predict import _validate_label_names
from birder.scripts.predict import handle_show_flags
from birder.scripts.predict import save_logits
from birder.scripts.predict import save_logits_parquet
from birder.scripts.predict import save_output
from birder.scripts.predict import save_output_parquet
from torch.utils.data import DataLoader

from birder_clip.common import fs_ops
from birder_clip.common import lib
from birder_clip.inference.zero_shot import build_class_text_embeddings
from birder_clip.inference.zero_shot import infer_dataloader_iter
from birder_clip.inference.zero_shot_templates import TEMPLATE_SETS
from birder_clip.model_registry import Task
from birder_clip.model_registry import registry
from birder_clip.tokenizers import get_tokenizer

logger = logging.getLogger(__name__)


def get_class_to_idx(args: argparse.Namespace) -> dict[str, int]:
    if args.class_file is not None:
        return read_class_file(args.class_file)
    if args.classes is not None:
        return {class_name: idx for idx, class_name in enumerate(args.classes)}

    return class_to_idx_from_paths(args.data_path, hierarchical=args.hierarchical)


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

    network_name = lib.get_image_text_network_name(
        args.network,
        tag=args.tag,
        image_encoder=args.image_encoder,
        text_encoder=args.text_encoder,
        embed_dim=args.embed_dim,
        tokenizer=args.tokenizer,
    )
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
        st=args.st,
        dtype=model_dtype,
    )
    logger.debug(f"Model loaded with signature: {model_info.signature}")
    logger.debug(f"RGB stats: {model_info.rgb_stats}")

    if args.tokenizer is not None:
        tokenizer_name = args.tokenizer
    else:
        tokenizer_name = net.tokenizer_name

    context_length = lib.get_context_length_from_signature(model_info.signature)
    tokenizer = get_tokenizer(tokenizer_name, context_length=context_length)
    logger.debug(f"Using tokenizer: {tokenizer_name}, context length={context_length}")

    class_to_idx = get_class_to_idx(args)
    class_names = list(class_to_idx.keys())

    if args.show_class is not None:
        if args.show_class not in class_to_idx:
            logger.warning("Selected 'show class' is not part of the model classes")

    if args.fast_matmul is True or args.amp is True:
        torch.set_float32_matmul_precision("high")

    if args.compile is True:
        net = torch.compile(net, mode=args.compile_mode)
        net.encode_image = torch.compile(net.encode_image, mode=args.compile_mode)
        net.encode_text = torch.compile(net.encode_text, mode=args.compile_mode)
        net.forward_logits = torch.compile(net.forward_logits, mode=args.compile_mode)

    if args.size is None:
        args.size = lib.get_size_from_signature(model_info.signature)
        logger.debug(f"Using size={args.size}")

    input_channels = lib.get_channels_from_signature(model_info.signature)
    batch_size = args.batch_size
    inference_transform = inference_preset(args.size, model_info.rgb_stats, args.center_crop, args.simple_crop)
    if args.wds is True:
        wds_path: str | list[str]
        if args.wds_info is not None:
            wds_path, dataset_size = wds_args_from_info(args.wds_info, args.wds_split)
            if args.wds_size is not None:
                dataset_size = args.wds_size
        else:
            wds_path, dataset_size = prepare_wds_args(args.data_path[0], args.wds_size, device)

        num_samples = dataset_size
        dataset = make_wds_dataset(
            wds_path,
            dataset_size=dataset_size,
            shuffle=args.shuffle,
            samples_names=True,
            transform=inference_transform,
            image_decoder=args.img_loader,
            channels=input_channels,
        )
        inference_loader = make_wds_loader(
            dataset,
            batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            collate_fn=None,
            world_size=1,
            pin_memory=False,
            exact=True,
        )

    else:
        loader = get_image_loader(args.img_loader, input_channels)
        dataset = make_image_dataset(
            args.data_path, class_to_idx, transforms=inference_transform, loader=loader, hierarchical=args.hierarchical
        )
        num_samples = len(dataset)
        inference_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )

    show_flag = (
        args.show is True
        or args.show_top_below is not None
        or args.show_target_below is not None
        or args.show_mistakes is True
        or args.show_out_of_k is True
        or args.show_class is not None
    )

    def batch_callback(
        file_paths: list[str], out: npt.NDArray[np.float32], batch_labels: npt.NDArray[np.int64]
    ) -> None:
        # Show flags
        if show_flag is True:
            for img_path, prob, label in zip(file_paths, out, batch_labels):
                handle_show_flags(args, img_path, prob, label, class_to_idx)

    # Sort out output file names
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

    results_path = f"{base_output_path}.csv"
    logits_path = birder_settings.RESULTS_DIR.joinpath(f"{base_output_path}_logits.{args.output_format}")
    output_path = birder_settings.RESULTS_DIR.joinpath(f"{base_output_path}_output.{args.output_format}")
    if args.save_logits is True or args.save_output is True:
        _validate_label_names(class_names)

    logits_writer: Optional[pq.ParquetWriter] = None
    output_writer: Optional[pq.ParquetWriter] = None
    metadata_columns = ["sample"]
    if args.save_labels is True:
        metadata_columns.append("label")

    # Inference
    templates = get_templates(args)
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
    tic = time.time()
    infer_iter = infer_dataloader_iter(
        device,
        net,
        inference_loader,
        text_embeddings,
        return_logits=args.save_logits,
        model_dtype=model_dtype,
        amp=args.amp,
        amp_dtype=amp_dtype,
        num_samples=num_samples,
        batch_callback=batch_callback,
        chunk_size=args.chunk_size,
    )
    append = False  # Append mode for raw outputs, only False for the first batch
    results_append = False
    summary_list = []
    with torch.inference_mode():
        for sample_paths, outs, labels in infer_iter:
            labels_to_save = labels if args.save_labels is True else None
            if args.save_logits is True:
                if args.output_format == "parquet":
                    if logits_writer is None:
                        logits_writer = _init_parquet_writer(metadata_columns, class_names, logits_path)

                    save_logits_parquet(logits_writer, sample_paths, class_names, outs, labels=labels_to_save)
                else:
                    save_logits(logits_path, sample_paths, class_names, outs, append=append, labels=labels_to_save)

            elif args.save_output is True:
                if args.output_format == "parquet":
                    if output_writer is None:
                        output_writer = _init_parquet_writer(
                            metadata_columns + ["prediction"], class_names, output_path
                        )

                    save_output_parquet(output_writer, sample_paths, class_names, outs, labels=labels_to_save)
                else:
                    save_output(output_path, sample_paths, class_names, outs, append=append, labels=labels_to_save)

            if args.save_logits is False and args.skip_results_analysis is False:
                # Handle results
                results_labels, num_invalid_results_labels = _sanitize_results_labels(labels, outs.shape[1])
                if num_invalid_results_labels > 0:
                    logger.warning(
                        f"Ignoring {num_invalid_results_labels} labels outside model output range "
                        f"[0, {outs.shape[1] - 1}] for results analysis"
                    )

                results = Results(sample_paths, results_labels, class_names, output=outs)
                if results.missing_all_labels is False:
                    if args.save_results is True:
                        results.save(results_path, append=results_append)
                        results_append = True
                    if args.chunk_size is None:
                        results.log_short_report()

                else:
                    logger.warning("No labeled samples found")

                # Summary
                if args.summary is True:
                    summary_list.append(results.prediction_names.value_counts())

            append = True

    if logits_writer is not None:
        logits_writer.close()
    if output_writer is not None:
        output_writer.close()

    toc = time.time()
    rate = num_samples / (toc - tic)
    logger.info(f"{birder_lib.format_duration(toc-tic)} to classify {num_samples:,} samples ({rate:.2f} samples/sec)")

    #  Print summary
    if args.summary is True:
        summary_df = pl.concat(summary_list).group_by("prediction_names").agg(pl.col("count").sum())
        summary_df = summary_df.sort(by="count", descending=True)
        indent_size = summary_df["prediction_names"].str.len_chars().max() + 2  # type: ignore[operator]
        for specie_name, count in summary_df.iter_rows():
            logger.info(f"{specie_name:<{indent_size}} {count}")


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Run zero-shot image classification",
        epilog=(
            "Usage example:\n"
            "python -m birder_clip.scripts.zero_shot --network openai_clip_vit_l14 --classes eagle hawk falcon --gpu "
            "--template-set identity data/validation_il-common_packed\n"
            "python -m birder_clip.scripts.zero_shot -n laion_clip_vit_h14 --gpu --amp "
            "--save-results ../birder/data/validation_il-common_packed\n"
            "python -m birder_clip.scripts.zero_shot -n siglip_v2_vit_so400m_p14 --template-set default --gpu "
            "--shuffle --show ../birder/data/validation_il-common_packed\n"
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
    parser.add_argument("--st", "--safetensors", default=False, action="store_true", help="load Safetensors weights")
    parser.add_argument("--class-file", type=str, help="Birder class file, one class per line")
    parser.add_argument(
        "--classes",
        type=str,
        nargs="*",
        help="class names to use for zero-shot classification (defaults to ImageFolder classes)",
    )
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
    parser.add_argument(
        "--chunk-size", type=int, metavar="N", help="process in chunks of N samples to reduce memory usage"
    )
    parser.add_argument("--center-crop", type=float, default=1.0, help="center crop ratio to use during inference")
    parser.add_argument(
        "--simple-crop",
        default=False,
        action="store_true",
        help="use a simple crop that preserves aspect ratio but may trim parts of the image",
    )
    parser.add_argument("--show", default=False, action="store_true", help="show image predictions")
    parser.add_argument("--show-top-below", type=float, help="show when top prediction is below given threshold")
    parser.add_argument("--show-target-below", type=float, help="show when target prediction is below given threshold")
    parser.add_argument("--show-mistakes", default=False, action="store_true", help="show only mis-classified images")
    parser.add_argument("--show-out-of-k", default=False, action="store_true", help="show images not in the top-k")
    parser.add_argument("--show-class", type=str, help="show specific class predictions")
    parser.add_argument("--shuffle", default=False, action="store_true", help="predict samples in random order")
    parser.add_argument("--summary", default=False, action="store_true", help="log prediction summary")
    parser.add_argument("--save-results", default=False, action="store_true", help="save results object")
    parser.add_argument(
        "--skip-results-analysis",
        default=False,
        action="store_true",
        help="skip results analysis and reporting even when labels are available",
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
        help="output format for raw data (logits, output) - does not affect results files",
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
    parser.add_argument("--wds", default=False, action="store_true", help="predict a webdataset directory")
    parser.add_argument("--wds-size", type=int, metavar="N", help="size of the wds dataset")
    parser.add_argument("--wds-info", type=str, action="append", metavar="FILE", help="one or more wds info file paths")
    parser.add_argument("--wds-split", type=str, default="validation", metavar="NAME", help="wds dataset split to load")
    parser.add_argument(
        "--hierarchical",
        default=False,
        action="store_true",
        help="use hierarchical directory structure for labels (e.g., 'dir1/subdir2' -> 'dir1_subdir2' label)",
    )
    parser.add_argument("data_path", nargs="*", help="data files path (directories and files)")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.size = cli.parse_size(args.size)
    show_flags_requested = (
        args.show is True
        or args.show_top_below is not None
        or args.show_target_below is not None
        or args.show_mistakes is True
        or args.show_out_of_k is True
        or args.show_class is not None
    )

    if args.network is None:
        raise cli.ValidationError("--network is required")
    if registry.exists(args.network, task=Task.IMAGE_TEXT) is False:
        raise cli.ValidationError(f"--network {args.network} not supported, see list-models tool for available options")
    if args.center_crop > 1 or args.center_crop <= 0.0:
        raise cli.ValidationError(f"--center-crop must be in range of (0, 1.0], got {args.center_crop}")

    reserved_model_config_keys = lib.get_reserved_model_config_keys(args.model_config)
    if len(reserved_model_config_keys) > 0:
        raise cli.ValidationError(
            "--model-config cannot contain keys with dedicated CLI flags: " f"{', '.join(reserved_model_config_keys)}"
        )
    if args.image_encoder_config is not None and "size" in args.image_encoder_config:
        raise cli.ValidationError("--image-encoder-config cannot contain size, use --size")
    if args.text_encoder_config is not None and "context_length" in args.text_encoder_config:
        raise cli.ValidationError("--text-encoder-config cannot contain context_length")

    if args.compile_mode is not None and args.compile is False:
        raise cli.ValidationError("--compile-mode requires --compile")
    if args.class_file is not None and args.classes is not None:
        raise cli.ValidationError("--class-file and --classes cannot be used together")
    if args.classes is not None and len(args.classes) == 0:
        raise cli.ValidationError("--classes requires at least one class name")
    if args.skip_results_analysis is True and args.save_results is True:
        raise cli.ValidationError("--skip-results-analysis cannot be used with --save-results")
    if args.skip_results_analysis is True and args.summary is True:
        raise cli.ValidationError("--skip-results-analysis cannot be used with --summary")
    if args.template is not None and args.template_file is not None:
        raise cli.ValidationError("--template and --template-file cannot be used together")
    if args.template_set != "default" and args.template is not None:
        raise cli.ValidationError("--template-set and --template cannot be used together")
    if args.template_set != "default" and args.template_file is not None:
        raise cli.ValidationError("--template-set and --template-file cannot be used together")
    if args.amp is True and args.model_dtype != "float32":
        raise cli.ValidationError("--amp can only be used with --model-dtype float32")
    if args.save_logits is True and args.save_results is True:
        raise cli.ValidationError("--save-logits cannot be used with --save-results")
    if args.save_logits is True and args.save_output is True:
        raise cli.ValidationError("--save-logits cannot be used with --save-output")
    if args.save_logits is True and args.summary is True:
        raise cli.ValidationError("--save-logits cannot be used with --summary")
    if args.save_logits is True and show_flags_requested is True:
        # Results are not calculated
        raise cli.ValidationError("--save-logits cannot be used with any of the 'show' flags")
    if args.wds is False and len(args.data_path) == 0:
        raise cli.ValidationError("Must provide at least one data source, --data-path or --wds")
    if args.wds is True:
        if args.class_file is None and args.classes is None:
            raise cli.ValidationError("--wds requires --class-file or --classes")
        if args.wds_info is None and len(args.data_path) == 0:
            raise cli.ValidationError("--wds requires a data path unless --wds-info is provided")
        if len(args.data_path) > 1:
            raise cli.ValidationError(f"--wds can have at most 1 --data-path, got {len(args.data_path)}")
        if args.wds_info is None and len(args.data_path) == 1:
            data_path = args.data_path[0]
            if "://" in data_path and args.wds_size is None:
                raise cli.ValidationError("--wds-size is required for remote --data-path")
    if args.wds is True and args.hierarchical is True:
        raise cli.ValidationError("--wds cannot be used with --hierarchical")
    if args.wds is True and show_flags_requested is True:
        raise cli.ValidationError("--wds cannot be used with any of the 'show' flags")


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

    if birder_settings.RESULTS_DIR.exists() is False:
        logger.info(f"Creating {birder_settings.RESULTS_DIR} directory...")
        birder_settings.RESULTS_DIR.mkdir(parents=True)

    predict(args)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
