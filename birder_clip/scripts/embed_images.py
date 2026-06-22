import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any
from typing import Optional
from typing import get_args

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from birder.common import cli
from birder.common import lib as birder_lib
from birder.conf import settings as birder_settings
from birder.data.dataloader.webdataset import make_wds_loader
from birder.data.datasets.directory import ImageLoaderName
from birder.data.datasets.directory import get_image_loader
from birder.data.datasets.directory import make_image_dataset
from birder.data.datasets.webdataset import make_wds_dataset
from birder.data.datasets.webdataset import prepare_wds_args
from birder.data.datasets.webdataset import wds_args_from_info
from birder.data.transforms.classification import inference_preset
from birder.inference.data_parallel import InferenceDataParallel
from torch.utils.data import DataLoader

from birder_clip.common import fs_ops
from birder_clip.common import lib
from birder_clip.inference.image_embeddings import infer_dataloader_iter
from birder_clip.model_registry import Task
from birder_clip.model_registry import registry
from birder_clip.version import __version__

logger = logging.getLogger(__name__)


def _metadata_pa_fields(columns: list[str]) -> list[pa.Field]:
    return [pa.field(name, pa.string()) for name in columns]


def _init_embeddings_parquet_writer(
    metadata_columns: list[str], embedding_size: int, metadata: dict[str, Any], path: Path
) -> pq.ParquetWriter:
    schema = pa.schema(
        _metadata_pa_fields(metadata_columns)
        + [pa.field("embedding", pa.list_(pa.float32(), list_size=embedding_size))]
    )
    schema = schema.with_metadata({"birder_clip.image_embeddings": json.dumps(metadata).encode("utf-8")})

    return pq.ParquetWriter(path, schema)


def save_embeddings_parquet(
    writer: pq.ParquetWriter,
    sample_paths: list[str],
    embeddings: npt.NDArray[np.float32],
) -> None:
    logger.info(f"Writing embeddings at {writer.where}")
    data = {"sample": pa.array(sample_paths, type=pa.string())}
    data["embedding"] = pa.array(embeddings.tolist(), type=writer.schema.field("embedding").type)
    table = pa.Table.from_pydict(data, schema=writer.schema)
    writer.write_table(table)


def embed(args: argparse.Namespace) -> None:
    if args.gpu is True:
        device = torch.device("cuda")
    elif args.mps is True:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.parallel is True and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} {device} devices")
    else:
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
        new_context_length=args.context_length,
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

    if args.context_length is not None:
        context_length = args.context_length
    else:
        context_length = lib.get_context_length_from_signature(model_info.signature)

    logger.debug(f"Using tokenizer: {tokenizer_name}, context length={context_length}")

    if args.fast_matmul is True or args.amp is True:
        torch.set_float32_matmul_precision("high")

    if args.parallel is True and torch.cuda.device_count() > 1:
        net = InferenceDataParallel(
            net,
            output_device="cpu",
            compile_replicas=args.compile,
            compile_methods=["encode_image"],
            compile_mode=args.compile_mode,
        )
    elif args.compile is True:
        net = torch.compile(net, mode=args.compile_mode)
        net.encode_image = torch.compile(net.encode_image, mode=args.compile_mode)

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
            wds_path, dataset_size = prepare_wds_args(
                args.data_path[0], args.wds_size, device, select_suffix=("jpg", "jpeg", "png", "webp")
            )

        num_samples = dataset_size
        dataset = make_wds_dataset(
            wds_path,
            dataset_size=dataset_size,
            shuffle=args.shuffle,
            samples_names=True,
            transform=inference_transform,
            image_decoder=args.img_loader,
            channels=input_channels,
            cls_key=None,
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
            args.data_path,
            {},
            transforms=inference_transform,
            loader=loader,
        )
        num_samples = len(dataset)
        inference_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )

    # Sort out output file names
    epoch_str = ""
    if args.epoch is not None:
        epoch_str = f"_e{args.epoch}"

    base_output_path = f"{network_name}{epoch_str}_{args.size[0]}px_crop{args.center_crop}_{num_samples}"
    if args.simple_crop is True:
        base_output_path = f"{base_output_path}_sc"
    if args.model_dtype != "float32":
        base_output_path = f"{base_output_path}_{args.model_dtype}"
    if args.prefix is not None:
        base_output_path = f"{args.prefix}_{base_output_path}"
    if args.suffix is not None:
        base_output_path = f"{base_output_path}_{args.suffix}"

    embeddings_path = birder_settings.RESULTS_DIR.joinpath(f"{base_output_path}_embeddings.parquet")

    embeddings_writer: Optional[pq.ParquetWriter] = None
    metadata_columns = ["sample"]

    parquet_metadata: dict[str, Any] = {
        "birder_clip_version": __version__,
        "network_name": network_name,
        "epoch": args.epoch,
        "size": args.size,
        "center_crop": args.center_crop,
        "simple_crop": args.simple_crop,
        "normalized": True,
    }

    # Inference
    tic = time.time()
    infer_iter = infer_dataloader_iter(
        device,
        net,
        inference_loader,
        model_dtype=model_dtype,
        amp=args.amp,
        amp_dtype=amp_dtype,
        num_samples=num_samples,
        chunk_size=args.chunk_size,
    )
    with torch.inference_mode():
        for sample_paths, embeddings in infer_iter:
            if embeddings_writer is None:
                embeddings_writer = _init_embeddings_parquet_writer(
                    metadata_columns, embeddings.shape[1], parquet_metadata, embeddings_path
                )

            save_embeddings_parquet(embeddings_writer, sample_paths, embeddings)

    if embeddings_writer is not None:
        embeddings_writer.close()

    toc = time.time()
    rate = num_samples / (toc - tic)
    logger.info(f"{birder_lib.format_duration(toc-tic)} to embed {num_samples:,} samples ({rate:.2f} samples/sec)")


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Generate image embeddings",
        epilog=(
            "Usage example:\n"
            "python -m birder_clip.scripts.embed_images --network laion_clip_vit_h14 --gpu "
            "data/validation_eu-common_packed\n"
            "python -m birder_clip.scripts.embed_images -n siglip_v2_vit_so400m_p14 --gpu "
            "--amp --amp-dtype bfloat16 --suffix raw_data data/raw_data\n"
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
    parser.add_argument(
        "--model-config",
        action=cli.FlexibleDictAction,
        help="override the model default configuration, accepts key-value pairs or JSON",
    )
    parser.add_argument("--compile", default=False, action="store_true", help="enable compilation")
    parser.add_argument(
        "--compile-mode", type=str, choices=list(torch._inductor.list_mode_options().keys()), help="torch.compile mode"
    )
    parser.add_argument("--batch-size", type=int, default=32, metavar="N", help="the batch size")
    parser.add_argument(
        "--size", type=int, nargs="+", metavar=("H", "W"), help="image size for inference (defaults to model signature)"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        metavar="N",
        help="text context length for inference (defaults to model signature)",
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
    parser.add_argument("--shuffle", default=False, action="store_true", help="embed samples in random order")
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
    parser.add_argument("--gpu-id", type=int, metavar="ID", help="gpu id to use (ignored in parallel mode)")
    parser.add_argument("--mps", default=False, action="store_true", help="use mps (Metal Performance Shaders) device")
    parser.add_argument("--parallel", default=False, action="store_true", help="use multiple gpus")
    parser.add_argument("--wds", default=False, action="store_true", help="embed a webdataset directory")
    parser.add_argument("--wds-size", type=int, metavar="N", help="size of the wds dataset")
    parser.add_argument("--wds-info", type=str, action="append", metavar="FILE", help="one or more wds info file paths")
    parser.add_argument("--wds-split", type=str, default="validation", metavar="NAME", help="wds dataset split to load")
    parser.add_argument("data_path", nargs="*", help="data files path (directories and files)")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.size = cli.parse_size(args.size)

    if args.network is None:
        raise cli.ValidationError("--network is required")
    if registry.exists(args.network, task=Task.IMAGE_TEXT) is False:
        raise cli.ValidationError(f"--network {args.network} not supported, see list-models tool for available options")
    if args.center_crop > 1 or args.center_crop <= 0.0:
        raise cli.ValidationError(f"--center-crop must be in range of (0, 1.0], got {args.center_crop}")
    if args.parallel is True and args.gpu is False:
        raise cli.ValidationError("--parallel requires --gpu to be set")
    if args.context_length is not None and args.context_length <= 0:
        raise cli.ValidationError("--context-length must be positive")

    reserved_model_config_keys = lib.get_reserved_model_config_keys(args.model_config)
    if len(reserved_model_config_keys) > 0:
        raise cli.ValidationError(
            "--model-config cannot contain keys with dedicated CLI flags: " f"{', '.join(reserved_model_config_keys)}"
        )
    if args.image_encoder_config is not None and "size" in args.image_encoder_config:
        raise cli.ValidationError("--image-encoder-config cannot contain size, use --size")
    if args.text_encoder_config is not None and "context_length" in args.text_encoder_config:
        raise cli.ValidationError("--text-encoder-config cannot contain context_length, use --context-length")

    if args.compile_mode is not None and args.compile is False:
        raise cli.ValidationError("--compile-mode requires --compile")
    if args.amp is True and args.model_dtype != "float32":
        raise cli.ValidationError("--amp can only be used with --model-dtype float32")
    if args.wds is False and len(args.data_path) == 0:
        raise cli.ValidationError("Must provide at least one data source, --data-path or --wds")
    if args.wds is True:
        if args.wds_info is None and len(args.data_path) == 0:
            raise cli.ValidationError("--wds requires a data path unless --wds-info is provided")
        if len(args.data_path) > 1:
            raise cli.ValidationError(f"--wds can have at most 1 --data-path, got {len(args.data_path)}")
        if args.wds_info is None and len(args.data_path) == 1:
            data_path = args.data_path[0]
            if "://" in data_path and args.wds_size is None:
                raise cli.ValidationError("--wds-size is required for remote --data-path")


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

    embed(args)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
