import argparse
import logging
from typing import Any
from typing import get_args

import torch
import torch.nn.functional as F
from birder.common import cli
from birder.common.fs_ops import read_class_file
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
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    with torch.amp.autocast(device.type, enabled=args.amp, dtype=amp_dtype):
        text_embeddings = build_class_text_embeddings(
            net,
            tokenizer,
            class_names,
            templates,
            device=device,
            batch_size=args.text_batch_size,
        )

    top_k = min(args.top_k, len(class_names))
    with torch.inference_mode():
        for paths, images, _ in dataloader:
            images = images.to(device, dtype=model_dtype, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=args.amp, dtype=amp_dtype):
                image_embeddings = net.encode_image(images, normalize=True)
                logits = net.forward_logits(image_embeddings, text_embeddings)
                prob = F.softmax(logits, dim=-1)

            values, indices = prob.topk(top_k, dim=-1)

            for path, sample_values, sample_indices in zip(paths, values, indices):
                predictions = [
                    f"{class_names[idx.item()]}: {value.item():.4f}"
                    for value, idx in zip(sample_values, sample_indices)
                ]
                logger.info(f"{path}: {', '.join(predictions)}")


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
    parser.add_argument(
        "--template-set",
        choices=sorted(TEMPLATE_SETS.keys()),
        default="default",
        help="built-in zero-shot template set",
    )
    parser.add_argument("--template", action="append", help="additional prompt template")
    parser.add_argument("--template-file", type=str, help="file with one prompt template per line")
    parser.add_argument("--top-k", type=int, default=5, metavar="N", help="number of predictions to print")
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
