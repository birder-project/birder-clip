import argparse
import logging
from typing import Any

from birder.common import cli
from transformers import AutoTokenizer

from birder_clip.conf import settings
from birder_clip.model_registry import Task
from birder_clip.model_registry import registry
from birder_clip.tokenizers.hf import HFTokenizer
from birder_clip.tokenizers.hf import hf_tokenizer_path
from birder_clip.tokenizers.registry import exists as tokenizer_exists
from birder_clip.tokenizers.registry import get_tokenizer_info

logger = logging.getLogger(__name__)


def _resolve_hf_tokenizer_name(tokenizer: str) -> str:
    if tokenizer.startswith("hf:"):
        return tokenizer.removeprefix("hf:")

    if tokenizer_exists(tokenizer) is True:
        factory, tokenizer_kwargs = get_tokenizer_info(tokenizer)
        if factory is not HFTokenizer:
            raise cli.ValidationError(f"{tokenizer} is not a Hugging Face tokenizer")
        if "pretrained_model_name_or_path" not in tokenizer_kwargs:
            raise cli.ValidationError(f"{tokenizer} does not define a Hugging Face tokenizer source")

        return tokenizer_kwargs["pretrained_model_name_or_path"]  # type: ignore[no-any-return]

    if registry.exists(tokenizer, task=Task.IMAGE_TEXT) is True:
        config = registry.all_nets[tokenizer.lower()].config  # type: ignore[misc]
        if config is None or "tokenizer" not in config:
            raise cli.ValidationError(f"{tokenizer} does not define a tokenizer")

        return _resolve_hf_tokenizer_name(config["tokenizer"])

    raise cli.ValidationError(f"{tokenizer} is not a registered tokenizer or image-text model")


def download_tokenizer(args: argparse.Namespace) -> None:
    tokenizer_name = _resolve_hf_tokenizer_name(args.tokenizer)
    path = hf_tokenizer_path(tokenizer_name)
    if path.exists() is True:
        logger.info(f"Tokenizer already exists at {path}, skipping download...")
        return

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)  # nosec B615
    if settings.TOKENIZERS_DIR.exists() is False:
        logger.info(f"Creating {settings.TOKENIZERS_DIR} directory...")
        settings.TOKENIZERS_DIR.mkdir(parents=True)

    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving tokenizer at {path}...")
    tokenizer.save_pretrained(path)


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "download-tokenizer",
        allow_abbrev=False,
        help="download a tokenizer",
        description="download a tokenizer",
        epilog=(
            "Usage example:\n"
            "python -m birder_clip.tools download-tokenizer --tokenizer siglip_v2_vit_so400m_p14\n"
            "python -m birder_clip.tools download-tokenizer --tokenizer siglip2_gemma\n"
            "python -m birder_clip.tools download-tokenizer --tokenizer hf:xlm-roberta-base\n"
            "python -m birder_clip.tools download-tokenizer --tokenizer hf:timm/ViT-SO400M-14-SigLIP2\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("--tokenizer", type=str, help="tokenizer, Hugging Face tokenizer or image-text model name")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    if args.tokenizer is None:
        raise cli.ValidationError("--tokenizer is required")

    download_tokenizer(args)
