import logging
from pathlib import Path
from typing import Any
from typing import Optional

import torch
from transformers import AutoTokenizer

from birder_clip.conf import settings
from birder_clip.tokenizers.base import get_clean_fn
from birder_clip.tokenizers.registry import exists as tokenizer_exists
from birder_clip.tokenizers.registry import get_tokenizer_info
from birder_clip.tokenizers.registry import register_tokenizer
from birder_clip.tokenizers.registry import register_tokenizer_prefix

logger = logging.getLogger(__name__)


def hf_tokenizer_path(source: str) -> Path:
    return settings.TOKENIZERS_DIR.joinpath(source)


def download_hf_tokenizer(source: str) -> None:
    path = hf_tokenizer_path(source)
    if path.exists() is True:
        logger.debug(f"Tokenizer already exists at {path}, skipping download...")
        return

    tokenizer = AutoTokenizer.from_pretrained(source)  # nosec B615
    if settings.TOKENIZERS_DIR.exists() is False:
        logger.info(f"Creating {settings.TOKENIZERS_DIR} directory...")
        settings.TOKENIZERS_DIR.mkdir(parents=True)

    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving tokenizer at {path}...")
    tokenizer.save_pretrained(path)


def get_hf_tokenizer_source(name: str) -> Optional[str]:
    if tokenizer_exists(name) is False:
        return None

    factory, factory_kwargs = get_tokenizer_info(name)
    if factory is not HFTokenizer:
        return None
    if "source" not in factory_kwargs:
        return None

    return factory_kwargs["source"]  # type: ignore[no-any-return]


class HFTokenizer:
    def __init__(
        self,
        source: str,
        *,
        context_length: int = 77,
        clean: str = "whitespace",
        **kwargs: Any,
    ) -> None:
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)
        self.tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            str(hf_tokenizer_path(source)),
            local_files_only=True,
            **kwargs,
        )

    def __call__(self, texts: str | list[str], context_length: Optional[int] = None) -> torch.LongTensor:
        if isinstance(texts, str):
            texts = [texts]

        return self.tokenizer(
            [self.clean_fn(text) for text in texts],
            return_tensors="pt",
            max_length=context_length or self.context_length,
            padding="max_length",
            truncation=True,
        ).input_ids


register_tokenizer_prefix("hf:", HFTokenizer)

register_tokenizer(
    "siglip2_gemma", HFTokenizer, source="timm/ViT-SO400M-14-SigLIP2", context_length=64, clean="canonicalize"
)
