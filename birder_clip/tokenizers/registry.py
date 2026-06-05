import warnings
from collections.abc import Callable
from typing import Any

from birder_clip.tokenizers.base import Tokenizer
from birder_clip.tokenizers.hf import HFTokenizer
from birder_clip.tokenizers.openai_clip_bpe import SimpleTokenizer

TokenizerFactory = Callable[..., Tokenizer]

_TOKENIZERS: dict[str, tuple[TokenizerFactory, dict[str, Any]]] = {}


def register_tokenizer(name: str, factory: TokenizerFactory, **kwargs: Any) -> None:
    name_key = name.lower()
    if name_key in _TOKENIZERS:
        warnings.warn(f"Tokenizer '{name}' is already registered and will be overwritten", UserWarning)

    _TOKENIZERS[name_key] = (factory, kwargs)


def get_tokenizer(name: str, **kwargs: Any) -> Tokenizer:
    if name.startswith("hf:"):
        return HFTokenizer(name.removeprefix("hf:"), **kwargs)

    name_key = name.lower()
    factory, factory_kwargs = _TOKENIZERS[name_key]
    return factory(**{**factory_kwargs, **kwargs})


def get_tokenizer_info(name: str) -> tuple[TokenizerFactory, dict[str, Any]]:
    name_key = name.lower()
    factory, factory_kwargs = _TOKENIZERS[name_key]
    return factory, factory_kwargs.copy()


def exists(name: str) -> bool:
    return name.lower() in _TOKENIZERS


def list_tokenizers() -> list[str]:
    return sorted(_TOKENIZERS)


register_tokenizer("openai_clip_bpe", SimpleTokenizer)
register_tokenizer("pe_base_bpe", SimpleTokenizer, context_length=32)
register_tokenizer(
    "siglip2_gemma",
    HFTokenizer,
    pretrained_model_name_or_path="timm/ViT-SO400M-14-SigLIP2",
    context_length=64,
    clean="canonicalize",
)
