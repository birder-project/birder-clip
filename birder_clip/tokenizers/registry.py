import warnings
from collections.abc import Callable
from typing import Any

from birder_clip.tokenizers.base import Tokenizer

TokenizerFactory = Callable[..., Tokenizer]

_TOKENIZERS: dict[str, tuple[TokenizerFactory, dict[str, Any]]] = {}
_TOKENIZER_PREFIXES: dict[str, tuple[TokenizerFactory, dict[str, Any]]] = {}


def register_tokenizer(name: str, factory: TokenizerFactory, **kwargs: Any) -> None:
    name_key = name.lower()
    if name_key in _TOKENIZERS:
        warnings.warn(f"Tokenizer '{name}' is already registered and will be overwritten", UserWarning)

    _TOKENIZERS[name_key] = (factory, kwargs)


def register_tokenizer_prefix(prefix: str, factory: TokenizerFactory, **kwargs: Any) -> None:
    prefix_key = prefix.lower()
    if prefix_key in _TOKENIZER_PREFIXES:
        warnings.warn(f"Tokenizer prefix '{prefix}' is already registered and will be overwritten", UserWarning)

    _TOKENIZER_PREFIXES[prefix_key] = (factory, kwargs)


def get_tokenizer(name: str, **kwargs: Any) -> Tokenizer:
    factory, factory_kwargs = get_tokenizer_info(name)
    return factory(**{**factory_kwargs, **kwargs})


def get_tokenizer_info(name: str) -> tuple[TokenizerFactory, dict[str, Any]]:
    name_key = name.lower()
    if name_key in _TOKENIZERS:
        factory, factory_kwargs = _TOKENIZERS[name_key]
        return factory, factory_kwargs.copy()

    for prefix, (factory, factory_kwargs) in _TOKENIZER_PREFIXES.items():
        if name_key.startswith(prefix):
            return factory, {**factory_kwargs, "source": name[len(prefix) :]}

    raise KeyError(name)


def exists(name: str) -> bool:
    name_key = name.lower()
    if name_key in _TOKENIZERS:
        return True

    return any(name_key.startswith(prefix) for prefix in _TOKENIZER_PREFIXES)


def list_tokenizers() -> list[str]:
    return sorted(_TOKENIZERS)
