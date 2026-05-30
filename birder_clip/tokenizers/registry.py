import warnings
from collections.abc import Callable

from birder_clip.tokenizers.base import Tokenizer
from birder_clip.tokenizers.openai_clip_bpe import SimpleTokenizer

TokenizerFactory = Callable[[], Tokenizer]

_TOKENIZERS: dict[str, TokenizerFactory] = {}


def register_tokenizer(name: str, factory: TokenizerFactory) -> None:
    name_key = name.lower()
    if name_key in _TOKENIZERS:
        warnings.warn(f"Tokenizer '{name}' is already registered and will be overwritten", UserWarning)

    _TOKENIZERS[name_key] = factory


def get_tokenizer(name: str) -> Tokenizer:
    name_key = name.lower()
    return _TOKENIZERS[name_key]()


def list_tokenizers() -> list[str]:
    return sorted(_TOKENIZERS)


register_tokenizer("openai_clip_bpe", SimpleTokenizer)
