import html
import string
from collections.abc import Callable
from typing import Optional
from typing import Protocol

import ftfy
import torch


class Tokenizer(Protocol):
    context_length: int
    num_special_tokens: int

    def encode(self, text: str) -> list[int]: ...

    def __call__(self, texts: str | list[str], context_length: Optional[int] = None) -> torch.LongTensor: ...


def basic_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    return " ".join(text.split()).strip()


def canonicalize_text(text: str) -> str:
    text = text.replace("_", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    return whitespace_clean(text)


def _clean_lower(text: str) -> str:
    return whitespace_clean(basic_clean(text)).lower()


def _clean_whitespace(text: str) -> str:
    return whitespace_clean(basic_clean(text))


def _clean_canonicalize(text: str) -> str:
    return canonicalize_text(basic_clean(text))


def get_clean_fn(clean: str) -> Callable[[str], str]:
    if clean == "lower":
        return _clean_lower
    if clean == "whitespace":
        return _clean_whitespace
    if clean == "canonicalize":
        return _clean_canonicalize

    raise ValueError(f"Unknown clean function: {clean}")
