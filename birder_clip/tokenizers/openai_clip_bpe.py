"""
OpenAI CLIP byte-pair encoding tokenizer, adapted from
https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py
and
https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/tokenizer.py
"""

# Reference license: MIT (both)

import gzip
import html
import string
from collections.abc import Callable
from collections.abc import Iterable
from importlib import resources
from pathlib import Path
from typing import Optional

import ftfy
import regex as re
import torch

DEFAULT_CONTEXT_LENGTH = 77


def _default_bpe_resource() -> resources.abc.Traversable:
    return resources.files("birder_clip.tokenizers").joinpath("bpe_simple_vocab_16e6.txt.gz")


def bytes_to_unicode() -> dict[int, str]:
    """
    Return a reversible mapping from bytes to unicode strings

    This lets the tokenizer represent arbitrary UTF-8 text without an unknown token.
    """

    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))

    unicode_values = byte_values[:]
    index = 0
    for byte in range(2**8):
        if byte not in byte_values:
            byte_values.append(byte)
            unicode_values.append(2**8 + index)
            index += 1

    return dict(zip(byte_values, [chr(value) for value in unicode_values]))


def get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    pairs = set()
    previous = word[0]
    for current in word[1:]:
        pairs.add((previous, current))
        previous = current

    return pairs


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


class SimpleTokenizer:
    def __init__(
        self,
        bpe_path: Optional[str | Path] = None,
        *,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        clean: str = "lower",
    ) -> None:
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {value: key for key, value in self.byte_encoder.items()}
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)

        merges = self._load_merges(bpe_path)
        base_vocab = list(self.byte_encoder.values())
        vocab = base_vocab + [token + "</w>" for token in base_vocab]
        vocab.extend("".join(merge) for merge in merges)

        special_tokens = ["<start_of_text>", "<end_of_text>"]
        vocab.extend(special_tokens)

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {value: key for key, value in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {token: token for token in special_tokens}

        special_pattern = "|".join(re.escape(token) for token in special_tokens)
        self.pat = re.compile(
            special_pattern + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

        self.vocab_size = len(self.encoder)
        self.sot_token_id = self.encoder["<start_of_text>"]
        self.eot_token_id = self.encoder["<end_of_text>"]

    def _load_merges(self, bpe_path: Optional[str | Path]) -> list[tuple[str, str]]:
        if bpe_path is None:
            with _default_bpe_resource().open("rb") as handle:
                contents = gzip.decompress(handle.read()).decode("utf-8")
        else:
            with gzip.open(bpe_path, "rt", encoding="utf-8") as handle:
                contents = handle.read()

        merges = contents.splitlines()
        merges = merges[1 : 49152 - 256 - 2 + 1]
        return [tuple(merge.split()) for merge in merges]  # type: ignore[misc]

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if len(pairs) == 0:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word: list[str] = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                    new_word.extend(word[index:next_index])
                    index = next_index
                except ValueError:
                    new_word.extend(word[index:])
                    break

                if word[index] == first and index < len(word) - 1 and word[index + 1] == second:
                    new_word.append(first + second)
                    index += 2
                else:
                    new_word.append(word[index])
                    index += 1

            word = tuple(new_word)
            if len(word) == 1:
                break

            pairs = get_pairs(word)

        word_str = " ".join(word)
        self.cache[token] = word_str
        return word_str

    def encode(self, text: str) -> list[int]:
        bpe_tokens: list[int] = []
        text = self.clean_fn(text)
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[byte] for byte in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))

        return bpe_tokens

    def decode(self, token_ids: Iterable[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()

        text = "".join(self.decoder[token_id] for token_id in token_ids)
        byte_values = bytearray(self.byte_decoder[char] for char in text)
        return byte_values.decode("utf-8", errors="replace").replace("</w>", " ")

    def __call__(self, texts: str | list[str], context_length: Optional[int] = None) -> torch.LongTensor:
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        all_tokens = [[self.sot_token_id] + self.encode(text) + [self.eot_token_id] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for index, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
                tokens[-1] = self.eot_token_id

            result[index, : len(tokens)] = torch.tensor(tokens)

        return result
