import logging
import unittest

import torch

from birder_clip.tokenizers import get_tokenizer
from birder_clip.tokenizers import list_tokenizers
from birder_clip.tokenizers import register_tokenizer
from birder_clip.tokenizers import simple_tokenizer
from birder_clip.tokenizers.registry import exists
from birder_clip.tokenizers.registry import get_tokenizer_info

logging.disable(logging.CRITICAL)


class TestTokenizerRegistry(unittest.TestCase):
    def test_tokenizer_registry(self) -> None:
        self.assertIn("simple_tokenizer", list_tokenizers())
        self.assertIn("openai_clip_bpe", list_tokenizers())
        self.assertTrue(exists("simple_tokenizer"))
        self.assertFalse(exists("missing_tokenizer"))
        self.assertIsInstance(get_tokenizer("simple_tokenizer"), simple_tokenizer.SimpleTokenizer)

    def test_duplicate_tokenizer_warning(self) -> None:
        with self.assertWarns(UserWarning):
            register_tokenizer(
                "simple_tokenizer",
                simple_tokenizer.SimpleTokenizer,
                context_length=simple_tokenizer.DEFAULT_CONTEXT_LENGTH,
            )

    def test_tokenizer_info_kwargs_are_copied(self) -> None:
        _, tokenizer_kwargs = get_tokenizer_info("pe_base_bpe")
        tokenizer_kwargs["context_length"] = 999

        _, next_tokenizer_kwargs = get_tokenizer_info("pe_base_bpe")
        self.assertEqual(next_tokenizer_kwargs["context_length"], 32)


class TestSimpleTokenizer(unittest.TestCase):
    def test_encode(self) -> None:
        tokenizer = simple_tokenizer.SimpleTokenizer()

        self.assertEqual(tokenizer.encode("a photo of a bird"), [320, 1125, 539, 320, 3329])
        self.assertEqual(tokenizer.encode("A   PHOTO of a bird"), [320, 1125, 539, 320, 3329])

    def test_encode_decode(self) -> None:
        tokenizer = simple_tokenizer.SimpleTokenizer()

        decoded = tokenizer.decode(tokenizer.encode("a photo of a bird"))
        self.assertEqual(decoded.strip(), "a photo of a bird")

    def test_call(self) -> None:
        tokenizer = simple_tokenizer.SimpleTokenizer()

        tokens = tokenizer(["a photo of a bird", "bird"])
        self.assertSequenceEqual(tokens.shape, (2, simple_tokenizer.DEFAULT_CONTEXT_LENGTH))
        self.assertEqual(tokens.dtype, torch.long)
        self.assertEqual(tokens[0, 0].item(), tokenizer.sot_token_id)
        self.assertEqual(tokens[0, 6].item(), tokenizer.eot_token_id)
        self.assertTrue(
            torch.equal(tokens[0, 7:], torch.zeros(simple_tokenizer.DEFAULT_CONTEXT_LENGTH - 7, dtype=torch.long))
        )

    def test_truncation_keeps_eot(self) -> None:
        tokenizer = simple_tokenizer.SimpleTokenizer()

        tokens = tokenizer("a photo of a bird", context_length=4)
        expected = torch.tensor([[tokenizer.sot_token_id, 320, 1125, tokenizer.eot_token_id]], dtype=torch.long)
        self.assertTrue(torch.equal(tokens, expected))
