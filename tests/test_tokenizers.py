import logging
import unittest

import torch

from birder_clip.tokenizers import openai_clip_bpe

logging.disable(logging.CRITICAL)


class TestOpenAICLIPBPE(unittest.TestCase):
    def test_encode(self) -> None:
        tokenizer = openai_clip_bpe.SimpleTokenizer()

        self.assertEqual(tokenizer.encode("a photo of a bird"), [320, 1125, 539, 320, 3329])
        self.assertEqual(tokenizer.encode("A   PHOTO of a bird"), [320, 1125, 539, 320, 3329])

    def test_encode_decode(self) -> None:
        tokenizer = openai_clip_bpe.SimpleTokenizer()

        decoded = tokenizer.decode(tokenizer.encode("a photo of a bird"))
        self.assertEqual(decoded.strip(), "a photo of a bird")

    def test_call(self) -> None:
        tokenizer = openai_clip_bpe.SimpleTokenizer()

        tokens = tokenizer(["a photo of a bird", "bird"])
        self.assertSequenceEqual(tokens.shape, (2, openai_clip_bpe.DEFAULT_CONTEXT_LENGTH))
        self.assertEqual(tokens.dtype, torch.long)
        self.assertEqual(tokens[0, 0].item(), tokenizer.sot_token_id)
        self.assertEqual(tokens[0, 6].item(), tokenizer.eot_token_id)
        self.assertTrue(
            torch.equal(tokens[0, 7:], torch.zeros(openai_clip_bpe.DEFAULT_CONTEXT_LENGTH - 7, dtype=torch.long))
        )

    def test_truncation_keeps_eot(self) -> None:
        tokenizer = openai_clip_bpe.SimpleTokenizer()

        tokens = tokenizer("a photo of a bird", context_length=4)
        expected = torch.tensor([[tokenizer.sot_token_id, 320, 1125, tokenizer.eot_token_id]], dtype=torch.long)
        self.assertTrue(torch.equal(tokens, expected))
