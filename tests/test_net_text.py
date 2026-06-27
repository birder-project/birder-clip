import logging
import unittest

import torch

from birder_clip.model_registry import registry
from birder_clip.net.text.base import VisualCausalTextDecoderBaseNet
from birder_clip.net.text.encoder import _global_pool

logging.disable(logging.CRITICAL)

TEST_CONTEXT_LENGTH = 8
TEST_EOS_TOKEN_ID = 49407
TEST_VOCAB_SIZE = 10


def _text_tokens() -> torch.Tensor:
    return torch.tensor(
        [
            [1, 2, TEST_EOS_TOKEN_ID, 0, 0, 0, 0, 0],
            [3, 4, 5, TEST_EOS_TOKEN_ID, 0, 0, 0, 0],
            [1, 2, 5, 6, 7, 2, 4, TEST_EOS_TOKEN_ID],
        ]
    )


class TestTextNetworks(unittest.TestCase):
    def test_text_global_pool(self) -> None:
        features = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
        tokens = torch.tensor([[1, 9, 2, 0], [3, 4, 9, 0]])

        first = _global_pool(features, tokens, pool_type="first")
        last = _global_pool(features, tokens, pool_type="last")
        eos = _global_pool(features, tokens, pool_type="eos", eos_token_id=9)

        self.assertTrue(torch.equal(first, features[:, 0]))
        self.assertTrue(torch.equal(last, features[:, -1]))
        self.assertTrue(torch.equal(eos, torch.stack([features[0, 1], features[1, 2]])))

    def test_text_transformer_encoder_padding_attention_mask(self) -> None:
        net = registry.text_encoder_factory(
            "transformer_encoder_small",
            config={"causal_mask": False, "pad_token_id": 0},
            context_length=TEST_CONTEXT_LENGTH,
        )
        tokens = _text_tokens()
        attn_mask = net._pad_attention_mask(tokens)

        self.assertIsNotNone(attn_mask)
        self.assertEqual(attn_mask.dtype, torch.bool)
        self.assertSequenceEqual(attn_mask.shape, (3, 1, 1, TEST_CONTEXT_LENGTH))
        self.assertTrue(attn_mask[0][0][0][0].item())
        self.assertFalse(attn_mask[0][0][0][-1].item())

    def test_text_transformer_encoder_pool_types_forward(self) -> None:
        tokens = _text_tokens()
        for pool_type in ("first", "last", "eos"):
            with self.subTest(pool_type=pool_type):
                net = registry.text_encoder_factory(
                    "transformer_encoder_small",
                    config={"pool_type": pool_type},
                    context_length=TEST_CONTEXT_LENGTH,
                )
                out = net(tokens)

                self.assertSequenceEqual(out.shape, (3, net.embedding_size))
                self.assertTrue(torch.isfinite(out).all())

    def test_text_conditioned_decoder_forward_features_from_embeddings(self) -> None:
        net = registry.text_decoder_factory(
            "conditioned_decoder",
            config={"vocab_size": TEST_VOCAB_SIZE},
            context_length=TEST_CONTEXT_LENGTH,
        )

        text_embeddings = torch.rand(2, 5, net.hidden_dim)
        image_tokens = torch.rand(2, 6, net.kv_dim)
        cross_attn_mask = torch.ones(text_embeddings.size(1), image_tokens.size(1), dtype=torch.bool)
        features = net.forward_features_from_embeddings(text_embeddings, image_tokens, cross_attn_mask=cross_attn_mask)

        self.assertSequenceEqual(features.shape, (2, 5, net.hidden_dim))
        self.assertTrue(torch.isfinite(features).all())

    def test_text_conditioned_decoder_forward_head(self) -> None:
        net = registry.text_decoder_factory(
            "conditioned_decoder",
            config={"vocab_size": TEST_VOCAB_SIZE},
            context_length=TEST_CONTEXT_LENGTH,
        )

        text_embeddings = torch.rand(2, 5, net.hidden_dim)
        image_tokens = torch.rand(2, 6, net.kv_dim)
        features = net.forward_features_from_embeddings(text_embeddings, image_tokens)
        logits = net.forward_head(features)

        self.assertSequenceEqual(logits.shape, (2, 5, TEST_VOCAB_SIZE))
        self.assertTrue(torch.isfinite(logits).all())

    def test_text_visual_causal_text_decoder_forward(self) -> None:
        net = registry.text_decoder_factory(
            "visual_causal_text_decoder",
            config={"vocab_size": TEST_VOCAB_SIZE, "hidden_dim": 192, "num_heads": 3, "num_layers": 2},
            context_length=TEST_CONTEXT_LENGTH,
        )

        text = torch.randint(0, TEST_VOCAB_SIZE, (2, 5))
        image_tokens = torch.rand(2, 6, net.kv_dim)
        self.assertIsInstance(net, VisualCausalTextDecoderBaseNet)
        logits = net(text, image_tokens)

        self.assertSequenceEqual(logits.shape, (2, 5, TEST_VOCAB_SIZE))
        self.assertTrue(torch.isfinite(logits).all())

        adjusted_context_length = TEST_CONTEXT_LENGTH + 4
        net.adjust_context_length(adjusted_context_length)
        text = torch.randint(0, TEST_VOCAB_SIZE, (2, adjusted_context_length))
        logits = net(text, image_tokens)

        self.assertSequenceEqual(logits.shape, (2, adjusted_context_length, TEST_VOCAB_SIZE))
        self.assertTrue(torch.isfinite(logits).all())

    def test_text_visual_causal_text_decoder_shift_input_tokens(self) -> None:
        net = registry.text_decoder_factory(
            "visual_causal_text_decoder",
            config={
                "vocab_size": TEST_VOCAB_SIZE,
                "hidden_dim": 192,
                "num_heads": 3,
                "num_layers": 2,
                "shift_input_tokens": True,
            },
            context_length=TEST_CONTEXT_LENGTH,
        )

        text = torch.randint(0, TEST_VOCAB_SIZE, (2, 5))
        image_tokens = torch.rand(2, 6, net.kv_dim)
        logits = net(text, image_tokens)

        self.assertSequenceEqual(logits.shape, (2, 4, TEST_VOCAB_SIZE))
        self.assertTrue(torch.isfinite(logits).all())
