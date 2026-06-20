import copy
import json
import logging
import unittest
from typing import Any

import torch
from birder.common.lib import env_bool
from birder.conf.settings import DEFAULT_NUM_CHANNELS
from parameterized import parameterized

from birder_clip.model_registry import registry
from birder_clip.net.base import BaseNet

logging.disable(logging.CRITICAL)

TEST_IMAGE_SIZE = (64, 64)
TEST_CONTEXT_LENGTH = 8
TEST_EMBED_DIM = 384
TEST_EOS_TOKEN_ID = 9
TEST_VOCAB_SIZE = 10

CLIP_TEST_CASES = [
    # Image encoder, text encoder, text config
    ("convnext_v1_atto", "transformer_encoder_small", {"pool_type": "first"}),
    ("rope_vit_s32", "transformer_encoder_small", {"causal_mask": False, "pad_token_id": 0}),
    ("swin_transformer_v2_t", "transformer_encoder_small", {"pool_type": "last"}),
    ("vit_s16", "transformer_encoder_small", {}),
]

COCA_TEST_CASES = [
    # Image encoder, text encoder, text config
    ("regnet_y_200m", "transformer_encoder_small", {"pool_type": "last"}),
    ("focalnet_t_srf", "transformer_encoder_small", {"pool_type": "last"}),
    ("vit_s16", "transformer_encoder_small", {"pool_type": "last"}),
]


def _text_tokens(count: int) -> torch.Tensor:
    tokens = torch.tensor(
        [
            [1, 2, TEST_EOS_TOKEN_ID, 0, 0, 0, 0, 0],
            [1, 2, 5, 6, TEST_EOS_TOKEN_ID, 0, 0, 0],
            [1, 2, 5, 6, 7, 2, 4, TEST_EOS_TOKEN_ID],
        ]
    )

    return tokens[:count]


class TestBase(unittest.TestCase):
    def test_config_override_does_not_mutate_registered_config(self) -> None:
        class RegisteredNet(BaseNet):  # pylint: disable=abstract-method
            config = {"name": "registered", "nested": {"value": 1}}

        net = RegisteredNet(config={"name": "override"})
        net.config["nested"]["value"] = 2
        next_net = RegisteredNet()

        self.assertEqual(net.config["name"], "override")
        self.assertEqual(next_net.config["name"], "registered")
        self.assertEqual(next_net.config["nested"]["value"], 1)
        self.assertEqual(RegisteredNet.config["nested"]["value"], 1)


class TestCLIP(unittest.TestCase):
    @parameterized.expand(CLIP_TEST_CASES)  # type: ignore[untyped-decorator]
    def test_net(self, image_encoder: str, text_encoder: str, text_config: dict[str, Any]) -> None:
        batch_size = 2
        n = registry.net_factory(
            "clip",
            config={
                "image": {
                    "network": image_encoder,
                    "input_channels": DEFAULT_NUM_CHANNELS,
                    "size": TEST_IMAGE_SIZE,
                },
                "text": {
                    "network": text_encoder,
                    "config": {
                        "vocab_size": TEST_VOCAB_SIZE,
                        "eos_token_id": TEST_EOS_TOKEN_ID,
                        **text_config,
                    },
                    "context_length": TEST_CONTEXT_LENGTH,
                },
                "embed_dim": TEST_EMBED_DIM,
                "tokenizer": "simple_tokenizer",
            },
        )
        size = TEST_IMAGE_SIZE

        self.assertEqual(n.embedding_size, TEST_EMBED_DIM)
        self.assertEqual(n.tokenizer_name, "simple_tokenizer")

        # Ensure config is serializable
        _ = json.dumps(n.config)

        # Test network
        image = torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size))
        text = _text_tokens(batch_size)
        logits = n(image, text)

        self.assertEqual(logits.numel(), batch_size * batch_size)
        self.assertFalse(torch.isnan(logits).any())

        # Non-equal batch size
        logits = n(torch.rand((1, DEFAULT_NUM_CHANNELS, *size)), _text_tokens(3))

        self.assertSequenceEqual(logits.shape, (1, 3))
        self.assertFalse(torch.isnan(logits).any())

        n.eval()
        with torch.inference_mode():
            image_features = n.encode_image(image)
            text_features = n.encode_text(text)
            logits = n(image, text)
            out = n(image, text, return_features=True)

        n.train()

        self.assertSequenceEqual(image_features.shape, (batch_size, TEST_EMBED_DIM))
        self.assertSequenceEqual(text_features.shape, (batch_size, TEST_EMBED_DIM))
        self.assertFalse(torch.isnan(image_features).any())
        self.assertFalse(torch.isnan(text_features).any())

        self.assertSetEqual(set(out.keys()), {"image_features", "text_features", "logit_scale", "logit_bias"})
        image_features = out["image_features"]
        text_features = out["text_features"]
        logit_scale = out["logit_scale"]

        self.assertSequenceEqual(image_features.shape, (batch_size, TEST_EMBED_DIM))
        self.assertSequenceEqual(text_features.shape, (batch_size, TEST_EMBED_DIM))
        self.assertEqual(logit_scale.ndim, 1)
        self.assertIsNone(out["logit_bias"])
        self.assertFalse(torch.isnan(image_features).any())
        self.assertFalse(torch.isnan(text_features).any())
        torch.testing.assert_close(n.forward_logits(image_features, text_features), logits)

        # Adjust size
        size = (n.image_encoder.size[0] + 32, n.image_encoder.size[1] + 32)
        n.adjust_image_size(size)
        logits = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)), text)

        self.assertEqual(logits.numel(), batch_size * batch_size)
        self.assertFalse(torch.isnan(logits).any())

        # Adjust context length
        context_length = 10
        n.adjust_context_length(context_length)
        text = torch.tensor(
            [
                [1, 2, TEST_EOS_TOKEN_ID, 0, 0, 0, 0, 0, 0, 0],
                [1, 2, 5, 6, TEST_EOS_TOKEN_ID, 0, 0, 0, 0, 0],
            ]
        )
        logits = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)), text)

        self.assertSequenceEqual(logits.shape, (batch_size, batch_size))
        self.assertFalse(torch.isnan(logits).any())

        # Ensure model is copyable
        n_copy = copy.deepcopy(n)
        self.assertIsNotNone(n_copy)

    @parameterized.expand(CLIP_TEST_CASES)  # type: ignore[untyped-decorator]
    @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    def test_net_backward(self, image_encoder: str, text_encoder: str, text_config: dict[str, Any]) -> None:
        batch_size = 2
        n = registry.net_factory(
            "clip",
            config={
                "image": {
                    "network": image_encoder,
                    "input_channels": DEFAULT_NUM_CHANNELS,
                    "size": TEST_IMAGE_SIZE,
                },
                "text": {
                    "network": text_encoder,
                    "config": {
                        "vocab_size": TEST_VOCAB_SIZE,
                        "eos_token_id": TEST_EOS_TOKEN_ID,
                        **text_config,
                    },
                    "context_length": TEST_CONTEXT_LENGTH,
                },
                "embed_dim": TEST_EMBED_DIM,
                "tokenizer": "simple_tokenizer",
            },
        )

        logits = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *TEST_IMAGE_SIZE)), _text_tokens(batch_size))

        loss = logits.sum()
        loss.backward()
        for name, param in n.named_parameters():
            self.assertIsNotNone(param.grad, msg=f"missing grad for {name}")
            self.assertTrue(torch.isfinite(param.grad).all().item(), msg=f"non-finite grad for {name}")

    @parameterized.expand(CLIP_TEST_CASES)  # type: ignore[untyped-decorator]
    @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    def test_grad_checkpointing(self, image_encoder: str, text_encoder: str, text_config: dict[str, Any]) -> None:
        batch_size = 1
        n = registry.net_factory(
            "clip",
            config={
                "image": {
                    "network": image_encoder,
                    "input_channels": DEFAULT_NUM_CHANNELS,
                    "size": TEST_IMAGE_SIZE,
                },
                "text": {
                    "network": text_encoder,
                    "config": {
                        "vocab_size": TEST_VOCAB_SIZE,
                        "eos_token_id": TEST_EOS_TOKEN_ID,
                        **text_config,
                    },
                    "context_length": TEST_CONTEXT_LENGTH,
                },
                "embed_dim": TEST_EMBED_DIM,
                "tokenizer": "simple_tokenizer",
            },
        )
        image = torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *TEST_IMAGE_SIZE))
        text = _text_tokens(batch_size)

        n.eval()
        with torch.no_grad():
            expected = n(image, text)

        n.set_grad_checkpointing(segments=4)
        out = n(image, text)

        # Verify forward with checkpointing
        self.assertEqual(out.size(), expected.size())
        self.assertTrue(torch.allclose(out, expected))
        self.assertEqual(out.numel(), batch_size * batch_size)
        self.assertTrue(torch.isfinite(out).all().item())

        # Check grads
        loss = out.sum()
        loss.backward()
        for name, param in n.named_parameters():
            self.assertIsNotNone(param.grad, msg=f"missing grad for {name}")
            self.assertTrue(torch.isfinite(param.grad).all().item(), msg=f"non-finite grad for {name}")

    @parameterized.expand(CLIP_TEST_CASES)  # type: ignore[untyped-decorator]
    @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    def test_net_pt2(self, image_encoder: str, text_encoder: str, text_config: dict[str, Any]) -> None:
        n = registry.net_factory(
            "clip",
            config={
                "image": {
                    "network": image_encoder,
                    "input_channels": DEFAULT_NUM_CHANNELS,
                    "size": TEST_IMAGE_SIZE,
                },
                "text": {
                    "network": text_encoder,
                    "config": {
                        "vocab_size": TEST_VOCAB_SIZE,
                        "eos_token_id": TEST_EOS_TOKEN_ID,
                        **text_config,
                    },
                    "context_length": TEST_CONTEXT_LENGTH,
                },
                "embed_dim": TEST_EMBED_DIM,
                "tokenizer": "simple_tokenizer",
            },
        )
        n.eval()

        image_batch_dim = torch.export.Dim("image_batch")
        text_batch_dim = torch.export.Dim("text_batch")
        with torch.no_grad():
            torch.export.export(
                n,
                (torch.randn(2, DEFAULT_NUM_CHANNELS, *TEST_IMAGE_SIZE), _text_tokens(2)),
                dynamic_shapes={
                    "image": {0: image_batch_dim},
                    "text": {0: text_batch_dim},
                },
            )


class TestCoCa(unittest.TestCase):
    @parameterized.expand(COCA_TEST_CASES)  # type: ignore[untyped-decorator]
    def test_net(self, image_encoder: str, text_encoder: str, text_config: dict[str, Any]) -> None:
        batch_size = 2
        n = registry.net_factory(
            "coca",
            config={
                "image": {
                    "network": image_encoder,
                    "input_channels": DEFAULT_NUM_CHANNELS,
                    "size": TEST_IMAGE_SIZE,
                },
                "text": {
                    "network": text_encoder,
                    "config": {
                        "vocab_size": TEST_VOCAB_SIZE,
                        "eos_token_id": TEST_EOS_TOKEN_ID,
                        "class_token": True,
                        "norm_after_pool": True,
                        **text_config,
                    },
                    "context_length": TEST_CONTEXT_LENGTH,
                },
                "decoder": {
                    "network": "conditioned_decoder",
                    "config": {
                        "num_heads": 6,
                        "num_layers": 2,
                    },
                    "context_length": TEST_CONTEXT_LENGTH,
                },
                "visual_pool": {
                    "num_heads": 6,
                    "latent_len": 8,
                },
                "embed_dim": TEST_EMBED_DIM,
                "tokenizer": "simple_tokenizer",
            },
        )
        size = TEST_IMAGE_SIZE

        self.assertEqual(n.embedding_size, TEST_EMBED_DIM)
        self.assertEqual(n.tokenizer_name, "simple_tokenizer")

        # Ensure config is serializable
        _ = json.dumps(n.config)

        # Test network
        image = torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size))
        text = _text_tokens(batch_size)
        logits = n(image, text)

        self.assertEqual(logits.numel(), batch_size * batch_size)
        self.assertFalse(torch.isnan(logits).any())

        # Non-equal batch size
        logits = n(torch.rand((1, DEFAULT_NUM_CHANNELS, *size)), _text_tokens(3))

        self.assertSequenceEqual(logits.shape, (1, 3))
        self.assertFalse(torch.isnan(logits).any())

        n.eval()
        with torch.inference_mode():
            image_features = n.encode_image(image)
            text_features = n.encode_text(text)
            logits = n(image, text)
            out = n(image, text, return_features=True)

        n.train()

        self.assertSequenceEqual(image_features.shape, (batch_size, TEST_EMBED_DIM))
        self.assertSequenceEqual(text_features.shape, (batch_size, TEST_EMBED_DIM))
        self.assertFalse(torch.isnan(image_features).any())
        self.assertFalse(torch.isnan(text_features).any())

        self.assertSetEqual(set(out.keys()), {"image_features", "text_features", "logits", "logit_scale", "logit_bias"})
        image_features = out["image_features"]
        text_features = out["text_features"]
        decoder_logits = out["logits"]
        logit_scale = out["logit_scale"]

        self.assertSequenceEqual(image_features.shape, (batch_size, TEST_EMBED_DIM))
        self.assertSequenceEqual(text_features.shape, (batch_size, TEST_EMBED_DIM))
        self.assertSequenceEqual(decoder_logits.shape, (batch_size, TEST_CONTEXT_LENGTH, TEST_VOCAB_SIZE))
        self.assertEqual(logit_scale.ndim, 1)
        self.assertIsNone(out["logit_bias"])
        self.assertFalse(torch.isnan(image_features).any())
        self.assertFalse(torch.isnan(text_features).any())
        self.assertFalse(torch.isnan(decoder_logits).any())
        torch.testing.assert_close(n.forward_logits(image_features, text_features), logits)

        # Adjust size
        size = (n.image_encoder.size[0] + 32, n.image_encoder.size[1] + 32)
        n.adjust_image_size(size)
        logits = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)), text)

        self.assertEqual(logits.numel(), batch_size * batch_size)
        self.assertFalse(torch.isnan(logits).any())

        # Adjust context length
        context_length = 10
        n.adjust_context_length(context_length)
        text = torch.tensor(
            [
                [1, 2, TEST_EOS_TOKEN_ID, 0, 0, 0, 0, 0, 0, 0],
                [1, 2, 5, 6, TEST_EOS_TOKEN_ID, 0, 0, 0, 0, 0],
            ]
        )
        out = n(torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *size)), text, return_features=True)

        self.assertSequenceEqual(out["image_features"].shape, (batch_size, TEST_EMBED_DIM))
        self.assertSequenceEqual(out["text_features"].shape, (batch_size, TEST_EMBED_DIM))
        self.assertSequenceEqual(out["logits"].shape, (batch_size, context_length, TEST_VOCAB_SIZE))
        self.assertFalse(torch.isnan(out["logits"]).any())

        # Ensure model is copyable
        n_copy = copy.deepcopy(n)
        self.assertIsNotNone(n_copy)

    @parameterized.expand(COCA_TEST_CASES)  # type: ignore[untyped-decorator]
    @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    def test_net_backward(self, image_encoder: str, text_encoder: str, text_config: dict[str, Any]) -> None:
        batch_size = 2
        n = registry.net_factory(
            "coca",
            config={
                "image": {
                    "network": image_encoder,
                    "input_channels": DEFAULT_NUM_CHANNELS,
                    "size": TEST_IMAGE_SIZE,
                },
                "text": {
                    "network": text_encoder,
                    "config": {
                        "vocab_size": TEST_VOCAB_SIZE,
                        "eos_token_id": TEST_EOS_TOKEN_ID,
                        "class_token": True,
                        "norm_after_pool": True,
                        **text_config,
                    },
                    "context_length": TEST_CONTEXT_LENGTH,
                },
                "decoder": {
                    "network": "conditioned_decoder",
                    "config": {
                        "num_heads": 6,
                        "num_layers": 2,
                    },
                    "context_length": TEST_CONTEXT_LENGTH,
                },
                "visual_pool": {
                    "num_heads": 6,
                    "latent_len": 8,
                },
                "embed_dim": TEST_EMBED_DIM,
                "tokenizer": "simple_tokenizer",
            },
        )

        out = n(
            torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *TEST_IMAGE_SIZE)),
            _text_tokens(batch_size),
            return_features=True,
        )

        loss = n.forward_logits(out["image_features"], out["text_features"]).sum() + out["logits"].sum()
        loss.backward()
        for name, param in n.named_parameters():
            self.assertIsNotNone(param.grad, msg=f"missing grad for {name}")
            self.assertTrue(torch.isfinite(param.grad).all().item(), msg=f"non-finite grad for {name}")

    @parameterized.expand(COCA_TEST_CASES)  # type: ignore[untyped-decorator]
    @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    def test_grad_checkpointing(self, image_encoder: str, text_encoder: str, text_config: dict[str, Any]) -> None:
        batch_size = 1
        n = registry.net_factory(
            "coca",
            config={
                "image": {
                    "network": image_encoder,
                    "input_channels": DEFAULT_NUM_CHANNELS,
                    "size": TEST_IMAGE_SIZE,
                },
                "text": {
                    "network": text_encoder,
                    "config": {
                        "vocab_size": TEST_VOCAB_SIZE,
                        "eos_token_id": TEST_EOS_TOKEN_ID,
                        "class_token": True,
                        "norm_after_pool": True,
                        **text_config,
                    },
                    "context_length": TEST_CONTEXT_LENGTH,
                },
                "decoder": {
                    "network": "conditioned_decoder",
                    "config": {
                        "num_heads": 6,
                        "num_layers": 2,
                    },
                    "context_length": TEST_CONTEXT_LENGTH,
                },
                "visual_pool": {
                    "num_heads": 6,
                    "latent_len": 8,
                },
                "embed_dim": TEST_EMBED_DIM,
                "tokenizer": "simple_tokenizer",
            },
        )
        image = torch.rand((batch_size, DEFAULT_NUM_CHANNELS, *TEST_IMAGE_SIZE))
        text = _text_tokens(batch_size)

        n.eval()
        with torch.no_grad():
            expected = n(image, text, return_features=True)

        n.set_grad_checkpointing(segments=4)
        out = n(image, text, return_features=True)

        # Verify forward with checkpointing
        torch.testing.assert_close(out["image_features"], expected["image_features"])
        torch.testing.assert_close(out["text_features"], expected["text_features"])
        torch.testing.assert_close(out["logits"], expected["logits"])
        self.assertTrue(torch.isfinite(out["image_features"]).all().item())
        self.assertTrue(torch.isfinite(out["text_features"]).all().item())
        self.assertTrue(torch.isfinite(out["logits"]).all().item())

        # Check grads
        loss = n.forward_logits(out["image_features"], out["text_features"]).sum() + out["logits"].sum()
        loss.backward()
        for name, param in n.named_parameters():
            self.assertIsNotNone(param.grad, msg=f"missing grad for {name}")
            self.assertTrue(torch.isfinite(param.grad).all().item(), msg=f"non-finite grad for {name}")

    @parameterized.expand(COCA_TEST_CASES)  # type: ignore[untyped-decorator]
    @unittest.skipUnless(env_bool("SLOW_TESTS"), "Avoid slow tests")
    def test_net_pt2(self, image_encoder: str, text_encoder: str, text_config: dict[str, Any]) -> None:
        n = registry.net_factory(
            "coca",
            config={
                "image": {
                    "network": image_encoder,
                    "input_channels": DEFAULT_NUM_CHANNELS,
                    "size": TEST_IMAGE_SIZE,
                },
                "text": {
                    "network": text_encoder,
                    "config": {
                        "vocab_size": TEST_VOCAB_SIZE,
                        "eos_token_id": TEST_EOS_TOKEN_ID,
                        "class_token": True,
                        "norm_after_pool": True,
                        **text_config,
                    },
                    "context_length": TEST_CONTEXT_LENGTH,
                },
                "decoder": {
                    "network": "conditioned_decoder",
                    "config": {
                        "num_heads": 6,
                        "num_layers": 2,
                    },
                    "context_length": TEST_CONTEXT_LENGTH,
                },
                "visual_pool": {
                    "num_heads": 6,
                    "latent_len": 8,
                },
                "embed_dim": TEST_EMBED_DIM,
                "tokenizer": "simple_tokenizer",
            },
        )
        n.eval()

        image_batch_dim = torch.export.Dim("image_batch")
        text_batch_dim = torch.export.Dim("text_batch")
        with torch.no_grad():
            torch.export.export(
                n,
                (torch.randn(2, DEFAULT_NUM_CHANNELS, *TEST_IMAGE_SIZE), _text_tokens(2)),
                dynamic_shapes={
                    "image": {0: image_batch_dim},
                    "text": {0: text_batch_dim},
                },
            )
