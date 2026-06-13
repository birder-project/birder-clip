import logging
import unittest
from typing import Any

import torch

from birder_clip.model_registry import registry
from birder_clip.net.base import BaseNet
from birder_clip.net.text.transformer import text_global_pool

logging.disable(logging.CRITICAL)


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


class TestTextNetworks(unittest.TestCase):
    @staticmethod
    def _small_text_config(**kwargs: Any) -> dict[str, Any]:
        config: dict[str, Any] = {
            "vocab_size": 10,
            "hidden_dim": 16,
            "num_heads": 4,
            "num_layers": 2,
            "output_dim": 12,
            "eos_token_id": 9,
        }
        config.update(kwargs)

        return config

    def test_text_global_pool(self) -> None:
        features = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
        tokens = torch.tensor([[1, 9, 2, 0], [3, 4, 9, 0]])

        first = text_global_pool(features, tokens, pool_type="first")
        last = text_global_pool(features, tokens, pool_type="last")
        eos = text_global_pool(features, tokens, pool_type="eos", eos_token_id=9)

        self.assertTrue(torch.equal(first, features[:, 0]))
        self.assertTrue(torch.equal(last, features[:, -1]))
        self.assertTrue(torch.equal(eos, torch.stack([features[0, 1], features[1, 2]])))

    def test_text_transformer_padding_attention_mask(self) -> None:
        net = registry.text_factory(
            "text_transformer",
            config=self._small_text_config(causal_mask=False, pad_token_id=0),
            context_length=8,
        )
        tokens = torch.tensor([[1, 2, 9, 0, 0], [3, 4, 5, 9, 0]])
        attn_mask = net._pad_attention_mask(tokens)

        self.assertIsNotNone(attn_mask)
        self.assertEqual(attn_mask.dtype, torch.bool)
        self.assertSequenceEqual(attn_mask.shape, (2, 1, 1, 5))
        self.assertTrue(attn_mask[0][0][0][0].item())
        self.assertFalse(attn_mask[0][0][0][-1].item())

    def test_text_transformer_pool_types_forward(self) -> None:
        tokens = torch.tensor([[1, 2, 9, 0, 0], [3, 4, 5, 9, 0]])
        for pool_type in ("first", "last", "eos"):
            with self.subTest(pool_type=pool_type):
                net = registry.text_factory(
                    "text_transformer",
                    config=self._small_text_config(pool_type=pool_type),
                    context_length=8,
                )
                out = net(tokens)

                self.assertSequenceEqual(out.shape, (2, 12))
                self.assertTrue(torch.isfinite(out).all())


class TestCLIPNetworks(unittest.TestCase):
    @staticmethod
    def _small_clip_config() -> dict[str, Any]:
        return {
            "image": {
                "network": "vit_s16",
                "size": (64, 64),
            },
            "text": {
                "network": "text_transformer",
                "config": TestTextNetworks._small_text_config(),
                "context_length": 8,
            },
            "embed_dim": 12,
            "tokenizer": "openai_clip_bpe",
        }

    def test_clip_model_forward(self) -> None:
        net = registry.net_factory("clip", config=self._small_clip_config())

        image = torch.rand(2, 3, 64, 64)
        text = torch.tensor([[1, 2, 9, 0, 0], [3, 4, 5, 9, 0]])
        logits = net(image, text)

        self.assertSequenceEqual(logits.shape, (2, 2))
        self.assertTrue(torch.isfinite(logits).all())

        # Non-equal batch size
        image = torch.rand(1, 3, 64, 64)
        text = torch.tensor([[1, 2, 9, 0, 0], [3, 4, 5, 9, 0], [1, 2, 5, 9, 0]])
        logits = net(image, text)

        self.assertSequenceEqual(logits.shape, (1, 3))
        self.assertTrue(torch.isfinite(logits).all())

    def test_clip_model_forward_features(self) -> None:
        net = registry.net_factory("clip", config=self._small_clip_config())

        image = torch.rand(2, 3, 64, 64)
        text = torch.tensor([[1, 2, 9, 0, 0], [3, 4, 5, 9, 0]])
        out = net(image, text, return_features=True)

        self.assertSetEqual(set(out.keys()), {"image_features", "text_features", "logit_scale", "logit_bias"})
        self.assertSequenceEqual(out["image_features"].shape, (2, 12))
        self.assertSequenceEqual(out["text_features"].shape, (2, 12))
        self.assertEqual(out["logit_scale"].ndim, 1)
        self.assertIsNone(out["logit_bias"])
        self.assertTrue(torch.isfinite(out["image_features"]).all())
        self.assertTrue(torch.isfinite(out["text_features"]).all())

    def test_clip_model_input_channels(self) -> None:
        config = self._small_clip_config()
        config["image"]["input_channels"] = 1
        net = registry.net_factory("clip", config=config)

        image = torch.rand(2, 1, 64, 64)
        text = torch.tensor([[1, 2, 9, 0, 0], [3, 4, 5, 9, 0]])
        logits = net(image, text)

        self.assertEqual(net.image_encoder.input_channels, 1)
        self.assertSequenceEqual(logits.shape, (2, 2))
        self.assertTrue(torch.isfinite(logits).all())
