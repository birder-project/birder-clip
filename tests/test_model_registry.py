import logging
import unittest

import requests
from birder.common.lib import env_bool

from birder_clip.common.lib import get_pretrained_model_url
from birder_clip.model_registry import Task
from birder_clip.model_registry import registry
from birder_clip.model_registry.model_registry import ModelRegistry
from birder_clip.net.base import BaseNet
from birder_clip.net.text.base import CrossAttentionTextDecoderBaseNet
from birder_clip.net.text.base import TextDecoderBaseNet
from birder_clip.net.text.base import TextEncoderBaseNet
from birder_clip.net.text.base import VisualCausalTextDecoderBaseNet

logging.disable(logging.CRITICAL)


class TestRegistry(unittest.TestCase):
    def test_registry_nets(self) -> None:
        for net in registry._image_text_nets.values():
            self.assertTrue(issubclass(net, BaseNet))

        for net in registry._text_encoder_nets.values():
            self.assertTrue(issubclass(net, TextEncoderBaseNet))

        for net in registry._text_decoder_nets.values():
            self.assertTrue(issubclass(net, TextDecoderBaseNet))

    def test_no_duplicates(self) -> None:
        all_names = []
        for net_name in registry._image_text_nets:
            all_names.append(net_name)

        for net_name in registry._text_encoder_nets:
            all_names.append(net_name)

        for net_name in registry._text_decoder_nets:
            all_names.append(net_name)

        self.assertEqual(len(all_names), len(set(all_names)))

    @unittest.skipUnless(env_bool("NETWORK_TESTS"), "Avoid tests that require network access")
    def test_manifest(self) -> None:
        for model_name, model_metadata in registry._pretrained_nets.items():
            for model_format in model_metadata["formats"]:
                _, url = get_pretrained_model_url(model_name, model_format)

                for _ in range(3):
                    try:
                        resp = requests.head(url, timeout=5, allow_redirects=True)
                        break
                    except requests.RequestException:
                        continue

                with self.subTest(model_name=model_name, model_format=model_format):
                    self.assertEqual(resp.status_code, 200, f"{model_name} not found at {url}")
                    self.assertGreater(int(resp.headers["Content-Length"]), 100000)


class TestModelRegistry(unittest.TestCase):
    def test_model_registry(self) -> None:
        model_registry = ModelRegistry()
        model_registry.register_model_config("image_text_net", BaseNet, config={"a": 1})
        model_registry.register_model_config("text_net", TextEncoderBaseNet, config={"b": 2})
        model_registry.register_model_config("decoder_net", TextDecoderBaseNet, config={"c": 3})

        self.assertListEqual(list(model_registry.all_nets.keys()), ["image_text_net", "text_net", "decoder_net"])
        self.assertListEqual(list(model_registry._image_text_nets.keys()), ["image_text_net"])
        self.assertListEqual(list(model_registry._text_encoder_nets.keys()), ["text_net"])
        self.assertListEqual(list(model_registry._text_decoder_nets.keys()), ["decoder_net"])
        self.assertListEqual(model_registry.list_models(task=Task.IMAGE_TEXT), ["image_text_net"])
        self.assertListEqual(model_registry.list_models(task=Task.TEXT_ENCODER), ["text_net"])
        self.assertListEqual(model_registry.list_models(task=Task.TEXT_DECODER), ["decoder_net"])
        self.assertListEqual(
            model_registry.list_models(task=Task.TEXT_DECODER, net_type=TextDecoderBaseNet),
            ["decoder_net"],
        )
        self.assertTrue(model_registry.exists("image_text_net", task=Task.IMAGE_TEXT))
        self.assertFalse(model_registry.exists("image_text_net", task=Task.TEXT_ENCODER))
        self.assertTrue(model_registry.exists("decoder_net", task=Task.TEXT_DECODER))

        with self.assertWarns(UserWarning):
            model_registry.register_model_config("image_text_net", BaseNet, config={"a": 2})

    def test_cross_task_duplicate_names_not_allowed(self) -> None:
        model_registry = ModelRegistry()
        model_registry.register_model_config("shared_name", BaseNet, config={"a": 1})

        with self.assertRaises(ValueError):
            model_registry.register_model_config("shared_name", TextEncoderBaseNet, config={"b": 2})

    def test_registry_task_filters(self) -> None:
        self.assertIn("clip", registry.list_models(task=Task.IMAGE_TEXT))
        self.assertIn("transformer_encoder", registry.list_models(task=Task.TEXT_ENCODER))
        self.assertIn("conditioned_decoder", registry.list_models(task=Task.TEXT_DECODER))
        self.assertIn("visual_causal_text_decoder", registry.list_models(task=Task.TEXT_DECODER))
        self.assertIn(
            "conditioned_decoder",
            registry.list_models(task=Task.TEXT_DECODER, net_type=CrossAttentionTextDecoderBaseNet),
        )
        self.assertIn(
            "visual_causal_text_decoder",
            registry.list_models(task=Task.TEXT_DECODER, net_type=VisualCausalTextDecoderBaseNet),
        )
        self.assertIn("clip", registry.list_models())
        self.assertIn("transformer_encoder", registry.list_models())
        self.assertIn("conditioned_decoder", registry.list_models())
        self.assertIn("visual_causal_text_decoder", registry.list_models())
        self.assertTrue(registry.exists("clip", task=Task.IMAGE_TEXT))
        self.assertTrue(registry.exists("transformer_encoder", task=Task.TEXT_ENCODER))
        self.assertTrue(registry.exists("conditioned_decoder", task=Task.TEXT_DECODER))
        self.assertTrue(registry.exists("visual_causal_text_decoder", task=Task.TEXT_DECODER))
        self.assertTrue(
            registry.exists("conditioned_decoder", task=Task.TEXT_DECODER, net_type=CrossAttentionTextDecoderBaseNet)
        )
        self.assertTrue(
            registry.exists(
                "visual_causal_text_decoder", task=Task.TEXT_DECODER, net_type=VisualCausalTextDecoderBaseNet
            )
        )

    def test_model_defaults(self) -> None:
        self.assertEqual(registry.get_default_size("openvision_v1_vit_s16"), (224, 224))
        self.assertEqual(registry.get_default_tokenizer("openvision_v1_vit_s16"), "openvision_v1")
        self.assertEqual(registry.get_default_context_length("openvision_v1_vit_s16"), 80)
