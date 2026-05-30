import logging
import unittest
from typing import Any

from birder_clip.model_registry import Task
from birder_clip.model_registry import registry
from birder_clip.model_registry.model_registry import ModelRegistry
from birder_clip.net.base import BaseNet
from birder_clip.net.text.base import TextBaseNet

logging.disable(logging.CRITICAL)


class TestRegistry(unittest.TestCase):
    def test_registry_nets(self) -> None:
        for net in registry._image_text_nets.values():
            self.assertTrue(issubclass(net, BaseNet))

        for net in registry._text_nets.values():
            self.assertTrue(issubclass(net, TextBaseNet))

    def test_no_duplicates(self) -> None:
        all_names = []
        for net_name in registry._image_text_nets:
            all_names.append(net_name)

        for net_name in registry._text_nets:
            all_names.append(net_name)

        self.assertEqual(len(all_names), len(set(all_names)))


class TestModelRegistry(unittest.TestCase):
    @staticmethod
    def _small_text_config(**kwargs: Any) -> dict[str, Any]:
        config: dict[str, Any] = {
            "context_length": 8,
            "vocab_size": 10,
            "hidden_dim": 16,
            "num_heads": 4,
            "num_layers": 2,
            "output_dim": 12,
            "eos_token_id": 9,
        }
        config.update(kwargs)

        return config

    def test_model_registry(self) -> None:
        model_registry = ModelRegistry()
        model_registry.register_model_config("image_text_net", BaseNet, config={"a": 1})
        model_registry.register_model_config("text_net", TextBaseNet, config={"b": 2})

        self.assertListEqual(list(model_registry.all_nets.keys()), ["image_text_net", "text_net"])
        self.assertListEqual(list(model_registry._image_text_nets.keys()), ["image_text_net"])
        self.assertListEqual(list(model_registry._text_nets.keys()), ["text_net"])
        self.assertListEqual(model_registry.list_models(task=Task.IMAGE_TEXT), ["image_text_net"])
        self.assertListEqual(model_registry.list_models(task=Task.TEXT), ["text_net"])
        self.assertTrue(model_registry.exists("image_text_net", task=Task.IMAGE_TEXT))
        self.assertFalse(model_registry.exists("image_text_net", task=Task.TEXT))

    def test_cross_task_duplicate_names_not_allowed(self) -> None:
        model_registry = ModelRegistry()
        model_registry.register_model_config("shared_name", BaseNet, config={"a": 1})

        with self.assertRaises(ValueError):
            model_registry.register_model_config("shared_name", TextBaseNet, config={"b": 2})

    def test_registry_task_filters(self) -> None:
        self.assertIn("clip", registry.list_models(task=Task.IMAGE_TEXT))
        self.assertIn("text_transformer", registry.list_models(task=Task.TEXT))
        self.assertIn("clip", registry.list_models())
        self.assertIn("text_transformer", registry.list_models())
        self.assertTrue(registry.exists("clip", task=Task.IMAGE_TEXT))
        self.assertTrue(registry.exists("text_transformer", task=Task.TEXT))
