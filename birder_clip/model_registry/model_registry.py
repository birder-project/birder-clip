import fnmatch
import warnings
from enum import StrEnum
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import Optional
from typing import TypeAlias

from birder.model_registry import registry as birder_registry
from birder.model_registry.model_registry import group_sort

from birder_clip.model_registry import manifest
from birder_clip.tokenizers.registry import get_tokenizer_info

if TYPE_CHECKING is True:
    from birder_clip.net.base import BaseNet  # pylint: disable=cyclic-import
    from birder_clip.net.text.base import TextBaseNet  # pylint: disable=cyclic-import

    NetType: TypeAlias = type[BaseNet] | type[TextBaseNet]


class Task(StrEnum):
    IMAGE_TEXT = "image_text"
    TEXT = "text"


class ModelRegistry:
    def __init__(self) -> None:
        self.registered_configs: dict[str, "NetType"] = {}
        self._image_text_nets: dict[str, type["BaseNet"]] = {}
        self._text_nets: dict[str, type[TextBaseNet]] = {}
        self._pretrained_nets = manifest.REGISTRY_MANIFEST

    @property
    def all_nets(self) -> dict[str, "NetType"]:
        return {**self._image_text_nets, **self._text_nets}

    def _get_models_for_task(self, task: Task) -> dict[str, "NetType"]:
        if task == Task.IMAGE_TEXT:
            nets: dict[str, "NetType"] = self._image_text_nets
        elif task == Task.TEXT:
            nets = self._text_nets
        else:
            raise ValueError(f"Unsupported model task: {task}")

        return nets

    def _register_model(self, name: str, net_type: "NetType") -> None:
        name_key = name.lower()
        task = Task(net_type.task)
        nets = self._get_models_for_task(task)
        if name_key in self.all_nets and name_key not in nets:
            raise ValueError(f"Registered model name '{name}' collides with an existing registered model name")
        if name_key in nets:
            warnings.warn(f"Network '{name}' is already registered and will be overwritten", UserWarning)

        nets[name_key] = net_type

    def register_model_config(self, name: str, net_type: "NetType", *, config: Optional[dict[str, Any]] = None) -> None:
        name_key = name.lower()
        registered_net_type = type(name, (net_type,), {"config": config})
        self._register_model(name_key, registered_net_type)
        if name_key in self.registered_configs:
            warnings.warn(f"Registered config '{name}' is already registered and will be overwritten", UserWarning)

        self.registered_configs[name_key] = registered_net_type

    def register_weights(self, name: str, weights_info: manifest.ModelMetadataType) -> None:
        if name in self._pretrained_nets:
            warnings.warn(f"Weights '{name}' are already registered and will be overwritten", UserWarning)

        if "task" not in weights_info:
            weights_info["task"] = self.all_nets[weights_info["net"]["network"]].task

        manifest.REGISTRY_MANIFEST[name] = weights_info
        self._pretrained_nets[name] = weights_info

    def list_models(
        self,
        include_filter: Optional[str] = None,
        *,
        task: Optional[Task] = None,
        net_type: Optional[type | tuple[type, ...]] = None,
        net_type_op: Literal["AND", "OR"] = "AND",
    ) -> list[str]:
        nets = self.all_nets
        if task is not None:
            nets = self._get_models_for_task(task)

        if net_type is not None:
            if not isinstance(net_type, tuple):
                net_type = (net_type,)

            if net_type_op == "OR":
                nets = {name: t for name, t in nets.items() if issubclass(t, net_type) is True}
            elif net_type_op == "AND":
                nets = {name: t for name, t in nets.items() if all(issubclass(t, nt) for nt in net_type)}
            else:
                raise ValueError(f"Unknown op {net_type_op}")

        model_list = list(nets.keys())
        if include_filter is not None:
            model_list = fnmatch.filter(model_list, include_filter)

        return group_sort(model_list)

    def list_pretrained_models(self, include_filter: Optional[str] = None, task: Optional[Task] = None) -> list[str]:
        model_list = list(self._pretrained_nets.keys())
        if include_filter is not None:
            model_list = fnmatch.filter(model_list, include_filter)

        if task is not None:
            model_list = [name for name in model_list if self._pretrained_nets[name]["task"] == task]

        return group_sort(model_list)

    def exists(self, name: str, task: Optional[Task] = None, net_type: Optional[type] = None) -> bool:
        nets = self.all_nets
        if task is not None:
            nets = self._get_models_for_task(task)

        if net_type is not None:
            nets = {name: t for name, t in nets.items() if issubclass(t, net_type) is True}

        return name.lower() in nets

    def pretrained_exists(self, name: str) -> bool:
        return name in self._pretrained_nets

    def get_default_size(self, name: str, *, image_encoder: Optional[str] = None) -> tuple[int, int]:
        if image_encoder is not None:
            return birder_registry.get_default_size(image_encoder)

        net_type = self._image_text_nets[name.lower()]
        config = net_type.config  # type: ignore[misc]
        if config is not None:
            image_size = config.get("image", {}).get("size")
            if image_size is not None:
                return image_size  # type: ignore[no-any-return]

            image_encoder = config.get("image", {}).get("network")
            if image_encoder is not None:
                return birder_registry.get_default_size(image_encoder)

        raise ValueError(f"Default size is not available for {name}")

    def get_default_tokenizer(self, name: str) -> Optional[str]:
        net_type = self._image_text_nets[name.lower()]
        config = net_type.config  # type: ignore[misc]
        if config is None:
            return None

        return config.get("tokenizer")

    def get_default_context_length(self, name: str, *, tokenizer: Optional[str] = None) -> int:
        net_type = self._image_text_nets[name.lower()]
        config = net_type.config  # type: ignore[misc]
        if config is not None:
            text_config = config.get("text", {})
            context_length = text_config.get("context_length")
            if context_length is not None:
                return context_length  # type: ignore[no-any-return]

            if tokenizer is None:
                tokenizer = config.get("tokenizer")

        if tokenizer is not None:
            _, tokenizer_kwargs = get_tokenizer_info(tokenizer)
            context_length = tokenizer_kwargs.get("context_length")
            if context_length is not None:
                return context_length  # type: ignore[no-any-return]

        raise ValueError(f"Default context length is not available for {name}")

    def get_pretrained_metadata(self, name: str) -> manifest.ModelMetadataType:
        metadata = self._pretrained_nets[name]
        if "task" not in metadata:
            metadata["task"] = self.all_nets[metadata["net"]["network"]].task

        return metadata

    def text_factory(
        self, name: str, *, config: Optional[dict[str, Any]] = None, context_length: Optional[int] = None
    ) -> "TextBaseNet":
        name_key = name.lower()
        return self._text_nets[name_key](config=config, context_length=context_length)

    def net_factory(self, name: str, *, config: Optional[dict[str, Any]] = None) -> "BaseNet":
        name_key = name.lower()
        return self._image_text_nets[name_key](config=config)

    def _metadata_type_name(self, model: "BaseNet | TextBaseNet") -> str:
        cls = model.__class__
        bases = cls.__bases__
        if len(bases) > 1 and bases[0].__name__ == "FSDPModule":
            return bases[1].__name__.lower()

        return cls.__name__.lower()

    def get_model_base_name(self, model: "BaseNet | TextBaseNet") -> str:
        type_name = self._metadata_type_name(model)
        if type_name in self.registered_configs:
            type_name = self.registered_configs[type_name].__bases__[0].__name__.lower()

        return type_name

    def get_registered_name(self, model: "BaseNet | TextBaseNet") -> Optional[str]:
        type_name = self._metadata_type_name(model)
        if type_name in self.registered_configs:
            return type_name

        return None


registry = ModelRegistry()
list_pretrained_models = registry.list_pretrained_models
