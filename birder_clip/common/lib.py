import copy
from typing import Any
from typing import Optional

from birder.data.transforms.classification import RGBType

from birder_clip.conf import settings
from birder_clip.model_registry import registry
from birder_clip.net.base import BaseNet
from birder_clip.net.base import SignatureType
from birder_clip.version import __version__

MODEL_CONFIG_RESERVED_KEYS = frozenset({"image", "text", "tokenizer", "embed_dim", "embed-dim"})


def get_size_from_signature(signature: SignatureType) -> tuple[int, int]:
    return tuple(signature["inputs"][0]["data_shape"][2:4])  # type: ignore[return-value]


def get_channels_from_signature(signature: SignatureType) -> int:
    return signature["inputs"][0]["data_shape"][1]


def get_context_length_from_signature(signature: SignatureType) -> int:
    return signature["inputs"][1]["data_shape"][1]


def get_reserved_model_config_keys(config: Optional[dict[str, Any]]) -> list[str]:
    if config is None:
        return []

    return sorted(MODEL_CONFIG_RESERVED_KEYS.intersection(config.keys()))


def get_image_text_network_name(
    network: str,
    tag: Optional[str] = None,
    image_encoder: Optional[str] = None,
    text_encoder: Optional[str] = None,
    embed_dim: Optional[int] = None,
    tokenizer: Optional[str] = None,
) -> str:
    parts = [network]
    if image_encoder is not None:
        parts.append(image_encoder)
    if text_encoder is not None and text_encoder != "text_transformer":
        parts.append(text_encoder)

    if registry.exists(network) is True:
        default_tokenizer = registry.get_default_tokenizer(network)
    else:
        default_tokenizer = "simple_tokenizer"
    if default_tokenizer is None:
        default_tokenizer = "simple_tokenizer"

    if tokenizer is not None and tokenizer != default_tokenizer:
        parts.append(tokenizer)
    if embed_dim is not None:
        parts.append(f"d{embed_dim}")

    network_name = "_".join(parts)
    if tag is not None:
        network_name = f"{network_name}_{tag}"

    return network_name


def get_image_text_model_config(
    base_config: Optional[dict[str, Any]] = None,
    config: Optional[dict[str, Any]] = None,
    *,
    image_encoder: Optional[str] = None,
    text_encoder: Optional[str] = None,
    embed_dim: Optional[int] = None,
    tokenizer: Optional[str] = None,
    image_encoder_config: Optional[dict[str, Any]] = None,
    text_encoder_config: Optional[dict[str, Any]] = None,
    input_channels: Optional[int] = None,
    image_size: Optional[tuple[int, int]] = None,
    context_length: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    if base_config is not None:
        model_config = copy.deepcopy(base_config)
    else:
        model_config = {}

    if config is not None:
        for key, value in config.items():
            if key in {"image", "text"} and isinstance(value, dict):
                model_config[key] = {**model_config.get(key, {}), **value}
            else:
                model_config[key] = value

    if (
        image_encoder is not None
        or image_encoder_config is not None
        or input_channels is not None
        or image_size is not None
    ):
        image_config = model_config.get("image", {}).copy()
        if image_encoder is not None:
            # String encoder metadata replaces only the encoder name.
            image_config["network"] = image_encoder
        if image_encoder_config is not None:
            # Dict encoder metadata adds constructor args under "config".
            image_config["config"] = {**image_config.get("config", {}), **image_encoder_config}
        if input_channels is not None:
            image_config["input_channels"] = input_channels
        if image_size is not None:
            image_config["size"] = image_size

        model_config["image"] = image_config

    if text_encoder is not None or text_encoder_config is not None or context_length is not None:
        text_config = model_config.get("text", {}).copy()
        if text_encoder is not None:
            # String encoder metadata replaces only the encoder name.
            text_config["network"] = text_encoder
        if text_encoder_config is not None:
            # Dict encoder metadata adds constructor args under "config".
            text_config["config"] = {**text_config.get("config", {}), **text_encoder_config}
        if context_length is not None:
            text_config["context_length"] = context_length

        model_config["text"] = text_config

    if embed_dim is not None:
        model_config["embed_dim"] = embed_dim
    if tokenizer is not None:
        model_config["tokenizer"] = tokenizer

    if len(model_config) == 0:
        return None

    return model_config


def get_image_text_network_config(net: BaseNet, signature: SignatureType, rgb_stats: RGBType) -> dict[str, Any]:
    model_name = registry.get_model_base_name(net)
    registered_name = registry.get_registered_name(net)
    model_config = None
    if net.config is not None:
        model_config = net.config

    return {
        "birder_clip_version": __version__,
        "name": model_name,
        "registered_name": registered_name,
        "task": net.task,
        "model_config": model_config,
        "signature": signature,
        "rgb_stats": rgb_stats,
    }


def get_pretrained_model_url(weights: str, file_format: str) -> tuple[str, str]:
    model_metadata = registry.get_pretrained_metadata(weights)
    model_file = f"{weights}.{file_format}"
    base_url = model_metadata.get("url", settings.REGISTRY_BASE_UTL)
    url = f"{base_url}/{model_file}"

    return (model_file, url)
