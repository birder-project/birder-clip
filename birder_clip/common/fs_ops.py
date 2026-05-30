import logging
from pathlib import Path
from typing import Any
from typing import NamedTuple
from typing import Optional

import torch
from birder.conf import settings
from birder.data.transforms.classification import RGBType

from birder_clip.common import lib
from birder_clip.model_registry import registry
from birder_clip.net.base import BaseNet

logger = logging.getLogger(__name__)


class ModelInfo(NamedTuple):
    signature: dict[str, Any]
    rgb_stats: RGBType
    custom_config: Optional[dict[str, Any]] = None


def model_path(network_name: str, *, epoch: Optional[int | str] = None) -> Path:
    if epoch is not None:
        file_name = f"{network_name}_{epoch}.pt"
    else:
        file_name = f"{network_name}.pt"

    return settings.MODELS_DIR.joinpath(file_name)


def load_model(
    device: torch.device,
    network: str,
    *,
    path: Optional[str | Path] = None,
    config: Optional[dict[str, Any]] = None,
    tag: Optional[str] = None,
    image_encoder: Optional[str] = None,
    text_encoder: Optional[str] = None,
    embed_dim: Optional[int] = None,
    tokenizer: Optional[str] = None,
    image_encoder_config: Optional[dict[str, Any]] = None,
    text_encoder_config: Optional[dict[str, Any]] = None,
    epoch: Optional[int] = None,
    new_size: Optional[tuple[int, int]] = None,
    inference: bool,
    dtype: Optional[torch.dtype] = None,
) -> tuple[BaseNet, ModelInfo]:
    if path is None:
        _network_name = lib.get_image_text_network_name(
            network,
            tag=tag,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            embed_dim=embed_dim,
            tokenizer=tokenizer,
        )
        path = model_path(_network_name, epoch=epoch)

    logger.info(f"Loading model from {path} on device {device}...")
    model_dict: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)

    loaded_config: dict[str, Any] = model_dict.get("config", {})
    merged_config = {**loaded_config}
    if tokenizer is not None:
        tokenizer_name: Optional[str] = tokenizer
        merged_config["tokenizer"] = tokenizer
    else:
        tokenizer_name = loaded_config.get("tokenizer")

    if image_encoder is not None and text_encoder is not None and embed_dim is not None and tokenizer_name is not None:
        image_size = loaded_config.get("image", {}).get("size")
        merged_config.update(
            lib.get_image_text_network_config(
                image_encoder,
                text_encoder,
                embed_dim,
                tokenizer_name,
                image_size=image_size,
                image_config=image_encoder_config,
                text_config=text_encoder_config,
            )
        )
    if config is not None:
        merged_config.update(config)
    if len(merged_config) == 0:
        merged_config = None  # type: ignore[assignment]

    net = registry.net_factory(network, config=merged_config)
    net.load_state_dict(model_dict["state"])
    if new_size is not None:
        net.adjust_image_size(new_size)

    net.to(device)
    if dtype is not None:
        net.to(dtype)
    if inference is True:
        for param in net.parameters():
            param.requires_grad_(False)

        net.eval()

    if len(loaded_config) == 0:
        custom_config = None
    else:
        custom_config = loaded_config
        logger.debug(f"Model loaded with custom config: {custom_config}")

    return (net, ModelInfo(model_dict["signature"], model_dict["rgb_stats"], custom_config))
