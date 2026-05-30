from typing import Any
from typing import Optional


def get_size_from_signature(signature: dict[str, Any]) -> tuple[int, int]:
    return tuple(signature["inputs"][0]["data_shape"][2:4])


def get_channels_from_signature(signature: dict[str, Any]) -> int:
    return signature["inputs"][0]["data_shape"][1]  # type: ignore[no-any-return]


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
    if text_encoder is not None:
        parts.append(text_encoder)
    if tokenizer is not None:
        parts.append(tokenizer)
    if embed_dim is not None:
        parts.append(f"d{embed_dim}")

    network_name = "_".join(parts)
    if tag is not None:
        network_name = f"{network_name}_{tag}"

    return network_name


def get_image_text_network_config(
    image_encoder: str,
    text_encoder: str,
    embed_dim: int,
    tokenizer: Optional[str],
    *,
    image_size: Optional[tuple[int, int]] = None,
    image_config: Optional[dict[str, Any]] = None,
    text_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    image: dict[str, Any] = {"network": image_encoder}
    if image_size is not None:
        image["size"] = image_size
    if image_config is not None:
        image["config"] = image_config

    text: dict[str, Any] = {"network": text_encoder}
    if text_config is not None:
        text["config"] = text_config

    network_config = {
        "image": image,
        "text": text,
        "embed_dim": embed_dim,
        "tokenizer": tokenizer,
    }

    return network_config
