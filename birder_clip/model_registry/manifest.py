from typing import Any
from typing import Literal
from typing import NotRequired
from typing import TypeAlias
from typing import TypedDict

FileFormatType = Literal["pt", "safetensors"]

FormatInfoType = TypedDict(
    "FormatInfoType",
    {"file_size": float, "sha256": str},
)

EncoderInfoType = TypedDict(
    "EncoderInfoType",
    {
        "network": str,
        "config": NotRequired[dict[str, Any]],
        "input_channels": NotRequired[int],
        "num_classes": NotRequired[int],
        "size": NotRequired[tuple[int, int]],
    },
)

EncoderMetadataType: TypeAlias = str | EncoderInfoType

NetworkInfoType = TypedDict(
    "NetworkInfoType",
    {
        "network": str,
        "tag": NotRequired[str],
        "image_encoder": NotRequired[EncoderMetadataType],
        "text_encoder": NotRequired[EncoderMetadataType],
        "embed_dim": NotRequired[int],
        "tokenizer": NotRequired[str],
    },
)

ModelMetadataType = TypedDict(
    "ModelMetadataType",
    {
        "url": NotRequired[str],
        "description": str,
        "resolution": tuple[int, int],
        "context_length": int,
        "formats": dict[FileFormatType, FormatInfoType],
        "net": NetworkInfoType,
        "task": NotRequired[str],
    },
)

REGISTRY_MANIFEST: dict[str, ModelMetadataType] = {}
