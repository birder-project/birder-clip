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

ImageEncoderInfoType = TypedDict(
    "ImageEncoderInfoType",
    {
        "network": str,
        "config": NotRequired[dict[str, Any]],
        "input_channels": NotRequired[int],
        "num_classes": NotRequired[int],
        "size": NotRequired[tuple[int, int]],
    },
)
TextEncoderInfoType = TypedDict(
    "TextEncoderInfoType",
    {
        "network": str,
        "config": NotRequired[dict[str, Any]],
    },
)
TextDecoderInfoType = TypedDict(
    "TextDecoderInfoType",
    {
        "network": str,
        "config": NotRequired[dict[str, Any]],
    },
)

ImageEncoderMetadataType: TypeAlias = str | ImageEncoderInfoType
TextEncoderMetadataType: TypeAlias = str | TextEncoderInfoType
TextDecoderMetadataType: TypeAlias = str | TextDecoderInfoType

NetworkInfoType = TypedDict(
    "NetworkInfoType",
    {
        "network": str,
        "tag": NotRequired[str],
        "image_encoder": NotRequired[ImageEncoderMetadataType],
        "text_encoder": NotRequired[TextEncoderMetadataType],
        "text_decoder": NotRequired[TextDecoderMetadataType],
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
