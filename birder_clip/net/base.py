import copy
from typing import Any
from typing import Optional
from typing import TypedDict

import torch
from birder.net.base import BaseNet as ImageEncoder
from torch import nn

from birder_clip.model_registry import Task
from birder_clip.net.text.base import TextBaseNet

DataShapeType = TypedDict("DataShapeType", {"data_shape": list[int]})
SignatureType = TypedDict("SignatureType", {"inputs": list[DataShapeType], "outputs": list[DataShapeType]})


def get_image_text_signature(input_shape: tuple[int, ...], context_length: int) -> SignatureType:
    return {
        "inputs": [
            {"data_shape": [0, *input_shape[1:]]},
            {"data_shape": [0, context_length]},
        ],
        "outputs": [{"data_shape": [0, 0]}],
    }


class BaseNet(nn.Module):
    task = str(Task.IMAGE_TEXT)

    def __init__(self, *, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__()
        if hasattr(self, "config") is False:
            self.config = config
        else:
            if self.config is not None:
                self.config = copy.deepcopy(self.config)  # Avoid mutating registered configs

        if config is not None:
            assert self.config is not None
            self.config.update(config)  # Override with custom config

        self.image_encoder: ImageEncoder
        self.text_encoder: TextBaseNet
        self.embedding_size: int
        self.tokenizer_name: str

    def encode_image(self, image: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def encode_text(self, text: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def forward_logits(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self, image: torch.Tensor, text: torch.Tensor, *, return_features: bool = False
    ) -> torch.Tensor | dict[str, Optional[torch.Tensor]]:
        raise NotImplementedError

    def adjust_image_size(self, new_size: tuple[int, int]) -> None:
        self.image_encoder.adjust_size(new_size)

    def adjust_context_length(self, new_context_length: int) -> None:
        self.text_encoder.adjust_context_length(new_context_length)
