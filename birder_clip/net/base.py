import copy
from typing import Any
from typing import Optional

import torch
from birder.net.base import BaseNet as ImageEncoder
from torch import nn

from birder_clip.model_registry import Task


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
        self.embedding_size: int
        self.tokenizer_name: str

    def encode_image(self, image: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def encode_text(self, text: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def forward_logits(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def adjust_image_size(self, new_size: tuple[int, int]) -> None:
        self.image_encoder.adjust_size(new_size)
