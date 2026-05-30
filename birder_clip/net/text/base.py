import copy
from typing import Any
from typing import Optional

import torch
from torch import nn

from birder_clip.model_registry import Task


class TextBaseNet(nn.Module):
    task = str(Task.TEXT)

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

        self.embedding_size: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
