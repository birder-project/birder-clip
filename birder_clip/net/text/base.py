import copy
import logging
from typing import Any
from typing import Optional

import torch
from torch import nn

from birder_clip.model_registry import Task

logger = logging.getLogger(__name__)


class TextBaseNet(nn.Module):
    default_context_length = 77
    block_group_regex: Optional[str]
    task = str(Task.TEXT)

    def __init__(self, *, config: Optional[dict[str, Any]] = None, context_length: Optional[int] = None) -> None:
        super().__init__()
        if hasattr(self, "config") is False:
            self.config = config
        else:
            if self.config is not None:
                self.config = copy.deepcopy(self.config)  # Avoid mutating registered configs

        if config is not None:
            assert self.config is not None
            self.config.update(config)  # Override with custom config

        if context_length is not None:
            self.context_length = context_length
        else:
            self.context_length = self.default_context_length

        self.embedding_size: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def adjust_context_length(self, new_context_length: int) -> None:
        if new_context_length == self.context_length:
            return

        logger.info(f"Adjusting model context length from {self.context_length} to {new_context_length}")
        self.context_length = new_context_length
