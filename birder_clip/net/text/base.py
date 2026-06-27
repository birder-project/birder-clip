import copy
import logging
from typing import Any
from typing import ClassVar
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from birder_clip.model_registry import Task

logger = logging.getLogger(__name__)


def adjust_position_embedding(pos_embedding: torch.Tensor, new_context_length: int) -> torch.Tensor:
    orig_dtype = pos_embedding.dtype
    pos_embedding = pos_embedding.float()
    pos_embedding = pos_embedding.permute(0, 2, 1)
    pos_embedding = F.interpolate(pos_embedding, size=new_context_length, mode="linear", align_corners=True)
    pos_embedding = pos_embedding.permute(0, 2, 1)

    return pos_embedding.to(orig_dtype)


class TextEncoderBaseNet(nn.Module):
    default_context_length: ClassVar[int] = 77
    block_group_regex: ClassVar[Optional[str]] = None
    task: ClassVar[str] = str(Task.TEXT_ENCODER)

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

        self.vocab_size: int
        self.hidden_dim: int
        self.embedding_size: int

    def set_grad_checkpointing(
        self,
        enable: bool = True,
        *,
        segments: Optional[int] = None,
        preserve_rng_state: bool = True,
        use_reentrant: bool = False,
    ) -> None:
        raise RuntimeError(f"{self.__class__.__name__} does not support gradient checkpointing")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_head(self, features: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def tokens_from_features(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.forward_head(features, x)

    def adjust_context_length(self, new_context_length: int) -> None:
        if new_context_length == self.context_length:
            return

        logger.info(f"Adjusting model context length from {self.context_length} to {new_context_length}")
        self.context_length = new_context_length


class TextDecoderBaseNet(nn.Module):
    default_context_length: ClassVar[int] = 77
    block_group_regex: ClassVar[Optional[str]] = None
    task: ClassVar[str] = str(Task.TEXT_DECODER)

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

        self.vocab_size: int
        self.hidden_dim: int
        self.kv_dim: int

    def set_grad_checkpointing(
        self,
        enable: bool = True,
        *,
        segments: Optional[int] = None,
        preserve_rng_state: bool = True,
        use_reentrant: bool = False,
    ) -> None:
        raise RuntimeError(f"{self.__class__.__name__} does not support gradient checkpointing")

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def adjust_context_length(self, new_context_length: int) -> None:
        if new_context_length == self.context_length:
            return

        logger.info(f"Adjusting decoder context length from {self.context_length} to {new_context_length}")
        self.context_length = new_context_length


class CrossAttentionTextDecoderBaseNet(TextDecoderBaseNet):
    def forward_features_from_embeddings(
        self, text_embeddings: torch.Tensor, image_tokens: torch.Tensor, cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward_from_embeddings(
        self, text_embeddings: torch.Tensor, image_tokens: torch.Tensor, cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        features = self.forward_features_from_embeddings(text_embeddings, image_tokens, cross_attn_mask=cross_attn_mask)
        return self.forward_head(features)

    def forward(
        self, text_embeddings: torch.Tensor, image_tokens: torch.Tensor, cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.forward_from_embeddings(text_embeddings, image_tokens, cross_attn_mask=cross_attn_mask)


class VisualCausalTextDecoderBaseNet(TextDecoderBaseNet):
    def forward_features(self, text: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_features_from_embeddings(
        self, text_embeddings: torch.Tensor, image_tokens: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward_from_embeddings(self, text_embeddings: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        features = self.forward_features_from_embeddings(text_embeddings, image_tokens)
        return self.forward_head(features)

    def forward(self, text: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(text, image_tokens)
        return self.forward_head(features)
