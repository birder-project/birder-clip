import logging
from typing import Any
from typing import Literal
from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig
from transformers import AutoModel

from birder_clip.model_registry import registry
from birder_clip.net.text.base import TextEncoderBaseNet

logger = logging.getLogger(__name__)


class HFTextEncoder(TextEncoderBaseNet):
    def __init__(self, *, config: Optional[dict[str, Any]] = None, context_length: Optional[int] = None) -> None:
        super().__init__(config=config, context_length=context_length)
        assert self.config is not None, "must set config"

        source: str = self.config["source"]
        hidden_dim: int = self.config["hidden_dim"]
        output_dim: int = self.config.get("output_dim", hidden_dim)
        pool_type: Literal["first", "last", "mean", "eos"] = self.config.get("pool_type", "first")
        proj_bias: bool = self.config.get("proj_bias", False)
        pad_token_id: Optional[int] = self.config.get("pad_token_id", None)
        eos_token_id: Optional[int] = self.config.get("eos_token_id", None)
        trust_remote_code: bool = self.config.get("trust_remote_code", False)
        local_files_only: bool = self.config.get("local_files_only", False)
        cache_dir: Optional[str] = self.config.get("cache_dir", None)
        hf_config_overrides: dict[str, Any] = self.config.get("hf_config", {})
        hf_model_kwargs: dict[str, Any] = self.config.get("model_kwargs", {})

        if pool_type not in ("first", "last", "mean", "eos"):
            raise ValueError(f"Unknown pool_type '{pool_type}'")

        hf_config = AutoConfig.from_pretrained(  # nosec B615
            source,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            **hf_config_overrides,
        )
        if pad_token_id is None:
            pad_token_id = hf_config.pad_token_id
        if eos_token_id is None:
            eos_token_id = hf_config.eos_token_id

        self.source = source
        self.vocab_size = hf_config.vocab_size
        self.hidden_dim = hidden_dim
        self.pool_type = pool_type
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        self.hf_model_kwargs = hf_model_kwargs

        self.model = AutoModel.from_config(hf_config, trust_remote_code=trust_remote_code, **hf_model_kwargs)
        self.text_projection = nn.Linear(hidden_dim, output_dim, bias=proj_bias)
        self.embedding_size = output_dim

        nn.init.normal_(self.text_projection.weight, std=hidden_dim**-0.5)
        if self.text_projection.bias is not None:
            nn.init.zeros_(self.text_projection.bias)

    def load_pretrained(self) -> None:
        self.model = AutoModel.from_pretrained(  # nosec B615
            self.source,
            config=self.model.config,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
            cache_dir=self.cache_dir,
            **self.hf_model_kwargs,
        )

    def set_grad_checkpointing(
        self,
        enable: bool = True,
        *,
        segments: Optional[int] = None,
        preserve_rng_state: bool = True,
        use_reentrant: bool = False,
    ) -> None:
        if enable is True:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
        else:
            self.model.gradient_checkpointing_disable()

    def adjust_context_length(self, new_context_length: int) -> None:
        if new_context_length == self.context_length:
            return

        logger.warning("Context length adjustment is not supported for HF text encoders")
        super().adjust_context_length(new_context_length)

    def _attention_mask(self, text: torch.Tensor) -> Optional[torch.Tensor]:
        if self.pad_token_id is None:
            return None

        return text.ne(self.pad_token_id)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=x, attention_mask=self._attention_mask(x))
        return outputs.last_hidden_state

    def forward_head(self, features: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "first":
            pooled = features[:, 0]

        elif self.pool_type == "last":
            if self.pad_token_id is None:
                pooled = features[:, -1]
            else:
                indices = text.ne(self.pad_token_id).sum(dim=-1).sub(1).clamp_min(0)
                pooled = features[torch.arange(features.size(0), device=features.device), indices]

        elif self.pool_type == "mean":
            attention_mask = self._attention_mask(text)
            if attention_mask is None:
                pooled = features.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).to(dtype=features.dtype)
                pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        elif self.pool_type == "eos":
            if self.eos_token_id is None:
                raise RuntimeError("pool_type='eos' requires eos_token_id")
            indices = text.eq(self.eos_token_id).int().argmax(dim=-1)
            pooled = features[torch.arange(features.size(0), device=features.device), indices]

        else:
            raise RuntimeError(f"Unknown pool_type '{self.pool_type}'")

        return self.text_projection(pooled)

    def tokens_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return features


registry.register_model_config("hf_text_encoder", HFTextEncoder, config={})

registry.register_model_config(
    "hf:bert-base-uncased",
    HFTextEncoder,
    config={
        "source": "bert-base-uncased",
        "hidden_dim": 768,
        "model_kwargs": {"add_pooling_layer": False},
    },
)

registry.register_model_config(
    "hf:roberta-base",
    HFTextEncoder,
    config={
        "source": "roberta-base",
        "hidden_dim": 768,
        "pool_type": "mean",
        "model_kwargs": {"add_pooling_layer": False},
    },
)
