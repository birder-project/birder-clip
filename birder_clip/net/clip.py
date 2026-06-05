import math
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from birder.model_registry import registry as birder_registry
from birder.net.base import BaseNet as ImageEncoder
from torch import nn

from birder_clip.model_registry import registry
from birder_clip.net.base import BaseNet
from birder_clip.net.text.base import TextBaseNet


class CLIP(BaseNet):
    def __init__(self, *, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        assert self.config is not None, "must set config"

        embed_dim: int = self.config["embed_dim"]
        image_config: dict[str, Any] = self.config["image"]
        text_config: dict[str, Any] = self.config["text"]
        tokenizer_name: str = self.config["tokenizer"]

        image_encoder_size: Optional[tuple[int, int]] = image_config.get("size", None)
        image_encoder: ImageEncoder = birder_registry.net_factory(
            image_config["network"],
            image_config.get("num_classes", embed_dim),
            config=image_config.get("config", None),
            size=image_encoder_size,
        )
        text_encoder = registry.text_factory(
            text_config["network"],
            config=text_config.get("config", None),
        )
        assert isinstance(text_encoder, TextBaseNet)
        assert text_encoder.embedding_size == embed_dim

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        if "init_logit_bias" in self.config:
            self.logit_bias = nn.Parameter(torch.ones([]) * self.config["init_logit_bias"])
        else:
            self.logit_bias = None

        self.embedding_size = embed_dim
        self.tokenizer_name = tokenizer_name

    def encode_image(self, image: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        features = self.image_encoder(image)
        if normalize is True:
            features = F.normalize(features, dim=-1)

        return features

    def encode_text(self, text: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        features = self.text_encoder(text)
        if normalize is True:
            features = F.normalize(features, dim=-1)

        return features

    def forward_logits(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        logits = self.logit_scale.exp() * (image_features @ text_features.T)
        if self.logit_bias is not None:
            logits = logits + self.logit_bias

        return logits

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)

        return self.forward_logits(image_features, text_features)


registry.register_model_config("clip", CLIP, config={})

# OpenAI CLIP - https://arxiv.org/abs/2103.00020
registry.register_model_config(
    "openai_clip_vit_l14",
    CLIP,
    config={
        "embed_dim": 768,
        "image": {
            "network": "vit_l14_pn_quick_gelu",
            "size": (224, 224),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "hidden_dim": 768,
                "num_heads": 12,
                "output_dim": 768,
                "act_layer_type": "quick_gelu",
            },
        },
        "tokenizer": "openai_clip_bpe",
    },
)

# PE Core - https://arxiv.org/abs/2504.13181
registry.register_model_config(
    "pe_core_s16",
    CLIP,
    config={
        "embed_dim": 512,
        "image": {
            "network": "rope_i_vit_s16_pn_aps_c1",
            "size": (384, 384),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "context_length": 32,
            },
        },
        "tokenizer": "pe_base_bpe",
    },
)
registry.register_model_config(
    "pe_core_b16",
    CLIP,
    config={
        "embed_dim": 1024,
        "image": {
            "network": "rope_i_vit_b16_pn_aps_c1",
            "size": (224, 224),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "context_length": 32,
                "hidden_dim": 1024,
                "num_heads": 16,
                "num_layers": 24,
                "output_dim": 1024,
            },
        },
        "tokenizer": "pe_base_bpe",
    },
)
registry.register_model_config(
    "pe_core_l14",
    CLIP,
    config={
        "embed_dim": 1024,
        "image": {
            "network": "rope_i_vit_l14_pn_aps_c1",
            "size": (336, 336),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "context_length": 32,
                "hidden_dim": 1024,
                "num_heads": 16,
                "num_layers": 24,
                "output_dim": 1024,
            },
        },
        "tokenizer": "pe_base_bpe",
    },
)

# SigLIP 2 - https://arxiv.org/abs/2502.14786
registry.register_model_config(
    "siglip_v2_vit_so400m_p14",
    CLIP,
    config={
        "embed_dim": 1152,
        "init_logit_bias": -10.0,
        "image": {
            "network": "vit_so400m_p14_ap",
            "num_classes": 0,
            "size": (224, 224),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "context_length": 64,
                "vocab_size": 256000,
                "hidden_dim": 1152,
                "num_heads": 16,
                "num_layers": 27,
                "mlp_dim": 4304,
                "output_dim": 1152,
                "causal_mask": False,
                "pool_type": "last",
                "proj_bias": True,
                "norm_layer_eps": 1e-6,
                "act_layer_type": "gelu_tanh",
            },
        },
        "tokenizer": "siglip2_gemma",
    },
)

# BioCLIP - https://arxiv.org/abs/2311.18803
registry.register_model_config(
    "bioclip_v1_vit_b16",
    CLIP,
    config={
        "embed_dim": 512,
        "image": {
            "network": "vit_b16_pn",
            "size": (224, 224),
        },
        "text": {
            "network": "text_transformer",
        },
        "tokenizer": "openai_clip_bpe",
    },
)

# BioCLIP 2 - https://arxiv.org/abs/2505.23883
registry.register_model_config(
    "bioclip_v2_vit_l14",
    CLIP,
    config={
        "embed_dim": 768,
        "image": {
            "network": "vit_l14_pn",
            "size": (224, 224),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "hidden_dim": 768,
                "num_heads": 12,
                "output_dim": 768,
            },
        },
        "tokenizer": "openai_clip_bpe",
    },
)
