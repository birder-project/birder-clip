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
            embed_dim,
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
        return self.logit_scale.exp() * (image_features @ text_features.T)

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)

        return self.forward_logits(image_features, text_features)


registry.register_model_config("clip", CLIP, config={})
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
