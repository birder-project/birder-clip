import math
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from birder.conf import settings
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

        image_config: dict[str, Any] = self.config["image"]
        text_config: dict[str, Any] = self.config["text"]
        embed_dim: int = self.config["embed_dim"]
        tokenizer_name: str = self.config["tokenizer"]

        image_encoder: ImageEncoder = birder_registry.net_factory(
            image_config["network"],
            image_config.get("num_classes", embed_dim),
            image_config.get("input_channels", settings.DEFAULT_NUM_CHANNELS),
            config=image_config.get("config", None),
            size=image_config.get("size", None),
        )
        text_encoder = registry.text_factory(
            text_config["network"],
            config=text_config.get("config", None),
            context_length=text_config.get("context_length", None),
        )
        assert isinstance(text_encoder, TextBaseNet)
        assert text_encoder.embedding_size == embed_dim

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(torch.ones(1) * math.log(1 / 0.07))
        if "init_logit_bias" in self.config:
            self.logit_bias = nn.Parameter(torch.ones(1) * self.config["init_logit_bias"])
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

    def forward(
        self, image: torch.Tensor, text: torch.Tensor, *, return_features: bool = False
    ) -> torch.Tensor | dict[str, Optional[torch.Tensor]]:
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        if return_features is True:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp(),
                "logit_bias": self.logit_bias,
            }

        return self.forward_logits(image_features, text_features)


registry.register_model_config("clip", CLIP, config={})

# OpenAI CLIP - https://arxiv.org/abs/2103.00020
registry.register_model_config(
    "openai_clip_vit_l14",
    CLIP,
    config={
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
        "embed_dim": 768,
        "tokenizer": "openai_clip_bpe",
    },
)

# LAION CLIP - laion/CLIP-ViT-L-14-laion2B-s32B-b82K
registry.register_model_config(
    "laion_clip_vit_l14",
    CLIP,
    config={
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
        "embed_dim": 768,
        "tokenizer": "openai_clip_bpe",
    },
)

# LAION CLIP - laion/CLIP-ViT-H-14-laion2B-s32B-b79K
registry.register_model_config(
    "laion_clip_vit_h14",
    CLIP,
    config={
        "image": {
            "network": "vit_h14_pn",
            "size": (224, 224),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "hidden_dim": 1024,
                "num_heads": 16,
                "num_layers": 24,
                "output_dim": 1024,
            },
        },
        "embed_dim": 1024,
        "tokenizer": "openai_clip_bpe",
    },
)

# LAION CLIP - laion/CLIP-convnext_base_w-laion2B-s13B-b82K
registry.register_model_config(
    "laion_clip_convnext_v1_base",
    CLIP,
    config={
        "image": {
            "network": "convnext_v1_base",
            "config": {"drop_path_rate": 0.1},
            "size": (320, 320),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "hidden_dim": 640,
                "num_heads": 10,
                "output_dim": 640,
            },
        },
        "embed_dim": 640,
        "tokenizer": "openai_clip_bpe",
    },
)

# OpenVision - https://arxiv.org/abs/2505.04601
registry.register_model_config(
    "openvision_v1_vit_b16",
    CLIP,
    config={
        "image": {
            "network": "vit_reg1_b16_nap_avg",
            "config": {"drop_path_rate": 0.0},
            "size": (384, 384),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "vocab_size": 32000,
                "causal_mask": False,
                "pool_type": "last",
                "norm_layer_eps": 1e-6,
                "act_layer_type": "gelu_tanh",
            },
            "context_length": 80,
        },
        "embed_dim": 512,
        "tokenizer": "openvision",
    },
)
registry.register_model_config(
    "openvision_v1_vit_l14",
    CLIP,
    config={
        "image": {
            "network": "vit_reg1_l14_nap_avg",
            "config": {"drop_path_rate": 0.0},
            "size": (224, 224),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "vocab_size": 32000,
                "hidden_dim": 768,
                "num_heads": 12,
                "output_dim": 768,
                "causal_mask": False,
                "pool_type": "last",
                "norm_layer_eps": 1e-6,
                "act_layer_type": "gelu_tanh",
            },
            "context_length": 80,
        },
        "embed_dim": 768,
        "tokenizer": "openvision",
    },
)

# PE Core - https://arxiv.org/abs/2504.13181
registry.register_model_config(
    "pe_core_s16",
    CLIP,
    config={
        "image": {
            "network": "rope_i_vit_s16_pn_aps_c1",
            "size": (384, 384),
        },
        "text": {
            "network": "text_transformer",
            "context_length": 32,
        },
        "embed_dim": 512,
        "tokenizer": "pe_base_bpe",
    },
)
registry.register_model_config(
    "pe_core_b16",
    CLIP,
    config={
        "image": {
            "network": "rope_i_vit_b16_pn_aps_c1",
            "size": (224, 224),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "hidden_dim": 1024,
                "num_heads": 16,
                "num_layers": 24,
                "output_dim": 1024,
            },
            "context_length": 32,
        },
        "embed_dim": 1024,
        "tokenizer": "pe_base_bpe",
    },
)
registry.register_model_config(
    "pe_core_l14",
    CLIP,
    config={
        "image": {
            "network": "rope_i_vit_l14_pn_aps_c1",
            "size": (336, 336),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "hidden_dim": 1024,
                "num_heads": 16,
                "num_layers": 24,
                "output_dim": 1024,
            },
            "context_length": 32,
        },
        "embed_dim": 1024,
        "tokenizer": "pe_base_bpe",
    },
)

# SigLIP - https://arxiv.org/abs/2303.15343
registry.register_model_config(
    "siglip_v1_vit_b16",
    CLIP,
    config={
        "image": {
            "network": "vit_b16_ap",  # NOTE: Change to vit_b16_ap_c1 when next version of Birder released
            "num_classes": 0,
            "size": (384, 384),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "vocab_size": 32000,
                "hidden_dim": 768,
                "num_heads": 12,
                "num_layers": 12,
                "output_dim": 768,
                "causal_mask": False,
                "pool_type": "last",
                "proj_bias": True,
                "norm_layer_eps": 1e-6,
            },
            "context_length": 64,
        },
        "embed_dim": 768,
        "tokenizer": "siglip_t5",
        "init_logit_bias": -10.0,
    },
)

# SigLIP 2 - https://arxiv.org/abs/2502.14786
registry.register_model_config(
    "siglip_v2_vit_l16",
    CLIP,
    config={
        "image": {
            "network": "vit_l16_ap_c1",
            "num_classes": 0,
            "size": (256, 256),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "vocab_size": 256000,
                "hidden_dim": 1024,
                "num_heads": 16,
                "num_layers": 24,
                "output_dim": 1024,
                "causal_mask": False,
                "pool_type": "last",
                "proj_bias": True,
                "norm_layer_eps": 1e-6,
                "act_layer_type": "gelu_tanh",
            },
            "context_length": 64,
        },
        "embed_dim": 1024,
        "tokenizer": "siglip2_gemma",
        "init_logit_bias": -10.0,
    },
)
registry.register_model_config(
    "siglip_v2_vit_so400m_p14",
    CLIP,
    config={
        "image": {
            "network": "vit_so400m_p14_ap_c1",
            "num_classes": 0,
            "size": (224, 224),
        },
        "text": {
            "network": "text_transformer",
            "config": {
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
            "context_length": 64,
        },
        "embed_dim": 1152,
        "tokenizer": "siglip2_gemma",
        "init_logit_bias": -10.0,
    },
)

# MobileCLIP 2 - https://arxiv.org/abs/2508.20691
registry.register_model_config(
    "mobileclip_v2_s0",
    CLIP,
    config={
        "image": {
            "network": "mobileclip_v1_i0",
            "size": (256, 256),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "causal_mask": False,
            },
        },
        "embed_dim": 512,
        "tokenizer": "openai_clip_bpe",
    },
)
registry.register_model_config(
    "mobileclip_v2_s2",
    CLIP,
    config={
        "image": {
            "network": "mobileclip_v1_i2",
            "size": (256, 256),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "causal_mask": False,
            },
        },
        "embed_dim": 512,
        "tokenizer": "openai_clip_bpe",
    },
)
registry.register_model_config(
    "mobileclip_v2_s3",
    CLIP,
    config={
        "image": {
            "network": "mobileclip_v2_i3",
            "size": (256, 256),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "hidden_dim": 768,
                "num_heads": 12,
                "output_dim": 768,
                "causal_mask": False,
            },
        },
        "embed_dim": 768,
        "tokenizer": "openai_clip_bpe",
    },
)
registry.register_model_config(
    "mobileclip_v2_s4",
    CLIP,
    config={
        "image": {
            "network": "mobileclip_v2_i4",
            "size": (256, 256),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "hidden_dim": 768,
                "num_heads": 12,
                "output_dim": 768,
                "causal_mask": False,
            },
        },
        "embed_dim": 768,
        "tokenizer": "openai_clip_bpe",
    },
)

# MetaCLIP 2 - https://arxiv.org/abs/2507.22062
registry.register_model_config(
    "metaclip_v2_worldwide_b16",
    CLIP,
    config={
        "image": {
            "network": "vit_b16_pn",
            "size": (384, 384),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "vocab_size": 901629,
                "eos_token_id": 2,
            },
        },
        "embed_dim": 512,
        "tokenizer": "metaclip2_worldwide_bpe",
    },
)
registry.register_model_config(
    "metaclip_v2_worldwide_l14",
    CLIP,
    config={
        "image": {
            "network": "vit_l14_pn",
            "size": (224, 224),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "vocab_size": 901629,
                "hidden_dim": 768,
                "num_heads": 12,
                "output_dim": 768,
                "eos_token_id": 2,
            },
        },
        "embed_dim": 768,
        "tokenizer": "metaclip2_worldwide_bpe",
    },
)

# BioCLIP - https://arxiv.org/abs/2311.18803
registry.register_model_config(
    "bioclip_v1_vit_b16",
    CLIP,
    config={
        "image": {
            "network": "vit_b16_pn",
            "size": (224, 224),
        },
        "text": {
            "network": "text_transformer",
        },
        "embed_dim": 512,
        "tokenizer": "openai_clip_bpe",
    },
)

# BioCLIP 2 - https://arxiv.org/abs/2505.23883
registry.register_model_config(
    "bioclip_v2_vit_l14",
    CLIP,
    config={
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
        "embed_dim": 768,
        "tokenizer": "openai_clip_bpe",
    },
)

# BioCLIP 2.5 - https://arxiv.org/abs/2505.23883
registry.register_model_config(
    "bioclip_v25_vit_h14",
    CLIP,
    config={
        "image": {
            "network": "vit_h14_pn",
            "size": (224, 224),
        },
        "text": {
            "network": "text_transformer",
            "config": {
                "hidden_dim": 1024,
                "num_heads": 16,
                "num_layers": 24,
                "output_dim": 1024,
            },
        },
        "embed_dim": 1024,
        "tokenizer": "openai_clip_bpe",
    },
)


# Weights
####################

registry.register_weights(
    "openai_clip_vit_l14",
    {
        "description": "ViT l14 image-text model pretrained by OpenAI using CLIP",
        "resolution": (224, 224),
        "context_length": 77,
        "formats": {
            "pt": {
                "file_size": 1468.1,
                "sha256": "1020f6b1b35a551c22993788d38e0dd4933af3616b120d7375364cde9786fcb6",
            }
        },
        "net": {"network": "openai_clip_vit_l14"},
    },
)
registry.register_weights(
    "pe_core_b16",
    {
        "description": "RoPEi ViT b16 image encoder pretrained by Meta FAIR using CLIP",
        "resolution": (224, 224),
        "context_length": 32,
        "formats": {
            "pt": {
                "file_size": 1707.8,
                "sha256": "11453d4a36fad6dbd802ec9fa35375ce0ae8b7b156a5ca45c0e87587df05290f",
            }
        },
        "net": {"network": "pe_core_b16"},
    },
)
