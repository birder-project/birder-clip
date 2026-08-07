from typing import Any
from typing import ClassVar
from typing import Optional

import torch
from birder.layers.activations import get_activation_module
from birder.net.base import stochastic_depth_rates
from torch import nn

from birder_clip.model_registry import registry
from birder_clip.net.text.base import VisualCausalTextDecoderBaseNet
from birder_clip.net.text.base import adjust_position_embedding
from birder_clip.net.text.encoder import Encoder


class VisualCausalTextDecoder(VisualCausalTextDecoderBaseNet):
    block_group_regex: ClassVar[str] = r"decoder\.block\.(\d+)"

    def __init__(self, *, config: Optional[dict[str, Any]] = None, context_length: Optional[int] = None) -> None:
        super().__init__(config=config, context_length=context_length)
        assert self.config is not None, "must set config"

        vocab_size: int = self.config.get("vocab_size", 32000)
        hidden_dim: int = self.config.get("hidden_dim", 768)
        num_heads: int = self.config.get("num_heads", 12)
        num_layers: int = self.config.get("num_layers", 12)
        mlp_dim: int = self.config.get("mlp_dim", hidden_dim * 4)
        kv_dim: int = self.config.get("kv_dim", hidden_dim)
        dropout: float = self.config.get("dropout", 0.0)
        attention_dropout: float = self.config.get("attention_dropout", 0.0)
        drop_path_rate: float = self.config.get("drop_path_rate", 0.0)
        norm_layer_type: str = self.config.get("norm_layer_type", "LayerNorm")
        norm_layer_eps: float = self.config.get("norm_layer_eps", 1e-5)
        act_layer_type: str = self.config.get("act_layer_type", "gelu")
        qkv_bias: bool = self.config.get("qkv_bias", True)
        qk_norm: bool = self.config.get("qk_norm", False)
        pre_norm: bool = self.config.get("pre_norm", False)
        image_projection_bias: bool = self.config.get("image_projection_bias", False)
        lm_head_bias: bool = self.config.get("lm_head_bias", False)
        layer_scale_init_value: Optional[float] = self.config.get("layer_scale_init_value", None)
        shift_input_tokens: bool = self.config.get("shift_input_tokens", False)

        if norm_layer_type == "LayerNorm":
            norm_layer = nn.LayerNorm
        elif norm_layer_type == "RMSNorm":
            norm_layer = nn.RMSNorm
        else:
            raise ValueError(f"Unknown norm_layer_type '{norm_layer_type}'")

        torch._assert(hidden_dim % num_heads == 0, "Hidden dim indivisible by num heads!")

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.kv_dim = kv_dim
        self.causal_mask = True
        self.shift_input_tokens = shift_input_tokens
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.empty(1, self.context_length, hidden_dim))
        self.image_projection = nn.Linear(kv_dim, hidden_dim, bias=image_projection_bias)

        dpr = stochastic_depth_rates(drop_path_rate, num_layers)
        self.decoder = Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            dpr,
            pre_norm=pre_norm,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            activation_layer=get_activation_module(act_layer_type),
            layer_scale_init_value=layer_scale_init_value,
            norm_layer=norm_layer,
            norm_layer_eps=norm_layer_eps,
        )
        self.decoder.set_causal_attention()

        self.norm = norm_layer(hidden_dim, eps=norm_layer_eps)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=lm_head_bias)

        # Weights initialization
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.01)
        nn.init.normal_(self.image_projection.weight, std=kv_dim**-0.5)
        if self.image_projection.bias is not None:
            nn.init.zeros_(self.image_projection.bias)

        proj_std = (self.hidden_dim**-0.5) * ((2 * self.num_layers) ** -0.5)
        attn_std = self.hidden_dim**-0.5
        fc_std = (2 * self.hidden_dim) ** -0.5
        for block in self.decoder.block:
            nn.init.normal_(block.attn.qkv.weight, std=attn_std)
            nn.init.normal_(block.attn.proj.weight, std=proj_std)
            nn.init.normal_(block.mlp[0].weight, std=fc_std)
            nn.init.normal_(block.mlp[3].weight, std=proj_std)

        nn.init.normal_(self.lm_head.weight, std=self.hidden_dim**-0.5)
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)

    def set_grad_checkpointing(
        self,
        enable: bool = True,
        *,
        segments: Optional[int] = None,
        preserve_rng_state: bool = True,
        use_reentrant: bool = False,
    ) -> None:
        self.decoder.set_grad_checkpointing(
            enable=enable,
            segments=segments,
            preserve_rng_state=preserve_rng_state,
            use_reentrant=use_reentrant,
        )

    def _prepare_text_embeddings(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        if self.shift_input_tokens is True:
            text_embeddings = text_embeddings[:, :-1]

        seq_len = text_embeddings.size(1)
        pos_embedding = self.pos_embedding[:, :seq_len, :].to(dtype=text_embeddings.dtype)

        return text_embeddings + pos_embedding

    def forward_features(self, text: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        text_embeddings = self.token_embedding(text)
        return self.forward_features_from_embeddings(text_embeddings, image_tokens)

    def forward_features_from_embeddings(
        self, text_embeddings: torch.Tensor, image_tokens: torch.Tensor
    ) -> torch.Tensor:
        text_embeddings = self._prepare_text_embeddings(text_embeddings)
        image_embeddings = self.image_projection(image_tokens)
        combined_embeddings = torch.concat([image_embeddings, text_embeddings], dim=1)
        features = self.decoder(combined_embeddings)
        features = features[:, image_embeddings.size(1) :]
        features = self.norm(features)

        return features

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        return self.lm_head(features)

    def adjust_context_length(self, new_context_length: int) -> None:
        if new_context_length == self.context_length:
            return

        super().adjust_context_length(new_context_length)
        self.pos_embedding = nn.Parameter(adjust_position_embedding(self.pos_embedding, new_context_length))


registry.register_model_config("visual_causal_text_decoder", VisualCausalTextDecoder, config={})
registry.register_model_config(
    "visual_causal_text_decoder_t",
    VisualCausalTextDecoder,
    config={"hidden_dim": 192, "num_layers": 12, "mlp_dim": 768, "num_heads": 3},
)
registry.register_model_config(
    "visual_causal_text_decoder_s",
    VisualCausalTextDecoder,
    config={"hidden_dim": 384, "num_layers": 12, "mlp_dim": 1536, "num_heads": 6},
)
registry.register_model_config(
    "visual_causal_text_decoder_m",
    VisualCausalTextDecoder,
    config={"hidden_dim": 512, "num_layers": 12, "mlp_dim": 2048, "num_heads": 8},
)
registry.register_model_config(
    "visual_causal_text_decoder_b",  # Matches the default
    VisualCausalTextDecoder,
    config={"hidden_dim": 768, "num_layers": 12, "mlp_dim": 3072, "num_heads": 12},
)
registry.register_model_config(
    "visual_causal_text_decoder_l",
    VisualCausalTextDecoder,
    config={"hidden_dim": 1024, "num_layers": 24, "mlp_dim": 4096, "num_heads": 16},
)
registry.register_model_config(
    "visual_causal_text_decoder_h",
    VisualCausalTextDecoder,
    config={"hidden_dim": 1280, "num_layers": 32, "mlp_dim": 5120, "num_heads": 16},
)
registry.register_model_config(
    "visual_causal_text_decoder_g",
    VisualCausalTextDecoder,
    config={"hidden_dim": 1408, "num_layers": 40, "mlp_dim": 6144, "num_heads": 16},
)
registry.register_model_config(
    "visual_causal_text_decoder_gigantic",
    VisualCausalTextDecoder,
    config={"hidden_dim": 1664, "num_layers": 48, "mlp_dim": 8192, "num_heads": 16},
)
