"""
CLIP text transformer, adapted from
https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
"""

# Reference license: MIT

from collections.abc import Callable
from typing import Any
from typing import Literal
from typing import Optional

import torch
import torch.nn.functional as F
from birder.layers import FFN
from birder.layers import LayerScale
from birder.layers.activations import get_activation_module
from torch import nn
from torchvision.ops import StochasticDepth

from birder_clip.model_registry import registry
from birder_clip.net.text.base import TextBaseNet


def text_global_pool(
    x: torch.Tensor,
    text: torch.Tensor,
    *,
    pool_type: Literal["first", "last", "eos"] = "eos",
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    if pool_type == "first":
        return x[:, 0]
    if pool_type == "last":
        return x[:, -1]
    if pool_type == "eos":
        assert eos_token_id is not None
        idx = (text == eos_token_id).int().argmax(dim=-1)
        return x[torch.arange(x.size(0), device=x.device), idx]

    raise ValueError(f"Unknown pool_type '{pool_type}'")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float,
        proj_drop: float,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        norm_layer_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        torch._assert(dim % num_heads == 0, "Dim should be divisible by num_heads")

        self.is_causal = False
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if qk_norm is True:
            self.q_norm = norm_layer(self.head_dim, eps=norm_layer_eps)
            self.k_norm = norm_layer(self.head_dim, eps=norm_layer_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.size()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        x = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=self.is_causal,
            scale=self.scale,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
        activation_layer: type[nn.Module],
        layer_scale_init_value: Optional[float] = None,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        norm_layer_eps: float = 1e-5,
        qkv_bias: bool = True,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(hidden_dim, eps=norm_layer_eps)
        self.attn = Attention(
            hidden_dim,
            num_heads,
            attn_drop=attention_dropout,
            proj_drop=0.0,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
            norm_layer_eps=norm_layer_eps,
        )
        self.drop_path = StochasticDepth(drop_path, mode="row")
        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale(hidden_dim, layer_scale_init_value)
            self.layer_scale_2 = LayerScale(hidden_dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()

        self.norm2 = norm_layer(hidden_dim, eps=norm_layer_eps)
        self.mlp = FFN(hidden_dim, mlp_dim, act_layer=activation_layer, dropout=dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path(self.layer_scale_1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x))))

        return x

    def set_causal_attention(self, is_causal: bool = True) -> None:
        self.attn.is_causal = is_causal


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        dpr: list[float],
        pre_norm: bool = False,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        activation_layer: type[nn.Module] = nn.GELU,
        layer_scale_init_value: Optional[float] = None,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        norm_layer_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        pre_layers = []
        if dropout > 0.0:
            pre_layers.append(nn.Dropout(dropout))
        if pre_norm is True:
            pre_layers.append(norm_layer(hidden_dim, eps=norm_layer_eps))

        self.pre_block = nn.Sequential(*pre_layers)
        self.block = nn.ModuleList(
            [
                EncoderBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    dropout,
                    attention_dropout,
                    dpr[i],
                    activation_layer=activation_layer,
                    layer_scale_init_value=layer_scale_init_value,
                    norm_layer=norm_layer,
                    norm_layer_eps=norm_layer_eps,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.pre_block(x)
        for b in self.block:
            x = b(x, attn_mask=attn_mask)

        return x

    def set_causal_attention(self, is_causal: bool = True) -> None:
        for b in self.block:
            b.set_causal_attention(is_causal)


class TextTransformer(TextBaseNet):
    def __init__(self, *, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        assert self.config is not None, "must set config"

        context_length: int = self.config.get("context_length", 77)
        vocab_size: int = self.config.get("vocab_size", 49408)
        hidden_dim: int = self.config.get("hidden_dim", 512)
        num_heads: int = self.config.get("num_heads", 8)
        num_layers: int = self.config.get("num_layers", 12)
        mlp_dim: int = self.config.get("mlp_dim", hidden_dim * 4)
        output_dim: int = self.config.get("output_dim", 512)
        dropout: float = self.config.get("dropout", 0.0)
        attention_dropout: float = self.config.get("attention_dropout", 0.0)
        drop_path_rate: float = self.config.get("drop_path_rate", 0.0)
        causal_mask: bool = self.config.get("causal_mask", True)
        pool_type: Literal["first", "last", "eos"] = self.config.get("pool_type", "eos")
        proj_bias: bool = self.config.get("proj_bias", False)
        pad_token_id: Optional[int] = self.config.get("pad_token_id", None)
        eos_token_id: Optional[int] = self.config.get("eos_token_id", 49407)
        norm_layer_type: str = self.config.get("norm_layer_type", "LayerNorm")
        norm_layer_eps: float = self.config.get("norm_layer_eps", 1e-5)
        act_layer_type: str = self.config.get("act_layer_type", "gelu")
        qkv_bias: bool = self.config.get("qkv_bias", True)
        qk_norm: bool = self.config.get("qk_norm", False)
        pre_norm: bool = self.config.get("pre_norm", False)
        layer_scale_init_value: Optional[float] = self.config.get("layer_scale_init_value", None)

        if pool_type not in ("first", "last", "eos"):
            raise ValueError(f"Unknown pool_type '{pool_type}'")

        if norm_layer_type == "LayerNorm":
            norm_layer = nn.LayerNorm
        elif norm_layer_type == "RMSNorm":
            norm_layer = nn.RMSNorm
        else:
            raise ValueError(f"Unknown norm_layer_type '{norm_layer_type}'")

        torch._assert(hidden_dim % num_heads == 0, "Hidden dim indivisible by num heads!")

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.causal_mask = causal_mask
        self.pool_type = pool_type
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.empty(1, context_length, hidden_dim).normal_(std=0.01))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.encoder = Encoder(
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
        if causal_mask is True:
            self.encoder.set_causal_attention()

        self.norm = norm_layer(hidden_dim, eps=norm_layer_eps)
        self.text_projection = nn.Linear(hidden_dim, output_dim, bias=proj_bias)
        self.embedding_size = output_dim

        # Weights initialization
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.01)

        proj_std = (self.hidden_dim**-0.5) * ((2 * self.num_layers) ** -0.5)
        attn_std = self.hidden_dim**-0.5
        fc_std = (2 * self.hidden_dim) ** -0.5
        for block in self.encoder.block:
            nn.init.normal_(block.attn.qkv.weight, std=attn_std)
            nn.init.normal_(block.attn.proj.weight, std=proj_std)
            nn.init.normal_(block.mlp[0].weight, std=fc_std)
            nn.init.normal_(block.mlp[3].weight, std=proj_std)

        nn.init.normal_(self.text_projection.weight, std=self.hidden_dim**-0.5)
        if self.text_projection.bias is not None:
            nn.init.zeros_(self.text_projection.bias)

    def set_causal_attention(self, is_causal: bool = True) -> None:
        self.causal_mask = is_causal
        self.encoder.set_causal_attention(is_causal)

    def _pad_attention_mask(self, token_ids: torch.Tensor) -> Optional[torch.Tensor]:
        if self.causal_mask is True or self.pad_token_id is None:
            return None

        return token_ids.ne(self.pad_token_id)[:, None, None, :]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        attn_mask = self._pad_attention_mask(x)
        x = self.token_embedding(x)
        x = x + self.pos_embedding[:, : x.size(1), :].to(dtype=x.dtype)
        x = self.encoder(x, attn_mask=attn_mask)
        x = self.norm(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_features(x)
        out = text_global_pool(tokens, x, pool_type=self.pool_type, eos_token_id=self.eos_token_id)

        return self.text_projection(out)


registry.register_model_config("text_transformer", TextTransformer, config={})
