"""
CLIP loss, adapted from
https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py
"""

# Reference license: MIT

from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn.functional as F
from birder.common import training_utils


def gather_features(features: torch.Tensor) -> torch.Tensor:
    if training_utils.is_dist_available_and_initialized() is False:
        return features

    return funcol.all_gather_tensor_autograd(features, gather_dim=0, group=dist.group.WORLD)


class CLIPLoss(torch.nn.Module):
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if image_features.shape[0] != text_features.shape[0]:
            raise ValueError("CLIP loss expects paired image and text batches")

        image_features = gather_features(image_features)
        text_features = gather_features(text_features)

        logits_per_image = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits_per_image = logits_per_image + logit_bias

        logits_per_text = logits_per_image.T
        labels = torch.arange(logits_per_image.shape[0], device=image_features.device, dtype=torch.long)
        contrastive_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

        return {"contrastive_loss": contrastive_loss}
