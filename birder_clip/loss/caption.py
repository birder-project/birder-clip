from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn.functional as F


class CaptionLoss(torch.nn.Module):
    """
    Autoregressive captioning cross entropy over decoder logits

    The loss consumes unshifted tokenized captions and supports two decoder output conventions:
    - logits length equals text length: the final logit is ignored.
    - logits length equals text length - 1: logits are used as-is.
    """

    def __init__(
        self,
        caption_loss_weight: float = 1.0,
        pad_token_id: int = 0,
        ignore_token_ids: Optional[Sequence[int]] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.caption_loss_weight = caption_loss_weight
        self.ignore_token_ids = tuple(ignore_token_ids) if ignore_token_ids is not None else (pad_token_id,)
        self.label_smoothing = label_smoothing

    def _align_logits_and_targets(self, logits: torch.Tensor, texts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.size(1) == texts.size(1):
            return logits[:, :-1], texts[:, 1:]
        if logits.size(1) == texts.size(1) - 1:
            return logits, texts[:, 1:]

        raise ValueError(
            "Expected logits sequence length to equal text sequence length or text sequence length - 1, "
            f"got logits={logits.size(1)}, texts={texts.size(1)}"
        )

    def _align_target_mask(self, target_mask: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        if target_mask.shape == texts.shape:
            target_mask = target_mask[:, 1:]

        return target_mask

    def forward(
        self, logits: torch.Tensor, texts: torch.Tensor, target_mask: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        if self.caption_loss_weight == 0.0:
            return {"caption_loss": logits.new_zeros(())}

        logits, targets = self._align_logits_and_targets(logits, texts)
        token_loss = F.cross_entropy(
            logits.permute(0, 2, 1),
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        loss_mask = torch.ones_like(targets, dtype=torch.bool)
        for token_id in self.ignore_token_ids:
            loss_mask = loss_mask & (targets != token_id)

        if target_mask is not None:
            target_mask = self._align_target_mask(target_mask, texts)
            loss_mask = loss_mask & target_mask.to(dtype=torch.bool)

        loss_mask_float = loss_mask.to(dtype=token_loss.dtype)
        caption_loss = (token_loss * loss_mask_float).sum() / loss_mask_float.sum().clamp_min(1)
        caption_loss = caption_loss * self.caption_loss_weight

        return {"caption_loss": caption_loss}
