import argparse
import logging
from pathlib import Path
from typing import Any
from typing import Optional

import torch
import torch.distributed as dist
from birder.common import training_utils as birder_training_utils

from birder_clip.common import fs_ops
from birder_clip.conf import settings


def setup_file_logging(log_file_path: str | Path) -> logging.Handler:
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter(
        fmt="{message}",
        style="{",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(settings.LOG_LEVEL)

    logging.getLogger("birder").addHandler(file_handler)
    logging.getLogger("birder_clip").addHandler(file_handler)

    return file_handler


def save_training_checkpoint(
    args: argparse.Namespace,
    network_name: str,
    epoch: int,
    net: torch.nn.Module,
    signature: Any,
    rgb_stats: Any,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.amp.grad_scaler.GradScaler],
    model_base: Optional[torch.nn.Module],
    *,
    external_config: Optional[dict[str, Any]] = None,
    **extra_states: Optional[dict[str, Any]],
) -> None:
    if birder_training_utils.is_global_primary(args) is True:
        fs_ops.checkpoint_model(
            network_name,
            epoch,
            net,
            signature,
            rgb_stats,
            optimizer,
            scheduler,
            scaler,
            model_base,
            external_config=external_config,
            **extra_states,
        )

    if birder_training_utils.is_dist_available_and_initialized() is True:
        dist.barrier()
