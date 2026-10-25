import argparse
import json
import logging
from pathlib import Path
from typing import Any
from typing import Optional

import torch
import torch.distributed as dist
from birder.common import fsdp_utils
from birder.common import training_utils as birder_training_utils
from birder.conf import settings as birder_settings

from birder_clip.common import fs_ops
from birder_clip.version import __version__ as birder_clip_version


def setup_file_logging(log_file_path: str | Path) -> logging.Handler:
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter(
        fmt="{message}",
        style="{",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(birder_settings.LOG_LEVEL)

    logging.getLogger("birder").addHandler(file_handler)
    logging.getLogger("birder_clip").addHandler(file_handler)

    return file_handler


def make_training_args_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "birder_clip_version": birder_clip_version,
        **birder_training_utils.make_training_args_payload(args),
    }


def write_training_args_json(path: Path, args: argparse.Namespace) -> None:
    with open(path.joinpath("training_args.json"), "w", encoding="utf-8") as handle:
        json.dump(make_training_args_payload(args), handle, indent=2)


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
    fsdp_mode: bool = False,
    fsdp_model_state: Optional[dict[str, Any]] = None,
    external_config: Optional[dict[str, Any]] = None,
    **extra_states: Optional[dict[str, Any]],
) -> None:
    if fsdp_mode is True:
        if fsdp_model_state is not None:
            model_state = fsdp_model_state
        else:
            model_state = fsdp_utils.gather_full_model_state_dict(net)

        optimizer_state = None
        scheduler_state = None
        scaler_state = None
        model_base_state = None
        if optimizer is not None and scheduler is not None:
            optimizer_state = fsdp_utils.gather_full_optimizer_state_dict(net, optimizer)
            scheduler_state = scheduler.state_dict()
            if scaler is not None:
                scaler_state = scaler.state_dict()
            if model_base is not None:
                model_base_state = model_base.state_dict()

        if birder_training_utils.is_global_primary(args) is True:
            fs_ops.checkpoint_model_from_state_dicts(
                network_name,
                epoch,
                model_state,
                net.task,
                signature,
                rgb_stats,
                optimizer_state,
                scheduler_state,
                scaler_state,
                model_base_state,
                external_config=external_config,
                **extra_states,
            )

        if birder_training_utils.is_dist_available_and_initialized() is True:
            dist.barrier()

    else:
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
