import argparse
import os
import typing
from typing import Optional
from typing import get_args

import torch
from birder.common import cli
from birder.common.training_utils import OptimizerType
from birder.common.training_utils import SchedulerType
from birder.conf import settings
from birder.data.datasets.directory import ImageLoaderName
from birder.data.transforms.classification import AugType
from birder.data.transforms.classification import RGBMode

from birder_clip.model_registry import Task
from birder_clip.model_registry import registry


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-n", "--network", type=str, help="the image-text network to train")
    parser.add_argument("-t", "--tag", type=str, help="add model tag")
    parser.add_argument("--image-encoder", type=str, help="the image encoder to use")
    parser.add_argument("--text-encoder", type=str, help="the text encoder to use")
    parser.add_argument("--embed-dim", type=int, metavar="N", help="shared image-text embedding dimension")
    parser.add_argument("--tokenizer", type=str, help="the tokenizer to use")
    parser.add_argument(
        "--model-config",
        action=cli.FlexibleDictAction,
        help="override the model default configuration, accepts key-value pairs or JSON",
    )
    parser.add_argument(
        "--image-encoder-config",
        action=cli.FlexibleDictAction,
        help="override the image encoder configuration, accepts key-value pairs or JSON",
    )
    parser.add_argument(
        "--text-encoder-config",
        action=cli.FlexibleDictAction,
        help="override the text encoder configuration, accepts key-value pairs or JSON",
    )


def add_loss_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Loss parameters")
    group.add_argument("--loss", type=str, choices=["clip"], default="clip", help="loss function to use")


def add_optimization_args(parser: argparse.ArgumentParser, default_batch_size: int = 32) -> None:
    group = parser.add_argument_group("Optimization parameters")
    group.add_argument("--batch-size", type=int, default=default_batch_size, metavar="N", help="the batch size")
    group.add_argument(
        "--opt", type=str, choices=list(get_args(OptimizerType)), default="sgd", help="optimizer to use"
    )
    group.add_argument("--opt-fused", default=False, action="store_true", help="use fused optimizer implementation")
    group.add_argument("--momentum", type=float, default=0.9, metavar="M", help="optimizer momentum")
    group.add_argument("--nesterov", default=False, action="store_true", help="use nesterov momentum")
    group.add_argument("--opt-eps", type=float, help="optimizer epsilon (None to use the optimizer default)")
    group.add_argument("--opt-betas", type=float, nargs="+", help="optimizer betas (None to use the optimizer default)")
    group.add_argument("--opt-alpha", type=float, help="optimizer alpha (None to use the optimizer default)")
    group.add_argument("--clip-grad-norm", type=float, help="the maximum gradient norm")
    group.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        metavar="N",
        help="number of iterations to accumulate gradients per optimizer step",
    )
    # NOTE: Add flag for negative sample caching in grad accum mode


def add_lr_wd_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Learning rate and regularization parameters")
    group.add_argument("--lr", type=float, default=0.001, metavar="LR", help="base learning rate")
    group.add_argument("--bias-lr", type=float, metavar="LR", help="learning rate of biases")
    group.add_argument(
        "--lr-scale", type=int, help="reference batch size for LR scaling, if provided, LR will be scaled accordingly"
    )
    group.add_argument(
        "--lr-scale-type", type=str, choices=["linear", "sqrt"], default="linear", help="learning rate scaling type"
    )
    group.add_argument("--wd", type=float, default=0.2, metavar="WD", help="weight decay")
    group.add_argument("--norm-wd", type=float, metavar="WD", help="weight decay for Normalization layers")
    group.add_argument(
        "--bias-weight-decay", type=float, metavar="WD", help="weight decay for bias parameters of all layers"
    )
    group.add_argument(
        "--transformer-embedding-decay",
        type=float,
        metavar="WD",
        help="weight decay for embedding parameters for vision transformer models",
    )
    group.add_argument(
        "--custom-layer-wd",
        action=cli.FlexibleDictAction,
        metavar="LAYER=WD",
        help="custom weight decay for specific layers by name (e.g., logit_scale=0.0)",
    )


def add_lr_scheduler_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Learning rate scheduler parameters")
    group.add_argument(
        "--lr-scheduler-update",
        type=str,
        choices=["epoch", "step"],
        default="epoch",
        help="when to apply learning rate scheduler update: epoch (once per epoch), step (each optimizer step)",
    )
    group.add_argument(
        "--lr-scheduler",
        type=str,
        choices=list(get_args(SchedulerType)),
        default="constant",
        help="learning rate scheduler",
    )
    group.add_argument(
        "--lr-step-size",
        type=int,
        default=40,
        metavar="N",
        help="decrease lr every N epochs/steps (relative to after warmup, step scheduler only)",
    )
    group.add_argument(
        "--lr-steps",
        type=int,
        nargs="+",
        help="absolute epoch/step milestones when to decrease lr (multistep scheduler only)",
    )
    group.add_argument(
        "--lr-step-gamma",
        type=float,
        default=0.75,
        help="multiplicative factor of learning rate decay (for step scheduler only)",
    )
    group.add_argument(
        "--lr-cosine-min",
        type=float,
        default=0.000001,
        help="minimum learning rate (for cosine annealing scheduler only)",
    )
    group.add_argument(
        "--lr-power", type=float, default=1.0, help="power of the polynomial (for polynomial scheduler only)"
    )
    group.add_argument(
        "--lr-warmup-decay",
        type=float,
        default=0.01,
        help="multiplicative factor for learning rate at the start of warmup",
    )


def add_training_schedule_args(parser: argparse.ArgumentParser, default_epochs: int = 100) -> None:
    group = parser.add_argument_group("Training schedule parameters")
    group.add_argument("--epochs", type=int, default=default_epochs, metavar="N", help="number of training epochs")
    group.add_argument(
        "--stop-epoch", type=int, metavar="N", help="epoch to stop the training at (multi stage training)"
    )
    group.add_argument(
        "--steps-per-epoch",
        type=int,
        metavar="N",
        help="virtual epoch length in steps, leave unset to use the full dataset",
    )
    group.add_argument("--warmup-epochs", type=int, metavar="N", help="number of warmup epochs")
    group.add_argument("--warmup-steps", type=int, metavar="N", help="number of warmup optimizer steps")
    group.add_argument("--cooldown-epochs", type=int, metavar="N", help="number of cooldown epochs (linear to zero)")
    group.add_argument(
        "--cooldown-steps", type=int, metavar="N", help="number of cooldown optimizer steps (linear to zero)"
    )


def add_ema_args(
    parser: argparse.ArgumentParser, default_ema_steps: int = 1, default_ema_decay: float = 0.9999
) -> None:
    group = parser.add_argument_group("Exponential moving average parameters")
    group.add_argument(
        "--model-ema",
        default=False,
        action="store_true",
        help="enable tracking exponential moving average of model parameters",
    )
    group.add_argument(
        "--model-ema-steps",
        type=int,
        default=default_ema_steps,
        metavar="N",
        help="number of optimizer steps between EMA updates",
    )
    group.add_argument(
        "--model-ema-decay",
        type=float,
        default=default_ema_decay,
        help="decay factor for exponential moving average of model parameters",
    )
    group.add_argument(
        "--model-ema-warmup",
        type=int,
        metavar="N",
        help="number of epochs/steps before EMA is applied (defaults to warmup epochs/steps, pass 0 to disable warmup)",
    )


def add_batch_norm_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Batch normalization parameters")
    group.add_argument(
        "--freeze-bn",
        default=False,
        action="store_true",
        help="freeze all batch statistics and affine parameters of batchnorm2d layers",
    )
    group.add_argument("--sync-bn", default=False, action="store_true", help="use synchronized BatchNorm")


def add_input_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Input parameters")
    group.add_argument(
        "--channels", type=int, default=settings.DEFAULT_NUM_CHANNELS, metavar="N", help="no. of image channels"
    )
    group.add_argument("--size", type=int, nargs="+", metavar=("H", "W"), help="image size")
    group.add_argument("--context-length", type=int, metavar="N", help="text context length")


def add_data_aug_args(
    parser: argparse.ArgumentParser,
    default_level: int = 4,
    default_min_scale: Optional[float] = None,
    default_re_prob: Optional[float] = None,
) -> None:
    group = parser.add_argument_group("Data augmentation parameters")
    group.add_argument(
        "--aug-type", type=str, choices=list(get_args(AugType)), default="birder", help="augmentation type"
    )
    group.add_argument(
        "--aug-level",
        type=int,
        choices=list(range(10 + 1)),
        default=default_level,
        help="magnitude of birder augmentations (0 off -> 10 highest)",
    )
    group.add_argument(
        "--use-grayscale", default=False, action="store_true", help="use grayscale augmentation (birder aug only)"
    )
    group.add_argument(
        "--ra-num-ops",
        type=int,
        default=2,
        metavar="N",
        help="number of augmentation transformations to apply sequentially",
    )
    group.add_argument("--ra-magnitude", type=int, default=9, help="magnitude for all the RandAugment transformations")
    group.add_argument("--augmix-severity", type=int, default=3, help="severity of AugMix policy")
    group.add_argument("--resize-min-scale", type=float, default=default_min_scale, help="random resize min scale")
    group.add_argument(
        "--re-prob",
        type=float,
        default=default_re_prob,
        metavar="P",
        help="random erase probability (default according to aug-level)",
    )
    group.add_argument(
        "--simple-crop", default=False, action="store_true", help="use simple random crop (SRC) instead of RRC"
    )
    group.add_argument("--center-crop", type=float, default=1.0, help="center crop ratio to use during validation")
    group.add_argument(
        "--rgb-mode",
        type=str,
        choices=list(typing.get_args(RGBMode)),
        default="birder",
        help="RGB mean and std to use for normalization",
    )
    group.add_argument(
        "--rgb-mean",
        type=float,
        nargs="+",
        metavar=("R", "G"),
        help="set custom RGB mean values (overrides values from selected RGB mode)",
    )
    group.add_argument(
        "--rgb-std",
        type=float,
        nargs="+",
        metavar=("R", "G"),
        help="set custom RGB std values (overrides values from selected RGB mode)",
    )


def add_dataloader_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Dataloader parameters")
    group.add_argument(
        "--img-loader",
        type=str,
        choices=get_args(ImageLoaderName),
        default="tv",
        help="backend to load and decode images",
    )

    default_num_workers = min(12, max(os.cpu_count() // 4, 4))  # type: ignore[operator]
    group.add_argument(
        "-j",
        "--num-workers",
        type=int,
        default=default_num_workers,
        metavar="N",
        help="number of preprocessing workers",
    )
    group.add_argument(
        "--prefetch-factor", type=int, metavar="N", help="number of batches loaded in advance by each worker"
    )
    group.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        default=True,
        action="store_false",
        help="disable memory pinning in dataloaders",
    )
    group.add_argument(
        "--persistent-workers",
        default=False,
        action="store_true",
        help="keep dataloader worker processes alive between epochs",
    )
    group.add_argument("--drop-last", default=False, action="store_true", help="drop the last incomplete batch")


def add_precision_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Precision parameters")
    group.add_argument(
        "--model-dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="model dtype to use",
    )
    group.add_argument(
        "--amp",
        default=False,
        action="store_true",
        help="enable automatic mixed precision (AMP) training via torch.amp",
    )
    group.add_argument(
        "--amp-dtype",
        type=str,
        choices=["float16", "bfloat16"],
        default="float16",
        help="whether to use float16 or bfloat16 for mixed precision",
    )
    group.add_argument(
        "--fast-matmul", default=False, action="store_true", help="use fast matrix multiplication (affects precision)"
    )


def add_compile_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Compilation parameters")
    group.add_argument("--compile", default=False, action="store_true", help="enable compilation")
    group.add_argument("--compile-fullgraph", default=False, action="store_true", help="compile using fullgraph=True")
    group.add_argument(
        "--compile-mode", type=str, choices=list(torch._inductor.list_mode_options().keys()), help="torch.compile mode"
    )
    group.add_argument(
        "--compile-opt", default=False, action="store_true", help="enable compilation for optimizer step"
    )
    group.add_argument(
        "--compile-recompile-limit",
        type=int,
        default=torch.compiler.config.recompile_limit,
        metavar="N",
        help="maximum recompilations per compiled function before eager fallback",
    )
    group.add_argument(
        "--compile-accumulated-recompile-limit",
        type=int,
        default=torch.compiler.config.accumulated_recompile_limit,
        metavar="N",
        help="maximum total recompilations across compiled functions",
    )


def add_checkpoint_args(parser: argparse.ArgumentParser, default_save_frequency: int = 1) -> None:
    group = parser.add_argument_group("Checkpoint parameters")
    group.add_argument(
        "--save-frequency", type=int, default=default_save_frequency, metavar="N", help="frequency of model saving"
    )
    group.add_argument(
        "--keep-last", type=int, metavar="N", help="number of recent checkpoints to keep (older ones are deleted)"
    )
    group.add_argument(
        "--pretrained",
        default=False,
        action="store_true",
        help="start with pretrained version of specified network (will download if not found locally)",
    )
    group.add_argument("--resume-epoch", type=int, metavar="N", help="epoch number to resume training from")
    group.add_argument(
        "--non-strict-weights",
        default=False,
        action="store_true",
        help="allow non-strict loading of model weights (missing or unexpected keys in state_dict)",
    )
    group.add_argument(
        "--load-states",
        default=False,
        action="store_true",
        help="load optimizer, scheduler and scaler states when resuming",
    )
    group.add_argument("--load-scheduler", default=False, action="store_true", help="load only scheduler when resuming")


def add_distributed_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Distributed training parameters")
    group.add_argument("--world-size", type=int, default=1, metavar="N", help="number of distributed processes")
    group.add_argument("--local-rank", type=int, metavar="N", help="local rank")
    group.add_argument("--dist-url", type=str, default="env://", help="URL used to initialize distributed training")
    group.add_argument("--dist-backend", type=str, default="nccl", help="distributed backend")
    group.add_argument(
        "--find-unused-parameters",
        default=False,
        action="store_true",
        help="enable searching for unused parameters in DistributedDataParallel (may impact performance)",
    )
    group.add_argument(
        "--no-broadcast-buffers",
        default=False,
        action="store_true",
        help="disable broadcasting of buffers from rank 0 in distributed training",
    )


def add_logging_and_debug_args(parser: argparse.ArgumentParser, default_log_interval: int = 50) -> None:
    group = parser.add_argument_group("Logging and debugging parameters")
    group.add_argument(
        "--experiment",
        "--exp",
        type=str,
        metavar="NAME",
        help="experiment name for logging (creates dedicated directory for the run)",
    )
    group.add_argument(
        "--log-interval",
        type=int,
        default=default_log_interval,
        metavar="N",
        help="how many iterations between summary writes",
    )
    group.add_argument(
        "--grad-anomaly-detection",
        default=False,
        action="store_true",
        help="enable the autograd anomaly detection (for debugging)",
    )
    group.add_argument(
        "--use-deterministic-algorithms", default=False, action="store_true", help="use only deterministic algorithms"
    )
    group.add_argument(
        "--plot-lr", default=False, action="store_true", help="plot learning rate and exit (skip training)"
    )
    group.add_argument("--no-summary", default=False, action="store_true", help="don't print model summary")
    group.add_argument(
        "--non-interactive",
        default=False,
        action="store_true",
        help="force non-interactive mode (disables progress bars)",
    )
    group.add_argument(
        "--seed", type=int, help="set random seed for better reproducibility (affects torch, numpy and random)"
    )
    group.add_argument("--cpu", default=False, action="store_true", help="use cpu (mostly for testing)")
    group.add_argument(
        "--use-fake-data",
        default=False,
        action="store_true",
        help="use fake data instead of real dataset",
    )


def add_training_data_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Training data parameters", description="WebDataset")
    group.add_argument("--wds", default=False, action="store_true", help="use webdataset for training")
    group.add_argument("--wds-info", type=str, action="append", metavar="FILE", help="one or more wds info file paths")
    group.add_argument("--wds-cache-dir", type=str, metavar="DIR", help="webdataset cache directory")
    group.add_argument("--wds-train-size", type=int, metavar="N", help="size of the wds training set")
    group.add_argument("--wds-val-size", type=int, metavar="N", help="size of the wds validation set")
    group.add_argument(
        "--wds-training-split", type=str, default="training", metavar="NAME", help="wds dataset train split"
    )
    group.add_argument(
        "--wds-val-split", type=str, default="validation", metavar="NAME", help="wds dataset validation split"
    )
    group.add_argument(
        "--wds-extra-shuffle",
        default=False,
        action="store_true",
        help="enable cross-worker batch shuffling after batching",
    )

    group = parser.add_argument_group(description="CSV")
    group.add_argument("--data-path", nargs="*", help="training CSV file paths (required columns: image_path, caption)")
    group.add_argument(
        "--val-path", nargs="*", help="validation CSV file paths (required columns: image_path, caption)"
    )


def common_args_validation(args: argparse.Namespace) -> None:
    if args.network is None:
        raise cli.ValidationError("--network is required")
    if registry.exists(args.network, task=Task.IMAGE_TEXT) is False:
        raise cli.ValidationError(f"--network {args.network} not supported, see list-models tool for available options")

    if args.stop_epoch is not None and args.stop_epoch > args.epochs:
        raise cli.ValidationError(
            f"--stop-epoch must be smaller than the total number of epochs ({args.epochs}), got {args.stop_epoch}"
        )
    if args.warmup_epochs is not None and args.warmup_steps is not None:
        raise cli.ValidationError("--warmup-epochs cannot be used with --warmup-steps")
    if args.cooldown_epochs is not None and args.cooldown_steps is not None:
        raise cli.ValidationError("--cooldown-epochs cannot be used with --cooldown-steps")
    if args.lr_scheduler_update != "step" and args.warmup_steps is not None:
        raise cli.ValidationError(
            "--warmup-steps can only be used when --lr-scheduler-update is 'step', "
            f"but it is set to '{args.lr_scheduler_update}'"
        )
    if args.lr_scheduler_update != "step" and args.cooldown_steps is not None:
        raise cli.ValidationError(
            "--cooldown-steps can only be used when --lr-scheduler-update is 'step', "
            f"but it is set to '{args.lr_scheduler_update}'"
        )

    if args.compile_fullgraph is True and args.compile is False:
        raise cli.ValidationError("--compile-fullgraph requires --compile")
    if args.compile_mode is not None and args.compile is False:
        raise cli.ValidationError("--compile-mode requires --compile")

    if args.load_states is True and args.resume_epoch is None:
        raise cli.ValidationError("--load-states requires --resume-epoch to be set")
    if args.load_scheduler is True and args.resume_epoch is None:
        raise cli.ValidationError("--load-scheduler requires --resume-epoch to be set")
    if hasattr(args, "pretrained") is True and args.pretrained is True and args.resume_epoch is not None:
        raise cli.ValidationError("--pretrained cannot be used with --resume-epoch")

    if args.freeze_bn is True and args.sync_bn is True:
        raise cli.ValidationError("--freeze-bn cannot be used with --sync-bn")

    if args.wds is False and args.data_path is None and args.use_fake_data is False:
        raise cli.ValidationError("Must provide at least one data source, --data-path or --wds")
    if args.wds is False and args.data_path is not None and len(args.data_path) == 0 and args.use_fake_data is False:
        raise cli.ValidationError("Must provide at least one data source, --data-path or --wds")
    if args.wds is True and args.data_path is not None and len(args.data_path) > 1:
        raise cli.ValidationError(f"--wds can have at most 1 --data-path, got {len(args.data_path)}")
    if args.val_path is not None and len(args.val_path) == 0:
        raise cli.ValidationError("--val-path must have at least one value")
    if args.use_fake_data is True and args.wds is True:
        raise cli.ValidationError("--use-fake-data cannot be used with --wds")
    if args.persistent_workers is True and args.num_workers == 0:
        raise cli.ValidationError("--persistent-workers requires --num-workers to be greater than 0")

    if args.amp is True and args.model_dtype != "float32":
        raise cli.ValidationError("--amp can only be used with --model-dtype float32")
    if args.embed_dim is not None and args.embed_dim <= 0:
        raise cli.ValidationError("--embed-dim must be positive")
    if args.context_length is not None and args.context_length <= 0:
        raise cli.ValidationError("--context-length must be positive")
    if args.grad_accum_steps < 1:
        raise cli.ValidationError("--grad-accum-steps must be >= 1")
    if args.model_ema_steps < 1:
        raise cli.ValidationError("--model-ema-steps must be >= 1")
