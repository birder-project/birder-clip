import argparse
import os
import typing
from typing import Optional
from typing import get_args

import torch
from birder.common.cli import FlexibleDictAction
from birder.common.cli import ValidationError
from birder.common.training_utils import DDPCommDType
from birder.common.training_utils import OptimizerType
from birder.common.training_utils import SchedulerType
from birder.conf import settings
from birder.data.datasets.directory import ImageLoaderName
from birder.data.transforms.classification import AugType
from birder.data.transforms.classification import RGBMode


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-n", "--network", type=str, help="the neural network to use")
    parser.add_argument("-t", "--tag", type=str, help="add model tag")
    parser.add_argument("--image-encoder", type=str, help="the image encoder to use")
    parser.add_argument(
        "--image-encoder-pretrained",
        type=str,
        help="pretrained Birder image model weights path to load into the image encoder",
    )
    parser.add_argument("--text-encoder", type=str, help="the text encoder to use")
    parser.add_argument("--text-decoder", type=str, help="the text decoder to use")
    parser.add_argument("--embed-dim", type=int, metavar="N", help="shared image-text embedding dimension")
    parser.add_argument("--tokenizer", type=str, help="the tokenizer to use")
    parser.add_argument(
        "--model-config",
        action=FlexibleDictAction,
        help=(
            "override the model default configuration, accepts key-value pairs or JSON "
            "('init_logit_bias=-10.0' or '{\"visual_pool\": {\"num_heads\": 16}}')"
        ),
    )
    parser.add_argument(
        "--image-encoder-config",
        action=FlexibleDictAction,
        help="override the image encoder configuration, accepts key-value pairs or JSON",
    )
    parser.add_argument(
        "--text-encoder-config",
        action=FlexibleDictAction,
        help="override the text encoder configuration, accepts key-value pairs or JSON",
    )
    parser.add_argument(
        "--text-decoder-config",
        action=FlexibleDictAction,
        help="override the text decoder configuration, accepts key-value pairs or JSON",
    )
    parser.add_argument(
        "--openvision-v2-keep-ratio", type=float, help="OpenVision v2 image token keep ratio for caption decoding"
    )


def add_loss_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Loss parameters")
    group.add_argument(
        "--loss", type=str, choices=["clip", "coca", "caption"], default="clip", help="loss function to use"
    )
    group.add_argument(
        "--coca-caption-loss-weight", type=float, default=1.0, help="weight assigned to CoCa caption loss"
    )
    group.add_argument(
        "--coca-contrastive-loss-weight", type=float, default=1.0, help="weight assigned to CoCa contrastive loss"
    )


def add_freeze_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Freeze parameters")
    group.add_argument(
        "--freeze-image-encoder",
        default=False,
        action="store_true",
        help="freeze image encoder body, leaving the projection head trainable",
    )


def add_optimization_args(parser: argparse.ArgumentParser, default_batch_size: int = 32) -> None:
    group = parser.add_argument_group("Optimization parameters")
    group.add_argument("--batch-size", type=int, default=default_batch_size, metavar="N", help="the batch size")
    group.add_argument(
        "--opt", type=str, choices=list(get_args(OptimizerType)), default="adamw", help="optimizer to use"
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
    group.add_argument(
        "--grad-accum-cache-negatives",
        default=False,
        action="store_true",
        help="cache features so CLIP loss uses all accumulated microbatches as negatives",
    )


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
        action=FlexibleDictAction,
        metavar="LAYER=WD",
        help="custom weight decay for specific layers by name (e.g., offset_conv=0.0)",
    )


def add_lr_scheduler_args(parser: argparse.ArgumentParser, default_cosine_fraction: float = 1.0) -> None:
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
        help=(
            "number of scheduler updates between lr decreases, counted from when StepLR becomes active "
            "(--lr-scheduler step only)"
        ),
    )
    group.add_argument(
        "--lr-steps",
        type=int,
        nargs="+",
        help=(
            "absolute training epochs/optimizer steps, counted from training start, at which to decrease lr "
            "(--lr-scheduler multistep only)"
        ),
    )
    group.add_argument(
        "--lr-step-gamma",
        type=float,
        default=0.75,
        help="multiplicative learning rate decay factor (--lr-scheduler step or multistep)",
    )
    group.add_argument(
        "--lr-cosine-min",
        type=float,
        default=0.000001,
        help="minimum learning rate (for cosine annealing scheduler only)",
    )
    group.add_argument(
        "--lr-cosine-fraction",
        type=float,
        default=default_cosine_fraction,
        help="fraction of the cosine half-period to use (for cosine annealing scheduler only)",
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
    group.add_argument(
        "--size", type=int, nargs="+", metavar=("H", "W"), help="image size (defaults to the network default size)"
    )
    group.add_argument("--naflex", default=False, action="store_true", help="use native aspect-ratio training")
    group.add_argument(
        "--naflex-sizes",
        type=int,
        nargs="+",
        metavar="SIZE",
        help=(
            "square-equivalent image sizes in pixels to sample once per training batch, actual image "
            "dimensions preserve aspect ratio (values must be divisible by the model patch size)"
        ),
    )
    group.add_argument(
        "--naflex-patch-sizes",
        type=int,
        nargs="+",
        metavar="PATCH_SIZE",
        help="patch sizes to sample once per training batch (values do not need to divide the image sizes)",
    )
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
        help="magnitude of birder augmentations (0 uses only resize and horizontal flip -> 10 highest)",
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
    group.add_argument(
        "--timm-num-ops",
        type=int,
        default=2,
        metavar="N",
        help="number of timm-style RandAugment transformations to select sequentially",
    )
    group.add_argument(
        "--timm-magnitude",
        type=int,
        choices=list(range(10 + 1)),
        default=9,
        help="magnitude of timm-style RandAugment transformations",
    )
    group.add_argument(
        "--timm-magnitude-std",
        type=float,
        default=0.5,
        help="standard deviation of timm-style RandAugment magnitude noise",
    )
    group.add_argument("--augmix-severity", type=int, default=3, help="severity of AugMix policy")
    group.add_argument("--clip-color-jitter-prob", type=float, default=0.0, help="CLIP color jitter probability")
    group.add_argument("--clip-gray-prob", type=float, default=0.0, help="CLIP grayscale probability")
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
    group.add_argument(
        "--no-drop-last",
        dest="drop_last",
        default=True,
        action="store_false",
        help="do not drop the last incomplete batch",
    )


def add_precision_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Device and precision parameters")
    group.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="device to use for training",
    )
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


def add_grad_checkpointing_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Gradient checkpointing parameters")
    group.add_argument(
        "--grad-checkpointing",
        default=False,
        action="store_true",
        help="enable gradient checkpointing for supported models",
    )
    group.add_argument(
        "--grad-checkpointing-segments",
        type=int,
        metavar="N",
        help="number of checkpoint segments to request from supported models",
    )
    group.add_argument(
        "--no-grad-checkpointing-preserve-rng-state",
        dest="grad_checkpointing_preserve_rng_state",
        default=True,
        action="store_false",
        help="disable RNG state preservation during gradient checkpointing recomputation",
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


def add_checkpoint_args(
    parser: argparse.ArgumentParser, default_save_frequency: int = 1, load_ema: bool = False
) -> None:
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
    if load_ema is True:
        group.add_argument(
            "--load-ema", default=False, action="store_true", help="load EMA model weights when resuming"
        )


def add_distributed_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Distributed training parameters")
    group.add_argument("--world-size", type=int, default=1, metavar="N", help="number of distributed processes")
    group.add_argument("--rank", type=int, default=0, metavar="N", help="global distributed process rank")
    group.add_argument("--local-rank", type=int, metavar="N", help="local rank")
    group.add_argument("--dist-url", type=str, default="env://", help="URL used to initialize distributed training")
    group.add_argument("--dist-backend", type=str, default="nccl", help="distributed backend")
    group.add_argument(
        "--distributed-mode", type=str, choices=["ddp", "fsdp"], default="ddp", help="distributed training mode"
    )
    group.add_argument(
        "--fsdp-sharding-strategy",
        type=str,
        choices=["shard-grad-op", "full-shard"],
        default="shard-grad-op",
        help="FSDP sharding strategy",
    )
    group.add_argument(
        "--fsdp-param-dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="FSDP mixed precision parameter dtype",
    )
    group.add_argument(
        "--fsdp-reduce-dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="FSDP mixed precision gradient reduction dtype",
    )
    group.add_argument(
        "--fsdp-wrap-policy",
        type=str,
        choices=["block-group-regex", "min-num-params"],
        default="block-group-regex",
        help="FSDP module wrapping policy",
    )
    group.add_argument(
        "--fsdp-wrap-min-num-params",
        type=float,
        metavar="M",
        help="minimum module parameter count in millions for wrapping when using --fsdp-wrap-policy min-num-params",
    )
    group.add_argument(
        "--fsdp-offload-policy", type=str, choices=["none", "cpu"], default="none", help="FSDP parameter offload policy"
    )
    group.add_argument(
        "--ddp-comm-dtype",
        type=str,
        choices=list(get_args(DDPCommDType)),
        help="compress DDP gradient communication to the specified dtype (DDP only)",
    )
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
    group.add_argument(
        "--use-fake-data",
        default=False,
        action="store_true",
        help="use fake data instead of real dataset (like torchvision.datasets.FakeData)",
    )


def add_training_data_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Training data parameters")
    group.add_argument(
        "--caption-fields",
        type=str,
        nargs="+",
        default=["caption"],
        metavar="FIELD",
        help=(
            "caption fields (CSV columns or WebDataset JSON keys) to sample uniformly from non-empty values "
            "(validation uses the first non-empty field)"
        ),
    )

    group = parser.add_argument_group(description="WebDataset")
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
        help=(
            "enable cross-worker batch shuffling after batching. Provides maximum sample diversity but incurs a "
            "notable performance penalty. Use with caution"
        ),
    )

    group = parser.add_argument_group(description="CSV")
    group.add_argument(
        "--data-path", nargs="*", help="training CSV file paths (requires image_path and the selected caption columns)"
    )
    group.add_argument(
        "--val-path", nargs="*", help="validation CSV file paths (requires image_path and the selected caption columns)"
    )


def common_args_validation(args: argparse.Namespace) -> None:
    if args.network is None:
        raise ValidationError("--network is required")

    # Training schedule args, shared by all scripts
    if args.stop_epoch is not None and args.stop_epoch > args.epochs:
        raise ValidationError(
            f"--stop-epoch must be smaller than the total number of epochs ({args.epochs}), got {args.stop_epoch}"
        )
    if args.warmup_epochs is not None and args.warmup_steps is not None:
        raise ValidationError("--warmup-epochs cannot be used with --warmup-steps")
    if args.cooldown_epochs is not None and args.cooldown_steps is not None:
        raise ValidationError("--cooldown-epochs cannot be used with --cooldown-steps")
    if args.keep_last is not None and args.keep_last < 1:
        raise ValidationError(f"--keep-last must be greater than or equal to 1, got {args.keep_last}")

    if args.lr_scheduler_update != "step" and args.warmup_steps is not None:
        raise ValidationError(
            "--warmup-steps can only be used when --lr-scheduler-update is 'step', "
            f"but it is set to '{args.lr_scheduler_update}'"
        )
    if args.lr_scheduler_update != "step" and args.cooldown_steps is not None:
        raise ValidationError(
            "--cooldown-steps can only be used when --lr-scheduler-update is 'step', "
            f"but it is set to '{args.lr_scheduler_update}'"
        )
    if args.lr_cosine_fraction <= 0.0 or args.lr_cosine_fraction > 1.0:
        raise ValidationError("--lr-cosine-fraction must be in the range (0, 1]")
    if args.lr_warmup_decay <= 0.0 or args.lr_warmup_decay > 1.0:
        raise ValidationError("--lr-warmup-decay must be in the range (0, 1]")

    # EMA
    if args.model_ema_steps < 1:
        raise ValidationError("--model-ema-steps must be >= 1")

    # Gradient checkpointing args
    if args.grad_checkpointing_segments is not None and args.grad_checkpointing_segments < 1:
        raise ValidationError("--grad-checkpointing-segments must be >= 1")
    if args.grad_checkpointing_segments is not None and args.grad_checkpointing is False:
        raise ValidationError("--grad-checkpointing-segments requires --grad-checkpointing")

    # Compile args, argument dependant
    if args.compile_fullgraph is True and args.compile is False:
        raise ValidationError("--compile-fullgraph requires --compile")
    if args.compile_mode is not None and args.compile is False:
        raise ValidationError("--compile-mode requires --compile")

    # Checkpoint args, shared by all scripts
    if args.load_states is True and args.resume_epoch is None:
        raise ValidationError("--load-states requires --resume-epoch to be set")
    if hasattr(args, "load_ema") is True and args.load_ema is True:
        if args.resume_epoch is None:
            raise ValidationError("--load-ema requires --resume-epoch to be set")
        if args.model_ema is False:
            raise ValidationError("--load-ema requires --model-ema")
    if args.load_scheduler is True and args.resume_epoch is None:
        raise ValidationError("--load-scheduler requires --resume-epoch to be set")
    if args.pretrained is True and args.resume_epoch is not None:
        raise ValidationError("--pretrained cannot be used with --resume-epoch")

    # Input args
    if args.naflex is True:
        if args.simple_crop is True:
            raise ValidationError("--naflex cannot be used with --simple-crop")
        if args.wds is True and args.wds_extra_shuffle is True:
            raise ValidationError("--naflex cannot be used with --wds-extra-shuffle")

    if args.naflex_sizes is not None:
        if args.naflex is False:
            raise ValidationError("--naflex-sizes requires --naflex")
        if any(size <= 0 for size in args.naflex_sizes):
            raise ValidationError(f"--naflex-sizes values must be positive, got {args.naflex_sizes}")
        if len(set(args.naflex_sizes)) != len(args.naflex_sizes):
            raise ValidationError(f"--naflex-sizes values must be unique, got {args.naflex_sizes}")

    if args.naflex_patch_sizes is not None:
        if args.naflex is False:
            raise ValidationError("--naflex-patch-sizes requires --naflex")
        if len(set(args.naflex_patch_sizes)) != len(args.naflex_patch_sizes):
            raise ValidationError(f"--naflex-patch-sizes values must be unique, got {args.naflex_patch_sizes}")

    # Data augmentation args
    if args.rgb_mean is not None and len(args.rgb_mean) != args.channels:
        raise ValidationError(f"--rgb-mean must have {args.channels} values, got {len(args.rgb_mean)}")
    if args.rgb_std is not None and len(args.rgb_std) != args.channels:
        raise ValidationError(f"--rgb-std must have {args.channels} values, got {len(args.rgb_std)}")

    if args.resize_min_scale is not None and (args.resize_min_scale <= 0.0 or args.resize_min_scale >= 1.0):
        raise ValidationError(f"--resize-min-scale must be in range of (0, 1.0), got {args.resize_min_scale}")
    if args.center_crop > 1 or args.center_crop <= 0.0:
        raise ValidationError(f"--center-crop must be in range of (0, 1.0], got {args.center_crop}")
    if args.clip_color_jitter_prob is not None and (
        args.clip_color_jitter_prob < 0.0 or args.clip_color_jitter_prob > 1.0
    ):
        raise ValidationError(f"--clip-color-jitter-prob must be in range of [0, 1], got {args.clip_color_jitter_prob}")
    if args.clip_gray_prob is not None and (args.clip_gray_prob < 0.0 or args.clip_gray_prob > 1.0):
        raise ValidationError(f"--clip-gray-prob must be in range of [0, 1], got {args.clip_gray_prob}")

    # Training data args
    if len(args.caption_fields) == 0 or any(len(field.strip()) == 0 for field in args.caption_fields):
        raise ValidationError("--caption-fields must contain non-empty field names")
    if len(set(args.caption_fields)) != len(args.caption_fields):
        raise ValidationError("--caption-fields must not contain duplicate field names")

    if args.wds is False and args.data_path is None and args.use_fake_data is False:
        raise ValidationError("Must provide at least one data source, --data-path or --wds")
    if args.wds is False and args.data_path is not None and len(args.data_path) == 0 and args.use_fake_data is False:
        raise ValidationError("Must provide at least one data source, --data-path or --wds")
    if args.wds is True and args.data_path is not None and len(args.data_path) > 1:
        raise ValidationError(f"--wds can have at most 1 --data-path, got {len(args.data_path)}")
    if args.val_path is not None and len(args.val_path) == 0:
        raise ValidationError("--val-path must have at least one value")
    if args.use_fake_data is True and args.wds is True:
        raise ValidationError("--use-fake-data cannot be used with --wds")
    if args.wds is True and args.data_path is None and args.wds_info is None:
        raise ValidationError("Must provide at least one data source, --data-path or --wds-info")
    if args.wds is True and args.wds_info is not None and args.val_path is not None:
        raise ValidationError("--val-path cannot be used with --wds-info")

    # Dataloader args
    if args.persistent_workers is True and args.num_workers == 0:
        raise ValidationError("--persistent-workers requires --num-workers to be greater than 0")

    # BatchNorm args, shared by some scripts
    if args.freeze_bn is True and args.sync_bn is True:
        raise ValidationError("--freeze-bn cannot be used with --sync-bn")

    # FSDP mode checks
    if args.distributed_mode == "fsdp":
        if args.grad_checkpointing is True:
            raise ValidationError("--grad-checkpointing cannot be used with --distributed-mode fsdp")
        if args.sync_bn is True:
            raise ValidationError("--sync-bn cannot be used with --distributed-mode fsdp")
        if args.find_unused_parameters is True:
            raise ValidationError("--find-unused-parameters cannot be used with --distributed-mode fsdp")
        if args.ddp_comm_dtype is not None:
            raise ValidationError(
                "--ddp-comm-dtype cannot be used with --distributed-mode fsdp, use --fsdp-reduce-dtype instead"
            )
        if args.compile_opt is True:
            raise ValidationError("--compile-opt cannot be used with --distributed-mode fsdp")
        if args.compile_fullgraph is True:
            raise ValidationError("--compile-fullgraph cannot be used with --distributed-mode fsdp")
        if torch.device(args.device).type == "cpu":
            raise ValidationError("--device cpu cannot be used with --distributed-mode fsdp")
        if args.model_ema is True:
            raise ValidationError("--model-ema cannot be used with --distributed-mode fsdp")
        if args.fsdp_wrap_policy == "min-num-params" and args.fsdp_wrap_min_num_params is None:
            raise ValidationError("--fsdp-wrap-min-num-params is required when --fsdp-wrap-policy is min-num-params")
        if args.fsdp_wrap_min_num_params is not None and args.fsdp_wrap_min_num_params <= 0:
            raise ValidationError("--fsdp-wrap-min-num-params must be > 0")

    # Precision args, shared by all scripts
    if args.amp is True and args.model_dtype != "float32":
        raise ValidationError("--amp can only be used with --model-dtype float32")

    # Optimization args
    if args.grad_accum_steps < 1:
        raise ValidationError("--grad-accum-steps must be >= 1")
