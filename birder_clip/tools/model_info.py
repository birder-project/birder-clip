import argparse
from typing import Any

import torch
from birder.common import cli
from rich.console import Console

from birder_clip.common import fs_ops
from birder_clip.common import lib
from birder_clip.model_registry import Task
from birder_clip.model_registry import registry


def get_model_info(net: torch.nn.Module) -> dict[str, float]:
    num_params = 0
    num_buffers = 0
    param_size = 0
    buffer_size = 0
    for param in net.parameters():
        num_params += param.numel()
        param_size += param.numel() * param.element_size()

    for buffer in net.buffers():
        num_buffers += buffer.numel()
        buffer_size += buffer.numel() * buffer.element_size()

    return {"num_params": num_params, "num_buffers": num_buffers, "model_size": param_size + buffer_size}


def print_model_info(console: Console, title: str, net: torch.nn.Module) -> None:
    model_info = get_model_info(net)
    console.print(f"[bold]{title}[/bold]")
    console.print(f"Network type: [bold]{type(net).__name__}[/bold]")
    console.print(f"Number of parameters: {model_info['num_params']:,}")
    console.print(f"Model size (inc. buffers): {(model_info['model_size']) / 1024**2:,.2f} [bold]MB[/bold]")
    console.print()


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "model-info",
        allow_abbrev=False,
        help="print information about the model",
        description="print information about the model",
        epilog=(
            "Usage examples:\n"
            "python -m birder_clip.tools model-info --network openai_clip_vit_l14 --epoch 0\n"
            "python -m birder_clip.tools model-info --network pe_core_l14 --st\n"
            "python -m birder_clip.tools model-info --network clip --image-encoder vit_b16 "
            "--text-encoder text_transformer --embed-dim 512 --tokenizer openai_clip_bpe\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "-n", "--network", type=str, required=True, help="the image-text network to load (i.e. openai_clip_vit_l14)"
    )
    subparser.add_argument(
        "--model-config",
        action=cli.FlexibleDictAction,
        help="override the model default configuration, accepts key-value pairs or JSON",
    )
    subparser.add_argument("--image-encoder", type=str, help="the image encoder to use")
    subparser.add_argument("--text-encoder", type=str, help="the text encoder to use")
    subparser.add_argument("--embed-dim", type=int, metavar="N", help="shared image-text embedding dimension")
    subparser.add_argument("--tokenizer", type=str, help="the tokenizer to use")
    subparser.add_argument(
        "--image-encoder-config",
        action=cli.FlexibleDictAction,
        help="override the image encoder configuration, accepts key-value pairs or JSON",
    )
    subparser.add_argument(
        "--text-encoder-config",
        action=cli.FlexibleDictAction,
        help="override the text encoder configuration, accepts key-value pairs or JSON",
    )
    subparser.add_argument("-e", "--epoch", type=int, metavar="N", help="model checkpoint to load")
    subparser.add_argument("-t", "--tag", type=str, help="model tag (from the training phase)")
    subparser.add_argument("--st", "--safetensors", default=False, action="store_true", help="load Safetensors weights")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    if registry.exists(args.network, task=Task.IMAGE_TEXT) is False:
        raise cli.ValidationError(f"--network {args.network} not supported, see list-models tool for available options")

    if args.tokenizer is None:
        args.tokenizer = registry.get_default_tokenizer(args.network)
    if args.tokenizer is None:
        args.tokenizer = "simple_tokenizer"

    registered_config = registry.all_nets[args.network.lower()].config  # type: ignore[misc]
    if registered_config is None:
        registered_config = {}
    if args.image_encoder is None and registered_config.get("image", {}).get("network") is None:
        raise cli.ValidationError("--image-encoder is required for this network")
    if args.text_encoder is None and registered_config.get("text", {}).get("network") is None:
        raise cli.ValidationError("--text-encoder is required for this network")

    reserved_model_config_keys = lib.get_reserved_model_config_keys(args.model_config)
    if len(reserved_model_config_keys) > 0:
        raise cli.ValidationError(
            "--model-config cannot contain keys with dedicated CLI flags: " f"{', '.join(reserved_model_config_keys)}"
        )
    if args.image_encoder_config is not None and "size" in args.image_encoder_config:
        raise cli.ValidationError("--image-encoder-config cannot contain size")
    if args.image_encoder_config is not None and "input_channels" in args.image_encoder_config:
        raise cli.ValidationError("--image-encoder-config cannot contain input_channels")
    if args.text_encoder_config is not None and "context_length" in args.text_encoder_config:
        raise cli.ValidationError("--text-encoder-config cannot contain context_length")

    # Load model
    device = torch.device("cpu")
    net, model_info = fs_ops.load_model(
        device,
        args.network,
        config=args.model_config,
        tag=args.tag,
        image_encoder=args.image_encoder,
        text_encoder=args.text_encoder,
        embed_dim=args.embed_dim,
        tokenizer=args.tokenizer,
        image_encoder_config=args.image_encoder_config,
        text_encoder_config=args.text_encoder_config,
        epoch=args.epoch,
        inference=True,
        st=args.st,
    )

    is_nan = torch.stack([torch.isnan(p).any() for p in net.parameters()]).any().item()

    console = Console()
    console.print(f"Network type: [bold]{type(net).__name__}[/bold], with task={net.task}")
    console.print(f"Network signature: {model_info.signature}")
    console.print(f"Network rgb values: {model_info.rgb_stats}")
    if model_info.custom_config is not None:
        console.print(f"Network has saved custom config: {model_info.custom_config}")

    console.print()
    print_model_info(console, "Image encoder", net.image_encoder)
    print_model_info(console, "Text encoder", net.text_encoder)
    print_model_info(console, "Full model", net)

    if is_nan is True:
        console.print()
        console.print("[red]Warning, NaN detected at the model weights[/red]")
