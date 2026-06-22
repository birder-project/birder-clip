import argparse
import json
import logging
from pathlib import Path
from typing import Any
from typing import Optional

import torch
from birder.common import cli
from birder.common import fs_ops as birder_fs_ops
from birder.common import lib as birder_lib
from birder.common.fs_ops import read_class_file
from birder.data.transforms.classification import RGBType
from birder.net.base import get_signature

from birder_clip.common import fs_ops
from birder_clip.common import lib
from birder_clip.inference.zero_shot import build_class_text_embeddings
from birder_clip.inference.zero_shot_templates import TEMPLATE_SETS
from birder_clip.model_registry import Task
from birder_clip.model_registry import registry
from birder_clip.net.base import SignatureType
from birder_clip.tokenizers import get_tokenizer

logger = logging.getLogger(__name__)


def config_export(net: torch.nn.Module, signature: SignatureType, rgb_stats: RGBType, model_path: str | Path) -> None:
    model_config = lib.get_image_text_network_config(net, signature, rgb_stats)
    logger.info("Saving model config json...")
    with open(f"{model_path}_config.json", "w", encoding="utf-8") as handle:
        json.dump(model_config, handle, indent=2)


def image_encoder_export(
    net: torch.nn.Module,
    network_name: str,
    epoch: int,
    signature: SignatureType,
    rgb_stats: RGBType,
    image_encoder_config: Optional[dict[str, Any]],
    class_to_idx: Optional[dict[str, int]] = None,
) -> None:
    if class_to_idx is None:
        class_to_idx = {str(idx): idx for idx in range(net.image_encoder.num_classes)}

    image_signature = get_signature(
        (
            0,
            signature["inputs"][0]["data_shape"][1],
            signature["inputs"][0]["data_shape"][2],
            signature["inputs"][0]["data_shape"][3],
        ),
        net.image_encoder.num_classes,
    )

    birder_fs_ops.checkpoint_model(
        network_name,
        epoch,
        net.image_encoder,
        image_signature,
        class_to_idx,
        rgb_stats,
        optimizer=None,
        scheduler=None,
        scaler=None,
        model_base=None,
        external_config=image_encoder_config,
    )


def get_class_to_idx(args: argparse.Namespace) -> dict[str, int]:
    if args.class_file is not None:
        return read_class_file(args.class_file)
    if args.classes is not None:
        return {class_name: idx for idx, class_name in enumerate(args.classes)}

    raise RuntimeError("Cannot infer classes")


def get_templates(args: argparse.Namespace) -> list[str]:
    if args.template is not None:
        return args.template  # type: ignore[no-any-return]
    if args.template_file is not None:
        with open(args.template_file, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if len(line.strip()) > 0]

    return list(TEMPLATE_SETS[args.template_set])


def set_image_encoder_classification_head(
    net: torch.nn.Module,
    tokenizer_name: str,
    class_names: list[str],
    templates: list[str],
    context_length: Optional[int],
    text_batch_size: Optional[int],
) -> None:
    device = torch.device("cpu")
    tokenizer = get_tokenizer(tokenizer_name, context_length=context_length)
    text_embeddings = build_class_text_embeddings(
        net,
        tokenizer,
        class_names,
        templates,
        device=device,
        context_length=context_length,
        batch_size=text_batch_size,
    )

    classifier = net.image_encoder.classifier
    logit_scale = net.logit_scale.exp().to(dtype=text_embeddings.dtype)
    head_weight = logit_scale * text_embeddings
    head_bias: Optional[torch.Tensor] = None
    text_embedding_dim = text_embeddings.size(1)

    if isinstance(classifier, torch.nn.Identity):
        image_embedding_dim = net.image_encoder.embedding_size
        if image_embedding_dim != text_embedding_dim:
            raise ValueError(
                "Cannot export classification head: image encoder embedding dimension "
                f"{image_embedding_dim} does not match text embedding dimension {text_embedding_dim}"
            )
    elif isinstance(classifier, torch.nn.Linear):
        if classifier.out_features != text_embedding_dim:
            raise ValueError(
                "Cannot export classification head: image encoder classifier output dimension "
                f"{classifier.out_features} does not match text embedding dimension {text_embedding_dim}"
            )

        classifier_weight = classifier.weight.to(dtype=head_weight.dtype)
        head_weight = head_weight @ classifier_weight
        if classifier.bias is not None:
            classifier_bias = classifier.bias.to(dtype=head_weight.dtype)
            head_bias = logit_scale * (text_embeddings @ classifier_bias)
    else:
        raise ValueError(
            "Cannot export classification head: image encoder classifier must be Linear or Identity, "
            f"got {type(classifier).__name__}"
        )

    logit_bias = getattr(net, "logit_bias", None)
    if logit_bias is not None:
        logit_bias = logit_bias.to(dtype=head_weight.dtype)
        if head_bias is None:
            head_bias = logit_bias.expand(len(class_names))
        else:
            head_bias = head_bias + logit_bias

    net.image_encoder.reset_classifier(len(class_names))
    with torch.no_grad():
        net.image_encoder.classifier.weight.copy_(head_weight.to(dtype=net.image_encoder.classifier.weight.dtype))
        if net.image_encoder.classifier.bias is not None:
            if head_bias is None:
                net.image_encoder.classifier.bias.zero_()
            else:
                net.image_encoder.classifier.bias.copy_(head_bias.to(dtype=net.image_encoder.classifier.bias.dtype))
        elif head_bias is not None:
            raise ValueError("Cannot export classification head: reset image encoder classifier has no bias")


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "convert-model",
        allow_abbrev=False,
        help="convert PyTorch model to various formats",
        description="convert PyTorch model to various formats",
        epilog=(
            "Usage examples:\n"
            "python -m birder_clip.tools convert-model --network pe_core_l14 --resize 252\n"
            "python -m birder_clip.tools convert-model --network siglip_v2_vit_so400m_p14 --context-length 128\n"
            "python -m birder_clip.tools convert-model --network laion_clip_vit_h14 --config\n"
            "python -m birder_clip.tools convert-model --network laion_clip_convnext_v1_base --image-encoder-only\n"
            "python -m birder_clip.tools convert-model --network laion_clip_vit_h14 --image-classifier "
            "--classes eagle hawk falcon --template-set default\n"
            "python -m birder_clip.tools convert-model --network clip --image-encoder vit_b16 "
            "--text-encoder transformer_encoder --embed-dim 512 --tokenizer openai_clip_bpe "
            "--epoch 100 --resize 384\n"
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
    subparser.add_argument("--force", action="store_true", help="override existing model")
    subparser.add_argument("--class-file", type=str, help="Birder class file, one class per line")
    subparser.add_argument("--classes", type=str, nargs="*", help="class names for the exported classification head")
    subparser.add_argument(
        "--template-set",
        choices=sorted(TEMPLATE_SETS.keys()),
        default="default",
        help="built-in zero-shot template set",
    )
    subparser.add_argument("--template", action="append", help="additional prompt template")
    subparser.add_argument("--template-file", type=str, help="file with one prompt template per line")
    subparser.add_argument("--text-batch-size", type=int, metavar="N", help="class batch size for text encoding")

    format_group = subparser.add_mutually_exclusive_group(required=True)
    format_group.add_argument("--resize", type=int, nargs="+", metavar=("H", "W"), help="resize model (pt)")
    format_group.add_argument("--context-length", type=int, metavar="N", help="change text context length (pt)")
    format_group.add_argument("--config", default=False, action="store_true", help="generate model config json")
    format_group.add_argument(
        "--st", "--safetensors", default=False, action="store_true", help="convert to Safetensors"
    )
    format_group.add_argument(
        "--image-encoder-only", default=False, action="store_true", help="extract image encoder as a Birder checkpoint"
    )
    format_group.add_argument(
        "--image-classifier",
        default=False,
        action="store_true",
        help="extract image encoder as a Birder classifier checkpoint with a linear zero-shot head",
    )

    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    args.resize = cli.parse_size(args.resize)
    if args.tokenizer is None:
        args.tokenizer = registry.get_default_tokenizer(args.network)
    if args.tokenizer is None:
        args.tokenizer = "simple_tokenizer"

    if registry.exists(args.network, task=Task.IMAGE_TEXT) is False:
        raise cli.ValidationError(f"--network {args.network} not supported, see list-models tool for available options")

    registered_config = registry.all_nets[args.network.lower()].config  # type: ignore[misc]
    if registered_config is None:
        registered_config = {}
    if args.image_encoder is None and registered_config.get("image", {}).get("network") is None:
        raise cli.ValidationError("--image-encoder is required for this network")
    if args.text_encoder is None and registered_config.get("text", {}).get("network") is None:
        raise cli.ValidationError("--text-encoder is required for this network")
    if args.embed_dim is None and registered_config.get("embed_dim") is None:
        raise cli.ValidationError("--embed-dim is required for this network")

    reserved_model_config_keys = lib.get_reserved_model_config_keys(args.model_config)
    if len(reserved_model_config_keys) > 0:
        raise cli.ValidationError(
            "--model-config cannot contain keys with dedicated CLI flags: " f"{', '.join(reserved_model_config_keys)}"
        )
    if args.image_encoder_config is not None and "size" in args.image_encoder_config:
        raise cli.ValidationError("--image-encoder-config cannot contain size, use --resize")
    if args.image_encoder_config is not None and "input_channels" in args.image_encoder_config:
        raise cli.ValidationError("--image-encoder-config cannot contain input_channels")
    if args.text_encoder_config is not None and "context_length" in args.text_encoder_config:
        raise cli.ValidationError("--text-encoder-config cannot contain context_length")
    if args.image_classifier is False:
        if args.class_file is not None or args.classes is not None:
            raise cli.ValidationError("--class-file and --classes require --image-classifier")
        if args.template is not None or args.template_file is not None:
            raise cli.ValidationError("--template and --template-file require --image-classifier")
        if args.text_batch_size is not None:
            raise cli.ValidationError("--text-batch-size requires --image-classifier")

    if args.class_file is not None and args.classes is not None:
        raise cli.ValidationError("--class-file and --classes cannot be used together")
    if args.image_classifier is True and args.class_file is None and args.classes is None:
        raise cli.ValidationError("--image-classifier requires --class-file or --classes")
    if args.classes is not None and len(args.classes) == 0:
        raise cli.ValidationError("--classes requires at least one class name")
    if args.template is not None and args.template_file is not None:
        raise cli.ValidationError("--template and --template-file cannot be used together")
    if args.template_set != "default" and args.template is not None:
        raise cli.ValidationError("--template-set and --template cannot be used together")
    if args.template_set != "default" and args.template_file is not None:
        raise cli.ValidationError("--template-set and --template-file cannot be used together")

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
    )

    network_name = lib.get_image_text_network_name(
        args.network,
        tag=args.tag,
        image_encoder=args.image_encoder,
        text_encoder=args.text_encoder,
        embed_dim=args.embed_dim,
        tokenizer=args.tokenizer,
    )

    if args.resize is not None:
        network_name = f"{network_name}_{args.resize[0]}px"
    elif args.context_length is not None:
        network_name = f"{network_name}_cl{args.context_length}"

    if args.image_encoder_only is True or args.image_classifier is True:
        image_encoder_network = args.image_encoder
        if image_encoder_network is None:
            image_encoder_network = registered_config["image"]["network"]

        network_name = birder_lib.get_network_name(image_encoder_network, tag=args.tag)
        model_path = birder_fs_ops.model_path(network_name, epoch=args.epoch)
    else:
        model_path = fs_ops.model_path(network_name, epoch=args.epoch, st=args.st)

    if (
        (args.image_encoder_only is True or args.image_classifier is True)
        and model_path.exists() is True
        and args.force is False
    ):
        logger.warning("Converted model already exists... aborting")
        raise SystemExit(1)
    if model_path.exists() is True and args.force is False and args.config is False:
        logger.warning("Converted model already exists... aborting")
        raise SystemExit(1)

    logger.info(f"Saving converted model {model_path}...")
    if args.resize is not None:
        net.adjust_image_size(args.resize)
        model_info.signature["inputs"][0]["data_shape"][2] = args.resize[0]
        model_info.signature["inputs"][0]["data_shape"][3] = args.resize[1]
        fs_ops.checkpoint_model(
            network_name,
            args.epoch,
            net,
            model_info.signature,
            model_info.rgb_stats,
            optimizer=None,
            scheduler=None,
            scaler=None,
            model_base=None,
            external_config=model_info.custom_config,
        )

    elif args.context_length is not None:
        net.adjust_context_length(args.context_length)
        model_info.signature["inputs"][1]["data_shape"][1] = args.context_length
        fs_ops.checkpoint_model(
            network_name,
            args.epoch,
            net,
            model_info.signature,
            model_info.rgb_stats,
            optimizer=None,
            scheduler=None,
            scaler=None,
            model_base=None,
            external_config=model_info.custom_config,
        )

    elif args.config is True:
        config_export(net, model_info.signature, model_info.rgb_stats, model_path)

    elif args.st is True:
        fs_ops.save_st(
            net,
            str(model_path),
            net.task,
            model_info.signature,
            model_info.rgb_stats,
            external_config=model_info.custom_config,
        )

    elif args.image_encoder_only is True or args.image_classifier is True:
        image_encoder_config = {}
        registered_image_config = registered_config.get("image", {})
        if isinstance(registered_image_config, dict):
            image_encoder_config.update(registered_image_config.get("config", {}))

        if model_info.custom_config is not None:
            custom_image_config = model_info.custom_config.get("image", {})
            if isinstance(custom_image_config, dict):
                image_encoder_config.update(custom_image_config.get("config", {}))

        if args.image_encoder_config is not None:
            image_encoder_config.update(args.image_encoder_config)
        if len(image_encoder_config) == 0:
            image_encoder_config = None  # type: ignore[assignment]

        class_to_idx = None
        if args.image_classifier is True:
            class_to_idx = get_class_to_idx(args)
            tokenizer_name = args.tokenizer if args.tokenizer is not None else net.tokenizer_name
            context_length = lib.get_context_length_from_signature(model_info.signature)
            set_image_encoder_classification_head(
                net,
                tokenizer_name,
                list(class_to_idx.keys()),
                get_templates(args),
                context_length,
                args.text_batch_size,
            )

        image_encoder_export(
            net,
            network_name,
            args.epoch,
            model_info.signature,
            model_info.rgb_stats,
            image_encoder_config,
            class_to_idx,
        )
