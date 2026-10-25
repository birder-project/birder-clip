import argparse
import logging
import math
import random
import textwrap
from collections.abc import Callable
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any
from typing import Optional
from typing import get_args

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.v2.functional as F
from birder.common import cli
from birder.common import training_utils
from birder.conf import settings
from birder.data.collators.naflex import NaFlexBatchProcessor
from birder.data.collators.naflex import NaFlexBatchSpec
from birder.data.collators.naflex import NaFlexPathCollator
from birder.data.collators.naflex import resolve_naflex_batch_specs
from birder.data.dataloader.webdataset import make_wds_loader
from birder.data.datasets.directory import ImageLoaderName
from birder.data.datasets.directory import get_image_loader
from birder.data.datasets.naflex import NaFlexMultiScaleDataset
from birder.data.datasets.webdataset import get_wds_num_shards
from birder.data.datasets.webdataset import wds_args_from_info
from birder.data.transforms.classification import get_rgb_stats
from birder.data.transforms.classification import inference_preset
from birder.data.transforms.classification import reverse_preset
from birder.data.transforms.classification import training_preset
from birder.data.transforms.naflex import inference_preset as naflex_inference_preset
from birder.tools.show_iterator import _unpatchify_naflex
from torch.utils.data import DataLoader

from birder_clip.common import training_cli
from birder_clip.data.datasets.csv import ImageTextCsvDataset
from birder_clip.data.datasets.webdataset import make_wds_dataset
from birder_clip.data.datasets.webdataset import prepare_wds_args

logger = logging.getLogger(__name__)


def _caption_title(caption: str, limit: int = 90) -> str:
    return textwrap.shorten(" ".join(caption.split()), width=limit, placeholder="...")


def _show_single_sample(
    dataset: ImageTextCsvDataset, transform: Any, reverse_transform: Any, args: argparse.Namespace
) -> None:
    cols = 4
    rows = 3
    no_iterations = min(6, len(dataset))
    for index in random.sample(range(len(dataset)), no_iterations):
        img_path, img, caption = dataset[index]

        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"{Path(img_path).name}\n{caption}", wrap=True)
        grid_spec = fig.add_gridspec(ncols=cols, nrows=rows)

        # Show original
        ax = fig.add_subplot(grid_spec[0, 0:cols])
        ax.imshow(np.asarray(F.to_pil_image(img)))
        aug_type = f"naflex/{args.aug_type}" if args.naflex is True else args.aug_type
        ax.set_title(f"Original, aug type: {aug_type}")
        ax.axis("off")

        # Show transformed
        counter = 0
        for i in range(cols):
            for j in range(1, rows):
                transformed_img = F.to_pil_image(reverse_transform(transform(img)))

                ax = fig.add_subplot(grid_spec[j, i])
                ax.imshow(np.asarray(transformed_img))
                if args.naflex is True:
                    ax.set_title(f"#{counter}, {transformed_img.height}x{transformed_img.width}")
                else:
                    ax.set_title(f"#{counter}")
                ax.axis("off")
                counter += 1

        plt.show()


def _show_batches(
    dataset: ImageTextCsvDataset | torch.utils.data.IterableDataset,
    reverse_transform: Any,
    args: argparse.Namespace,
    naflex_specs: tuple[NaFlexBatchSpec, ...],
    naflex_transforms: Optional[dict[NaFlexBatchSpec, Callable[..., torch.Tensor]]],
    fixed_naflex_patch_size: int,
) -> None:
    cols = 4
    rows = 2
    batch_size = cols * rows
    collate_fn: Optional[Callable[[Any], Any]] = None
    naflex_batch_processor: Optional[NaFlexBatchProcessor] = None
    if args.naflex is True:
        collator_patch_size = None if naflex_transforms is not None else fixed_naflex_patch_size
        naflex_collator = NaFlexPathCollator(collator_patch_size)
        if naflex_transforms is not None:
            naflex_batch_processor = NaFlexBatchProcessor(naflex_collator, naflex_specs, naflex_transforms, seed=0)
            if args.wds is False:
                dataset = NaFlexMultiScaleDataset(dataset, naflex_batch_processor)

            collate_fn = naflex_batch_processor.base_collator
        else:
            collate_fn = naflex_collator

    if args.wds is True:
        wds_batcher: Optional[Callable[..., Iterator[tuple[Any, ...]]]] = None
        wds_num_shards: Optional[int] = None
        if naflex_batch_processor is not None:
            wds_batcher = partial(
                naflex_batch_processor.iter_batches,
                batch_size=batch_size,
                drop_last=False,
            )
            wds_num_shards = get_wds_num_shards(dataset)

        data_loader = make_wds_loader(
            dataset,
            batch_size,
            num_workers=1,
            prefetch_factor=1,
            collate_fn=None if naflex_batch_processor is not None else collate_fn,
            world_size=1,
            pin_memory=False,
            shuffle=args.wds_extra_shuffle,
            batcher=wds_batcher,
            num_shards=wds_num_shards,
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    for batch_idx, (paths, images, captions) in enumerate(data_loader):
        if batch_idx >= 6:
            break

        fig = plt.figure(constrained_layout=True)
        grid_spec = fig.add_gridspec(ncols=cols, nrows=rows)
        batch_patch_size = args.patch_size

        if args.naflex is True:
            patches, grid_sizes, valid_mask = images
            batch_patch_size = math.isqrt(patches.size(2) // args.channels)
            images = [
                _unpatchify_naflex(image_patches, grid_size, image_valid_mask, channels=args.channels)
                for image_patches, grid_size, image_valid_mask in zip(patches, grid_sizes, valid_mask)
            ]

        # Show transformed
        for index in range(min(batch_size, len(images))):
            transformed_img = F.to_pil_image(reverse_transform(images[index]))

            ax = fig.add_subplot(grid_spec[index // cols, index % cols])
            ax.imshow(np.asarray(transformed_img))
            title = f"{Path(paths[index]).name}\n{_caption_title(captions[index])}"
            if args.naflex is True:
                ax.set_title(
                    f"#{index}, {transformed_img.height}x{transformed_img.width}, p{batch_patch_size}\n{title}",
                    fontsize=8,
                )
            else:
                ax.set_title(title, fontsize=8)
            ax.axis("off")

        plt.show()


def show_iterator(args: argparse.Namespace) -> None:
    rgb_stats = get_rgb_stats(args.rgb_mode, args.rgb_mean, args.rgb_std)
    reverse_transform = reverse_preset(rgb_stats)
    naflex_specs: tuple[NaFlexBatchSpec, ...] = ()
    naflex_transforms: Optional[dict[NaFlexBatchSpec, Callable[..., torch.Tensor]]] = None
    fixed_naflex_patch_size = args.patch_size
    if args.naflex is True:
        inference_spec = resolve_naflex_batch_specs(args.size, args.patch_size)[0]
        if args.mode == "training":
            naflex_specs = resolve_naflex_batch_specs(
                args.size, args.patch_size, args.naflex_sizes, args.naflex_patch_sizes
            )
            logger.info(f"Resolved NaFlex batch specifications: {list(naflex_specs)}")
            if len(set(naflex_specs)) > 1:
                naflex_transforms = {
                    spec: training_utils.get_naflex_training_transform(args, spec) for spec in naflex_specs
                }
                transform = None if args.batch is True else naflex_transforms[naflex_specs[0]]
            else:
                spec = naflex_specs[0]
                fixed_naflex_patch_size = spec.patch_size
                transform = training_utils.get_naflex_training_transform(args, spec)
        elif args.mode == "inference":
            transform = naflex_inference_preset(
                inference_spec.patch_size, inference_spec.max_seq_len, rgb_stats, args.center_crop
            )
        else:
            raise ValueError(f"Unknown mode={args.mode}")
    elif args.mode == "training":
        transform = training_preset(
            size=args.size,
            aug_type=args.aug_type,
            level=args.aug_level,
            rgv_values=rgb_stats,
            resize_min_scale=args.resize_min_scale,
            re_prob=args.re_prob,
            use_grayscale=args.use_grayscale,
            ra_num_ops=args.ra_num_ops,
            ra_magnitude=args.ra_magnitude,
            augmix_severity=args.augmix_severity,
            clip_color_jitter_prob=args.clip_color_jitter_prob,
            clip_gray_prob=args.clip_gray_prob,
            timm_magnitude=args.timm_magnitude,
            timm_num_ops=args.timm_num_ops,
            timm_magnitude_std=args.timm_magnitude_std,
            simple_crop=args.simple_crop,
        )
    elif args.mode == "inference":
        transform = inference_preset(args.size, rgb_stats, args.center_crop, args.simple_crop)
    else:
        raise ValueError(f"Unknown mode={args.mode}")

    random_caption = args.mode == "training"
    if args.wds is True:
        wds_path: str | list[str]
        if args.wds_info is not None:
            wds_path, dataset_size = wds_args_from_info(args.wds_info, args.wds_split)
            if args.wds_size is not None:
                dataset_size = args.wds_size
        else:
            wds_path, dataset_size = prepare_wds_args(args.data_path, args.wds_size, torch.device("cpu"))

        dataset = make_wds_dataset(
            wds_path,
            dataset_size=dataset_size,
            shuffle=True,
            samples_names=True,
            transform=transform,
            image_decoder=args.img_loader,
            channels=args.channels,
            caption_fields=args.caption_fields,
            random_caption=random_caption,
            cache_dir=args.wds_cache_dir,
        )
    else:
        dataset = ImageTextCsvDataset(
            args.data_path,
            transforms=transform,
            loader=get_image_loader(args.img_loader, args.channels),
            caption_fields=args.caption_fields,
            random_caption=random_caption,
        )

    if args.batch is False:
        assert transform is not None
        raw_dataset = ImageTextCsvDataset(
            args.data_path,
            loader=get_image_loader(args.img_loader, args.channels),
            caption_fields=args.caption_fields,
            random_caption=random_caption,
        )
        _show_single_sample(raw_dataset, transform, reverse_transform, args)
    else:
        _show_batches(dataset, reverse_transform, args, naflex_specs, naflex_transforms, fixed_naflex_patch_size)


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "show-iterator",
        allow_abbrev=False,
        help="show image-text CSV iterator output vs input",
        description="show image-text CSV iterator output vs input",
        epilog=(
            "Usage examples:\n"
            "python -m birder_clip.tools show-iterator --mode training --aug-level 1 --data-path data/training.csv\n"
            "python -m birder_clip.tools show-iterator --mode inference --size 336 --batch "
            "--data-path data/validation.csv\n"
            "python -m birder_clip.tools show-iterator --mode training --aug-level 4 --batch --wds "
            "--wds-info https://huggingface.co/datasets/pixparse/cc12m-wds/resolve/main/_info.json "
            "--wds-split train\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--mode", type=str, choices=["training", "inference"], default="training", help="iterator mode"
    )
    subparser.add_argument("--size", type=int, nargs="+", default=[224], metavar=("H", "W"), help="image size")
    training_cli.add_data_aug_args(subparser)
    subparser.add_argument(
        "--batch", default=False, action="store_true", help="show a batch instead of a single sample"
    )
    subparser.add_argument("--naflex", default=False, action="store_true", help="use native aspect-ratio transforms")
    subparser.add_argument(
        "--naflex-sizes",
        type=int,
        nargs="+",
        metavar="SIZE",
        help=(
            "square-equivalent image sizes in pixels to sample once per training batch, actual image "
            "dimensions preserve aspect ratio (values must be divisible by --patch-size)"
        ),
    )
    subparser.add_argument(
        "--naflex-patch-sizes",
        type=int,
        nargs="+",
        metavar="PATCH_SIZE",
        help="patch sizes to sample once per training batch (values do not need to divide the image sizes)",
    )
    subparser.add_argument("--patch-size", type=int, default=16, help="patch size for NaFlex")
    subparser.add_argument(
        "--caption-fields",
        type=str,
        nargs="+",
        default=["caption"],
        metavar="FIELD",
        help=(
            "caption fields (CSV columns or WebDataset JSON keys) to sample uniformly from non-empty values "
            "in training mode (inference uses the first non-empty field)"
        ),
    )
    subparser.add_argument("--data-path", nargs="+", help="image-text CSV file paths")
    subparser.add_argument(
        "--img-loader",
        type=str,
        choices=get_args(ImageLoaderName),
        default="tv",
        help="backend to load and decode images",
    )
    subparser.add_argument(
        "--channels", type=int, default=settings.DEFAULT_NUM_CHANNELS, metavar="N", help="no. of image channels"
    )
    subparser.add_argument("--wds", default=False, action="store_true", help="use webdataset")
    subparser.add_argument(
        "--wds-info", type=str, action="append", metavar="FILE", help="one or more wds info file paths"
    )
    subparser.add_argument("--wds-cache-dir", type=str, metavar="DIR", help="webdataset cache directory")
    subparser.add_argument("--wds-size", type=int, metavar="N", help="size of the wds directory")
    subparser.add_argument(
        "--wds-split", type=str, default="training", metavar="NAME", help="wds dataset split to load"
    )
    subparser.add_argument(
        "--wds-extra-shuffle",
        default=False,
        action="store_true",
        help="enable cross-worker batch shuffling after batching",
    )
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    if args.wds is True and args.batch is False:
        raise cli.ValidationError("--wds requires --batch to be set")
    if args.naflex is True and args.simple_crop is True:
        raise cli.ValidationError("--naflex does not support --simple-crop")
    if args.naflex is True and args.wds is True and args.wds_extra_shuffle is True:
        raise cli.ValidationError("--naflex cannot be used with --wds-extra-shuffle")
    if args.naflex_sizes is not None and args.naflex is False:
        raise cli.ValidationError("--naflex-sizes requires --naflex")
    if args.naflex_sizes is not None and args.mode != "training":
        raise cli.ValidationError("--naflex-sizes is only supported in training mode")
    if args.naflex_patch_sizes is not None:
        if args.naflex is False:
            raise cli.ValidationError("--naflex-patch-sizes requires --naflex")
        if args.mode != "training":
            raise cli.ValidationError("--naflex-patch-sizes is only supported in training mode")
        if len(set(args.naflex_patch_sizes)) != len(args.naflex_patch_sizes):
            raise cli.ValidationError(f"--naflex-patch-sizes values must be unique, got {args.naflex_patch_sizes}")
    if len(args.caption_fields) == 0 or any(len(field.strip()) == 0 for field in args.caption_fields):
        raise cli.ValidationError("--caption-fields must contain non-empty field names")
    if len(set(args.caption_fields)) != len(args.caption_fields):
        raise cli.ValidationError("--caption-fields must not contain duplicate field names")
    if args.wds is False and args.data_path is None:
        raise cli.ValidationError("--data-path is required")
    if args.wds is True and args.data_path is None and args.wds_info is None:
        raise cli.ValidationError("--data-path is required")
    if args.wds is True and args.data_path is not None and len(args.data_path) > 1:
        raise cli.ValidationError(f"--wds can have at most 1 --data-path, got {len(args.data_path)}")
    if args.rgb_mean is not None and len(args.rgb_mean) != args.channels:
        raise cli.ValidationError(f"--rgb-mean must have {args.channels} values, got {len(args.rgb_mean)}")
    if args.rgb_std is not None and len(args.rgb_std) != args.channels:
        raise cli.ValidationError(f"--rgb-std must have {args.channels} values, got {len(args.rgb_std)}")

    if args.data_path is not None:
        if args.wds is True:
            args.data_path = args.data_path[0]
        else:
            args.data_path = [Path(path) for path in args.data_path]
    args.size = cli.parse_size(args.size)
    if args.naflex is True:
        if args.patch_size <= 0:
            raise cli.ValidationError("--patch-size must be positive")
        if args.size[0] % args.patch_size != 0 or args.size[1] % args.patch_size != 0:
            raise cli.ValidationError("--size must be divisible by --patch-size when using --naflex")
    show_iterator(args)
