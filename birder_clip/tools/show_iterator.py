import argparse
import logging
import random
import textwrap
from pathlib import Path
from typing import Any
from typing import get_args

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.v2.functional as F
from birder.common import cli
from birder.common import training_cli
from birder.conf import settings
from birder.data.datasets.directory import ImageLoaderName
from birder.data.datasets.directory import get_image_loader
from birder.data.transforms.classification import get_rgb_stats
from birder.data.transforms.classification import inference_preset
from birder.data.transforms.classification import reverse_preset
from birder.data.transforms.classification import training_preset
from torch.utils.data import DataLoader

from birder_clip.data.datasets.csv import ImageTextCsvDataset

logger = logging.getLogger(__name__)


def _caption_title(caption: str, limit: int = 90) -> str:
    return textwrap.shorten(" ".join(caption.split()), width=limit, placeholder="...")


def _show_single_sample(dataset: ImageTextCsvDataset, transform: Any, reverse_transform: Any) -> None:
    cols = 4
    rows = 3
    no_iterations = min(6, len(dataset))
    for index in random.sample(range(len(dataset)), no_iterations):
        img_path, img, caption = dataset[index]

        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"{Path(img_path).name}\n{caption}", wrap=True)
        grid_spec = fig.add_gridspec(ncols=cols, nrows=rows)

        ax = fig.add_subplot(grid_spec[0, 0:cols])
        ax.imshow(np.asarray(F.to_pil_image(img)))
        ax.set_title("Original")
        ax.axis("off")

        counter = 0
        for i in range(cols):
            for j in range(1, rows):
                transformed_img = F.to_pil_image(reverse_transform(transform(img)))

                ax = fig.add_subplot(grid_spec[j, i])
                ax.imshow(np.asarray(transformed_img))
                ax.set_title(f"#{counter}")
                ax.axis("off")
                counter += 1

        plt.show()


def _show_batches(dataset: ImageTextCsvDataset, reverse_transform: Any) -> None:
    cols = 4
    rows = 2
    batch_size = cols * rows
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch_idx, (paths, images, captions) in enumerate(data_loader):
        if batch_idx >= 6:
            break

        fig = plt.figure(constrained_layout=True)
        grid_spec = fig.add_gridspec(ncols=cols, nrows=rows)

        for index in range(min(batch_size, len(images))):
            img = F.to_pil_image(reverse_transform(images[index]))

            ax = fig.add_subplot(grid_spec[index // cols, index % cols])
            ax.imshow(np.asarray(img))
            ax.set_title(f"{Path(paths[index]).name}\n{_caption_title(captions[index])}", fontsize=8)
            ax.axis("off")

        plt.show()


def show_iterator(args: argparse.Namespace) -> None:
    rgb_stats = get_rgb_stats(args.rgb_mode, args.rgb_mean, args.rgb_std)
    reverse_transform = reverse_preset(rgb_stats)
    if args.mode == "training":
        transform = training_preset(
            args.size,
            args.aug_type,
            args.aug_level,
            rgb_stats,
            args.resize_min_scale,
            args.re_prob,
            args.use_grayscale,
            args.ra_num_ops,
            args.ra_magnitude,
            args.augmix_severity,
            args.simple_crop,
        )
    elif args.mode == "inference":
        transform = inference_preset(args.size, rgb_stats, args.center_crop, args.simple_crop)
    else:
        raise ValueError(f"Unknown mode={args.mode}")

    dataset = ImageTextCsvDataset(
        args.data_path,
        transforms=transform,
        loader=get_image_loader(args.img_loader, args.channels),
    )
    logger.info(dataset)

    if args.batch is False:
        raw_dataset = ImageTextCsvDataset(
            args.data_path,
            loader=get_image_loader(args.img_loader, args.channels),
        )
        _show_single_sample(raw_dataset, transform, reverse_transform)
    else:
        _show_batches(dataset, reverse_transform)


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
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--mode", type=str, choices=["training", "inference"], default="training", help="iterator mode"
    )
    subparser.add_argument("--size", type=int, nargs="+", default=[224], metavar=("H", "W"), help="image size")
    training_cli.add_data_aug_args(subparser)
    subparser.add_argument("--center-crop", type=float, default=1.0, help="center crop ratio during inference")
    subparser.add_argument(
        "--batch", default=False, action="store_true", help="show a batch instead of a single sample"
    )
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
    subparser.add_argument("--data-path", nargs="+", help="image-text CSV file paths")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    if args.data_path is None:
        raise cli.ValidationError("--data-path is required")
    if args.rgb_mean is not None and len(args.rgb_mean) != args.channels:
        raise cli.ValidationError(f"--rgb-mean must have {args.channels} values, got {len(args.rgb_mean)}")
    if args.rgb_std is not None and len(args.rgb_std) != args.channels:
        raise cli.ValidationError(f"--rgb-std must have {args.channels} values, got {len(args.rgb_std)}")

    args.data_path = [Path(path) for path in args.data_path]
    args.size = cli.parse_size(args.size)
    show_iterator(args)
