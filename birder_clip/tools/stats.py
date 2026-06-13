import argparse
import logging
import math
from collections import Counter
from collections.abc import Iterable
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from typing import Optional

import polars as pl
import torch
import webdataset as wds
from birder.common import cli
from birder.data.datasets.webdataset import wds_args_from_info

from birder_clip.data.datasets.csv import CAPTION_COLUMN
from birder_clip.data.datasets.webdataset import decode_caption
from birder_clip.data.datasets.webdataset import prepare_wds_args
from birder_clip.model_registry import Task
from birder_clip.model_registry import registry
from birder_clip.tokenizers import Tokenizer
from birder_clip.tokenizers import get_tokenizer
from birder_clip.tokenizers.registry import exists as tokenizer_exists

logger = logging.getLogger(__name__)


class TokenLengthStats:
    def __init__(self, context_length: int) -> None:
        self.context_length = context_length
        self.histogram: Counter[int] = Counter()
        self.count = 0
        self.total = 0
        self.total_sq = 0
        self.overflow_count = 0
        self.min_length: Optional[int] = None
        self.max_length: Optional[int] = None

    def update(self, length: int) -> None:
        self.histogram[length] += 1
        self.count += 1
        self.total += length
        self.total_sq += length * length
        if length > self.context_length:
            self.overflow_count += 1
        if self.min_length is None or length < self.min_length:
            self.min_length = length
        if self.max_length is None or length > self.max_length:
            self.max_length = length

    def mean(self) -> float:
        return self.total / self.count

    def std(self) -> float:
        mean = self.mean()
        variance = self.total_sq / self.count - mean * mean
        return math.sqrt(max(0.0, variance))

    def percentile(self, percentile: float) -> float:
        if self.count == 1:
            return float(self.max_length or 0)

        rank = (self.count - 1) * percentile / 100
        lower_idx = math.floor(rank)
        upper_idx = math.ceil(rank)
        lower_value = self._value_at(lower_idx)
        if lower_idx == upper_idx:
            return float(lower_value)

        upper_value = self._value_at(upper_idx)
        return lower_value + (upper_value - lower_value) * (rank - lower_idx)

    def _value_at(self, index: int) -> int:
        seen = 0
        for length, count in sorted(self.histogram.items()):
            seen += count
            if index < seen:
                return length

        raise IndexError(index)


def load_tokenizer(tokenizer: str) -> Tokenizer:
    if tokenizer_exists(tokenizer) is True:
        return get_tokenizer(tokenizer)

    if registry.exists(tokenizer, task=Task.IMAGE_TEXT) is True:
        tokenizer_name = registry.get_default_tokenizer(tokenizer)
        if tokenizer_name is None:
            raise cli.ValidationError(f"{tokenizer} does not define a tokenizer")

        return get_tokenizer(tokenizer_name)

    raise cli.ValidationError(f"{tokenizer} is not a registered tokenizer or image-text model")


def _iter_csv_captions(data_paths: Iterable[Path], batch_size: int) -> Iterator[str]:
    for data_path in data_paths:
        batches = (
            pl.scan_csv(
                Path(data_path),
                schema_overrides={CAPTION_COLUMN: pl.String},
            )
            .select(CAPTION_COLUMN)
            .collect_batches(chunk_size=batch_size)
        )

        for batch in batches:
            yield from batch.get_column(CAPTION_COLUMN)


def _iter_wds_captions(wds_path: str | list[str]) -> Iterator[str]:
    caption_key = "txt;json"
    caption_json_key = "caption"
    suffixes = tuple(caption_key.split(";"))

    dataset = (
        wds.WebDataset(
            wds_path,
            shardshuffle=False,
            nodesplitter=wds.split_by_node,
            select_files=lambda key_name: key_name.endswith(suffixes),
            empty_check=False,
        )
        .decode()
        .to_tuple(caption_key)
    )
    for (caption,) in dataset:
        yield decode_caption(caption, caption_json_key=caption_json_key)


def prompt_tokens(args: argparse.Namespace) -> None:
    tokenizer = load_tokenizer(args.tokenizer)
    if args.wds is True:
        wds_path: str | list[str]
        if args.wds_info is not None:
            wds_path, expected_size = wds_args_from_info(args.wds_info, args.wds_split)
            logger.info(f"Reading WDS split '{args.wds_split}' with {expected_size:,} expected samples")
        else:
            wds_path, _ = prepare_wds_args(args.data_path, 0, torch.device("cpu"))

        captions = _iter_wds_captions(wds_path)
    else:
        captions = _iter_csv_captions(args.data_path, args.csv_batch_size)

    stats = TokenLengthStats(tokenizer.context_length)
    for caption in captions:
        stats.update(len(tokenizer.encode(caption)) + tokenizer.num_special_tokens)

    logger.info(f"Loaded {stats.count:,} prompts")
    logger.info(f"Context length: {stats.context_length}")
    logger.info(f"Special tokens: {tokenizer.num_special_tokens}")
    logger.info(f"Min tokens: {stats.min_length:,}")
    logger.info(f"Max tokens: {stats.max_length:,}")
    logger.info(f"Mean tokens: {stats.mean():.2f}")
    logger.info(f"Median tokens: {stats.percentile(50):.2f}")
    logger.info(f"Std tokens: {stats.std():.2f}")
    logger.info(f"P95 tokens: {stats.percentile(95):.2f}")
    logger.info(f"P99 tokens: {stats.percentile(99):.2f}")
    logger.info(
        f"Prompts exceeding context length: {stats.overflow_count:,} ({stats.overflow_count / stats.count:.2%})"
    )


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "stats",
        allow_abbrev=False,
        help="show image-text dataset statistics",
        description="show image-text dataset statistics",
        epilog=(
            "Usage examples:\n"
            "python -m birder_clip.tools stats --prompt-tokens --tokenizer siglip_v2_vit_so400m_p14 "
            "--data-path data/training.csv data/validation.csv\n"
            "python -m birder_clip.tools stats --prompt-tokens --tokenizer openai_clip_bpe --wds "
            "--data-path ~/Datasets/cc12m\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("--prompt-tokens", default=False, action="store_true", help="show prompt token statistics")
    subparser.add_argument(
        "--tokenizer", type=str, default="simple_tokenizer", help="tokenizer or image-text model name"
    )
    subparser.add_argument("--data-path", nargs="+", help="image-text CSV file paths")
    subparser.add_argument("--csv-batch-size", type=int, default=50_000, metavar="N", help="CSV read batch size")
    subparser.add_argument("--wds", default=False, action="store_true", help="use webdataset")
    subparser.add_argument(
        "--wds-info", type=str, action="append", metavar="FILE", help="one or more wds info file paths"
    )
    subparser.add_argument(
        "--wds-split", type=str, default="training", metavar="NAME", help="wds dataset split to load"
    )
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    if args.wds is True and args.data_path is not None and args.wds_info is not None:
        raise cli.ValidationError("--data-path and --wds-info are mutually exclusive when --wds is set")
    if args.wds is True and args.data_path is not None and len(args.data_path) > 1:
        raise cli.ValidationError(f"--wds can have at most 1 --data-path, got {len(args.data_path)}")

    if args.data_path is not None:
        if args.wds is True:
            args.data_path = args.data_path[0]
        else:
            args.data_path = [Path(path) for path in args.data_path]

    if args.prompt_tokens is True:
        prompt_tokens(args)
