import logging
from collections.abc import Callable
from functools import partial
from typing import Any
from typing import Optional

import torch
import webdataset as wds
from birder.conf import settings
from birder.data.datasets import webdataset as birder_wds

from birder_clip.tokenizers import Tokenizer

logger = logging.getLogger(__name__)


def decode_caption(caption: Any, caption_json_key: str = "caption") -> str:
    if isinstance(caption, dict):
        if caption_json_key not in caption:
            raise ValueError(f"WebDataset JSON sample missing '{caption_json_key}' key")

        caption = caption[caption_json_key]

    if isinstance(caption, bytes):
        caption = caption.decode("utf-8")

    if isinstance(caption, str) is False:
        raise TypeError(f"WebDataset caption must be a string, got {type(caption).__name__}")

    return caption  # type: ignore[no-any-return]


def tokenize_caption(caption: str, tokenizer: Tokenizer) -> torch.Tensor:
    return tokenizer([caption])[0]


def make_wds_dataset(
    wds_path: str | list[str],
    dataset_size: int,
    shuffle: bool,
    samples_names: bool,
    transform: Callable[..., torch.Tensor],
    image_decoder: birder_wds.WDSImageDecoderSpec = "tv",
    channels: int = settings.DEFAULT_NUM_CHANNELS,
    tokenizer: Optional[Tokenizer] = None,
    *,
    caption_key: str = "txt;json",  # WebDataset picks the first present key, so txt takes precedence over json
    caption_json_key: str = "caption",
    cache_dir: Optional[str] = None,
    shuffle_buffer_size: Optional[int] = None,
    shuffle_initial_size: Optional[int] = None,
) -> torch.utils.data.IterableDataset:
    if shuffle is True:
        shardshuffle = 500
    else:
        shardshuffle = False

    dataset = wds.WebDataset(
        wds_path, shardshuffle=shardshuffle, nodesplitter=wds.split_by_node, cache_dir=cache_dir, empty_check=False
    )
    if shuffle is True:
        if shuffle_buffer_size is None:
            shuffle_buffer_size = birder_wds.WDS_SHUFFLE_SIZE
        if shuffle_initial_size is None:
            shuffle_initial_size = birder_wds.WDS_INITIAL_SIZE

        logger.debug(f"Using buffer size of {shuffle_buffer_size} for shuffle with {shuffle_initial_size} initial size")
        dataset = dataset.shuffle(shuffle_buffer_size, initial=shuffle_initial_size)

    return_keys = ["jpeg;jpg;png;webp"]
    return_keys = return_keys + [caption_key]
    if samples_names is True:
        return_keys = ["__url__", "__key__"] + return_keys

    if isinstance(image_decoder, str):
        decoder = birder_wds.get_wds_image_decoder(image_decoder, channels)
    else:
        decoder = image_decoder

    dataset = dataset.with_length(dataset_size, silent=True).decode(decoder).to_tuple(*return_keys)

    caption_decoder = partial(decode_caption, caption_json_key=caption_json_key)
    if samples_names is True:
        dataset = dataset.map(birder_wds.decode_sample_name)
        dataset = dataset.map_tuple(birder_wds.identity, transform, caption_decoder)
    else:
        dataset = dataset.map_tuple(transform, caption_decoder)

    if tokenizer is not None:
        text_transform = partial(tokenize_caption, tokenizer=tokenizer)
        if samples_names is True:
            dataset = dataset.map_tuple(birder_wds.identity, birder_wds.identity, text_transform)
        else:
            dataset = dataset.map_tuple(birder_wds.identity, text_transform)

    return dataset


def wds_size(wds_path: str, device: torch.device, select_suffix: str | tuple[str, ...] = ("txt", "json")) -> int:
    return birder_wds.wds_size(wds_path, device, select_suffix=select_suffix)


def prepare_wds_args(
    data_path: str, size: Optional[int], device: torch.device, select_suffix: str | tuple[str, ...] = ("txt", "json")
) -> tuple[str, int]:
    return birder_wds.prepare_wds_args(data_path, size, device, select_suffix=select_suffix)
