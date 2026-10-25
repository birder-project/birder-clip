import sys
from collections.abc import Iterator
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from birder_clip.net.base import BaseNet

DataloaderInferenceResult = tuple[list[str], npt.NDArray[np.float32]]


def infer_dataloader_iter(
    device: torch.device,
    net: BaseNet,
    dataloader: DataLoader,
    model_dtype: torch.dtype = torch.float32,
    amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    num_samples: Optional[int] = None,
    chunk_size: Optional[float] = None,
) -> Iterator[DataloaderInferenceResult]:
    if chunk_size is None:
        chunk_size = float("inf")

    net.to(device, dtype=model_dtype)
    embeddings_list: list[npt.NDArray[np.float32]] = []
    sample_paths: list[str] = []
    sample_count = 0
    with tqdm(total=num_samples, initial=0, unit="images", unit_scale=True, leave=False) as progress:
        for file_paths, inputs, _targets in dataloader:
            batch_kwargs = {}
            if isinstance(inputs, (tuple, list)):  # Assumes NaFlex collator inputs
                inputs, grid_sizes, valid_mask = inputs
                batch_kwargs = {
                    "grid_sizes": grid_sizes.to(device, non_blocking=True),
                    "valid_mask": valid_mask.to(device, non_blocking=True),
                }

            batch_size = inputs.size(0)

            # Inference
            inputs = inputs.to(device, dtype=model_dtype)
            with torch.amp.autocast(device.type, enabled=amp, dtype=amp_dtype):
                embeddings = net.encode_image(inputs, normalize=True, **batch_kwargs)
                embeddings = embeddings.cpu().float().numpy()

            embeddings_list.append(embeddings)

            # Set sample list
            sample_paths.extend(file_paths)

            # Update progress bar
            progress.update(n=batch_size)

            # Yield results when we reach chunk_size
            sample_count += batch_size
            if sample_count >= chunk_size:
                with tqdm.external_write_mode(file=sys.stderr):
                    yield (sample_paths, np.concatenate(embeddings_list, axis=0))

                # Reset for next chunk
                embeddings_list = []
                sample_paths = []
                sample_count = 0

    if len(embeddings_list) > 0:
        yield (sample_paths, np.concatenate(embeddings_list, axis=0))
