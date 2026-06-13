"""
Zero-shot text embedding helpers

Zero-shot classification compares image features against one text feature per
candidate class. When multiple prompt templates are used, this module follows
the OpenCLIP/OpenAI CLIP convention: encode every class/template prompt,
normalize prompt embeddings, average them per class and normalize the averaged
class embedding again.
"""

import sys
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from birder_clip.net.base import BaseNet
from birder_clip.tokenizers.base import Tokenizer


def render_prompts(class_names: Sequence[str], templates: Sequence[str]) -> list[str]:
    return [template.format(class_name) for class_name in class_names for template in templates]


def build_class_text_embeddings(
    net: BaseNet,
    tokenizer: Tokenizer,
    class_names: Sequence[str],
    templates: Sequence[str],
    *,
    device: torch.device,
    context_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    num_templates = len(templates)
    if batch_size is None:
        batch_size = len(class_names)

    class_text_embeddings = []
    with torch.inference_mode():
        for start in range(0, len(class_names), batch_size):
            batch_class_names = class_names[start : start + batch_size]
            prompts = render_prompts(batch_class_names, templates)
            tokens = tokenizer(prompts, context_length=context_length).to(device)
            with torch.amp.autocast(device.type, enabled=amp, dtype=amp_dtype):
                class_embeddings = net.encode_text(tokens, normalize=True)

            class_embeddings = class_embeddings.reshape(len(batch_class_names), num_templates, -1).mean(dim=1)
            class_embeddings = F.normalize(class_embeddings, dim=-1)
            class_text_embeddings.append(class_embeddings)

    return torch.concat(class_text_embeddings, dim=0)


DataloaderInferenceResult = tuple[list[str], npt.NDArray[np.float32], npt.NDArray[np.int64]]


def infer_dataloader_iter(
    device: torch.device,
    net: BaseNet | torch.ScriptModule,
    dataloader: DataLoader,
    text_embeddings: torch.Tensor,
    return_logits: bool = False,
    model_dtype: torch.dtype = torch.float32,
    amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    num_samples: Optional[int] = None,
    batch_callback: Optional[Callable[[list[str], npt.NDArray[np.float32], npt.NDArray[np.int64]], None]] = None,
    chunk_size: Optional[float] = None,
) -> Iterator[DataloaderInferenceResult]:
    if chunk_size is None:
        chunk_size = float("inf")

    net.to(device, dtype=model_dtype)
    out_list: list[npt.NDArray[np.float32]] = []
    labels_list: list[npt.NDArray[np.int64]] = []
    sample_paths: list[str] = []
    sample_count = 0
    with tqdm(total=num_samples, initial=0, unit="images", unit_scale=True, leave=False) as progress:
        for file_paths, inputs, targets in dataloader:
            batch_size = inputs.size(0)

            # Inference
            inputs = inputs.to(device, dtype=model_dtype)
            with torch.amp.autocast(device.type, enabled=amp, dtype=amp_dtype):
                image_embeddings = net.encode_image(inputs, normalize=True)
                logits = net.forward_logits(image_embeddings, text_embeddings)
                if return_logits is True:
                    out = logits.cpu().float().numpy()
                else:
                    out = F.softmax(logits, dim=-1).cpu().float().numpy()

            out_list.append(out)

            # Set labels and sample list
            batch_labels = targets.cpu().numpy()
            labels_list.append(batch_labels)
            sample_paths.extend(file_paths)

            if batch_callback is not None:
                batch_callback(file_paths, out, batch_labels)

            # Update progress bar
            progress.update(n=batch_size)

            # Yield results when we reach chunk_size
            sample_count += batch_size
            if sample_count >= chunk_size:
                with tqdm.external_write_mode(file=sys.stderr):
                    yield (sample_paths, np.concatenate(out_list, axis=0), np.concatenate(labels_list))

                # Reset for next chunk
                out_list = []
                labels_list = []
                sample_paths = []
                sample_count = 0

    if len(out_list) > 0:
        yield (sample_paths, np.concatenate(out_list, axis=0), np.concatenate(labels_list))
