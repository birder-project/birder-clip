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
from typing import Protocol

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import five_crop
from tqdm import tqdm

from birder_clip.net.base import BaseNet
from birder_clip.tokenizers.base import Tokenizer


class ZeroShotInferenceModule(Protocol):
    def __call__(
        self,
        inputs: torch.Tensor,
        *,
        tta: bool = False,
        return_logits: bool = False,
        grid_sizes: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...


class ZeroShotInference:
    def __init__(self, net: BaseNet, text_embeddings: torch.Tensor) -> None:
        self.net = net
        self.text_embeddings = text_embeddings

    def __call__(
        self,
        inputs: torch.Tensor,
        *,
        tta: bool = False,
        return_logits: bool = False,
        grid_sizes: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inputs = inputs.to(self.text_embeddings.device, non_blocking=True)
        batch_kwargs = {}
        if grid_sizes is not None:
            batch_kwargs = {
                "grid_sizes": grid_sizes.to(self.text_embeddings.device, non_blocking=True),
                "valid_mask": valid_mask.to(self.text_embeddings.device, non_blocking=True),  # type: ignore[union-attr]
            }

        if tta is True:
            _, _, H, W = inputs.size()
            crop_h = int(H * 0.8)
            crop_w = int(W * 0.8)
            tta_inputs = five_crop(inputs, size=[crop_h, crop_w])
            t = v2.Resize((H, W), interpolation=v2.InterpolationMode.BICUBIC, antialias=True)
            outs = []
            for tta_input in tta_inputs:
                image_embeddings = self.net.encode_image(t(tta_input), normalize=True)
                logits = self.net.forward_logits(image_embeddings, self.text_embeddings)
                if return_logits is True:
                    outs.append(logits)
                else:
                    outs.append(F.softmax(logits, dim=-1))

            return torch.stack(outs).mean(dim=0)

        image_embeddings = self.net.encode_image(inputs, normalize=True, **batch_kwargs)
        logits = self.net.forward_logits(image_embeddings, self.text_embeddings)
        if return_logits is True:
            return logits

        return F.softmax(logits, dim=-1)


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
    net: ZeroShotInferenceModule,
    dataloader: DataLoader,
    tta: bool = False,
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

    out_list: list[npt.NDArray[np.float32]] = []
    labels_list: list[npt.NDArray[np.int64]] = []
    sample_paths: list[str] = []
    sample_count = 0
    with tqdm(total=num_samples, initial=0, unit="images", unit_scale=True, leave=False) as progress:
        for file_paths, inputs, targets in dataloader:
            batch_kwargs = {}
            if isinstance(inputs, (tuple, list)):  # Assumes NaFlex collator inputs
                inputs, grid_sizes, valid_mask = inputs
                batch_kwargs = {
                    "grid_sizes": grid_sizes.to(device, non_blocking=True),
                    "valid_mask": valid_mask.to(device, non_blocking=True),
                }

            batch_size = inputs.size(0)

            # Inference
            inputs = inputs.to(dtype=model_dtype)
            with torch.amp.autocast(device.type, enabled=amp, dtype=amp_dtype):
                out = net(inputs, return_logits=return_logits, tta=tta, **batch_kwargs)
                out = out.cpu().float().numpy()

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
