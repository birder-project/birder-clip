"""
Zero-shot text embedding helpers

Zero-shot classification compares image features against one text feature per
candidate class. When multiple prompt templates are used, this module follows
the OpenCLIP/OpenAI CLIP convention: encode every class/template prompt,
normalize prompt embeddings, average them per class and normalize the averaged
class embedding again.
"""

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn.functional as F

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
