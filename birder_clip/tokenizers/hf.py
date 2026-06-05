from pathlib import Path
from typing import Any
from typing import Optional

import torch
from transformers import AutoTokenizer

from birder_clip.conf import settings
from birder_clip.tokenizers.openai_clip_bpe import get_clean_fn


def hf_tokenizer_path(pretrained_model_name_or_path: str) -> Path:
    return settings.TOKENIZERS_DIR.joinpath(pretrained_model_name_or_path)


class HFTokenizer:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        *,
        context_length: int = 77,
        clean: str = "whitespace",
        **kwargs: Any,
    ) -> None:
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)
        self.tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            str(hf_tokenizer_path(pretrained_model_name_or_path)),
            local_files_only=True,
            **kwargs,
        )

    def __call__(self, texts: str | list[str], context_length: Optional[int] = None) -> torch.LongTensor:
        if isinstance(texts, str):
            texts = [texts]

        return self.tokenizer(
            [self.clean_fn(text) for text in texts],
            return_tensors="pt",
            max_length=context_length or self.context_length,
            padding="max_length",
            truncation=True,
        ).input_ids
