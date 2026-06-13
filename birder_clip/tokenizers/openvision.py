from typing import Any
from typing import Optional

import torch
from transformers import AutoTokenizer

from birder_clip.conf import settings
from birder_clip.tokenizers.base import get_clean_fn
from birder_clip.tokenizers.registry import register_tokenizer


class OpenVisionTokenizer:
    def __init__(
        self,
        source: str,
        *,
        context_length: int = 80,
        clean: str = "whitespace",
        vocab_size: int = 32000,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        cls_token_id: int = 101,
        pad_token_id: int = 0,
        **kwargs: Any,
    ) -> None:
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.cls_token_id = cls_token_id
        self.pad_token_id = pad_token_id
        self.num_special_tokens = 3
        self.tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            str(settings.TOKENIZERS_DIR.joinpath(source)),
            local_files_only=True,
            **kwargs,
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(  # type: ignore[no-any-return]
            self.clean_fn(text),
            add_special_tokens=False,
            truncation=False,
            padding=False,
        ).input_ids

    def __call__(self, texts: str | list[str], context_length: Optional[int] = None) -> torch.LongTensor:
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        max_body_length = context_length - self.num_special_tokens
        input_ids = torch.full((len(texts), context_length), self.pad_token_id, dtype=torch.long)
        for idx, text in enumerate(texts):
            body_token_ids = self.encode(text)[:max_body_length]
            token_ids = [self.bos_token_id] + body_token_ids + [self.eos_token_id]
            input_ids[idx, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
            input_ids[idx, -1] = self.cls_token_id

        return input_ids


register_tokenizer("openvision", OpenVisionTokenizer, source="bert-base-uncased", context_length=80, vocab_size=32000)
