from typing import Optional
from typing import Protocol

import torch


class Tokenizer(Protocol):
    def __call__(self, texts: str | list[str], context_length: Optional[int] = None) -> torch.LongTensor: ...
