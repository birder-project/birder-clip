from typing import Any
from typing import Optional

from torchvision.datasets import FakeData

from birder_clip.tokenizers import Tokenizer


class FakeImageTextData(FakeData):
    def __init__(self, *args: Any, tokenizer: Optional[Tokenizer] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def __getitem__(self, index: int) -> tuple[str, Any, Any]:
        img, target = super().__getitem__(index)
        path = f"fake/path/{index}.jpeg"
        caption = f"fake caption {target}"
        if self.tokenizer is not None:
            text = self.tokenizer([caption])[0]
        else:
            text = caption

        return (path, img, text)
