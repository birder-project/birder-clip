from bisect import bisect_right
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Optional

import polars as pl
import torch
from birder.data.datasets.directory import tv_rgb_loader

from birder_clip.common.lib import select_caption
from birder_clip.tokenizers import Tokenizer

IMAGE_PATH_COLUMN = "image_path"
CAPTION_COLUMN = "caption"


def _resolve_image_path(path: str, csv_dir: Path) -> str:
    image_path = Path(path)
    if image_path.is_absolute() is True:
        return path

    return str(csv_dir / image_path)


class ImageTextCsvDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_paths: list[str | Path],
        transforms: Optional[Callable[..., Any]] = None,
        tokenizer: Optional[Tokenizer] = None,
        loader: Callable[[str], Any] = tv_rgb_loader,
        caption_fields: Sequence[str] = (CAPTION_COLUMN,),
        random_caption: bool = False,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.loader = loader
        self.random_caption = random_caption
        self.csv_dirs: list[Path] = []
        self.csv_offsets: list[int] = []

        frames = []
        length = 0
        columns = [IMAGE_PATH_COLUMN, *caption_fields]
        for csv_path in csv_paths:
            csv_path = Path(csv_path).expanduser().absolute()
            frame = pl.read_csv(
                csv_path,
                columns=columns,
                schema_overrides={column: pl.String for column in columns},
            )
            frames.append(frame)
            length += len(frame)
            self.csv_offsets.append(length)
            self.csv_dirs.append(csv_path.parent)

        frame = pl.concat(frames)
        self.paths = frame.get_column(IMAGE_PATH_COLUMN)
        self.captions = frame.select(caption_fields)

    def __getitem__(self, index: int) -> tuple[str, Any, Any]:
        path = self.paths[index]
        csv_idx = bisect_right(self.csv_offsets, index)
        image_path = _resolve_image_path(path, self.csv_dirs[csv_idx])
        caption = select_caption(self.captions.row(index), random_caption=self.random_caption)
        image = self.loader(image_path)
        if self.transforms is not None:
            image = self.transforms(image)

        if self.tokenizer is not None:
            text = self.tokenizer([caption])[0]
        else:
            text = caption

        return (path, image, text)

    def __len__(self) -> int:
        return len(self.paths)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of data points: {self.__len__()}"]
        if self.transforms is not None:
            body += [repr(self.transforms)]
        if self.tokenizer is not None:
            body += [repr(self.tokenizer)]

        lines = [head] + ["    " + line for line in body]

        return "\n".join(lines)
