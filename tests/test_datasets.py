import unittest
from pathlib import Path

import torch

from birder_clip.data.datasets.csv import ImageTextCsvDataset
from birder_clip.data.datasets.csv import _resolve_image_path
from birder_clip.data.datasets.fake import FakeImageTextData
from birder_clip.tokenizers import get_tokenizer

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class TestDatasets(unittest.TestCase):
    def test_resolve_image_path(self) -> None:
        self.assertEqual(_resolve_image_path("/data/image.jpeg", Path("/csv")), "/data/image.jpeg")
        self.assertEqual(_resolve_image_path("image.jpeg", Path("/csv")), "/csv/image.jpeg")

    def test_image_text_csv_dataset(self) -> None:
        tokenizer = get_tokenizer("simple_tokenizer")
        dataset = ImageTextCsvDataset(
            [FIXTURES_DIR / "image_text.csv"],
            transforms=lambda x: x + ".data",
            tokenizer=tokenizer,
            loader=lambda x: x,
        )

        self.assertEqual(len(dataset), 2)

        path, image, text = dataset[0]
        self.assertEqual(path, "images/bird1.jpeg")
        self.assertEqual(image, str(FIXTURES_DIR / "images/bird1.jpeg") + ".data")
        self.assertSequenceEqual(text.shape, (tokenizer.context_length,))
        self.assertEqual(text.dtype, torch.long)

        path, image, text = dataset[1]
        self.assertEqual(path, "images/bird2.jpeg")
        self.assertEqual(image, str(FIXTURES_DIR / "images/bird2.jpeg") + ".data")
        self.assertSequenceEqual(text.shape, (tokenizer.context_length,))
        self.assertEqual(text.dtype, torch.long)

    def test_fake_image_text_data(self) -> None:
        tokenizer = get_tokenizer("simple_tokenizer")
        dataset = FakeImageTextData(
            2,
            image_size=(3, 16, 16),
            num_classes=3,
            transform=lambda x: x,
            tokenizer=tokenizer,
        )

        path, image, text = dataset[1]

        self.assertEqual(path, "fake/path/1.jpeg")
        self.assertIsNotNone(image)
        self.assertSequenceEqual(text.shape, (tokenizer.context_length,))
        self.assertEqual(text.dtype, torch.long)
