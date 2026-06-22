import logging
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from birder_clip.inference.data_parallel import ZeroShotInferenceDataParallel
from birder_clip.inference.zero_shot import ZeroShotInference

logging.disable(logging.CRITICAL)


class ZeroShotTestModel(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.embedding_size = embedding_size

    def encode_image(self, image: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        features = image.flatten(1)[:, : self.embedding_size]
        if normalize is True:
            features = F.normalize(features, dim=-1)

        return features

    def forward_logits(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        return image_features @ text_features.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestZeroShotInferenceDataParallelCUDA(unittest.TestCase):
    def setUp(self) -> None:
        self.size = (32, 32)
        self.embedding_size = 16
        self.num_classes = 5
        self.device = torch.device("cuda")

        self.model = ZeroShotTestModel(self.embedding_size)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        text_embeddings = torch.eye(self.num_classes, self.embedding_size)
        self.text_embeddings = F.normalize(text_embeddings, dim=-1)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires at least 2 GPUs")
    def test_basic_forward(self) -> None:
        model_parallel = ZeroShotInferenceDataParallel(self.model, self.text_embeddings)

        batch_sizes = [1, 4, 7, 16, 31]
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                x = torch.randn(batch_size, 3, *self.size)
                with torch.inference_mode():
                    # Single GPU reference
                    model_single = self.model.to(self.device)
                    inference_single = ZeroShotInference(model_single, self.text_embeddings.to(self.device))
                    out_single = inference_single(
                        x.to(self.device),
                        return_logits=True,
                    )

                    # Multi-GPU
                    out_parallel = model_parallel(x, return_logits=True)

                # Check shapes match
                self.assertEqual(out_single.size(), out_parallel.size())

                # Check outputs are close
                diff = (out_single - out_parallel).abs().max()
                self.assertLess(diff.item(), 1e-5, f"Output mismatch for batch_size={batch_size}")

        # Test with CPU output
        model_parallel = ZeroShotInferenceDataParallel(self.model, self.text_embeddings, output_device="cpu")
        x = torch.randn(batch_size, 3, *self.size)
        with torch.inference_mode():
            out_parallel = model_parallel(x)

        self.assertEqual(out_parallel.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires at least 2 GPUs")
    def test_output_order_preservation(self) -> None:
        model_parallel = ZeroShotInferenceDataParallel(self.model, self.text_embeddings)

        # Create input where each sample is identifiable by its values
        batch_size = 17
        x = torch.zeros(batch_size, 3, *self.size)
        for i in range(batch_size):
            x[i, 0, 0, 0] = float(i)  # Each sample has unique value
            x[i, 0, 0, 1] = 1.0

        with torch.inference_mode():
            model_single = self.model.to(self.device)
            inference_single = ZeroShotInference(model_single, self.text_embeddings.to(self.device))
            out_single = inference_single(
                x.to(self.device),
                return_logits=True,
            )
            out_parallel = model_parallel(x, return_logits=True)

        torch.testing.assert_close(out_parallel, out_single, rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires at least 2 GPUs")
    def test_integration_with_zero_shot_inference_options(self) -> None:
        model_parallel = ZeroShotInferenceDataParallel(self.model, self.text_embeddings)

        batch_size = 4
        x = torch.randn(batch_size, 3, *self.size)
        with torch.inference_mode():
            out = model_parallel(x)
            self.assertEqual(out.shape, (batch_size, self.num_classes))
            torch.testing.assert_close(out.sum(dim=1), torch.ones(batch_size, device=out.device), rtol=1e-5, atol=1e-5)

            out = model_parallel(x, tta=True)
            self.assertEqual(out.shape, (batch_size, self.num_classes))
            torch.testing.assert_close(out.sum(dim=1), torch.ones(batch_size, device=out.device), rtol=1e-5, atol=1e-5)

            out = model_parallel(x, return_logits=True)
            self.assertEqual(out.shape, (batch_size, self.num_classes))

    def test_single_gpu_fallback(self) -> None:
        model_parallel = ZeroShotInferenceDataParallel(self.model, self.text_embeddings, device_ids=[0])
        batch_size = 4
        x = torch.randn(batch_size, 3, *self.size)

        with torch.inference_mode():
            out = model_parallel(x, return_logits=True)

        self.assertEqual(out.size(), (batch_size, self.num_classes))

        # Verify it produces same output as direct model
        with torch.inference_mode():
            model_single = self.model.to(self.device)
            inference_single = ZeroShotInference(model_single, self.text_embeddings.to(self.device))
            out_direct = inference_single(
                x.to(self.device),
                return_logits=True,
            )

        torch.testing.assert_close(out, out_direct, rtol=1e-5, atol=1e-5)
