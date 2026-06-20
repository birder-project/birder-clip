import unittest

import torch
import torch.nn.functional as F

from birder_clip.loss.coca import CoCaLoss
from birder_clip.loss.contrastive import CLIPLoss


class TestContrastiveLoss(unittest.TestCase):
    def test_clip_loss(self) -> None:
        criterion = CLIPLoss()
        image_features = torch.rand(4, 8, requires_grad=True)
        text_features = torch.rand(4, 8, requires_grad=True)
        logit_scale = torch.ones([])

        losses = criterion(image_features, text_features, logit_scale)
        loss = losses["contrastive_loss"]

        self.assertFalse(torch.isnan(loss).any())
        self.assertEqual(loss.ndim, 0)

        loss.backward()
        self.assertIsNotNone(image_features.grad)
        self.assertIsNotNone(text_features.grad)
        self.assertTrue(torch.isfinite(image_features.grad).all().item())
        self.assertTrue(torch.isfinite(text_features.grad).all().item())

    def test_clip_loss_with_logit_bias(self) -> None:
        criterion = CLIPLoss()
        image_features = torch.rand(4, 8)
        text_features = torch.rand(4, 8)
        logit_scale = torch.ones([])
        logit_bias = torch.ones([])

        losses = criterion(image_features, text_features, logit_scale, logit_bias=logit_bias)

        self.assertListEqual(list(losses.keys()), ["contrastive_loss"])
        self.assertFalse(torch.isnan(losses["contrastive_loss"]).any())
        self.assertEqual(losses["contrastive_loss"].ndim, 0)

    def test_clip_loss_unpaired_batch(self) -> None:
        criterion = CLIPLoss()
        image_features = torch.rand(4, 8)
        text_features = torch.rand(3, 8)
        logit_scale = torch.ones([])

        with self.assertRaises(ValueError):
            criterion(image_features, text_features, logit_scale)


class TestCoCaLoss(unittest.TestCase):
    def test_coca_loss(self) -> None:
        criterion = CoCaLoss()
        image_features = torch.rand(4, 8, requires_grad=True)
        text_features = torch.rand(4, 8, requires_grad=True)
        logits = torch.rand(4, 5, 10, requires_grad=True)
        texts = torch.tensor(
            [
                [1, 2, 3, 0, 0],
                [1, 4, 5, 6, 0],
                [1, 7, 8, 9, 0],
                [1, 2, 4, 6, 8],
            ]
        )
        logit_scale = torch.ones([])

        losses = criterion(image_features, text_features, logits, texts, logit_scale)

        self.assertListEqual(list(losses.keys()), ["contrastive_loss", "caption_loss"])
        self.assertFalse(torch.isnan(losses["contrastive_loss"]).any())
        self.assertFalse(torch.isnan(losses["caption_loss"]).any())
        self.assertEqual(losses["contrastive_loss"].ndim, 0)
        self.assertEqual(losses["caption_loss"].ndim, 0)

        loss = sum(losses.values())
        loss.backward()
        self.assertIsNotNone(image_features.grad)
        self.assertIsNotNone(text_features.grad)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(image_features.grad).all().item())
        self.assertTrue(torch.isfinite(text_features.grad).all().item())
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_coca_loss_weights(self) -> None:
        image_features = torch.rand(4, 8)
        text_features = torch.rand(4, 8)
        logits = torch.rand(4, 5, 10)
        texts = torch.tensor(
            [
                [1, 2, 3, 0, 0],
                [1, 4, 5, 6, 0],
                [1, 7, 8, 9, 0],
                [1, 2, 4, 6, 8],
            ]
        )
        logit_scale = torch.ones([])

        unweighted_losses = CoCaLoss()(image_features, text_features, logits, texts, logit_scale)
        weighted_losses = CoCaLoss(caption_loss_weight=2.0, clip_loss_weight=0.5)(
            image_features,
            text_features,
            logits,
            texts,
            logit_scale,
        )

        self.assertTrue(
            torch.allclose(weighted_losses["contrastive_loss"], unweighted_losses["contrastive_loss"] * 0.5)
        )
        self.assertTrue(torch.allclose(weighted_losses["caption_loss"], unweighted_losses["caption_loss"] * 2.0))

    def test_coca_caption_loss_ignores_padding(self) -> None:
        criterion = CoCaLoss(clip_loss_weight=0.0, pad_token_id=0)
        image_features = torch.rand(2, 8)
        text_features = torch.rand(2, 8)
        logits = torch.rand(2, 4, 6)
        texts = torch.tensor([[1, 2, 0, 0], [1, 3, 4, 0]])
        logit_scale = torch.ones([])

        losses = criterion(image_features, text_features, logits, texts, logit_scale)
        expected_caption_loss = F.cross_entropy(
            logits[:, :-1].permute(0, 2, 1),
            texts[:, 1:],
            ignore_index=0,
        )

        self.assertEqual(losses["contrastive_loss"].item(), 0.0)
        self.assertTrue(torch.allclose(losses["caption_loss"], expected_caption_loss))
