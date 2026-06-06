import unittest

import torch

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
