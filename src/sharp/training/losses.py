"""Losses for SHARP fine-tuning.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sharp.models.normalizers import MeanStdNormalizer
from torch import nn
from torchvision import models


@dataclass
class FineTuneLossWeights:
    """Weights for fine-tuning losses."""

    color: float = 1.0
    percep: float = 0.1
    alpha: float = 0.05
    tv: float = 0.01
    occlusion_color: float = 1.0
    occlusion_delta: float = 0.01


class VGGPerceptualLoss(nn.Module):
    """Perceptual + Gram loss on target view (paper style)."""

    def __init__(self) -> None:
        """Initialize frozen VGG16 feature blocks for perceptual supervision."""
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features.eval()
        self.blocks = nn.ModuleList(
            [
                vgg[:4],
                vgg[4:9],
                vgg[9:16],
                vgg[16:23],
            ]
        )
        for p in self.parameters():
            p.requires_grad_(False)
        self.normalizer = MeanStdNormalizer(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    @staticmethod
    def _gram(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        f = x.flatten(2)
        return (f @ f.transpose(1, 2)) / (c * h * w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute feature and Gram losses across multiple VGG feature scales."""
        pred = self.normalizer(pred)
        target = self.normalizer(target)

        loss = pred.new_zeros(())
        feat_pred = pred
        feat_tgt = target
        for block in self.blocks:
            feat_pred = block(feat_pred)
            feat_tgt = block(feat_tgt)
            loss = loss + F.l1_loss(feat_pred, feat_tgt)
            loss = loss + F.l1_loss(self._gram(feat_pred), self._gram(feat_tgt))
        return loss


def total_variation_loss(depth: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV for smoothness regularization."""
    dx = torch.abs(depth[..., :, 1:] - depth[..., :, :-1]).mean()
    dy = torch.abs(depth[..., 1:, :] - depth[..., :-1, :]).mean()
    return dx + dy
