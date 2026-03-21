"""Loss functions for SHARP fine-tuning.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import torch
import torch.nn.functional as F
import torchvision.models as tv_models
from torch import nn

from sharp.utils.gsplat import RenderingOutputs


@dataclasses.dataclass
class FineTuneLossWeights:
    """Weights for the fine-tuning losses."""

    color: float = 1.0
    alpha: float = 0.05
    perceptual: float = 0.1
    depth: float = 0.0
    depth_tv: float = 0.01
    scale_reg: float = 0.0


class FineTuneLossOutputs(NamedTuple):
    """Structured loss output."""

    total: torch.Tensor
    color: torch.Tensor
    alpha: torch.Tensor
    perceptual: torch.Tensor
    depth: torch.Tensor
    depth_tv: torch.Tensor
    scale_reg: torch.Tensor


class VGGPerceptualLoss(nn.Module):
    """Lightweight VGG perceptual loss used on the novel view."""

    def __init__(self) -> None:
        """Initialize the frozen VGG16 feature extractor."""
        super().__init__()
        features = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        split_indices = {3, 8, 15, 22}
        blocks = []
        start = 0
        for end in sorted(split_indices):
            blocks.append(nn.Sequential(*features[start : end + 1]))
            start = end + 1
        self.blocks = nn.ModuleList(blocks)
        self.requires_grad_(False)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[None, :, None, None],
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[None, :, None, None],
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the perceptual loss between prediction and target."""
        x = (prediction - self.mean) / self.std
        y = (target - self.mean) / self.std
        total = x.new_tensor(0.0)
        for block in self.blocks:
            x = block(x)
            y = block(y)
            total = total + F.l1_loss(x, y)
        return total


class FineTuneLoss(nn.Module):
    """Paper-inspired fine-tuning objectives.

    When `source_depth` is absent, depth supervision is disabled and only the
    image-space and regularization terms are optimized.
    """

    def __init__(
        self,
        weights: FineTuneLossWeights | None = None,
        use_perceptual: bool = True,
    ) -> None:
        """Initialize the fine-tuning loss."""
        super().__init__()
        self.weights = weights or FineTuneLossWeights()
        self.perceptual = VGGPerceptualLoss() if use_perceptual else None

    @staticmethod
    def _depth_tv(depth_layers: torch.Tensor) -> torch.Tensor:
        """Total-variation regularizer on the second depth layer in inverse depth."""
        if depth_layers.shape[1] < 2:
            return depth_layers.new_tensor(0.0)
        inverse_depth = depth_layers[:, 1:2].clamp(min=1e-4).reciprocal()
        grad_x = torch.abs(inverse_depth[..., :, 1:] - inverse_depth[..., :, :-1]).mean()
        grad_y = torch.abs(inverse_depth[..., 1:, :] - inverse_depth[..., :-1, :]).mean()
        return grad_x + grad_y

    def forward(
        self,
        source_render: RenderingOutputs,
        target_render: RenderingOutputs,
        batch: dict[str, torch.Tensor | None],
        aligned_depth: torch.Tensor,
        alignment_map: torch.Tensor | None = None,
    ) -> FineTuneLossOutputs:
        """Compute the weighted training loss for one batch."""
        source_image = batch["source_image"]
        target_image = batch["target_image"]
        source_depth = batch.get("source_depth")

        assert isinstance(source_image, torch.Tensor)
        assert isinstance(target_image, torch.Tensor)

        color = F.l1_loss(source_render.color, source_image) + F.l1_loss(
            target_render.color,
            target_image,
        )

        alpha_target_src = torch.ones_like(source_render.alpha)
        alpha_target_tgt = torch.ones_like(target_render.alpha)
        alpha = F.binary_cross_entropy(
            source_render.alpha.clamp(1e-6, 1 - 1e-6),
            alpha_target_src,
        ) + F.binary_cross_entropy(
            target_render.alpha.clamp(1e-6, 1 - 1e-6),
            alpha_target_tgt,
        )

        perceptual = color.new_tensor(0.0)
        if self.perceptual is not None:
            perceptual = self.perceptual(target_render.color, target_image)

        depth = color.new_tensor(0.0)
        if isinstance(source_depth, torch.Tensor):
            predicted_disparity = aligned_depth[:, 0:1].clamp(min=1e-4).reciprocal()
            target_disparity = source_depth[:, 0:1].clamp(min=1e-4).reciprocal()
            depth = F.l1_loss(predicted_disparity, target_disparity)

        depth_tv = self._depth_tv(aligned_depth)

        scale_reg = color.new_tensor(0.0)
        if alignment_map is not None:
            scale_reg = torch.abs(alignment_map - 1.0).mean()

        total = (
            self.weights.color * color
            + self.weights.alpha * alpha
            + self.weights.perceptual * perceptual
            + self.weights.depth * depth
            + self.weights.depth_tv * depth_tv
            + self.weights.scale_reg * scale_reg
        )
        return FineTuneLossOutputs(
            total=total,
            color=color,
            alpha=alpha,
            perceptual=perceptual,
            depth=depth,
            depth_tv=depth_tv,
            scale_reg=scale_reg,
        )
