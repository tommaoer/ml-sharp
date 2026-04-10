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
from sharp.utils.gaussians import Gaussians3D


@dataclasses.dataclass
class FineTuneLossWeights:
    """Weights for the fine-tuning losses."""

    color: float = 1.0
    alpha: float = 0.05
    perceptual: float = 0.1
    depth: float = 0.0
    depth_tv: float = 0.0
    grad: float = 0.0
    delta: float = 0.0
    splat: float = 0.0
    scale: float = 0.0
    scale_tv: float = 0.0
    keep: float = 0.0
    target_global: float = 0.0


class FineTuneLossOutputs(NamedTuple):
    """Structured loss output."""

    total: torch.Tensor
    color: torch.Tensor
    alpha: torch.Tensor
    perceptual: torch.Tensor
    depth: torch.Tensor
    depth_tv: torch.Tensor
    grad: torch.Tensor
    delta: torch.Tensor
    splat: torch.Tensor
    scale: torch.Tensor
    scale_tv: torch.Tensor
    keep: torch.Tensor
    target_global: torch.Tensor


class VGGPerceptualLoss(nn.Module):
    """VGG perceptual loss with feature and Gram terms on the novel view."""

    def __init__(self) -> None:
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

    @staticmethod
    def _gram_matrix(x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        features = x.view(batch, channels, height * width)
        gram = features @ features.transpose(-1, -2)
        return gram / max(channels * height * width, 1)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = (prediction - self.mean) / self.std
        y = (target - self.mean) / self.std
        total = x.new_tensor(0.0)
        for block in self.blocks:
            x = block(x)
            y = block(y)
            total = total + F.mse_loss(x, y)
            total = total + F.mse_loss(self._gram_matrix(x), self._gram_matrix(y))
        return total


class FineTuneLoss(nn.Module):
    """Training losses aligned with the SHARP paper objectives."""

    def __init__(
        self,
        weights: FineTuneLossWeights | None = None,
        use_perceptual: bool = True,
        delta_xy_scale: float = 0.001,
        delta_threshold_px: float = 400.0,
        splat_sigma_min: float = 1e-1,
        splat_sigma_max: float = 1e2,
        grad_sigma: float = 1e-2,
        grad_eps: float = 1e-2,
    ) -> None:
        super().__init__()
        self.weights = weights or FineTuneLossWeights()
        self.perceptual = VGGPerceptualLoss() if use_perceptual else None
        self.delta_xy_scale = delta_xy_scale
        self.delta_threshold_px = delta_threshold_px
        self.splat_sigma_min = splat_sigma_min
        self.splat_sigma_max = splat_sigma_max
        self.grad_sigma = grad_sigma
        self.grad_eps = grad_eps

    @staticmethod
    def _enabled(weight: float) -> bool:
        return weight > 0.0

    @staticmethod
    def _inverse_depth(depth: torch.Tensor) -> torch.Tensor:
        return depth.clamp(min=1e-4).reciprocal()

    @staticmethod
    def _depth_tv(depth_layers: torch.Tensor) -> torch.Tensor:
        if depth_layers.shape[1] < 2:
            return depth_layers.new_tensor(0.0)
        inverse_depth = depth_layers[:, 1:2].clamp(min=1e-4).reciprocal()
        grad_x = torch.abs(inverse_depth[..., :, 1:] - inverse_depth[..., :, :-1]).mean()
        grad_y = torch.abs(inverse_depth[..., 1:, :] - inverse_depth[..., :-1, :]).mean()
        return grad_x + grad_y

    @staticmethod
    def _multiscale_tv(scale_map: torch.Tensor, levels: int = 6) -> torch.Tensor:
        total = scale_map.new_tensor(0.0)
        current = scale_map
        for _ in range(levels):
            if min(current.shape[-2:]) < 2:
                break
            total = total + torch.abs(current[..., :, 1:] - current[..., :, :-1]).mean()
            total = total + torch.abs(current[..., 1:, :] - current[..., :-1, :]).mean()
            current = F.avg_pool2d(current, kernel_size=2, stride=2, ceil_mode=True)
        return total

    def _grad_regularizer(
        self,
        aligned_depth: torch.Tensor,
        gaussians_ndc: Gaussians3D,
        delta_values: torch.Tensor,
    ) -> torch.Tensor:
        inverse_depth = self._inverse_depth(aligned_depth[:, 0:1])
        grad_x = F.pad(torch.abs(inverse_depth[..., :, 1:] - inverse_depth[..., :, :-1]), (0, 1, 0, 0))
        grad_y = F.pad(torch.abs(inverse_depth[..., 1:, :] - inverse_depth[..., :-1, :]), (0, 0, 0, 1))
        grad_mag = grad_x + grad_y
        pooled_grad = F.interpolate(
            grad_mag,
            size=delta_values.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )[:, 0]
        opacity_grid = gaussians_ndc.opacities.view(
            gaussians_ndc.opacities.shape[0],
            delta_values.shape[2],
            delta_values.shape[-2],
            delta_values.shape[-1],
        )[:, 0]
        penalty = 1.0 - torch.exp(
            -(torch.relu(pooled_grad - self.grad_eps) / max(self.grad_sigma, 1e-8))
        )
        return (opacity_grid * penalty).mean()

    def _delta_regularizer(self, delta_values: torch.Tensor, image_width: int) -> torch.Tensor:
        delta_xy = delta_values[:, 0:2] * (self.delta_xy_scale * image_width)
        penalty = torch.relu(delta_xy.abs() - self.delta_threshold_px)
        return penalty.mean()

    def _splat_regularizer(
        self,
        gaussians_world: Gaussians3D,
        source_intrinsics: torch.Tensor,
        delta_values: torch.Tensor,
    ) -> torch.Tensor:
        grid_h = delta_values.shape[-2]
        grid_w = delta_values.shape[-1]
        first_layer_count = grid_h * grid_w
        means = gaussians_world.mean_vectors[:, :first_layer_count]
        scales = gaussians_world.singular_values[:, :first_layer_count]
        fx = source_intrinsics[:, 0, 0][:, None]
        fy = source_intrinsics[:, 1, 1][:, None]
        depth = means[..., 2].clamp(min=1e-4)
        sigma_x = fx * scales[..., 0] / depth
        sigma_y = fy * scales[..., 1] / depth
        penalty = (
            torch.relu(sigma_x - self.splat_sigma_max)
            + torch.relu(self.splat_sigma_min - sigma_x)
            + torch.relu(sigma_y - self.splat_sigma_max)
            + torch.relu(self.splat_sigma_min - sigma_y)
        )
        return penalty.mean()

    def forward(
        self,
        source_render: RenderingOutputs,
        target_render: RenderingOutputs,
        batch: dict[str, torch.Tensor | None],
        aligned_depth: torch.Tensor,
        delta_values: torch.Tensor,
        gaussians_ndc: Gaussians3D,
        gaussians_world: Gaussians3D,
        depth_alignment_map: torch.Tensor,
        loss_region_mask: torch.Tensor | None = None,
    ) -> FineTuneLossOutputs:
        source_image = batch["source_image"]
        target_image = batch["target_image"]
        source_depth = batch.get("source_depth")
        source_intrinsics = batch["source_intrinsics"]

        assert isinstance(source_image, torch.Tensor)
        assert isinstance(target_image, torch.Tensor)
        assert isinstance(source_intrinsics, torch.Tensor)

        zero = source_image.new_tensor(0.0)
        mask = loss_region_mask
        if mask is None:
            mask = torch.ones_like(source_image[:, 0:1])
        mask = mask.to(dtype=source_image.dtype)
        masked_target_render = target_render.color * mask
        masked_target_image = target_image * mask
        outside_mask = 1.0 - mask

        color = zero
        if self._enabled(self.weights.color):
            color = F.l1_loss(masked_target_render, masked_target_image)

        alpha = zero
        if self._enabled(self.weights.alpha):
            alpha = F.binary_cross_entropy(target_render.alpha * mask, mask)

        perceptual = zero
        if self._enabled(self.weights.perceptual) and self.perceptual is not None:
            perceptual = self.perceptual(masked_target_render, masked_target_image)

        keep = zero
        if self._enabled(self.weights.keep):
            keep = F.l1_loss(target_render.color * outside_mask, source_render.color * outside_mask)

        target_global = zero
        if self._enabled(self.weights.target_global):
            target_global = F.l1_loss(target_render.color, target_image)

        depth = zero
        if self._enabled(self.weights.depth) and isinstance(source_depth, torch.Tensor):
            predicted_disparity = self._inverse_depth(aligned_depth[:, 0:1])
            target_disparity = self._inverse_depth(source_depth[:, 0:1])
            depth = F.l1_loss(predicted_disparity, target_disparity)

        depth_tv = zero
        if self._enabled(self.weights.depth_tv):
            depth_tv = self._depth_tv(aligned_depth)

        grad = zero
        if self._enabled(self.weights.grad):
            grad = self._grad_regularizer(aligned_depth, gaussians_ndc, delta_values)

        delta = zero
        if self._enabled(self.weights.delta):
            delta = self._delta_regularizer(delta_values, source_image.shape[-1])

        splat = zero
        if self._enabled(self.weights.splat):
            splat = self._splat_regularizer(gaussians_world, source_intrinsics, delta_values)

        scale = zero
        if self._enabled(self.weights.scale) and isinstance(source_depth, torch.Tensor):
            scale = torch.abs(depth_alignment_map - 1.0).mean()

        scale_tv = zero
        if self._enabled(self.weights.scale_tv) and isinstance(source_depth, torch.Tensor):
            scale_tv = self._multiscale_tv(depth_alignment_map)

        total = (
            self.weights.color * color
            + self.weights.alpha * alpha
            + self.weights.perceptual * perceptual
            + self.weights.depth * depth
            + self.weights.depth_tv * depth_tv
            + self.weights.grad * grad
            + self.weights.delta * delta
            + self.weights.splat * splat
            + self.weights.scale * scale
            + self.weights.scale_tv * scale_tv
            + self.weights.keep * keep
            + self.weights.target_global * target_global
        )
        return FineTuneLossOutputs(
            total=total,
            color=color,
            alpha=alpha,
            perceptual=perceptual,
            depth=depth,
            depth_tv=depth_tv,
            grad=grad,
            delta=delta,
            splat=splat,
            scale=scale,
            scale_tv=scale_tv,
            keep=keep,
            target_global=target_global,
        )
