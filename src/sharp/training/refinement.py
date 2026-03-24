"""Mask-guided Gaussian refinement modules.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import torch
from torch import nn

from sharp.models.decoders import UNetDecoder
from sharp.models.encoders import UNetEncoder


class MaskDeltaRefiner(nn.Module):
    """Predicts additive Gaussian delta corrections, optionally masked per region.

    The refiner only updates covariance-related and color channels. Position and
    opacity deltas are forced to zero.
    """

    def __init__(self, num_layers: int, width: list[int] | None = None, steps: int = 4) -> None:
        """Initialize the Gaussian delta refiner."""
        super().__init__()
        if width is None:
            width = [32, 64, 128, 256, 256]
        self.num_layers = num_layers
        self.encoder = UNetEncoder(dim_in=10, width=width, steps=steps, norm_num_groups=4)
        self.decoder = UNetDecoder(dim_out=width[0], width=width, steps=steps, norm_num_groups=4)
        self.conv_out = nn.Conv2d(width[0], 14 * num_layers, kernel_size=1, stride=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
        channel_update_mask = torch.zeros(1, 14, 1, 1, 1, dtype=torch.float32)
        # Keep covariance-related channels (scales and quaternions) and color channels:
        # 3:6 (scale), 6:10 (quaternion), 10:13 (color).
        channel_update_mask[:, 3:13] = 1.0
        self.register_buffer("channel_update_mask", channel_update_mask)

    def forward(
        self,
        source_image: torch.Tensor,
        target_render: torch.Tensor,
        masked_target_render: torch.Tensor,
        invisible_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict additive Gaussian deltas at decoder resolution."""
        features = torch.cat(
            [source_image, target_render, masked_target_render, invisible_mask],
            dim=1,
        )
        encoded = self.encoder(features)
        delta = self.conv_out(self.decoder(encoded))
        _, _, height, width = delta.shape
        delta = delta.unflatten(1, (14, self.num_layers))
        # `invisible_mask` is already BCHW here; keep it 4D for interpolate and only
        # expand to 5D afterwards to match [B, 14, L, H, W] delta tensors.
        mask = torch.nn.functional.interpolate(
            invisible_mask,
            size=(height, width),
            mode="nearest",
        )
        return delta * mask[:, None] * self.channel_update_mask


__all__ = ["MaskDeltaRefiner"]
