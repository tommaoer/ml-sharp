"""Fine-tuning pipeline for SHARP using video + camera trajectories.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sharp.models.decoders import UNetDecoder
from sharp.models.encoders import UNetEncoder
from sharp.utils import io
from sharp.utils.gaussians import Gaussians3D, unproject_gaussians
from sharp.utils.gsplat import GSplatRenderer
from torch import nn
from torch.utils.data import DataLoader

from .dataset import VideoCameraFineTuneDataset
from .losses import FineTuneLossWeights, VGGPerceptualLoss, total_variation_loss

LOGGER = logging.getLogger(__name__)


@dataclass
class FineTuneConfig:
    """Configuration for SHARP fine-tuning."""

    dataset_root: Path
    output_dir: Path
    checkpoint_path: Path | None = None
    batch_size: int = 1
    epochs: int = 1
    steps_per_epoch: int = 1000
    lr: float = 1e-5
    vis_interval: int = 100
    train_gaussian_decoder: bool = True
    enable_depth_loss: bool = False
    device: str = "cuda"
    min_view_distance: float = 0.05
    max_view_distance: float = 2.0
    disable_updates: bool = False


class OcclusionGaussianRefiner(nn.Module):
    """Predicts Gaussian deltas for occluded target regions using UNet blocks from SHARP."""

    def __init__(self, width: int = 32, steps: int = 4, num_layers: int = 2):
        """Initialize an occlusion refiner network."""
        super().__init__()
        self.steps = steps
        widths = [width << i for i in range(steps + 1)]
        self.encoder = UNetEncoder(dim_in=7, width=widths, steps=steps, norm_num_groups=4)
        self.decoder = UNetDecoder(dim_out=widths[0], width=widths, steps=steps, norm_num_groups=4)
        self.head = nn.Conv2d(widths[0], 14 * num_layers, kernel_size=1)
        self.num_layers = num_layers

    def forward(
        self,
        src_render: torch.Tensor,
        tgt_render: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict per-Gaussian attribute deltas for the occlusion copy."""
        x = torch.cat([src_render, tgt_render, mask], dim=1)
        # UNetEncoder uses stride-2 pooling at each level. For odd resolutions
        # (e.g. 1080p), skip-connections may mismatch by 1 pixel after repeated
        # down/up-sampling. We pad to a multiple of 2**steps and crop back.
        stride = 1 << self.steps
        h, w = x.shape[-2:]
        pad_h = (stride - (h % stride)) % stride
        pad_w = (stride - (w % stride)) % stride
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        features = self.encoder(x)
        out = self.head(self.decoder(features))
        if pad_h > 0 or pad_w > 0:
            out = out[..., :h, :w]

        b, _, h, w = out.shape
        return out.view(b, 14, self.num_layers, h, w)


def _to_image_u8(x: torch.Tensor) -> np.ndarray:
    x = x.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    return (x * 255.0).astype(np.uint8)


def _make_intrinsics_resized(
    intr: torch.Tensor,
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
):
    intr_resized = intr.clone()
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    intr_resized[0] *= dst_w / src_w
    intr_resized[1] *= dst_h / src_h
    return intr_resized


def _compute_gaussian_visibility(
    gaussians: Gaussians3D,
    w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    means = gaussians.mean_vectors
    b, n, _ = means.shape
    ones = torch.ones(b, n, 1, device=means.device, dtype=means.dtype)
    points_h = torch.cat([means, ones], dim=-1)

    cam = points_h @ w2c.transpose(-1, -2)
    z = cam[..., 2].clamp_min(1e-6)
    uv_h = cam @ intrinsics.transpose(-1, -2)
    u = uv_h[..., 0] / z
    v = uv_h[..., 1] / z
    return (z > 1e-3) & (u >= 0) & (u < width) & (v >= 0) & (v < height)


def _apply_gaussian_delta(
    base: Gaussians3D,
    delta: torch.Tensor,
    delta_scale: float = 0.01,
) -> Gaussians3D:
    b, n, _ = base.mean_vectors.shape
    delta_flat = delta.permute(0, 2, 3, 4, 1).reshape(b, n, 14)
    return Gaussians3D(
        mean_vectors=base.mean_vectors + delta_scale * delta_flat[..., 0:3],
        singular_values=(base.singular_values + delta_scale * delta_flat[..., 3:6]).clamp_min(1e-4),
        quaternions=base.quaternions + delta_scale * delta_flat[..., 6:10],
        colors=(base.colors + delta_scale * delta_flat[..., 10:13]).clamp(0.0, 1.0),
        opacities=(base.opacities + delta_scale * delta_flat[..., 13]).clamp(0.0, 1.0),
    )


def _concat_gaussians(a: Gaussians3D, b: Gaussians3D) -> Gaussians3D:
    return Gaussians3D(
        mean_vectors=torch.cat([a.mean_vectors, b.mean_vectors], dim=1),
        singular_values=torch.cat([a.singular_values, b.singular_values], dim=1),
        quaternions=torch.cat([a.quaternions, b.quaternions], dim=1),
        colors=torch.cat([a.colors, b.colors], dim=1),
        opacities=torch.cat([a.opacities, b.opacities], dim=1),
    )


def _mask_gaussians(gaussians: Gaussians3D, visible_mask: torch.Tensor) -> Gaussians3D:
    """Mask Gaussians by visibility in source view."""
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors,
        singular_values=gaussians.singular_values,
        quaternions=gaussians.quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities * visible_mask.float(),
    )


def save_debug_visualization(
    output_dir: Path,
    epoch: int,
    step: int,
    src_image: torch.Tensor,
    src_render: torch.Tensor,
    tgt_image: torch.Tensor,
    tgt_render: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Save the requested training visualizations for one training step."""
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"e{epoch:03d}_s{step:06d}"

    io.save_image(_to_image_u8(src_image[0]), vis_dir / f"{prefix}_src.png")
    io.save_image(_to_image_u8(src_render[0]), vis_dir / f"{prefix}_src_render.png")
    io.save_image(_to_image_u8(tgt_image[0]), vis_dir / f"{prefix}_tgt.png")
    io.save_image(_to_image_u8(tgt_render[0]), vis_dir / f"{prefix}_tgt_render.png")
    io.save_image(_to_image_u8(mask[0].repeat(3, 1, 1)), vis_dir / f"{prefix}_mask.png")


def run_finetuning(config: FineTuneConfig, predictor: nn.Module, num_layers: int = 2) -> None:
    """Run fine-tuning with source-target pair rendering supervision."""
    device = torch.device(config.device)
    predictor = predictor.to(device)
    predictor.train()

    if not config.train_gaussian_decoder or config.disable_updates:
        for p in predictor.parameters():
            p.requires_grad_(False)

    occlusion_refiner = OcclusionGaussianRefiner(num_layers=num_layers).to(device)
    renderer = GSplatRenderer(color_space="linearRGB", background_color="black").to(device)

    if config.disable_updates:
        LOGGER.info(
            "disable_updates=True: running forward/render only, "
            "without any parameter updates."
        )
        occlusion_refiner.requires_grad_(False)
        trainable: list[torch.Tensor] = []
        optimizer = None
    else:
        trainable = [p for p in predictor.parameters() if p.requires_grad]
        trainable += list(occlusion_refiner.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=config.lr)

    loss_weights = FineTuneLossWeights()
    perceptual = VGGPerceptualLoss().to(device)

    dataset = VideoCameraFineTuneDataset(
        dataset_root=config.dataset_root,
        min_view_distance=config.min_view_distance,
        max_view_distance=config.max_view_distance,
        max_samples=config.epochs * config.steps_per_epoch,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=0)

    internal_size = (predictor.internal_resolution(), predictor.internal_resolution())
    output_res = predictor.output_resolution

    global_step = 0
    for epoch in range(config.epochs):
        for batch in loader:
            if global_step >= (epoch + 1) * config.steps_per_epoch:
                break

            src_image = batch["src_image"].to(device)
            tgt_image = batch["tgt_image"].to(device)
            src_w2c = batch["src_w2c"].to(device)
            tgt_w2c = batch["tgt_w2c"].to(device)
            src_intr = batch["src_intrinsics"].to(device)

            b, _, h, w = src_image.shape
            src_resized = F.interpolate(
                src_image,
                size=internal_size[::-1],
                mode="bilinear",
                align_corners=True,
            )
            disparity_factor = (src_intr[:, 0, 0] / float(w)).float()

            intr_src_internal = torch.stack(
                [_make_intrinsics_resized(src_intr[i], (w, h), internal_size) for i in range(b)],
                dim=0,
            )
            # Render and losses are computed at original resolution.
            intr_src_render = src_intr.clone()
            intr_tgt_render = src_intr.clone()
            identity_w2c = torch.eye(4, device=device, dtype=src_w2c.dtype)[None].repeat(b, 1, 1)
            # SHARP predicts Gaussians in the source camera frame. Since the source
            # extrinsics are implicit in SHARP (identity), we convert target camera
            # pose to a relative transform from source->target.
            rel_tgt_w2c = tgt_w2c @ torch.linalg.inv(src_w2c)

            gaussians_ndc = predictor(src_resized, disparity_factor)

            gaussians_world = unproject_gaussians(
                gaussians_ndc,
                identity_w2c,
                intr_src_internal,
                internal_size,
            )

            render_src = renderer(
                gaussians_world,
                identity_w2c,
                intr_src_render,
                image_width=w,
                image_height=h,
            )
            render_tgt = renderer(
                gaussians_world,
                rel_tgt_w2c,
                intr_tgt_render,
                image_width=w,
                image_height=h,
            )

            vis_src = _compute_gaussian_visibility(
                gaussians_world,
                identity_w2c,
                intr_src_render,
                w,
                h,
            )
            vis_tgt = _compute_gaussian_visibility(
                gaussians_world,
                rel_tgt_w2c,
                intr_tgt_render,
                w,
                h,
            )
            invisible_tgt = vis_tgt & (~vis_src)

            # Disoccluded region: invisible in source view but visible in target view.
            invisible_target_gaussians = _mask_gaussians(gaussians_world, invisible_tgt)
            render_tgt_invisible = renderer(
                invisible_target_gaussians,
                rel_tgt_w2c,
                intr_tgt_render,
                image_width=w,
                image_height=h,
            )
            mask = (render_tgt_invisible.alpha > 1e-3).float()

            # Gaussian deltas live on predictor output grid (output_res x output_res),
            # so we run the occlusion refiner on that grid as well.
            src_render_for_refiner = F.interpolate(
                render_src.color.detach(),
                size=(output_res, output_res),
                mode="bilinear",
                align_corners=False,
            )
            tgt_render_for_refiner = F.interpolate(
                render_tgt.color.detach(),
                size=(output_res, output_res),
                mode="bilinear",
                align_corners=False,
            )
            mask_for_refiner = F.interpolate(mask, size=(output_res, output_res), mode="nearest")
            delta_map = occlusion_refiner(
                src_render_for_refiner,
                tgt_render_for_refiner,
                mask_for_refiner,
            )

            if config.disable_updates:
                delta_mask = torch.zeros_like(delta_map[:, :1])
                masked_delta = torch.zeros_like(delta_map)
                gaussians_aug = gaussians_world
            else:
                delta_mask = (
                    invisible_tgt.float().view(b, num_layers, output_res, output_res)[:, None]
                )
                masked_delta = delta_map * delta_mask
                occluded_copy = _apply_gaussian_delta(gaussians_world, masked_delta)
                occluded_copy = Gaussians3D(
                    mean_vectors=occluded_copy.mean_vectors,
                    singular_values=occluded_copy.singular_values,
                    quaternions=occluded_copy.quaternions,
                    colors=occluded_copy.colors,
                    opacities=occluded_copy.opacities * invisible_tgt.float(),
                )
                gaussians_aug = _concat_gaussians(gaussians_world, occluded_copy)
            render_tgt_aug = renderer(
                gaussians_aug,
                rel_tgt_w2c,
                intr_tgt_render,
                image_width=w,
                image_height=h,
            )

            loss_color = F.l1_loss(render_src.color, src_image) + F.l1_loss(
                render_tgt_aug.color,
                tgt_image,
            )
            loss_percep = perceptual(render_tgt_aug.color, tgt_image)
            loss_alpha = F.binary_cross_entropy(
                render_src.alpha.clamp(1e-5, 1 - 1e-5),
                torch.ones_like(render_src.alpha),
            )
            loss_tv = total_variation_loss(render_tgt_aug.depth)
            loss_occ_color = (
                (torch.abs(render_tgt_aug.color - tgt_image) * mask).sum()
                / mask.sum().clamp_min(1.0)
            )
            loss_occ_delta = masked_delta.abs().mean()

            loss = (
                loss_weights.color * loss_color
                + loss_weights.percep * loss_percep
                + loss_weights.alpha * loss_alpha
                + loss_weights.tv * loss_tv
                + loss_weights.occlusion_color * loss_occ_color
                + loss_weights.occlusion_delta * loss_occ_delta
            )

            if config.enable_depth_loss:
                LOGGER.warning(
                    "Depth loss requested, but depth GT is not wired in this training "
                    "dataset yet."
                )

            if global_step == 0 or global_step % config.vis_interval == 0:
                save_debug_visualization(
                    config.output_dir,
                    epoch,
                    global_step,
                    src_image,
                    render_src.color,
                    tgt_image,
                    render_tgt_aug.color,
                    mask,
                )

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            if global_step % 10 == 0:
                LOGGER.info(
                    "epoch=%d step=%d loss=%.4f color=%.4f percep=%.4f occ=%.4f",
                    epoch,
                    global_step,
                    loss.item(),
                    loss_color.item(),
                    loss_percep.item(),
                    loss_occ_color.item(),
                )

            global_step += 1

        ckpt_dir = config.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "predictor": predictor.state_dict(),
                "occlusion_refiner": occlusion_refiner.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            },
            ckpt_dir / f"epoch_{epoch:03d}.pt",
        )
