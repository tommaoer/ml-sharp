"""Fine-tuning pipeline for SHARP using video + camera trajectories.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import copy
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
    min_frame_gap: int = 1
    max_frame_gap: int = 30
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


class GaussianDeltaAdaptor(nn.Module):
    """Gaussian-decoder-style delta predictor initialized from SHARP pretrained modules."""

    def __init__(self, feature_model: nn.Module, prediction_head: nn.Module) -> None:
        """Initialize by cloning SHARP pretrained decoder/head weights."""
        super().__init__()
        self.feature_model = copy.deepcopy(feature_model)
        self.prediction_head = copy.deepcopy(prediction_head)

    def forward(
        self,
        feature_input: torch.Tensor,
        encodings: list[torch.Tensor],
    ) -> torch.Tensor:
        """Predict Gaussian attribute deltas from SHARP decoder-style features."""
        features = self.feature_model(feature_input, encodings=encodings)
        return self.prediction_head(features)


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
    rendered_depth: torch.Tensor | None = None,
    depth_eps: float = 0.05,
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
    in_frustum = (z > 1e-3) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if rendered_depth is None:
        return in_frustum

    # Depth-aware visibility: Gaussian must lie close to rendered surface depth.
    width_denom = max(width - 1, 1)
    height_denom = max(height - 1, 1)
    grid_x = (u / width_denom * 2.0 - 1.0).clamp(-1.0, 1.0)
    grid_y = (v / height_denom * 2.0 - 1.0).clamp(-1.0, 1.0)
    sample_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)  # [B, N, 1, 2]
    sampled_depth = F.grid_sample(
        rendered_depth,
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(1).squeeze(-1)  # [B, N]
    return in_frustum & (z <= sampled_depth + depth_eps)


def _compute_pixel_disocclusion_mask(
    src_depth: torch.Tensor,
    src_alpha: torch.Tensor,
    tgt_alpha: torch.Tensor,
    rel_tgt_w2c: torch.Tensor,
    intr_src: torch.Tensor,
    intr_tgt: torch.Tensor,
    alpha_thr: float = 1e-3,
) -> torch.Tensor:
    """Compute disocclusion mask in target view using pixel reprojection only."""
    b, _, h, w = src_depth.shape
    device = src_depth.device
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=src_depth.dtype),
        torch.arange(w, device=device, dtype=src_depth.dtype),
        indexing="ij",
    )
    xx = xx[None].expand(b, -1, -1)
    yy = yy[None].expand(b, -1, -1)

    z = src_depth[:, 0]
    src_valid = (src_alpha[:, 0] > alpha_thr) & (z > 1e-6)

    fx = intr_src[:, 0, 0][:, None, None]
    fy = intr_src[:, 1, 1][:, None, None]
    cx = intr_src[:, 0, 2][:, None, None]
    cy = intr_src[:, 1, 2][:, None, None]

    x = (xx - cx) / fx * z
    y = (yy - cy) / fy * z
    ones = torch.ones_like(z)
    pts_src = torch.stack([x, y, z, ones], dim=-1)  # [B, H, W, 4]

    pts_tgt = pts_src @ rel_tgt_w2c.transpose(-1, -2)
    z_tgt = pts_tgt[..., 2].clamp_min(1e-6)

    fx_t = intr_tgt[:, 0, 0][:, None, None]
    fy_t = intr_tgt[:, 1, 1][:, None, None]
    cx_t = intr_tgt[:, 0, 2][:, None, None]
    cy_t = intr_tgt[:, 1, 2][:, None, None]
    u_t = (pts_tgt[..., 0] / z_tgt) * fx_t + cx_t
    v_t = (pts_tgt[..., 1] / z_tgt) * fy_t + cy_t

    reproj_visible = torch.zeros((b, 1, h, w), device=device, dtype=torch.bool)
    for ib in range(b):
        valid = src_valid[ib] & (z_tgt[ib] > 0)
        u = u_t[ib][valid].round().long()
        v = v_t[ib][valid].round().long()
        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        reproj_visible[ib, 0, v[in_img], u[in_img]] = True

    tgt_visible = tgt_alpha > alpha_thr
    return (tgt_visible & (~reproj_visible)).float()


def _morphological_smooth_mask(
    mask: torch.Tensor,
    open_kernel: int = 3,
    close_kernel: int = 9,
    speckle_kernel: int = 7,
    speckle_ratio: float = 0.08,
) -> torch.Tensor:
    """Apply morphology to remove thin lines and preserve large white regions."""
    out = mask

    if open_kernel > 1:
        pad = open_kernel // 2
        # Opening (erode -> dilate): removes thin structures.
        eroded = 1.0 - F.max_pool2d(1.0 - out, kernel_size=open_kernel, stride=1, padding=pad)
        out = F.max_pool2d(eroded, kernel_size=open_kernel, stride=1, padding=pad)

    if close_kernel > 1:
        pad = close_kernel // 2
        # Closing fills interior holes in large disocclusion regions.
        dilated = F.max_pool2d(out, kernel_size=close_kernel, stride=1, padding=pad)
        out = 1.0 - F.max_pool2d(1.0 - dilated, kernel_size=close_kernel, stride=1, padding=pad)

    if speckle_kernel > 1:
        pad = speckle_kernel // 2
        local_ratio = F.avg_pool2d(out, kernel_size=speckle_kernel, stride=1, padding=pad)
        out = out * (local_ratio > speckle_ratio).float()

    return out


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
    predictor.eval()
    # Keep SHARP base predictor frozen; only train the newly added delta branch.
    for p in predictor.parameters():
        p.requires_grad_(False)

    gaussian_delta_adaptor = GaussianDeltaAdaptor(
        predictor.feature_model,
        predictor.prediction_head,
    ).to(device)
    renderer = GSplatRenderer(color_space="linearRGB", background_color="black").to(device)

    if config.disable_updates:
        LOGGER.info(
            "disable_updates=True: running forward/render only, "
            "without any parameter updates."
        )
        gaussian_delta_adaptor.requires_grad_(False)
        trainable: list[torch.Tensor] = []
        optimizer = None
    else:
        trainable = list(gaussian_delta_adaptor.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=config.lr)
        num_trainable = sum(p.numel() for p in trainable if p.requires_grad)
        LOGGER.info("Trainable parameters (gaussian_delta_adaptor only): %d", num_trainable)

    loss_weights = FineTuneLossWeights()
    perceptual = VGGPerceptualLoss().to(device)

    dataset = VideoCameraFineTuneDataset(
        dataset_root=config.dataset_root,
        min_frame_gap=config.min_frame_gap,
        max_frame_gap=config.max_frame_gap,
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

            # Run SHARP pipeline explicitly so we can add a decoder-like delta branch
            # initialized from the pretrained Gaussian decoder/head.
            monodepth_output = predictor.monodepth_model(src_resized)
            monodepth_disparity = monodepth_output.disparity
            monodepth = (
                disparity_factor[:, None, None, None]
                / monodepth_disparity.clamp(min=1e-4, max=1e4)
            )
            monodepth, _ = predictor.depth_alignment(
                monodepth,
                None,
                monodepth_output.decoder_features,
            )

            init_output = predictor.init_model(src_resized, monodepth)
            gaussian_features = predictor.feature_model(
                init_output.feature_input,
                encodings=monodepth_output.output_features,
            )
            delta_base = predictor.prediction_head(gaussian_features)
            gaussians_ndc = predictor.gaussian_composer(
                delta=delta_base,
                base_values=init_output.gaussian_base_values,
                global_scale=init_output.global_scale,
            )

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

            # Pixel-based disocclusion mask (no Gaussian-level mask rendering).
            mask = _compute_pixel_disocclusion_mask(
                src_depth=render_src.depth,
                src_alpha=render_src.alpha,
                tgt_alpha=render_tgt.alpha,
                rel_tgt_w2c=rel_tgt_w2c,
                intr_src=intr_src_render,
                intr_tgt=intr_tgt_render,
            )
            mask = _morphological_smooth_mask(mask)
            # Keep only central valid region (remove image borders), then intersect.
            center_mask = torch.zeros_like(mask)
            margin_h = int(0.08 * h)
            margin_w = int(0.08 * w)
            center_mask[:, :, margin_h : h - margin_h, margin_w : w - margin_w] = 1.0
            mask = mask * center_mask

            # Decoder-style delta branch initialized from SHARP pretrained Gaussian decoder.
            mask_for_refiner = F.interpolate(mask, size=(output_res, output_res), mode="nearest")
            delta_map = gaussian_delta_adaptor(
                init_output.feature_input,
                monodepth_output.output_features,
            )

            if config.disable_updates:
                delta_mask = torch.zeros_like(delta_map[:, :1, ...])
                masked_delta = torch.zeros_like(delta_map)
                gaussians_aug = gaussians_world
            else:
                delta_mask = mask_for_refiner[:, :, None].repeat(
                    1,
                    1,
                    num_layers,
                    1,
                    1,
                )
                masked_delta = delta_map * delta_mask
                delta_total = delta_base + masked_delta
                gaussians_ndc_aug = predictor.gaussian_composer(
                    delta=delta_total,
                    base_values=init_output.gaussian_base_values,
                    global_scale=init_output.global_scale,
                )
                gaussians_aug = unproject_gaussians(
                    gaussians_ndc_aug,
                    identity_w2c,
                    intr_src_internal,
                    internal_size,
                )
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
                "gaussian_delta_adaptor": gaussian_delta_adaptor.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            },
            ckpt_dir / f"epoch_{epoch:03d}.pt",
        )
