"""Contains `sharp render-invisible-mask` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import imageio.v2 as iio
import numpy as np
import torch
import torch.nn.functional as F

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils import logging as logging_utils
from sharp.utils.camera import create_camera_matrix
from sharp.utils.gaussians import Gaussians3D
from sharp.utils.gsplat import GSplatRenderer

from .predict import DEFAULT_MODEL_URL, predict_image

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input-image",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to a single input image for SHARP inference.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Directory for mask frames and optional video.",
)
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Path to SHARP checkpoint. Defaults to released checkpoint.",
)
@click.option("--fps", type=float, default=30.0, show_default=True)
@click.option("--alpha-threshold", type=float, default=0.01, show_default=True)
@click.option("--morph-radius", type=int, default=2, show_default=True)
@click.option("--num-views", type=int, default=81, show_default=True)
@click.option("--max-yaw-deg", type=float, default=50.0, show_default=True)
@click.option("--save-video/--no-save-video", default=True, show_default=True)
@click.option("--device", type=str, default="default", help="cuda / cpu / mps / default")
@click.option("-v", "--verbose", is_flag=True)
def render_invisible_mask_cli(
    input_image: Path,
    output_dir: Path,
    checkpoint_path: Path | None,
    fps: float,
    alpha_threshold: float,
    morph_radius: int,
    num_views: int,
    max_yaw_deg: float,
    save_video: bool,
    device: str,
    verbose: bool,
) -> None:
    """Predict gaussians from one image and render invisible-region masks for target cameras."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)
    device_t = resolve_device(device)
    if device_t.type != "cuda":
        raise RuntimeError("Rendering with gsplat requires CUDA.")

    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    state_dict = load_weights(checkpoint_path)
    predictor = create_predictor(PredictorParams()).to(device_t)
    predictor.load_state_dict(state_dict)
    predictor.eval()

    image, _, f_px = io.load_rgb(input_image)
    image_h, image_w = image.shape[:2]
    gaussians = predict_image(predictor, image, f_px, device_t)
    intrinsics = build_intrinsics_from_image(f_px, image_w, image_h, device_t)
    extrinsics = create_orbit_extrinsics(
        gaussians=gaussians,
        num_views=num_views,
        max_yaw_deg=max_yaw_deg,
        device=device_t,
    )

    renderer = GSplatRenderer(color_space="linearRGB", background_color="black").to(device_t)
    video_writer = None
    if save_video:
        video_writer = iio.get_writer(output_dir / "invisible_mask.mp4", fps=fps)

    gaussians = gaussians.to(device_t)
    for frame_index in range(extrinsics.shape[0]):
        render_out = renderer(
            gaussians=gaussians,
            extrinsics=extrinsics[frame_index : frame_index + 1],
            intrinsics=intrinsics[None],
            image_width=image_w,
            image_height=image_h,
        )
        visible = render_out.alpha[0:1, 0:1]
        invisible_mask = (visible <= alpha_threshold).float()
        invisible_mask = apply_morphology(invisible_mask, morph_radius)

        mask_np = (invisible_mask[0, 0] * 255.0).to(dtype=torch.uint8).detach().cpu().numpy()
        mask_rgb = np.repeat(mask_np[..., None], 3, axis=-1)
        iio.imwrite(masks_dir / f"{frame_index:06d}.png", mask_rgb)
        if video_writer is not None:
            video_writer.append_data(mask_rgb)

    if video_writer is not None:
        video_writer.close()


def resolve_device(device: str) -> torch.device:
    """Resolve runtime device."""
    if device == "default":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_weights(checkpoint_path: Path | None) -> dict[str, torch.Tensor]:
    """Load predictor checkpoint state dict."""
    if checkpoint_path is None:
        return torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "predictor" in checkpoint:
        return checkpoint["predictor"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def build_intrinsics_from_image(f_px: float, width: int, height: int, device: torch.device) -> torch.Tensor:
    """Build OpenCV-style 4x4 intrinsics from the input image camera."""
    fx = float(f_px)
    fy = float(f_px)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return torch.tensor(
        [
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def create_orbit_extrinsics(
    gaussians: Gaussians3D,
    num_views: int,
    max_yaw_deg: float,
    device: torch.device,
) -> torch.Tensor:
    """Create left-right orbit around scene centroid with vertical Y-axis."""
    if num_views < 2:
        raise ValueError("num_views must be >= 2.")
    means = gaussians.mean_vectors[0].to(device)
    opacities = gaussians.opacities[0].flatten().to(device)
    weights = opacities / opacities.sum().clamp(min=1e-6)
    center = (means * weights[:, None]).sum(dim=0)

    camera_origin = torch.zeros(3, dtype=torch.float32, device=device)
    rel = camera_origin - center
    xz_radius = torch.linalg.norm(rel[[0, 2]])
    if xz_radius < 1e-3:
        xz_std = means[:, [0, 2]].std(dim=0).mean().clamp(min=0.5)
        rel = torch.tensor([xz_std, rel[1], 0.0], device=device)

    yaw_values = torch.linspace(-max_yaw_deg, max_yaw_deg, num_views, device=device)
    extrinsics = []
    world_up = torch.tensor([0.0, -1.0, 0.0], device=device)
    for yaw_deg in yaw_values:
        yaw = torch.deg2rad(yaw_deg)
        cos_v = torch.cos(yaw)
        sin_v = torch.sin(yaw)
        rot_y = torch.tensor(
            [
                [cos_v, 0.0, sin_v],
                [0.0, 1.0, 0.0],
                [-sin_v, 0.0, cos_v],
            ],
            device=device,
            dtype=torch.float32,
        )
        eye = center + rot_y @ rel
        extrinsics.append(
            create_camera_matrix(
                position=eye,
                look_at_position=center,
                world_up=world_up,
                inverse=True,
            )
        )
    return torch.stack(extrinsics, dim=0)


def apply_morphology(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Apply light closing+opening on a BCHW binary mask."""
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    # Closing: dilate then erode.
    dilated = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)
    closed = 1.0 - F.max_pool2d(1.0 - dilated, kernel_size=kernel, stride=1, padding=radius)
    # Opening: erode then dilate.
    eroded = 1.0 - F.max_pool2d(1.0 - closed, kernel_size=kernel, stride=1, padding=radius)
    opened = F.max_pool2d(eroded, kernel_size=kernel, stride=1, padding=radius)
    return (opened > 0.5).float()
