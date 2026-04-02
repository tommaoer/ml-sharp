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
from sharp.utils import camera
from sharp.utils.gsplat import GSplatRenderer

from .finetune import compute_target_invisible_mask
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
@click.option("--morph-radius", type=int, default=2, show_default=True)
@click.option("--num-views", type=int, default=81, show_default=True)
@click.option("--save-video/--no-save-video", default=True, show_default=True)
@click.option("--device", type=str, default="default", help="cuda / cpu / mps / default")
@click.option("-v", "--verbose", is_flag=True)
def render_invisible_mask_cli(
    input_image: Path,
    output_dir: Path,
    checkpoint_path: Path | None,
    fps: float,
    morph_radius: int,
    num_views: int,
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
    intrinsics_cpu = build_intrinsics_from_image(f_px, image_w, image_h, torch.device("cpu"))
    gaussians_cpu = gaussians.to(torch.device("cpu"))
    camera_model = camera.create_camera_model(
        gaussians_cpu,
        intrinsics_cpu,
        resolution_px=(image_w, image_h),
    )
    trajectory_params = camera.TrajectoryParams(
        type="rotate_forward",
        num_steps=num_views,
        num_repeats=1,
    )
    trajectory = camera.create_eye_trajectory(
        gaussians_cpu,
        trajectory_params,
        resolution_px=(image_w, image_h),
        f_px=float(f_px),
    )

    renderer = GSplatRenderer(color_space="linearRGB", background_color="black").to(device_t)
    video_writer = None
    if save_video:
        video_writer = iio.get_writer(output_dir / "invisible_mask.mp4", fps=fps)

    gaussians = gaussians.to(device_t)
    source_intrinsics = build_intrinsics_from_image(f_px, image_w, image_h, device_t)[None]
    source_extrinsics = torch.eye(4, dtype=torch.float32, device=device_t)[None]
    source_render = renderer(
        gaussians=gaussians,
        extrinsics=source_extrinsics,
        intrinsics=source_intrinsics,
        image_width=image_w,
        image_height=image_h,
    )
    source_depth = source_render.depth[:, 0:1]

    rendered_dir = output_dir / "rendered_color"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    for frame_index, eye_position in enumerate(trajectory):
        camera_info = camera_model.compute(eye_position)
        target_extrinsics = camera_info.extrinsics[None].to(device_t)
        target_intrinsics = camera_info.intrinsics[None].to(device_t)
        render_out = renderer(
            gaussians=gaussians,
            extrinsics=target_extrinsics,
            intrinsics=target_intrinsics,
            image_width=camera_info.width,
            image_height=camera_info.height,
        )
        invisible_mask = compute_target_invisible_mask(
            source_depth=source_depth,
            source_intrinsics=source_intrinsics,
            source_extrinsics=source_extrinsics,
            target_intrinsics=target_intrinsics,
            target_extrinsics=target_extrinsics,
        )
        invisible_mask = apply_morphology(invisible_mask, morph_radius)

        mask_np = (invisible_mask[0, 0] * 255.0).to(dtype=torch.uint8).detach().cpu().numpy()
        mask_rgb = np.repeat(mask_np[..., None], 3, axis=-1)
        color_np = (render_out.color[0].permute(1, 2, 0) * 255.0).to(dtype=torch.uint8).cpu().numpy()
        iio.imwrite(masks_dir / f"{frame_index:06d}.png", mask_rgb)
        iio.imwrite(rendered_dir / f"{frame_index:06d}.png", color_np)
        if video_writer is not None:
            preview_frame = np.concatenate([color_np, mask_rgb], axis=1)
            video_writer.append_data(preview_frame)

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

def apply_morphology(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Remove isolated black holes and smooth black/white boundaries."""
    if radius <= 0:
        return (mask > 0.5).float()
    binary = (mask > 0.5).float()
    # 1) Remove isolated black points/holes inside white regions.
    black = 1.0 - binary
    black_support = F.avg_pool2d(black, kernel_size=3, stride=1, padding=1) * 9.0
    black = black * (black_support >= 4.0).float()
    filled = 1.0 - black

    # 2) Smooth boundary with local voting.
    smoothed = (F.avg_pool2d(filled, kernel_size=5, stride=1, padding=2) > 0.5).float()

    # 3) One close/open round to regularize contour.
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(smoothed, kernel_size=kernel, stride=1, padding=radius)
    closed = 1.0 - F.max_pool2d(1.0 - dilated, kernel_size=kernel, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - closed, kernel_size=kernel, stride=1, padding=radius)
    opened = F.max_pool2d(eroded, kernel_size=kernel, stride=1, padding=radius)
    return (opened > 0.5).float()
