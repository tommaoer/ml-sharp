"""Contains `sharp render-invisible-mask` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import imageio.v2 as iio
import numpy as np
import torch

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils import logging as logging_utils
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
    "--camera-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to camera json with fl_x/fl_y/cx/cy/c2ws.",
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
@click.option("--save-video/--no-save-video", default=True, show_default=True)
@click.option("--device", type=str, default="default", help="cuda / cpu / mps / default")
@click.option("-v", "--verbose", is_flag=True)
def render_invisible_mask_cli(
    input_image: Path,
    camera_path: Path,
    output_dir: Path,
    checkpoint_path: Path | None,
    fps: float,
    alpha_threshold: float,
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

    camera_payload = json.loads(camera_path.read_text(encoding="utf-8"))
    intrinsics = build_intrinsics(camera_payload, image_w, image_h, device_t)
    c2ws = torch.tensor(camera_payload["c2ws"], dtype=torch.float32, device=device_t)
    if c2ws.ndim != 3 or c2ws.shape[-2:] != (4, 4):
        raise ValueError(f"Expected c2ws shape [T, 4, 4], got {tuple(c2ws.shape)}.")
    extrinsics = torch.linalg.inv(c2ws)

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
        visible = render_out.alpha[0, 0]
        invisible_mask = (visible <= alpha_threshold).to(dtype=torch.uint8) * 255
        mask_np = invisible_mask.detach().cpu().numpy()
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


def build_intrinsics(
    payload: dict[str, object], width: int, height: int, device: torch.device
) -> torch.Tensor:
    """Build OpenCV-style 4x4 intrinsics from json payload."""
    fx = float(payload["fl_x"])
    fy = float(payload.get("fl_y", payload["fl_x"]))
    cx = float(payload.get("cx", (width - 1) / 2.0))
    cy = float(payload.get("cy", (height - 1) / 2.0))
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

