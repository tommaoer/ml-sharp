"""Contains `sharp predict-finetune` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click
import torch

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import SceneMetaData, save_ply

from .predict import DEFAULT_MODEL_URL, predict_image
from .render import render_gaussians

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Path to an image or a directory of images.",
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Path to save predicted Gaussians (.ply) and optional renderings.",
)
@click.option(
    "-c",
    "--checkpoint-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to checkpoint. Supports raw predictor state_dict or {predictor: ...} payload.",
)
@click.option(
    "--strict/--no-strict",
    default=False,
    show_default=True,
    help="Whether to require an exact state_dict key match when loading checkpoint.",
)
@click.option(
    "--render/--no-render",
    "with_rendering",
    is_flag=True,
    default=False,
    help="Whether to render trajectory for each predicted scene (CUDA only).",
)
@click.option(
    "--device",
    type=str,
    default="default",
    help="Device to run on. ['cpu', 'mps', 'cuda']",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def predict_finetune_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path | None,
    strict: bool,
    with_rendering: bool,
    device: str,
    verbose: bool,
):
    """Predict Gaussians with tolerant checkpoint loading for fine-tuned models."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    extensions = io.get_supported_image_extensions()
    image_paths = [input_path] if input_path.is_file() else list(input_path.glob("*"))
    image_paths = [path for path in image_paths if path.suffix in extensions]
    if not image_paths:
        LOGGER.warning("No valid image files found in %s", input_path)
        return

    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    LOGGER.info("Using device %s", device)

    if with_rendering and device != "cuda":
        LOGGER.warning("Can only run rendering with gsplat on CUDA. Rendering is disabled.")
        with_rendering = False

    state_dict = load_predictor_state_dict(checkpoint_path)

    gaussian_predictor = create_predictor(PredictorParams())
    incompatibility = gaussian_predictor.load_state_dict(state_dict, strict=strict)
    if incompatibility.missing_keys:
        LOGGER.warning("Missing keys when loading checkpoint: %d", len(incompatibility.missing_keys))
    if incompatibility.unexpected_keys:
        LOGGER.warning(
            "Unexpected keys when loading checkpoint: %d", len(incompatibility.unexpected_keys)
        )
    gaussian_predictor.eval()
    gaussian_predictor.to(device)

    output_path.mkdir(exist_ok=True, parents=True)
    for image_path in image_paths:
        LOGGER.info("Processing %s", image_path)
        image, _, f_px = io.load_rgb(image_path)
        height, width = image.shape[:2]
        gaussians = predict_image(gaussian_predictor, image, f_px, torch.device(device))
        save_ply(gaussians, f_px, (height, width), output_path / f"{image_path.stem}.ply")

        if with_rendering:
            metadata = SceneMetaData(f_px, (width, height), "linearRGB")
            render_gaussians(gaussians, metadata, (output_path / image_path.stem).with_suffix(".mp4"))


def load_predictor_state_dict(checkpoint_path: Path | None) -> dict[str, Any]:
    """Load predictor state dict from raw or wrapped checkpoint format."""
    if checkpoint_path is None:
        LOGGER.info("No checkpoint provided. Downloading default model from %s", DEFAULT_MODEL_URL)
        return torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)

    LOGGER.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "predictor" in checkpoint:
        return checkpoint["predictor"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint
