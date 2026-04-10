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
from sharp.utils.gaussians import SceneMetaData, save_ply, unproject_gaussians

from .finetune import LightweightDeltaDecoder
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
    "--use-delta/--ignore-delta",
    default=True,
    show_default=True,
    help="Whether to use delta_decoder weights from finetune checkpoints during inference.",
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
    use_delta: bool,
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

    state_dict, has_delta_decoder = load_predictor_state_dict(checkpoint_path)

    gaussian_predictor = create_predictor(PredictorParams())
    if has_delta_decoder and use_delta:
        add_delta_decoder_from_checkpoint(gaussian_predictor, state_dict)
    elif has_delta_decoder and not use_delta:
        state_dict = strip_delta_decoder_keys(state_dict)

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
        if has_delta_decoder and use_delta:
            gaussians = predict_image_with_delta(gaussian_predictor, image, f_px, torch.device(device))
        else:
            gaussians = predict_image(gaussian_predictor, image, f_px, torch.device(device))
        save_ply(gaussians, f_px, (height, width), output_path / f"{image_path.stem}.ply")

        if with_rendering:
            metadata = SceneMetaData(f_px, (width, height), "linearRGB")
            render_gaussians(gaussians, metadata, (output_path / image_path.stem).with_suffix(".mp4"))


def load_predictor_state_dict(checkpoint_path: Path | None) -> tuple[dict[str, Any], bool]:
    """Load predictor state dict from raw or wrapped checkpoint format."""
    if checkpoint_path is None:
        LOGGER.info("No checkpoint provided. Downloading default model from %s", DEFAULT_MODEL_URL)
        return torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True), False

    LOGGER.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "predictor" in checkpoint:
        state_dict = checkpoint["predictor"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(state_dict)!r}")

    normalized_state_dict = normalize_state_dict_keys(state_dict)
    has_delta_decoder = any(key.startswith("delta_decoder.") for key in normalized_state_dict)
    return normalized_state_dict, has_delta_decoder


def normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize checkpoint keys for loading."""
    return {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def strip_delta_decoder_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Drop delta_decoder keys (e.g. for baseline inference comparisons)."""
    cleaned: dict[str, Any] = {}
    dropped_keys: list[str] = []
    for key, value in state_dict.items():
        if key.startswith("delta_decoder."):
            dropped_keys.append(key)
            continue
        cleaned[key] = value

    if dropped_keys:
        LOGGER.info(
            "Ignoring %d delta_decoder keys (e.g. %s) before loading predictor weights.",
            len(dropped_keys),
            dropped_keys[0],
        )
    return cleaned


def add_delta_decoder_from_checkpoint(predictor, state_dict: dict[str, Any]) -> None:
    """Attach a delta decoder module so finetune checkpoints can be used during inference."""
    first_weight = state_dict.get("delta_decoder.net.0.weight")
    if first_weight is None:
        raise KeyError("Checkpoint indicates delta_decoder keys but delta_decoder.net.0.weight is missing.")
    hidden_dim = int(first_weight.shape[0])
    feature_dim = predictor.prediction_head.geometry_prediction_head.in_channels
    num_layers = predictor.prediction_head.num_layers
    predictor.delta_decoder = LightweightDeltaDecoder(
        feature_dim=feature_dim,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        geometry_scale=0.05,
        texture_scale=1.0,
    )
    LOGGER.info("Attached delta_decoder for inference (hidden_dim=%d).", hidden_dim)


def predict_image_with_delta(predictor, image, f_px: float, device: torch.device):
    """Predict gaussians and apply finetuned delta decoder."""
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width], dtype=torch.float32, device=device)
    internal_shape = (1536, 1536)
    image_resized_pt = torch.nn.functional.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    monodepth_output = predictor.monodepth_model(image_resized_pt)
    monodepth_disparity = monodepth_output.disparity
    monodepth = disparity_factor[:, None, None, None] / monodepth_disparity.clamp(min=1e-4, max=1e4)
    monodepth, _ = predictor.depth_alignment(monodepth, None, monodepth_output.decoder_features)
    init_output = predictor.init_model(image_resized_pt, monodepth)
    image_features = predictor.feature_model(
        init_output.feature_input, encodings=monodepth_output.output_features
    )
    delta_values = predictor.prediction_head(image_features)
    delta_values = delta_values + predictor.delta_decoder(image_features)
    gaussians_ndc = predictor.gaussian_composer(
        delta=delta_values,
        base_values=init_output.gaussian_base_values,
        global_scale=init_output.global_scale,
    )

    intrinsics = torch.tensor(
        [[f_px, 0, width / 2, 0], [0, f_px, height / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height
    return unproject_gaussians(gaussians_ndc, torch.eye(4, device=device), intrinsics_resized, internal_shape)
