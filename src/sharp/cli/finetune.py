"""Contains `sharp finetune` CLI implementation."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import torch

from sharp.models import PredictorParams, create_predictor
from sharp.training.finetune import FineTuneConfig, run_finetuning
from sharp.utils import logging as logging_utils

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


@click.command()
@click.option("--dataset-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--checkpoint-path", type=click.Path(path_type=Path, dir_okay=False), default=None)
@click.option("--batch-size", type=int, default=1)
@click.option("--epochs", type=int, default=1)
@click.option("--steps-per-epoch", type=int, default=1000)
@click.option("--lr", type=float, default=1e-5)
@click.option("--vis-interval", type=int, default=100)
@click.option("--train-gaussian-decoder/--freeze-gaussian-decoder", default=True)
@click.option(
    "--train-gaussian-delta-adaptor/--freeze-gaussian-delta-adaptor",
    default=True,
    help="Enable/disable training for the added Gaussian delta adaptor branch.",
)
@click.option("--enable-depth-loss/--disable-depth-loss", default=False)
@click.option("--device", type=str, default="cuda")
@click.option("--min-frame-gap", type=int, default=1)
@click.option("--max-frame-gap", type=int, default=30)
@click.option(
    "--mask-mode",
    type=click.Choice(["geometry", "rgb", "hybrid"], case_sensitive=False),
    default="geometry",
    show_default=True,
    help="Mask source: geometry reprojection, direct RGB difference, or hybrid union.",
)
@click.option(
    "--rgb-mask-threshold",
    type=float,
    default=0.12,
    show_default=True,
    help="Threshold for RGB-difference mask when mask-mode uses rgb/hybrid.",
)
@click.option(
    "--disable-updates/--enable-updates",
    default=False,
    help="Disable all parameter updates and keep output identical to base SHARP prediction.",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def finetune_cli(
    dataset_root: Path,
    output_dir: Path,
    checkpoint_path: Path | None,
    batch_size: int,
    epochs: int,
    steps_per_epoch: int,
    lr: float,
    vis_interval: int,
    train_gaussian_decoder: bool,
    train_gaussian_delta_adaptor: bool,
    enable_depth_loss: bool,
    device: str,
    min_frame_gap: int,
    max_frame_gap: int,
    mask_mode: str,
    rgb_mask_threshold: float,
    disable_updates: bool,
    verbose: bool,
):
    """Fine-tune SHARP with source/target frame supervision from videos."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_path is None:
        LOGGER.info("Downloading default checkpoint from %s", DEFAULT_MODEL_URL)
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, weights_only=True)

    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)

    cfg = FineTuneConfig(
        dataset_root=dataset_root,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        lr=lr,
        vis_interval=vis_interval,
        train_gaussian_decoder=train_gaussian_decoder,
        train_gaussian_delta_adaptor=train_gaussian_delta_adaptor,
        enable_depth_loss=enable_depth_loss,
        device=device,
        min_frame_gap=min_frame_gap,
        max_frame_gap=max_frame_gap,
        mask_mode=mask_mode.lower(),
        rgb_mask_threshold=rgb_mask_threshold,
        disable_updates=disable_updates,
    )
    run_finetuning(cfg, predictor)
