"""Contains `sharp finetune` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click
import torch
from torch.utils.data import DataLoader

from sharp.cli.predict import DEFAULT_MODEL_URL
from sharp.models import PredictorParams, create_predictor
from sharp.training.dataset import VideoFolderSceneDataset, collate_view_pairs
from sharp.training.losses import FineTuneLoss, FineTuneLossWeights
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import Gaussians3D, unproject_gaussians
from sharp.utils.gsplat import GSplatRenderer

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "--data-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Root folder that contains one scene per directory.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Directory to save checkpoints and logs.",
)
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to a pretrained SHARP checkpoint. Defaults to the released checkpoint.",
)
@click.option("--device", type=str, default="default", help="cpu / cuda / mps / default")
@click.option("--batch-size", type=int, default=1, show_default=True)
@click.option("--num-workers", type=int, default=2, show_default=True)
@click.option("--epochs", type=int, default=1, show_default=True)
@click.option("--lr", type=float, default=1e-5, show_default=True)
@click.option("--weight-decay", type=float, default=1e-4, show_default=True)
@click.option("--samples-per-scene", type=int, default=32, show_default=True)
@click.option("--max-frame-gap", type=int, default=24, show_default=True)
@click.option("--min-frame-distance", type=int, default=1, show_default=True)
@click.option("--log-every", type=int, default=10, show_default=True)
@click.option("--save-every", type=int, default=200, show_default=True)
@click.option("--max-steps", type=int, default=0, show_default=True)
@click.option("--preload-video/--stream-video", default=False, show_default=True)
@click.option("--perceptual/--no-perceptual", default=True, show_default=True)
@click.option("--depth-loss/--no-depth-loss", default=False, show_default=True)
@click.option("--color-weight", type=float, default=1.0, show_default=True)
@click.option("--alpha-weight", type=float, default=0.05, show_default=True)
@click.option("--perceptual-weight", type=float, default=0.1, show_default=True)
@click.option("--depth-tv-weight", type=float, default=0.01, show_default=True)
@click.option("--scale-reg-weight", type=float, default=0.0, show_default=True)
@click.option("--verbose", is_flag=True, default=False)
def finetune_cli(
    data_root: Path,
    output_dir: Path,
    checkpoint_path: Path | None,
    device: str,
    batch_size: int,
    num_workers: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    samples_per_scene: int,
    max_frame_gap: int,
    min_frame_distance: int,
    log_every: int,
    save_every: int,
    max_steps: int,
    preload_video: bool,
    perceptual: bool,
    depth_loss: bool,
    color_weight: float,
    alpha_weight: float,
    perceptual_weight: float,
    depth_tv_weight: float,
    scale_reg_weight: float,
    verbose: bool,
) -> None:
    """Fine-tune SHARP on folder-based video scenes with camera poses."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_t = resolve_device(device)
    if device_t.type != "cuda":
        raise RuntimeError(
            "Fine-tuning requires CUDA because differentiable gsplat rendering is used "
            "inside the training loop."
        )
    LOGGER.info("Using device %s", device_t)

    params = PredictorParams()
    params.monodepth.unfreeze_decoder = True
    params.monodepth.unfreeze_head = True
    params.monodepth.unfreeze_image_encoder = True
    params.depth_alignment.frozen = False

    model = create_predictor(params)
    state_dict = load_pretrained_weights(checkpoint_path)
    model.load_state_dict(state_dict)
    model.to(device_t)
    model.train()

    dataset = VideoFolderSceneDataset(
        data_root,
        internal_resolution=(1536, 1536),
        samples_per_scene=samples_per_scene,
        max_frame_gap=max_frame_gap,
        min_frame_distance=min_frame_distance,
        preload=preload_video,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_view_pairs,
        pin_memory=True,
        drop_last=False,
    )

    renderer = GSplatRenderer(
        color_space=params.color_space,
        background_color="black",
        low_pass_filter_eps=params.low_pass_filter_eps,
    ).to(device_t)
    loss_module = FineTuneLoss(
        weights=FineTuneLossWeights(
            color=color_weight,
            alpha=alpha_weight,
            perceptual=perceptual_weight,
            depth=1.0 if depth_loss else 0.0,
            depth_tv=depth_tv_weight,
            scale_reg=scale_reg_weight,
        ),
        use_perceptual=perceptual,
    ).to(device_t)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    write_config(
        output_dir / "finetune_config.json",
        {
            "data_root": str(data_root),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else DEFAULT_MODEL_URL,
            "device": str(device_t),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "samples_per_scene": samples_per_scene,
            "max_frame_gap": max_frame_gap,
            "min_frame_distance": min_frame_distance,
            "preload_video": preload_video,
            "perceptual": perceptual,
            "depth_loss": depth_loss,
        },
    )

    global_step = 0
    for epoch in range(epochs):
        for batch in loader:
            batch = move_batch_to_device(batch, device_t)
            optimizer.zero_grad(set_to_none=True)

            outputs = forward_training_pass(model, renderer, batch)
            losses = loss_module(
                source_render=outputs["source_render"],
                target_render=outputs["target_render"],
                batch=batch,
                aligned_depth=outputs["aligned_depth"],
                alignment_map=outputs["alignment_map"],
            )
            losses.total.backward()
            optimizer.step()

            global_step += 1
            if global_step % log_every == 0:
                LOGGER.info(
                    (
                        "epoch=%d step=%d total=%.4f color=%.4f alpha=%.4f "
                        "perceptual=%.4f depth=%.4f depth_tv=%.4f scale=%.4f"
                    ),
                    epoch,
                    global_step,
                    losses.total.item(),
                    losses.color.item(),
                    losses.alpha.item(),
                    losses.perceptual.item(),
                    losses.depth.item(),
                    losses.depth_tv.item(),
                    losses.scale_reg.item(),
                )

            if global_step % save_every == 0:
                save_checkpoint(
                    output_dir / f"step_{global_step:06d}.pt",
                    model,
                    optimizer,
                    global_step,
                )

            if max_steps > 0 and global_step >= max_steps:
                break
        if max_steps > 0 and global_step >= max_steps:
            break

    save_checkpoint(output_dir / "last.pt", model, optimizer, global_step)


def resolve_device(device: str) -> torch.device:
    """Resolve the requested device string."""
    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return torch.device(device)


def load_pretrained_weights(checkpoint_path: Path | None) -> dict[str, Any]:
    """Load pretrained predictor weights."""
    if checkpoint_path is None:
        LOGGER.info("Downloading pretrained checkpoint from %s", DEFAULT_MODEL_URL)
        return torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)

    LOGGER.info("Loading pretrained checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def write_config(path: Path, config: dict[str, Any]) -> None:
    """Persist the run configuration to disk."""
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def move_batch_to_device(
    batch: dict[str, torch.Tensor | None],
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    """Move tensors in a batch dict to the target device."""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def forward_training_pass(
    model,
    renderer: GSplatRenderer,
    batch: dict[str, torch.Tensor | None],
) -> dict[str, torch.Tensor | Gaussians3D | Any]:
    """Run the paper-style training forward pass."""
    source_image = batch["source_image"]
    disparity_factor = batch["disparity_factor"]
    source_depth = batch["source_depth"]
    source_intrinsics = batch["source_intrinsics"]
    source_extrinsics = batch["source_extrinsics"]
    target_intrinsics = batch["target_intrinsics"]
    target_extrinsics = batch["target_extrinsics"]

    assert isinstance(source_image, torch.Tensor)
    assert isinstance(disparity_factor, torch.Tensor)
    assert isinstance(source_intrinsics, torch.Tensor)
    assert isinstance(source_extrinsics, torch.Tensor)
    assert isinstance(target_intrinsics, torch.Tensor)
    assert isinstance(target_extrinsics, torch.Tensor)

    monodepth_output = model.monodepth_model(source_image)
    predicted_disparity = monodepth_output.disparity
    metric_depth = disparity_factor[:, :, None, None] / predicted_disparity.clamp(min=1e-4)

    aligned_depth, alignment_map = model.depth_alignment(
        metric_depth,
        source_depth,
        monodepth_output.decoder_features,
    )
    init_output = model.init_model(source_image, aligned_depth)
    image_features = model.feature_model(
        init_output.feature_input,
        encodings=monodepth_output.output_features,
    )
    delta_values = model.prediction_head(image_features)
    gaussians_ndc = model.gaussian_composer(
        delta=delta_values,
        base_values=init_output.gaussian_base_values,
        global_scale=init_output.global_scale,
    )

    image_shape = (source_image.shape[-1], source_image.shape[-2])
    gaussians_world = batch_unproject_gaussians(
        gaussians_ndc,
        source_extrinsics,
        source_intrinsics,
        image_shape,
    )
    source_render = renderer(
        gaussians_world,
        source_extrinsics,
        source_intrinsics,
        image_width=source_image.shape[-1],
        image_height=source_image.shape[-2],
    )
    target_render = renderer(
        gaussians_world,
        target_extrinsics,
        target_intrinsics,
        image_width=source_image.shape[-1],
        image_height=source_image.shape[-2],
    )

    return {
        "gaussians_ndc": gaussians_ndc,
        "gaussians_world": gaussians_world,
        "source_render": source_render,
        "target_render": target_render,
        "aligned_depth": aligned_depth,
        "alignment_map": alignment_map,
    }


def batch_unproject_gaussians(
    gaussians_ndc: Gaussians3D,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
) -> Gaussians3D:
    """Batch wrapper over `unproject_gaussians`."""
    items = []
    for index in range(gaussians_ndc.mean_vectors.shape[0]):
        items.append(
            unproject_gaussians(
                Gaussians3D(
                    mean_vectors=gaussians_ndc.mean_vectors[index : index + 1],
                    singular_values=gaussians_ndc.singular_values[index : index + 1],
                    quaternions=gaussians_ndc.quaternions[index : index + 1],
                    colors=gaussians_ndc.colors[index : index + 1],
                    opacities=gaussians_ndc.opacities[index : index + 1],
                ),
                extrinsics[index],
                intrinsics[index],
                image_shape,
            )
        )
    return Gaussians3D(
        mean_vectors=torch.cat([item.mean_vectors for item in items], dim=0),
        singular_values=torch.cat([item.singular_values for item in items], dim=0),
        quaternions=torch.cat([item.quaternions for item in items], dim=0),
        colors=torch.cat([item.colors for item in items], dim=0),
        opacities=torch.cat([item.opacities for item in items], dim=0),
    )


def save_checkpoint(
    path: Path,
    model,
    optimizer: torch.optim.Optimizer,
    global_step: int,
) -> None:
    """Save a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
        },
        path,
    )
    LOGGER.info("Saved checkpoint to %s", path)
