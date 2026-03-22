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
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sharp.cli.predict import DEFAULT_MODEL_URL
from sharp.models import PredictorParams, create_predictor
from sharp.training.dataset import (
    MultiScenePosedVideoDataset,
    PosedVideoDataset,
    collate_view_pairs,
)
from sharp.training.losses import FineTuneLoss, FineTuneLossWeights
from sharp.training.refinement import MaskDeltaRefiner
from sharp.utils import io, vis
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import Gaussians3D, unproject_gaussians
from sharp.utils.gsplat import GSplatRenderer, RenderingOutputs

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "--data-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="Root path containing many scene folders, each with one mp4 and one json.",
)
@click.option(
    "--video-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Path to the training video.",
)
@click.option(
    "--pose-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Path to the JSON file containing fl_x/fl_y/cx/cy/c2ws.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Directory to save checkpoints, logs, and visualizations.",
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
@click.option("--samples-per-epoch", type=int, default=256, show_default=True)
@click.option("--min-frame-distance", type=int, default=4, show_default=True)
@click.option("--max-frame-distance", type=int, default=48, show_default=True)
@click.option("--log-every", type=int, default=10, show_default=True)
@click.option("--save-every", type=int, default=200, show_default=True)
@click.option("--visualize-every", type=int, default=50, show_default=True)
@click.option("--max-steps", type=int, default=0, show_default=True)
@click.option("--preload-video/--stream-video", default=False, show_default=True)
@click.option("--perceptual/--no-perceptual", default=True, show_default=True)
@click.option("--depth-loss/--no-depth-loss", default=False, show_default=True)
@click.option("--color-weight", type=float, default=1.0, show_default=True)
@click.option("--alpha-weight", type=float, default=0.05, show_default=True)
@click.option("--perceptual-weight", type=float, default=0.1, show_default=True)
@click.option("--depth-tv-weight", type=float, default=0.01, show_default=True)
@click.option("--low-pass-filter-eps", type=float, default=0.0, show_default=True)
@click.option("--verbose", is_flag=True, default=False)
def finetune_cli(
    data_root: Path | None,
    video_path: Path | None,
    pose_path: Path | None,
    output_dir: Path,
    checkpoint_path: Path | None,
    device: str,
    batch_size: int,
    num_workers: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    samples_per_epoch: int,
    min_frame_distance: int,
    max_frame_distance: int,
    log_every: int,
    save_every: int,
    visualize_every: int,
    max_steps: int,
    preload_video: bool,
    perceptual: bool,
    depth_loss: bool,
    color_weight: float,
    alpha_weight: float,
    perceptual_weight: float,
    depth_tv_weight: float,
    low_pass_filter_eps: float,
    verbose: bool,
) -> None:
    """Fine-tune SHARP on a posed video sequence."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)
    if data_root is None and (video_path is None or pose_path is None):
        raise click.UsageError("Please provide --data-root or both --video-path and --pose-path.")

    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    device_t = resolve_device(device)
    if device_t.type != "cuda":
        raise RuntimeError(
            "Fine-tuning requires CUDA because differentiable gsplat rendering is used "
            "inside the training loop."
        )
    LOGGER.info("Using device %s", device_t)

    predictor = build_finetune_predictor(checkpoint_path).to(device_t)
    refiner = MaskDeltaRefiner(num_layers=predictor.init_model.num_layers).to(device_t)
    if data_root is not None:
        dataset = MultiScenePosedVideoDataset(
            data_root=data_root,
            internal_resolution=(1536, 1536),
            min_frame_distance=min_frame_distance,
            max_frame_distance=max_frame_distance,
            samples_per_scene=samples_per_epoch,
            preload=preload_video,
        )
    else:
        assert video_path is not None and pose_path is not None
        dataset = PosedVideoDataset(
            video_path=video_path,
            pose_path=pose_path,
            internal_resolution=(1536, 1536),
            min_frame_distance=min_frame_distance,
            max_frame_distance=max_frame_distance,
            samples_per_epoch=samples_per_epoch,
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
        color_space="linearRGB",
        background_color="black",
        low_pass_filter_eps=low_pass_filter_eps,
    ).to(device_t)
    loss_module = FineTuneLoss(
        weights=FineTuneLossWeights(
            color=color_weight,
            alpha=alpha_weight,
            perceptual=perceptual_weight,
            depth=1.0 if depth_loss else 0.0,
            depth_tv=depth_tv_weight,
        ),
        use_perceptual=perceptual,
    ).to(device_t)

    trainable_module_names = ["feature_model", "refiner"]
    trainable_parameter_names = [
        f"predictor.{name}" for name, param in predictor.named_parameters() if param.requires_grad
    ] + [
        f"refiner.{name}" for name, _ in refiner.named_parameters()
    ]
    trainable_parameter_count = sum(
        param.numel() for _, param in predictor.named_parameters() if param.requires_grad
    ) + sum(param.numel() for param in refiner.parameters())
    LOGGER.info(
        "Optimizing modules: %s (%d parameters)",
        ", ".join(trainable_module_names),
        trainable_parameter_count,
    )
    trainable_parameters = [
        param for _, param in predictor.named_parameters() if param.requires_grad
    ] + list(refiner.parameters())
    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)

    write_config(
        output_dir / "finetune_config.json",
        {
            "data_root": str(data_root) if data_root is not None else None,
            "video_path": str(video_path) if video_path is not None else None,
            "pose_path": str(pose_path) if pose_path is not None else None,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else DEFAULT_MODEL_URL,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "samples_per_epoch": samples_per_epoch,
            "min_frame_distance": min_frame_distance,
            "max_frame_distance": max_frame_distance,
            "visualize_every": visualize_every,
            "low_pass_filter_eps": low_pass_filter_eps,
            "trainable_modules": trainable_module_names,
            "trainable_parameter_count": trainable_parameter_count,
            "trainable_parameter_names": trainable_parameter_names,
        },
    )

    global_step = 0
    for epoch in range(epochs):
        for batch in loader:
            batch = move_batch_to_device(batch, device_t)

            optimizer.zero_grad(set_to_none=True)
            outputs = forward_training_pass(predictor, renderer, batch, refiner)
            losses = loss_module(
                source_render=outputs["source_render"],
                target_render=outputs["refined_target_render"],
                batch=batch,
                aligned_depth=outputs["aligned_depth"],
            )

            if epoch == 0 and global_step == 0:
                LOGGER.info("Saving visualization at epoch=0 step=0 before optimization")
                save_visualization_batch(visualization_dir, global_step, batch, outputs)

            losses.total.backward()
            optimizer.step()

            global_step += 1
            if global_step % log_every == 0:
                LOGGER.info(
                    (
                        "epoch=%d step=%d total=%.4f color=%.4f "
                        "alpha=%.4f perceptual=%.4f depth=%.4f depth_tv=%.4f"
                    ),
                    epoch,
                    global_step,
                    losses.total.item(),
                    losses.color.item(),
                    losses.alpha.item(),
                    losses.perceptual.item(),
                    losses.depth.item(),
                    losses.depth_tv.item(),
                )

            if global_step % visualize_every == 0:
                save_visualization_batch(visualization_dir, global_step, batch, outputs)

            if global_step % save_every == 0:
                save_checkpoint(
                    output_dir / f"step_{global_step:06d}.pt",
                    predictor,
                    refiner,
                    optimizer,
                    global_step,
                )

            if max_steps > 0 and global_step >= max_steps:
                break
        if max_steps > 0 and global_step >= max_steps:
            break

    save_checkpoint(output_dir / "last.pt", predictor, refiner, optimizer, global_step)


def resolve_device(device: str) -> torch.device:
    """Resolve the requested device string."""
    if device == "default":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def build_finetune_predictor(checkpoint_path: Path | None):
    """Create SHARP predictor and unfreeze only the feature model."""
    params = PredictorParams()
    predictor = create_predictor(params)
    predictor.load_state_dict(load_pretrained_weights(checkpoint_path))
    predictor.requires_grad_(False)
    predictor.feature_model.requires_grad_(True)
    predictor.prediction_head.requires_grad_(False)
    predictor.train()
    predictor.monodepth_model.eval()
    predictor.init_model.eval()
    predictor.gaussian_composer.eval()
    predictor.depth_alignment.eval()
    return predictor


def load_pretrained_weights(checkpoint_path: Path | None) -> dict[str, Any]:
    """Load pretrained predictor weights."""
    if checkpoint_path is None:
        LOGGER.info("Downloading pretrained checkpoint from %s", DEFAULT_MODEL_URL)
        return torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    LOGGER.info("Loading pretrained checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "predictor" in checkpoint:
        return checkpoint["predictor"]
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
    predictor,
    renderer: GSplatRenderer,
    batch: dict[str, torch.Tensor | None],
    refiner: MaskDeltaRefiner,
) -> dict[str, torch.Tensor | RenderingOutputs | Gaussians3D]:
    """Run input-frame -> NDC Gaussians -> world-space -> target rendering."""
    source_image = batch["source_image"]
    source_depth = batch["source_depth"]
    disparity_factor = batch["disparity_factor"]
    source_intrinsics = batch["source_intrinsics"]
    source_original_intrinsics = batch["source_original_intrinsics"]
    source_original_size = batch["source_original_size"]
    source_extrinsics = batch["source_extrinsics"]
    target_intrinsics = batch["target_intrinsics"]
    target_original_intrinsics = batch["target_original_intrinsics"]
    target_original_size = batch["target_original_size"]
    target_extrinsics = batch["target_extrinsics"]

    assert isinstance(source_image, torch.Tensor)
    assert isinstance(disparity_factor, torch.Tensor)
    assert isinstance(source_intrinsics, torch.Tensor)
    assert isinstance(source_original_intrinsics, torch.Tensor)
    assert isinstance(source_original_size, torch.Tensor)
    assert isinstance(source_extrinsics, torch.Tensor)
    assert isinstance(target_intrinsics, torch.Tensor)
    assert isinstance(target_original_intrinsics, torch.Tensor)
    assert isinstance(target_original_size, torch.Tensor)
    assert isinstance(target_extrinsics, torch.Tensor)

    monodepth_output = predictor.monodepth_model(source_image)
    predicted_disparity = monodepth_output.disparity
    metric_depth = disparity_factor[:, :, None, None] / predicted_disparity.clamp(min=1e-4)
    aligned_depth, _ = predictor.depth_alignment(
        metric_depth,
        source_depth,
        monodepth_output.decoder_features,
    )

    init_output = predictor.init_model(source_image, aligned_depth)
    image_features = predictor.feature_model(
        init_output.feature_input,
        encodings=monodepth_output.output_features,
    )
    delta_values = predictor.prediction_head(image_features)
    gaussians_ndc = predictor.gaussian_composer(
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

    invisible_mask = compute_target_invisible_mask(
        aligned_depth[:, 0:1],
        source_intrinsics,
        source_extrinsics,
        target_intrinsics,
        target_extrinsics,
    )
    gaussian_invisible_gate = compute_gaussian_invisible_gate(
        gaussians_world,
        target_intrinsics,
        target_extrinsics,
        invisible_mask,
        delta_values.shape[2],
        delta_values.shape[-2],
        delta_values.shape[-1],
    )
    refiner_size = delta_values.shape[-2:]
    invisible_target_render = target_render.color * invisible_mask
    delta_correction = refiner(
        F.interpolate(source_image, size=refiner_size, mode="bilinear", align_corners=True),
        F.interpolate(target_render.color, size=refiner_size, mode="bilinear", align_corners=True),
        F.interpolate(invisible_target_render, size=refiner_size, mode="bilinear", align_corners=True),
        F.interpolate(invisible_mask, size=refiner_size, mode="nearest"),
    )
    delta_correction = delta_correction * gaussian_invisible_gate
    refined_gaussians_ndc = predictor.gaussian_composer(
        delta=delta_values + delta_correction,
        base_values=init_output.gaussian_base_values,
        global_scale=init_output.global_scale,
    )
    refined_gaussians_world = batch_unproject_gaussians(
        refined_gaussians_ndc,
        source_extrinsics,
        source_intrinsics,
        image_shape,
    )
    refined_target_render = renderer(
        refined_gaussians_world,
        target_extrinsics,
        target_intrinsics,
        image_width=source_image.shape[-1],
        image_height=source_image.shape[-2],
    )

    source_render_original = render_batch_at_sizes(
        renderer,
        gaussians_world,
        source_extrinsics,
        source_original_intrinsics,
        source_original_size,
    )
    refined_target_render_original = render_batch_at_sizes(
        renderer,
        refined_gaussians_world,
        target_extrinsics,
        target_original_intrinsics,
        target_original_size,
    )

    return {
        "gaussians_world": gaussians_world,
        "refined_gaussians_world": refined_gaussians_world,
        "source_render": source_render,
        "target_render": target_render,
        "refined_target_render": refined_target_render,
        "source_render_original": source_render_original,
        "refined_target_render_original": refined_target_render_original,
        "aligned_depth": aligned_depth,
        "invisible_mask": invisible_mask,
        "invisible_target_render": invisible_target_render,
    }



def compute_target_invisible_mask(
    source_depth: torch.Tensor,
    source_intrinsics: torch.Tensor,
    source_extrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    target_extrinsics: torch.Tensor,
) -> torch.Tensor:
    """Estimate target-view regions visible in target but not source."""
    batch_size, _, height, width = source_depth.shape
    device = source_depth.device
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xx = xx[None].expand(batch_size, -1, -1)
    yy = yy[None].expand(batch_size, -1, -1)
    z = source_depth[:, 0]

    fx = source_intrinsics[:, 0, 0][:, None, None]
    fy = source_intrinsics[:, 1, 1][:, None, None]
    cx = source_intrinsics[:, 0, 2][:, None, None]
    cy = source_intrinsics[:, 1, 2][:, None, None]

    x = (xx - cx) / fx * z
    y = (yy - cy) / fy * z
    points_cam = torch.stack([x, y, z, torch.ones_like(z)], dim=-1)

    source_c2w = torch.linalg.inv(source_extrinsics)
    points_world = points_cam.reshape(batch_size, -1, 4) @ source_c2w.transpose(-1, -2)
    points_target = points_world @ target_extrinsics.transpose(-1, -2)
    points_target = points_target[..., :3]

    z_t = points_target[..., 2]
    u = target_intrinsics[:, 0, 0][:, None] * (points_target[..., 0] / z_t.clamp(min=1e-6))
    u = u + target_intrinsics[:, 0, 2][:, None]
    v = target_intrinsics[:, 1, 1][:, None] * (points_target[..., 1] / z_t.clamp(min=1e-6))
    v = v + target_intrinsics[:, 1, 2][:, None]

    u0 = torch.floor(u)
    v0 = torch.floor(v)
    z_buffer = torch.full((batch_size, height * width), float("inf"), device=device)

    for du, dv in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
        uu = (u0 + du).long()
        vv = (v0 + dv).long()
        valid = (z_t > 1e-4) & (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
        flat_indices = vv * width + uu
        for batch_index in range(batch_size):
            valid_indices = flat_indices[batch_index][valid[batch_index]]
            valid_depths = z_t[batch_index][valid[batch_index]]
            if valid_indices.numel() > 0:
                z_buffer[batch_index].scatter_reduce_(
                    0, valid_indices, valid_depths, reduce="amin", include_self=True
                )

    coverage = torch.isfinite(z_buffer).float().view(batch_size, 1, height, width)
    coverage = F.max_pool2d(coverage, kernel_size=3, stride=1, padding=1)
    return 1.0 - coverage


def compute_gaussian_invisible_gate(
    gaussians_world: Gaussians3D,
    target_intrinsics: torch.Tensor,
    target_extrinsics: torch.Tensor,
    invisible_mask: torch.Tensor,
    num_layers: int,
    grid_height: int,
    grid_width: int,
) -> torch.Tensor:
    """Gate updates to gaussians projected into target-invisible regions."""
    batch_size, num_gaussians, _ = gaussians_world.mean_vectors.shape
    means_world = torch.cat(
        [
            gaussians_world.mean_vectors,
            torch.ones(batch_size, num_gaussians, 1, device=gaussians_world.mean_vectors.device),
        ],
        dim=-1,
    )
    means_target = means_world @ target_extrinsics.transpose(-1, -2)
    means_target = means_target[..., :3]

    z = means_target[..., 2].clamp(min=1e-6)
    u = target_intrinsics[:, 0, 0][:, None] * (means_target[..., 0] / z)
    u = u + target_intrinsics[:, 0, 2][:, None]
    v = target_intrinsics[:, 1, 1][:, None] * (means_target[..., 1] / z)
    v = v + target_intrinsics[:, 1, 2][:, None]

    norm_u = 2.0 * (u / max(invisible_mask.shape[-1] - 1, 1)) - 1.0
    norm_v = 2.0 * (v / max(invisible_mask.shape[-2] - 1, 1)) - 1.0
    grid = torch.stack([norm_u, norm_v], dim=-1).view(batch_size, num_gaussians, 1, 2)
    sampled_mask = F.grid_sample(
        invisible_mask,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(batch_size, num_gaussians)

    valid = (means_target[..., 2] > 1e-4).float()
    sampled_mask = sampled_mask * valid
    gate = sampled_mask.view(batch_size, num_layers, grid_height, grid_width)
    return gate[:, None]


def render_batch_at_sizes(
    renderer: GSplatRenderer,
    gaussians_world: Gaussians3D,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_sizes: torch.Tensor,
) -> list[RenderingOutputs]:
    """Render each batch item at its requested image size."""
    renders = []
    for index in range(gaussians_world.mean_vectors.shape[0]):
        renders.append(
            renderer(
                Gaussians3D(
                    mean_vectors=gaussians_world.mean_vectors[index : index + 1],
                    singular_values=gaussians_world.singular_values[index : index + 1],
                    quaternions=gaussians_world.quaternions[index : index + 1],
                    colors=gaussians_world.colors[index : index + 1],
                    opacities=gaussians_world.opacities[index : index + 1],
                ),
                extrinsics[index : index + 1],
                intrinsics[index : index + 1],
                image_width=int(image_sizes[index, 1].item()),
                image_height=int(image_sizes[index, 0].item()),
            )
        )
    return renders



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


def save_visualization_batch(
    output_dir: Path,
    global_step: int,
    batch: dict[str, torch.Tensor | None],
    outputs: dict[str, torch.Tensor | RenderingOutputs | Gaussians3D],
) -> None:
    """Save intermediate visualizations for debugging."""
    source_image = batch["source_image"]
    target_image = batch["target_image"]
    source_original_image = batch["source_original_image"]
    target_original_image = batch["target_original_image"]
    assert isinstance(source_image, torch.Tensor)
    assert isinstance(target_image, torch.Tensor)
    assert isinstance(source_original_image, list)
    assert isinstance(target_original_image, list)

    source_render = outputs["source_render"]
    target_render = outputs["target_render"]
    refined_target_render = outputs["refined_target_render"]
    source_render_original = outputs["source_render_original"]
    refined_target_render_original = outputs["refined_target_render_original"]
    invisible_mask = outputs["invisible_mask"]
    invisible_target_render = outputs["invisible_target_render"]
    assert isinstance(source_render, RenderingOutputs)
    assert isinstance(target_render, RenderingOutputs)
    assert isinstance(refined_target_render, RenderingOutputs)
    assert isinstance(source_render_original, list)
    assert isinstance(refined_target_render_original, list)
    assert isinstance(invisible_mask, torch.Tensor)
    assert isinstance(invisible_target_render, torch.Tensor)

    prefix = output_dir / f"step_{global_step:06d}"
    save_tensor_image(source_image[0], prefix.with_name(prefix.name + ".source.train.png"))
    save_tensor_image(target_image[0], prefix.with_name(prefix.name + ".target.train.png"))
    save_tensor_image(source_original_image[0], prefix.with_name(prefix.name + ".source.png"))
    save_tensor_image(target_original_image[0], prefix.with_name(prefix.name + ".target.png"))
    save_mask_image(invisible_mask[0], prefix.with_name(prefix.name + ".invisible_mask.png"))
    save_tensor_image(
        (invisible_target_render[0]).clamp(0.0, 1.0),
        prefix.with_name(prefix.name + ".invisible_target_render.png"),
    )
    save_tensor_image(
        source_render.color[0].clamp(0.0, 1.0),
        prefix.with_name(prefix.name + ".source_render.train.png"),
    )
    save_tensor_image(
        source_render_original[0].color[0].clamp(0.0, 1.0),
        prefix.with_name(prefix.name + ".source_render.png"),
    )
    save_tensor_image(
        refined_target_render.color[0].clamp(0.0, 1.0),
        prefix.with_name(prefix.name + ".target_render.train.png"),
    )
    save_tensor_image(
        refined_target_render_original[0].color[0].clamp(0.0, 1.0),
        prefix.with_name(prefix.name + ".target_render.png"),
    )


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    """Save a float tensor image in CHW format to disk."""
    image = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    io.save_image((image * 255.0).astype(np.uint8), path)


def save_mask_image(mask: torch.Tensor, path: Path) -> None:
    """Save a mask visualization."""
    colorized = vis.colorize_alpha(mask.detach().cpu())[None][0].permute(1, 2, 0).numpy()
    io.save_image(colorized.astype(np.uint8), path)


def save_checkpoint(
    path: Path,
    predictor,
    refiner,
    optimizer: torch.optim.Optimizer,
    global_step: int,
) -> None:
    """Save a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "predictor": predictor.state_dict(),
            "refiner": refiner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
        },
        path,
    )
    LOGGER.info("Saved checkpoint to %s", path)
