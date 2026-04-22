"""Contains `sharp finetune-ddp` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import click
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from sharp.cli.finetune import (
    build_finetune_predictor,
    forward_training_pass,
    move_batch_to_device,
    save_visualization_batch,
)
from sharp.cli.predict import DEFAULT_MODEL_URL
from sharp.training.dataset import (
    MultiScenePosedVideoDataset,
    PosedVideoDataset,
    collate_view_pairs,
)
from sharp.training.losses import FineTuneLoss, FineTuneLossWeights
from sharp.utils import logging as logging_utils
from sharp.utils.gsplat import GSplatRenderer

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--data-root", type=click.Path(path_type=Path, exists=True, file_okay=False), default=None)
@click.option("--video-path", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None)
@click.option("--pose-path", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None)
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--checkpoint-path", type=click.Path(path_type=Path, dir_okay=False), default=None)
@click.option("--batch-size", type=int, default=1, show_default=True)
@click.option("--num-workers", type=int, default=2, show_default=True)
@click.option("--epochs", type=int, default=1, show_default=True)
@click.option("--lr", type=float, default=1e-5, show_default=True)
@click.option("--weight-decay", type=float, default=1e-4, show_default=True)
@click.option("--samples-per-epoch", type=int, default=256, show_default=True)
@click.option("--min-frame-distance", type=int, default=4, show_default=True)
@click.option("--max-frame-distance", type=int, default=48, show_default=True)
@click.option("--log-every", type=int, default=10, show_default=True)
@click.option("--visualize-every", type=int, default=50, show_default=True)
@click.option("--save-every", type=int, default=200, show_default=True)
@click.option("--max-steps", type=int, default=0, show_default=True)
@click.option("--preload-video/--stream-video", default=False, show_default=True)
@click.option("--perceptual/--no-perceptual", default=True, show_default=True)
@click.option("--color-weight", type=float, default=1.0, show_default=True)
@click.option("--alpha-weight", type=float, default=0.05, show_default=True)
@click.option("--perceptual-weight", type=float, default=0.1, show_default=True)
@click.option("--loss-border-ratio", type=float, default=0.0, show_default=True)
@click.option("--invisible-mask-dilation-px", type=int, default=6, show_default=True)
@click.option("--invisible-loss-boost", type=float, default=1.5, show_default=True)
@click.option("--gaussian-mask-dilation-px", type=int, default=8, show_default=True)
@click.option("--train-prediction-head/--freeze-prediction-head", default=False, show_default=True)
@click.option("-v", "--verbose", is_flag=True, default=False)
def finetune_ddp_cli(
    data_root: Path | None,
    video_path: Path | None,
    pose_path: Path | None,
    output_dir: Path,
    checkpoint_path: Path | None,
    batch_size: int,
    num_workers: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    samples_per_epoch: int,
    min_frame_distance: int,
    max_frame_distance: int,
    log_every: int,
    visualize_every: int,
    save_every: int,
    max_steps: int,
    preload_video: bool,
    perceptual: bool,
    color_weight: float,
    alpha_weight: float,
    perceptual_weight: float,
    loss_border_ratio: float,
    invisible_mask_dilation_px: int,
    invisible_loss_boost: float,
    gaussian_mask_dilation_px: int,
    train_prediction_head: bool,
    verbose: bool,
) -> None:
    """DDP multi-GPU fine-tuning entrypoint (launch with torchrun)."""
    if data_root is None and (video_path is None or pose_path is None):
        raise click.UsageError("Please provide --data-root or both --video-path and --pose-path.")

    if not torch.cuda.is_available():
        raise RuntimeError("finetune-ddp requires CUDA.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        raise RuntimeError("finetune-ddp must be launched with torchrun and WORLD_SIZE > 1.")

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device_t = torch.device("cuda", local_rank)

    logging_utils.configure(logging.DEBUG if verbose and rank == 0 else logging.INFO)
    is_main_process = rank == 0

    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / "visualizations"
    if is_main_process:
        visualization_dir.mkdir(parents=True, exist_ok=True)

    predictor = build_finetune_predictor(
        checkpoint_path,
        train_prediction_head=train_prediction_head,
    ).to(device_t)
    predictor = DDP(
        predictor,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    internal_resolution = (1536, 1536)
    if data_root is not None:
        dataset = MultiScenePosedVideoDataset(
            data_root=data_root,
            internal_resolution=internal_resolution,
            min_frame_distance=min_frame_distance,
            max_frame_distance=max_frame_distance,
            samples_per_scene=samples_per_epoch,
            preload=preload_video,
            load_depth=False,
        )
    else:
        assert video_path is not None and pose_path is not None
        dataset = PosedVideoDataset(
            video_path=video_path,
            pose_path=pose_path,
            internal_resolution=internal_resolution,
            min_frame_distance=min_frame_distance,
            max_frame_distance=max_frame_distance,
            samples_per_epoch=samples_per_epoch,
            preload=preload_video,
            load_depth=False,
        )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_view_pairs,
        pin_memory=True,
        drop_last=False,
    )

    renderer = GSplatRenderer(color_space="linearRGB", background_color="black").to(device_t)
    loss_module = FineTuneLoss(
        weights=FineTuneLossWeights(
            color=color_weight * max(float(invisible_loss_boost), 0.0),
            alpha=alpha_weight * max(float(invisible_loss_boost), 0.0),
            perceptual=perceptual_weight * max(float(invisible_loss_boost), 0.0),
        ),
        use_perceptual=perceptual,
    ).to(device_t)

    trainable_parameters = [p for p in predictor.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise click.UsageError(
            "No trainable parameters selected. Use --train-prediction-head to enable fine-tuning."
        )
    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)

    if is_main_process:
        write_ddp_config(
            output_dir / "finetune_ddp_config.json",
            {
                "data_root": str(data_root) if data_root is not None else None,
                "video_path": str(video_path) if video_path is not None else None,
                "pose_path": str(pose_path) if pose_path is not None else None,
                "checkpoint_path": str(checkpoint_path) if checkpoint_path else DEFAULT_MODEL_URL,
                "world_size": world_size,
                "batch_size_per_gpu": batch_size,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "samples_per_epoch": samples_per_epoch,
                "invisible_mask_dilation_px": invisible_mask_dilation_px,
                "invisible_loss_boost": invisible_loss_boost,
                "gaussian_mask_dilation_px": gaussian_mask_dilation_px,
                "train_prediction_head": train_prediction_head,
            },
        )

    global_step = 0
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            batch = move_batch_to_device(batch, device_t)
            optimizer.zero_grad(set_to_none=True)
            outputs = forward_training_pass(
                predictor,
                renderer,
                batch,
                loss_border_ratio=loss_border_ratio,
                invisible_mask_dilation_px=invisible_mask_dilation_px,
                gaussian_mask_dilation_px=gaussian_mask_dilation_px,
            )
            losses = loss_module(
                source_render=outputs["source_render"],
                target_render=outputs["refined_target_render"],
                batch=batch,
                aligned_depth=outputs["aligned_depth"],
                delta_values=outputs["delta_values"],
                gaussians_ndc=outputs["gaussians_ndc"],
                gaussians_world=outputs["gaussians_world"],
                depth_alignment_map=outputs["depth_alignment_map"],
                loss_region_mask=outputs["loss_region_mask"],
            )
            losses.total.backward()
            optimizer.step()
            global_step += 1

            if is_main_process and global_step % log_every == 0:
                LOGGER.info("epoch=%d step=%d total=%.4f", epoch, global_step, losses.total.item())
            if is_main_process and global_step % visualize_every == 0:
                save_visualization_batch(visualization_dir, global_step, batch, outputs)
            if is_main_process and global_step % save_every == 0:
                save_ddp_checkpoint(output_dir / f"step_{global_step:06d}.pt", predictor, optimizer, global_step)

            if max_steps > 0 and global_step >= max_steps:
                break
        if max_steps > 0 and global_step >= max_steps:
            break

    if is_main_process:
        save_ddp_checkpoint(output_dir / "last.pt", predictor, optimizer, global_step)

    dist.barrier()
    dist.destroy_process_group()


def save_ddp_checkpoint(
    path: Path,
    predictor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    payload = {
        "predictor": {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in predictor.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    torch.save(payload, path)


def write_ddp_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
