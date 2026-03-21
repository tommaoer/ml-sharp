"""Dataset helpers for SHARP fine-tuning.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as iio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FrameRecord:
    """Frame metadata for one frame inside a sequence."""

    image: torch.Tensor
    intrinsics: torch.Tensor
    extrinsics: torch.Tensor
    frame_index: int
    depth: torch.Tensor | None = None


@dataclass(frozen=True)
class ViewPairSample:
    """A training pair consisting of an input view and a target view."""

    source: FrameRecord
    target: FrameRecord
    disparity_factor: torch.Tensor


class PosedVideoDataset(Dataset[ViewPairSample]):
    """Loads a single video sequence with per-frame camera intrinsics/extrinsics."""

    def __init__(
        self,
        video_path: str | Path,
        pose_path: str | Path,
        internal_resolution: tuple[int, int] = (1536, 1536),
        min_frame_distance: int = 4,
        max_frame_distance: int = 48,
        samples_per_epoch: int | None = None,
        preload: bool = False,
    ) -> None:
        """Initialize the dataset."""
        self.video_path = Path(video_path)
        self.pose_path = Path(pose_path)
        self.internal_resolution = internal_resolution
        self.min_frame_distance = min_frame_distance
        self.max_frame_distance = max_frame_distance
        self.preload = preload

        with self.pose_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        self.c2ws = torch.tensor(metadata["c2ws"], dtype=torch.float32)
        if self.c2ws.ndim != 3 or self.c2ws.shape[-2:] != (4, 4):
            raise ValueError(f"Expected c2ws to have shape [T, 4, 4] in {self.pose_path}.")

        self.num_frames = int(self.c2ws.shape[0])
        if self.num_frames < 2:
            raise ValueError("At least two frames are required for fine-tuning.")

        self.intrinsics = self._create_intrinsics(metadata)
        self.frames = self._load_video_frames() if preload else None
        self.samples_per_epoch = samples_per_epoch or self.num_frames

    @staticmethod
    def _create_intrinsics(metadata: dict[str, Any]) -> torch.Tensor:
        intrinsics = torch.eye(4, dtype=torch.float32)
        intrinsics[0, 0] = float(metadata["fl_x"])
        intrinsics[1, 1] = float(metadata.get("fl_y", metadata["fl_x"]))
        intrinsics[0, 2] = float(metadata["cx"])
        intrinsics[1, 2] = float(metadata["cy"])
        return intrinsics

    def _load_video_frames(self) -> list[torch.Tensor]:
        frames = list(iio.imiter(self.video_path))
        if len(frames) < self.num_frames:
            raise ValueError(
                f"Video {self.video_path} has {len(frames)} frames but json expects "
                f"{self.num_frames}."
            )
        return [self._frame_to_tensor(frame) for frame in frames[: self.num_frames]]

    @staticmethod
    def _frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], repeats=3, axis=-1)
        return torch.from_numpy(frame[..., :3].copy()).float().permute(2, 0, 1) / 255.0

    def _load_frame(self, frame_index: int) -> torch.Tensor:
        if self.frames is not None:
            return self.frames[frame_index].clone()
        reader = iio.get_reader(self.video_path)
        try:
            frame = reader.get_data(frame_index)
        finally:
            reader.close()
        return self._frame_to_tensor(frame)

    @staticmethod
    def _scale_intrinsics(
        intrinsics: torch.Tensor,
        width: int,
        height: int,
        target_width: int,
        target_height: int,
    ) -> torch.Tensor:
        scaled = intrinsics.clone()
        scaled[0] *= float(target_width) / float(width)
        scaled[1] *= float(target_height) / float(height)
        return scaled

    def _build_frame_record(self, frame_index: int) -> FrameRecord:
        image = self._load_frame(frame_index)
        _, height, width = image.shape
        target_height, target_width = self.internal_resolution
        image = F.interpolate(
            image[None],
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=True,
        )[0]
        intrinsics = self._scale_intrinsics(
            self.intrinsics,
            width,
            height,
            target_width,
            target_height,
        )
        return FrameRecord(
            image=image,
            intrinsics=intrinsics,
            extrinsics=torch.linalg.inv(self.c2ws[frame_index]),
            frame_index=frame_index,
        )

    def _sample_target_index(self, source_index: int) -> int:
        candidates = [
            index
            for index in range(self.num_frames)
            if self.min_frame_distance <= abs(index - source_index) <= self.max_frame_distance
        ]
        if not candidates:
            candidates = [
                index for index in range(self.num_frames) if index != source_index
            ]
        if not candidates:
            raise ValueError("Could not sample a target frame different from the source frame.")
        return random.choice(candidates)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> ViewPairSample:
        del index
        source_index = random.randrange(self.num_frames)
        target_index = self._sample_target_index(source_index)

        source = self._build_frame_record(source_index)
        target = self._build_frame_record(target_index)
        disparity_factor = torch.tensor(
            [source.intrinsics[0, 0] / float(source.image.shape[-1])], dtype=torch.float32
        )
        return ViewPairSample(source=source, target=target, disparity_factor=disparity_factor)


def collate_view_pairs(batch: list[ViewPairSample]) -> dict[str, Any]:
    """Collate function for fine-tuning batches."""
    return {
        "source_image": torch.stack([item.source.image for item in batch], dim=0),
        "target_image": torch.stack([item.target.image for item in batch], dim=0),
        "source_intrinsics": torch.stack([item.source.intrinsics for item in batch], dim=0),
        "target_intrinsics": torch.stack([item.target.intrinsics for item in batch], dim=0),
        "source_extrinsics": torch.stack([item.source.extrinsics for item in batch], dim=0),
        "target_extrinsics": torch.stack([item.target.extrinsics for item in batch], dim=0),
        "source_depth": None,
        "target_depth": None,
        "disparity_factor": torch.stack([item.disparity_factor for item in batch], dim=0),
        "source_frame_index": torch.tensor([item.source.frame_index for item in batch]),
        "target_frame_index": torch.tensor([item.target.frame_index for item in batch]),
    }


__all__ = [
    "FrameRecord",
    "PosedVideoDataset",
    "ViewPairSample",
    "collate_view_pairs",
]
