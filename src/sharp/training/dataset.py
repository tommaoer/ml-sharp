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
    refiner_image: torch.Tensor
    refiner_intrinsics: torch.Tensor
    original_image: torch.Tensor
    original_intrinsics: torch.Tensor
    extrinsics: torch.Tensor
    frame_index: int
    depth: torch.Tensor | None = None


@dataclass(frozen=True)
class ViewPairSample:
    """A training pair consisting of an input view and a target view."""

    scene_name: str
    source: FrameRecord
    target: FrameRecord
    disparity_factor: torch.Tensor


class PosedVideoScene:
    """Represents one video sequence with per-frame camera intrinsics/extrinsics."""

    def __init__(
        self,
        video_path: str | Path,
        pose_path: str | Path,
        internal_resolution: tuple[int, int] = (1536, 1536),
        refiner_resolution: tuple[int, int] | None = None,
        preload: bool = False,
        load_depth: bool = True,
    ) -> None:
        """Initialize one posed-video scene."""
        self.video_path = Path(video_path)
        self.pose_path = Path(pose_path)
        self.scene_name = self.video_path.parent.name
        self.internal_resolution = internal_resolution
        self.refiner_resolution = refiner_resolution or internal_resolution
        self.preload = preload
        self.load_depth = load_depth
        self.depth_path = self.video_path.parent / "depth_sequence.npy"
        self.frame_cache: dict[int, torch.Tensor] | None = {} if self.preload else None

        with self.pose_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        self.c2ws = torch.tensor(metadata["c2ws"], dtype=torch.float32)
        if self.c2ws.ndim != 3 or self.c2ws.shape[-2:] != (4, 4):
            raise ValueError(f"Expected c2ws to have shape [T, 4, 4] in {self.pose_path}.")

        self.num_frames = int(self.c2ws.shape[0])
        if self.num_frames < 2:
            raise ValueError(f"Scene {self.scene_name} must contain at least two frames.")

        self.intrinsics = self._create_intrinsics(metadata)
        # Depth loading is lazy to avoid long startup on very large multi-scene datasets.
        self.depth_sequence = None

    def _load_depth_sequence(self) -> np.ndarray | None:
        if not self.depth_path.exists():
            return None

        # Use eager loading instead of memmap to avoid keeping one file descriptor
        # open per scene (can hit "Too many open files" on large multi-scene runs).
        depth_np = np.load(self.depth_path)
        if depth_np.ndim == 3:
            depth_np = depth_np[:, None, :, :]
        elif depth_np.ndim == 4 and depth_np.shape[-1] == 1:
            depth_np = np.transpose(depth_np, (0, 3, 1, 2))
        if depth_np.ndim != 4 or depth_np.shape[1] != 1:
            raise ValueError(
                f"Expected depth_sequence.npy to have shape [T, H, W], [T, 1, H, W], or "
                f"[T, H, W, 1], got "
                f"{tuple(depth_np.shape)} in {self.depth_path}."
            )
        if depth_np.shape[0] < self.num_frames:
            raise ValueError(
                f"Depth sequence {self.depth_path} has {depth_np.shape[0]} frames but pose json "
                f"expects {self.num_frames}."
            )
        return depth_np[: self.num_frames]

    @staticmethod
    def _create_intrinsics(metadata: dict[str, Any]) -> torch.Tensor:
        intrinsics = torch.eye(4, dtype=torch.float32)
        intrinsics[0, 0] = float(metadata["fl_x"])
        intrinsics[1, 1] = float(metadata.get("fl_y", metadata["fl_x"]))
        intrinsics[0, 2] = float(metadata["cx"])
        intrinsics[1, 2] = float(metadata["cy"])
        return intrinsics

    @staticmethod
    def _frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], repeats=3, axis=-1)
        return torch.from_numpy(frame[..., :3].copy()).float().permute(2, 0, 1) / 255.0

    def load_frame(self, frame_index: int) -> torch.Tensor:
        """Load one RGB frame from the scene video."""
        if self.frame_cache is not None and frame_index in self.frame_cache:
            return self.frame_cache[frame_index].clone()
        reader = iio.get_reader(self.video_path)
        try:
            frame = reader.get_data(frame_index)
        finally:
            reader.close()
        frame_tensor = self._frame_to_tensor(frame)
        if self.frame_cache is not None:
            self.frame_cache[frame_index] = frame_tensor
        return frame_tensor

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

    def build_frame_record(self, frame_index: int) -> FrameRecord:
        """Build one frame record with both training and original-resolution views."""
        original_image = self.load_frame(frame_index)
        _, height, width = original_image.shape
        target_height, target_width = self.internal_resolution
        refiner_height, refiner_width = self.refiner_resolution
        image = F.interpolate(
            original_image[None],
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=True,
        )[0]
        refiner_image = F.interpolate(
            original_image[None],
            size=(refiner_height, refiner_width),
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
        refiner_intrinsics = self._scale_intrinsics(
            self.intrinsics,
            width,
            height,
            refiner_width,
            refiner_height,
        )
        depth = None
        if self.load_depth:
            if self.depth_sequence is None:
                self.depth_sequence = self._load_depth_sequence()
        if self.depth_sequence is not None:
            depth_frame = torch.from_numpy(
                np.array(self.depth_sequence[frame_index], copy=True)
            ).float()[None]
            depth = F.interpolate(
                depth_frame,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=True,
            )[0]
        return FrameRecord(
            image=image,
            intrinsics=intrinsics,
            refiner_image=refiner_image,
            refiner_intrinsics=refiner_intrinsics,
            original_image=original_image,
            original_intrinsics=self.intrinsics.clone(),
            extrinsics=torch.linalg.inv(self.c2ws[frame_index]),
            frame_index=frame_index,
            depth=depth,
        )


class PosedVideoDataset(Dataset[ViewPairSample]):
    """Loads a single video sequence with per-frame camera intrinsics/extrinsics."""

    def __init__(
        self,
        video_path: str | Path,
        pose_path: str | Path,
        internal_resolution: tuple[int, int] = (1536, 1536),
        refiner_resolution: tuple[int, int] | None = None,
        min_frame_distance: int = 4,
        max_frame_distance: int = 48,
        samples_per_epoch: int | None = None,
        preload: bool = False,
        load_depth: bool = True,
    ) -> None:
        """Initialize the dataset."""
        self.scene = PosedVideoScene(
            video_path=video_path,
            pose_path=pose_path,
            internal_resolution=internal_resolution,
            refiner_resolution=refiner_resolution,
            preload=preload,
            load_depth=load_depth,
        )
        self.min_frame_distance = min_frame_distance
        self.max_frame_distance = max_frame_distance
        self.samples_per_epoch = samples_per_epoch or self.scene.num_frames

    def _sample_target_index(self, source_index: int) -> int:
        candidates = [
            index
            for index in range(self.scene.num_frames)
            if self.min_frame_distance <= abs(index - source_index) <= self.max_frame_distance
        ]
        if not candidates:
            candidates = [index for index in range(self.scene.num_frames) if index != source_index]
        if not candidates:
            raise ValueError("Could not sample a target frame different from the source frame.")
        return random.choice(candidates)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> ViewPairSample:
        del index
        source_index = random.randrange(self.scene.num_frames)
        target_index = self._sample_target_index(source_index)

        source = self.scene.build_frame_record(source_index)
        target = self.scene.build_frame_record(target_index)
        disparity_factor = torch.tensor(
            [source.intrinsics[0, 0] / float(source.image.shape[-1])], dtype=torch.float32
        )
        return ViewPairSample(
            scene_name=self.scene.scene_name,
            source=source,
            target=target,
            disparity_factor=disparity_factor,
        )


class MultiScenePosedVideoDataset(Dataset[ViewPairSample]):
    """Loads many scene folders, each containing one video and one pose json."""

    def __init__(
        self,
        data_root: str | Path,
        internal_resolution: tuple[int, int] = (1536, 1536),
        refiner_resolution: tuple[int, int] | None = None,
        min_frame_distance: int = 4,
        max_frame_distance: int = 48,
        samples_per_scene: int = 32,
        preload: bool = False,
        load_depth: bool = True,
        video_extensions: tuple[str, ...] = (".mp4", ".MP4"),
    ) -> None:
        """Initialize the multi-scene dataset."""
        self.data_root = Path(data_root)
        self.min_frame_distance = min_frame_distance
        self.max_frame_distance = max_frame_distance
        self.samples_per_scene = samples_per_scene
        self.load_depth = load_depth

        scene_dirs = sorted(path for path in self.data_root.iterdir() if path.is_dir())
        if not scene_dirs:
            raise ValueError(f"No scene folders found in {self.data_root}.")

        self.scenes = []
        for scene_dir in scene_dirs:
            video_candidates = [
                path
                for path in sorted(scene_dir.iterdir())
                if path.is_file() and path.suffix in video_extensions
            ]
            json_candidates = [
                path
                for path in sorted(scene_dir.iterdir())
                if path.is_file() and path.suffix == ".json"
            ]
            pose_json = self._select_pose_json(scene_dir, json_candidates)
            if len(video_candidates) != 1 or pose_json is None:
                raise ValueError(
                    f"Each scene folder must contain exactly one video and at least one valid pose json: "
                    f"{scene_dir}."
                )
            self.scenes.append(
                PosedVideoScene(
                    video_candidates[0],
                    pose_json,
                    internal_resolution=internal_resolution,
                    refiner_resolution=refiner_resolution,
                    preload=preload,
                    load_depth=self.load_depth,
                )
            )

    @staticmethod
    def _select_pose_json(scene_dir: Path, json_candidates: list[Path]) -> Path | None:
        """Select preferred pose json from a scene directory.

        Preference order:
          1) camera_params.json
          2) poses.json
          3) cameras.json
          4) any other json that is not a known legacy/backup file.
        """
        if not json_candidates:
            return None

        preferred_names = ("camera_params.json", "poses.json", "cameras.json")
        for name in preferred_names:
            for path in json_candidates:
                if path.name == name:
                    return path

        filtered = [
            path for path in json_candidates if path.stem not in {"camera_params_old", "poses_old"}
        ]
        return filtered[0] if filtered else json_candidates[0]

    def _sample_target_index(self, scene: PosedVideoScene, source_index: int) -> int:
        candidates = [
            index
            for index in range(scene.num_frames)
            if self.min_frame_distance <= abs(index - source_index) <= self.max_frame_distance
        ]
        if not candidates:
            candidates = [index for index in range(scene.num_frames) if index != source_index]
        if not candidates:
            raise ValueError(
                f"Could not sample a target frame different from the source in {scene.scene_name}."
            )
        return random.choice(candidates)

    def __len__(self) -> int:
        return len(self.scenes) * self.samples_per_scene

    def __getitem__(self, index: int) -> ViewPairSample:
        scene = self.scenes[index % len(self.scenes)]
        source_index = random.randrange(scene.num_frames)
        target_index = self._sample_target_index(scene, source_index)

        source = scene.build_frame_record(source_index)
        target = scene.build_frame_record(target_index)
        disparity_factor = torch.tensor(
            [source.intrinsics[0, 0] / float(source.image.shape[-1])], dtype=torch.float32
        )
        return ViewPairSample(
            scene_name=scene.scene_name,
            source=source,
            target=target,
            disparity_factor=disparity_factor,
        )


def collate_view_pairs(batch: list[ViewPairSample]) -> dict[str, Any]:
    """Collate function for fine-tuning batches."""
    return {
        "scene_name": [item.scene_name for item in batch],
        "source_image": torch.stack([item.source.image for item in batch], dim=0),
        "target_image": torch.stack([item.target.image for item in batch], dim=0),
        "source_refiner_image": torch.stack([item.source.refiner_image for item in batch], dim=0),
        "source_original_image": [item.source.original_image for item in batch],
        "target_original_image": [item.target.original_image for item in batch],
        "source_intrinsics": torch.stack([item.source.intrinsics for item in batch], dim=0),
        "target_intrinsics": torch.stack([item.target.intrinsics for item in batch], dim=0),
        "source_refiner_intrinsics": torch.stack(
            [item.source.refiner_intrinsics for item in batch], dim=0
        ),
        "target_refiner_intrinsics": torch.stack(
            [item.target.refiner_intrinsics for item in batch], dim=0
        ),
        "source_original_intrinsics": torch.stack(
            [item.source.original_intrinsics for item in batch], dim=0
        ),
        "target_original_intrinsics": torch.stack(
            [item.target.original_intrinsics for item in batch], dim=0
        ),
        "source_extrinsics": torch.stack([item.source.extrinsics for item in batch], dim=0),
        "target_extrinsics": torch.stack([item.target.extrinsics for item in batch], dim=0),
        "source_depth": (
            torch.stack([item.source.depth for item in batch], dim=0)
            if all(item.source.depth is not None for item in batch)
            else None
        ),
        "target_depth": (
            torch.stack([item.target.depth for item in batch], dim=0)
            if all(item.target.depth is not None for item in batch)
            else None
        ),
        "disparity_factor": torch.stack([item.disparity_factor for item in batch], dim=0),
        "source_frame_index": torch.tensor([item.source.frame_index for item in batch]),
        "target_frame_index": torch.tensor([item.target.frame_index for item in batch]),
        "source_original_size": torch.tensor(
            [[item.source.original_image.shape[-2], item.source.original_image.shape[-1]] for item in batch]
        ),
        "target_original_size": torch.tensor(
            [[item.target.original_image.shape[-2], item.target.original_image.shape[-1]] for item in batch]
        ),
    }


__all__ = [
    "FrameRecord",
    "MultiScenePosedVideoDataset",
    "PosedVideoDataset",
    "PosedVideoScene",
    "ViewPairSample",
    "collate_view_pairs",
]
