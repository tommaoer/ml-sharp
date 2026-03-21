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
    """Frame metadata for one frame inside a scene video."""

    image: torch.Tensor
    intrinsics: torch.Tensor
    extrinsics: torch.Tensor
    frame_index: int
    depth: torch.Tensor | None = None


@dataclass(frozen=True)
class ViewPairSample:
    """A training pair consisting of an input view and a novel view."""

    scene_name: str
    source: FrameRecord
    target: FrameRecord
    disparity_factor: torch.Tensor


class VideoFolderSceneDataset(Dataset[ViewPairSample]):
    """Loads multi-view training samples from folder-based video scenes.

    Each scene directory must contain one video file and one json file. The json is
    expected to contain camera intrinsics (`fl_x`, `fl_y`, `cx`, `cy`) and a `c2ws`
    array with one 4x4 camera-to-world transform per frame.
    """

    def __init__(
        self,
        root: str | Path,
        internal_resolution: tuple[int, int] = (1536, 1536),
        max_scenes: int | None = None,
        max_frame_gap: int | None = None,
        min_frame_distance: int = 1,
        samples_per_scene: int | None = None,
        file_extensions: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi"),
        json_name: str | None = None,
        preload: bool = False,
        random_target: bool = True,
    ) -> None:
        """Initialize the dataset."""
        self.root = Path(root)
        self.internal_resolution = internal_resolution
        self.max_frame_gap = max_frame_gap
        self.min_frame_distance = min_frame_distance
        self.samples_per_scene = samples_per_scene
        self.file_extensions = file_extensions
        self.json_name = json_name
        self.preload = preload
        self.random_target = random_target

        scene_dirs = sorted(path for path in self.root.iterdir() if path.is_dir())
        if max_scenes is not None:
            scene_dirs = scene_dirs[:max_scenes]

        self.scenes = [self._load_scene(scene_dir) for scene_dir in scene_dirs]
        self.scenes = [scene for scene in self.scenes if scene["num_frames"] >= 2]
        if not self.scenes:
            raise ValueError(f"No valid training scenes found in {self.root}.")

        if self.samples_per_scene is None:
            self.samples_per_scene = max(scene["num_frames"] for scene in self.scenes)

    def _find_video_path(self, scene_dir: Path) -> Path:
        candidates = [
            path
            for path in sorted(scene_dir.iterdir())
            if path.is_file() and path.suffix.lower() in self.file_extensions
        ]
        if not candidates:
            raise FileNotFoundError(f"No supported video file found in {scene_dir}.")
        if len(candidates) > 1:
            raise ValueError(
                f"Found multiple video files in {scene_dir}; please keep only one video file."
            )
        return candidates[0]

    def _find_json_path(self, scene_dir: Path) -> Path:
        if self.json_name is not None:
            path = scene_dir / self.json_name
            if not path.exists():
                raise FileNotFoundError(f"Metadata file {path} not found.")
            return path

        candidates = [
            path
            for path in sorted(scene_dir.iterdir())
            if path.is_file() and path.suffix == ".json"
        ]
        if not candidates:
            raise FileNotFoundError(f"No json metadata file found in {scene_dir}.")
        if len(candidates) > 1:
            raise ValueError(
                f"Found multiple json files in {scene_dir}; please pass json_name explicitly."
            )
        return candidates[0]

    def _load_scene(self, scene_dir: Path) -> dict[str, Any]:
        video_path = self._find_video_path(scene_dir)
        json_path = self._find_json_path(scene_dir)

        with json_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        c2ws = torch.tensor(metadata["c2ws"], dtype=torch.float32)
        if c2ws.ndim != 3 or c2ws.shape[-2:] != (4, 4):
            raise ValueError(f"Expected c2ws to have shape [T, 4, 4] in {json_path}.")

        num_frames = int(c2ws.shape[0])
        intrinsics = self._create_intrinsics(metadata)
        frames = self._load_video_frames(video_path, num_frames) if self.preload else None
        return {
            "name": scene_dir.name,
            "video_path": video_path,
            "json_path": json_path,
            "intrinsics": intrinsics,
            "c2ws": c2ws,
            "num_frames": num_frames,
            "frames": frames,
        }

    @staticmethod
    def _create_intrinsics(metadata: dict[str, Any]) -> torch.Tensor:
        intrinsics = torch.eye(4, dtype=torch.float32)
        intrinsics[0, 0] = float(metadata["fl_x"])
        intrinsics[1, 1] = float(metadata.get("fl_y", metadata["fl_x"]))
        intrinsics[0, 2] = float(metadata["cx"])
        intrinsics[1, 2] = float(metadata["cy"])
        return intrinsics

    def _load_video_frames(self, video_path: Path, num_frames: int) -> list[torch.Tensor]:
        frames = list(iio.imiter(video_path))
        if len(frames) < num_frames:
            raise ValueError(
                f"Video {video_path} has {len(frames)} frames but json expects {num_frames}."
            )
        return [self._frame_to_tensor(frame) for frame in frames[:num_frames]]

    @staticmethod
    def _frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], repeats=3, axis=-1)
        frame = frame[..., :3]
        return torch.from_numpy(frame.copy()).float().permute(2, 0, 1) / 255.0

    def _load_frame(self, scene: dict[str, Any], frame_index: int) -> torch.Tensor:
        if scene["frames"] is not None:
            return scene["frames"][frame_index].clone()

        reader = iio.get_reader(scene["video_path"])
        try:
            frame = reader.get_data(frame_index)
        finally:
            reader.close()
        return self._frame_to_tensor(frame)

    def _resize_frame(self, frame: torch.Tensor) -> torch.Tensor:
        target_height, target_width = self.internal_resolution
        return F.interpolate(
            frame[None],
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=True,
        )[0]

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

    def _build_frame_record(self, scene: dict[str, Any], frame_index: int) -> FrameRecord:
        image = self._load_frame(scene, frame_index)
        _, height, width = image.shape
        target_height, target_width = self.internal_resolution
        image = self._resize_frame(image)
        intrinsics = self._scale_intrinsics(
            scene["intrinsics"],
            width,
            height,
            target_width,
            target_height,
        )
        c2w = scene["c2ws"][frame_index]
        return FrameRecord(
            image=image,
            intrinsics=intrinsics,
            extrinsics=torch.linalg.inv(c2w),
            frame_index=frame_index,
        )

    def _choose_target_index(self, num_frames: int, src_index: int) -> int:
        lower = max(0, src_index - self.max_frame_gap) if self.max_frame_gap is not None else 0
        upper = (
            min(num_frames - 1, src_index + self.max_frame_gap)
            if self.max_frame_gap is not None
            else num_frames - 1
        )
        candidates = [
            index
            for index in range(lower, upper + 1)
            if abs(index - src_index) >= self.min_frame_distance
        ]
        if not candidates:
            candidates = [index for index in range(num_frames) if index != src_index]
        return random.choice(candidates) if self.random_target else candidates[0]

    def __len__(self) -> int:
        return len(self.scenes) * self.samples_per_scene

    def __getitem__(self, index: int) -> ViewPairSample:
        scene = self.scenes[index % len(self.scenes)]
        num_frames = scene["num_frames"]
        source_index = random.randrange(num_frames)
        target_index = self._choose_target_index(num_frames, source_index)

        source = self._build_frame_record(scene, source_index)
        target = self._build_frame_record(scene, target_index)
        disparity_factor = torch.tensor(
            [source.intrinsics[0, 0] / float(source.image.shape[-1])],
            dtype=torch.float32,
        )
        return ViewPairSample(
            scene_name=scene["name"],
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
    "ViewPairSample",
    "VideoFolderSceneDataset",
    "collate_view_pairs",
]
