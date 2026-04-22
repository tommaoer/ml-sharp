"""Dataset utilities for SHARP fine-tuning on video folders.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import torch
from sharp.utils import io
from torch.utils.data import Dataset


@dataclass
class VideoSequenceRecord:
    """A single sequence folder with one video + one json camera file."""

    root: Path
    video_path: Path
    json_path: Path


class VideoCameraFineTuneDataset(Dataset):
    """Loads random source/target frame pairs from multi-sequence folders.

    Expected layout:
        dataset_root/
          seq_000/
            *.mp4 (or *.mov)
            *.json
          seq_001/
            *.mp4
            *.json
    """

    def __init__(
        self,
        dataset_root: Path,
        min_view_distance: float = 0.05,
        max_view_distance: float = 2.0,
        max_samples: int = 100000,
    ) -> None:
        """Initialize dataset and discover valid sequence folders."""
        self.dataset_root = dataset_root
        self.min_view_distance = min_view_distance
        self.max_view_distance = max_view_distance
        self.max_samples = max_samples

        self.records = self._discover_records(dataset_root)
        if len(self.records) == 0:
            raise RuntimeError(f"No valid video+json folders found under {dataset_root}")

        self._cache: dict[Path, tuple[list[np.ndarray], dict]] = {}

    def _discover_records(self, dataset_root: Path) -> list[VideoSequenceRecord]:
        video_exts = set(io.get_supported_video_extensions())
        records: list[VideoSequenceRecord] = []
        for folder in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
            videos = [p for p in folder.iterdir() if p.suffix.lower() in video_exts]
            jsons = [p for p in folder.iterdir() if p.suffix.lower() == ".json"]
            if len(videos) == 0 or len(jsons) == 0:
                continue
            records.append(
                VideoSequenceRecord(
                    root=folder,
                    video_path=videos[0],
                    json_path=jsons[0],
                )
            )
        return records

    def _load_sequence(self, record: VideoSequenceRecord) -> tuple[list[np.ndarray], dict]:
        if record.root in self._cache:
            return self._cache[record.root]

        with record.json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        reader = iio.get_reader(record.video_path)
        frames = [frame[..., :3] for frame in reader]
        reader.close()

        if "c2ws" not in meta:
            raise KeyError(f"Missing c2ws in {record.json_path}")
        if len(meta["c2ws"]) != len(frames):
            raise ValueError(
                f"Frames/c2ws mismatch in {record.root}: {len(frames)} vs {len(meta['c2ws'])}"
            )

        self._cache[record.root] = (frames, meta)
        return frames, meta

    def _sample_pair_indices(self, c2ws: torch.Tensor) -> tuple[int, int]:
        num_frames = c2ws.shape[0]
        for _ in range(64):
            src = random.randrange(num_frames)
            tgt = random.randrange(num_frames)
            if src == tgt:
                continue
            src_t = c2ws[src, :3, 3]
            tgt_t = c2ws[tgt, :3, 3]
            distance = torch.linalg.norm(src_t - tgt_t).item()
            if self.min_view_distance <= distance <= self.max_view_distance:
                return src, tgt
        src = random.randrange(num_frames)
        tgt = (src + random.randrange(1, num_frames)) % num_frames
        return src, tgt

    def __len__(self) -> int:
        return self.max_samples

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        record = random.choice(self.records)
        frames, meta = self._load_sequence(record)

        c2ws = torch.tensor(meta["c2ws"], dtype=torch.float32)
        src_idx, tgt_idx = self._sample_pair_indices(c2ws)

        src_np = frames[src_idx]
        tgt_np = frames[tgt_idx]

        src = torch.from_numpy(src_np).float().permute(2, 0, 1) / 255.0
        tgt = torch.from_numpy(tgt_np).float().permute(2, 0, 1) / 255.0

        fl_x = float(meta["fl_x"])
        fl_y = float(meta["fl_y"])
        cx = float(meta["cx"])
        cy = float(meta["cy"])
        intrinsics = torch.tensor(
            [
                [fl_x, 0.0, cx, 0.0],
                [0.0, fl_y, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        return {
            "src_image": src,
            "tgt_image": tgt,
            "src_c2w": c2ws[src_idx],
            "tgt_c2w": c2ws[tgt_idx],
            "src_w2c": torch.linalg.inv(c2ws[src_idx]),
            "tgt_w2c": torch.linalg.inv(c2ws[tgt_idx]),
            "src_intrinsics": intrinsics,
            "tgt_intrinsics": intrinsics,
            "src_idx": torch.tensor(src_idx),
            "tgt_idx": torch.tensor(tgt_idx),
        }
