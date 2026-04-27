#!/usr/bin/env python3
"""Foreground/background Gaussian decomposition + inpainting pipeline for SHARP.

Workflow:
1) Segment image A into foreground A1 (person-prioritized) and background A2.
2) Predict Gaussians from A with SHARP and split into G1 (foreground) / G2 (background).
3) Inpaint the foreground hole in A2.
4) Predict Gaussians from inpainted background => G2'.
5) Align G2' to G2 on known background overlap and export combined G1+G2'.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sharp.cli.predict import DEFAULT_MODEL_URL
from sharp.cli.render import render_gaussians
from sharp.models import PredictorParams, create_predictor
from sharp.utils import camera
from sharp.utils.gaussians import (
    Gaussians3D,
    compose_covariance_matrices,
    decompose_covariance_matrices,
    save_ply,
)
from sharp.utils.io import load_rgb, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Input image A")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder")
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="Optional SHARP checkpoint path."
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--segmentation-model",
        default="facebook/mask2former-swin-large-coco-panoptic",
        help="Transformers segmentation model id.",
    )
    parser.add_argument(
        "--inpaint-model",
        default="sd2-community/stable-diffusion-2-inpainting",
        help="Diffusers inpainting model id.",
    )
    parser.add_argument(
        "--prompt",
        default="clean realistic background, high detail, consistent lighting",
        help="Inpainting prompt.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render output gaussians with SHARP built-in camera trajectory (CUDA only).",
    )
    parser.add_argument(
        "--trajectory-spatial-scale",
        type=float,
        default=1.0,
        help="Scale factor for trajectory spatial amplitude (lateral + zoom motion).",
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device == "mps" and not torch.mps.is_available():
        return torch.device("cpu")
    return torch.device(device)


def load_sharp_predictor(
    device: torch.device,
    checkpoint: Path | None,
):
    if checkpoint is None:
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        state_dict = torch.load(checkpoint, weights_only=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)
    return predictor


def _ensure_batched_gaussians(gaussians: Gaussians3D) -> Gaussians3D:
    if gaussians.mean_vectors.ndim == 2:
        return Gaussians3D(
            mean_vectors=gaussians.mean_vectors.unsqueeze(0),
            singular_values=gaussians.singular_values.unsqueeze(0),
            quaternions=gaussians.quaternions.unsqueeze(0),
            colors=gaussians.colors.unsqueeze(0),
            opacities=gaussians.opacities.unsqueeze(0),
        )
    return gaussians


def _robust_unproject_gaussians(
    gaussians_ndc: Gaussians3D,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
) -> Gaussians3D:
    gaussians_ndc = _ensure_batched_gaussians(gaussians_ndc)
    width, height = image_shape
    device = intrinsics.device
    ndc_matrix = torch.tensor(
        [
            [2.0 / width, 0.0, -1.0, 0.0],
            [0.0, 2.0 / height, -1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=intrinsics.dtype,
    )
    transform = torch.linalg.inv(ndc_matrix @ intrinsics)[:3]  # 3x4
    linear = transform[:, :3]  # 3x3
    offset = transform[:, 3]  # 3

    means = gaussians_ndc.mean_vectors @ linear.T + offset
    cov = compose_covariance_matrices(gaussians_ndc.quaternions, gaussians_ndc.singular_values)
    cov = linear[None, None] @ cov @ linear.T[None, None]
    quaternions, singular_values = decompose_covariance_matrices(cov)

    return Gaussians3D(
        mean_vectors=means,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=gaussians_ndc.colors,
        opacities=gaussians_ndc.opacities,
    )


@torch.no_grad()
def predict_gaussians_from_image(
    predictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
) -> Gaussians3D:
    internal_shape = (1536, 1536)
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width], dtype=torch.float32, device=device)
    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )
    gaussians_ndc = predictor(image_resized_pt, disparity_factor)
    gaussians_ndc = _ensure_batched_gaussians(gaussians_ndc)

    intrinsics = torch.tensor(
        [
            [f_px, 0, width / 2, 0],
            [0, f_px, height / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height
    return _robust_unproject_gaussians(gaussians_ndc, intrinsics_resized, internal_shape)


def segment_foreground_person(image: np.ndarray, model_id: str, device: torch.device) -> np.ndarray:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Missing transformers dependency. Install with: pip install transformers"
        ) from exc

    hf_device = 0 if device.type == "cuda" else -1
    segmenter = pipeline("image-segmentation", model=model_id, device=hf_device)

    outputs = segmenter(Image.fromarray(image))
    if not outputs:
        raise RuntimeError("Segmentation model returned no masks.")

    person_masks = [entry for entry in outputs if str(entry.get("label", "")).lower() == "person"]
    selected = person_masks if person_masks else [max(outputs, key=lambda x: x.get("score", 0.0))]

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for entry in selected:
        raw_mask = np.array(entry["mask"].resize((w, h), Image.NEAREST), dtype=np.uint8)
        mask |= raw_mask > 0

    if not np.any(mask):
        raise RuntimeError("Failed to produce a non-empty foreground mask.")
    return mask


def inpaint_background(
    image: np.ndarray,
    fg_mask: np.ndarray,
    model_id: str,
    prompt: str,
    device: torch.device,
) -> np.ndarray:
    try:
        from diffusers import AutoPipelineForInpainting
    except ImportError as exc:
        raise RuntimeError(
            "Missing diffusers dependency. Install with: pip install diffusers accelerate"
        ) from exc

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    image_pil = Image.fromarray(image)
    width, height = image_pil.size
    mask_pil = Image.fromarray((fg_mask.astype(np.uint8) * 255), mode="L")

    result = pipe(
        prompt=prompt,
        image=image_pil,
        mask_image=mask_pil,
        guidance_scale=7.5,
        num_inference_steps=40,
    ).images[0]
    if result.size != (width, height):
        result = result.resize((width, height), Image.BICUBIC)
    return np.array(result)


def project_to_pixels(gaussians: Gaussians3D, f_px: float, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    means = gaussians.mean_vectors[0].detach().cpu().numpy()
    z = np.clip(means[:, 2], 1e-6, None)
    u = (f_px * means[:, 0] / z) + (width / 2.0)
    v = (f_px * means[:, 1] / z) + (height / 2.0)
    valid = (means[:, 2] > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.stack([u, v], axis=1), valid


def split_gaussians_by_mask(
    gaussians: Gaussians3D,
    mask_fg: np.ndarray,
    f_px: float,
    image_shape: tuple[int, int],
) -> tuple[Gaussians3D, Gaussians3D]:
    h, w = image_shape
    uv, valid = project_to_pixels(gaussians, f_px=f_px, height=h, width=w)

    fg_idx = np.zeros((gaussians.mean_vectors.shape[1],), dtype=bool)
    if np.any(valid):
        uv_valid = uv[valid]
        u_int = np.clip(np.round(uv_valid[:, 0]).astype(int), 0, w - 1)
        v_int = np.clip(np.round(uv_valid[:, 1]).astype(int), 0, h - 1)
        fg_idx[valid] = mask_fg[v_int, u_int]

    bg_idx = ~fg_idx

    def select(index: np.ndarray) -> Gaussians3D:
        idx = torch.from_numpy(index).to(gaussians.mean_vectors.device)
        return Gaussians3D(
            mean_vectors=gaussians.mean_vectors[:, idx, :],
            singular_values=gaussians.singular_values[:, idx, :],
            quaternions=gaussians.quaternions[:, idx, :],
            colors=gaussians.colors[:, idx, :],
            opacities=gaussians.opacities[:, idx],
        )

    return select(fg_idx), select(bg_idx)


def align_background_gaussians(
    g2_ref: Gaussians3D,
    g2_new: Gaussians3D,
    known_bg_mask: np.ndarray,
    f_px: float,
    image_shape: tuple[int, int],
) -> Gaussians3D:
    h, w = image_shape
    uv_ref, valid_ref = project_to_pixels(g2_ref, f_px=f_px, height=h, width=w)
    uv_new, valid_new = project_to_pixels(g2_new, f_px=f_px, height=h, width=w)
    if not np.any(valid_ref) or not np.any(valid_new):
        return g2_new

    def collect_pixel_means(g: Gaussians3D, uv: np.ndarray, valid: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
        means = g.mean_vectors[0].detach().cpu().numpy()
        u = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)

        accum: dict[tuple[int, int], list[np.ndarray]] = {}
        for idx in np.where(valid)[0]:
            key = (v[idx], u[idx])
            if not known_bg_mask[key[0], key[1]]:
                continue
            accum.setdefault(key, []).append(means[idx])

        reduced: dict[tuple[int, int], np.ndarray] = {}
        for key, pts in accum.items():
            reduced[key] = np.mean(np.stack(pts, axis=0), axis=0)
        return reduced

    ref_map = collect_pixel_means(g2_ref, uv_ref, valid_ref)
    new_map = collect_pixel_means(g2_new, uv_new, valid_new)
    common_keys = list(set(ref_map.keys()) & set(new_map.keys()))
    if len(common_keys) < 64:
        return g2_new

    deltas = np.stack([ref_map[k] - new_map[k] for k in common_keys], axis=0)
    translation = np.median(deltas, axis=0)

    # Clamp over-aggressive translation to avoid breaking well-aligned results.
    ref_depth = g2_ref.mean_vectors[0, :, 2].detach().cpu().numpy()
    depth_scale = max(float(np.median(ref_depth[ref_depth > 1e-6])), 1e-3)
    max_shift = 0.05 * depth_scale
    shift_norm = float(np.linalg.norm(translation))
    if shift_norm > max_shift:
        translation = translation * (max_shift / max(shift_norm, 1e-8))

    translation_t = torch.tensor(
        translation,
        dtype=g2_new.mean_vectors.dtype,
        device=g2_new.mean_vectors.device,
    )
    aligned_means = g2_new.mean_vectors + translation_t[None, None, :]

    return Gaussians3D(
        mean_vectors=aligned_means,
        singular_values=g2_new.singular_values,
        quaternions=g2_new.quaternions,
        colors=g2_new.colors,
        opacities=g2_new.opacities,
    )


def concat_gaussians(g1: Gaussians3D, g2: Gaussians3D) -> Gaussians3D:
    return Gaussians3D(
        mean_vectors=torch.cat([g1.mean_vectors, g2.mean_vectors], dim=1),
        singular_values=torch.cat([g1.singular_values, g2.singular_values], dim=1),
        quaternions=torch.cat([g1.quaternions, g2.quaternions], dim=1),
        colors=torch.cat([g1.colors, g2.colors], dim=1),
        opacities=torch.cat([g1.opacities, g2.opacities], dim=1),
    )


def render_outputs(
    output_dir: Path,
    f_px: float,
    image_shape: tuple[int, int],
    trajectory_spatial_scale: float,
    gaussian_outputs: dict[str, Gaussians3D],
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("`--render` requires CUDA (same as sharp render).")

    params = camera.TrajectoryParams(
        max_disparity=0.08 * trajectory_spatial_scale,
        max_zoom=0.15 * trajectory_spatial_scale,
    )
    metadata = camera_metadata(f_px=f_px, image_shape=image_shape)

    video_dir = output_dir / "renderings"
    video_dir.mkdir(parents=True, exist_ok=True)
    for name, gaussians in gaussian_outputs.items():
        render_gaussians(
            gaussians=gaussians,
            metadata=metadata,
            params=params,
            output_path=video_dir / f"{name}.mp4",
        )


def camera_metadata(f_px: float, image_shape: tuple[int, int]):
    from sharp.utils.gaussians import SceneMetaData

    height, width = image_shape
    return SceneMetaData(focal_length_px=f_px, resolution_px=(width, height), color_space="linearRGB")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    image_a, icc_profile, f_px = load_rgb(args.image)
    height, width = image_a.shape[:2]

    fg_mask = segment_foreground_person(image=image_a, model_id=args.segmentation_model, device=device)
    bg_mask = ~fg_mask

    predictor = load_sharp_predictor(device=device, checkpoint=args.checkpoint)
    gaussians_a = predict_gaussians_from_image(predictor, image=image_a, f_px=f_px, device=device)

    g1, g2 = split_gaussians_by_mask(gaussians_a, fg_mask, f_px=f_px, image_shape=(height, width))

    inpainted_bg = inpaint_background(
        image=image_a,
        fg_mask=fg_mask,
        model_id=args.inpaint_model,
        prompt=args.prompt,
        device=device,
    )
    g2_prime = predict_gaussians_from_image(predictor, image=inpainted_bg, f_px=f_px, device=device)
    g2_prime_aligned = align_background_gaussians(
        g2_ref=g2,
        g2_new=g2_prime,
        known_bg_mask=bg_mask,
        f_px=f_px,
        image_shape=(height, width),
    )

    merged = concat_gaussians(g1, g2_prime_aligned)

    save_image(image_a, args.output_dir / "input_image.png", icc_profile=icc_profile)
    save_image((fg_mask.astype(np.uint8) * 255), args.output_dir / "mask_fg.png", icc_profile=None)
    save_image((bg_mask.astype(np.uint8) * 255), args.output_dir / "mask_bg.png", icc_profile=None)
    segmented_preview = image_a.copy()
    segmented_preview[~fg_mask] = (segmented_preview[~fg_mask] * 0.25).astype(np.uint8)
    save_image(segmented_preview, args.output_dir / "segmented_preview.png", icc_profile=icc_profile)
    save_image(inpainted_bg, args.output_dir / "background_inpainted.png", icc_profile=icc_profile)

    save_ply(
        gaussians_a,
        f_px=f_px,
        image_shape=(height, width),
        path=args.output_dir / "G0_original_sharp.ply",
    )
    save_ply(g1, f_px=f_px, image_shape=(height, width), path=args.output_dir / "G1_foreground.ply")
    save_ply(g2, f_px=f_px, image_shape=(height, width), path=args.output_dir / "G2_background.ply")
    save_ply(
        g2_prime,
        f_px=f_px,
        image_shape=(height, width),
        path=args.output_dir / "G2_prime_inpainted_raw.ply",
    )
    save_ply(
        g2_prime_aligned,
        f_px=f_px,
        image_shape=(height, width),
        path=args.output_dir / "G2_prime_aligned.ply",
    )
    save_ply(
        merged,
        f_px=f_px,
        image_shape=(height, width),
        path=args.output_dir / "G1_plus_G2_prime_aligned.ply",
    )

    if args.render:
        render_outputs(
            output_dir=args.output_dir,
            f_px=f_px,
            image_shape=(height, width),
            trajectory_spatial_scale=args.trajectory_spatial_scale,
            gaussian_outputs={
                "G0_original_sharp": gaussians_a,
                "G1_foreground": g1,
                "G2_background": g2,
                "G2_prime_inpainted_raw": g2_prime,
                "G2_prime_aligned": g2_prime_aligned,
                "G1_plus_G2_prime_aligned": merged,
            },
        )

    print("Done. Generated:")
    print(f"- {args.output_dir / 'input_image.png'}")
    print(f"- {args.output_dir / 'G0_original_sharp.ply'}")
    print(f"- {args.output_dir / 'mask_fg.png'}")
    print(f"- {args.output_dir / 'mask_bg.png'}")
    print(f"- {args.output_dir / 'segmented_preview.png'}")
    print(f"- {args.output_dir / 'background_inpainted.png'}")
    print(f"- {args.output_dir / 'G1_foreground.ply'}")
    print(f"- {args.output_dir / 'G2_background.ply'}")
    print(f"- {args.output_dir / 'G2_prime_inpainted_raw.ply'}")
    print(f"- {args.output_dir / 'G2_prime_aligned.ply'}")
    print(f"- {args.output_dir / 'G1_plus_G2_prime_aligned.ply'}")
    if args.render:
        print(f"- {args.output_dir / 'renderings'}/*.mp4")


if __name__ == "__main__":
    main()
