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
from PIL import Image

from sharp.cli.predict import DEFAULT_MODEL_URL, predict_image
from sharp.models import PredictorParams, create_predictor
from sharp.utils.gaussians import Gaussians3D, save_ply
from sharp.utils.io import load_rgb, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Input image A")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional SHARP checkpoint path. If omitted, download default.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--segmentation-model",
        default="facebook/mask2former-swin-large-coco-panoptic",
        help="Transformers segmentation model id.",
    )
    parser.add_argument(
        "--inpaint-model",
        default="stabilityai/stable-diffusion-2-inpainting",
        help="Diffusers inpainting model id.",
    )
    parser.add_argument(
        "--prompt",
        default="clean realistic background, high detail, consistent lighting",
        help="Inpainting prompt.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device == "mps" and not torch.mps.is_available():
        return torch.device("cpu")
    return torch.device(device)


def load_sharp_predictor(device: torch.device, checkpoint: Path | None):
    if checkpoint is None:
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        state_dict = torch.load(checkpoint, weights_only=True)

    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)
    return predictor


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
    mask_pil = Image.fromarray((fg_mask.astype(np.uint8) * 255), mode="L")

    result = pipe(
        prompt=prompt,
        image=image_pil,
        mask_image=mask_pil,
        guidance_scale=7.5,
        num_inference_steps=40,
    ).images[0]
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

    def visible_subset(g: Gaussians3D) -> torch.Tensor:
        uv, valid = project_to_pixels(g, f_px=f_px, height=h, width=w)
        keep = np.zeros((g.mean_vectors.shape[1],), dtype=bool)
        if np.any(valid):
            uv_valid = uv[valid]
            u_int = np.clip(np.round(uv_valid[:, 0]).astype(int), 0, w - 1)
            v_int = np.clip(np.round(uv_valid[:, 1]).astype(int), 0, h - 1)
            keep[valid] = known_bg_mask[v_int, u_int]
        keep_t = torch.from_numpy(keep).to(g.mean_vectors.device)
        return g.mean_vectors[0, keep_t, :]

    ref_pts = visible_subset(g2_ref)
    new_pts = visible_subset(g2_new)
    if ref_pts.shape[0] < 32 or new_pts.shape[0] < 32:
        return g2_new

    ref_center = ref_pts.mean(dim=0)
    new_center = new_pts.mean(dim=0)
    ref_scale = ref_pts.std(dim=0).mean().clamp(min=1e-6)
    new_scale = new_pts.std(dim=0).mean().clamp(min=1e-6)
    scale = (ref_scale / new_scale).detach()

    aligned_means = (g2_new.mean_vectors - new_center) * scale + ref_center
    aligned_scales = g2_new.singular_values * scale

    return Gaussians3D(
        mean_vectors=aligned_means,
        singular_values=aligned_scales,
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    image_a, icc_profile, f_px = load_rgb(args.image)
    height, width = image_a.shape[:2]

    fg_mask = segment_foreground_person(image=image_a, model_id=args.segmentation_model, device=device)
    bg_mask = ~fg_mask

    predictor = load_sharp_predictor(device=device, checkpoint=args.checkpoint)
    gaussians_a = predict_image(predictor, image_a, f_px=f_px, device=device)

    g1, g2 = split_gaussians_by_mask(gaussians_a, fg_mask, f_px=f_px, image_shape=(height, width))

    inpainted_bg = inpaint_background(
        image=image_a,
        fg_mask=fg_mask,
        model_id=args.inpaint_model,
        prompt=args.prompt,
        device=device,
    )
    g2_prime = predict_image(predictor, inpainted_bg, f_px=f_px, device=device)
    g2_prime_aligned = align_background_gaussians(
        g2_ref=g2,
        g2_new=g2_prime,
        known_bg_mask=bg_mask,
        f_px=f_px,
        image_shape=(height, width),
    )

    merged = concat_gaussians(g1, g2_prime_aligned)

    save_image((fg_mask.astype(np.uint8) * 255), args.output_dir / "mask_fg.png", icc_profile=None)
    save_image((bg_mask.astype(np.uint8) * 255), args.output_dir / "mask_bg.png", icc_profile=None)
    save_image(inpainted_bg, args.output_dir / "background_inpainted.png", icc_profile=icc_profile)

    save_ply(g1, f_px=f_px, image_shape=(height, width), path=args.output_dir / "G1_foreground.ply")
    save_ply(g2, f_px=f_px, image_shape=(height, width), path=args.output_dir / "G2_background.ply")
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

    print("Done. Generated:")
    print(f"- {args.output_dir / 'mask_fg.png'}")
    print(f"- {args.output_dir / 'mask_bg.png'}")
    print(f"- {args.output_dir / 'background_inpainted.png'}")
    print(f"- {args.output_dir / 'G1_foreground.ply'}")
    print(f"- {args.output_dir / 'G2_background.ply'}")
    print(f"- {args.output_dir / 'G2_prime_aligned.ply'}")
    print(f"- {args.output_dir / 'G1_plus_G2_prime_aligned.ply'}")


if __name__ == "__main__":
    main()
