# Sharp Monocular View Synthesis in Less Than a Second

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://apple.github.io/ml-sharp/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.10685-b31b1b.svg)](https://arxiv.org/abs/2512.10685)

This software project accompanies the research paper: _Sharp Monocular View Synthesis in Less Than a Second_
by _Lars Mescheder, Wei Dong, Shiwei Li, Xuyang Bai, Marcel Santos, Peiyun Hu, Bruno Lecouat, Mingmin Zhen, Amaël Delaunoy,
Tian Fang, Yanghai Tsin, Stephan Richter and Vladlen Koltun_.

![](data/teaser.jpg)

We present SHARP, an approach to photorealistic view synthesis from a single image. Given a single photograph, SHARP regresses the parameters of a 3D Gaussian representation of the depicted scene. This is done in less than a second on a standard GPU via a single feedforward pass through a neural network. The 3D Gaussian representation produced by SHARP can then be rendered in real time, yielding high-resolution photorealistic images for nearby views. The representation is metric, with absolute scale, supporting metric camera movements. Experimental results demonstrate that SHARP delivers robust zero-shot generalization across datasets. It sets a new state of the art on multiple datasets, reducing LPIPS by 25–34% and DISTS by 21–43% versus the best prior model, while lowering the synthesis time by three orders of magnitude.

## Getting started

We recommend to first create a python environment:

```
conda create -n sharp python=3.13
```

Afterwards, you can install the project using

```
pip install -r requirements.txt
```

To test the installation, run

```
sharp --help
```

## Using the CLI

To run prediction:

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians
```

The model checkpoint will be downloaded automatically on first run and cached locally at `~/.cache/torch/hub/checkpoints/`.

Alternatively, you can download the model directly:

```
wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt
```

To use a manually downloaded checkpoint, specify it with the `-c` flag:

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians -c sharp_2572gikvuh.pt
```

If you want to run inference with fine-tuning checkpoints saved by `sharp finetune` / `sharp finetune-ddp`
while tolerating architecture/key differences, use:

```
sharp predict-finetune -i /path/to/input/images -o /path/to/output/gaussians -c /path/to/finetune_checkpoint.pt
```

`predict-finetune` uses `delta_decoder.*` by default so finetuned deltas participate in inference.
If you want baseline comparison behavior, pass `--ignore-delta`.

The results will be 3D gaussian splats (3DGS) in the output folder. The 3DGS `.ply` files are compatible to various public 3DGS renderers. We follow the OpenCV coordinate convention (x right, y down, z forward). The 3DGS scene center is roughly at (0, 0, +z). When dealing with 3rdparty renderers, please scale and rotate to re-center the scene accordingly.

### Rendering trajectories (CUDA GPU only)

Additionally you can render videos with a camera trajectory. While the gaussians prediction works for all CPU, CUDA, and MPS, rendering videos via the `--render` option currently requires a CUDA GPU. The gsplat renderer takes a while to initialize at the first launch.

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians --render

# Or from the intermediate gaussians:
sharp render -i /path/to/output/gaussians -o /path/to/output/renderings

# Increase/decrease camera motion range:
sharp render -i /path/to/output/gaussians -o /path/to/output/renderings --trajectory-scale 1.4
```


## Fine-tuning on posed videos

This repository now also includes a scene fine-tuning entrypoint for either a **single posed video** or a **directory of many scene folders**. The training loop follows the requested pipeline:

1. Randomly sample an input frame.
2. Run the pretrained SHARP predictor to obtain Gaussians in NDC space.
3. Map these Gaussians to world space with the input-frame intrinsics/extrinsics.
4. Randomly sample a target frame with a configurable frame-distance range.
5. Render the world-space Gaussians in the target camera and optimize the Gaussian Decoder.

Expected multi-scene input layout:

```
/path/to/data_root/
  scene_000/
    video.mp4
    camera_params.json
  scene_001/
    clip.mp4
    camera_params_old.json
    camera_params.json
```

Each pose JSON file should contain `fl_x`, `fl_y`, `cx`, `cy`, and a `c2ws` array with one 4x4 camera-to-world matrix per video frame. When multiple JSON files exist in one scene folder, the loader prefers `camera_params.json` (and ignores legacy `camera_params_old.json` when possible). A typical multi-scene command is:

```
sharp finetune \
  --data-root /path/to/data_root \
  --output-dir /path/to/output_dir \
  --checkpoint-path /path/to/sharp_2572gikvuh.pt \
  --min-frame-distance 4 \
  --max-frame-distance 48
```

If you only want to fine-tune on one video, you can still pass `--video-path` and `--pose-path`.

If mask-focused improvement is too weak, increase `--invisible-mask-dilation-px` (expand supervised mask area) and/or `--invisible-loss-boost` (upweight mask-region reconstruction losses).
The `--loss-border-ratio` controls the center-region crop before intersecting with the invisible-mask region. Set `--loss-border-ratio 0.0` to use the full image ∩ invisible-mask intersection.

The command keeps the network input at `1536x1536`, but it also preserves each frame's original resolution for debugging renders. Fine-tuning updates the predictor `feature_model` together with a mask-guided Gaussian refiner, while the other predictor modules stay frozen. The target-view supervision now detects regions that are visible in the target view but invisible from the source view, predicts additive Gaussian deltas for those gated Gaussians, and re-renders the refined target view. Each visualization step saves 10 images: source/target frames at training and original resolution, the target-invisible mask, the masked target-region render, source-view Gaussian renders at training and original resolution, and target-view Gaussian renders at training and original resolution. A `step_000000.*` visualization is still saved at `epoch=0, step=0` before that iteration's optimizer update. By default, fine-tuning uses `--low-pass-filter-eps 0.0` so the debug renders match the standard SHARP render path instead of adding extra smoothing. Fine-tuning currently requires CUDA because the training loop uses differentiable `gsplat` rendering.

## Evaluation

Please refer to the paper for both quantitative and qualitative evaluations.
Additionally, please check out this [qualitative examples page](https://apple.github.io/ml-sharp/) containing several video comparisons against related work.

## Citation

If you find our work useful, please cite the following paper:

```bibtex
@inproceedings{Sharp2025:arxiv,
  title      = {Sharp Monocular View Synthesis in Less Than a Second},
  author     = {Lars Mescheder and Wei Dong and Shiwei Li and Xuyang Bai and Marcel Santos and Peiyun Hu and Bruno Lecouat and Mingmin Zhen and Ama\"{e}l Delaunoy and Tian Fang and Yanghai Tsin and Stephan R. Richter and Vladlen Koltun},
  journal    = {arXiv preprint arXiv:2512.10685},
  year       = {2025},
  url        = {https://arxiv.org/abs/2512.10685},
}
```

## Acknowledgements

Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details.

## License

Please check out the repository [LICENSE](LICENSE) before using the provided code and
[LICENSE_MODEL](LICENSE_MODEL) for the released models.
