# GIM (Geometric Initialization Module)

Provides an initial camera pose estimate between two frames, combining
semantic features, metric depth, semi-dense matching, and PnP solving.

## Submodules

| File | Purpose | External dependency |
|---|---|---|
| `feature_extractor.py` | DINOv2 patch-level feature extraction | `torch.hub` (auto-downloaded) |
| `depth_estimator.py` | Metric3D metric depth estimation | `torch.hub` (auto-downloaded) |
| `matcher.py` | EfficientLoFTR semi-dense matching | Manual clone + manual weight download |
| `pose_solver.py` | Backprojects matches to 3D, solves PnP | OpenCV only |
| `feature_fusion.py` | Projects DINOv2 features + G_t to a shared dim and fuses them into F_t | None |
| `GIM_pipeline.py` | `GIMPipeline`: wires all of the above into a single `run()` call | None |
| `device_utils.py` | Auto-detects CUDA availability, falls back to CPU | None |

## Pipeline (how the 4 modules connect)
frame_t (reference), frame_t+1 (current)
-> EfficientLoFTR: get matched 2D keypoints (mkpts0, mkpts1) and per-match
geometric feature G_t (captured via a forward hook on fine_matching)
-> Metric3D: get depth map for frame_t
-> DINOv2: sample image features at each match location (mkpts0) in frame_t
-> FeatureFusion: project G_t (64-dim) and DINOv2 features (384-dim) to a
shared dim and combine -> F_t (matching feature, per match)
-> PoseSolver: backproject mkpts0 to 3D using depth map + camera intrinsics,
then solvePnP with mkpts1 -> initial pose T_t (R, t) of frame_t+1
relative to frame_t

`GIMPipeline.run()` returns a dict with `F_t`, `p_t` (mkpts1), `T_t` (R, t),
`pose_success`, and `num_inliers`. `F_t` and `p_t` are intended as input to
the downstream GMM module; `T_t` is intended as input to the BA layer
(see MambaVO paper Figure 1).

## Input formats

All image-consuming modules (`feature_extractor.py`, `depth_estimator.py`,
`matcher.py`) accept either:

- a file path (`str`) -- convenient for testing and single-image inspection
- a preloaded tensor `[N, 3, H, W]` in `[0, 1]` range -- matches the output of
  a training dataloader's `ToTensor()`

Both paths run the same resize/pad/normalize logic internally, so results are
equivalent up to small interpolation differences (see Known issues).
Camera intrinsics are always passed as a `(3, 3)` matrix `K`; modules that
need only the focal length read it from `K[0, 0]`.

## Setup

### 1. DINOv2 / Metric3D
No manual setup needed -- `torch.hub.load(...)` downloads weights automatically
on first run.

### 2. EfficientLoFTR (manual setup required)

Clone the official repo into `external/` (not tracked by git, see `.gitignore`):
```bash
mkdir -p external
git clone https://github.com/zju3dv/EfficientLoFTR.git external/EfficientLoFTR
```

Download the pretrained checkpoint manually (not available via torch.hub):
1. Open https://drive.google.com/drive/folders/1GOw6iVqsB-f1vmG6rNmdCcgwfB4VZ7_Q
2. Download `eloftr_outdoor.ckpt`
3. Place it at `external/EfficientLoFTR/weights/eloftr_outdoor.ckpt`

Required pip packages for EfficientLoFTR: `loguru`, `joblib`, `pytorch_lightning`.

## Known issues / gotchas

- **Metric3D CPU patch**: the official Metric3D source hardcodes `device="cuda"`
  inside `RAFTDepthNormalDPTDecoder5.py`'s `get_bins()` function
  (in the local `torch.hub` cache, e.g.
  `~/.cache/torch/hub/yvanyin_metric3d_main/mono/model/decode_heads/RAFTDepthNormalDPTDecoder5.py`).
  On a CPU-only machine (no NVIDIA GPU), this line must be manually changed to
  `device="cpu"`, or the model will crash with
  `AssertionError: Torch not compiled with CUDA enabled`.
  **GPU users do not need this patch** -- `torch.cuda.is_available()` will be
  True and the hardcoded `"cuda"` will work as-is.

- **EfficientLoFTR `reparameter()` step**: must be called after loading the
  checkpoint (`matcher = reparameter(matcher)`), or matching quality will be
  poor. This is required regardless of CPU/GPU.

- **G_t extraction**: EfficientLoFTR does not expose per-match fine-level
  features (G_t) in its output dict by default. `matcher.py` captures them via
  a `register_forward_hook` on `self.matcher.fine_matching`, avoiding any edit
  to the external repo's source. G_t is taken as the center of each match's
  local window feature, shape `[M, 64]`.

- **FeatureFusion is untrained**: `feature_fusion.py`'s two `nn.Linear`
  projection layers are randomly initialized and have not been trained.
  The pipeline is verified to be shape- and value-sane (no NaN/inf, correct
  dimensions), but the fused F_t values themselves are not yet meaningful.
  Training this layer (likely jointly with GMM) is still pending.

- **Camera intrinsics (`K` / `fx`)**: `depth_estimator.py` and `pose_solver.py`
  both require real camera intrinsics for metrically accurate results.
  Current test scripts use a placeholder `fx=500.0` for pipeline validation
  only -- replace with the real intrinsics from the dataset (TartanAir / EuRoC)
  before using outputs for anything beyond a smoke test.


## Tests

Each submodule has a corresponding test in `tests/`, validated by matching an
image against itself (self-match sanity check): matching confidence should be
near 1.0, and the solved pose should be near-identity (R ~= I, t ~= 0).

```bash
python mambavo/gim/tests/test_feature_extractor.py
python mambavo/gim/tests/test_depth_estimator.py
python mambavo/gim/tests/test_matcher.py
python mambavo/gim/tests/test_pose_solver.py
python mambavo/gim/tests/test_GIM_pipeline.py
python mambavo/gim/tests/test_tensor_input.py
```