# MambaVO Reproduction

Reproduction of MambaVO (CVPR 2025) for the FAU "Reproducing Research Results" course.

**Paper:** MambaVO: Deep Visual Odometry Based on Sequential Matching Refinement and Training Smoothing

## Team
- Xi Cao: Environment setup, TAP (Trending Aware Penalty), training
- Wenwen: GIM (Geometric Initialization Module)
- Jathursajan: GMM (Geometric Mamba Module)
- Mohammed: System Integration, data loading, training, evaluation

## Environment Setup

```bash
pip install torch==2.5.0+cu118 torchvision==0.20.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r mambavo_requirements.txt
```

## Reproduction Targets
From the MambaVO paper:
- Comparison of mean absolute trajectory errors between models and datasets (Tables 2 and 3)
- Analysis matching (Table 5)
- Ablation study (Table 7)

### Data Loader

Dataset intrinsics:
- TartanAir: `[320.0, 320.0, 320.0, 240.0]`
- EuRoC: `[458.654, 457.296, 367.215, 248.375]`

## Integration and Training

`mambavo/pfg/pfg.py` holds the Point-Frame Graph that puts a point id to each match, keeps a sliding window of frames, and stores GMM's pixel refinements per edge.

`mambavo/ba/ba_layer.py` wraps DPVO's differentiable BA (see `SETUP.md`)

`mambavo/losses.py` computes the pose loss and the matching loss then combines them through TAP

`mambavo/model.py` defines `MambaVO`, combining all the modules together and returns the predicted poses.

`train.py` trains `MambaVO` on TartanAir clips by running:

```bash
python train.py --datapath [tartanAIR-path]
```

## Ablation

Pass `--ablate <name>` to disable one component at a time

- GIM: `context_feature`, `geometric_feature`, `pnp`
- GMM: `history_fusion`, `mamba`, `gru`
- TAP: `grad_weight`, `history_balance`

example:
```bash
python train.py --datapath [tartanAIR-path] --ablate mamba
```

Each ablation run writes to its own TensorBoard log dir (`runs/mambavo_ablate_<name>`)

## Evaluation

Run the trained checkpoint against EuRoC and KITTI:

```bash
python evaluate.py --dataset euroc --datapath EuRoC --checkpoint checkpoint.pt
python evaluate.py --dataset kitti --datapath KITTI --checkpoint checkpoint.pt
python evaluate_matching.py --datapath EuRoC
```

Each script prints per-sequence
results and a final summary block; the tables below come straight from
that output.

The ablation study needs eight separately trained ablation
checkpoints, so we just take the averages done the comparison of mean absolute trajectory errors between models and datasets KITTI and EuRoc.

