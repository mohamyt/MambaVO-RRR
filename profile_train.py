"""Quick profiler: runs a few real training steps and reports where time goes.
Usage:
    python profile_train.py --datapath data/TartanAir --steps 5
"""
import argparse
import sys
import time
import torch
from torch.utils.data import DataLoader
from mambavo.data_readers.tartan import TartanAir
from mambavo.model import MambaVO
from mambavo.losses import pose_loss, match_loss
from mambavo.tap.tap import TrendingAwarePenalty

def build_gt_dicts(poses, disps, intrinsics):
    n = poses.shape[0]
    gt_poses = {i: poses[i] for i in range(n)}
    gt_disps = {i: disps[i] for i in range(n)}
    intr = {i: intrinsics[i] for i in range(n)}
    return gt_poses, gt_disps, intr

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datapath", required=True)
    p.add_argument("--loftr_repo_path", default="external/EfficientLoFTR")
    p.add_argument("--loftr_weights_path", default="external/EfficientLoFTR/weights/eloftr_outdoor.ckpt")
    p.add_argument("--n_frames", type=int, default=4)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    log(f"device: {args.device}")

    t0 = time.time()
    dataset = TartanAir(args.datapath, n_frames=args.n_frames)
    log(f"dataset init done in {time.time() - t0:.1f}s, {len(dataset)} samples")

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    t0 = time.time()
    model = MambaVO(args.loftr_repo_path, args.loftr_weights_path, device=args.device, ablate="none").to(args.device)
    log(f"model init done in {time.time() - t0:.1f}s")

    tap = TrendingAwarePenalty(ablate="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    it = iter(loader)

    # one warmup step outside the profiler (excludes one-time CUDA context / cudnn autotune cost)
    t0 = time.time()
    images, poses, disps, intrinsics = next(it)
    images = images[0].to(args.device); poses = poses[0].to(args.device)
    disps = disps[0].to(args.device); intrinsics = intrinsics[0].to(args.device)
    gt_poses, gt_disps, intr = build_gt_dicts(poses, disps, intrinsics)
    out = model(images, intrinsics)
    l_pose = pose_loss(out["poses"], gt_poses)
    l_match = match_loss(out["pfg"], gt_poses, gt_disps, intr)
    loss = tap.compute_loss(l_pose, l_match, list(model.parameters()))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if args.device == "cuda":
        torch.cuda.synchronize()
    log(f"warmup step done in {time.time() - t0:.1f}s")

    def sync():
        if args.device == "cuda": torch.cuda.synchronize()

    for step in range(args.steps):
        t0 = time.time()
        images, poses, disps, intrinsics = next(it)
        images = images[0].to(args.device); poses = poses[0].to(args.device)
        disps = disps[0].to(args.device); intrinsics = intrinsics[0].to(args.device)
        gt_poses, gt_disps, intr = build_gt_dicts(poses, disps, intrinsics)

        sync(); t_fwd0 = time.time()
        out = model(images, intrinsics)
        sync(); t_fwd1 = time.time()

        l_pose = pose_loss(out["poses"], gt_poses)
        l_match = match_loss(out["pfg"], gt_poses, gt_disps, intr)
        loss = tap.compute_loss(l_pose, l_match, list(model.parameters()))
        sync(); t_loss1 = time.time()

        optimizer.zero_grad()
        loss.backward()
        sync(); t_bwd1 = time.time()
        optimizer.step()
        sync(); t_opt1 = time.time()

        log(f"step {step} done in {time.time() - t0:.1f}s "
            f"(forward={t_fwd1-t_fwd0:.1f}s loss={t_loss1-t_fwd1:.1f}s backward={t_bwd1-t_loss1:.1f}s opt_step={t_opt1-t_bwd1:.1f}s)")

if __name__ == "__main__":
    main()
