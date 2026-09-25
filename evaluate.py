# Evaluates a MambaVO checkpoint on EuRoC/KITTI, reproducing paper Tables 2/3 (and eventually table 7 for all the models)
# Usage: 'python evaluate.py --dataset kitti --datapath EuRoC/KITTI --checkpoint checkpoints/step_10000.pt'

#Adapted from: [Teed, Lipson, Deng, "Deep Patch Visual Odometry," NeurIPS 36, 2023 — github.com/princeton-vl/DPVO]

import argparse
import os
import re
import subprocess
import torch
from mambavo.inference import MambaVOTracker

EUROC_SEQUENCES = ["MH01", "MH02", "MH03", "MH04", "MH05", "V101", "V102", "V103", "V201", "V202", "V203"]
KITTI_SEQUENCES = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]


def save_tum(path, frame_indices, poses):
    with open(path, "w") as f:
        for idx, pose in zip(frame_indices, poses):
            vals = [float(v) for v in pose]
            f.write(f"{idx} " + " ".join(f"{v:.9f}" for v in vals) + "\n")


def run_evo_ape(gt_path, est_path):
    result = subprocess.run(["evo_ape", "tum", gt_path, est_path, "-as"], capture_output=True, text=True)
    out = result.stdout + result.stderr
    mean = rmse = None
    for line in out.splitlines():
        line = line.strip()
        m = re.match(r"mean\s+([\d.]+)", line)
        if m: mean = float(m.group(1))
        m = re.match(r"rmse\s+([\d.]+)", line)
        if m: rmse = float(m.group(1))
    return mean, rmse, out


def evaluate_sequence(tracker, dataset, gt_poses, tmp_dir, name):
    trajectory = tracker.track_sequence(dataset)
    if not trajectory:
        return None, None, "no keyframes produced"
    frame_indices = sorted(trajectory.keys())
    est_poses = [trajectory[i].cpu().numpy() for i in frame_indices]
    est_path = os.path.join(tmp_dir, f"{name}_est.tum")
    gt_path = os.path.join(tmp_dir, f"{name}_gt.tum")
    save_tum(est_path, frame_indices, est_poses)
    save_tum(gt_path, frame_indices, [gt_poses[i].numpy() for i in frame_indices])
    return run_evo_ape(gt_path, est_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["euroc", "kitti"], required=True)
    p.add_argument("--datapath", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--loftr_repo_path", default="external/EfficientLoFTR")
    p.add_argument("--loftr_weights_path", default="external/EfficientLoFTR/weights/eloftr_outdoor.ckpt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="eval_tmp")
    p.add_argument("--sequences", nargs="*", default=None)
    p.add_argument("--repeat", type=int, default=1)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tracker = MambaVOTracker(args.loftr_repo_path, args.loftr_weights_path, device=args.device)
    state_dict = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    tracker.load_checkpoint(state_dict)

    if args.dataset == "euroc":
        from mambavo.data_readers.euroc import EuRoC as Reader
        sequences = args.sequences or EUROC_SEQUENCES
    else:
        from mambavo.data_readers.kitti import KITTI as Reader
        sequences = args.sequences or KITTI_SEQUENCES

    results = {}
    for seq in sequences:
        print(f"=== {args.dataset.upper()} {seq} ===", flush=True)
        try:
            dataset = Reader(args.datapath, seq)
        except Exception as e:
            print(f"  SKIP (failed to load: {e})", flush=True); results[seq] = None; continue
        gt_poses = dataset.get_all_poses()
        if gt_poses is None:
            print("  SKIP (no ground truth)", flush=True); results[seq] = None; continue

        runs = []
        for r in range(args.repeat):
            mean, rmse, raw = evaluate_sequence(tracker, dataset, gt_poses, args.out_dir, f"{seq}_r{r}")
            print(f"  run {r}: ATE mean={mean} rmse={rmse}", flush=True)
            if mean is not None: runs.append(mean)
        results[seq] = sum(runs) / len(runs) if runs else None

    avg = [v for v in results.values() if v is not None]
    print(f"\n=== Summary: ATE[m] mean, {args.dataset} (w.o. loop, {args.repeat}x per sequence) ===", flush=True)
    for seq, v in results.items():
        print(f"  {seq}: {v if v is not None else 'X'}", flush=True)
    if avg:
        print(f"  AVG: {sum(avg)/len(avg):.4f}", flush=True)


if __name__ == "__main__":
    main()
