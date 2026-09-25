# Reproduces paper Table 5: matching-only pose-estimation AUC on EuRoC.

# uses the raw GIM (EfficientLoFTR) matches for each
# keyframe pair, estimates the relative pose via PoseLib 

# usage: 'python evaluate_matching.py --datapath EuRoC'
#
# Adapted from: [Wang et al., "Efficient LoFTRSemi-Dense Local Feature Matching with Sparse-Like Speed", 
# CVPR 2024, https://arxiv.org/abs/2403.04765] 
import argparse
import numpy as np
import torch
import poselib

from mambavo.gim.GIM_pipeline import GIMPipeline
from mambavo.model import _to_K_matrix
from mambavo.data_readers.euroc import EuRoC

EUROC_SEQUENCES = ["MH01", "MH02", "MH03", "MH04", "MH05", "V101", "V102", "V103", "V201", "V202", "V203"]


def compute_auc(errors, thresholds):
    # standard AUC-over-pose-error-CDF, as used in SuperGlue/LoFTR-style eval code.
    errors = np.sort(np.array(errors))
    n = len(errors)
    recall = (np.arange(n) + 1) / n
    aucs = []
    for t in thresholds:
        last = np.searchsorted(errors, t)
        r = np.r_[0.0, recall[:last], recall[last - 1] if last > 0 else 0.0]
        e = np.r_[0.0, errors[:last], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs


def relative_gt_pose(pose_r, pose_t):
    #convert dataset GT poses to PoseLib's se3 representation, 
    #so compute relative pose and return as rotation matrix and translation vector
    import pypose as pp
    T_r = pp.SE3(torch.as_tensor(pose_r, dtype=torch.float32))
    T_t = pp.SE3(torch.as_tensor(pose_t, dtype=torch.float32))
    T_rel = T_t.Inv() @ T_r
    m = T_rel.matrix().numpy()
    return m[:3, :3], m[:3, 3]


def pose_errors_deg(R_est, t_est, R_gt, t_gt):
    cos_r = np.clip((np.trace(R_est.T @ R_gt) - 1) / 2, -1, 1)
    rot_err = np.degrees(np.arccos(cos_r))
    t_est_n = t_est / (np.linalg.norm(t_est) + 1e-12)
    t_gt_n = t_gt / (np.linalg.norm(t_gt) + 1e-12)
    cos_t = np.clip(np.dot(t_est_n, t_gt_n), -1, 1)
    trans_err = np.degrees(np.arccos(cos_t))
    return max(rot_err, trans_err)


@torch.no_grad()
def evaluate_sequence(gim, dataset, K, W, H, keyframe_parallax_px=30.0, keyframe_max_gap=3, device="cuda"):
    errors = []
    ref_idx = 0
    ref_image, ref_pose, ref_intr = dataset[0]
    ref_image_n = (ref_image / 255.0).unsqueeze(0).to(device)
    frames_since_kf = 0

    for idx in range(1, len(dataset)):
        image, pose, intr = dataset[idx]
        image_n = (image / 255.0).unsqueeze(0).to(device)
        mkpts0, mkpts1, mconf, g_t = gim.matcher.match(ref_image_n, image_n)
        if len(mkpts0) < 8:
            frames_since_kf += 1
            continue
        mkpts0 = np.asarray(mkpts0); mkpts1 = np.asarray(mkpts1)
        parallax = float(np.median(np.linalg.norm(mkpts1 - mkpts0, axis=1)))
        is_keyframe = (parallax > keyframe_parallax_px) or (frames_since_kf >= keyframe_max_gap)
        if not is_keyframe:
            frames_since_kf += 1
            continue

        camera = {"model": "PINHOLE", "width": W, "height": H,
                  "params": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]}
        ransac_opt = {"max_epipolar_error": 1.0}
        try:
            campose, info = poselib.estimate_relative_pose(mkpts0.astype(np.float64), mkpts1.astype(np.float64),
                                                             camera, camera, ransac_opt)
            R_est, t_est = campose.R, campose.t
            R_gt, t_gt = relative_gt_pose(ref_pose.numpy(), pose.numpy())
            errors.append(pose_errors_deg(R_est, t_est, R_gt, t_gt))
        except Exception as e:
            print(f"  [warn] pair ({ref_idx},{idx}) failed: {e}", flush=True)

        ref_idx = idx; ref_image_n = image_n; ref_pose = pose
        frames_since_kf = 0

    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datapath", required=True)
    p.add_argument("--loftr_repo_path", default="external/EfficientLoFTR")
    p.add_argument("--loftr_weights_path", default="external/EfficientLoFTR/weights/eloftr_outdoor.ckpt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sequences", nargs="*", default=None)
    args = p.parse_args()

    gim = GIMPipeline(args.loftr_repo_path, args.loftr_weights_path, device=args.device)

    all_errors = []
    for seq in (args.sequences or EUROC_SEQUENCES):
        print(f"=== {seq} ===", flush=True)
        try:
            dataset = EuRoC(args.datapath, seq)
        except Exception as e:
            print(f"  SKIP ({e})", flush=True); continue
        sample_img, _, sample_intr = dataset[0]
        K = _to_K_matrix(sample_intr)
        H, W = sample_img.shape[1], sample_img.shape[2]
        errors = evaluate_sequence(gim, dataset, K, W, H, device=args.device)
        print(f"  {len(errors)} keyframe pairs, mean_err={np.mean(errors) if errors else float('nan'):.2f} deg", flush=True)
        all_errors.extend(errors)

    thresholds = [1, 2, 5, 10]
    aucs = compute_auc(all_errors, thresholds) if all_errors else [float("nan")] * 4
    print(f"\n=== Table 5: MambaVO matching-only AUC on EuRoC ({len(all_errors)} pairs total) ===", flush=True)
    for t, a in zip(thresholds, aucs):
        print(f"  AUC@{t}: {a:.3f}", flush=True)


if __name__ == "__main__":
    main()
