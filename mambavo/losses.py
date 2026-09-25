from typing import Dict
import torch
import pypose as pp
def pose_loss(pred_poses: Dict[int, torch.Tensor], gt_poses: Dict[int, torch.Tensor]) -> torch.Tensor:
#Eq. 11: relative-pose error, summed over each unordered pair of frame vertices
    frame_ids = sorted(set(pred_poses) & set(gt_poses))
    total = torch.zeros(())
    for a in range(len(frame_ids)):
        for b in range(a + 1, len(frame_ids)):
            f1, f2 = frame_ids[a], frame_ids[b]
            pred_rel = pp.SE3(pred_poses[f1]).Inv() @ pp.SE3(pred_poses[f2]); gt_rel = pp.SE3(gt_poses[f1]).Inv() @ pp.SE3(gt_poses[f2])
            err = (gt_rel.Inv() @ pred_rel).Log(); total = total + err.norm()
    return total
def _backproject(pixel: torch.Tensor, depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    fx, fy, cx, cy = K
    x = (pixel[..., 0] - cx) * depth / fx; y = (pixel[..., 1] - cy) * depth / fy
    return torch.stack([x, y, depth], dim=-1)
def _project(point3d: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    fx, fy, cx, cy = K
    u = point3d[..., 0] * fx / point3d[..., 2] + cx; v = point3d[..., 1] * fy / point3d[..., 2] + cy
    return torch.stack([u, v], dim=-1)
def match_loss(pfg, gt_poses: Dict[int, torch.Tensor], gt_disps: Dict[int, torch.Tensor], intrinsics: Dict[int, torch.Tensor]) -> torch.Tensor:
# Eq. 12.
    home_frame = {}
    for fid in pfg._frame_order:
        for pid in pfg.frame_pixels.get(fid, {}): home_frame.setdefault(pid, fid)
    groups: Dict = {}
    for fid, pixels in pfg.frame_pixels.items():
        if fid not in gt_poses: continue
        refine = pfg.frame_refine.get(fid, {})
        for pid, p_pred in pixels.items():
            r = home_frame[pid]
            if r not in gt_poses or r not in gt_disps: continue
            delta = refine.get(pid, {}).get("delta_pixel", torch.zeros(2, device=p_pred.device)); w = refine.get(pid, {}).get("weight", torch.ones(2, device=p_pred.device))
            groups.setdefault((r, fid), []).append((p_pred, delta, w, pfg.frame_pixels[r][pid]))
    total = torch.zeros(())
    for (r, fid), items in groups.items():
        p_preds = torch.stack([p for p, _, _, _ in items]); deltas = torch.stack([d for _, d, _, _ in items]); weights = torch.stack([w for _, _, w, _ in items])
        home_px = torch.stack([hp for _, _, _, hp in items])
        uv = home_px.long()  # matches the old int(home_px[0])/int(home_px[1]) truncation
        disp = gt_disps[r][uv[:, 1], uv[:, 0]].clamp_min(1e-6)
        pt_r = _backproject(home_px, 1.0 / disp, intrinsics[r])
        T_rel = pp.SE3(gt_poses[r]).Inv() @ pp.SE3(gt_poses[fid])
        gt_px = _project(T_rel.Act(pt_r), intrinsics[fid])
        err = (p_preds + deltas) - gt_px
        total = total + torch.sqrt((weights * err.pow(2)).sum(dim=-1) + 1e-12).sum()
    return total
