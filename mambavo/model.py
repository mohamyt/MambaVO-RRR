#MambaVO model: wires GIM -> PFG -> GMM -> BA over one training clip
import os
import time
import numpy as np
import torch
import torch.nn as nn
from mambavo.gim.GIM_pipeline import GIMPipeline
from mambavo.gmm.gmm import GeometricMambaModule
from mambavo.pfg.pfg import PointFrameGraph
from mambavo.ba.ba_layer import DifferentiableBA

_DEBUG_TIMING = os.environ.get("MAMBAVO_DEBUG_TIMING") == "1"
def _tick(device):
    if _DEBUG_TIMING and str(device).startswith("cuda"): torch.cuda.synchronize()
    return time.time()
def _to_K_matrix(intrinsics_4: torch.Tensor) -> np.ndarray:
    fx, fy, cx, cy = [float(v) for v in intrinsics_4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
def _Rt_to_pose(R: np.ndarray, t: np.ndarray) -> torch.Tensor:
    #cv2 (R, t) -> [px,py,pz,qx,qy,qz,qw] (the pose format used across data_readers/losses)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2; qw = 0.25 * s; qx = (R[2, 1] - R[1, 2]) / s; qy = (R[0, 2] - R[2, 0]) / s; qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; qw = (R[2, 1] - R[1, 2]) / s; qx = 0.25 * s; qy = (R[0, 1] + R[1, 0]) / s; qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; qw = (R[0, 2] - R[2, 0]) / s; qx = (R[0, 1] + R[1, 0]) / s; qy = 0.25 * s; qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; qw = (R[1, 0] - R[0, 1]) / s; qx = (R[0, 2] + R[2, 0]) / s; qy = (R[1, 2] + R[2, 1]) / s; qz = 0.25 * s
    return torch.tensor([t[0], t[1], t[2], qx, qy, qz, qw], dtype=torch.float32)
# ablate options, matches the groups in ablation_results.xlsx:
#   gim group: "context_feature", "geometric_feature", "pnp"
#   gmm group: "history_fusion", "mamba", "gru"
#   tap group: "grad_weight", "history_balance"
#   "none" = full model, no ablation
class MambaVO(nn.Module):
    #every frame in the tartanair dataset is matched against the ref frame (frame 0) and then the matches are added to the PFG
    def __init__(self, loftr_repo_path, loftr_weights_path, device=None, window_size: int = 10, match_thresh_px: float = 3.0,
                 images_are_0_255: bool = True, ablate: str = "none"):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ablate = ablate 
        self.gim = GIMPipeline(loftr_repo_path, loftr_weights_path, device=device); self.gmm = GeometricMambaModule(window_size=window_size, ablate=ablate); self.ba = DifferentiableBA()
        self.fusion = self.gim.fusion
        self.window_size = window_size; self.match_thresh_px = match_thresh_px; self.images_are_0_255 = images_are_0_255
    def _run_gim_frame(self, image_ref: torch.Tensor, image_cur: torch.Tensor, K: np.ndarray, ref_depth_map=None, ref_dino_grid=None):
        t0 = _tick(self.device)
        mkpts0, mkpts1, mconf, g_t = self.gim.matcher.match(image_ref, image_cur)
        t1 = _tick(self.device)
        if ref_depth_map is None:
            depth_map, _ = self.gim.depth_estimator.estimate(image_ref, K)
            depth_map = depth_map.detach().cpu().numpy()
        else:
            depth_map = ref_depth_map
        h, w = depth_map.shape
        dino_feat = self.gim.feature_extractor.extract(image_ref, points=mkpts0, original_hw=(h, w), grid=ref_dino_grid)
        if self.ablate == "context_feature": dino_feat = dino_feat * 0  # kill the dinov2 context feature, fusion just gets zeros for it
        g_t_use = g_t
        if self.ablate == "geometric_feature": g_t_use = g_t * 0  # kill the loftr geometric feature instead
        dino_feat_t = torch.as_tensor(dino_feat, dtype=torch.float32, device=self.device)
        g_t_t = torch.as_tensor(g_t_use, dtype=torch.float32, device=self.device)
        F_t = self.gim.fusion(dino_feat_t, g_t_t).to(self.device)
        t2 = _tick(self.device)
        if self.ablate == "pnp":
            # skip pnp entirely for the ablation, no initial pose guess for this frame
            success, R, t, num_inliers = False, None, None, 0
        else:
            success, R, t, num_inliers = self.gim.pose_solver.solve(mkpts0, mkpts1, depth_map, K)
        t3 = _tick(self.device)
        mkpts0_arr = np.asarray(mkpts0)
        us = np.clip(np.round(mkpts0_arr[:, 0]).astype(np.int64), 0, w - 1); vs = np.clip(np.round(mkpts0_arr[:, 1]).astype(np.int64), 0, h - 1)
        ref_depths = depth_map[vs, us].astype(np.float32)
        if _DEBUG_TIMING:
            print(f"[timing] loftr_match={t1-t0:.2f}s dino_extract+fusion={t2-t1:.2f}s pnp_solve={t3-t2:.2f}s (k={len(mkpts0)} matches)", flush=True)
        return mkpts0, mkpts1, F_t, success, R, t, ref_depths
    def forward(self, images: torch.Tensor, intrinsics: torch.Tensor) -> dict:
        # images:      [N,3,H,W]
        # intrinsics:  [N,4], [fx,fy,cx,cy]
        if self.images_are_0_255: images = images / 255.0
        n_frames, _, H, W = images.shape
        pfg = PointFrameGraph(match_thresh_px=self.match_thresh_px, device=self.device, image_hw=(H, W))
        pfg.poses[0] = torch.tensor([0., 0., 0., 0., 0., 0., 1.], device=self.device)
        history = {}; ref_id = 0
        image_ref = images[ref_id:ref_id + 1]; K_ref = _to_K_matrix(intrinsics[ref_id])
        ref_depth_map, _ = self.gim.depth_estimator.estimate(image_ref, K_ref); ref_depth_map = ref_depth_map.detach().cpu().numpy()
        ref_dino_grid = self.gim.feature_extractor.extract(image_ref, points=None)
        for t in range(1, n_frames):
            K = _to_K_matrix(intrinsics[t])
            mkpts_ref, mkpts_cur, F_t, success, R, tvec, ref_depths = self._run_gim_frame(image_ref, images[t:t + 1], K, ref_depth_map=ref_depth_map, ref_dino_grid=ref_dino_grid)
            init_pose = _Rt_to_pose(R, tvec).to(self.device) if success else None
            t_pfg0 = _tick(self.device)
            point_ids = pfg.add_frame(t, ref_id, mkpts_ref, mkpts_cur, F_t, init_pose=init_pose, intrinsics=intrinsics[t], ref_depths=ref_depths)
            window_frames = pfg.get_window_frames(upto_frame_id=t, window_size=self.window_size)
            t_pfg1 = _tick(self.device)
            out = self.gmm(window_frames=window_frames, current_frame_id=t, current_feat=F_t, current_point_ids=point_ids, history=history)
            t_gmm1 = _tick(self.device)
            history = out["history"]
            for fid, refinement in out["refinements"].items(): pfg.apply_refinements(fid, refinement["point_ids"], refinement["delta_pixels"], refinement["weights"])
            pfg.evict(self.window_size)
            if _DEBUG_TIMING:
                print(f"[timing] pfg_add_frame+window={t_pfg1-t_pfg0:.2f}s gmm={t_gmm1-t_pfg1:.2f}s (P={window_frames and sum(wf['point_ids'].numel() for wf in window_frames)})", flush=True)
        t_ba0 = _tick(self.device)
        ba_inputs = pfg.to_ba_inputs(); ba_out = self.ba(ba_inputs)
        if _DEBUG_TIMING:
            t_ba1 = _tick(self.device); print(f"[timing] ba={t_ba1-t_ba0:.2f}s (edges={ba_inputs['ii'].numel()})", flush=True)
        pred_poses = {fid: ba_out["poses"][i] for i, fid in enumerate(ba_out["frame_ids"])}
        return {"poses": pred_poses, "pfg": pfg} #returns {"poses": {frame_id: [7]}, "pfg": PointFrameGraph}
