# Adapted from : [Teed, Lipson, Deng, "Deep Patch Visual Odometry," NeurIPS 36, 2023 — github.com/princeton-vl/DPVO]

from typing import Dict, Optional
import numpy as np
import torch
import pypose as pp

from mambavo.gim.GIM_pipeline import GIMPipeline
from mambavo.gmm.gmm import GeometricMambaModule
from mambavo.pfg.pfg import PointFrameGraph
from mambavo.ba.ba_layer import DifferentiableBA
from mambavo.model import _to_K_matrix, _Rt_to_pose


class MambaVOTracker:
    def __init__(self, loftr_repo_path: str, loftr_weights_path: str, device: str = "cuda",
                 window_size: int = 10, match_thresh_px: float = 3.0,
                 keyframe_parallax_px: float = 30.0, keyframe_max_gap: int = 3,
                 images_are_0_255: bool = True):
        self.device = device
        self.gim = GIMPipeline(loftr_repo_path, loftr_weights_path, device=device)
        self.gmm = GeometricMambaModule(window_size=window_size, ablate="none").to(device).eval()
        self.ba = DifferentiableBA()
        self.window_size = window_size
        self.match_thresh_px = match_thresh_px
        self.keyframe_parallax_px = keyframe_parallax_px
        self.keyframe_max_gap = keyframe_max_gap
        self.images_are_0_255 = images_are_0_255

    def load_checkpoint(self, state_dict: Dict[str, torch.Tensor]):
        gmm_sd = {k[len("gmm."):]: v for k, v in state_dict.items() if k.startswith("gmm.")}
        missing, unexpected = self.gmm.load_state_dict(gmm_sd, strict=False)
        fusion_sd = {k[len("fusion."):]: v for k, v in state_dict.items() if k.startswith("fusion.")}
        if fusion_sd:
            f_missing, f_unexpected = self.gim.fusion.load_state_dict(fusion_sd, strict=False)
            if f_missing or f_unexpected:
                print(f"[tracker] warning: fusion load_state_dict missing={f_missing} unexpected={f_unexpected}", flush=True)
        else:
            print("[tracker] warning: checkpoint has no fusion.* keys -- gim.fusion stays at random init", flush=True)
        other_keys = [k for k in state_dict if not k.startswith("gmm.") and not k.startswith("fusion.")]
        if other_keys:
            print(f"[tracker] warning: checkpoint has unrecognized keys not loaded: {other_keys[:5]}...", flush=True)
        if missing or unexpected:
            print(f"[tracker] warning: gmm load_state_dict missing={missing} unexpected={unexpected}", flush=True)

    def _make_pfg(self, H: int, W: int) -> PointFrameGraph:
        return PointFrameGraph(match_thresh_px=self.match_thresh_px, device=self.device, image_hw=(H, W))

    @torch.no_grad()
    def track_sequence(self, dataset, log_every: int = 50) -> Dict[int, torch.Tensor]:
        pfg: Optional[PointFrameGraph] = None
        history: Dict[int, torch.Tensor] = {}
        kf_cache: Dict[int, tuple] = {}
        kf_to_frame_idx: Dict[int, int] = {}
        trajectory: Dict[int, torch.Tensor] = {}
        frames_since_kf = 0
        last_kf_id = None
        next_kf_id = 0

        n = len(dataset)
        for idx in range(n):
            image, _, intrinsics = dataset[idx]
            image = image.to(self.device); intrinsics = intrinsics.to(self.device)
            image_n = (image / 255.0 if self.images_are_0_255 else image).unsqueeze(0)
            _, H, W = image.shape

            if pfg is None:
                pfg = self._make_pfg(H, W)
                pfg.poses[next_kf_id] = torch.tensor([0., 0., 0., 0., 0., 0., 1.], device=self.device)
                K = _to_K_matrix(intrinsics)
                depth_map, _ = self.gim.depth_estimator.estimate(image_n, K)
                depth_map = depth_map.detach().cpu().numpy()
                dino_grid = self.gim.feature_extractor.extract(image_n, points=None)
                kf_cache[next_kf_id] = (depth_map, dino_grid, image_n, K, intrinsics)
                pfg._frame_order.append(next_kf_id)
                pfg.frame_intrinsics[next_kf_id] = intrinsics
                kf_to_frame_idx[next_kf_id] = idx
                trajectory[idx] = pp.SE3(pfg.poses[next_kf_id]).Inv().data.detach().clone()
                last_kf_id = next_kf_id
                next_kf_id += 1
                frames_since_kf = 0
                continue

            ref_depth_map, ref_dino_grid, ref_image, ref_K, ref_intr = kf_cache[last_kf_id]
            mkpts0, mkpts1, mconf, g_t = self.gim.matcher.match(ref_image, image_n)
            if len(mkpts0) < 8:
                frames_since_kf += 1
                continue
            mkpts0_arr = np.asarray(mkpts0); mkpts1_arr = np.asarray(mkpts1)
            parallax = float(np.median(np.linalg.norm(mkpts1_arr - mkpts0_arr, axis=1)))
            is_keyframe = (parallax > self.keyframe_parallax_px) or (frames_since_kf >= self.keyframe_max_gap)
            if not is_keyframe:
                frames_since_kf += 1
                continue

            h, w = ref_depth_map.shape
            dino_feat = self.gim.feature_extractor.extract(ref_image, points=mkpts0, original_hw=(h, w), grid=ref_dino_grid)
            F_t = self.gim.fusion(dino_feat, g_t).to(self.device)
            success, R, tvec, _ = self.gim.pose_solver.solve(mkpts0, mkpts1, ref_depth_map, ref_K)
            us = np.clip(np.round(mkpts0_arr[:, 0]).astype(np.int64), 0, w - 1)
            vs = np.clip(np.round(mkpts0_arr[:, 1]).astype(np.int64), 0, h - 1)
            ref_depths = ref_depth_map[vs, us].astype(np.float32)

            init_pose_global = None
            if success:
                T_cur_from_ref = pp.SE3(_Rt_to_pose(R, tvec).to(self.device))
                T_ref_from_world = pp.SE3(pfg.poses[last_kf_id])
                init_pose_global = (T_cur_from_ref @ T_ref_from_world).data

            kf_id = next_kf_id
            point_ids = pfg.add_frame(kf_id, last_kf_id, mkpts0, mkpts1, F_t, init_pose=init_pose_global,
                                       intrinsics=intrinsics, ref_depths=ref_depths)
            window_frames = pfg.get_window_frames(upto_frame_id=kf_id, window_size=self.window_size)
            out = self.gmm(window_frames=window_frames, current_frame_id=kf_id, current_feat=F_t,
                            current_point_ids=point_ids, history=history)
            history = out["history"]
            for fid, refinement in out["refinements"].items():
                pfg.apply_refinements(fid, refinement["point_ids"], refinement["delta_pixels"], refinement["weights"])
            pfg.evict(self.window_size)

            ba_inputs = pfg.to_ba_inputs()
            ba_out = self.ba(ba_inputs)
            for i, fid in enumerate(ba_out["frame_ids"]):
                refined_pose = ba_out["poses"][i].detach()
                pfg.poses[fid] = refined_pose
                trajectory[kf_to_frame_idx.get(fid, fid)] = pp.SE3(refined_pose).Inv().data.clone()

            K = _to_K_matrix(intrinsics)
            depth_map, _ = self.gim.depth_estimator.estimate(image_n, K)
            depth_map = depth_map.detach().cpu().numpy()
            dino_grid = self.gim.feature_extractor.extract(image_n, points=None)
            kf_cache[kf_id] = (depth_map, dino_grid, image_n, K, intrinsics)
            kf_to_frame_idx[kf_id] = idx
            for old_kf in list(kf_cache.keys()):
                if old_kf not in pfg.frame_pixels: del kf_cache[old_kf]

            last_kf_id = kf_id
            next_kf_id += 1
            frames_since_kf = 0

            if log_every and kf_id % log_every == 0:
                print(f"[track] frame {idx}/{n} -> keyframe {kf_id} (parallax={parallax:.1f}px, "
                      f"{len(mkpts0)} matches, window={len(pfg._frame_order)})", flush=True)

        return trajectory
