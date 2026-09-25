#Adapted from: [Teed, Lipson, Deng, "Deep Patch Visual Odometry," NeurIPS 36, 2023 — github.com/princeton-vl/DPVO]

from typing import Dict, List, Optional
import torch


class PointFrameGraph:
    def __init__(self, match_thresh_px: float = 3.0, device=None, image_hw=None):
        self.match_thresh_px = match_thresh_px
        self.device = device
        self.image_hw = image_hw
        self.frame_pixels: Dict[int, Dict[int, torch.Tensor]] = {}
        self.frame_tokens: Dict[int, Dict[int, torch.Tensor]] = {}
        self.frame_refine: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
        self.frame_intrinsics: Dict[int, torch.Tensor] = {}
        self.poses: Dict[int, torch.Tensor] = {}
        self.point_home_frame: Dict[int, int] = {}
        self.point_depth: Dict[int, torch.Tensor] = {}
        self._next_point_id = 0
        self._frame_order: List[int] = []

    def _as_tensor(self, x, dtype=torch.float32):
        t = x if torch.is_tensor(x) else torch.as_tensor(x, dtype=dtype)
        return t.to(self.device) if self.device is not None else t

    def add_frame(self, frame_id: int, ref_frame_id: int, mkpts_ref, mkpts_cur, F_t,
                  init_pose: Optional[torch.Tensor] = None, intrinsics=None, ref_depths=None):
        mkpts_ref = self._as_tensor(mkpts_ref)
        mkpts_cur = self._as_tensor(mkpts_cur)
        F_t = self._as_tensor(F_t)
        if ref_depths is not None:
            ref_depths = self._as_tensor(ref_depths)
        if intrinsics is not None:
            intr_t = self._as_tensor(intrinsics)
            self.frame_intrinsics.setdefault(frame_id, intr_t)
            self.frame_intrinsics.setdefault(ref_frame_id, intr_t)
        ref_pixels = self.frame_pixels.setdefault(ref_frame_id, {})
        k = mkpts_ref.shape[0]
        if ref_pixels:
            ref_pids = list(ref_pixels.keys())
            ref_px = torch.stack([ref_pixels[p] for p in ref_pids])
            dist = torch.cdist(mkpts_ref, ref_px)
            min_dist, min_idx = dist.min(dim=1)
            matched = min_dist < self.match_thresh_px
        else:
            ref_pids = []
            matched = torch.zeros(k, dtype=torch.bool, device=self.device)
        point_ids = torch.empty(k, dtype=torch.long, device=self.device)
        if matched.any():
            ref_pids_t = torch.tensor(ref_pids, dtype=torch.long, device=self.device)
            point_ids[matched] = ref_pids_t[min_idx[matched]]
        new_mask = ~matched
        n_new = int(new_mask.sum())
        if n_new > 0:
            new_ids = torch.arange(self._next_point_id, self._next_point_id + n_new, dtype=torch.long, device=self.device)
            point_ids[new_mask] = new_ids
            self._next_point_id += n_new
            new_idx = new_mask.nonzero(as_tuple=True)[0].tolist()
            for i, pid in zip(new_idx, new_ids.tolist()):
                ref_pixels[pid] = mkpts_ref[i].detach()
                self.point_home_frame[pid] = ref_frame_id
                self.point_depth[pid] = ref_depths[i].detach() if ref_depths is not None else torch.tensor(1.0, device=self.device)
        cur_pixels = self.frame_pixels.setdefault(frame_id, {})
        cur_tokens = self.frame_tokens.setdefault(frame_id, {})
        point_ids_list = point_ids.tolist()
        for i, pid in enumerate(point_ids_list):
            cur_pixels[pid] = mkpts_cur[i].detach()
            cur_tokens[pid] = F_t[i]
        if frame_id not in self._frame_order:
            self._frame_order.append(frame_id)
        if ref_frame_id not in self._frame_order:
            self._frame_order.append(ref_frame_id)
        if init_pose is not None:
            self.poses.setdefault(frame_id, self._as_tensor(init_pose))
        return point_ids

    def get_window_frames(self, upto_frame_id: int, window_size: int):
        order = [f for f in self._frame_order if f != upto_frame_id]
        window = order[-window_size:]
        frames = []
        for fid in window:
            tokens = self.frame_tokens.get(fid, {})
            if not tokens:
                continue
            pids = sorted(tokens.keys())
            frames.append({
                "frame_id": fid,
                "tokens": torch.stack([tokens[p] for p in pids]),
                "point_ids": torch.tensor(pids, dtype=torch.long, device=self.device),
            })
        return frames

    def apply_refinements(self, frame_id: int, point_ids: torch.Tensor, delta_pixels: torch.Tensor, weights: torch.Tensor):
        refine = self.frame_refine.setdefault(frame_id, {})
        for i, pid in enumerate(point_ids.tolist()):
            refine[pid] = {"delta_pixel": delta_pixels[i], "weight": weights[i]}

    def to_ba_inputs(self):
        frame_ids = sorted(self.frame_pixels.keys())
        frame_idx = {f: i for i, f in enumerate(frame_ids)}
        point_ids = sorted({p for pix in self.frame_pixels.values() for p in pix})
        point_idx = {p: i for i, p in enumerate(point_ids)}
        ii, jj, kk, targets, weights = [], [], [], [], []
        for fid in frame_ids:
            refine = self.frame_refine.get(fid, {})
            for pid, px in self.frame_pixels[fid].items():
                delta = refine.get(pid, {}).get("delta_pixel")
                w = refine.get(pid, {}).get("weight")
                home = self.point_home_frame.get(pid, frame_ids[0])
                ii.append(frame_idx[home])
                jj.append(frame_idx[fid])
                kk.append(point_idx[pid])
                targets.append(px + delta if delta is not None else px)
                weights.append(w if w is not None else torch.ones(2, device=self.device))
        poses_init = torch.stack([self.poses.get(f, torch.tensor([0., 0., 0., 0., 0., 0., 1.], device=self.device)) for f in frame_ids])
        intrinsics_init = torch.stack([self.frame_intrinsics.get(f, torch.zeros(4, device=self.device)) for f in frame_ids])
        xy = torch.stack([self.frame_pixels[self.point_home_frame.get(p, frame_ids[0])][p] for p in point_ids])
        depth = torch.stack([self.point_depth.get(p, torch.tensor(1.0, device=self.device)) for p in point_ids]).clamp(min=1e-3)
        disp = (1.0 / depth).unsqueeze(-1)
        patches_init = torch.cat([xy, disp], dim=-1)
        H, W = self.image_hw if self.image_hw is not None else (0, 0)
        bounds = torch.tensor([0, 0, W, H], dtype=torch.float32, device=self.device)
        return {
            "frame_ids": frame_ids,
            "point_ids": point_ids,
            "poses_init": poses_init,
            "intrinsics_init": intrinsics_init,
            "patches_init": patches_init,
            "bounds": bounds,
            "ii": torch.tensor(ii, dtype=torch.long, device=self.device),
            "jj": torch.tensor(jj, dtype=torch.long, device=self.device),
            "kk": torch.tensor(kk, dtype=torch.long, device=self.device),
            "targets": torch.stack(targets) if targets else torch.empty(0, 2, device=self.device),
            "weights": torch.stack(weights) if weights else torch.empty(0, 2, device=self.device),
        }

    def evict(self, window_size: int):
        if len(self._frame_order) <= window_size:
            return
        drop = self._frame_order[:-window_size]
        self._frame_order = self._frame_order[-window_size:]
        drop_set = set(drop)
        orphaned = {pid for pid, home in self.point_home_frame.items() if home in drop_set}
        for fid in drop:
            self.frame_pixels.pop(fid, None)
            self.frame_tokens.pop(fid, None)
            self.frame_refine.pop(fid, None)
            self.poses.pop(fid, None)
        for pid in orphaned:
            self.point_home_frame.pop(pid, None)
            self.point_depth.pop(pid, None)
        for fid in self._frame_order:
            for pid in orphaned:
                self.frame_pixels.get(fid, {}).pop(pid, None)
                self.frame_tokens.get(fid, {}).pop(pid, None)
                self.frame_refine.get(fid, {}).pop(pid, None)
        active_points = {p for pix in self.frame_pixels.values() for p in pix}
        for pid in list(self.point_depth.keys()):
            if pid not in active_points:
                del self.point_depth[pid]
                self.point_home_frame.pop(pid, None)
