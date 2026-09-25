# GMM - refines the matches using history + mamba blocks. see eqs 5-9 in the paper
# also has some ablate switches now so we can turn stuff off and see what breaks lol
import warnings
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
class HistoryFusion(nn.Module):
    # cross attention + a 1x1 conv thing (basically just a linear per point, doesnt mix points)
    def __init__(self, feat_dim: int = 384, n_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=n_heads, batch_first=True)
        self.conv1d = nn.Conv1d(feat_dim, feat_dim, kernel_size=1, bias=True); self.norm = nn.LayerNorm(feat_dim)
    def forward(self, F_t: Tensor, H_prev: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
        # unobserved rows still go through here, caller just ignores them later w the mask
        F_t_b = F_t.unsqueeze(0); H_b = H_prev.unsqueeze(0)
        M_hat, _ = self.cross_attn(F_t_b, H_b, H_b); M_hat = M_hat.squeeze(0)
        x = M_hat.T.unsqueeze(0); x = self.conv1d(x); M_t = x.squeeze(0).T
        return self.norm(M_t)
def _build_mamba_block(d_model: int, **kwargs):
    # tries real mamba, falls back to a gru if mamba_ssm isnt installed (not the same weights btw!!)
    try:
        from mamba_ssm import Mamba
        return Mamba(d_model=d_model, **kwargs)
    except ImportError:
        warnings.warn("mamba-ssm not installed. Using GRU fallback for Mamba block. Install with: pip install mamba-ssm. Checkpoints trained with this fallback are NOT compatible with real Mamba blocks.")
        return _GRUFallbackBlock(d_model)
class _GRUFallbackBlock(nn.Module):
    # just a gru pretending to be a mamba block, for when we cant install mamba_ssm
    def __init__(self, d_model: int):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True); self.norm = nn.LayerNorm(d_model)
    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.gru(x)
        return self.norm(out)
class RefineHead(nn.Module):
    # spits out a pixel correction and a weight for each token, eq 8
    def __init__(self, feat_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        self.mlp_delta = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 2))
        self.mlp_weight = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 2), nn.Softplus())
    def forward(self, token: Tensor) -> Tuple[Tensor, Tensor]:
        delta_p = self.mlp_delta(token); weight = self.mlp_weight(token)
        return delta_p, weight
class GeometricMambaModule(nn.Module):
    # the whole gmm thing, eqs 5-9. history dict lives outside and gets passed in/out every call
    UNOBSERVED_INIT_STD = 0.0
    def __init__(self, feat_dim: int = 384, n_mamba_layers: int = 4, n_attn_heads: int = 4, hidden_dim: int = 128,
                 window_size: int = 10, max_points_per_step: int = 4096, ablate: str = "none"):
        super().__init__()
        self.feat_dim = feat_dim; self.W = window_size; self.max_points_per_step = max_points_per_step
        self.ablate = ablate  # can be "none", "history_fusion", "mamba", or "gru" -- see ablation_results.xlsx
        self.history_fusion = HistoryFusion(feat_dim, n_attn_heads)
        self.mamba_blocks = nn.ModuleList([_build_mamba_block(d_model=feat_dim) for _ in range(n_mamba_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(feat_dim) for _ in range(n_mamba_layers)])
        self.refine_head = RefineHead(feat_dim, hidden_dim)
        self.gru = nn.GRUCell(feat_dim, feat_dim)
        # embedding for "this point wasnt seen this frame" slots, not just zeros
        self.unobserved_token = nn.Parameter(torch.zeros(feat_dim))
        nn.init.normal_(self.unobserved_token, std=0.02)
        # turns a points first ever GIM feature into its starting history vector
        self.history_init_proj = nn.Linear(feat_dim, feat_dim)
    def init_point_history(self, point_ids: Tensor, first_feats: Tensor) -> Dict[int, Tensor]:
        # only call for genuinely NEW point ids, dont overwrite existing history with this
        h0 = self.history_init_proj(first_feats)
        return {int(pid): h0[i] for i, pid in enumerate(point_ids)}
    def _gather_padded(self, all_pids_sorted: Tensor, point_ids: Tensor, feats: Tensor, fill_value: Tensor) -> Tuple[Tensor, Tensor]:
        # scatters feats into a (P,D) tensor lined up w all_pids_sorted, missing rows get fill_value
        P = all_pids_sorted.shape[0]; D = feats.shape[-1] if feats.numel() > 0 else fill_value.shape[-1]; device = all_pids_sorted.device
        out = fill_value.unsqueeze(0).expand(P, D).clone(); mask = torch.zeros(P, dtype=torch.bool, device=device)
        if point_ids.numel() == 0: return out, mask
        idx = torch.searchsorted(all_pids_sorted, point_ids)  # works cuz all_pids_sorted is sorted+unique
        out[idx] = feats.to(out.dtype); mask[idx] = True
        return out, mask
    def forward(self, window_frames: List[Dict], current_frame_id: int, current_feat: Tensor,
                current_point_ids: Tensor, history: Dict[int, Tensor]) -> Dict:
        device = current_feat.device; D = self.feat_dim
        # step 1, figure out every point id thats active right now (window + current frame)
        all_ids = [current_point_ids] + [wf["point_ids"] for wf in window_frames]
        all_pids = torch.unique(torch.cat(all_ids)) if all_ids else current_point_ids.new_empty(0)
        all_pids, _ = torch.sort(all_pids)
        if all_pids.numel() > self.max_points_per_step:
            # too many points, just keep the newest ones so we dont OOM
            warnings.warn(f"GMM: {all_pids.numel()} active points exceeds max_points_per_step={self.max_points_per_step}; truncating to the most recent ones.")
            all_pids = all_pids[-self.max_points_per_step:]
        P = all_pids.numel()
        if P == 0: return {"history": history, "refinements": {}}
        # one .tolist() sync instead of a .item() per point (see below) -- each .item() call
        # blocks python on the gpu catching up, and with up to max_points_per_step points that
        # was thousands of stalls per call.
        all_pids_list = all_pids.tolist()
        # step 2, brand new points get their history seeded from their own gim feature
        new_mask = torch.tensor([pid not in history for pid in all_pids_list], device=device, dtype=torch.bool)
        if new_mask.any():
            new_pids = all_pids[new_mask]
            frames_list = window_frames + [{"frame_id": current_frame_id, "tokens": current_feat, "point_ids": current_point_ids}]
            frames_list = [f for f in frames_list if f["point_ids"].numel() > 0]
            src_feats = torch.zeros(new_pids.numel(), D, device=device)
            if frames_list:
                # vectorized version of "first frame in frames_list that has this point id wins" --
                # point ids are unique within a single frame (they come from dict keys upstream),
                # so exactly one entry can win per point id.
                cat_pids = torch.cat([f["point_ids"] for f in frames_list])
                cat_toks = torch.cat([f["tokens"] for f in frames_list], dim=0)
                cat_rank = torch.cat([torch.full((f["point_ids"].numel(),), r, dtype=torch.long, device=device) for r, f in enumerate(frames_list)])
                uniq_pids, inv = torch.unique(cat_pids, return_inverse=True)
                min_rank = torch.full((uniq_pids.numel(),), len(frames_list), dtype=torch.long, device=device)
                min_rank.scatter_reduce_(0, inv, cat_rank, reduce="amin", include_self=True)
                is_winner = cat_rank == min_rank[inv]
                winner_tok = torch.zeros(uniq_pids.numel(), D, device=device, dtype=cat_toks.dtype)
                winner_tok[inv[is_winner]] = cat_toks[is_winner]
                pos = torch.searchsorted(uniq_pids, new_pids).clamp(max=uniq_pids.numel() - 1)
                hit = uniq_pids[pos] == new_pids
                src_feats[hit] = winner_tok[pos[hit]]
            init = self.init_point_history(new_pids, src_feats)
            history.update(init)
        # step 3, stack up the (P,D) history tensor lined up w all_pids
        H_prev = torch.stack([history[pid].to(device) for pid in all_pids_list], dim=0)
        # step 4, history fusion for this frame (eq 5-6) -- unless were ablating it away
        F_t_padded, cur_mask = self._gather_padded(all_pids, current_point_ids, current_feat, self.unobserved_token)
        if self.ablate == "history_fusion":
            M_t = F_t_padded  # skip the fusion, just use the raw feature as is (no history mixed in)
        else:
            M_t = self.history_fusion(F_t_padded, H_prev, cur_mask)
        # step 5, build the window sequence lined up by point id, oldest to newest
        frame_ids: List[int] = [wf["frame_id"] for wf in window_frames] + [current_frame_id]
        frame_tensors: List[Tensor] = []; frame_masks: List[Tensor] = []
        for wf in window_frames:
            padded, mask = self._gather_padded(all_pids, wf["point_ids"], wf["tokens"], self.unobserved_token)
            frame_tensors.append(padded); frame_masks.append(mask)
        frame_tensors.append(M_t); frame_masks.append(cur_mask)
        seq = torch.stack(frame_tensors, dim=1); obs_mask = torch.stack(frame_masks, dim=1)
        # step 6, run the mamba blocks (eq 7) -- unless ablated, then we just skip refining entirely
        if self.ablate == "mamba":
            refined = seq  # no temporal refinement happening here, just pass the raw sequence through
        else:
            x = seq
            for mamba, norm in zip(self.mamba_blocks, self.layer_norms):
                residual = x
                x = mamba(x)  # each row does its own timeline so padding cant leak between points
                x = norm(x + residual)
            refined = x
        H_hat = refined[:, -1, :]
        # step 7, refine head per frame (eq 8)
        refinements: Dict[int, Dict[str, Tensor]] = {}
        for t_idx, fid in enumerate(frame_ids):
            token = refined[:, t_idx, :]
            dp, w = self.refine_head(token)
            valid = obs_mask[:, t_idx]
            refinements[fid] = {"point_ids": all_pids[valid], "delta_pixels": dp[valid], "weights": w[valid]}
        # step 8, gru history update (eq 9) -- if ablated we just use the raw token as the new history
        if self.ablate == "gru":
            H_new_full = H_hat  # no gru gating, just take Hhat straight as the new history
        else:
            H_new_full = self.gru(H_hat, H_prev)
        H_updated = torch.where(cur_mask.unsqueeze(-1), H_new_full, H_prev)
        for i, pid in enumerate(all_pids_list):
            # staying attached to the graph on purpose, training clips are short segments so its fine
            history[pid] = H_updated[i]
        return {"history": history, "refinements": refinements}
    @staticmethod
    def prune_history(history: Dict[int, Tensor], active_point_ids: Tensor) -> None:
        # drops history for points that fell out of the window, otherwise it just grows forever
        active = set(int(p) for p in active_point_ids.tolist())
        for pid in list(history.keys()):
            if pid not in active: del history[pid]
