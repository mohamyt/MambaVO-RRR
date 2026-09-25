import torch
import torch.nn as nn
class TrendingAwarePenalty:
    # balances pose loss vs match loss so training doesnt go crazy, eqs 13-15
    def __init__(self, grad_update_freq=50, history_len=4, ablate="none"):
        self.grad_update_freq = grad_update_freq; self.history_len = history_len
        self.ablate = ablate  # "none", "grad_weight", or "history_balance" -- see ablation_results.xlsx
        self.pose_loss_history = []; self.match_loss_history = []
        self.grad_weight = 1.0  # starts at 1 so early steps arent weighted weird
        self.step = 0
    def _compute_grad_weight(self, pose_loss, match_loss, model_params):
        pose_grads = torch.autograd.grad(pose_loss, model_params, retain_graph=True, allow_unused=True)
        pose_grad_norm = torch.sqrt(sum(g.norm() ** 2 for g in pose_grads if g is not None))
        match_grads = torch.autograd.grad(match_loss, model_params, retain_graph=True, allow_unused=True)
        match_grad_norm = torch.sqrt(sum(g.norm() ** 2 for g in match_grads if g is not None))
        if match_grad_norm < 1e-8: return 1.0  # dont wanna divide by basically zero
        return (pose_grad_norm / match_grad_norm).item()
    def _compute_trend(self, history):
        if len(history) < self.history_len: return 1.0  # not enough data yet, just skip it
        current = history[-1]; past_avg = sum(history[-self.history_len:]) / self.history_len
        if past_avg < 1e-8: return 1.0
        ratio = max(0.5, min(2.0, current / past_avg))  # clamp pre-exp: unclamped, a ratio of just ~4.5 already produced the observed ~90x runaway
        return torch.exp(torch.tensor(ratio)).item()
    def compute_loss(self, pose_loss, match_loss, model_params):
        self.step += 1
        # every 50 steps we recompute how much to weight the match loss based on gradient sizes
        # (unless ablated, then grad_weight just stays stuck at 1.0 forever)
        if self.ablate != "grad_weight" and self.step % self.grad_update_freq == 0:
            self.grad_weight = self._compute_grad_weight(pose_loss, match_loss, model_params)
        scaled_match_loss = self.grad_weight * match_loss
        self.pose_loss_history.append(pose_loss.item()); self.match_loss_history.append(scaled_match_loss.item())
        # trend based balance thing, eq 14-15. ablate this and its just a flat 1.0 multiplier
        if self.ablate == "history_balance":
            lambda_total = 1.0
        else:
            lambda_match = self._compute_trend(self.match_loss_history)
            lambda_pose = self._compute_trend(self.pose_loss_history)
            if lambda_pose < 1e-8: lambda_pose = 1e-8  # avoid blowing up
            lambda_total = lambda_match / lambda_pose
        total_loss = pose_loss + lambda_total * scaled_match_loss
        return total_loss
