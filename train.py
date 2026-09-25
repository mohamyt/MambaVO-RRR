import argparse
import os
import glob
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from mambavo.data_readers.tartan import TartanAir
from mambavo.model import MambaVO
from mambavo.losses import pose_loss, match_loss
from mambavo.tap.tap import TrendingAwarePenalty
def build_gt_dicts(poses, disps, intrinsics):
    n = poses.shape[0]
    gt_poses = {i: poses[i] for i in range(n)}; gt_disps = {i: disps[i] for i in range(n)}; intr = {i: intrinsics[i] for i in range(n)}
    return gt_poses, gt_disps, intr
def find_latest_checkpoint(ckpt_dir):
    # just grabs whichever step_*.pt has the biggest number, thats our latest one
    ckpts = glob.glob(os.path.join(ckpt_dir, "step_*.pt"))
    if not ckpts: return None, 0
    steps = [int(os.path.basename(c).replace("step_", "").replace(".pt", "")) for c in ckpts]
    latest_step = max(steps)
    return os.path.join(ckpt_dir, f"step_{latest_step}.pt"), latest_step
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datapath", required=True); p.add_argument("--loftr_repo_path", default="external/EfficientLoFTR"); p.add_argument("--loftr_weights_path", default="external/EfficientLoFTR/weights/eloftr_outdoor.ckpt")
    p.add_argument("--n_frames", type=int, default=4); p.add_argument("--steps", type=int, default=196000); p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--ckpt_dir", default="checkpoints"); p.add_argument("--log_dir", default="runs/mambavo"); p.add_argument("--ckpt_every", type=int, default=5000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", action="store_true")  # picks up from the latest checkpoint in ckpt_dir if theres one
    p.add_argument("--ablate", default="none", choices=["none", "context_feature", "geometric_feature", "pnp",
                                                          "history_fusion", "mamba", "gru", "grad_weight", "history_balance"]) # for ablation studies, "none" = normal full model
    return p.parse_args()
def main():
    args = parse_args(); os.makedirs(args.ckpt_dir, exist_ok=True)
    log_dir = args.log_dir if args.ablate == "none" else args.log_dir + "_ablate_" + args.ablate  # so ablation runs dont stomp each other
    dataset = TartanAir(args.datapath, n_frames=args.n_frames); loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    model = MambaVO(args.loftr_repo_path, args.loftr_weights_path, device=args.device, ablate=args.ablate).to(args.device)
    tap = TrendingAwarePenalty(ablate=args.ablate); optimizer = torch.optim.Adam(model.parameters(), lr=args.lr); writer = SummaryWriter(log_dir)
    step = 0
    if args.resume:
        # just restores model weights, doesnt save/restore optimizer or tap state, good enough for now
        ckpt_path, step = find_latest_checkpoint(args.ckpt_dir)
        if ckpt_path is not None:
            print(f"resuming from {ckpt_path} (step {step})")
            model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
        else:
            print("--resume was set but no checkpoint found, starting fresh")
    while step < args.steps:
        for images, poses, disps, intrinsics in loader:
            images = images[0].to(args.device); poses = poses[0].to(args.device); disps = disps[0].to(args.device); intrinsics = intrinsics[0].to(args.device)
            gt_poses, gt_disps, intr = build_gt_dicts(poses, disps, intrinsics)
            try:
                out = model(images, intrinsics)
                l_pose = pose_loss(out["poses"], gt_poses); l_match = match_loss(out["pfg"], gt_poses, gt_disps, intr)
                # TAP's grad-norm ratio (Eq. 13) is scoped to gmm only, not model.parameters()
                # including fusion here (now that it's a registered submodule) fed a feedback loop:
                # inflated lambda_total -> bigger real gradients -> next grad_weight recompute shifts
                # further -> runaway. gmm-only matches the ratio's behavior before fusion was ever trained.
                loss = tap.compute_loss(l_pose, l_match, list(model.gmm.parameters()))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            except RuntimeError as e:
                # degenerate clip: GIM found zero valid matches across every frame in this
                # window (weak texture / motion blur / extreme viewpoint change), so
                # pfg.to_ba_inputs() has nothing to torch.stack(). not a code bug, so we skip
                # this sample and move on without counting it toward step count
                if "stack expects a non-empty TensorList" in str(e):
                    print(f"[warn] skipping degenerate clip at step {step} (no valid matches): {e}", flush=True)
                    optimizer.zero_grad()
                    continue
                raise
            writer.add_scalar("loss/total", loss.item(), step); writer.add_scalar("loss/pose", l_pose.item(), step); writer.add_scalar("loss/match", l_match.item(), step)
            if step % args.ckpt_every == 0: torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"step_{step}.pt"))
            step += 1
            if step >= args.steps: break
    writer.close()
if __name__ == "__main__":
    main()
