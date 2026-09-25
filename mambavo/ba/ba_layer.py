#BA layer that wraps DPVO's differentiable bundle adjustment

from typing import Dict


class DifferentiableBA:
    def __init__(self, num_iters: int = 10, lmbda: float = 1e-4, ep: float = 100.0, fixedp: int = 1):
        self.num_iters = num_iters
        self.lmbda = lmbda
        self.ep = ep
        self.fixedp = fixedp

    def __call__(self, ba_inputs: Dict) -> Dict:
        try:
            from dpvo.ba import BA
            from dpvo.lietorch import SE3
        except ImportError as e:
            raise ImportError("environment isn't loaded correctly") from e
        poses = SE3(ba_inputs["poses_init"].unsqueeze(0))
        patches = ba_inputs["patches_init"].view(1, -1, 3, 1, 1)
        intrinsics = ba_inputs["intrinsics_init"].unsqueeze(0)
        targets = ba_inputs["targets"].unsqueeze(0)
        weights = ba_inputs["weights"].unsqueeze(0)
        poses_opt, patches_opt = BA(
            poses, patches, intrinsics, targets, weights, self.lmbda,
            ba_inputs["ii"], ba_inputs["jj"], ba_inputs["kk"], ba_inputs["bounds"],
            ep=self.ep, fixedp=self.fixedp,
        )
        return {
            "frame_ids": ba_inputs["frame_ids"],
            "point_ids": ba_inputs["point_ids"],
            "poses": poses_opt.data.squeeze(0),
            "patches": patches_opt.squeeze(0),
        }
