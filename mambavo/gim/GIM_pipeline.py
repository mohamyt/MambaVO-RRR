import numpy as np

from .feature_extractor import DINOv2FeatureExtractor
from .depth_estimator import MetricDepthEstimator
from .matcher import EfficientLoFTRMatcher
from .pose_solver import PoseSolver
from .feature_fusion import FeatureFusion


class GIMPipeline:
    def __init__(self, loftr_repo_path, loftr_weights_path, device=None):
        self.feature_extractor = DINOv2FeatureExtractor(device=device)
        self.depth_estimator = MetricDepthEstimator(device=device)
        self.matcher = EfficientLoFTRMatcher(repo_path=loftr_repo_path, weights_path=loftr_weights_path, device=device)
        self.pose_solver = PoseSolver()
        self.fusion = FeatureFusion()

    # NOTE: image0 = reference/keyframe, image1 = new frame. K = camera intrinsics for image0.
    def run(self, image0_path, image1_path, K):
        mkpts0, mkpts1, mconf, g_t = self.matcher.match(image0_path, image1_path)

        depth_map, confidence = self.depth_estimator.estimate(image0_path, K)
        depth_map = depth_map.numpy()
        h, w = depth_map.shape

        dino_feat = self.feature_extractor.extract(image0_path, points=mkpts0, original_hw=(h, w))

        F_t = self.fusion(dino_feat, g_t).detach().numpy()

        success, R, t, num_inliers = self.pose_solver.solve(mkpts0, mkpts1, depth_map, K)

        return {
            "F_t": F_t,
            "p_t": mkpts1,
            "T_t": (R, t),
            "pose_success": success,
            "num_inliers": num_inliers,
        }