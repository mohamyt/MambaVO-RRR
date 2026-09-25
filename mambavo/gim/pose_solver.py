import numpy as np
import cv2


class PoseSolver:
    def backproject(self, mkpts0, depth_map, K):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        h, w = depth_map.shape

        points_3d = []
        valid_indices = []

        for i in range(len(mkpts0)):
            u, v = mkpts0[i]
            ui, vi = int(round(u)), int(round(v))
            if ui < 0 or ui >= w or vi < 0 or vi >= h:
                continue
            z = depth_map[vi, ui]
            if z <= 0:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points_3d.append([x, y, z])
            valid_indices.append(i)

        return np.array(points_3d, dtype=np.float64), valid_indices

    # NOTE: K must be the real camera intrinsics for the dataset, not a placeholder value.
    def solve(self, mkpts0, mkpts1, depth_map, K):
        points_3d, valid_indices = self.backproject(mkpts0, depth_map, K)
        points_2d = mkpts1[valid_indices]

        if len(points_3d) < 6:
            return False, None, None, 0

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, K, None
        )

        if not success:
            return False, None, None, 0

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()
        num_inliers = len(inliers) if inliers is not None else 0

        return True, R, t, num_inliers