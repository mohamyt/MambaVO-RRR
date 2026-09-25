#Adapted from: [Teed, Lipson, Deng, "Deep Patch Visual Odometry," NeurIPS 36, 2023 — github.com/princeton-vl/DPVO] 
# and [Messikommer et al., "Reinforcement Learning Meets Visual Odometry", ECCV 2024. https://github.com/uzh-rpg/rl_vo]


import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class KITTI(Dataset):
    """
    KITTI odometry dataset loader for evaluation.
    Returns one frame at a time with its ground truth pose and intrinsics.

    Expected directory structure:
        datapath/
            sequences/
                00/
                    image_0/    <- left grayscale images
                    calib.txt   <- camera calibration (P0..P3)
            poses/
                00.txt          <- ground truth poses (only sequences 00-10 have these)
    """

    def __init__(self, datapath, sequence):
        seq_dir = os.path.join(datapath, 'sequences', sequence)
        img_dir = os.path.join(seq_dir, 'image_0')
        self.image_paths = sorted(
            os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')
        )

        K = self._read_calib(os.path.join(seq_dir, 'calib.txt'))
        intr = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)
        self.intrinsics = np.tile(intr, (len(self.image_paths), 1))

        pose_path = os.path.join(datapath, 'poses', f'{sequence}.txt')
        self.poses = self._read_poses(pose_path) if os.path.exists(pose_path) else None

    def _read_calib(self, path):
        with open(path) as f:
            for line in f:
                if line.startswith('P0:'):
                    return np.array(line.split()[1:], dtype=np.float64).reshape(3, 4)
        raise ValueError(f"P0 not found in {path}")

    def _read_poses(self, path):
        # each line: 12 floats, a flattened 3x4 [R|t] world<-cam0 pose (world = frame 0's own frame)
        raw = np.loadtxt(path).reshape(-1, 3, 4)
        poses = np.zeros((raw.shape[0], 7), dtype=np.float32)
        for i in range(raw.shape[0]):
            R, t = raw[i, :, :3], raw[i, :, 3]
            poses[i, :3] = t
            poses[i, 3:] = self._rot_to_quat(R)
        return poses

    @staticmethod
    def _rot_to_quat(R):
        # same formula as mambavo/model.py's _Rt_to_pose, kept in sync for consistency
        tr = np.trace(R)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2; qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s; qy = (R[0, 2] - R[2, 0]) / s; qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s; qy = (R[0, 1] + R[1, 0]) / s; qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s; qy = 0.25 * s; qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s; qy = (R[1, 2] + R[2, 1]) / s; qz = 0.25 * s
        return np.array([qx, qy, qz, qw], dtype=np.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32)
        image = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]

        if self.poses is not None:
            pose = torch.from_numpy(self.poses[idx])
        else:
            pose = torch.zeros(7)
        intrinsics = torch.from_numpy(self.intrinsics[idx])

        return image, pose, intrinsics

    def get_all_poses(self):
        return torch.from_numpy(self.poses) if self.poses is not None else None
