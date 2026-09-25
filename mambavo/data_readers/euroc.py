#Adapted from: [Teed, Lipson, Deng, "Deep Patch Visual Odometry," NeurIPS 36, 2023 — github.com/princeton-vl/DPVO] 
# and [Messikommer et al., "Reinforcement Learning Meets Visual Odometry", ECCV 2024. https://github.com/uzh-rpg/rl_vo]

import os
import re
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# EuRoC left camera intrinsics [fx, fy, cx, cy]
INTRINSICS = np.array([458.654, 457.296, 367.215, 248.375], dtype=np.float32)


class EuRoC(Dataset):
    """
    EuRoC dataset loader for evaluation.
    Returns one frame at a time with its ground truth pose and intrinsics.

    Expected directory structure:
        datapath/
            MH01/
                mav0/
                    cam0/data/          <- images
                    cam0/data.csv       <- timestamps + filenames
                    state_groundtruth_estimate0/data.csv  <- ground truth poses
    """

    def __init__(self, datapath, sequence):
        seq_path = os.path.join(datapath, sequence, 'mav0')

        # Load image list
        cam_csv = os.path.join(seq_path, 'cam0', 'data.csv')
        cam_data = np.loadtxt(cam_csv, delimiter=',', dtype=str, skiprows=1)
        img_timestamps = cam_data[:, 0].astype(np.int64)
        img_files = [
            os.path.join(seq_path, 'cam0', 'data', f)
            for f in cam_data[:, 1]
        ]

        # Load ground truth poses. state_groundtruth_estimate0 is the Vicon/Leica
        # tracked BODY (IMU) frame in world coordinates -- NOT cam0's pose. EuRoC
        # ships a separate body-to-cam0 extrinsic (T_BS, in cam0/sensor.yaml) that
        # must be composed in, or every "camera pose" is off by that fixed
        # extrinsic rotation (~90 deg for the EuRoC VI-sensor rig). This mostly
        # doesn't show up in Sim(3)-aligned ATE (a single global rotation offset
        # gets absorbed by the alignment step), but it fully corrupts any
        # per-pair relative-pose metric (e.g. Table 5's matching AUC), which has
        # no such alignment to hide behind.
        gt_csv = os.path.join(
            seq_path, 'state_groundtruth_estimate0', 'data.csv')
        gt_data = np.loadtxt(gt_csv, delimiter=',', skiprows=1)
        gt_timestamps = gt_data[:, 0].astype(np.int64)
        # Raw format: [px, py, pz, qw, qx, qy, qz]
        # Reorder to:  [px, py, pz, qx, qy, qz, qw]
        gt_poses = gt_data[:, 1:8][:, [0, 1, 2, 4, 5, 6, 3]].astype(np.float32)

        # Match each image to the nearest ground truth pose by timestamp
        self.image_paths = img_files
        body_poses = self._associate(img_timestamps, gt_timestamps, gt_poses)
        sensor_yaml_path = os.path.join(seq_path, 'cam0', 'sensor.yaml')
        T_BS = self._read_T_BS(sensor_yaml_path)
        self.poses = self._apply_extrinsics(body_poses, T_BS)
        self.intrinsics = np.tile(INTRINSICS, (len(img_files), 1))

        # EuRoC ships RAW (distorted) images -- cam0/sensor.yaml's pinhole
        # intrinsics are only valid for the rectified image. Without undistorting
        # first, every pixel coordinate used by GIM's matching/PnP and the BA
        # layer is systematically off (worse near the edges), which corrupts
        # ATE across the whole dataset. KITTI's odometry benchmark images ship
        # pre-rectified, so this only affects EuRoC.
        K, dist, resolution = self._read_camera_params(sensor_yaml_path)
        w, h = resolution
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            K, dist, None, K, (w, h), cv2.CV_32FC1)

    def _read_T_BS(self, sensor_yaml_path):
        with open(sensor_yaml_path) as f:
            text = f.read()
        m = re.search(r'T_BS:.*?data:\s*\[([^\]]+)\]', text, re.S)
        vals = [float(x) for x in m.group(1).split(',')]
        return np.array(vals, dtype=np.float64).reshape(4, 4)

    def _read_camera_params(self, sensor_yaml_path):
        with open(sensor_yaml_path) as f:
            text = f.read()
        fu, fv, cu, cv_ = (float(x) for x in re.search(
            r'intrinsics:\s*\[([^\]]+)\]', text).group(1).split(','))
        K = np.array([[fu, 0, cu], [0, fv, cv_], [0, 0, 1]], dtype=np.float64)
        dist = np.array([float(x) for x in re.search(
            r'distortion_coefficients:\s*\[([^\]]+)\]', text).group(1).split(',')],
            dtype=np.float64)
        w, h = (int(x) for x in re.search(
            r'resolution:\s*\[([^\]]+)\]', text).group(1).split(','))
        return K, dist, (w, h)

    def _apply_extrinsics(self, body_poses, T_BS):
        # body_poses: (N,7) [px,py,pz,qx,qy,qz,qw], T_world<-body.
        # T_BS = T_body<-cam0. T_world<-cam0 = T_world<-body @ T_body<-cam0.
        out = np.zeros_like(body_poses)
        for i in range(len(body_poses)):
            px, py, pz, qx, qy, qz, qw = body_poses[i]
            T_wb = np.eye(4)
            T_wb[:3, :3] = self._quat_to_rot(qx, qy, qz, qw)
            T_wb[:3, 3] = [px, py, pz]
            T_wc = T_wb @ T_BS
            out[i, :3] = T_wc[:3, 3]
            out[i, 3:] = self._rot_to_quat(T_wc[:3, :3])
        return out

    @staticmethod
    def _quat_to_rot(qx, qy, qz, qw):
        n = qx * qx + qy * qy + qz * qz + qw * qw
        s = 2.0 / n if n > 0 else 0.0
        return np.array([
            [1 - s * (qy * qy + qz * qz), s * (qx * qy - qw * qz), s * (qx * qz + qw * qy)],
            [s * (qx * qy + qw * qz), 1 - s * (qx * qx + qz * qz), s * (qy * qz - qw * qx)],
            [s * (qx * qz - qw * qy), s * (qy * qz + qw * qx), 1 - s * (qx * qx + qy * qy)],
        ])

    @staticmethod
    def _rot_to_quat(R):
        # same formula as mambavo/data_readers/kitti.py's _rot_to_quat, kept in sync
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
        return np.array([qx, qy, qz, qw], dtype=np.float64)

    def _associate(self, img_ts, gt_ts, gt_poses):
        poses = []
        for t in img_ts:
            idx = np.argmin(np.abs(gt_ts - t))
            poses.append(gt_poses[idx])
        return np.array(poses, dtype=np.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # EuRoC images are grayscale, convert to 3-channel for feature extractors
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32)
        image = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]

        pose = torch.from_numpy(self.poses[idx])         # [7]
        intrinsics = torch.from_numpy(self.intrinsics[idx])  # [4]

        return image, pose, intrinsics

    def get_all_poses(self):
        """Return all ground truth poses as a tensor, used for ATE evaluation."""
        return torch.from_numpy(self.poses)
