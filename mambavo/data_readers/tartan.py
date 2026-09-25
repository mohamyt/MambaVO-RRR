#Adapted from: [Teed, Lipson, Deng, "Deep Patch Visual Odometry," NeurIPS 36, 2023 — github.com/princeton-vl/DPVO] 
# and [Messikommer et al., "Reinforcement Learning Meets Visual Odometry", ECCV 2024. https://github.com/uzh-rpg/rl_vo]


import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from .augmentation import RGBDAugmentor

# Scenes reserved for validation, excluded from training
TEST_SCENES = [
    "abandonedfactory/abandonedfactory/Easy/P011",
    "abandonedfactory/abandonedfactory/Hard/P011",
    "abandonedfactory_night/abandonedfactory_night/Easy/P013",
    "abandonedfactory_night/abandonedfactory_night/Hard/P014",
    "amusement/amusement/Easy/P008",
    "amusement/amusement/Hard/P007",
    "carwelding/carwelding/Easy/P007",
    "endofworld/endofworld/Easy/P009",
    "gascola/gascola/Easy/P008",
    "gascola/gascola/Hard/P009",
    "hospital/hospital/Easy/P036",
    "hospital/hospital/Hard/P049",
    "japanesealley/japanesealley/Easy/P007",
    "japanesealley/japanesealley/Hard/P005",
    "neighborhood/neighborhood/Easy/P021",
    "neighborhood/neighborhood/Hard/P017",
    "ocean/ocean/Easy/P013",
    "ocean/ocean/Hard/P009",
    "office2/office2/Easy/P011",
    "office2/office2/Hard/P010",
    "office/office/Hard/P007",
    "oldtown/oldtown/Easy/P007",
    "oldtown/oldtown/Hard/P008",
    "seasidetown/seasidetown/Easy/P009",
    "seasonsforest/seasonsforest/Easy/P011",
    "seasonsforest/seasonsforest/Hard/P006",
    "seasonsforest_winter/seasonsforest_winter/Easy/P009",
    "seasonsforest_winter/seasonsforest_winter/Hard/P018",
    "soulcity/soulcity/Easy/P012",
    "soulcity/soulcity/Hard/P009",
    "westerndesert/westerndesert/Easy/P013",
    "westerndesert/westerndesert/Hard/P007",
]

DEPTH_SCALE = 5.0
INTRINSICS = np.array([320.0, 320.0, 320.0, 240.0], dtype=np.float32)


class TartanAir(Dataset):
    """
    TartanAir dataset loader for training.
    Returns clips of n_frames consecutive frames per sample.
    """

    def __init__(self, datapath, n_frames=4, crop_size=[480, 640], aug=True):
        self.n_frames = n_frames
        self.aug = RGBDAugmentor(crop_size=crop_size) if aug else None

        # Collect all valid (scene, start_index) pairs
        self.samples = []
        scenes = sorted(glob.glob(os.path.join(datapath, '*/*/*/*')))
        for scene in scenes:
            # Skip test scenes
            if any(t in scene for t in TEST_SCENES):
                continue

            images = sorted(glob.glob(os.path.join(scene, 'image_left/*.png')))
            depths = sorted(glob.glob(os.path.join(scene, 'depth_left/*.npy')))
            pose_file = os.path.join(scene, 'pose_left.txt')

            if len(images) != len(depths) or not os.path.exists(pose_file):
                continue
            if len(images) < n_frames:
                continue

            poses = np.loadtxt(pose_file, delimiter=' ')
            # Reorder pose axes from TartanAir convention to standard
            poses = poses[:, [1, 2, 0, 4, 5, 3, 6]].astype(np.float32)
            poses[:, :3] /= DEPTH_SCALE

            for i in range(len(images) - n_frames + 1):
                self.samples.append((images, depths, poses, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        images_list, depths_list, poses, start = self.samples[idx]
        inds = range(start, start + self.n_frames)

        images, depths, pose_seq = [], [], []
        for i in inds:
            img = cv2.imread(images_list[i])
            images.append(img.astype(np.float32))

            d = np.load(depths_list[i]).astype(np.float32) / DEPTH_SCALE
            d[~np.isfinite(d)] = 1.0
            depths.append(d)

            pose_seq.append(poses[i])

        # Stack and convert to tensors
        images = torch.from_numpy(
            np.stack(images)).permute(0, 3, 1, 2)  # [N, 3, H, W]
        disps = torch.from_numpy(
            1.0 / np.stack(depths))                # [N, H, W]
        poses = torch.from_numpy(np.stack(pose_seq))  # [N, 7]
        intrinsics = torch.from_numpy(
            np.tile(INTRINSICS, (self.n_frames, 1)))  # [N, 4]

        if self.aug:
            images, poses, disps, intrinsics = self.aug(
                images, poses, disps, intrinsics)

        # Normalize disparity
        s = 0.7 * torch.quantile(disps, 0.98)
        disps = disps / s
        poses[:, :3] *= s

        return images, poses, disps, intrinsics
