import sys
import os
from copy import deepcopy

import torch
import cv2

from .device_utils import get_default_device

class EfficientLoFTRMatcher:
    # NOTE: depends on external repo zju3dv/EfficientLoFTR (not bundled in this
    # project repo) and a checkpoint that must be downloaded manually (not via
    # torch.hub). See GIM README for setup steps.

    def __init__(self, repo_path, weights_path, device="cpu"):
        sys.path.append(repo_path)
        from src.loftr import LoFTR, full_default_cfg, reparameter

        self.device = device if device is not None else get_default_device()

        cfg = deepcopy(full_default_cfg)
        self.matcher = LoFTR(config=cfg)
        self.matcher.load_state_dict(torch.load(weights_path, map_location=self.device)['state_dict'])
        self.matcher = reparameter(self.matcher)  # required step, do not skip
        self.matcher = self.matcher.eval().to(self.device)
        self._captured_feat_f0 = None
        self.matcher.fine_matching.register_forward_hook(self._capture_fine_feat)

    def _capture_fine_feat(self, module, inputs, output):
        # inputs[0] = feat_f0_unfold, shape [M, WW, C=64]
        self._captured_feat_f0 = inputs[0].detach()

    # NOTE: accepts either a file path (str) or a preloaded tensor [N,3,H,W]
    # in [0,1] range. Converts to grayscale if given a tensor (EfficientLoFTR
    # expects single-channel input).
    def _load_gray(self, image):
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {image}")
            img = cv2.resize(img, (img.shape[1] // 32 * 32, img.shape[0] // 32 * 32))
            return torch.from_numpy(img)[None][None].to(self.device) / 255.

        if torch.is_tensor(image):
            weights = torch.tensor([0.299, 0.587, 0.114], device=image.device).view(1, 3, 1, 1)
            gray = (image * weights).sum(dim=1, keepdim=True)
            h, w = gray.shape[2], gray.shape[3]
            new_h, new_w = h // 32 * 32, w // 32 * 32
            gray = torch.nn.functional.interpolate(gray, size=(new_h, new_w), mode="bilinear", align_corners=False)
            return gray.to(self.device)

        raise TypeError(f"Unsupported image input type: {type(image)}")

    @torch.no_grad()
    def match(self, image0, image1):

        img0 = self._load_gray(image0)
        img1 = self._load_gray(image1)
        batch = {'image0': img0, 'image1': img1}
        batch = {'image0': img0, 'image1': img1}

        # no_grad already, so autocast here is a pure inference speedup -- no backward
        # pass touches this, nothing to worry about numerically.
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(self.device).startswith("cuda")):
            self.matcher(batch)

        mkpts0 = batch['mkpts0_f'].float().cpu().numpy()
        mkpts1 = batch['mkpts1_f'].float().cpu().numpy()
        mconf = batch['mconf'].float().cpu().numpy()

        window_center = self._captured_feat_f0.shape[1] // 2
        g_t = self._captured_feat_f0[:, window_center, :].float().cpu().numpy()

        return mkpts0, mkpts1, mconf, g_t
