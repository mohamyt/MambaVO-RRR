import torch
import cv2
import numpy as np

from .device_utils import get_default_device

class MetricDepthEstimator:
    # NOTE: device hardcoded to "cpu" (Mac, no NVIDIA GPU). GPU users can pass device="cuda".
    # NOTE: CPU-only setup required patching the Metric3D source in torch.hub cache
    # (mono/model/decode_heads/RAFTDepthNormalDPTDecoder5.py, get_bins() hardcodes device="cuda").
    # GPU users do not need this patch. See GIM README for details.
    def __init__(self, model_name="metric3d_vit_small", device="cpu", input_size=(616, 1064)):
        self.device = device if device is not None else get_default_device()
        self.input_size = input_size  # (H, W), keep-aspect-ratio target for ViT model

        self.model = torch.hub.load('yvanyin/metric3d', model_name, pretrain=True)
        self.model.to(self.device)
        self.model.eval()

        # normalization uses 0-255 pixel range (not 0-1 like DINOv2)
        self.mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        self.std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        self.padding_value = [123.675, 116.28, 103.53]

    # NOTE: accepts either a file path (str) or a preloaded tensor [N,3,H,W]
    # in [0,1] range.
    def preprocess(self, image):
        if isinstance(image, str):
            rgb_bgr = cv2.imread(image)
            if rgb_bgr is None:
                raise FileNotFoundError(f"Image not found: {image}")
            rgb_origin = rgb_bgr[:, :, ::-1].copy()  # BGR -> RGB
            rgb_tensor_raw = torch.from_numpy(rgb_origin.transpose((2, 0, 1))).float()[None] / 255.
            original_hw = rgb_origin.shape[:2]
        elif torch.is_tensor(image):
            rgb_tensor_raw = image.float()
            original_hw = (image.shape[2], image.shape[3])
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        h, w = original_hw
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = torch.nn.functional.interpolate(rgb_tensor_raw, size=(new_h, new_w), mode="bilinear", align_corners=False)

        pad_h = self.input_size[0] - new_h
        pad_w = self.input_size[1] - new_w
        pad_h_half, pad_w_half = pad_h // 2, pad_w // 2
        pad_val = torch.tensor(self.padding_value, device=resized.device).view(1, 3, 1, 1) / 255.
        padded = torch.nn.functional.pad(resized, (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half))
        pad_mask = torch.nn.functional.pad(torch.ones_like(resized), (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half))
        padded = padded * pad_mask + pad_val * (1 - pad_mask)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        rgb_tensor = padded * 255.
        rgb_tensor = torch.div((rgb_tensor - self.mean.to(padded.device)), self.std.to(padded.device))
        rgb_tensor = rgb_tensor.to(self.device)

        return rgb_tensor, pad_info, original_hw, scale


    # NOTE: K is the (3,3) camera intrinsics matrix. Metric3D's canonical-space
    # depth is converted to metric depth via fx/1000, but that fx must be the
    # focal length AFTER preprocess()'s resize to the model's fixed input_size,
    # not the dataset's raw fx -- otherwise the metric scale is off by exactly
    # the resize ratio, which differs per dataset (aspect ratio dependent) and
    # was previously silently using the unscaled fx.
    @torch.no_grad()
    def estimate(self, image_path, K):
        rgb_tensor, pad_info, original_hw, scale = self.preprocess(image_path)

        # no_grad already, so autocast here is a pure inference speedup -- no backward
        # pass touches this, nothing to worry about numerically.
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(self.device).startswith("cuda")):
            pred_depth, confidence, output_dict = self.model.inference({'input': rgb_tensor})
        pred_depth = pred_depth.float()

        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[
            pad_info[0]: pred_depth.shape[0] - pad_info[1],
            pad_info[2]: pred_depth.shape[1] - pad_info[3]
        ]
        pred_depth = torch.nn.functional.interpolate(
            pred_depth[None, None, :, :], original_hw, mode='bilinear'
        ).squeeze()

        fx = K[0, 0] * scale
        canonical_to_real_scale = fx / 1000.0
        pred_depth_metric = pred_depth * canonical_to_real_scale
        pred_depth_metric = torch.clamp(pred_depth_metric, 0, 300)

        return pred_depth_metric, confidence
