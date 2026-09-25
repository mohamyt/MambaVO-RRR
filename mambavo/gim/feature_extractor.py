import torch
from torchvision import transforms
from PIL import Image

from .device_utils import get_default_device

class DINOv2FeatureExtractor:
    def __init__(self, model_name="dinov2_vits14", device="cpu", input_size=(224, 224)):
        self.device = device if device is not None else get_default_device()
        self.patch_size = 14
        self.input_size = input_size

        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0)

    # NOTE: accepts either a file path (str) or a preloaded tensor [N,3,H,W]
    # in [0,1] range (e.g. from a training dataloader's ToTensor output).
    def _load_image(self, image):
        if isinstance(image, str):
            return self.preprocess(image)

        if torch.is_tensor(image):
            resized = torch.nn.functional.interpolate(
                image, size=self.input_size, mode="bilinear", align_corners=False
            )
            mean = torch.tensor([0.485, 0.456, 0.406], device=image.device)[None, :, None, None]
            std = torch.tensor([0.229, 0.224, 0.225], device=image.device)[None, :, None, None]
            return (resized - mean) / std

        raise TypeError(f"Unsupported image input type: {type(image)}")

    # NOTE: pass `grid` (a previously-returned grid, e.g. from extract(image, points=None))
    # to sample new points against an already-computed feature grid instead of rerunning
    # the ViT forward pass -- useful when the same image is sampled multiple times per step
    # (e.g. a reference frame matched against several other frames).
    @torch.no_grad()
    def extract(self, image, points=None, original_hw=None, grid=None):
        if grid is None:
            image_tensor = self._load_image(image).to(self.device)
            # no_grad already, so autocast here is a pure inference speedup -- no backward
            # pass touches this, nothing to worry about numerically.
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(self.device).startswith("cuda")):
                features = self.model.forward_features(image_tensor)
            patch_tokens = features["x_norm_patchtokens"].float()

            h, w = self.input_size
            grid_h, grid_w = h // self.patch_size, w // self.patch_size
            B, N, C = patch_tokens.shape
            grid = patch_tokens.reshape(B, grid_h, grid_w, C)  # [1, gh, gw, C]

        if points is None:
            return grid

        if original_hw is None:
            raise ValueError("original_hw is required when points is provided")

        grid = grid.permute(0, 3, 1, 2)  # [1, C, gh, gw]

        h_orig, w_orig = original_hw
        h_in, w_in = self.input_size
        scale_x = w_in / w_orig
        scale_y = h_in / h_orig

        pts = torch.tensor(points, dtype=torch.float32, device=self.device)
        px = pts[:, 0] * scale_x
        py = pts[:, 1] * scale_y

        norm_x = (px / w_in) * 2 - 1  # normalize to [-1, 1] for grid_sample
        norm_y = (py / h_in) * 2 - 1

        sample_grid = torch.stack([norm_x, norm_y], dim=-1)[None, :, None, :]  # [1, N, 1, 2]
        sampled = torch.nn.functional.grid_sample(grid, sample_grid, align_corners=True)  # [1, C, N, 1]
        sampled = sampled.squeeze(-1).squeeze(0).permute(1, 0)  # [N, C]

        return sampled.cpu().numpy()
