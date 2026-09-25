import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    # NOTE: DINOv2 patch features and G_t from EfficientLoFTR are projected to a shared 
    # dimension, then summed to form F_t for GMM.
    def __init__(self, dino_dim=384, loftr_dim=64, fusion_dim=384):
        super().__init__()
        self.dino_proj = nn.Linear(dino_dim, fusion_dim)
        self.loftr_proj = nn.Linear(loftr_dim, fusion_dim)
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, dino_feat, g_t):
        if not torch.is_tensor(dino_feat):
            dino_feat = torch.tensor(dino_feat, dtype=torch.float32)
        if not torch.is_tensor(g_t):
            g_t = torch.tensor(g_t, dtype=torch.float32)

        fused = self.dino_proj(dino_feat) + self.loftr_proj(g_t)
        return self.norm(fused)  # F_t: [N, fusion_dim]