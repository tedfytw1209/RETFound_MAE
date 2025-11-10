import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from SAM2UNet.SAM2UNet import SAM2UNet

class GAPClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden: int = 512, p: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden), nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, feat):                 # feat: [B,C,H,W]
        z = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # [B,C]
        return self.mlp(z)

class SAM2UNetClassifier(nn.Module):
    def __init__(self, num_classes: int, seg_ckpt: str = None, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = SAM2UNet()
        if seg_ckpt and os.path.exists(seg_ckpt):
            sd = torch.load(seg_ckpt, map_location="cpu")
            state = sd.get("model", sd)
            self.backbone.load_state_dict(state, strict=False)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 512, 512)
            dec_feat = self.backbone(dummy, return_dec_feat=True)["dec_feat"]
            in_ch = dec_feat.shape[1]
        self.cls_head = GAPClassifier(in_channels=in_ch, num_classes=num_classes)

    def forward(self, x):
        out = self.backbone(x, return_dec_feat=True)
        dec = out["dec_feat"]             # [B,C,H',W']
        logits = self.cls_head(dec)       # [B,num_classes]
        return {"logits": logits, "dec_feat": dec, "seg_pred": out["pred"]}
