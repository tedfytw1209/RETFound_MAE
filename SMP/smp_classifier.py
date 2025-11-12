from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

__all__ = ["SimpleSMPClassifier"]

_DEC_TOP_RULES = {
    "DeepLabV3": 256, "DeepLabV3Plus": 256,
    "Unet": "enc1", "UnetPlusPlus": "enc1", "FPN": "enc1",
    "Linknet": "enc1", "PSPNet": "enc1", "MAnet": "enc1", "PAN": "enc1",
}
def _infer_decoder_out_ch(seg_arch: str, enc_chs) -> int:
    rule = _DEC_TOP_RULES.get(str(seg_arch), "enc1")
    return int(enc_chs[0] if rule == "enc1" else rule)

class ConvGAPHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        if in_ch is None:
            raise ValueError("in_ch must be specified, cannot be None")
        self.dropout = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()
        self.cls = nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=bias)
    
    @torch.no_grad()
    def _norm_cam(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.amin(dim=(-2, -1), keepdim=True)
        return x / x.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    
    def forward(self, fmap: torch.Tensor):
        x = self.dropout(fmap)
        logits_map = self.cls(x)
        logits = F.adaptive_avg_pool2d(logits_map, 1).flatten(1)
        cam = self._norm_cam(logits_map)
        return logits, logits_map, cam

class SMPClassifier(nn.Module):
    """
    mode:
      - "enc"  : encoder
      - "dec"  : decoder
      - "fuse" : encoder and decoder
    fuse_mode:
       - "sum"  : f = α * f_enc + (1 - α) * Align1x1(f_dec), learnable_alpha = True/False
       - "concat": concatenate along channel dim
    """
    def __init__(
        self,
        seg_arch: str = "Unet",
        encoder_name: str = "resnet50",
        encoder_weights: Optional[str] = None,
        in_channels: int = 3,
        num_classes: int = 2,

        mode: str = "enc",
        decoder_out_ch: Optional[int] = None,
        fuse_mode: str = "sum",
        fuse_dim: Optional[int] = None,

        learnable_alpha: bool = True,
        alpha: float = 0.5,

        pretrained_seg_ckpt: Optional[str] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert mode in ("enc", "dec", "fuse"), f"mode must be 'enc', 'dec', or 'fuse', got {mode}"
        assert fuse_mode in ("sum", "concat"), f"fuse_mode must be 'sum' or 'concat', got {fuse_mode}"
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha should be in (0, 1), got {alpha}")
        if seg_arch not in _DEC_TOP_RULES:
            raise ValueError(f"Unsupported seg_arch: {seg_arch}. Supported: {list(_DEC_TOP_RULES.keys())}")
        
        self.mode, self.fuse_mode = mode, fuse_mode
        self.seg_arch, self.learnable_alpha = seg_arch, learnable_alpha

        SegCls = getattr(smp, seg_arch)
        self.seg_model = SegCls(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
        if pretrained_seg_ckpt is not None:
            sd = torch.load(pretrained_seg_ckpt, map_location="cpu")
            sd = sd.get("state_dict", sd)
            self.seg_model.load_state_dict(sd, strict=False)

        self.encoder = self.seg_model.encoder
        enc_chs = list(self.encoder.out_channels)
        self.enc_last_ch = int(enc_chs[-1])
        
        # Infer decoder output channels by running a dummy forward pass
        if decoder_out_ch is None:
            with torch.no_grad():
                dummy = torch.randn(1, in_channels, 64, 64)
                enc_feats = self.encoder(dummy)
                print(enc_feats) #encoder is each layers output [x, layer1, layer2, layer3, layer4...]
                dec_out = self.seg_model.decoder(enc_feats)
                if isinstance(dec_out, (list, tuple)):
                    dec_out = dec_out[-1]
                self.dec_out_ch = dec_out.shape[1]
        else:
            self.dec_out_ch = decoder_out_ch

        # Determine final feature channels based on mode
        if self.mode == "enc":
            final_ch = self.enc_last_ch
        elif self.mode == "dec":
            final_ch = self.dec_out_ch
        else:  # fuse
            if self.fuse_mode == "concat":
                final_ch = int(fuse_dim) if fuse_dim is not None else self.enc_last_ch
                # Input to fuse_proj is enc_last_ch + dec_out_ch
                self.fuse_proj = nn.Conv2d(self.enc_last_ch + self.dec_out_ch, final_ch, 1, bias=False)
            else:  # sum
                final_ch = self.enc_last_ch
                self.dec_align = nn.Conv2d(self.dec_out_ch, self.enc_last_ch, 1, bias=False)
                if self.learnable_alpha:
                    init_logit = math.log(alpha) - math.log(1 - alpha)
                    self.alpha_logit = nn.Parameter(torch.tensor([init_logit], dtype=torch.float32))
                else:
                    self.alpha = alpha

        self.head = ConvGAPHead(final_ch, num_classes, bias=False, dropout=dropout)

    def _get_enc_last(self, x): 
        return self.encoder(x)[-1]
    
    def _get_dec_last(self, x):
        enc_feats = self.encoder(x)
        dec = self.seg_model.decoder(enc_feats)
        # Handle different decoder output formats
        if isinstance(dec, (list, tuple)):
            return dec[-1]
        return dec
    
    def _get_enc_and_dec(self, x):
        """Efficiently compute both encoder and decoder features with single encoder pass."""
        enc_feats = self.encoder(x)
        f_enc = enc_feats[-1]
        dec = self.seg_model.decoder(enc_feats)
        f_dec = dec[-1] if isinstance(dec, (list, tuple)) else dec
        return f_enc, f_dec

    def forward(self, x: torch.Tensor, mode_dict=False) -> Dict[str, Dict[str, torch.Tensor]]:
        out: Dict[str, Dict[str, torch.Tensor]] = {}

        if self.mode == "enc":
            f = self._get_enc_last(x)
            logits, logits_map, cam = self.head(f)
            if mode_dict:
                out["enc"] = {"logits": logits, "logits_map": logits_map, "cam": cam}
                return out
            else:
                return logits

        if self.mode == "dec":
            f = self._get_dec_last(x)
            logits, logits_map, cam = self.head(f)
            if mode_dict:
                out["dec"] = {"logits": logits, "logits_map": logits_map, "cam": cam}
                return out
            else:
                return logits

        # --- fuse ---
        f_enc, f_dec = self._get_enc_and_dec(x)  # Use efficient single-pass method
        H, W = max(f_enc.shape[-2], f_dec.shape[-2]), max(f_enc.shape[-1], f_dec.shape[-1])
        if f_enc.shape[-2:] != (H, W): 
            f_enc = F.interpolate(f_enc, (H, W), mode="bilinear", align_corners=False)
        if f_dec.shape[-2:] != (H, W): 
            f_dec = F.interpolate(f_dec, (H, W), mode="bilinear", align_corners=False)

        if self.fuse_mode == "sum":
            f_dec_aligned = self.dec_align(f_dec)
            alpha = torch.sigmoid(self.alpha_logit) if self.learnable_alpha else self.alpha
            f = alpha * f_enc + (1 - alpha) * f_dec_aligned
        else:
            f = self.fuse_proj(torch.cat([f_enc, f_dec], dim=1))

        logits, logits_map, cam = self.head(f)
        if mode_dict:
            out["fuse"] = {"logits": logits, "logits_map": logits_map, "cam": cam}
            return out
        else:
            return logits