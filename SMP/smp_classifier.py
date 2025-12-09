from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from .special_header import GeneralFusionHead

__all__ = ["SimpleSMPClassifier"]

# Configuration
class Config:
    # Paths
    DATA_ROOT = "/data/tl28853/eye/"  # Root directory containing images
    TRAIN_CSV = "/data/tl28853/eye/OCTDL/dme_train.csv"  # CSV with image and label columns
    VAL_CSV = "/data/tl28853/eye/OCTDL/dme_test.csv"  # CSV with image and label columns
    CHECKPOINT_DIR = "/data/tl28853/eye/segmentation_models.pytorch/checkpoints_octdl_dme_dec"
    
    # Model parameters
    SEG_ARCH = 'Unet'  # Unet, UnetPlusPlus, FPN, Linknet, PSPNet, MAnet, PAN, DeepLabV3, DeepLabV3Plus
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    NUM_CLASSES = 2
    IN_CHANNELS = 3
    MODE = 'dec'  # enc, dec, fuse
    FUSE_MODE = 'sum'  # sum, concat
    LEARNABLE_ALPHA = False
    ALPHA = 0.5
    PRETRAINED_SEG_CKPT = '/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth'
    SEG_CLASSES = 9  # number of segmentation classes
    ACTIVATION = 'softmax'  # softmax for multiclass, sigmoid for binary
    DROPOUT = 0.0
    
    # Training parameters
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 20
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    IMAGE_SIZE = 512
    NUM_WORKERS = 4
    USE_AMP = True

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
    
    # Why first classify in conv2d?
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
        seg_classes: int = 1,
        seg_activation: Optional[str] = None,

        mode: str = "enc",
        decoder_out_ch: Optional[int] = None,
        fuse_mode: str = "weighted_sum",

        learnable_alpha: bool = True,
        alpha: float = 0.5,

        pretrained_seg_ckpt: Optional[str] = None,
        dropout: float = 0.0,
        size_match: str = "decoder_to_encoder",
        use_mask: bool = False,
        fusion_dim: Optional[int] = None,
        
        enc_idx: int = -1,
        dec_idx: int = -1,
    ):
        super().__init__()
        assert mode in ("enc", "dec", "fuse"), f"mode must be 'enc', 'dec', or 'fuse', got {mode}"
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha should be in (0, 1), got {alpha}")
        if seg_arch not in _DEC_TOP_RULES:
            raise ValueError(f"Unsupported seg_arch: {seg_arch}. Supported: {list(_DEC_TOP_RULES.keys())}")
        
        self.mode, self.fuse_mode = mode, fuse_mode
        self.seg_arch, self.learnable_alpha = seg_arch, learnable_alpha
        self.use_mask = use_mask
        self.enc_idx, self.dec_idx = enc_idx, dec_idx

        SegCls = getattr(smp, seg_arch)
        self.seg_model = SegCls(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=seg_classes,
            activation=seg_activation,
        )
        if pretrained_seg_ckpt is not None:
            sd = torch.load(pretrained_seg_ckpt, map_location="cpu")
            sd = sd.get("model_state_dict", sd)
            result = self.seg_model.load_state_dict(sd, strict=False)
            print("\n=== MISSING KEYS ===")
            print(result.missing_keys)
            print("\n=== UNEXPECTED KEYS ===")
            print(result.unexpected_keys)

        #self.encoder = self.seg_model.encoder
        enc_chs = list(self.seg_model.encoder.out_channels)
        self.enc_last_ch = int(enc_chs[self.enc_idx])
        
        # Infer decoder output channels by running a dummy forward pass
        if use_mask:
            self.dec_out_ch = seg_classes
        elif decoder_out_ch is None:
            with torch.no_grad():
                dummy = torch.randn(1, in_channels, 64, 64)
                enc_feats = self.seg_model.encoder(dummy)
                #encoder is each layers output [x, layer1, layer2, layer3, layer4...]
                dec_out = self.seg_model.decoder(enc_feats)
                if isinstance(dec_out, (list, tuple)):
                    dec_out = dec_out[self.dec_idx]
                self.dec_out_ch = dec_out.shape[1]
        else:
            self.dec_out_ch = decoder_out_ch

        # Determine final feature channels based on mode
        if self.mode == "enc":
            final_ch = self.enc_last_ch
            self.head = ConvGAPHead(final_ch, num_classes, bias=False, dropout=dropout)
        elif self.mode == "dec":
            final_ch = self.dec_out_ch
            self.head = ConvGAPHead(final_ch, num_classes, bias=False, dropout=dropout)
        else:  # fuse
            '''
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
            '''
            self.head = GeneralFusionHead(
                enc_channels=self.enc_last_ch,
                dec_channels=self.dec_out_ch,
                num_classes=num_classes,
                merge_method=self.fuse_mode,
                pooling="gap",
                fusion_dim=fusion_dim,
                learnable_alpha=self.learnable_alpha,
                alpha_init=alpha,
                size_match=size_match,
                resize_backend="interpolate",
                channel_multiply_ignore_background=True,
                classifier_dropout=dropout,
                classifier_bias=False,
                use_mask=self.use_mask,
            )

        

    def _get_enc_last(self, x): 
        return self.seg_model.encoder(x)[-1]
    
    def _get_dec_last(self, x):
        enc_feats = self.seg_model.encoder(x)
        dec = self.seg_model.decoder(enc_feats)
        # Handle different decoder output formats
        if isinstance(dec, (list, tuple)):
            return dec[-1]
        if self.use_mask:
            dec = self.seg_model.segmentation_head(dec)
        return dec
    
    def _get_enc_and_dec(self, x, enc_idx: int = -1, dec_idx: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Efficiently compute both encoder and decoder features with single encoder pass."""
        enc_feats = self.seg_model.encoder(x)
        if enc_idx != -1:
            enc_list = [enc_feats[i] for i in range(enc_idx,len(enc_feats))]
            first_enc_shape = enc_list[0].shape[2:]
            for i in range(len(enc_list)):
                if enc_list[i].shape[2:] != first_enc_shape:
                    enc_list[i] = F.interpolate(enc_list[i], size=first_enc_shape, mode='bilinear', align_corners=False)
            final_enc = torch.cat(enc_list, dim=1)
        else:
            final_enc = enc_feats[enc_idx]
        dec = self.seg_model.decoder(enc_feats)
        final_dec = dec[dec_idx] if isinstance(dec, (list, tuple)) else dec
        if self.use_mask:
            final_dec = self.seg_model.segmentation_head(final_dec)
        return final_enc, final_dec

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
        elif self.mode == "dec":
            f = self._get_dec_last(x)
            logits, logits_map, cam = self.head(f)
            if mode_dict:
                out["dec"] = {"logits": logits, "logits_map": logits_map, "cam": cam}
                return out
            else:
                return logits
        else:
            # --- fuse ---
            # Use GeneralFusionHead to combine encoder & decoder features.
            f_enc, f_dec = self._get_enc_and_dec(x, self.enc_idx, self.dec_idx)
            print('enc shape:', f_enc.shape)
            print('dec shape:', f_dec.shape)

            logits = self.head(
                enc_feats=f_enc,
                dec_feats=f_dec,
                return_fused_feature=False,
            )

            if mode_dict:
                out["fuse"] = {"logits": logits}
                return out
            else:
                return logits