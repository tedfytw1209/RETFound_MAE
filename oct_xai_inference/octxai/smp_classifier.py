"""SMP classifier: a segmentation network (encoder + decoder) with a classification head.

Vendored from RETFound_MAE/SMP/smp_classifier.py (SMPClassifier, ConvGAPHead). Attribute names
(seg_model, head, ...) must stay identical to the original so the fine-tuned checkpoints load by
state_dict key. Differences from the original: the training-time Config paths are dropped, and
construction never reads a segmentation checkpoint (the fine-tuned checkpoint already contains
the segmentation weights) and never downloads encoder weights.

mode:
  - "enc"  : encoder features -> conv head
  - "dec"  : decoder features -> conv head
  - "fuse" : encoder + decoder features -> GeneralFusionHead (the SMP-fuse model)
"""

from typing import Optional, Tuple

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion_head import GeneralFusionHead

_SUPPORTED_ARCHS = ("DeepLabV3", "DeepLabV3Plus", "Unet", "UnetPlusPlus", "FPN",
                    "Linknet", "PSPNet", "MAnet", "PAN")


class ConvGAPHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        if in_ch is None:
            raise ValueError("in_ch must be specified, cannot be None")
        self.dropout = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()
        self.cls = nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=bias)

    def forward(self, fmap: torch.Tensor):
        x = self.dropout(fmap)
        logits_map = self.cls(x)
        logits = F.adaptive_avg_pool2d(logits_map, 1).flatten(1)
        return logits, logits_map


class SMPClassifier(nn.Module):
    def __init__(
        self,
        seg_arch: str = "Unet",
        encoder_name: str = "resnet50",
        in_channels: int = 3,
        num_classes: int = 2,
        seg_classes: int = 9,
        seg_activation: Optional[str] = "softmax",
        mode: str = "fuse",
        fuse_mode: str = "weighted_sum",
        learnable_alpha: bool = True,
        alpha: float = 0.5,
        alpha_type: str = "scalar",
        dropout: float = 0.0,
        size_match: str = "decoder_to_encoder",
        use_mask: bool = False,
        mask_softmax: bool = False,
        ignore_background: bool = False,
        fusion_dim: Optional[int] = None,
        align: str = "pre",
        enc_idx: int = -1,
        dec_idx: int = -1,
        smp_classifier: str = "linear",
    ):
        super().__init__()
        if mode not in ("enc", "dec", "fuse"):
            raise ValueError(f"mode must be 'enc', 'dec', or 'fuse', got {mode}")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha should be in (0, 1), got {alpha}")
        if seg_arch not in _SUPPORTED_ARCHS:
            raise ValueError(f"Unsupported seg_arch: {seg_arch}. Supported: {list(_SUPPORTED_ARCHS)}")

        self.mode, self.fuse_mode = mode, fuse_mode
        self.seg_arch, self.learnable_alpha = seg_arch, learnable_alpha
        self.alpha = alpha
        self.alpha_type = alpha_type
        self.dropout = dropout
        self.encoder_name = encoder_name
        self.fusion_dim = fusion_dim
        self.align = align
        self.size_match = size_match
        self.use_mask = use_mask
        self.ignore_background = ignore_background
        self.mask_softmax = mask_softmax
        self.enc_idx, self.dec_idx = enc_idx, dec_idx
        self.smp_classifier = smp_classifier

        self.seg_model = getattr(smp, seg_arch)(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=seg_classes,
            activation=seg_activation,
        )

        enc_chs = list(self.seg_model.encoder.out_channels)
        self.enc_last_ch = int(enc_chs[self.enc_idx])
        if use_mask and ignore_background:
            self.dec_out_ch = seg_classes - 1
        elif use_mask:
            self.dec_out_ch = seg_classes
        else:
            with torch.no_grad():
                dummy = torch.randn(1, in_channels, 64, 64)
                dec_out = self.seg_model.decoder(self.seg_model.encoder(dummy))
                if isinstance(dec_out, (list, tuple)):
                    dec_out = dec_out[self.dec_idx]
                self.dec_out_ch = dec_out.shape[1]

        if self.mode == "enc":
            self.head = ConvGAPHead(self.enc_last_ch, num_classes, bias=False, dropout=dropout)
        elif self.mode == "dec":
            self.head = ConvGAPHead(self.dec_out_ch, num_classes, bias=False, dropout=dropout)
        else:
            self.head = GeneralFusionHead(
                enc_channels=self.enc_last_ch,
                dec_channels=self.dec_out_ch,
                num_classes=num_classes,
                merge_method=fuse_mode,
                pooling="gap",
                fusion_dim=fusion_dim,
                align=align,
                learnable_alpha=learnable_alpha,
                alpha_init=alpha,
                alpha_type=alpha_type,
                size_match=size_match,
                resize_backend="interpolate",
                channel_multiply_ignore_background=True,
                classifier_dropout=dropout,
                classifier_bias=False,
                use_mask=use_mask,
                smp_classifier=smp_classifier,
            )

    def _get_enc_last(self, x):
        return self.seg_model.encoder(x)[-1]

    def _get_dec_last(self, x):
        dec = self.seg_model.decoder(self.seg_model.encoder(x))
        if isinstance(dec, (list, tuple)):
            return dec[-1]
        if self.use_mask:
            dec = self.seg_model.segmentation_head(dec)
            if self.mask_softmax:
                dec = F.softmax(dec, dim=1)
            if self.ignore_background:
                dec = dec[:, 1:, :, :]
        return dec

    def _get_enc_and_dec(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder and decoder features from a single encoder pass."""
        enc_feats = self.seg_model.encoder(x)
        final_enc = enc_feats[self.enc_idx]
        dec = self.seg_model.decoder(enc_feats)
        final_dec = dec[self.dec_idx] if isinstance(dec, (list, tuple)) else dec
        if self.use_mask:
            final_dec = self.seg_model.segmentation_head(final_dec)
            if self.mask_softmax:
                final_dec = F.softmax(final_dec, dim=1)
            if self.ignore_background:
                final_dec = final_dec[:, 1:, :, :]
        return final_enc, final_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class logits (B, num_classes)."""
        if self.mode == "enc":
            return self.head(self._get_enc_last(x))[0]
        if self.mode == "dec":
            return self.head(self._get_dec_last(x))[0]
        f_enc, f_dec = self._get_enc_and_dec(x)
        return self.head(enc_feats=f_enc, dec_feats=f_dec, return_fused_feature=False)
