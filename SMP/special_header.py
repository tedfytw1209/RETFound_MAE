import math
from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

class SegGatedClassifierHead(nn.Module):
    r"""
        SegGatedClassifierHead (Architecture A)

        Goal:
            Use the segmentation output to gate encoder features so that
            standard Grad-CAM applied on the gated feature automatically
            suppresses background responses.

        Notation (per sample, omitting batch dimension):
            - Encoder feature map:
                A^k ∈ R^{H × W}, k = 1,...,C
            - Segmentation logits:
                Z^{seg}_{p,m}, p ∈ Ω (pixels), m = 0,...,M
                m = 0 : background
                m ≥ 1 : foreground objects / structures
            - Segmentation probabilities:
                P^{seg}_{p,m} = softmax_m(Z^{seg}_{p,m})

            Define foreground weight:
                F_p = 1 - P^{seg}_{p,0} = ∑_{m=1}^M P^{seg}_{p,m}
                (optionally, F_p^(β) if you want stronger suppression)

            Segmentation-gated feature:
                B^k_p = F_p ⋅ A^k_p

            Global Average Pooling:
                g_k = (1 / Z) ∑_p B^k_p,   Z = H × W

            Linear classifier:
                y^c = b_c + ∑_k w_{c,k} ⋅ g_k

            Standard Grad-CAM on B:
                α_k^c = (1 / Z) ∑_p ∂y^c / ∂B^k_p
                L_GradCAM^c(p) = ReLU(∑_k α_k^c ⋅ B^k_p)

            Because F_p ≈ 0 on background pixels, both B^k_p and
            gradients there are strongly suppressed, so Grad-CAM
            naturally highlights only foreground regions.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        seg_channels: int,
        fg_power: float = 1.0,
        use_softmax: bool = True,
        pooling: str = "gap",
    ):
        """
        Args:
            in_channels:  number of channels of encoder feature A (C).
            num_classes:  number of classification classes.
            seg_channels: number of segmentation output channels (M+1),
                          including background channel at index 0.
            fg_power:    exponent β for foreground weight F^β (>= 1).
            use_softmax: if True, apply softmax over seg_channels to get
                         P^{seg}; if False, assume seg_logits already ~ probs.
            pooling:     'gap' (global average pooling) or 'gmp' (max pooling).
        """
        super().__init__()
        assert seg_channels >= 2, "seg_channels should be background + at least 1 object."
        assert pooling in ("gap", "gmp")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.seg_channels = seg_channels
        self.fg_power = fg_power
        self.use_softmax = use_softmax
        self.pooling = pooling

        # Simple linear classifier on pooled features
        self.classifier = nn.Linear(in_channels, num_classes)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] -> [B, C]
        """
        if self.pooling == "gap":
            return x.mean(dim=(2, 3))
        else:  # 'gmp'
            return x.amax(dim=(2, 3))

    def forward(
        self,
        feats: torch.Tensor,
        seg_logits: torch.Tensor,
        return_gated_feature: bool = False,
    ):
        """
        Args:
            feats:      encoder feature map A, shape [B, C, H, W].
            seg_logits: segmentation logits Z^{seg}, shape [B, M+1, H_s, W_s].
                        Channel 0 is background.
            return_gated_feature:
                        if True, also return the gated feature B,
                        which is the correct target layer for Grad-CAM.

        Returns:
            logits: [B, num_classes]
            gated_feats (optional): [B, C, H, W] B = F * A
        """
        B, C, H, W = feats.shape
        assert C == self.in_channels

        # 1) Resize segmentation logits to match feature spatial size if needed
        if seg_logits.shape[2:] != (H, W):
            seg_logits_resized = F.interpolate(
                seg_logits,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
        else:
            seg_logits_resized = seg_logits

        # 2) Compute segmentation probabilities P^{seg}
        if self.use_softmax:
            # P^{seg} ∈ [0,1], sum over channel = 1
            seg_probs = F.softmax(seg_logits_resized, dim=1)
        else:
            seg_probs = seg_logits_resized  # assume already probability-like

        # 3) Foreground weight F_p = 1 - P^{seg}_{p,0}
        #    seg_probs[:, 0, :, :] is background probability
        fg_weight = 1.0 - seg_probs[:, 0:1, :, :]  # [B, 1, H, W]

        if self.fg_power != 1.0:
            fg_weight = fg_weight.pow(self.fg_power)

        # 4) Segmentation-gated feature B^k_p = F_p ⋅ A^k_p
        gated_feats = feats * fg_weight  # broadcast over channel dimension

        # 5) Global pooling and classification
        pooled = self._pool(gated_feats)       # [B, C]
        logits = self.classifier(pooled)       # [B, num_classes]

        if return_gated_feature:
            # This gated_feats should be used as the Grad-CAM target feature.
            return logits, gated_feats
        else:
            return logits


class ObjectDecomposedClassifierHead(nn.Module):
    r"""
        ObjectDecomposedClassifierHead (Architecture B)

        Goal:
            Decompose the encoder feature map into object-specific feature groups,
            so that standard Grad-CAM applied to the concatenated feature
            can be interpreted per object by grouping channels.

        Notation (per sample, omitting batch dimension):
            - Encoder feature map:
                A^k ∈ R^{H × W}, k = 1,...,C
            - Segmentation logits:
                Z^{seg}_{p,m}, p ∈ Ω, m = 0,...,M
                m = 0 : background
                m ≥ 1 : foreground objects / structures
            - Segmentation probabilities:
                P^{seg}_{p,m} = softmax_m(Z^{seg}_{p,m})

            For each object class m = 1,...,M, define a (soft) object mask:
                M^{(m)}_p = P^{seg}_{p,m}

            Object-masked feature:
                A^{(m),k}_p = M^{(m)}_p ⋅ A^k_p

            We then stack these object features in the channel dimension:
                B^{(m-1)C + k}_p = A^{(m),k}_p
                ⇒ B^ℓ ∈ R^{H × W}, ℓ = 1,..., M⋅C

            Global Average Pooling:
                g_ℓ = (1 / Z) ∑_p B^ℓ_p,   Z = H × W

            Classifier:
                y^c = b_c + ∑_{ℓ=1}^{MC} w_{c,ℓ} ⋅ g_ℓ

            If we denote:
                ℓ ↔ (m, k)  where ℓ = (m-1)C + k,

            then we can interpret:
                y^c = b_c + ∑_{m=1}^{M} ∑_{k=1}^{C} w_{c,m,k} ⋅ g_{m,k}

            Standard Grad-CAM on B:
                α_ℓ^c = (1 / Z) ∑_p ∂y^c / ∂B^ℓ_p
                L_GradCAM^c(p) = ReLU(∑_{ℓ} α_ℓ^c ⋅ B^ℓ_p)

            Object-specific Grad-CAM can be recovered by grouping channels:
                For object m:
                    α_{m,k}^c := α_{(m-1)C + k}^c
                    L^{c,m}(p) = ReLU(∑_{k} α_{m,k}^c ⋅ B^{(m-1)C + k}_p)
                            = ReLU(∑_{k} α_{m,k}^c ⋅ M^{(m)}_p A^k_p)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        seg_channels: int,
        use_softmax: bool = True,
        pooling: str = "gap",
        ignore_background: bool = True,
    ):
        """
        Args:
            in_channels:   number of encoder feature channels C.
            num_classes:   number of classification classes.
            seg_channels:  number of segmentation channels (M+1), including background.
            use_softmax:   if True, apply softmax to seg_logits; else assume prob-like input.
            pooling:       'gap' (global average pooling) or 'gmp'.
            ignore_background:
                           if True, only use classes 1..M as objects.
                           If False, background (m=0) will be treated as an object as well.
        """
        super().__init__()
        assert seg_channels >= 2, "seg_channels should be background + at least 1 object."
        assert pooling in ("gap", "gmp")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.seg_channels = seg_channels
        self.use_softmax = use_softmax
        self.pooling = pooling
        self.ignore_background = ignore_background

        # Number of object classes we use for decomposition
        if ignore_background:
            self.num_objects = seg_channels - 1  # m = 1..M
            self.object_offset = 1               # skip channel 0
        else:
            self.num_objects = seg_channels      # m = 0..M
            self.object_offset = 0

        # Total channels after object decomposition: M * C (or (M+1)*C)
        self.decomposed_channels = self.num_objects * in_channels

        # Linear classifier on pooled decomposed feature
        self.classifier = nn.Linear(self.decomposed_channels, num_classes)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] -> [B, C]
        """
        if self.pooling == "gap":
            return x.mean(dim=(2, 3))
        else:
            return x.amax(dim=(2, 3))

    def forward(
        self,
        feats: torch.Tensor,
        seg_logits: torch.Tensor,
        return_decomposed_feature: bool = False,
    ):
        """
        Args:
            feats:      encoder feature map A, shape [B, C, H, W].
            seg_logits: segmentation logits Z^{seg}, shape [B, M+1, H_s, W_s].
                        Channel 0 is background.
            return_decomposed_feature:
                        if True, also return the decomposed feature B
                        with shape [B, M*C, H, W], which is the correct
                        target layer for Grad-CAM (with channel grouping).

        Returns:
            logits: [B, num_classes]
            decomposed_feats (optional): [B, M*C, H, W]
        """
        B, C, H, W = feats.shape
        assert C == self.in_channels

        # 1) Resize segmentation logits to feature spatial resolution
        if seg_logits.shape[2:] != (H, W):
            seg_logits_resized = F.interpolate(
                seg_logits,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
        else:
            seg_logits_resized = seg_logits

        # 2) Compute segmentation probabilities
        if self.use_softmax:
            seg_probs = F.softmax(seg_logits_resized, dim=1)  # [B, M+1, H, W]
        else:
            seg_probs = seg_logits_resized

        # 3) Select object channels (skip background if ignore_background=True)
        #    masks: [B, num_objects, H, W]
        masks = seg_probs[:, self.object_offset : self.object_offset + self.num_objects, :, :]

        # 4) Build object-masked feature:
        #    A^{(m),k}_p = M^{(m)}_p ⋅ A^k_p
        #
        #    Implementation trick:
        #      feats:  [B, C,   H, W]
        #      masks:  [B, M,   H, W]
        #    -> feats.unsqueeze(1): [B, 1, C, H, W]
        #    -> masks.unsqueeze(2): [B, M, 1, H, W]
        #    -> multiplied:        [B, M, C, H, W]
        #    -> reshape to:        [B, M*C, H, W]  (concatenate over channel)
        B_feats = (feats.unsqueeze(1) * masks.unsqueeze(2)).reshape(
            B, self.num_objects * C, H, W
        )  # [B, M*C, H, W]

        # 5) Global pooling and classification
        pooled = self._pool(B_feats)              # [B, M*C]
        logits = self.classifier(pooled)          # [B, num_classes]

        if return_decomposed_feature:
            # B_feats is the decomposed feature.
            # For Grad-CAM:
            #   - Hook this tensor as the target layer.
            #   - After obtaining α_ℓ^c and L^c(p),
            #     you can group channels:
            #       ℓ = (m-1)*C + k,  m ∈ {1..M}, k ∈ {1..C}
            #     to get object-specific heatmaps.
            return logits, B_feats
        else:
            return logits

#tmp code from smp
class Activation(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)

class SpatialGate(nn.Module):
    """Simple spatially-varying gate: cat(enc, dec) → Conv1×1 → ReLU → Conv1×1 → Sigmoid.

    Produces a per-pixel (H, W) alpha map in (0, 1).  Lightweight and fully
    convolutional; no global pooling, so every pixel gets its own gate value.

    Args:
        in_ch: input channels (= target_dim * 2)
        mid_ch: intermediate channel width (default 16)
    """
    def __init__(self, in_ch: int, mid_ch: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2C, H, W)  →  gate (B, 1, H, W)"""
        return self.net(x)


class SE_gate(nn.Module):
    """SE-style + CBAM spatial attention fusion gate.

    Combines two complementary attention paths:
      1. SE path (global channel context): GAP → FC → ReLU → FC → scalar logit (B,1,1,1)
         Squeezes global channel statistics into a per-image bias term.
      2. Spatial path (local structure): channel-avg + channel-max → Conv1×1 → ReLU → Conv1×1
         CBAM-style; produces a spatially varying logit map (B,1,H,W).

    Gate = Sigmoid(se_logit + spatial_logit)  →  (B, 1, H, W) ∈ (0, 1)

    Args:
        in_ch: number of input channels (= target_dim * 2, i.e. cat(enc, dec))
        reduction: channel reduction ratio for the SE path (default 4)
        spatial_mid: intermediate channels in the spatial conv path (default 16)
    """
    def __init__(self, in_ch: int, reduction: int = 4, spatial_mid: int = 16):
        super().__init__()
        r = max(in_ch // reduction, 8)
        # SE path: global average pool → FC → ReLU → FC → scalar gate logit
        self.se_fc = nn.Sequential(
            nn.Linear(in_ch, r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(r, 1, bias=False),
        )
        # Spatial path: channel avg + max pool → spatially varying logit map
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, spatial_mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_mid, 1, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2C, H, W)  →  gate (B, 1, H, W)"""
        # SE: global context → scalar logit bias (B, 1, 1, 1)
        se_logit = self.se_fc(x.mean(dim=(2, 3)))         # (B, 1)
        se_logit = se_logit.unsqueeze(-1).unsqueeze(-1)   # (B, 1, 1, 1)

        # Spatial: channel-pool stats → spatial logit (B, 1, H, W)
        avg_map = x.mean(dim=1, keepdim=True)             # (B, 1, H, W)
        max_map = x.amax(dim=1, keepdim=True)             # (B, 1, H, W)
        spatial_logit = self.spatial_conv(
            torch.cat([avg_map, max_map], dim=1)
        )                                                  # (B, 1, H, W)

        # SE global bias + spatial local detail, combined before sigmoid
        return torch.sigmoid(se_logit + spatial_logit)    # (B, 1, H, W)


class AttnGate(nn.Module):
    """Cross-attention gate: encoder global query attends over decoder spatial keys.

    Splits cat(enc, dec) back into two streams, then:
      Q = GAP(enc) → Linear(C, d)   → (B, d)          global "what to look for"
      K = Conv1×1(dec, d)           → (B, d, H, W)     local  "where is it"
      gate = Sigmoid(Q · K / √d)   → (B, 1, H, W)

    The encoder's global context selects which decoder spatial positions are
    task-relevant.  The resulting gate is directly interpretable as a spatial
    attention map, linking CAM faithfulness (encoder focus) to decoder alignment.

    Args:
        in_ch: total input channels = target_dim * 2 (enc + dec concatenated)
        dim: projection dimension for Q and K (default 32)
    """
    def __init__(self, in_ch: int, dim: int = 32):
        super().__init__()
        half = in_ch // 2              # = target_dim
        self.q_proj = nn.Linear(half, dim, bias=False)
        self.k_proj = nn.Conv2d(half, dim, kernel_size=1, bias=False)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2C, H, W)  →  gate (B, 1, H, W)"""
        half = x.shape[1] // 2
        enc, dec = x[:, :half], x[:, half:]              # (B, C, H, W) each

        q = self.q_proj(enc.mean(dim=(2, 3)))             # (B, d)
        k = self.k_proj(dec)                              # (B, d, H, W)

        # Dot product: q (B,d) · k (B,d,H,W) → (B,H,W)
        attn = torch.einsum('bd,bdhw->bhw', q, k) * self.scale
        return torch.sigmoid(attn.unsqueeze(1))           # (B, 1, H, W)


class GeneralFusionHead(nn.Module):
    """
    General header that fuses encoder and decoder features before pooling.

    The design targets ablations requested in the header selection study:
        1. Merge method: weighted sum, element-wise add, channel merge, channel-wise mask multiply, or element-wise multiply.
        2. Optional segmentation-driven gating on decoder features (SoftMax vs raw).
        3. Size matching policy between encoder/decoder maps.
    """

    _MERGE_METHODS = ("weighted_sum", "add", "channel_merge", "channel_multiply", "multiply")
    _SIZE_MATCH = (
        "upsample_both",       # both -> max(H, W)
        "downsample_both",     # both -> min(H, W)
        "fixed",               # fixed size H, W assigned
        "encoder_to_decoder",  # upsample encoder to decoder spatial size
        "decoder_to_encoder",  # downsample decoder to encoder spatial size
    )

    def __init__(
        self,
        enc_channels: int,
        dec_channels: int,
        num_classes: int,
        *,
        merge_method: str = "weighted_sum",
        pooling: str = "gap",
        fusion_dim: Optional[int] = 0,
        align: str = "pre",
        learnable_alpha: bool = True,
        alpha_init: float = 0.5,
        alpha_type: str = "scalar",
        size_match: str = "encoder_to_decoder",
        resize_backend: str = "interpolate",
        channel_multiply_ignore_background: bool = True,
        classifier_dropout: float = 0.0,
        classifier_bias: bool = False,
        fixed_size: Optional[Tuple[int, int]] = None,
        use_mask: bool = False,
        smp_classifier: str = "linear",
    ):
        super().__init__()
        merge_method = merge_method.lower()
        size_match = size_match.lower()
        if merge_method not in self._MERGE_METHODS:
            raise ValueError(f"merge_method must be one of {self._MERGE_METHODS}, got {merge_method}")
        if size_match not in self._SIZE_MATCH and not size_match.isdigit():
            raise ValueError(f"size_match must be in {self._SIZE_MATCH}, got {size_match}")
        resize_backend = resize_backend.lower()
        assert pooling in ("gap", "gmp")
        if merge_method == "weighted_sum" and not (0.0 < alpha_init < 1.0):
            raise ValueError(f"alpha_init should be in (0, 1), got {alpha_init}")
        if resize_backend not in ("interpolate", "conv"):
            raise ValueError("resize_backend must be 'interpolate' or 'conv'")
        if alpha_type not in ("scalar", "channel", "spatial", "se", "attn"):
            raise ValueError(f"alpha_type must be 'scalar', 'channel', 'spatial', 'se', or 'attn', got {alpha_type}")

        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        self.num_classes = num_classes
        self.merge_method = merge_method
        self.pooling = pooling
        self.size_match = size_match
        self.learnable_alpha = learnable_alpha and merge_method == "weighted_sum"
        self.alpha_type = alpha_type if (learnable_alpha and merge_method == "weighted_sum") else "scalar"
        self.resize_backend = resize_backend
        self.channel_multiply_ignore_background = channel_multiply_ignore_background
        self.channel_multiply_layers: Optional[int] = None
        self.fusion_dim = fusion_dim
        self.align = align
        self.fixed_size = fixed_size
        self.use_mask = use_mask
        self.smp_classifier = smp_classifier
        # Running cache for spatial alpha stats (updated each forward pass)
        self._last_spatial_alpha_mean: Optional[float] = None
        if resize_backend == "conv":
            self._upsample_layers = nn.ModuleDict()
            self._downsample_layers = nn.ModuleDict()
        else:
            self._upsample_layers = None
            self._downsample_layers = None
        #print(f"GeneralFusionHead: merge_method={merge_method}, size_match={size_match}, fusion_dim={fusion_dim}")
        #print("Encoder channels:", enc_channels, "Decoder channels:", dec_channels)
        self.enc_align = nn.Identity()
        self.dec_align = nn.Identity()
        if merge_method in ("weighted_sum", "add", "multiply"):
            target_dim = max(enc_channels, dec_channels) if not self.fusion_dim else self.fusion_dim
            self.enc_align = self._make_align_layer(enc_channels, target_dim)
            self.dec_align = self._make_align_layer(dec_channels, target_dim)
            final_dim = target_dim
        elif merge_method == "channel_merge":
            merged_dim = enc_channels + dec_channels
            final_dim = merged_dim if not self.fusion_dim else self.fusion_dim
            '''
            if final_dim != merged_dim:
                self.channel_reduce = nn.Conv2d(merged_dim, final_dim, kernel_size=1, bias=False)
            else:
                self.channel_reduce = nn.Identity()
            '''
            if final_dim != merged_dim:
                self.enc_align = self._make_align_layer(enc_channels, final_dim // 2)
                self.dec_align = self._make_align_layer(dec_channels, final_dim // 2)
        else:  # channel_multiply
            if dec_channels <= 0:
                raise ValueError("dec_channels must be > 0 for channel_multiply.")
            if self.channel_multiply_ignore_background and self.use_mask:
                if dec_channels == 9: #NOTE: hardcoded for 8-layer decoder output
                    effective_layers = dec_channels - 1
                elif dec_channels == 8: # Already no background channel
                    effective_layers = dec_channels
                else:
                    print("Warning: channel_multiply_ignore_background is True but dec_channels != 9.")
                    raise ValueError("Please ensure the decoder output has background channel to ignore.")
            else:
                effective_layers = dec_channels
            self.channel_multiply_layers = effective_layers
            multiply_dim = enc_channels * effective_layers
            final_dim = multiply_dim if not self.fusion_dim else self.fusion_dim
            if final_dim != multiply_dim:
                self.enc_align = self._make_align_layer(enc_channels, final_dim // effective_layers)
        #print("Final fused feature dim:", final_dim)

        if merge_method == "weighted_sum":
            init_logit = math.log(alpha_init) - math.log(1 - alpha_init)
            if self.learnable_alpha:
                if self.alpha_type == "scalar":
                    # Single scalar gate shared across all channels and spatial positions.
                    self.alpha_logit = nn.Parameter(
                        torch.tensor([init_logit], dtype=torch.float32)
                    )
                elif self.alpha_type == "channel":
                    # Independent gate per channel — learns which channels rely more on
                    # encoder vs decoder.  Shape: (target_dim,) → broadcast (1,C,1,1).
                    self.alpha_logit = nn.Parameter(
                        torch.full((target_dim,), init_logit, dtype=torch.float32)
                    )
                elif self.alpha_type == "spatial":
                    # Simple per-pixel gate: cat(enc,dec) → Conv → ReLU → Conv → Sigmoid.
                    # Produces independent (H,W) alpha values with no global context.
                    self.spatial_gate = SpatialGate(target_dim * 2)
                elif self.alpha_type == "se":
                    # SE + CBAM gate: global SE bias + channel-pool spatial attention.
                    # Richer than 'spatial'; captures both global and local context.
                    self.spatial_gate = SE_gate(target_dim * 2)
                elif self.alpha_type == "attn":
                    # Cross-attention gate: encoder global query × decoder spatial keys.
                    # Gate map = the cross-attention weights → directly interpretable as
                    # "which decoder positions the encoder classification focus attends to".
                    self.spatial_gate = AttnGate(target_dim * 2)
            else:
                self.register_buffer("alpha_fixed", torch.tensor(alpha_init, dtype=torch.float32))

        # Classifier (Linear)
        self.dropout = nn.Dropout2d(classifier_dropout) if classifier_dropout and classifier_dropout > 0 else nn.Identity()
        if self.smp_classifier == "conv":
            self.classifier = nn.Conv2d(final_dim, num_classes, kernel_size=1, bias=classifier_bias)
        elif self.smp_classifier == "linear":
            self.classifier = nn.Linear(final_dim, num_classes, bias=classifier_bias)
        else:
            raise ValueError(f"Unsupported smp_classifier type: {self.smp_classifier}")

    @staticmethod
    def _make_align_layer(in_ch: int, out_ch: int) -> nn.Module:
        return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == "gap":
            return x.mean(dim=(2, 3))
        elif self.pooling == "gmp":
            return x.amax(dim=(2, 3))
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")

    def _match_spatial(self, enc_feats: torch.Tensor, dec_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Match the spatial dimensions of the encoder and decoder features.

        Args:
            enc_feats (torch.Tensor): Encoder features. (B, C, H_1, W_1)
            dec_feats (torch.Tensor): Decoder features. (B, C, H_2, W_2)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Aligned encoder and decoder features. (B, C, H_f, W_f)
        """
        h_enc, w_enc = enc_feats.shape[-2:]
        h_dec, w_dec = dec_feats.shape[-2:]
        if self.size_match == "upsample_both":
            target = (max(h_enc, h_dec), max(w_enc, w_dec))
            enc_feats = self._resize(enc_feats, target, "upsample")
            dec_feats = self._resize(dec_feats, target, "upsample")
        elif self.size_match == "downsample_both":
            target = (min(h_enc, h_dec), min(w_enc, w_dec))
            enc_feats = self._resize(enc_feats, target, "downsample")
            dec_feats = self._resize(dec_feats, target, "downsample")
        elif self.size_match == "encoder_to_decoder":
            target = (h_dec, w_dec)
            enc_feats = self._resize(enc_feats, target, "upsample")
        elif self.size_match == "decoder_to_encoder":
            target = (h_enc, w_enc)
            dec_feats = self._resize(dec_feats, target, "downsample")
        elif self.size_match.isdigit():
            target = (int(self.size_match), int(self.size_match))
            enc_feats = self._resize(enc_feats, target, "resize")
            dec_feats = self._resize(dec_feats, target, "resize")
        return enc_feats, dec_feats

    def _resize(self, feat: torch.Tensor, target_spatial: Tuple[int, int], mode: str) -> torch.Tensor:
        if feat.shape[-2:] == target_spatial:
            return feat
        if mode == "resize":
            return F.interpolate(feat, size=target_spatial, mode="bilinear", align_corners=False)
        
        if self.resize_backend == "interpolate":
            if mode == "upsample":
                return F.interpolate(feat, size=target_spatial, mode="bilinear", align_corners=False)
            elif mode == "downsample":
                return F.adaptive_avg_pool2d(feat, output_size=target_spatial)
        elif self.resize_backend == "conv":  # conv backend, not true
            if mode == "upsample":
                return self._resize_with_deconv(feat, target_spatial)
            elif mode == "downsample":
                return self._resize_with_conv(feat, target_spatial)
        else:
            raise ValueError(f"Unsupported resize_backend: {self.resize_backend}")

    def _get_or_create_deconv(self, channels: int) -> nn.ConvTranspose2d:
        key = str(channels)
        if key not in self._upsample_layers:
            self._upsample_layers[key] = nn.ConvTranspose2d(
                channels, channels, kernel_size=2, stride=2, padding=0
            )
        return self._upsample_layers[key]

    def _get_or_create_downsample(self, channels: int) -> nn.Conv2d:
        key = str(channels)
        if key not in self._downsample_layers:
            self._downsample_layers[key] = nn.Conv2d(
                channels, channels, kernel_size=3, stride=2, padding=1, bias=False
            )
        return self._downsample_layers[key]

    def _resize_with_deconv(self, feat: torch.Tensor, target_spatial: Tuple[int, int]) -> torch.Tensor:
        target_h, target_w = target_spatial
        deconv = self._get_or_create_deconv(feat.shape[1])
        h, w = feat.shape[-2:]
        while (h < target_h) or (w < target_w):
            feat = deconv(feat)
            h, w = feat.shape[-2:]
            if h == target_h and w == target_w:
                return feat
            if h > target_h or w > target_w:
                return F.interpolate(feat, size=target_spatial, mode="bilinear", align_corners=False)
        return feat

    def _resize_with_conv(self, feat: torch.Tensor, target_spatial: Tuple[int, int]) -> torch.Tensor:
        target_h, target_w = target_spatial
        conv = self._get_or_create_downsample(feat.shape[1])
        h, w = feat.shape[-2:]
        while (h > target_h) or (w > target_w):
            feat = conv(feat)
            h, w = feat.shape[-2:]
            if h == target_h and w == target_w:
                return feat
            if h < target_h or w < target_w:
                return F.adaptive_avg_pool2d(feat, output_size=target_spatial)
        if (h, w) != target_spatial:
            feat = F.adaptive_avg_pool2d(feat, output_size=target_spatial)
        return feat

    def _apply_decoder_mask(self, feats: torch.Tensor, logits: Optional[torch.Tensor]) -> torch.Tensor:
        probs = logits # softmax already done in segmentation header
        if probs.shape[1] == 1:
            mask = probs
        else:
            # Treat channel 0 as background when available (typical 8-layer decoder output)
            mask = 1.0 - probs[:, 0:1]
        return feats * mask

    def _merge(self, enc_feats: torch.Tensor, dec_feats: torch.Tensor) -> torch.Tensor:
        if self.merge_method == "channel_merge":
            merged = torch.cat([enc_feats, dec_feats], dim=1)
            return merged

        if self.merge_method == "weighted_sum":
            if self.learnable_alpha:
                if self.alpha_type == "scalar":
                    # α ∈ ℝ, broadcast over (B, C, H, W)
                    alpha = torch.sigmoid(self.alpha_logit)
                elif self.alpha_type == "channel":
                    # α ∈ ℝ^C, broadcast over (B, C, H, W) via (1, C, 1, 1)
                    alpha = torch.sigmoid(self.alpha_logit).view(1, -1, 1, 1)
                elif self.alpha_type in ("spatial", "se", "attn"):
                    # α ∈ (0,1)^{H×W}: per-pixel gate predicted from cat(enc, dec).
                    # 'spatial': simple conv; 'se': SE+CBAM; 'attn': cross-attention.
                    concat = torch.cat([enc_feats, dec_feats], dim=1)  # (B, 2C, H, W)
                    alpha = self.spatial_gate(concat)                   # (B, 1, H, W)
                    # Cache mean for logging (detached; no graph retention)
                    self._last_spatial_alpha_mean = float(alpha.detach().mean())
                else:
                    alpha = torch.sigmoid(self.alpha_logit)
            else:
                alpha = self.alpha_fixed
            return alpha * enc_feats + (1 - alpha) * dec_feats

        if self.merge_method == "add":
            return enc_feats + dec_feats
        if self.merge_method == "multiply":
            return enc_feats * dec_feats
        raise RuntimeError(f"Unsupported merge_method {self.merge_method}")

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def get_alpha_stats(self) -> Optional[Dict[str, float]]:
        """Return current fusion-gate statistics (no grad).

        Returns a dict suitable for logging to wandb / CSV, or None when the
        merge method is not weighted_sum (gate concept does not apply).

        Keys depend on alpha_type:
          scalar  → {'alpha': float}
          channel → {'alpha_mean', 'alpha_std', 'alpha_min', 'alpha_max'}
          spatial → {'alpha_mean'}   (mean over last forward-pass spatial map)
          fixed   → {'alpha': float}
        """
        if self.merge_method != "weighted_sum":
            return None

        if not self.learnable_alpha:
            return {"alpha": float(self.alpha_fixed)}

        with torch.no_grad():
            if self.alpha_type == "scalar":
                return {"alpha": float(torch.sigmoid(self.alpha_logit))}
            elif self.alpha_type == "channel":
                v = torch.sigmoid(self.alpha_logit)
                return {
                    "alpha_mean": float(v.mean()),
                    "alpha_std":  float(v.std()),
                    "alpha_min":  float(v.min()),
                    "alpha_max":  float(v.max()),
                }
            elif self.alpha_type in ("spatial", "se", "attn"):
                if self._last_spatial_alpha_mean is None:
                    return None
                return {"alpha_mean": self._last_spatial_alpha_mean}
        return None

    def _channel_multiply(self, enc_feats: torch.Tensor, dec_feats: torch.Tensor) -> torch.Tensor:
        if dec_feats is None:
            raise ValueError("dec_feats must be provided for channel_multiply mode.")
        masks = dec_feats # softmax already done in segmentation header
        if self.use_mask and self.channel_multiply_ignore_background:
            if masks.shape[1] <= 1:
                raise ValueError("Decoder output must have background channel to ignore.")
            masks = masks[:, 1:, :, :]
        if masks.shape[1] != self.channel_multiply_layers:
            raise ValueError(
                f"Expected {self.channel_multiply_layers} decoder channels after processing, got {masks.shape[1]}"
            )
        # [B, C_enc, H, W] -> unsqueeze(1) -> [B, 1, C_enc, H, W]
        # [B, M, H, W] -> unsqueeze(2) -> [B, M, 1, H, W]
        # [B, 1, C_enc, H, W] * [B, M, 1, H, W] -> [B, M, C_enc, H, W]
        # -> reshape(B, M*C_enc, H, W) -> [B, M*C_enc, H, W]
        fused = (enc_feats.unsqueeze(1) * masks.unsqueeze(2)).reshape(
            enc_feats.shape[0],
            self.channel_multiply_layers * enc_feats.shape[1],
            enc_feats.shape[2],
            enc_feats.shape[3],
        )
        return fused

    def forward(
        self,
        enc_feats: torch.Tensor,
        dec_feats: torch.Tensor,
        *,
        return_fused_feature: bool = False,
    ):
        """
        Args:
            enc_feats: encoder map [B, C_enc, H1, W1]
            dec_feats: decoder map [B, C_dec, H2, W2]
            decoder_logits: optional decoder segmentation logits (e.g. 8 layers)
        """
        if self.align=='pre':
            enc_feats = self.enc_align(enc_feats)
            dec_feats = self.dec_align(dec_feats)
            enc_feats, dec_feats = self._match_spatial(enc_feats, dec_feats)
        else:
            enc_feats, dec_feats = self._match_spatial(enc_feats, dec_feats)
            enc_feats = self.enc_align(enc_feats)
            dec_feats = self.dec_align(dec_feats)
        
        if self.merge_method == "channel_multiply":
            fused = self._channel_multiply(enc_feats, dec_feats)
        else:
            fused = self._merge(enc_feats, dec_feats)
        #print("Fused feature shape:", fused.shape)
        if self.smp_classifier == "linear":
            fused = self.dropout(fused)
            pooled = self._pool(fused)
            #print("Pooled feature shape:", pooled.shape)
            logits = self.classifier(pooled)
        else:  # conv classifier
            logits_map = self.classifier(self.dropout(fused))
            #print("Conv feature shape:", logits_map.shape)
            logits = self._pool(logits_map)
        #print("Logits shape:", logits.shape)
        if return_fused_feature:
            return logits, fused
        return logits
