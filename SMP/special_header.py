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
        fusion_dim: Optional[int] = None,
        learnable_alpha: bool = True,
        alpha_init: float = 0.5,
        decoder_softmax: bool = True,
        size_match: str = "encoder_to_decoder",
        resize_backend: str = "interpolate",
        channel_multiply_ignore_background: bool = True,
        classifier_dropout: float = 0.0,
        classifier_bias: bool = False,
    ):
        super().__init__()
        merge_method = merge_method.lower()
        size_match = size_match.lower()
        if merge_method not in self._MERGE_METHODS:
            raise ValueError(f"merge_method must be one of {self._MERGE_METHODS}, got {merge_method}")
        if size_match not in self._SIZE_MATCH:
            raise ValueError(f"size_match must be in {self._SIZE_MATCH}, got {size_match}")
        resize_backend = resize_backend.lower()
        assert pooling in ("gap", "gmp")
        if merge_method == "weighted_sum" and not (0.0 < alpha_init < 1.0):
            raise ValueError(f"alpha_init should be in (0, 1), got {alpha_init}")
        if resize_backend not in ("interpolate", "conv"):
            raise ValueError("resize_backend must be 'interpolate' or 'conv'")

        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        self.num_classes = num_classes
        self.merge_method = merge_method
        self.pooling = pooling
        self.decoder_softmax = decoder_softmax
        self.size_match = size_match
        self.learnable_alpha = learnable_alpha and merge_method == "weighted_sum"
        self.resize_backend = resize_backend
        self.channel_multiply_ignore_background = channel_multiply_ignore_background
        self.channel_multiply_layers: Optional[int] = None
        if resize_backend == "conv":
            self._upsample_layers = nn.ModuleDict()
            self._downsample_layers = nn.ModuleDict()
        else:
            self._upsample_layers = None
            self._downsample_layers = None

        if merge_method in ("weighted_sum", "add", "multiply"):
            target_dim = fusion_dim if fusion_dim is not None else max(enc_channels, dec_channels)
            self.enc_align = self._make_align_layer(enc_channels, target_dim)
            self.dec_align = self._make_align_layer(dec_channels, target_dim)
            final_dim = target_dim
        elif merge_method == "channel_merge":
            merged_dim = enc_channels + dec_channels
            final_dim = fusion_dim if fusion_dim is not None else merged_dim
            if final_dim != merged_dim:
                self.channel_reduce = nn.Conv2d(merged_dim, final_dim, kernel_size=1, bias=False)
            else:
                self.channel_reduce = nn.Identity()
            self.enc_align = nn.Identity()
            self.dec_align = nn.Identity()
        else:  # channel_multiply
            if dec_channels <= 0:
                raise ValueError("dec_channels must be > 0 for channel_multiply.")
            if self.channel_multiply_ignore_background:
                if dec_channels <= 1:
                    raise ValueError("channel_multiply with ignore_background=True requires dec_channels >= 2.")
                effective_layers = dec_channels - 1
            else:
                effective_layers = dec_channels
            self.channel_multiply_layers = effective_layers
            final_dim = enc_channels * effective_layers
            self.enc_align = nn.Identity()
            self.dec_align = nn.Identity()

        if merge_method == "weighted_sum":
            init_logit = math.log(alpha_init) - math.log(1 - alpha_init)
            if self.learnable_alpha:
                self.alpha_logit = nn.Parameter(torch.tensor([init_logit], dtype=torch.float32))
            else:
                self.register_buffer("alpha_fixed", torch.tensor(alpha_init, dtype=torch.float32))

        # Classifier (Linear)
        self.dropout = nn.Dropout2d(classifier_dropout) if classifier_dropout and classifier_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(final_dim, num_classes, bias=classifier_bias)

    @staticmethod
    def _make_align_layer(in_ch: int, out_ch: int) -> nn.Module:
        return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == "gap":
            return x.mean(dim=(2, 3))
        return x.amax(dim=(2, 3))

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
        else:  # decoder_to_encoder
            target = (h_enc, w_enc)
            dec_feats = self._resize(dec_feats, target, "downsample")
        return enc_feats, dec_feats

    def _resize(self, feat: torch.Tensor, target_spatial: Tuple[int, int], mode: str) -> torch.Tensor:
        if feat.shape[-2:] == target_spatial:
            return feat
        if self.resize_backend == "interpolate":
            if mode == "upsample":
                return F.interpolate(feat, size=target_spatial, mode="bilinear", align_corners=False)
            return F.adaptive_avg_pool2d(feat, output_size=target_spatial)
        if mode == "upsample":
            return self._resize_with_deconv(feat, target_spatial)
        return self._resize_with_conv(feat, target_spatial)

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

    def _apply_decoder_mask(self, dec_feats: torch.Tensor, decoder_logits: Optional[torch.Tensor]) -> torch.Tensor:
        if decoder_logits is None:
            return dec_feats
        logits = decoder_logits
        if logits.shape[2:] != dec_feats.shape[2:]:
            logits = F.interpolate(logits, size=dec_feats.shape[2:], mode="bilinear", align_corners=False)
        if self.decoder_softmax:
            probs = F.softmax(logits, dim=1)
        else:
            probs = logits
        if probs.shape[1] == 1:
            mask = probs
        else:
            # Treat channel 0 as background when available (typical 8-layer decoder output)
            mask = 1.0 - probs[:, 0:1]
        return dec_feats * mask

    def _merge(self, enc_feats: torch.Tensor, dec_feats: torch.Tensor) -> torch.Tensor:
        if self.merge_method == "channel_merge":
            merged = torch.cat([enc_feats, dec_feats], dim=1)
            return self.channel_reduce(merged)

        enc_aligned = self.enc_align(enc_feats)
        dec_aligned = self.dec_align(dec_feats)

        if self.merge_method == "weighted_sum":
            if self.learnable_alpha:
                alpha = torch.sigmoid(self.alpha_logit)
            else:
                alpha = self.alpha_fixed
            return alpha * enc_aligned + (1 - alpha) * dec_aligned
        if self.merge_method == "add":
            return enc_aligned + dec_aligned
        if self.merge_method == "multiply":
            return enc_aligned * dec_aligned
        raise RuntimeError(f"Unsupported merge_method {self.merge_method}")

    def _channel_multiply(self, enc_feats: torch.Tensor, dec_feats: torch.Tensor) -> torch.Tensor:
        if dec_feats is None:
            raise ValueError("dec_feats must be provided for channel_multiply mode.")
        if self.decoder_softmax:
            masks = F.softmax(dec_feats, dim=1)
        else:
            masks = dec_feats
        if self.channel_multiply_ignore_background:
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
        decoder_logits: Optional[torch.Tensor] = None,
        return_fused_feature: bool = False,
    ):
        """
        Args:
            enc_feats: encoder map [B, C_enc, H1, W1]
            dec_feats: decoder map [B, C_dec, H2, W2]
            decoder_logits: optional decoder segmentation logits (e.g. 8 layers)
        """
        enc_feats, dec_feats = self._match_spatial(enc_feats, dec_feats)
        if self.merge_method == "channel_multiply":
            fused = self._channel_multiply(enc_feats, dec_feats)
        else:
            dec_feats = self._apply_decoder_mask(dec_feats, decoder_logits)
            fused = self._merge(enc_feats, dec_feats)
        pooled = self._pool(fused)
        logits = self.classifier(pooled)
        if return_fused_feature:
            return logits, fused
        return logits
