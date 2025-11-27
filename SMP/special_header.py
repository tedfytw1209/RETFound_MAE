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
