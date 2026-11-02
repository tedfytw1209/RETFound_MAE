"""RETFound ViT-Large (MAE) classifier.

Vendored from RETFound_MAE/models_vit.py (VisionTransformer, RETFound_mae). Attribute names
(patch_embed, blocks, fc_norm, head, ...) must stay identical to the original so the fine-tuned
checkpoints load by state_dict key.
"""

import os
from functools import partial
from typing import Optional

# The XAI evaluation runs with the plain attention path (TIMM_FUSED_ATTN=0); match it unless the
# caller already chose. Must be set before timm is imported.
os.environ.setdefault("TIMM_FUSED_ATTN", "0")

import timm.models.vision_transformer  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling."""

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x, attn_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1, keepdim=True)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    # The checkpoints were trained with timm 0.9.x, where forward_head() takes token 0 of the sequence
    # and applies fc_norm once more (fc_norm is therefore applied twice when global_pool=True: here and
    # in forward_features). timm >= 1.0 pools differently, so the 0.9.x behaviour is pinned explicitly
    # instead of depending on the installed timm version.
    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None, **kwargs):
        return self.forward_head(self.forward_features(x))


def RETFound_mae(**kwargs):
    return VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
