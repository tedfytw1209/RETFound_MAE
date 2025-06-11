
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

def generate_attention_map_single(attentions, use_rollout=True):
    if use_rollout:
        avg_attentions = [att.mean(dim=1).squeeze(0).cpu().numpy() for att in attentions]
        num_tokens = avg_attentions[0].shape[0]
        rollout = np.eye(num_tokens)

        for att in avg_attentions:
            att = att + np.eye(num_tokens)
            att = att / att.sum(axis=-1, keepdims=True)
            rollout = np.matmul(rollout, att)

        cls_attention = rollout[0, 1:]
    else:
        last_layer_attention = attentions[-1][0]  # shape: (1, num_heads, num_tokens, num_tokens)
        cls_attention = last_layer_attention[:, 0, 1:].mean(dim=0).cpu().numpy()

    num_patches = int(cls_attention.shape[0] ** 0.5)
    attention_map = cls_attention.reshape(num_patches, num_patches)

    attention_map = np.array(Image.fromarray(attention_map).resize((image_tensor.shape[-1], image_tensor.shape[-2]), resample=Image.BILINEAR))
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    return attention_map