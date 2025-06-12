
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from pprint import pprint

import timm
from timm.models.layers import PatchEmbed
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

def generate_attention_map_single(attentions, img_size=224, use_rollout=True):
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

    attention_map = np.array(Image.fromarray(attention_map).resize((img_size, img_size), resample=Image.BILINEAR))
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    return attention_map

class Attention_Map(torch.nn.Module):
    def __init__(self, model, input_size, N=11, use_rollout=True):
        super(Attention_Map, self).__init__()
        self.model = model
        self.input_size = input_size
        self.use_rollout = use_rollout
        
        self.print_model(model)
        return_attns = [f'blocks.{i}.attn.softmax' for i in range(N)]
        # This is the "one line of code" that does what you want
        self.feature_extractor = create_feature_extractor(
            model, return_nodes=return_attns,
            tracer_kwargs={'leaf_modules': [PatchEmbed]})

    def forward(self, x):
        bs = x.shape[0]
        self.model.eval()
        with torch.no_grad():
            attentions = self.feature_extractor(x) #(B, ...)

        attention_maps = []
        for i in range(bs):
            attention_map = generate_attention_map_single(attentions, img_size=self.input_size, use_rollout=self.use_rollout)
            attention_maps.append(attention_map)
        attention_maps = np.array(attention_maps)
        attention_maps = torch.from_numpy(attention_maps).float().cuda()
        return attention_maps #.unsqueeze(1)  # Add channel dimension
    
    def print_model(self,model):
        nodes, _ = get_graph_node_names(model, tracer_kwargs={'leaf_modules': [PatchEmbed]})
        pprint(nodes)
    

if __name__ == "__main__":
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    input_size = 224
    attention_map_model = Attention_Map(model, input_size, N=11, use_rollout=True)

    # Example input
    x = torch.randn(2, 3, input_size, input_size).cuda()  # Batch size of 2
    attention_maps = attention_map_model(x)

    print("Attention Maps Shape:", attention_maps.shape)  # Should be (2, input_size, input_size)