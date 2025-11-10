
from PIL import Image
import numpy as np
import torch
from pprint import pprint

import timm
from timm.models.layers import PatchEmbed
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from util.misc import to_tensor

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

def generate_attention_map_batch(attentions, img_size=224, use_rollout=True):
    """
    Generate attention maps for a batch of images using transformer attentions.

    Parameters:
        attentions (list of torch.Tensor): List of attention tensors from each layer,
                                           each of shape (B, num_heads, num_tokens, num_tokens)
        img_size (int): Target spatial size (e.g., 224).
        use_rollout (bool): Whether to apply attention rollout across layers.

    Returns:
        np.ndarray: Attention maps of shape (B, img_size, img_size)
    """
    B = attentions[0].shape[0]
    num_tokens = attentions[0].shape[-1]
    patch_tokens = num_tokens - 1  # exclude cls token
    num_patches = int(patch_tokens ** 0.5)
    attention_maps = []

    for i in range(B):
        if use_rollout:
            rollout = np.eye(num_tokens)
            for att in attentions:
                avg_att = att[i].mean(dim=0).cpu().numpy()  # shape: (num_tokens, num_tokens)
                avg_att = avg_att + np.eye(num_tokens)
                avg_att = avg_att / avg_att.sum(axis=-1, keepdims=True)
                rollout = np.matmul(rollout, avg_att)
            cls_attention = rollout[0, 1:]  # exclude cls token
        else:
            last_layer = attentions[-1][i]  # shape: (num_heads, num_tokens, num_tokens)
            cls_attention = last_layer[:, 0, 1:].mean(dim=0).cpu().numpy()  # mean over heads

        att_map = cls_attention.reshape(num_patches, num_patches)
        att_map = att_map.astype(np.float32)
        att_map = np.array(Image.fromarray(att_map).resize((img_size, img_size), resample=Image.BILINEAR))
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
        attention_maps.append(att_map)

    return np.stack(attention_maps)  # shape: (B, img_size, img_size)

class Attention_Map(torch.nn.Module):
    def __init__(self, model, model_name, input_size, N=12, use_rollout=True, print_layers=False, device=None):
        super(Attention_Map, self).__init__()
        self.model = model
        self.input_size = input_size
        self.use_rollout = use_rollout
        self.model_name = model_name
        self.device = device
        
        if print_layers:
            self.print_model(model, use_timm=False)

        if 'retfound' in model_name.lower(): #timm RETFound model
            # for timm ViT model (e.g., vit_base_patch16_224, RETFound_mae, etc.)
            self.return_attns = [f'blocks.{i}.attn.softmax' for i in range(N)]
            # This is the "one line of code" that does what you want
            self.feature_extractor = create_feature_extractor(
                model, return_nodes=self.return_attns,
                tracer_kwargs={'leaf_modules': [PatchEmbed]})
            self.use_timm = True
        elif 'vit' in model_name or 'dino' in model_name: #huggingface ViT model
            # for HuggingFace ViT model (e.g., vit_base_patch16_224, dino_vitb16, etc.)
            self.return_attns = []
            self.feature_extractor = None
            self.use_timm = False
        else:
            raise ValueError(f"Model {model_name} is not supported for attention map extraction.")

    #model=model, inputs=x_batch, targets=y_batch, **self.explain_func_kwargs
    def forward(self, inputs=None, targets=None, model=None, **kwargs):
        if model is None:
            model = self.model
        if inputs is None:
            raise ValueError("inputs parameter is required")
        inputs = to_tensor(inputs, device=self.device)
        model.eval()
        with torch.no_grad():
            if self.use_timm:
                attentions = self.feature_extractor(inputs) #(B, n_heads, num_tokens, num_tokens)
                attentions = [attentions[key] for key in self.return_attns]
            else:
                # Make sure x is shaped [B,3,224,224] and already normalized by the processor
                # Some HF models will propagate flags through the classifier
                outputs = model(pixel_values=inputs, output_attentions=True, return_dict=True)
                attentions = getattr(outputs, "attentions", None)

                if attentions is None:
                    # Call the base ViT encoder directly (works reliably)
                    outputs = model.vit(pixel_values=inputs, output_attentions=True, return_dict=True)
                    attentions = outputs.attentions

                if attentions is None:
                    print("Model attentions:", outputs)
                    raise RuntimeError("ViT did not return attentions; check that x is passed as pixel_values and flags are set.")

        attention_maps = generate_attention_map_batch(attentions, img_size=self.input_size, use_rollout=self.use_rollout)
        #attention_maps = torch.from_numpy(attention_maps).float().cuda()
        return attention_maps #.unsqueeze(1)  # Add channel dimension
    
    def print_model(self,model,use_timm): #TODO: Need fix for None timm models
        if use_timm:
            print("Timm Model Layers:")
            nodes, _ = get_graph_node_names(model, tracer_kwargs={'leaf_modules': [PatchEmbed]})
            pprint(nodes)
        else:
            print("HuggingFace Model Layers:")
            print(model)
    

if __name__ == "__main__":
    #Need TIMM_FUSED_ATTN=0
    model = timm.create_model('vit_base_patch16_224', pretrained=True).cuda()
    input_size = 224
    attention_map_model = Attention_Map(model, input_size, N=11, use_rollout=True).cuda()

    # Example input
    x = torch.randn(2, 3, input_size, input_size).cuda()  # Batch size of 2
    attention_maps = attention_map_model(x)

    print("Attention Maps Shape:", attention_maps.shape)  # Should be (2, input_size, input_size)