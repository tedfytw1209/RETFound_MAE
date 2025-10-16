import torch
import itertools
from PIL import Image
from crp.concepts import ChannelConcept

from zennit.image import imgify
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules
from zennit.composites import EpsilonPlusFlat
from zennit.canonizers import SequentialMergeBatchNorm
from crp.attribution import CondAttribution

from lxt.efficient import monkey_patch, monkey_patch_zennit
from crp.helper import get_layer_names

""" Model wrapper to return a tensor"""
class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        try:
            out = self.model(pixel_values=x)
        except TypeError:
            out = self.model(x)

        # 3) 统一抽取 logits
        if isinstance(out, torch.Tensor):
            logits = out
        elif hasattr(out, "logits"):
            logits = out.logits
        elif isinstance(out, dict) and "logits" in out:
            logits = out["logits"]
        elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            logits = out[0]
        else:
            raise TypeError(
                f"Cannot extract logits from output of type {type(out)}. "
                f"Expected Tensor / object with .logits / dict['logits'] / tuple[Tensor,...]."
            )

        return logits

def _get(obj, name, default=None):
    return getattr(obj, name, default)
## CRP
class CRP(torch.nn.Module):
    def __init__(self, model, model_name, img_size, patch_size):
        super(CRP, self).__init__()
        self.model = model
        self.model_name = model_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.cc = ChannelConcept()
        self.composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
        if _get(self.model, 'resnet', None) is not None:
            hook = _get(self.model, 'resnet', None)
        elif _get(self.model, 'vit', None) is not None:
            hook = _get(self.model, 'vit', None)
        else:
            hook = self.model
        self.attribution = CondAttribution(HuggingfaceToTensorModelWrapper(self.model), no_param_grad=True)

    def generate_heatmap(self, x, target_class=None):
        # compute heatmap wrt. output 46 (green lizard class)
        conditions = [{"y": target_class}]
        # zennit requires gradients
        x.requires_grad = True
        attr = self.attribution(x, conditions, self.composite, mask_map=self.cc.mask)
        # or use a dictionary for mask_map
        layer_names = get_layer_names(self.model, [torch.nn.Conv2d, torch.nn.Linear])
        mask_map = {name: self.cc.mask for name in layer_names}
        attr = self.attribution(x, conditions, self.composite, mask_map=mask_map)
        heatmap = attr.heatmap
        B = heatmap.shape[0]
        # Normalize heatmap between [0, 1] for plotting
        cam_min = heatmap.view(B, -1).min(dim=1)[0].view(B, 1, 1)
        cam_max = heatmap.view(B, -1).max(dim=1)[0].view(B, 1, 1)
        cam = (heatmap - cam_min) / (cam_max - cam_min + 1e-8)
        return cam # (B, H, W)
    def forward(self, image_tensor, target_class=None):
        """Generate CRP heatmap"""
        if target_class is None:
            # Get predicted class
            with torch.no_grad():
                outputs = HuggingfaceToTensorModelWrapper(self.model)(image_tensor)
                target_class = outputs.argmax(dim=1).item()
        heatmaps = self.generate_heatmap(image_tensor, target_class)
        return heatmaps.detach().cpu().numpy()

## LXT

# Modify the Vision Transformer module to compute Layer-wise Relevance Propagation (LRP)
# in the backward pass. For ViTs, we utilize the LRP Gamma rule. It is implemented
# inside the 'zennit' library. To make it compatible with LXT, we also monkey patch it. That's it.
class LXT(torch.nn.Module):
    def __init__(self, model, model_name, img_size, patch_size, conv_gamma = 0.25, lin_gamma = 0.05):
        super(LXT, self).__init__()
        self.model = model
        self.model_name = model_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.conv_gamma = conv_gamma #[0.1, 0.25, 100]
        self.lin_gamma = lin_gamma #[0, 0.01, 0.05, 0.1, 1]
        if _get(self.model, 'resnet', None) is not None:
            hook = _get(self.model, 'resnet', None)
        elif _get(self.model, 'vit', None) is not None:
            hook = _get(self.model, 'vit', None)
        else:
            hook = self.model
        # Apply monkey patch to the model to enable LRP with Gamma rule
        monkey_patch(type(hook), verbose=True)
        #monkey_patch_zennit(verbose=True)

    def generate_heatmap(self, x, target_class=None):
        x.grad = None  # Reset gradients
        # Define rules for the Conv2d and Linear layers using 'zennit'
        # LayerMapComposite maps specific layer types to specific LRP rule implementations
        zennit_comp = LayerMapComposite([
            (torch.nn.Conv2d, z_rules.Gamma(self.conv_gamma)),
            (torch.nn.Linear, z_rules.Gamma(self.lin_gamma)),
        ])
        # Register the composite rules with the model
        zennit_comp.register(self.model)
        # Forward pass with gradient tracking enabled
        y = HuggingfaceToTensorModelWrapper(self.model)(x.requires_grad_())
        # Backward pass for the highest probability class
        # This initiates the LRP computation through the network
        y[0, target_class].backward()
        # Remove the registered composite to prevent interference in future iterations
        zennit_comp.remove()
        # Calculate the relevance by computing Gradient * Input
        # This is the final step of LRP to get the pixel-wise explanation
        heatmap = (x * x.grad).sum(1)
        # Normalize relevance between [-1, 1] for plotting
        #heatmap = heatmap / abs(heatmap).max()
        B = heatmap.shape[0]
        cam_min = heatmap.view(B, -1).min(dim=1)[0].view(B, 1, 1)
        cam_max = heatmap.view(B, -1).max(dim=1)[0].view(B, 1, 1)
        cam = (heatmap - cam_min) / (cam_max - cam_min + 1e-8)
        return cam # (B, H, W)
    
    def forward(self, image_tensor, target_class=None):
        """Generate LXT heatmap"""
        if target_class is None:
            # Get predicted class
            with torch.no_grad():
                outputs = HuggingfaceToTensorModelWrapper(self.model)(image_tensor)
                target_class = outputs.argmax(dim=1).item()
        heatmaps = self.generate_heatmap(image_tensor, target_class)
        return heatmaps.detach().cpu().numpy()

class CRP_LXT(torch.nn.Module):
    def __init__(self, model, model_name, img_size, patch_size=14):
        super(CRP_LXT, self).__init__()
        self.model = model
        self.model_name = model_name
        self.model.eval()
        self.img_size = img_size
        self.patch_size = patch_size

        if 'retfound' in model_name.lower() or 'vit' in model_name.lower() or 'dino' in model_name.lower():
            # timm or HuggingFace ViT
            self.mode = 'LXT'
        else:
            self.mode = 'CRP'
        #define LXT or CRP
        if self.mode == 'LXT':
            self.method = LXT(model, model_name, img_size, patch_size)
        elif self.mode == 'CRP':
            self.method = CRP(model, model_name, img_size, patch_size)
        else:
            raise ValueError(f"Model {model_name} is not supported for CRP_LXT.")
    ##model=model, inputs=x_batch, targets=y_batch, **self.explain_func_kwargs
    def forward(self, inputs=None, targets=None, model=None, **kwargs):
        return self.method(inputs=inputs, targets=targets, model=model, **kwargs)
        

