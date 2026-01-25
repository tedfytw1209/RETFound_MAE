import warnings
warnings.filterwarnings('ignore')
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import numpy as np
from typing import List, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification, ViTImageProcessor
from util.misc import to_tensor

def _get(obj, name, default=None):
    return getattr(obj, name, default)
def _resolve_target_layer(model, model_name=None, module_name=None, select_index=-1, debug=False):
    """Resolve target layer for CAM methods with optional debug logging"""

    if debug:
        print(f"\n{'='*60}")
        print(f"[DEBUG] _resolve_target_layer called:")
        print(f"  model_name: {model_name}")
        print(f"  module_name: {module_name}")
        print(f"  select_index: {select_index}")
        print(f"{'='*60}")

    # --- RETFound
    if 'RETFound' in model_name:
        target = model.blocks[select_index]
        if debug:
            print(f"[DEBUG] RETFound path: Resolved to model.blocks[{select_index}]")
            print(f"  Target layer type: {type(target).__name__}")
        return target
    
    # --- SMP Classifier (Segmentation Models PyTorch)
    # Check for SMP model structure: has encoder and seg_model
    seg_model = _get(model, "seg_model")
    mode = _get(model, "mode")
    
    if seg_model is not None:
        # This is an SMP-based model
        encoder = _get(seg_model, "encoder")
        decoder = _get(seg_model, "decoder")
        head = _get(model, "head")
        #Manually select the target module
        if module_name is not None:
            if module_name == "encoder" and encoder is not None:
                target_module = encoder
            elif module_name == "decoder" and decoder is not None:
                target_module = decoder
            elif module_name == "head" and head is not None:
                target_module = head
            else:
                raise ValueError(f"Unsupported SMP module_name: {module_name}")
        # Automatically select the target module
        elif mode == "enc" and encoder is not None:
            # For encoder or fuse mode, target the last encoder layer
            # SMP encoders typically have stages/layers
            target_module = encoder
        elif mode == "dec" and decoder is not None:
            # For decoder mode, target the decoder output
            # Get the last conv layer in the decoder
            target_module = decoder
        # !!! Need to check SMP fuse mode !!!
        elif mode == "fuse": #get general classifier layer, tmporary solution
            if debug:
                print(f"[DEBUG] SMP fuse mode detected:")
                print(f"  size_match: {getattr(model, 'size_match', 'N/A')}")

            if encoder is not None and model.size_match=="decoder_to_encoder":
                target_module = encoder
                if debug:
                    print(f"  Routing to encoder (decoder_to_encoder)")
            elif decoder is not None and model.size_match=="encoder_to_decoder":
                target_module = decoder
                if debug:
                    print(f"  Routing to decoder (encoder_to_decoder)")
            elif head is not None:
                target_module = head
                if debug:
                    print(f"  Routing to head (fallback)")
            else:
                raise ValueError(f"Unsupported SMP size_match: {model.size_match}")
        else:
            raise ValueError(f"Unsupported SMP mode: {mode}")
        
        if target_module is not None:
            conv_list = []
            for name, module in target_module.named_modules():
                if isinstance(module, nn.Conv2d):
                    conv_list.append((name, module))

            if debug:
                print(f"[DEBUG] SMP path - collecting Conv2d layers:")
                print(f"  Mode: {mode}")
                print(f"  Module name: {module_name}")
                print(f"  Target module type: {type(target_module).__name__}")
                print(f"  Found {len(conv_list)} Conv2d layers")
                if len(conv_list) > 0:
                    print(f"  First 3 Conv2d layers:")
                    for i, (name, layer) in enumerate(conv_list[:3]):
                        print(f"    [{i}] {name}: {layer.in_channels} -> {layer.out_channels}, kernel={layer.kernel_size}, stride={layer.stride}")
                    if len(conv_list) > 3:
                        print(f"  Last 3 Conv2d layers:")
                        for i, (name, layer) in enumerate(conv_list[-3:], start=len(conv_list)-3):
                            print(f"    [{i}] {name}: {layer.in_channels} -> {layer.out_channels}, kernel={layer.kernel_size}, stride={layer.stride}")
                    print(f"  Selecting index: {select_index} -> layer: {conv_list[select_index][0]}")

            if len(conv_list) > 0:
                selected_layer = conv_list[select_index][1]  # Extract module from tuple
                if debug:
                    print(f"[DEBUG] Resolved to Conv2d layer: {conv_list[select_index][0]}")
                    print(f"  In channels: {selected_layer.in_channels}")
                    print(f"  Out channels: {selected_layer.out_channels}")
                    print(f"  Kernel size: {selected_layer.kernel_size}")
                    print(f"  Stride: {selected_layer.stride}")
                return selected_layer
            else:
                raise ValueError(f"Cannot resolve target layer for SMP model. {target_module} {select_index}")
        else:
            raise ValueError(f"Cannot resolve mode {mode} {module_name} target layer {target_module} {select_index} for SMP model.")
    
    # --- HuggingFace ViT 風格
    vit = _get(model, "vit")
    if vit is not None:
        enc = _get(vit, "encoder")
        layers = _get(enc, "layer")
        if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
            layer = layers[select_index]
            if debug:
                print(f"[DEBUG] HuggingFace ViT path: Resolved to vit.encoder.layer[{select_index}]")
                print(f"  Target layer type: {type(layer).__name__}")
            if hasattr(layer, "layernorm_before"):
                return layer.layernorm_before
            if hasattr(layer, "layernorm"):
                return layer.layernorm
            return layer
    
    # --- timm ViT 風格
    if _get(model, "blocks") is not None:
        blocks = model.blocks
        if isinstance(blocks, (nn.ModuleList, list)) and len(blocks) > 0:
            blk = blocks[select_index]
            if debug:
                print(f"[DEBUG] timm ViT path: Resolved to blocks[{select_index}]")
                print(f"  Target layer type: {type(blk).__name__}")
            if hasattr(blk, "norm2"):
                return blk.norm2
            if hasattr(blk, "norm1"):
                return blk.norm1
            return blk
    
    # --- HF 某些包裝在 base_model
    base = _get(model, "base_model")
    if base is not None:
        vit = _get(base, "vit")
        if vit is not None:
            enc = _get(vit, "encoder")
            layers = _get(enc, "layer")
            if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
                layer = layers[select_index]
                if debug:
                    print(f"[DEBUG] HF base_model: Resolved to base_model.vit.encoder.layer[{select_index}]")
                    print(f"  Target layer type: {type(layer).__name__}")
                return layer

    # --- timm Swin
    if _get(model, "layers") is not None:
        layers = model.layers
        if len(layers) > 0 and _get(layers[-1], "blocks") is not None:
            blks = layers[-1].blocks
            if len(blks) > 0:
                return blks[select_index]

    # --- HF Swin
    swin = _get(model, "swin")
    if swin is not None:
        enc = _get(swin, "encoder")
        layers = _get(enc, "layers")
        if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
            blks = _get(layers[-1], "blocks")
            if isinstance(blks, (nn.ModuleList, list)) and len(blks) > 0:
                return blks[select_index]

    # --- torchvision ResNet
    if _get(model, "layer4") is not None and len(model.layer4) > 0:
        target = model.layer4[select_index]
        if debug:
            print(f"[DEBUG] torchvision ResNet path: model.layer4[{select_index}]")
            print(f"  Target layer type: {type(target).__name__}")
        return target

    # --- HF ResNet (wrapped under .resnet)
    resnet = _get(model, "resnet")
    if resnet is not None:
        conv_list = []
        for name, module in resnet.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_list.append((name, module))
        if len(conv_list) > 0:
            if debug:
                print(f"[DEBUG] HF ResNet path:")
                print(f"  Found {len(conv_list)} Conv2d layers in .resnet")
                print(f"  Selected: {conv_list[select_index][0]}")
                selected = conv_list[select_index][1]
                print(f"  {selected.in_channels} -> {selected.out_channels}")
            return conv_list[select_index][1]

    # --- HF EfficientNet often wrapped under .efficientnet
    eff = _get(model, "efficientnet")
    if eff is not None:
        last_name, last_conv = None, None
        for name, m in eff.named_modules():
            if isinstance(m, nn.Conv2d):
                last_name, last_conv = name, m
        return last_conv

    # --- EfficientNet / MobileNet at top-level
    for name in ["features", "blocks"]:
        seq = _get(model, name)
        if isinstance(seq, (nn.Sequential, nn.ModuleList, list)) and len(seq) > 0:
            return seq[select_index]

    print(model_name, 'select_index=', select_index)
    print(model)
    raise ValueError("Unsupported model for GradCAM: cannot resolve target layer automatically.")

def reshape_transform_vit_huggingface(x, num_patches=14):
    activations = x[:, 1:, :]
    # x.shape: (B, num_patches*num_patches, C)=>(B, num_patches, num_patches, C)
    activations = activations.reshape(x.shape[0],
                                   num_patches, num_patches, activations.shape[2])
    activations = activations.transpose(2, 3).transpose(1, 2) #(B, num_patches, num_patches, C)=>(B, C, num_patches, num_patches)
    return activations

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

""" Translate the category name to the category index.
    Some models aren't trained on Imagenet but on even larger datasets,
    so we can't just assume that 761 will always be remote-control.

"""
def category_name_to_index(model, category_name):
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]
    
""" Helper function to run GradCAM on an image and create a visualization.
    (note to myself: this is probably useful enough to move into the package)
    If several targets are passed in targets_for_gradcam,
    e.g different categories,
    a visualization for each of them will be created.
    
"""
def run_grad_cam_on_image(model: torch.nn.Module,
                          target_layer: torch.nn.Module,
                          targets_for_gradcam: List[Callable],
                          reshape_transform: Optional[Callable],
                          input_tensor: torch.Tensor,
                          input_image: Image,
                          method: Callable=GradCAM):
    with method(model=HuggingfaceToTensorModelWrapper(model),
                 target_layers=[target_layer],
                 reshape_transform=reshape_transform) as cam:

        # Replicate the tensor for each of the categories we want to create Grad-CAM for:
        repeated_tensor = input_tensor[None, :].repeat(len(targets_for_gradcam), 1, 1, 1)

        batch_results = cam(input_tensor=repeated_tensor,
                            targets=targets_for_gradcam)
        results = []
        for grayscale_cam in batch_results:
            visualization = show_cam_on_image(np.float32(input_image)/255,
                                              grayscale_cam,
                                              use_rgb=True)
            results.append(visualization)
        return np.hstack(results)

class PytorchCAM(torch.nn.Module):
    def __init__(self, model, model_name, img_size, target_module=None, select_index=-1, patch_size=14, method=GradCAM, reshape_transform=None, normalize_cam=True, device=None, debug=False):
        super(PytorchCAM, self).__init__()
        self.model = model
        self.model_name = model_name
        self.model.eval()
        self.img_size = img_size
        self.patch_size = patch_size
        self.target_module = target_module
        self.select_index = select_index
        self.features = None
        self.gradients = None
        self.device = device
        self.debug = debug
        self.debug_forward_count = 0
        self.debug_backward_count = 0
        self.debug_handles = []

        # Register hooks on the last layer of the encoder
        self.target_layer = _resolve_target_layer(model, model_name, module_name=target_module, select_index=select_index, debug=debug)
        print(f"Resolved target layer for GradCAM: {type(self.target_layer).__name__}")
        print(self.target_layer)

        if debug:
            print(f"\n[DEBUG] PytorchCAM initialized:")
            print(f"  Model: {model_name}")
            print(f"  Target layer: {type(self.target_layer).__name__}")
            print(f"  Target layer module path: {self._get_module_path(model, self.target_layer)}")
            print(f"  Image size: {img_size}")
            print(f"  Patch size: {patch_size}")
            print(f"  Method: {method.__name__}")

            # Check if target layer has parameters and if they require gradients
            param_count = sum(p.numel() for p in self.target_layer.parameters())
            grad_enabled_params = sum(p.numel() for p in self.target_layer.parameters() if p.requires_grad)
            print(f"  Target layer parameters: {param_count} total, {grad_enabled_params} require grad")
            if param_count > 0 and grad_enabled_params == 0:
                print(f"  WARNING: Target layer has parameters but none require gradients!")

            # Register debug hooks to monitor activations and gradients
            def debug_forward_hook(module, input, output):
                self.debug_forward_count += 1
                if isinstance(output, torch.Tensor):
                    print(f"  [DEBUG Hook] Forward pass #{self.debug_forward_count}:")
                    print(f"    Output shape: {output.shape}")
                    print(f"    Output min/max: {output.min():.4f} / {output.max():.4f}")
                    print(f"    Output mean/std: {output.mean():.4f} / {output.std():.4f}")
                    print(f"    Output requires_grad: {output.requires_grad}")
                    self.features = output
                else:
                    print(f"  [DEBUG Hook] Forward pass #{self.debug_forward_count}: output is not a tensor, type={type(output)}")

            def debug_backward_hook(module, grad_input, grad_output):
                self.debug_backward_count += 1
                print(f"  [DEBUG Hook] Backward pass #{self.debug_backward_count}:")
                if grad_output[0] is not None:
                    print(f"    Grad output shape: {grad_output[0].shape}")
                    print(f"    Grad output min/max: {grad_output[0].min():.4f} / {grad_output[0].max():.4f}")
                    print(f"    Grad output mean/std: {grad_output[0].mean():.4f} / {grad_output[0].std():.4f}")
                    self.gradients = grad_output[0]
                else:
                    print(f"    Grad output is None!")
                if grad_input[0] is not None:
                    print(f"    Grad input shape: {grad_input[0].shape}")
                else:
                    print(f"    Grad input is None!")

            # Register our debug hooks
            h1 = self.target_layer.register_forward_hook(debug_forward_hook)
            h2 = self.target_layer.register_full_backward_hook(debug_backward_hook)
            self.debug_handles.extend([h1, h2])

        # Set reshape transform if needed
        if reshape_transform is None:
            if 'vit' in model_name.lower() or 'dino' in model_name.lower() or 'retfound' in model_name.lower():
                reshape_transform = lambda x: reshape_transform_vit_huggingface(x, num_patches=img_size // patch_size)
            elif 'smp' in model_name.lower():
                # SMP models don't need reshape transform (CNN-based)
                reshape_transform = None
            else:
                reshape_transform = None
        self.method = method(model=HuggingfaceToTensorModelWrapper(model), target_layers=[self.target_layer], reshape_transform=reshape_transform)
        self.normalize_cam = normalize_cam

    def _get_module_path(self, model, target_module):
        """Helper to find the path to a target module in the model tree"""
        if self.debug:
            for name, module in model.named_modules():
                if module is target_module:
                    return name
        return "unknown"

    def cleanup_debug_hooks(self):
        """Remove debug hooks if they were registered"""
        if self.debug:
            for handle in self.debug_handles:
                handle.remove()
            self.debug_handles.clear()

    def compute_cam(self, pixel_values, targets_for_gradcam: List[Callable]):
        """Compute the CAM for the given pixel values and targets for Grad-CAM.

        Args:
            pixel_values (torch.Tensor): The pixel values of the image.
            targets_for_gradcam (List[Callable]): The targets index for Grad-CAM.

        Returns:
            torch.Tensor: The CAM for the given pixel values and targets for Grad-CAM.
        """
        pixel_values = to_tensor(pixel_values, device=self.device)
        # Ensure 4D input [B, C, H, W]
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        B = pixel_values.size(0)

        if self.debug:
            print(f"\n[DEBUG] compute_cam called:")
            print(f"  Input shape: {pixel_values.shape}")
            print(f"  Batch size: {B}")
            print(f"  Targets: {[t.category for t in targets_for_gradcam] if targets_for_gradcam else None}")
            print(f"  Input requires_grad: {pixel_values.requires_grad}")
            print(f"  Model training mode: {self.model.training}")
            print(f"  Resetting debug hook counters...")
            self.debug_forward_count = 0
            self.debug_backward_count = 0

        if self.debug:
            print(f"\n[DEBUG] Calling pytorch-grad-cam method...")

        with torch.set_grad_enabled(True):
            batch_results = torch.as_tensor(self.method(input_tensor=pixel_values, targets=targets_for_gradcam))  # shape: (B', H', W')

        if self.debug:
            print(f"\n[DEBUG] After pytorch-grad-cam method call:")
            print(f"  Forward hook fired {self.debug_forward_count} times")
            print(f"  Backward hook fired {self.debug_backward_count} times")
            if self.features is not None:
                print(f"  Captured features shape: {self.features.shape}")
                print(f"  Captured features min/max: {self.features.min():.4f} / {self.features.max():.4f}")
            else:
                print(f"  WARNING: No features captured!")
            if self.gradients is not None:
                print(f"  Captured gradients shape: {self.gradients.shape}")
                print(f"  Captured gradients min/max: {self.gradients.min():.4f} / {self.gradients.max():.4f}")
            else:
                print(f"  WARNING: No gradients captured!")

        if self.debug:
            print(f"  CAM output shape (before normalization): {batch_results.shape}")
            print(f"  CAM min/max: {batch_results.min():.4f} / {batch_results.max():.4f}")
            print(f"  CAM mean/std: {batch_results.mean():.4f} / {batch_results.std():.4f}")

            # Show histogram binning (10 bins) for distribution analysis
            hist = torch.histc(batch_results, bins=10, min=batch_results.min().item(), max=batch_results.max().item())
            bin_edges = torch.linspace(batch_results.min().item(), batch_results.max().item(), 11)
            print(f"  CAM binning result (10 bins):")
            total_pixels = batch_results.numel()
            for i in range(10):
                bin_start = bin_edges[i].item()
                bin_end = bin_edges[i+1].item()
                count = hist[i].item()
                pct = (count / total_pixels) * 100
                bar = '█' * int(pct / 2)  # Visual bar (50% = 25 chars)
                print(f"    [{bin_start:7.3f}, {bin_end:7.3f}): {int(count):6d} ({pct:5.1f}%) {bar}")

        # Normalize per image using actual output size to 0~1
        if self.normalize_cam:
            cam_min = batch_results.view(B, -1).min(dim=1)[0].view(B, 1, 1)
            cam_max = batch_results.view(B, -1).max(dim=1)[0].view(B, 1, 1)
            cam = (batch_results - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = batch_results

        return cam  # shape: (B, H, W)

    def forward(self, inputs=None, targets=None, model=None, **kwargs):
        if inputs is None:
            raise ValueError("inputs parameter is required")
        if targets is None:
            raise ValueError("targets parameter is required")
        # Convert targets to ClassifierOutputTarget objects
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if isinstance(targets, np.ndarray):
            targets = [ClassifierOutputTarget(int(t)) for t in targets]
        elif isinstance(targets, (list, tuple)) and not callable(targets[0]):
            targets = [ClassifierOutputTarget(int(t)) for t in targets]
        elif targets is None:
            pass
        elif isinstance(targets, int):
            targets = [ClassifierOutputTarget(targets)]
        else:
            raise ValueError(f"Unsupported targets type: {type(targets)}, {targets}")
        #print(targets)
        cam_bs = self.compute_cam(inputs, targets).detach().cpu()
        # back to original image size
        cam_bs = F.interpolate(cam_bs.unsqueeze(1), size=(self.img_size, self.img_size), mode='bilinear', align_corners=False) #add fake channel dimension

        if self.debug:
            print(f"  Final heatmap shape (after resize to {self.img_size}x{self.img_size}): {cam_bs.shape}")

        return cam_bs.squeeze(1).numpy() #remove fake channel dimension, shape: (B, img_size, img_size)

    def overlay_cam(self, image, cam):
        cam = np.uint8(255 * cam.detach().cpu().numpy())
        cam_img = Image.fromarray(cam).resize(image.size, resample=Image.BILINEAR)
        cmap = plt.get_cmap("jet")
        cam_colored = np.array(cmap(np.array(cam_img) / 255.0))[:, :, :3]
        overlay = 0.5 * (np.array(image) / 255.0) + 0.5 * cam_colored
        overlay = np.clip(overlay, 0, 1)
        return Image.fromarray(np.uint8(overlay * 255))

    def visualize(self, image_path):
        image, pixel_values = self.load_image(image_path)
        cam = self.compute_cam(pixel_values)
        overlay = self.overlay_cam(image, cam)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title("Grad-CAM Overlay")
        plt.axis('off')
        plt.show()

        

if __name__ == "__main__":
    # Load model and processor
    input_size = 224
    model_name = "google/vit-base-patch16-224-in21k"
    model = ViTForImageClassification.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)

    # Initialize GradCAM
    grad_cam = GradCAM(model=model, model_name=model_name, patch_size=14)

    # Load image and preprocess
    image = torch.randn(2, 3, input_size, input_size).cuda()  # Batch size of 2
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    pixel_values = inputs["pixel_values"]

    # Run GradCAM
    with torch.no_grad():
        cam = grad_cam(pixel_values)

    # Overlay heatmap
    overlay = grad_cam.overlay_cam(image, cam)

    # Display result
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Grad-CAM Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Clean up hooks
    grad_cam.cleanup()