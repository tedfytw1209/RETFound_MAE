import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
from util.misc import to_tensor

def _get(obj, name, default=None):
    return getattr(obj, name, default)

def _resolve_target_layer(model, model_name=None):
    """
    Resolve the target layer for various model architectures.
    For segmentation models, we typically want the last encoder layer
    or the bottleneck layer before the decoder.
    """
    # --- SMP Classifier (Segmentation Models PyTorch)
    encoder = _get(model, "encoder")
    seg_model = _get(model, "seg_model")
    mode = _get(model, "mode")
    
    if encoder is not None and seg_model is not None:
        # This is an SMP-based model
        if mode == "enc" or mode == "fuse":
            # For encoder or fuse mode, target the last encoder layer
            last_conv = None
            for name, module in encoder.named_modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
            if last_conv is not None:
                return last_conv
        elif mode == "dec":
            # For decoder mode, target the decoder output
            decoder = _get(seg_model, "decoder")
            if decoder is not None:
                last_conv = None
                for name, module in decoder.named_modules():
                    if isinstance(module, nn.Conv2d):
                        last_conv = module
                if last_conv is not None:
                    return last_conv
    
    # --- timm ViT style
    if _get(model, "blocks") is not None:
        blocks = model.blocks
        if isinstance(blocks, (nn.ModuleList, list)) and len(blocks) > 0:
            return blocks[-1]

    # --- HuggingFace ViT style
    vit = _get(model, "vit")
    if vit is not None:
        enc = _get(vit, "encoder")
        layers = _get(enc, "layer")
        if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
            return layers[-1]

    # --- HF wrapped in base_model
    base = _get(model, "base_model")
    if base is not None:
        vit = _get(base, "vit")
        if vit is not None:
            enc = _get(vit, "encoder")
            layers = _get(enc, "layer")
            if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
                return layers[-1]

    # --- timm Swin
    if _get(model, "layers") is not None:
        layers = model.layers
        if len(layers) > 0 and _get(layers[-1], "blocks") is not None:
            blks = layers[-1].blocks
            if len(blks) > 0:
                return blks[-1]

    # --- HF Swin
    swin = _get(model, "swin")
    if swin is not None:
        enc = _get(swin, "encoder")
        layers = _get(enc, "layers")
        if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
            blks = _get(layers[-1], "blocks")
            if isinstance(blks, (nn.ModuleList, list)) and len(blks) > 0:
                return blks[-1]

    # --- torchvision ResNet
    if _get(model, "layer4") is not None and len(model.layer4) > 0:
        return model.layer4[-1]

    # --- EfficientNet / MobileNet
    for name in ["features", "blocks"]:
        seq = _get(model, name)
        if isinstance(seq, (nn.Sequential, nn.ModuleList, list)) and len(seq) > 0:
            return seq[-1]

    raise ValueError("Unsupported model for SegCAM: cannot resolve target layer automatically.")


class SegCAM(torch.nn.Module):
    """Seg-Grad-CAM and HiRes-Seg-Grad-CAM implementation for segmentation tasks.
    
    Citations:
        Kira Vinogradova, Alexandr Dibrov, & Eugene W. Myers (2020). Towards Interpretable 
        Semantic Segmentation via Gradient-weighted Class Activation Mapping. In Proceedings 
        of the AAAI Conference on Artificial Intelligence.
        
        Draelos, R., & Carin, L.. (2020). Use HiResCAM instead of Grad-CAM for faithful 
        explanations of convolutional neural networks.
    
    Args:
        model: The segmentation model
        model_name: Name of the model (for automatic layer resolution)
        img_size: Input image size
        cam_type: "gradcam" for Seg-Grad-CAM, "hirescam" for HiRes-Seg-Grad-CAM
        pixel_set: "image", "class", "point", or "zero" - defines the pixel set for weighting
        pixel_set_point: (x, y) coordinate for pixel_set="point"
        target_layer: Specific layer to hook (optional, auto-resolved if None)
        device: Torch device
    """
    
    def __init__(self, model, model_name, img_size, cam_type="gradcam", 
                 pixel_set="class", pixel_set_point=None, target_layer=None, n_classes=1000,
                 device=None, normalize_cam=True):
        super(SegCAM, self).__init__()
        self.model = model
        self.model_name = model_name
        self.model.eval()
        self.img_size = img_size
        self.device = device
        self.cam_type = cam_type
        self.pixel_set = pixel_set
        self.pixel_set_point = pixel_set_point
        self.n_classes = n_classes
        self.normalize_cam = normalize_cam
        
        self.features = None
        self.gradients = None

        # Register hooks on the target layer
        if target_layer is not None:
            self.target_layer = target_layer
        elif 'RETFound' in model_name:
            self.target_layer = model.blocks[-1]
        else:
            self.target_layer = _resolve_target_layer(model, model_name)
            
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.features = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def backward_cam(self, prediction, backward_class, pixel_set_mask=None):
        """Perform backward pass for a specific class with pixel set weighting.
        
        Args:
            prediction: Model prediction tensor [B, C, H, W]
            backward_class: Class index to compute CAM for
            pixel_set_mask: Pre-computed pixel set mask (optional)
        """
        # Get segmentation map
        if prediction.shape[1] > 1:
            # For multi-class segmentation, use softmax
            segmentation = nn.Softmax(dim=1)(prediction)
        else:
            segmentation = prediction
        
        # Get pixel set mask
        if pixel_set_mask is not None:
            px_set_m = pixel_set_mask.to(self.device)
        elif self.pixel_set == "image":
            # Use entire image
            px_set_m = torch.ones_like(segmentation[:, 0, :, :]).to(self.device)
        elif self.pixel_set == "class":
            # Use pixels belonging to the target class
            class_probs = segmentation[:, backward_class, :, :]
            px_set_m = (class_probs > 0.5).float().to(self.device)
        elif self.pixel_set == "zero":
            # Empty pixel set (sanity check)
            px_set_m = torch.zeros_like(segmentation[:, 0, :, :]).to(self.device)
        elif self.pixel_set == "point":
            # Single point
            if self.pixel_set_point is None:
                raise ValueError("pixel_set_point must be provided when pixel_set='point'")
            px_set_m = torch.zeros_like(segmentation[:, 0, :, :]).to(self.device)
            px_set_m[:, self.pixel_set_point[0], self.pixel_set_point[1]] = 1
        else:
            raise ValueError(f"Unknown pixel_set: {self.pixel_set}")
        
        # Backward pass weighted by pixel set
        self.model.zero_grad()
        weighted_output = prediction[:, backward_class, :, :].squeeze() * px_set_m
        torch.sum(weighted_output).backward(retain_graph=True)

    def compute_cam(self, pixel_values, target_class=None, pixel_set_mask=None):
        """Compute Seg-CAM for given input.
        
        Args:
            pixel_values: Input tensor [B, C, H, W]
            target_class: Target class index (or indices for batch)
            pixel_set_mask: Optional pre-computed pixel set mask
            
        Returns:
            CAM heatmap [B, H, W]
        """
        pixel_values = to_tensor(pixel_values, device=self.device)
        is_batch = pixel_values.dim() == 4
        B = pixel_values.size(0) if is_batch else 1
        if not is_batch:
            pixel_values = pixel_values.unsqueeze(0)

        # Forward pass
        outputs = self.model(pixel_values)
        
        # Extract logits/predictions
        if hasattr(outputs, 'logits'):
            prediction = outputs.logits
        else:
            prediction = outputs
            
        # Determine target class
        if target_class is None:
            # Use argmax per spatial location
            target_class = prediction.argmax(dim=1).flatten()[0].item()
        
        # Handle batch of target classes
        if isinstance(target_class, (list, tuple, np.ndarray, torch.Tensor)):
            if len(target_class) != B:
                raise ValueError(f"target_class length {len(target_class)} != batch size {B}")
            target_classes = target_class
        else:
            target_classes = [target_class] * B
        
        # Compute CAM for each sample in batch
        cams = []
        for i in range(B):
            # Backward pass for this class
            self.backward_cam(prediction[i:i+1], target_classes[i], 
                            pixel_set_mask[i:i+1] if pixel_set_mask is not None else None)
            
            # Compute activations based on CAM type
            if self.cam_type == "gradcam":
                # Seg-Grad-CAM: pool gradients spatially
                pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3]).detach()
                pooled_gradients *= 1000  # Scale factor from reference
                
                # Weight features by pooled gradients
                activations = self.features[i:i+1].detach().clone()
                for c in range(activations.shape[1]):
                    activations[:, c, :, :] *= pooled_gradients[c]
                    
            elif self.cam_type == "hirescam":
                # HiRes-Seg-Grad-CAM: use full gradient resolution
                gradients = self.gradients[i:i+1].detach()
                activations = self.features[i:i+1].detach() * gradients
            else:
                raise ValueError(f"Unknown cam_type: {self.cam_type}")
            
            # Sum over channels and apply ReLU
            heatmap = torch.sum(activations, dim=1).squeeze(0)
            heatmap = F.relu(heatmap)
            
            # Normalize
            if torch.max(heatmap) > 0 and self.normalize_cam:
                heatmap = heatmap / torch.max(heatmap)
            
            cams.append(heatmap)
        
        cam = torch.stack(cams, dim=0)
        return cam if is_batch else cam.squeeze(0)

    def forward(self, inputs=None, targets=None, model=None, pixel_set_mask=None, **kwargs):
        """Forward pass compatible with XAI evaluation framework.
        
        Args:
            inputs: Input images [B, C, H, W]
            targets: Target class indices (int, list, or tensor)
            model: Optional model override
            pixel_set_mask: Optional pixel set masks [B, H, W]
            
        Returns:
            CAM heatmaps [B, H, W] as numpy array
        """
        if inputs is None:
            raise ValueError("inputs parameter is required")
        
        # Convert targets to appropriate format
        if targets is not None:
            if isinstance(targets, torch.Tensor):
                targets = targets.cpu().numpy()
            if isinstance(targets, np.ndarray):
                targets = targets.tolist()
        
        with torch.set_grad_enabled(True):
            cam_bs = self.compute_cam(inputs, targets, pixel_set_mask).detach().cpu()
        
        # Resize to original image size
        if cam_bs.dim() == 2:
            cam_bs = cam_bs.unsqueeze(0)
        cam_bs = F.interpolate(cam_bs.unsqueeze(1), size=(self.img_size, self.img_size), 
                              mode='bilinear', align_corners=False)
        return cam_bs.squeeze(1).numpy()  # [B, H, W]

    def overlay_cam(self, image, cam, alpha=0.4):
        """Create overlay of CAM on original image.
        
        Args:
            image: PIL Image or numpy array
            cam: CAM heatmap (2D array or tensor)
            alpha: Blending factor for heatmap
            
        Returns:
            PIL Image with CAM overlay
        """
        if isinstance(cam, torch.Tensor):
            cam = cam.detach().cpu().numpy()
        
        # Normalize cam
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        
        # Convert image to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
            
        # Handle different image formats
        if img_array.ndim == 3 and img_array.shape[0] == 3:
            # CHW to HWC
            img_array = img_array.transpose(1, 2, 0)
        elif img_array.ndim == 2:
            # Grayscale to RGB
            img_array = np.stack([img_array] * 3, axis=-1)
            
        # Normalize image to [0, 1]
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        
        # Resize heatmap to match image size
        if cam.shape != img_array.shape[:2]:
            cam = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]), 
                           interpolation=cv2.INTER_LINEAR)
        
        # Apply colormap
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Blend
        overlay = alpha * heatmap + (1 - alpha) * img_array
        overlay = np.clip(overlay, 0, 1)
        
        return Image.fromarray(np.uint8(overlay * 255))

    def visualize(self, image_path, target_class=None, save_path=None):
        """Visualize CAM for an image.
        
        Args:
            image_path: Path to image or PIL Image
            target_class: Target class index
            save_path: Optional path to save visualization
        """
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
            
        # Preprocess (assuming standard normalization)
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Compute CAM
        cam = self.compute_cam(img_tensor, target_class)
        overlay = self.overlay_cam(image, cam)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title(f"Seg-CAM Overlay (class {target_class})")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()

    def cleanup(self):
        """Remove hooks."""
        self.forward_handle.remove()
        self.backward_handle.remove()

