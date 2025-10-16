import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification, ViTImageProcessor
import re
import torch.nn as nn


def _get(obj, name, default=None):
    return getattr(obj, name, default)
def _resolve_target_layer(model, model_name=None):
    """
    嘗試解析各框架最後一個適合掛勾的層：
    - timm ViT:           model.blocks[-1]
    - HF ViT:             model.vit.encoder.layer[-1]
    - HF ViT (base_model):model.base_model.vit.encoder.layer[-1]
    - timm Swin:          model.layers[-1].blocks[-1]
    - HF Swin:            model.swin.encoder.layers[-1].blocks[-1]
    - torchvision ResNet: model.layer4[-1]
    - EfficientNet/MobileNet: model.features[-1] 或 blocks[-1]
    找不到就丟 ValueError，請外部顯式傳入。
    """
    # --- timm ViT 風格
    if _get(model, "blocks") is not None:
        blocks = model.blocks
        if isinstance(blocks, (nn.ModuleList, list)) and len(blocks) > 0:
            return blocks[-1]

    # --- HuggingFace ViT 風格
    vit = _get(model, "vit")
    if vit is not None:
        enc = _get(vit, "encoder")
        layers = _get(enc, "layer")
        if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
            return layers[-1]

    # --- HF 某些包裝在 base_model
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

    # --- EfficientNet / MobileNet 系
    for name in ["features", "blocks"]:
        seq = _get(model, name)
        if isinstance(seq, (nn.Sequential, nn.ModuleList, list)) and len(seq) > 0:
            return seq[-1]

    raise ValueError("Unsupported model for GradCAM: cannot resolve target layer automatically.")

class GradCAM(torch.nn.Module):
    def __init__(self, model, model_name, img_size, patch_size=14, target_layer=None):
        super(GradCAM, self).__init__()
        self.model = model
        self.model_name = model_name
        self.model.eval()
        self.img_size = img_size
        self.patch_size = patch_size

        self.features = None
        self.gradients = None

        # Register hooks on the last layer of the encoder
        if 'RETFound' in model_name:
            # timm or HuggingFace ViT
            self.target_layer = model.blocks[-1]
        else:
            self.target_layer = _resolve_target_layer(model, model_name)
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.features = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def compute_cam(self, pixel_values, target_class=None):
        is_batch = pixel_values.dim() == 4
        B = pixel_values.size(0) if is_batch else 1
        if not is_batch:
            pixel_values = pixel_values.unsqueeze(0)

        outputs = self.model(pixel_values)
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        if target_class is None:
            target_class = logits.argmax(dim=-1)
        elif isinstance(target_class, int):
            target_class = torch.tensor([target_class] * B, device=logits.device)

        scores = logits[range(B), target_class]
        self.model.zero_grad()
        scores.sum().backward()

        weights = self.gradients.mean(dim=1, keepdim=True)
        cam = (weights * self.features).sum(dim=-1)
        print(weights.shape, self.features.shape)
        print(cam.shape)
        if 'vit' in self.model_name.lower() or 'retfound' in self.model_name.lower():
            cam = F.relu(cam[:, 1:])  # Skip [CLS] token
            cam = cam.reshape(B, self.img_size // self.patch_size, self.img_size // self.patch_size)
        else:
            cam = F.relu(cam)

        # Normalize per image
        cam_min = cam.view(B, -1).min(dim=1)[0].view(B, 1, 1)
        cam_max = cam.view(B, -1).max(dim=1)[0].view(B, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam if is_batch else cam.squeeze(0)

    def forward(self, inputs=None, targets=None, model=None, **kwargs):
        if inputs is None:
            raise ValueError("inputs parameter is required")
        cam_bs = self.compute_cam(inputs, targets).detach().cpu()
        # back to original image size
        cam_bs = F.interpolate(cam_bs.unsqueeze(1), size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return cam_bs.squeeze(1).numpy() #shape: (B, img_size, img_size)

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

    def cleanup(self):
        self.forward_handle.remove()
        self.backward_handle.remove()
        

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