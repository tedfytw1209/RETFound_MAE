import warnings
warnings.filterwarnings('ignore')
from torchvision import transforms
from datasets import load_dataset
from pytorch_grad_cam import run_dff_on_image, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import cv2
from typing import List, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification, ViTImageProcessor

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

    # --- HF ResNet (wrapped under .resnet)
    resnet = _get(model, "resnet")
    if resnet is not None:
        last_conv_name, last_conv = None, None
        for name, module in resnet.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_name, last_conv = name, module
        return last_conv

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
            return seq[-1]

    raise ValueError("Unsupported model for GradCAM: cannot resolve target layer automatically.")

""" Model wrapper to return a tensor"""
class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits

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
    def __init__(self, model, model_name, img_size, patch_size=14, method=GradCAM, reshape_transform=None, normalize_cam=True):
        super(PytorchCAM, self).__init__()
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
        self.method = method(model=HuggingfaceToTensorModelWrapper(model),target_layers=[self.target_layer],reshape_transform=reshape_transform)
        self.normalize_cam = normalize_cam
        
    def compute_cam(self, pixel_values, targets_for_gradcam: List[Callable]):
        """Compute the CAM for the given pixel values and targets for Grad-CAM.

        Args:
            pixel_values (torch.Tensor): The pixel values of the image.
            targets_for_gradcam (List[Callable]): The targets index for Grad-CAM.

        Returns:
            torch.Tensor: The CAM for the given pixel values and targets for Grad-CAM.
        """
        # Ensure 4D input [B, C, H, W]
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        B = pixel_values.size(0)

        # Expand inputs/targets to match pytorch-grad-cam expectations (batch len == len(targets))
        if len(targets_for_gradcam) == 1 and B > 1:
            targets_expanded = [targets_for_gradcam[0]] * B
            repeated_tensor = pixel_values
        elif len(targets_for_gradcam) > 1 and B == 1:
            targets_expanded = targets_for_gradcam
            repeated_tensor = pixel_values.repeat(len(targets_for_gradcam), 1, 1, 1)
            B = repeated_tensor.size(0)
        elif len(targets_for_gradcam) > 1 and B > 1:
            # replicate each image for each target
            repeated_tensor = pixel_values.repeat_interleave(len(targets_for_gradcam), dim=0)
            targets_expanded = targets_for_gradcam * B
            B = repeated_tensor.size(0)
        else:
            targets_expanded = targets_for_gradcam
            repeated_tensor = pixel_values
        print(repeated_tensor.shape, targets_expanded)  # (B', C, H, W), B' = B or B * len(targets)
        batch_results = self.method(input_tensor=repeated_tensor, targets=targets_expanded)
        print(batch_results)  # shape: (B', H', W')
        # Normalize per image
        if self.normalize_cam:
            cam_min = batch_results.view(B * len(targets_for_gradcam), -1).min(dim=1)[0].view(B * len(targets_for_gradcam), 1, 1)
            cam_max = batch_results.view(B * len(targets_for_gradcam), -1).max(dim=1)[0].view(B * len(targets_for_gradcam), 1, 1)
            cam = (batch_results - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = batch_results

        return cam  # shape: (B', H, W)

    def forward(self, pixel_values, targets_for_gradcam: List[Callable]):
        cam_bs = self.compute_cam(pixel_values, targets_for_gradcam).detach().cpu()
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