import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification, ViTImageProcessor

class GradCAM(torch.nn.Module):
    def __init__(self, model, model_name, patch_size=14):
        super(GradCAM, self).__init__()
        self.model = model
        self.model_name = model_name
        self.model.eval()
        self.patch_size = patch_size

        self.features = None
        self.gradients = None

        # Register hooks on the last layer of the encoder
        target_layer = self.model.vit.encoder.layer[-1].output
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.features = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, pixel_values, target_class=None):
        outputs = self.model(pixel_values)
        logits = outputs.logits
        if target_class is None:
            target_class = logits.argmax(dim=-1).item()

        score = logits[0, target_class]
        self.model.zero_grad()
        score.backward()

        weights = self.gradients.mean(dim=1, keepdim=True)
        cam = (weights * self.features).sum(dim=-1).squeeze()
        cam = F.relu(cam[1:].reshape(self.patch_size, self.patch_size))
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

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