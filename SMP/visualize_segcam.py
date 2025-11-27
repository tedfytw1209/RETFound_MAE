import os
import sys
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from baselines.SegCAM import SegCAM
import segmentation_models_pytorch as smp


class VisualizationConfig:
    # Model parameters (should match training)
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = 1
    ACTIVATION = 'sigmoid'
    IMAGE_SIZE = 512
    
    # Model parameters (should match training)
    CHECKPOINT_PATH = "/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_binary.pth"
    
    # Paths
    DATA_CSV = "/orange/ruogu.fang/tienyuchang/CellData/OCT/DME_all.csv"  # CSV with image and label columns
    INPUT_DIR = "/orange/ruogu.fang/tienyuchang/CellData"
    OUTPUT_DIR = "/orange/ruogu.fang/tienyuchang/SegCAM_visualizations_celldata"
    
    # Image processing
    IMAGE_SIZE = 512
    
    # SegCAM parameters
    CAM_TYPE = "hirescam"  # "gradcam" or "hirescam"
    PIXEL_SET = "class"  # "image", "class", "point", or "zero"
    NORMALIZE_CAM = True
    # Target layer options: None/'auto', 'encoder_last', 'encoder_0', 'decoder_last', 'decoder_0', etc.
    # Examples: 'decoder_0' for first decoder block, 'encoder_last' for bottleneck
    TARGET_LAYER = None  # None for auto-detect (uses last decoder layer)
    
    # Visualization parameters
    ALPHA = 0.4  # Blending factor for heatmap overlay
    COLORMAP = cv2.COLORMAP_JET  # OpenCV colormap
    SAVE_INDIVIDUAL = True  # Save individual images
    SAVE_COMBINED = True  # Save combined grid
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    THRESHOLD = 0.5


def get_target_layer(model, target_layer_option=None):
    """Get target layer for SegCAM based on model architecture.
    
    Args:
        model: SMP model (e.g., Unet)
        target_layer_option: String specifying which layer to target:
            - None or 'auto': automatic selection (last decoder layer)
            - 'encoder_last': last encoder layer
            - 'encoder_0', 'encoder_1', etc.: specific encoder level
            - 'bottleneck': bottleneck layer (between encoder and decoder)
            - 'decoder_last': last decoder layer (default)
            - 'decoder_0', 'decoder_1', etc.: specific decoder level
    
    Returns:
        Target layer module
    """
    import torch.nn as nn
    
    if target_layer_option is None or target_layer_option == 'auto' or target_layer_option == 'decoder_last':
        # Default: last decoder layer
        last_conv = None
        for name, module in model.decoder.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            return last_conv
            
    elif target_layer_option == 'encoder_last':
        # Last encoder layer
        last_conv = None
        for name, module in model.encoder.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            return last_conv
            
    elif target_layer_option.startswith('encoder_'):
        # Specific encoder level
        try:
            level = int(target_layer_option.split('_')[1])
            encoder_layers = []
            for name, module in model.encoder.named_children():
                if hasattr(module, 'conv1') or isinstance(module, nn.Sequential):
                    encoder_layers.append(module)
            if level < len(encoder_layers):
                # Get last conv in this layer
                last_conv = None
                for name, module in encoder_layers[level].named_modules():
                    if isinstance(module, nn.Conv2d):
                        last_conv = module
                if last_conv is not None:
                    return last_conv
        except (ValueError, IndexError):
            pass
            
    elif target_layer_option == 'bottleneck':
        # Bottleneck (last encoder layer before decoder)
        last_conv = None
        for name, module in model.encoder.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            return last_conv
            
    elif target_layer_option.startswith('decoder_'):
        # Specific decoder level
        try:
            level = int(target_layer_option.split('_')[1])
            decoder_blocks = list(model.decoder.children())
            if level < len(decoder_blocks):
                # Get last conv in this block
                last_conv = None
                for name, module in decoder_blocks[level].named_modules():
                    if isinstance(module, nn.Conv2d):
                        last_conv = module
                if last_conv is not None:
                    return last_conv
        except (ValueError, IndexError):
            pass
    
    # Fallback: return last decoder layer
    last_conv = None
    for name, module in model.decoder.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        return last_conv
    
    raise ValueError(f"Could not find target layer for option: {target_layer_option}")


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    model = smp.Unet(
        encoder_name=VisualizationConfig.ENCODER,
        encoder_weights=None,
        classes=VisualizationConfig.CLASSES,
        activation=VisualizationConfig.ACTIVATION,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model loaded from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"Val Loss: {checkpoint['val_loss']:.4f}")
    if 'val_acc' in checkpoint:
        print(f"Val Acc: {checkpoint['val_acc']:.4f}")
    
    return model


def preprocess_image(image_path, image_size=512):
    """Load and preprocess image for inference"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image_tensor = TF.resize(image_tensor, [image_size, image_size])
    image_tensor = TF.normalize(
        image_tensor, 
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    
    return image_tensor, image, original_size


def denormalize_image(image_tensor):
    """Denormalize image tensor for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    image = image_tensor * std + mean
    image = torch.clamp(image, 0, 1)
    return image


def overlay_cam_on_image(image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Create overlay of CAM on original image.
    
    Args:
        image: numpy array [H, W, 3] in range [0, 1] or [0, 255]
        cam: CAM heatmap [H, W] in range [0, 1]
        alpha: Blending factor for heatmap
        colormap: OpenCV colormap
        
    Returns:
        numpy array [H, W, 3] with CAM overlay
    """
    # Ensure image is in [0, 1] range
    if image.max() > 1.0:
        image = image / 255.0
    
    # Normalize cam to [0, 1]
    if cam.max() > 0:
        cam = cam / cam.max()
    
    # Resize cam to match image size if needed
    if cam.shape != image.shape[:2]:
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]), 
                        interpolation=cv2.INTER_LINEAR)
    
    # Apply colormap
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Blend
    overlay = alpha * heatmap + (1 - alpha) * image
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def visualize_single_image(image_path, model, segcam, config, save_dir=None):
    """Visualize SegCAM for a single image.
    
    Args:
        image_path: Path to input image
        model: Trained model
        segcam: SegCAM instance
        config: Configuration object
        save_dir: Directory to save visualizations
        
    Returns:
        Dictionary with original image, mask, CAM, overlay, and prediction info
    """
    # Load and preprocess
    image_tensor, original_image, original_size = preprocess_image(
        image_path, config.IMAGE_SIZE
    )
    image_tensor_batch = image_tensor.unsqueeze(0).to(config.DEVICE)
    
    # Get predicted mask from model
    with torch.no_grad():
        pred_mask = model(image_tensor_batch)
        pred_mask_binary = (pred_mask.squeeze().cpu().numpy() > config.THRESHOLD).astype(np.uint8)
    
    # Generate SegCAM for segmentation
    # For segmentation, targets=None uses the predicted mask
    cam = segcam(
        inputs=image_tensor_batch,
        targets=None,  # Let SegCAM use predicted segmentation mask
        model=model
    )
    
    # Handle different output formats
    if isinstance(cam, torch.Tensor):
        cam = cam.cpu().numpy()
    if len(cam.shape) > 2:
        cam = cam[0]  # Get first image from batch if needed
    if len(cam.shape) > 2:
        cam = cam.squeeze()  # Remove any extra dimensions
    
    # Resize original image to match processed size for visualization
    vis_image = cv2.resize(original_image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    vis_image = vis_image / 255.0
    
    # Create overlay
    overlay = overlay_cam_on_image(
        vis_image, cam, 
        alpha=config.ALPHA, 
        colormap=config.COLORMAP
    )
    
    # Create visualization with 3 panels: image, mask, heatmap overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(vis_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Predicted mask
    axes[1].imshow(pred_mask_binary, cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    # SegCAM overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f'{config.CAM_TYPE.upper()} Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save if requested
    if save_dir and config.SAVE_INDIVIDUAL:
        save_path = Path(save_dir) / f"{Path(image_path).stem}_segcam.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        # print(f"Saved visualization to {save_path}")
    
    plt.close()
    
    return {
        'image': vis_image,
        'mask': pred_mask_binary,
        'cam': cam,
        'overlay': overlay,
        'image_path': image_path
    }


def create_summary_grid(results, save_path, max_images=16):
    """Create a summary grid of multiple visualizations.
    
    Args:
        results: List of result dictionaries from visualize_single_image
        save_path: Path to save summary grid
        max_images: Maximum number of images to include in grid
    """
    n_images = min(len(results), max_images)
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx in range(n_images):
        result = results[idx]
        axes[idx].imshow(result['overlay'])
        axes[idx].set_title(
            f"{Path(result['image_path']).stem}",
            fontsize=8
        )
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved summary grid to {save_path}")
    plt.close()

class CSVImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        if "image" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV file should include 'image' and 'label' columns.")
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row["image"]))
        # For segmentation, label might not be needed or could be mask path
        try:
            label = int(row["label"])
        except (ValueError, TypeError) as e:
            # If conversion fails, default to 0 (segmentation doesn't need label)
            label = 0
        return label, img_path

def main():
    config = VisualizationConfig()
    
    input_path = Path(config.INPUT_DIR)
    output_path = Path(config.OUTPUT_DIR)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    all_datasets = CSVImageDataset(config.DATA_CSV, config.INPUT_DIR)
    image_files_names = [n for n in all_datasets]

    print(f"Found {len(image_files_names)} images in {config.INPUT_DIR}")
    print("Sample image files:")
    print(image_files_names[:5])
    print(f"Saving results to {config.OUTPUT_DIR}")
    
    print("="*60)
    print("SegCAM Visualization for SMP Classifier")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    model = load_model(config.CHECKPOINT_PATH, config.DEVICE)
    print(model)
    
    # Initialize SegCAM with optional target layer
    print("\nInitializing SegCAM...")
    
    # Get target layer if specified
    target_layer = None
    if config.TARGET_LAYER is not None:
        target_layer = get_target_layer(model, config.TARGET_LAYER)
        print(f"Using specified target layer: {config.TARGET_LAYER}")
    
    segcam = SegCAM(
        model=model,
        model_name="SMP_dec",
        img_size=config.IMAGE_SIZE,
        cam_type=config.CAM_TYPE,
        pixel_set=config.PIXEL_SET,
        n_classes=VisualizationConfig.CLASSES,
        device=config.DEVICE,
        normalize_cam=config.NORMALIZE_CAM,
        target_layer=target_layer
    )
    
    print(f"SegCAM initialized with:")
    print(f"  - CAM type: {config.CAM_TYPE}")
    print(f"  - Pixel set: {config.PIXEL_SET}")
    print(f"  - Target layer: {segcam.target_layer.__class__.__name__}")
    
    # Process images
    results = []
    print("\nProcessing images...")
    for label, image_file_name in tqdm(image_files_names, desc="Generating SegCAMs"):
        result = visualize_single_image(
            image_file_name, 
            model, 
            segcam, 
            config,
            save_dir=output_path if config.SAVE_INDIVIDUAL else None
        )
        results.append(result)
    
    # Create summary grid
    if results and config.SAVE_COMBINED:
        print("\nCreating summary grid...")
        summary_path = output_path / "segcam_summary_grid.png"
        create_summary_grid(results, summary_path)
    
    # Print statistics
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
    print(f"Total images processed: {len(results)}")
    
    # Cleanup
    segcam.cleanup()
    print("\nDone!")


def visualize_single_sample(image_path, checkpoint_path, save_path=None, target_layer_option=None):
    """Convenience function to visualize a single image.
    
    Args:
        image_path: Path to image file
        checkpoint_path: Path to model checkpoint
        save_path: Optional path to save visualization
        target_layer_option: Optional target layer specification (e.g., 'encoder_last', 'decoder_0')
    """
    config = VisualizationConfig()
    config.CHECKPOINT_PATH = checkpoint_path
    config.SAVE_INDIVIDUAL = save_path is not None
    if target_layer_option is not None:
        config.TARGET_LAYER = target_layer_option
    
    # Load model
    model = load_model(checkpoint_path, config.DEVICE)
    
    # Get target layer if specified
    target_layer = None
    if config.TARGET_LAYER is not None:
        target_layer = get_target_layer(model, config.TARGET_LAYER)
    
    # Initialize SegCAM
    segcam = SegCAM(
        model=model,
        model_name="SMP_dec",
        img_size=config.IMAGE_SIZE,
        cam_type=config.CAM_TYPE,
        pixel_set=config.PIXEL_SET,
        n_classes=VisualizationConfig.CLASSES,
        device=config.DEVICE,
        normalize_cam=config.NORMALIZE_CAM,
        target_layer=target_layer
    )
    
    # Visualize
    result = visualize_single_image(
        image_path, 
        model, 
        segcam, 
        config,
        save_dir=Path(save_path).parent if save_path else None
    )
    
    # Show plot if not saving
    if save_path is None:
        plt.show()
    
    # Cleanup
    segcam.cleanup()
    
    return result


if __name__ == '__main__':
    main()

