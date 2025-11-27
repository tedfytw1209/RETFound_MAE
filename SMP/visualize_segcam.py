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
    
    # Visualization parameters
    ALPHA = 0.4  # Blending factor for heatmap overlay
    COLORMAP = cv2.COLORMAP_JET  # OpenCV colormap
    SAVE_INDIVIDUAL = True  # Save individual images
    SAVE_COMBINED = True  # Save combined grid
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    THRESHOLD = 0.5


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
        Dictionary with original image, CAM, overlay, and prediction info
    """
    # Load and preprocess
    image_tensor, original_image, original_size = preprocess_image(
        image_path, config.IMAGE_SIZE
    )
    image_tensor_batch = image_tensor.unsqueeze(0).to(config.DEVICE)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor_batch)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0, pred_class].item()
    
    # Generate SegCAM
    cam = segcam(
        inputs=image_tensor_batch,
        targets=pred_class,
        model=model
    )
    cam = cam[0]  # Get first image from batch
    
    # Resize original image to match processed size for visualization
    vis_image = cv2.resize(original_image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    vis_image = vis_image / 255.0
    
    # Create overlay
    overlay = overlay_cam_on_image(
        vis_image, cam, 
        alpha=config.ALPHA, 
        colormap=config.COLORMAP
    )
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(vis_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # CAM heatmap
    im = axes[1].imshow(cam, cmap='jet')
    axes[1].set_title(f'SegCAM Heatmap\n({config.CAM_TYPE})')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay\nPred: Class {pred_class} ({pred_prob:.3f})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save if requested
    if save_dir and config.SAVE_INDIVIDUAL:
        save_path = Path(save_dir) / f"{Path(image_path).stem}_segcam.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")
    
    plt.close()
    
    return {
        'image': vis_image,
        'cam': cam,
        'overlay': overlay,
        'pred_class': pred_class,
        'pred_prob': pred_prob,
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
            f"{Path(result['image_path']).stem}\n"
            f"Class {result['pred_class']} ({result['pred_prob']:.3f})",
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
        label = int(row["label"])
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
    
    # Initialize SegCAM
    print("\nInitializing SegCAM...")
    segcam = SegCAM(
        model=model,
        model_name="SMP_dec",
        img_size=config.IMAGE_SIZE,
        cam_type=config.CAM_TYPE,
        pixel_set=config.PIXEL_SET,
        n_classes=VisualizationConfig.CLASSES,
        device=config.DEVICE,
        normalize_cam=config.NORMALIZE_CAM
    )
    
    print(f"SegCAM initialized with:")
    print(f"  - CAM type: {config.CAM_TYPE}")
    print(f"  - Pixel set: {config.PIXEL_SET}")
    print(f"  - Target layer: {segcam.target_layer.__class__.__name__}")
    
    # Process images
    results = []
    print("\nProcessing images...")
    for label, image_file_name in tqdm(image_files_names, desc="Generating SegCAMs"):
        try:
            result = visualize_single_image(
                image_file_name, 
                model, 
                segcam, 
                config,
                save_dir=output_path if config.SAVE_INDIVIDUAL else None
            )
            results.append(result)
        except Exception as e:
            print(f"\nError processing {image_file_name}: {str(e)}")
            continue
    
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
    
    if results:
        # Count predictions per class
        from collections import Counter
        class_counts = Counter([r['pred_class'] for r in results])
        print("\nPrediction distribution:")
        for cls, count in sorted(class_counts.items()):
            print(f"  Class {cls}: {count} images ({count/len(results)*100:.1f}%)")
        
        # Average confidence
        avg_conf = np.mean([r['pred_prob'] for r in results])
        print(f"\nAverage prediction confidence: {avg_conf:.3f}")
    
    # Cleanup
    segcam.cleanup()
    print("\nDone!")


def visualize_single_sample(image_path, checkpoint_path, save_path=None):
    """Convenience function to visualize a single image.
    
    Args:
        image_path: Path to image file
        checkpoint_path: Path to model checkpoint
        save_path: Optional path to save visualization
    """
    config = VisualizationConfig()
    config.CHECKPOINT_PATH = checkpoint_path
    config.SAVE_INDIVIDUAL = save_path is not None
    
    # Load model
    model = load_model(checkpoint_path, config.DEVICE)
    
    # Initialize SegCAM
    segcam = SegCAM(
        model=model,
        model_name="SMP_Classifier",
        img_size=config.IMAGE_SIZE,
        cam_type=config.CAM_TYPE,
        pixel_set=config.PIXEL_SET,
        n_classes=VisualizationConfig.CLASSES,
        device=config.DEVICE,
        normalize_cam=config.NORMALIZE_CAM
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

