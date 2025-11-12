import os
import torch
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


class Config:
    # Model parameters (should match training)
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = 1
    ACTIVATION = 'sigmoid'
    IMAGE_SIZE = 512
    
    # Paths
    CHECKPOINT_PATH = "/data/tl28853/eye/segmentation_models.pytorch/checkpoints/best_model_binary.pth"
    INPUT_DIR = "/data/tl28853/eye/NR206/test_512"
    OUTPUT_DIR = "/data/tl28853/eye/NR206/pred_binary"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    THRESHOLD = 0.5


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    model = smp.Unet(
        encoder_name=Config.ENCODER,
        encoder_weights=None,
        classes=Config.CLASSES,
        activation=Config.ACTIVATION,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")
    
    return model


def preprocess_image(image_path, image_size=512):
    """Load and preprocess image for inference"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image_tensor = TF.resize(image_tensor, [image_size, image_size])
    image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return image_tensor, original_size


def postprocess_mask(mask_tensor, original_size, threshold=0.5):
    """Convert model output to binary mask"""
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = (mask > threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    
    return mask


def main():
    # Load model
    model = load_model(Config.CHECKPOINT_PATH, Config.DEVICE)
    
    input_path = Path(Config.INPUT_DIR)
    output_path = Path(Config.OUTPUT_DIR)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images in {Config.INPUT_DIR}")
    print(f"Saving masks to {Config.OUTPUT_DIR}")
    
    # Process images
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Load and preprocess
            image_tensor, original_size = preprocess_image(str(image_file), Config.IMAGE_SIZE)
            image_tensor = image_tensor.unsqueeze(0).to(Config.DEVICE)
            
            # Predict
            with torch.no_grad():
                output = model(image_tensor)
            
            # Postprocess
            mask = postprocess_mask(output, original_size, Config.THRESHOLD)
            
            # Save mask
            mask_path = output_path / image_file.name
            cv2.imwrite(str(mask_path), mask)
        except Exception as e:
            print(f"\nError processing {image_file.name}: {str(e)}")
            continue
    
    print(f"\nInference complete! {len(image_files)} masks saved to {Config.OUTPUT_DIR}")


if __name__ == '__main__':
    main()
