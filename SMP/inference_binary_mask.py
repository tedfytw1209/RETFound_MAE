import os
import torch
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class Config:
    # Model parameters (should match training)
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = 1
    ACTIVATION = 'sigmoid'
    IMAGE_SIZE = 512
    
    # Paths
    CHECKPOINT_PATH = "/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_binary.pth"
    DATA_CSV = "/orange/ruogu.fang/tienyuchang/OCTDL/DME_all.csv"  # CSV with image and label columns
    INPUT_DIR = ""
    OUTPUT_DIR = "/orange/ruogu.fang/tienyuchang/OCTDL_masks"
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
    # Load model
    model = load_model(Config.CHECKPOINT_PATH, Config.DEVICE)
    
    input_path = Path(Config.INPUT_DIR)
    output_path = Path(Config.OUTPUT_DIR)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    all_datasets = CSVImageDataset(Config.DATA_CSV, Config.INPUT_DIR)
    image_files_names = [n for n in all_datasets]

    print(f"Found {len(image_files_names)} images in {Config.INPUT_DIR}")
    print(f"Saving masks to {Config.OUTPUT_DIR}")
    
    # Process images
    i = 0
    for label, img_path in tqdm(image_files_names, desc="Processing images"):
        try:
            # Load and preprocess
            image_tensor, original_size = preprocess_image(str(img_path), Config.IMAGE_SIZE)
            image_tensor = image_tensor.unsqueeze(0).to(Config.DEVICE)
            
            # Predict
            with torch.no_grad():
                output = model(image_tensor)
            
            # Postprocess
            mask = postprocess_mask(output, original_size, Config.THRESHOLD)
            # Save mask to .npy file
            mask_path = output_path / Path(img_path).name
            np.save(str(mask_path.with_suffix('.npy')), mask)
            if i<5:
                cv2.imwrite(str(mask_path.with_suffix('.png')), mask)
            i += 1
        except Exception as e:
            print(f"\nError processing {Path(img_path).name}: {str(e)}")
            continue

    print(f"\nInference complete! {len(image_files_names)} masks saved to {Config.OUTPUT_DIR}")


if __name__ == '__main__':
    main()
