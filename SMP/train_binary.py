import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2
import numpy as np
from tqdm import tqdm
import random

# Configuration
class Config:
    # Paths
    TRAIN_IMG_DIR = "/data/tl28853/eye/NR206/train_512"
    TRAIN_MASK_DIR = "/data/tl28853/eye/NR206/binary_masks"
    VAL_IMG_DIR = "/data/tl28853/eye/NR206/val_512"
    VAL_MASK_DIR = "/data/tl28853/eye/NR206/binary_masks"
    CHECKPOINT_DIR = "/data/tl28853/eye/segmentation_models.pytorch/checkpoints"
    
    # Model parameters
    ENCODER = 'resnet50'  # encoder: resnet34, efficientnet-b0, etc.
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = 1  # number of classes (binary segmentation)
    ACTIVATION = 'sigmoid'  # sigmoid for binary, softmax for multiclass
    
    # Training parameters
    DEVICE = 'cuda:7' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 50
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 512
    NUM_WORKERS = 4


# Dataset class
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images = sorted(os.listdir(images_dir))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert to Tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        
        # Resize
        image = TF.resize(image, [Config.IMAGE_SIZE, Config.IMAGE_SIZE])
        mask = TF.resize(mask, [Config.IMAGE_SIZE, Config.IMAGE_SIZE])
        
        # Normalize image
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return image, mask


# Training function
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    
    loop = tqdm(loader, desc='Training')
    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    return total_loss / len(loader)


# Validation function
def validate_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        loop = tqdm(loader, desc='Validation')
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
    
    return total_loss / len(loader)


# Main training loop
def main():
    # Create checkpoint directory
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # Create datasets
    train_dataset = SegmentationDataset(
        Config.TRAIN_IMG_DIR,
        Config.TRAIN_MASK_DIR,
    )
    
    val_dataset = SegmentationDataset(
        Config.VAL_IMG_DIR,
        Config.VAL_MASK_DIR,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Create model
    model = smp.Unet(
        encoder_name=Config.ENCODER,
        encoder_weights=Config.ENCODER_WEIGHTS,
        classes=Config.CLASSES,
        activation=Config.ACTIVATION,
    )
    model = model.to(Config.DEVICE)
    
    # Loss function and optimizer
    loss_fn = smp.losses.DiceLoss(mode='binary')
    # Alternative: smp.losses.JaccardLoss(mode='binary')
    # Or combine: loss_fn = smp.losses.DiceLoss(mode='binary') + nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(Config.EPOCHS):
        print(f'\nEpoch {epoch+1}/{Config.EPOCHS}')
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, Config.DEVICE)
        val_loss = validate_epoch(model, val_loader, loss_fn, Config.DEVICE)
        
        scheduler.step(val_loss)
        
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(Config.CHECKPOINT_DIR, 'best_model_binary.pth'))
            print(f'Best model saved with val_loss: {val_loss:.4f}')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(Config.CHECKPOINT_DIR, f'checkpoint_binary_epoch_{epoch+1}.pth'))


if __name__ == '__main__':
    main()
