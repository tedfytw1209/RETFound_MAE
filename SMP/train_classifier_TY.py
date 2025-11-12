import os
from pathlib import Path
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from .smp_classifier import SMPClassifier


# Configuration
class Config:
    # Paths
    DATA_ROOT = "/orange/ruogu.fang/tienyuchang/OCTDL/"  # Root directory containing images
    TRAIN_CSV = "/orange/ruogu.fang/tienyuchang/OCTDL/DME_train.csv"  # CSV with image and label columns
    VAL_CSV = "/orange/ruogu.fang/tienyuchang/OCTDL/DME_test.csv"  # CSV with image and label columns
    CHECKPOINT_DIR = "/orange/ruogu.fang/tienyuchang/RETfound_results/checkpoints_octdl_dme_dec"
    
    # Model parameters
    SEG_ARCH = 'Unet'  # Unet, UnetPlusPlus, FPN, Linknet, PSPNet, MAnet, PAN, DeepLabV3, DeepLabV3Plus
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    NUM_CLASSES = 2
    IN_CHANNELS = 3
    MODE = 'dec'  # enc, dec, fuse
    FUSE_MODE = 'sum'  # sum, concat
    LEARNABLE_ALPHA = False
    ALPHA = 0.5
    PRETRAINED_SEG_CKPT = '/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth'
    DROPOUT = 0.0
    
    # Training parameters
    DEVICE = 'cuda:7' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 20
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    IMAGE_SIZE = 512
    NUM_WORKERS = 4
    USE_AMP = True


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
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(row["label"])
        return img, label


def get_transforms(img_size=512):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# Training function
def train_epoch(model, loader, optimizer, loss_fn, scaler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc='Training')
    for images, targets in loop:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=Config.USE_AMP):
            outputs = model(images, mode_dict=True)
            logits = outputs[Config.MODE]["logits"]
            loss = loss_fn(logits, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
        acc = 100.0 * correct / total
        postfix = {'loss': loss.item(), 'acc': f'{acc:.2f}%'}
        
        # Add alpha to postfix if learnable
        if Config.LEARNABLE_ALPHA and hasattr(model, 'fusion_alpha'):
            postfix['alpha'] = f'{model.fusion_alpha.item():.4f}'
        
        loop.set_postfix(postfix)
    
    return total_loss / len(loader), 100.0 * correct / total


# Validation function
def validate_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        loop = tqdm(loader, desc='Validation')
        for images, targets in loop:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images, mode_dict=True)
            logits = outputs[Config.MODE]["logits"]
            loss = loss_fn(logits, targets)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            acc = 100.0 * correct / total
            loop.set_postfix(loss=loss.item(), acc=f'{acc:.2f}%')
    
    return total_loss / len(loader), 100.0 * correct / total


# Main training loop
def main():
    # Create checkpoint directory
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # Save config
    with open(os.path.join(Config.CHECKPOINT_DIR, "config.yaml"), "w") as f:
        yaml.dump({k: v for k, v in vars(Config).items() if not k.startswith('_')}, f)
    
    # Create datasets
    train_dataset = CSVImageDataset(
        Config.TRAIN_CSV,
        Config.DATA_ROOT,
        transform=get_transforms(Config.IMAGE_SIZE)
    )
    
    val_dataset = CSVImageDataset(
        Config.VAL_CSV,
        Config.DATA_ROOT,
        transform=get_transforms(Config.IMAGE_SIZE)
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
    
    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
    
    # Create model
    model = SMPClassifier(
        seg_arch=Config.SEG_ARCH,
        encoder_name=Config.ENCODER,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        mode=Config.MODE,
        fuse_mode=Config.FUSE_MODE,
        learnable_alpha=Config.LEARNABLE_ALPHA,
        alpha=Config.ALPHA,
        pretrained_seg_ckpt=Config.PRETRAINED_SEG_CKPT,
        dropout=Config.DROPOUT,
    )
    model = model.to(Config.DEVICE)
    
    print(f'Model built: mode={Config.MODE}, seg_arch={Config.SEG_ARCH}, encoder={Config.ENCODER}')
    
    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    # AMP scaler
    scaler = GradScaler(enabled=Config.USE_AMP)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(Config.EPOCHS):
        print(f'\nEpoch {epoch+1}/{Config.EPOCHS}')
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, scaler, Config.DEVICE)
        val_loss, val_acc = validate_epoch(model, val_loader, loss_fn, Config.DEVICE)
        
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Print alpha if learnable
        if Config.LEARNABLE_ALPHA and hasattr(model, 'fusion_alpha'):
            print(f'Alpha: {model.fusion_alpha.item():.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Prepare checkpoint with alpha information
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': {
                    'mode': Config.MODE,
                    'seg_arch': Config.SEG_ARCH,
                    'encoder': Config.ENCODER,
                    'num_classes': Config.NUM_CLASSES,
                }
            }
            
            # Only save alpha-related config for fuse mode
            if Config.MODE == 'fuse':
                checkpoint['config']['fuse_mode'] = Config.FUSE_MODE
                checkpoint['config']['learnable_alpha'] = Config.LEARNABLE_ALPHA
                checkpoint['config']['alpha'] = model.fusion_alpha.item() if Config.LEARNABLE_ALPHA and hasattr(model, 'fusion_alpha') else Config.ALPHA
            
            torch.save(checkpoint, os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth'))
            print(f'Best model saved with val_acc: {val_acc:.2f}%')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': {
                    'mode': Config.MODE,
                    'seg_arch': Config.SEG_ARCH,
                    'encoder': Config.ENCODER,
                    'num_classes': Config.NUM_CLASSES,
                }
            }
            
            # Only save alpha-related config for fuse mode
            if Config.MODE == 'fuse':
                checkpoint['config']['fuse_mode'] = Config.FUSE_MODE
                checkpoint['config']['learnable_alpha'] = Config.LEARNABLE_ALPHA
                checkpoint['config']['alpha'] = model.fusion_alpha.item() if Config.LEARNABLE_ALPHA and hasattr(model, 'fusion_alpha') else Config.ALPHA
            
            torch.save(checkpoint, os.path.join(Config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print(f'\nTraining completed! Best Val Acc: {best_val_acc:.2f}%')


if __name__ == '__main__':
    main()
