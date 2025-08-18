import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

train_size  = 0.05  # Portion of the training set to train
batch_size  = 16
num_epochs  = 1
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CSVImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.img_dir}/{row['image']}"
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = row['label']
        return img, label

tfms = transforms.Compose([
    transforms.Resize((380,380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

train_ds = CSVImageDataset('CellData/OCT/DME_train.csv', img_dir='.', transform=tfms)
val_ds   = CSVImageDataset('CellData/OCT/DME_test.csv', img_dir='.', transform=tfms)

seed_value = 42
torch.manual_seed(seed_value)

full_size = len(train_ds)
partial_size = int(train_size * full_size)
rest_size    = full_size - partial_size
partial_ds, _ = random_split(train_ds, [partial_size, rest_size])

train_loader = DataLoader(partial_ds, batch_size=16, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Train Loop
def train_epoch():
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(train_loader, desc='Train'):
        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) > 0.5).int()
        correct += (preds == labels.int()).sum().item()
        total   += labels.size(0)

    return running_loss/total, correct/total

# Eval Loop
def eval_epoch():
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs  = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Val'):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_probs  = np.vstack(all_probs)   # shape (N,1)
    all_labels = np.vstack(all_labels)  # shape (N,1)

    y_true = all_labels.ravel()
    y_pred = (all_probs > 0.5).astype(int).ravel()
    y_score = all_probs.ravel()         # for AUC, use the raw prob

    epoch_loss = running_loss / len(val_ds)
    epoch_acc = accuracy_score(y_true, y_pred)
    epoch_auc = roc_auc_score(y_true, y_score)

    return epoch_loss, epoch_acc, epoch_auc

# Main
best_auc = 0.0
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch()
    val_loss, val_acc, val_auc = eval_epoch()

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train — loss: {train_loss:.4f}, acc: {train_acc:.4f}")
    print(f"  Val   — loss: {val_loss:.4f}, acc: {val_acc:.4f}, auc: {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), 'CellData/OCT/effnetb4_DME.pth')
        print(f"  ✓ New best AUC: {best_auc:.4f}")