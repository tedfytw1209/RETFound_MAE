import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from networks.net_api import sub_module as sm

class ReLayNetClassifier(nn.Module):
    def __init__(self, params, num_classes=2):
        super(ReLayNetClassifier, self).__init__()
        
        self.encode1 = sm.EncoderBlock(params)
        params['num_channels'] = 64
        self.encode2 = sm.EncoderBlock(params)
        self.encode3 = sm.EncoderBlock(params)
        self.bottleneck = sm.BasicBlock(params)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(params['num_filters'], num_classes)

    def forward(self, x):
        x, _, _ = self.encode1(x)
        x, _, _ = self.encode2(x)
        x, _, _ = self.encode3(x)
        x = self.bottleneck(x)

        x = self.pool(x)             # (B, C, 1, 1)
        x = torch.flatten(x, 1)      # (B, C)
        logits = self.classifier(x)  # (B, 2)
        return logits

params = {
    'num_channels': 1,
    'num_filters': 64,
    'kernel_h': 3,
    'kernel_w': 7,
    'stride_conv': 1,
    'pool': 2,
    'stride_pool': 2,
    'num_classes': 1
}

batch_size  = 16
num_epochs  = 50
learning_rate = 1e-3
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

model = ReLayNetClassifier(params, num_classes=1).to(device)

pretrained_model = torch.load('models/Exp01/relaynet_epoch20.model', map_location=device)
model.encode1.load_state_dict(pretrained_model.encode1.state_dict(), strict=False)
model.encode2.load_state_dict(pretrained_model.encode2.state_dict(), strict=False)
model.encode3.load_state_dict(pretrained_model.encode3.state_dict(), strict=False)
model.bottleneck.load_state_dict(pretrained_model.bottleneck.state_dict(), strict=False)

for param in model.encode1.parameters(): param.requires_grad = False
for param in model.encode2.parameters(): param.requires_grad = False
for param in model.encode3.parameters(): param.requires_grad = False
for param in model.bottleneck.parameters(): param.requires_grad = False

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
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        label = row['label'].copy()
        return img[0:1], label

tfms = transforms.Compose([
    transforms.Resize((496,512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

train_ds = CSVImageDataset('../OCTDL/dme_train.csv', img_dir='..', transform=tfms)
val_ds   = CSVImageDataset('../OCTDL/dme_test.csv', img_dir='..', transform=tfms)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

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

            # sigmoid to get probabilities for the positive ("yes") class
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

            # store true labels
            all_labels.append(labels.cpu().numpy())

    # concatenate over batches
    all_probs  = np.vstack(all_probs)   # shape (N,1)
    all_labels = np.vstack(all_labels)  # shape (N,1)

    # flatten
    y_true = all_labels.ravel()
    y_pred = (all_probs > 0.5).astype(int).ravel()
    y_score = all_probs.ravel()         # for AUC, use the raw prob

    # compute metrics
    epoch_loss = running_loss / len(val_ds)
    epoch_acc = accuracy_score(y_true, y_pred)
    epoch_auc = roc_auc_score(y_true, y_score)

    return epoch_loss, epoch_acc, epoch_auc

best_auc = 0.0
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch()
    val_loss, val_acc, val_auc = eval_epoch()

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train — loss: {train_loss:.4f}, acc: {train_acc:.4f}")
    print(f"  Val   — loss: {val_loss:.4f}, acc: {val_acc:.4f}, auc: {val_auc:.4f}")

    # you can choose to checkpoint on best AUC instead of (or in addition to) acc:
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), 'classification_model_octdl_dme.pth')
        print(f"  ✓ New best AUC: {best_auc:.4f}")
