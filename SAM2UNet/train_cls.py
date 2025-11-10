import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from SAM2UNet.SAM2UNet_classifier import SAM2UNetClassifier


# -------------------- Dataset --------------------
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


# -------------------- Args --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--train_img_dir", required=True)
    ap.add_argument("--val_img_dir", required=True)
    ap.add_argument("--seg_ckpt", required=True)
    ap.add_argument("--save_dir", default="ckpts_cls")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=5e-2)
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--threshold_scan", action="store_true")
    ap.add_argument("--pos_weight", type=float, default=None)
    return ap.parse_args()


# -------------------- Main --------------------
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor()
    ])

    train_ds = CSVImageDataset(args.train_csv, args.train_img_dir, transform=tfm)
    val_ds   = CSVImageDataset(args.val_csv,   args.val_img_dir,   transform=tfm)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    model = SAM2UNetClassifier(num_classes=1,
                               seg_ckpt=args.seg_ckpt,
                               freeze_backbone=args.freeze_backbone).cuda()

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.wd)

    pos_weight = None
    if args.pos_weight is not None:
        pos_weight = torch.tensor([args.pos_weight], device="cuda")

    best_metric = -1.0
    for ep in range(args.epochs):
        # ---------------- Train ----------------
        model.train()
        for img, y in train_ld:
            img = img.cuda(non_blocking=True)
            y = torch.as_tensor(y, device=img.device).float().unsqueeze(1)  # [B,1]

            logits = model(img)["logits"]  # [B,1]
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # ---------------- Validate ----------------
        model.eval()
        all_probs, all_labels = [], []
        correct = total = 0
        with torch.no_grad():
            for img, y in val_ld:
                img = img.cuda(non_blocking=True)
                logits = model(img)["logits"]               # [B,1]
                prob = torch.sigmoid(logits).squeeze(1)     # [B]
                all_probs.append(prob.cpu())
                y_cpu = torch.as_tensor(y)
                all_labels.append(y_cpu)
                pred = (prob.cpu() > 0.5).long()
                correct += (pred == y_cpu).sum().item()
                total += y_cpu.numel()

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            roc_auc = float("nan")
        try:
            pr_auc = average_precision_score(all_labels, all_probs)
        except ValueError:
            pr_auc = float("nan")

        acc = correct / max(1, total)
        f1_at_05 = f1_score(all_labels, (all_probs > 0.5).astype(int))

        best_t, best_f1 = 0.5, f1_at_05
        if args.threshold_scan:
            ths = np.linspace(0.05, 0.95, 19)
            scores = [(t, f1_score(all_labels, (all_probs > t).astype(int))) for t in ths]
            best_t, best_f1 = max(scores, key=lambda x: x[1])

        print(f"[Epoch {ep}] "
              f"ACC={acc:.4f} | ROC-AUC={roc_auc:.4f} | PR-AUC={pr_auc:.4f} "
              f"| F1@0.5={f1_at_05:.4f} "
              f"{'(bestT=%.2f,F1=%.4f)' % (best_t,best_f1) if args.threshold_scan else ''}")

        metric_for_best = roc_auc if not np.isnan(roc_auc) else acc
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"epoch{ep:03d}_metric{metric_for_best:.4f}.pt"))
        if metric_for_best > best_metric:
            best_metric = metric_for_best
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pt"))

    print(f"Training done. Best metric={best_metric:.4f}. "
          f"Best weights saved to {os.path.join(args.save_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
