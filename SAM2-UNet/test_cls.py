# test_cls.py
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

from SAM2UNet_classifier import SAM2UNetClassifier


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
        return img, label, str(row["image"])


# -------------------- Args --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--test_img_dir", required=True)
    ap.add_argument("--seg_ckpt", required=True)
    ap.add_argument("--cls_ckpt", required=True)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--threshold_scan", action="store_true")
    ap.add_argument("--save_preds", type=str, default=None)
    return ap.parse_args()


# -------------------- Main --------------------
def main():
    args = parse_args()

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor()
    ])

    ds = CSVImageDataset(args.test_csv, args.test_img_dir, transform=tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    model = SAM2UNetClassifier(num_classes=1,
                               seg_ckpt=args.seg_ckpt,
                               freeze_backbone=True).cuda()
    state = torch.load(args.cls_ckpt, map_location="cuda")
    model.load_state_dict(state, strict=False)
    model.eval()

    all_probs, all_labels, all_names = [], [], []
    with torch.no_grad():
        for img, y, names in dl:
            img = img.cuda(non_blocking=True)
            logits = model(img)["logits"]               # [B,1]
            prob = torch.sigmoid(logits).squeeze(1)     # [B]
            all_probs.append(prob.cpu())
            all_labels.append(torch.as_tensor(y))
            all_names.extend(list(names))

    all_probs = torch.cat(all_probs).numpy()     # shape [N]
    all_labels = torch.cat(all_labels).numpy()   # shape [N]

    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(all_labels, all_probs)
    except ValueError:
        pr_auc = float("nan")

    th = float(args.threshold)
    preds = (all_probs > th).astype(int)
    acc = accuracy_score(all_labels, preds)
    f1_at_th = f1_score(all_labels, preds)

    msg = f"[TEST] ACC@{th:.2f}={acc:.4f} | F1@{th:.2f}={f1_at_th:.4f} | ROC-AUC={roc_auc:.4f} | PR-AUC={pr_auc:.4f}"
    print(msg)

    if args.threshold_scan:
        ths = np.linspace(0.05, 0.95, 19)
        scores = [(t, f1_score(all_labels, (all_probs > t).astype(int))) for t in ths]
        best_t, best_f1 = max(scores, key=lambda x: x[1])
        print(f"[TEST] Scan best threshold: t={best_t:.2f}, F1={best_f1:.4f}")

    if args.save_preds:
        df_out = pd.DataFrame({
            "image": all_names,
            "label": all_labels.astype(int),
            "prob": all_probs.astype(float),
            "pred@{:.2f}".format(th): preds.astype(int)
        })
        out_path = args.save_preds
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_out.to_csv(out_path, index=False)
        print(f"[TEST] Predictions saved to: {out_path}")


if __name__ == "__main__":
    main()
