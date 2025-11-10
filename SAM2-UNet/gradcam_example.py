import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from SAM2UNet_classifier import SAM2UNetClassifier


class GradCAMOnDecFeat:
    def __init__(self, model):
        self.model = model.eval()

    def _norm(self, x):
        x = x - x.min()
        x = x / (x.max() + 1e-6)
        return x

    def generate(self, img):
        """
        img: [1,3,H,W]
        """
        self.model.zero_grad(set_to_none=True)
        
        # Enable gradient computation
        torch.set_grad_enabled(True)
        
        out = self.model(img)
        logits = out["logits"]              # [1,1]
        feat = out["dec_feat"]              # [1,C,H',W']
        
        # Ensure feat requires gradient
        feat.requires_grad_(True)
        
        prob = torch.sigmoid(logits)[0, 0].item()

        score = logits[:, 0].sum()
        score.backward()

        grad = feat.grad
        
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feat).sum(dim=1, keepdim=True))  # [1,1,H',W']
        cam = F.interpolate(cam, size=img.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = self._norm(cam).cpu().numpy()
        
        torch.set_grad_enabled(False)

        return cam, prob


def overlay_heatmap(img_path, cam, out_path, alpha=0.45):
    img = np.array(Image.open(img_path).convert("RGB")) / 255.0
    hmap = plt.cm.jet(cam)[..., :3]  # jet colormap (RGB)
    heat = (1 - alpha) * img + alpha * hmap
    heat = np.clip(heat, 0, 1)
    plt.imsave(out_path, heat)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_ckpt", required=True)
    ap.add_argument("--cls_ckpt", required=True)
    ap.add_argument("--img_path", required=True)
    ap.add_argument("--save_path", default="heatmap_overlay.png")
    ap.add_argument("--img_size", type=int, default=512)
    args = ap.parse_args()

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor()
    ])
    img_pil = Image.open(args.img_path).convert("RGB")
    img_tensor = tfm(img_pil).unsqueeze(0).cuda()

    model = SAM2UNetClassifier(num_classes=1,
                               seg_ckpt=args.seg_ckpt,
                               freeze_backbone=True).cuda()
    state = torch.load(args.cls_ckpt, map_location="cuda")
    model.load_state_dict(state, strict=False)

    cam_gen = GradCAMOnDecFeat(model)
    cam, prob = cam_gen.generate(img_tensor)
    overlay_heatmap(args.img_path, cam, args.save_path)

    print(f"Heatmap saved to {args.save_path}")
    print(f"Predicted positive probability: {prob:.4f}")


if __name__ == "__main__":
    main()
