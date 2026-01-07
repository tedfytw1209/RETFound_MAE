import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from tqdm import tqdm
from torch.utils.data import Dataset
from util.datasets import CSV_Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


@dataclass
class DatasetPreset:
    """Holds default paths for a dataset."""

    input_dir: Optional[str]
    csv_template: Optional[str]
    output_dir: Optional[str]
    image_column: str = "image"
    description: str = ""
    category: str = "public"
    modality: str = "OCT"


@dataclass
class RuntimeConfig:
    """Resolved runtime configuration."""

    dataset: str
    csv_path: Optional[Path]
    input_dir: Path
    output_dir: Path
    image_column: str
    category: str
    modality: str


@dataclass
class ImageRecord:
    """Internal representation of one inference item."""

    path: Path
    output_stem: str
    metadata: dict


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "celldata": DatasetPreset(
        input_dir="/orange/ruogu.fang/tienyuchang/CellData",
        csv_template="/orange/ruogu.fang/tienyuchang/CellData/OCT/{study}.csv",
        output_dir="/orange/ruogu.fang/tienyuchang/CellData_masks_{class_mode}",
        image_column="image",
        description="Mendeley CellData OCT classification splits (see util/datasets.py).",
        category="public",
    ),
    "octdl": DatasetPreset(
        input_dir="/orange/ruogu.fang/tienyuchang/OCTDL",
        csv_template="/orange/ruogu.fang/tienyuchang/OCTDL/{study}.csv",
        output_dir="/orange/ruogu.fang/tienyuchang/OCTDL_masks_{class_mode}",
        image_column="image",
        description="Kaggle OCTDL (see finetune_retfound_OCTDL_*.sh).",
        category="public",
    ),
    "uf": DatasetPreset(
        input_dir="/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired",
        csv_template="/orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/IRB2024_v5/split/tune5-eval5/{study}.csv",
        output_dir="/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired_masks_{class_mode}",
        image_column="OCT",
        description="UF benchmark IRB2024_v5 splits (see util/datasets.py).",
        category="uf",
        modality="OCT",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified segmentation inference for CellData, OCTDL, UF, or custom datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(list(DATASET_PRESETS.keys()) + ["custom"]),
        default="custom",
        help="Dataset preset. Use 'custom' to fully control paths.",
    )
    parser.add_argument(
        "--study",
        default="DME_all",
        help="Study name inserted into preset CSV templates (ignored for custom paths).",
    )
    parser.add_argument("--data-csv", type=str, help="CSV listing images (otherwise inferred from preset).")
    parser.add_argument("--input-dir", type=str, help="Directory that stores the raw images.")
    parser.add_argument("--output-dir", type=str, help="Directory to store segmentation masks.")
    parser.add_argument("--image-column", type=str, help="Column that stores the image paths inside the CSV.")
    parser.add_argument(
        "--dataset-category",
        choices=["public", "uf"],
        help="Override dataset category (public vs UF). Required for custom UF configs.",
    )
    parser.add_argument(
        "--class-mode",
        choices=["binary", "multiclass", "multiclass_resnet50", "multiclass_efficientb4"],
        default="multiclass",
        help="Binary: background vs retinal layer. Multiclass: background + layer per class.",
    )
    parser.add_argument("--num-classes", type=int, help="Model output classes. Defaults to 1 or 9 based on class-mode.")
    parser.add_argument(
        "--encoder",
        default="resnet50",
        help="Backbone used during training.",
    )
    parser.add_argument("--encoder-weights", default=None, help="smp encoder weights setting (usually None at inference).")
    parser.add_argument("--checkpoint", default="/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_binary.pth", help="Checkpoint (.pth) that stores model_state_dict.")
    parser.add_argument("--image-size", type=int, default=512, help="Square resize for inference.")
    parser.add_argument(
        "--activation",
        choices=["sigmoid", "softmax", "none", "auto"],
        default="auto",
        help="Override model activation. 'auto' picks sigmoid for binary and softmax for multiclass.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary mask probability threshold.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--export-formats",
        nargs="+",
        choices=["png", "npy"],
        default=["png"],
        help="Mask export formats.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose outputs already exist for all requested formats.",
    )
    parser.add_argument(
        "--uf-splits",
        default="test",
        help="Comma-separated split names used when loading UF CSV datasets (e.g., 'test' or 'val,test').",
    )
    parser.add_argument(
        "--uf-modality",
        default="OCT",
        help="Modality flag forwarded to CSV_Dataset for UF data (OCT/CFP/Thickness).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Stop after processing this many images (useful for smoke tests).",
    )
    parser.add_argument(
        "--save-composite",
        action="store_true",
        help="If set, save a side-by-side original+overlay visualization for each mask.",
    )
    return parser.parse_args()


def resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    preset = DATASET_PRESETS.get(args.dataset)

    if preset is None and args.dataset != "custom":
        raise ValueError(f"Unknown dataset preset: {args.dataset}")

    csv_path = args.data_csv
    input_dir = args.input_dir or (preset.input_dir if preset else None)
    output_dir = args.output_dir or (preset.output_dir.format(class_mode=args.class_mode) if preset and preset.output_dir else None)
    image_column = args.image_column or (preset.image_column if preset else "image")
    category = args.dataset_category or (preset.category if preset else "public")
    modality = preset.modality if preset else args.uf_modality

    if preset and not args.data_csv and preset.csv_template:
        csv_path = preset.csv_template.format(study=args.study)

    if input_dir is None:
        raise ValueError("input-dir is required for custom datasets.")

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else (input_dir / "pred_masks")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path_path: Optional[Path] = Path(csv_path) if csv_path else None
    if csv_path_path is not None and not csv_path_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path_path}")
    if category == "uf" and csv_path_path is None:
        raise ValueError("UF datasets require an accompanying CSV file.")

    return RuntimeConfig(
        dataset=args.dataset,
        csv_path=csv_path_path,
        input_dir=input_dir,
        output_dir=output_dir,
        image_column=image_column,
        category=category,
        modality=modality,
    )


class CSVImageDataset(Dataset):
    """Minimal dataset for public CSV files (CellData/OCTDL)."""

    def __init__(self, csv_path: Path, img_dir: Path, image_column: str = "image", label_column: str = "label"):
        self.df = pd.read_csv(csv_path)
        if image_column not in self.df.columns or label_column not in self.df.columns:
            raise ValueError(
                f"CSV must provide '{image_column}' and '{label_column}' columns. "
                f"Columns found: {list(self.df.columns)}"
            )
        self.img_dir = img_dir
        self.image_column = image_column
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row[self.image_column]))
        label = int(row[self.label_column])
        return label, img_path


def _build_records_from_public(cfg: RuntimeConfig) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    if cfg.csv_path:
        dataset = CSVImageDataset(cfg.csv_path, cfg.input_dir, cfg.image_column)
        missing = 0
        for _, img_path in dataset:
            path = Path(img_path).resolve()
            if not path.exists():
                missing += 1
                continue
            if cfg.input_dir in path.parents or path.parent == cfg.input_dir:
                metadata_rel = str(path.relative_to(cfg.input_dir))
            else:
                metadata_rel = str(path)
            records.append(ImageRecord(path=path, output_stem=path.stem, metadata={"relative_path": metadata_rel}))
        if missing:
            print(f"[WARN] Skipped {missing} rows because files were not found under {cfg.input_dir}.")
    else:
        for file in cfg.input_dir.iterdir():
            if not file.is_file():
                continue
            if file.suffix.lower() not in IMAGE_EXTENSIONS and file.suffix.lower() != ".npy":
                continue
            records.append(ImageRecord(path=file.resolve(), output_stem=file.stem, metadata={"relative_path": file.name}))

    if not records:
        raise RuntimeError("No images found for inference.")
    return records


def _parse_comma_separated(value: str) -> List[str]:
    splits = [token.strip() for token in value.split(",") if token.strip()]
    return splits or ["test"]


def _build_records_from_uf(cfg: RuntimeConfig, args: argparse.Namespace) -> List[ImageRecord]:
    if cfg.csv_path is None:
        raise ValueError("UF datasets require CSV metadata.")
    split_list = _parse_comma_separated(args.uf_splits)
    is_train_arg: List[str] | str = split_list if len(split_list) > 1 else split_list[0]

    uf_dataset = CSV_Dataset(
        str(cfg.csv_path),
        str(cfg.input_dir),
        is_train=is_train_arg,
        transfroms=None,
        k=0,
        modality=cfg.modality,
    )

    root_path = Path(uf_dataset.root_dir)
    records: List[ImageRecord] = []
    missing = 0
    for sample in uf_dataset.samples:
        image_entry = sample[0]
        if isinstance(image_entry, list):
            raise NotImplementedError("Half3D UF samples (k>0 for eval) are not supported in inference_general.")
        image_path = Path(image_entry)
        if not image_path.is_absolute():
            image_path = (root_path / image_entry).resolve()
        if not image_path.exists():
            missing += 1
            continue
        records.append(
            ImageRecord(
                path=image_path,
                output_stem=Path(image_entry).stem,
                metadata={"relative_path": str(Path(image_entry)), "label": int(sample[1])},
            )
        )
    if missing:
        print(f"[WARN] UF dataset skipped {missing} entries due to missing files under {root_path}.")
    if not records:
        raise RuntimeError("UF dataset produced zero valid samples.")
    return records


def build_image_records(cfg: RuntimeConfig, args: argparse.Namespace) -> List[ImageRecord]:
    if cfg.category == "uf":
        return _build_records_from_uf(cfg, args)
    return _build_records_from_public(cfg)


def create_model(args: argparse.Namespace) -> torch.nn.Module:
    num_classes = args.num_classes
    if num_classes is None:
        num_classes = 1 if args.class_mode == "binary" else 9

    activation = args.activation
    if activation == "auto":
        activation = "sigmoid" if num_classes == 1 else "softmax"

    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        classes=num_classes,
        activation=None if activation == "none" else activation,
    )

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    print(f"Checkpoint loaded from {args.checkpoint}")
    if "epoch" in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if "val_loss" in checkpoint:
        print(f"Validation loss: {checkpoint['val_loss']}")

    args.num_classes = num_classes
    args.activation = activation
    return model


def _ensure_rgb(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[2] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    elif array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return array


def _prepare_rgb_image(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = np.clip(image, 0.0, 1.0) * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return image


def load_image_as_tensor(image_path: Path, image_size: int) -> tuple[torch.Tensor, np.ndarray]:
    if image_path.suffix.lower() == ".npy":
        image = np.load(image_path)
        image = _ensure_rgb(image)
        image_rgb = _prepare_rgb_image(image)
    else:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Unable to read {image_path}")
        if image.ndim == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_image = image_rgb.copy()
    image_float = image_rgb.astype(np.float32)

    image_tensor = torch.from_numpy(image_float).permute(2, 0, 1) / 255.0
    image_tensor = TF.resize(image_tensor, [image_size, image_size])
    image_tensor = TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return image_tensor, original_image


def postprocess(output: torch.Tensor, original_size: Sequence[int], args: argparse.Namespace) -> np.ndarray:
    if args.num_classes == 1:
        mask = output.squeeze().cpu().numpy()
        mask = (mask > args.threshold).astype(np.uint8) * 255
    else:
        mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy().astype(np.uint8)

    mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def should_skip_outputs(record: ImageRecord, cfg: RuntimeConfig, export_formats: Sequence[str]) -> bool:
    existing = []
    for fmt in export_formats:
        ext = ".png" if fmt == "png" else ".npy"
        existing.append((cfg.output_dir / f"{record.output_stem}{ext}").exists())
    return all(existing)


def save_mask(mask: np.ndarray, record: ImageRecord, cfg: RuntimeConfig, export_formats: Sequence[str]) -> None:
    for fmt in export_formats:
        if fmt == "png":
            target = cfg.output_dir / f"{record.output_stem}.png"
            cv2.imwrite(str(target), mask)
        elif fmt == "npy":
            target = cfg.output_dir / f"{record.output_stem}.npy"
            np.save(str(target), mask)


def _colorize_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    if num_classes <= 1:
        mask_binary = (mask > 0).astype(np.uint8) * 255
        mask_color = np.zeros((*mask_binary.shape, 3), dtype=np.uint8)
        mask_color[..., 0] = mask_binary  # Red channel
    else:
        max_class = max(num_classes - 1, 1)
        mask_scaled = (mask.astype(np.float32) / max_class * 255).astype(np.uint8)
        mask_color_bgr = cv2.applyColorMap(mask_scaled, cv2.COLORMAP_JET)
        mask_color = cv2.cvtColor(mask_color_bgr, cv2.COLOR_BGR2RGB)
    return mask_color


def save_composite_image(
    original_image: np.ndarray,
    mask: np.ndarray,
    record: ImageRecord,
    cfg: RuntimeConfig,
    args: argparse.Namespace,
) -> None:
    mask_color = _colorize_mask(mask, args.num_classes)
    mask_color = cv2.resize(mask_color, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    target = cfg.output_dir / f"{record.output_stem}_vis.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[1].imshow(mask_color)
    axes[1].set_title("Mask")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(target), dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = resolve_runtime_config(args)
    records = build_image_records(cfg, args)
    model = create_model(args)

    print(f"Dataset preset: {args.dataset}")
    print(f"Total images: {len(records)}")
    print(f"Saving masks to: {cfg.output_dir}")
    print(f"Export formats: {args.export_formats}")
    if args.max_images is not None:
        print(f"Limiting run to first {args.max_images} images.")

    processed = 0
    for record in tqdm(records, desc="Running inference"):
        if args.max_images is not None and processed >= args.max_images:
            break
        if args.skip_existing and should_skip_outputs(record, cfg, args.export_formats):
            continue

        image_tensor, original_image = load_image_as_tensor(record.path, args.image_size)
        original_size = original_image.shape[:2]
        image_tensor = image_tensor.unsqueeze(0).to(args.device)

        with torch.inference_mode():
            output = model(image_tensor)

        mask = postprocess(output, original_size, args)
        if args.save_composite:
            save_composite_image(original_image, mask, record, cfg, args)
        else:
            save_mask(mask, record, cfg, args.export_formats)
        processed += 1

    print(f"Inference complete! Processed {processed} images.")


if __name__ == "__main__":
    main()

