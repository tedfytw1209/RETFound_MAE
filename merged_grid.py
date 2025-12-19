from pathlib import Path
from collections import defaultdict, OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import cv2
import torch
from torchvision import transforms

LOAD_MASK = True
IMG_MASK = False
HEATMAP_MASK = False
DRAW_LAYER = True
Thickness_DIR = "/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"
Thickness_CSV = "/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/thickness_map.csv"
Task_list = ['DME']
dataset_fname = 'sampled_labels01.csv'
dataset_dir = '/blue/ruogu.fang/tienyuchang/OCT_EDA'
img_p_fmt = "label_%d/%s" #label index and oct_img name

#mask function
def masked_img_func(img, mask_slice):
    binary_mask = np.zeros_like(img, dtype=np.uint8)
    for i in range(mask_slice.shape[0]-1):
        upper = mask_slice[i].astype(int)
        lower = mask_slice[i+1].astype(int)
        for x in range(img.shape[1]):
            binary_mask[upper[x]:lower[x], x] = 1

    # 套用 mask (把 mask=0 的地方設為 0)
    masked_img = img.copy()
    masked_img[binary_mask == 0] = 0

    return masked_img

# Data loading and preprocessing functions
def load_sample_data(task, num_sample=-1):
    """Load sample images for a given task"""
    df = pd.read_csv(os.path.join(dataset_dir, "%s_sampled"%task, dataset_fname))
    if LOAD_MASK:
        masked_df = pd.read_csv(Thickness_CSV)
        masked_df = masked_df.rename(columns={'OCT':'folder'}).dropna(subset=['Surface Name'])
        df = df.merge(masked_df,on='folder',how='inner').reset_index(drop=True)
        print('After adding mask, data len: ', df.shape[0])
    task_df = df[df['label'].isin([0, 1])]  # Adjust based on actual DME labels
    # Sample random images
    if num_sample > 0:
        task_df = task_df.sample(n=num_sample, random_state=42).reset_index(drop=True)
    else:
        task_df = task_df.reset_index(drop=True)
    
    images = []
    labels = []
    filenames = []
    mask_slices = []
    
    for _, row in task_df.iterrows():
        # Extract just the filename from oct_img
        filename = os.path.basename(row['OCT']) if isinstance(row['OCT'], str) else row['OCT']
        img_path = os.path.join(dataset_dir, "%s_sampled"%task, img_p_fmt % (row['label'], filename))
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                if LOAD_MASK:
                    mask_path = os.path.join(Thickness_DIR, row['folder'], row['Surface Name'])
                    mask = np.load(mask_path) # (Layer, slice, W)

                    # 假設我們要套用其中某一 slice 的 mask，例如 slice_index = 13
                    slice_index = int(os.path.basename(img_path).split("_")[-1].split(".")[0])  # 從檔名抓 13
                    mask_slice = mask[:, slice_index, :]  # shape: (Layer, W)
                    mask_slices.append(mask_slice)
                else:
                    mask_slices.append(None)
                    
                if IMG_MASK:
                    img_np = np.array(img)  # Convert PIL image to numpy array
                    masked_img_np = masked_img_func(img_np, mask_slice)
                    masked_img = Image.fromarray(masked_img_np)
                    images.append(masked_img)
                else:
                    images.append(img)
                labels.append(row['label'])
                # Store filename without extension for directory naming
                image_name = os.path.splitext(filename)[0]
                filenames.append(image_name)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue


    return images, labels, filenames, mask_slices

#test dataset
dme_imgs, dme_labels, dme_img_names, dme_mask_slices = load_sample_data('DME',-1)

def add_layer_line(overlay, mask_slice, width=1, cmap_name="rainbow"):
    # Convert to PIL.Image if needed
    if isinstance(overlay, np.ndarray):
        if overlay.dtype != np.uint8:
            overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
        overlay_img = Image.fromarray(overlay)
    else:
        overlay_img = overlay.convert("RGB")
    draw = ImageDraw.Draw(overlay_img)
    n_layers, W = mask_slice.shape
    xs = np.arange(W)
    # Generate rainbow colors for each layer
    cmap = plt.get_cmap(cmap_name)
    colors = (np.array([cmap(i / max(1, n_layers - 1))[:3] for i in range(n_layers)]) * 255).astype(int)
    # Draw each layer line
    for i in range(n_layers):
        ys = np.nan_to_num(mask_slice[i].astype(float), nan=0.0)
        ys = np.clip(ys, 0, overlay_img.height - 1)
        points = list(zip(xs, ys))
        color = tuple(colors[i])
        draw.line(points, fill=color, width=width)

    return overlay_img

# 你的資料夾結構（可依需要調整）
# ./heatmap_results/<task>/<label_idx>/<image_name>/<model>/<module_layer>/XAI.jpg
BASE_DIR = Path("./heatmap_results_production")
OUT_DIR  = Path("./grid_heatmap_production")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Model order for display
FIXED_MODEL_ORDER = [
    'SMP_enc', 'SMP_dec', 'SMP_enc_fix', 'SMP_dec_fix',
    'SMP_fuse_multiply_fus0enc-1dec-1_seg',
    'SMP_fuse_channel_merge_fus0enc-1dec-1_seg',
    'SMP_fuse_channel_multiply_fus0enc-1dec-1_seg',
    'SMP_fuse_weighted_sum_fus0enc-1dec-1_seg',
    'SMP_fuse_channel_merge_fus0enc-1dec-1_dec',
    'SMP_fuse_add_fus0enc-1dec-1_dec',
    'SMP_fuse_multiply_fus0enc-1dec-1_dec',
    'SMP_fuse_weighted_sum_fus0enc-1dec-1_dec',
    'SMP_fuse_add_fus8enc-2dec-1_seg',
    'SMP_fuse_channel_merge_fus8enc-2dec-1_seg',
    'SMP_fuse_channel_multiply_fus0enc-2dec-1_seg',
    'SMP_fuse_multiply_fus8enc-2dec-1_seg',
    'SMP_fuse_weighted_sum_fus8enc-2dec-1_seg',
]
FIXED_METHOD_ORDER = ['GradCAM', 'HiResCAM', 'GradCAMPlusPlus']

# 支援的影像副檔名
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS and p.is_file()

def _safe_open(path: Path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None

def _collect_index_per_model(task_dir: Path):
    """
    Collect index organized per model with module-layer information.
    
    Returns:
    - index: {(label, image_name): {model: {'original': Path, 'mask': Path, 'layers': {module_layer: {method: Path}}}}}
    - models: set([...])
    - methods: set([...])
    - module_layers: set([...])
    """
    index = defaultdict(lambda: defaultdict(lambda: {'original': None, 'mask': None, 'layers': defaultdict(dict)}))
    models, methods, module_layers = set(), set(), set()

    # Walk through task directory
    for path in task_dir.rglob("*"):
        if not path.is_file():
            continue
        
        # Check for original image and mask at model level
        if path.name == "original_image.jpg":
            try:
                model = path.parent.name
                image_name = path.parent.parent.name
                label = path.parent.parent.parent.name
                key = (label, image_name)
                index[key][model]['original'] = path
                models.add(model)
            except Exception:
                continue
        elif path.name == "mask.jpg":
            try:
                model = path.parent.name
                image_name = path.parent.parent.name
                label = path.parent.parent.parent.name
                key = (label, image_name)
                index[key][model]['mask'] = path
            except Exception:
                continue
        elif path.suffix.lower() in IMG_EXTS and path.stem in ['GradCAM', 'HiResCAM', 'GradCAMPlusPlus', 'ScoreCAM', 'RISE', 'Attention']:
            # XAI heatmap: task/label/image_name/model/module_layer/XAI.jpg
            try:
                method = path.stem
                module_layer = path.parent.name  # e.g., encoder_10, decoder_0, head_-1
                model = path.parent.parent.name
                image_name = path.parent.parent.parent.name
                label = path.parent.parent.parent.parent.name
                
                key = (label, image_name)
                index[key][model]['layers'][module_layer][method] = path
                models.add(model)
                methods.add(method)
                module_layers.add(module_layer)
            except Exception:
                continue

    return index, models, methods, module_layers

def _order(items, fixed):
    """若有固定順序，先照固定清單排序，其餘依字母序接在後面"""
    if not fixed:
        return sorted(items)
    seen = set()
    ordered = []
    # 先加入 fixed 中存在的
    for x in fixed:
        if x in items and x not in seen:
            ordered.append(x); seen.add(x)
    # 再加上剩餘（字母序）
    for x in sorted(items):
        if x not in seen:
            ordered.append(x)
    return ordered

def _order_module_layers(module_layers):
    """Order module layers: encoder first, then decoder, then head, sorted by index
    
    Order: encoder_10, encoder_23, encoder_42, encoder_52, decoder_0, decoder_1, ..., head_0, head_-1
    Negative indices (like -1 for last layer) are sorted at the end of each module group.
    """
    def sort_key(ml):
        parts = ml.rsplit('_', 1)
        if len(parts) == 2:
            module, idx = parts[0], parts[1]
            try:
                idx_num = int(idx)
                # Handle negative indices: sort them at the end (e.g., -1 means last layer)
                # Map negative to large positive for sorting: -1 -> 10000-1=9999, -2 -> 9998, etc.
                if idx_num < 0:
                    idx_num = 10000 + idx_num  # -1 becomes 9999, -2 becomes 9998
            except ValueError:
                idx_num = 999
        else:
            module, idx_num = ml, 0
        
        # Priority: encoder=0, decoder=1, head=2, others=3
        if 'encoder' in module:
            priority = 0
        elif 'decoder' in module:
            priority = 1
        elif 'head' in module:
            priority = 2
        else:
            priority = 3
        
        return (priority, idx_num)
    
    return sorted(module_layers, key=sort_key)

def _draw_grid_per_model(task_name, key, model_name, model_data, methods, module_layers, 
                          save_dir: Path, cell_size=(256, 256), dpi=150, mask_index=None, line_width=2):
    """
    Draw a grid for a single model showing:
    - Row 0: Image, Mask (spanning first 2 columns)
    - Other rows: Each module-layer output
    - Columns: XAI methods
    
    model_data: {'original': Path, 'mask': Path, 'layers': {module_layer: {method: Path}}}
    """
    # Filter module_layers to only those that have data for this model
    available_layers = [ml for ml in module_layers if ml in model_data['layers']]
    if not available_layers:
        return None
    
    n_methods = len(methods)
    n_layers = len(available_layers)
    
    # Grid layout: 
    # Row 0: Image | Mask | (empty cells for remaining methods)
    # Row 1+: Module-Layer heatmaps for each XAI method
    n_rows = 1 + n_layers  # 1 for image/mask row + rows for each layer
    n_cols = max(n_methods, 2)  # At least 2 columns for image and mask
    
    fig_w = max(8, n_cols * 3)
    fig_h = max(4, n_rows * 2.5)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    
    # Handle single row/column cases
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    
    # Turn off ticks but keep axes for labels
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
            for spine in axes[r][c].spines.values():
                spine.set_visible(False)
    
    # Column headers in row 0 only (XAI method names)
    for c, method in enumerate(methods):
        axes[0][c].set_title(method, fontsize=10)
    
    # Row label for row 0 (first column only)
    axes[0][0].set_ylabel("Input", fontsize=10, rotation=0, labelpad=55, va='center', fontweight='bold')
    
    # Row 0: Image (col 0) and Mask (col 1)
    # Image
    if model_data['original'] is not None:
        im = _safe_open(model_data['original'])
        if im is not None:
            if cell_size:
                im = im.resize(cell_size)
            axes[0][0].imshow(im)
    else:
        axes[0][0].text(0.5, 0.5, "No Image", ha="center", va="center", fontsize=10, transform=axes[0][0].transAxes)
    
    # Mask
    if n_cols > 1:
        if model_data['mask'] is not None:
            im = _safe_open(model_data['mask'])
            if im is not None:
                if cell_size:
                    im = im.resize(cell_size)
                axes[0][1].imshow(im)
        else:
            axes[0][1].text(0.5, 0.5, "No Mask", ha="center", va="center", fontsize=10, transform=axes[0][1].transAxes)
    
    # Rows 1+: Module-Layer heatmaps
    for r_idx, module_layer in enumerate(available_layers):
        row = r_idx + 1  # Skip first row (image/mask)
        layer_data = model_data['layers'].get(module_layer, {})
        
        # Row label - use full layer name (e.g., encoder_52, decoder_9, head_0)
        display_label = module_layer
        
        # Add y-axis label on the first column for each row
        axes[row][0].set_ylabel(display_label, fontsize=9, rotation=0, labelpad=55, va='center')
        
        for c, method in enumerate(methods):
            ax = axes[row][c]
            
            path = layer_data.get(method, None)
            
            if path is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10, transform=ax.transAxes)
                continue
            
            im = _safe_open(path)
            if im is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10, transform=ax.transAxes)
                continue
            
            if cell_size is not None:
                im = im.resize(cell_size)
            
            ax.imshow(im)
    
    # Overall title
    label, image_name = key
    # Shorten model name for title
    short_model = model_name.replace('_fus0enc-1dec-1', '').replace('_fus8enc-2dec-1', '_fea')
    fig.suptitle(f"{task_name} | {short_model}\nImage: {image_name} | Label: {label}", fontsize=12)
    
    plt.tight_layout(rect=[0.08, 0, 1, 0.93])
    # Save in <img_name> directory
    img_save_dir = save_dir / f"{label}_{image_name}"
    img_save_dir.mkdir(parents=True, exist_ok=True)
    out_path = img_save_dir / f"{model_name}_grid.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path

def build_all_grids_per_model(base_dir: Path = BASE_DIR, out_dir: Path = OUT_DIR, 
                               mask_index=None, line_width=2, selected_models=None):
    """Build grids for each model separately, showing image, mask, and all module-layer outputs"""
    tasks = [p for p in base_dir.iterdir() if p.is_dir()]
    if not tasks:
        print(f"[WARN] 在 {base_dir} 底下沒有找到任何 task 資料夾。")
        return

    for task_dir in tasks:
        task_name = task_dir.name
        print(f"[Task] {task_name} → 掃描中…")

        index, models_set, methods_set, module_layers_set = _collect_index_per_model(task_dir)
        if not index:
            print(f"  - 找不到任何影像（{task_dir}）。略過。")
            continue

        # Filter and order models
        if selected_models:
            models = [m for m in _order(models_set, FIXED_MODEL_ORDER) if m in selected_models]
        else:
            models = [m for m in _order(models_set, FIXED_MODEL_ORDER)]
        
        methods = _order(methods_set, FIXED_METHOD_ORDER)
        module_layers = _order_module_layers(module_layers_set)

        print(f"  - 模型：{models}")
        print(f"  - 方法：{methods}")
        print(f"  - 模組層：{module_layers}")
        
        save_dir = out_dir / task_name

        # Process each sample
        for key in tqdm(sorted(index.keys()), desc=f"  產出 {task_name} grids"):
            sample_data = index[key]
            
            # Create grid for each model
            for model_name in models:
                if model_name not in sample_data:
                    continue
                model_data = sample_data[model_name]
                
                _ = _draw_grid_per_model(
                    task_name, key, model_name, model_data, 
                    methods, module_layers, save_dir,
                    mask_index=mask_index, line_width=line_width
                )

    print("✅ 全部完成！")


# ============================================================================
# Alternative: Draw comparison grid across models for same layer
# ============================================================================

def _draw_comparison_grid(task_name, key, sample_data, models, methods, module_layer,
                           save_dir: Path, cell_size=(256, 256), dpi=150):
    """
    Draw a comparison grid for a specific module-layer across all models.
    Rows: Models
    Columns: Image | Mask | XAI methods
    """
    n_models = len(models)
    n_cols = 2 + len(methods)  # Image + Mask + XAI methods
    
    fig_w = max(10, n_cols * 2.5)
    fig_h = max(6, n_models * 2.5)
    
    fig, axes = plt.subplots(n_models, n_cols, figsize=(fig_w, fig_h))
    
    if n_models == 1:
        axes = [axes]
    
    # Turn off ticks but keep axes for labels
    for r in range(n_models):
        for c in range(n_cols):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
            for spine in axes[r][c].spines.values():
                spine.set_visible(False)
    
    # Column headers
    col_headers = ['Image', 'Mask'] + list(methods)
    for c, header in enumerate(col_headers):
        axes[0][c].set_title(header, fontsize=10)
    
    for r, model_name in enumerate(models):
        model_data = sample_data.get(model_name, {'original': None, 'mask': None, 'layers': {}})
        
        # Row label (model name) - y-axis label
        short_model = model_name.replace('_fus0enc-1dec-1', '').replace('_fus8enc-2dec-1', '_fea')
        axes[r][0].set_ylabel(short_model, fontsize=8, rotation=0, labelpad=70, va='center')
        
        # Image
        if model_data['original'] is not None:
            im = _safe_open(model_data['original'])
            if im is not None:
                if cell_size:
                    im = im.resize(cell_size)
                axes[r][0].imshow(im)
        
        # Mask
        if model_data['mask'] is not None:
            im = _safe_open(model_data['mask'])
            if im is not None:
                if cell_size:
                    im = im.resize(cell_size)
                axes[r][1].imshow(im)
        
        # XAI methods
        layer_data = model_data['layers'].get(module_layer, {})
        for c, method in enumerate(methods):
            ax = axes[r][2 + c]
            
            path = layer_data.get(method, None)
            if path is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10, transform=ax.transAxes)
                continue
            
            im = _safe_open(path)
            if im is not None:
                if cell_size:
                    im = im.resize(cell_size)
                ax.imshow(im)
    
    label, image_name = key
    fig.suptitle(f"{task_name} | {module_layer}\nImage: {image_name} | Label: {label}", fontsize=12)
    
    plt.tight_layout(rect=[0.08, 0, 1, 0.93])
    # Save in <img_name> directory
    img_save_dir = save_dir / f"{label}_{image_name}"
    img_save_dir.mkdir(parents=True, exist_ok=True)
    out_path = img_save_dir / f"{module_layer}_comparison.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_comparison_grids(base_dir: Path = BASE_DIR, out_dir: Path = OUT_DIR,
                           selected_models=None, selected_layers=None):
    """Build comparison grids showing all models for each module-layer"""
    tasks = [p for p in base_dir.iterdir() if p.is_dir()]
    if not tasks:
        print(f"[WARN] 在 {base_dir} 底下沒有找到任何 task 資料夾。")
        return

    for task_dir in tasks:
        task_name = task_dir.name
        print(f"[Task] {task_name} → 掃描中…")

        index, models_set, methods_set, module_layers_set = _collect_index_per_model(task_dir)
        if not index:
            print(f"  - 找不到任何影像（{task_dir}）。略過。")
            continue

        # Filter and order
        if selected_models:
            models = [m for m in _order(models_set, FIXED_MODEL_ORDER) if m in selected_models]
        else:
            models = _order(models_set, FIXED_MODEL_ORDER)
        
        methods = _order(methods_set, FIXED_METHOD_ORDER)
        module_layers = _order_module_layers(module_layers_set)
        
        if selected_layers:
            module_layers = [ml for ml in module_layers if ml in selected_layers]

        print(f"  - 模型：{models}")
        print(f"  - 方法：{methods}")
        print(f"  - 模組層：{module_layers}")
        
        save_dir = out_dir / task_name / "comparison"

        for key in tqdm(sorted(index.keys()), desc=f"  產出 {task_name} comparison grids"):
            sample_data = index[key]
            
            for module_layer in module_layers:
                _ = _draw_comparison_grid(
                    task_name, key, sample_data, models, methods, module_layer, save_dir
                )

    print("✅ Comparison grids 完成！")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate grid visualizations for XAI heatmaps')
    parser.add_argument('--base_dir', type=str, default='./heatmap_results_production',
                        help='Base directory containing heatmap results')
    parser.add_argument('--out_dir', type=str, default='./grid_heatmap_production',
                        help='Output directory for grid images')
    parser.add_argument('--mode', type=str, choices=['per_model', 'comparison', 'both'], default='per_model',
                        help='Grid mode: per_model (one grid per model), comparison (compare models), or both')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Specific models to include (default: all)')
    parser.add_argument('--layers', type=str, nargs='+', default=None,
                        help='Specific module-layers to include for comparison mode (default: all)')
    
    args = parser.parse_args()
    
    BASE_DIR = Path(args.base_dir)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build mask index
    mask_index = {}
    for name, label, ms in zip(dme_img_names, dme_labels, dme_mask_slices):
        mask_index[(str(label), name)] = ms
    
    if args.mode in ['per_model', 'both']:
        print("\n" + "="*60)
        print("Generating per-model grids...")
        print("="*60)
        build_all_grids_per_model(BASE_DIR, OUT_DIR, mask_index=mask_index, 
                                   selected_models=args.models)
    
    if args.mode in ['comparison', 'both']:
        print("\n" + "="*60)
        print("Generating comparison grids...")
        print("="*60)
        build_comparison_grids(BASE_DIR, OUT_DIR / "comparison", 
                               selected_models=args.models, selected_layers=args.layers)
    
    print("\n✅ All grids generated!")
