import os
import argparse
import numpy as np
import cv2
import torch
from typing import List, Callable, Optional
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm

# Set environment variable to avoid symbolic tracing issues
os.environ['TIMM_FUSED_ATTN'] = '0'
from torchvision import transforms
from pytorch_grad_cam import run_dff_on_image, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Import XAI methods
from baselines.GradCAM_v2 import PytorchCAM
from baselines.RISE import RISEBatch
from baselines.Attention import Attention_Map
#from baselines.CRP_LXT import CRP_LXT
from pytorch_grad_cam import GradCAM as GradCAMv2, ScoreCAM, HiResCAM, GradCAMPlusPlus

from transformers import (
    ViTImageProcessor, ViTForImageClassification,
    AutoImageProcessor, EfficientNetForImageClassification,
    ResNetForImageClassification, AutoModel
)
import models_vit as models
from util.datasets import TransformWrapper
import timm
import wandb
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

#transform
def eval_transform(input_size):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    t = []
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

#get model
def get_model(task,model,input_size,nb_classes, param_dict={}):
    if 'ADCon' in task:
        id2label = {0: "control", 1: "ad"}
        label2id = {v: k for k, v in id2label.items()}
    else:
        id2label = {i: f"class_{i}" for i in range(nb_classes)}
        label2id = {v: k for k, v in id2label.items()}
    processor = None
    if 'SMP' in model:
        from SMP.smp_classifier import SMPClassifier, Config as SMPConfig
        if 'fuse' in model:
            mode = 'fuse'
        elif 'dec' in model:
            mode = 'dec'
        else:
            mode = 'enc'
        model = SMPClassifier(
            seg_arch=SMPConfig.SEG_ARCH,
            encoder_name=SMPConfig.ENCODER,
            encoder_weights=SMPConfig.ENCODER_WEIGHTS,
            in_channels=SMPConfig.IN_CHANNELS,
            num_classes=nb_classes,
            seg_classes=SMPConfig.SEG_CLASSES,
            seg_activation=SMPConfig.ACTIVATION,
            mode=mode,
            fuse_mode=param_dict.get('smp_fuse_mode'),
            fusion_dim= None if param_dict.get('fusion_dim',0)==0 else param_dict.get('fusion_dim'),
            learnable_alpha=param_dict.get('smp_learnable_alpha'),
            pretrained_seg_ckpt="/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth",
            alpha=0.5,
            dropout=SMPConfig.DROPOUT,
            size_match=param_dict.get('size_match'),
            use_mask=param_dict.get('seg_mask'),
            enc_idx=param_dict.get('enc_idx',-1),
            dec_idx=param_dict.get('dec_idx',-1),
        )
    return model, processor

# task and dataset
#Task_list = ['ADCon','DME']
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

# model
Model_root = "/orange/ruogu.fang/tienyuchang/RETfound_results"
Model_fname = "checkpoint-best.pth"
# NOTE:
# - The previous preset-based multi-model lists (Model_list/Model_param_dict_list/Model_image_size_list)
#   have been removed.
# - We now build the model from CLI args (main_XAI_evaluation.py style) and optionally load --resume.

Module_list = ['encoder', 'decoder', 'head']
# Visualization functions
def _build_binary_mask(mask_slice, heatmap):
    binary_mask = np.zeros((heatmap.shape[0],heatmap.shape[1]), dtype=np.uint8)
    for i in range(mask_slice.shape[0]-1):
        upper = mask_slice[i].astype(int)
        lower = mask_slice[i+1].astype(int)
        for x in range(heatmap.shape[1]):
            binary_mask[upper[x]:lower[x], x] = 1
    return binary_mask
#mask function
def masked_img_func(img, mask_slice):
    binary_mask = _build_binary_mask(mask_slice, img)

    # 套用 mask (把 mask=0 的地方設為 0)
    masked_img = img.copy()
    masked_img[binary_mask == 0] = 0

    return masked_img

# Data loading and preprocessing functions
def load_sample_data(task, num_sample=-1, save_mask=False):
    """Load sample images for a given task"""
    df = pd.read_csv(os.path.join(dataset_dir, "%s_sampled"%task, dataset_fname))
    if LOAD_MASK and 'Surface Name' not in df.columns:
        masked_df = pd.read_csv(Thickness_CSV)
        masked_df = masked_df.rename(columns={'OCT':'folder'}).dropna(subset=['Surface Name'])
        df = df.merge(masked_df,on='folder',how='inner').reset_index(drop=True)
        #print('After adding mask, data len: ', df.shape[0])
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
            img = Image.open(img_path).convert('RGB')
            if LOAD_MASK:
                try:
                    mask_path = os.path.join(Thickness_DIR, row['folder'], row['Surface Name'])
                    mask = np.load(mask_path) # (Layer, slice, W)
                    # 假設我們要套用其中某一 slice 的 mask，例如 slice_index = 13
                    slice_index = int(os.path.basename(img_path).split("_")[-1].split(".")[0])  # 從檔名抓 13
                    mask_slice = mask[:, slice_index, :]  # shape: (Layer, W)
                    mask_slices.append(mask_slice)
                except Exception as e:
                    mask_slices.append(None)
            else:
                mask_slices.append(None)
            
            img_np = np.array(img)  # Convert PIL image to numpy array
            if IMG_MASK:
                masked_img_np = masked_img_func(img_np, mask_slice)
                masked_img = Image.fromarray(masked_img_np)
                images.append(masked_img)
            else:
                images.append(img)
            
            if save_mask:
                binary_mask = _build_binary_mask(mask_slice, img_np)
                mask_path = img_path.replace('.jpg','.npy')
                layer_path = img_path.replace('.jpg','_layer.npy')
                np.save(str(mask_path), binary_mask)
                np.save(str(layer_path), mask_slice)
            
            
            labels.append(row['label'])
            # Store filename without extension for directory naming
            image_name = os.path.splitext(filename)[0]
            filenames.append(image_name)

    return images, labels, filenames, mask_slices

def preprocess_image(image, processor=None, input_size=224, device=None, dtype=torch.float32):
    assert isinstance(image, Image.Image), f"expect PIL.Image, got {type(image)}"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if processor is not None:
        # A) 先尝试“直接可调用”形式（多数 timm/torchvision transform）
        try:
            out = processor(image)
            if isinstance(out, torch.Tensor):
                x = out
                if x.ndim == 3:  # [C,H,W] -> [1,C,H,W]
                    x = x.unsqueeze(0)
                return x.to(device=device, dtype=dtype)
            if isinstance(out, dict) and "pixel_values" in out:
                x = out["pixel_values"]
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x)
                if x.ndim == 3:
                    x = x.unsqueeze(0)
                return x.to(device=device, dtype=dtype)
        except TypeError:
            pass

        # B) 再尝试 HuggingFace 风格（不使用 images= 关键字）
        try:
            out = processor(image, return_tensors="pt")
            if isinstance(out, dict) and "pixel_values" in out:
                x = out["pixel_values"]  # [1,3,H,W]
                return x.to(device=device, dtype=dtype)
            if isinstance(out, torch.Tensor):
                x = out
                if x.ndim == 3:
                    x = x.unsqueeze(0)
                return x.to(device=device, dtype=dtype)
        except TypeError:
            pass

        # C) 某些实现仅接受列表
        for attempt in (lambda: processor([image], return_tensors="pt"),
                        lambda: processor([image])):
            try:
                out = attempt()
                if isinstance(out, dict) and "pixel_values" in out:
                    x = out["pixel_values"]
                    if isinstance(x, np.ndarray):
                        x = torch.from_numpy(x)
                    return x.to(device=device, dtype=dtype)
                if isinstance(out, torch.Tensor):
                    x = out
                    if x.ndim == 3:
                        x = x.unsqueeze(0)
                    return x.to(device=device, dtype=dtype)
            except TypeError:
                pass

    # D) 回退：标准 ImageNet 预处理
    fallback = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),  # [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    x = fallback(image)            # [3,H,W]
    x = x.unsqueeze(0)             # [1,3,H,W]
    return x.to(device=device, dtype=dtype)


# Load trained models function
def load_trained_model(task, model_name, Model_fname, input_size=224, nb_classes=2, model_param_dict={}):
    """Load a trained model for a specific task"""
    model, processor = get_model(task, model_name, input_size, nb_classes, param_dict=model_param_dict)
    model_path = os.path.join(Model_root, Model_fname)
    print(f"Loading {model_name} model from: {model_path}")
    
    # Load finetuned model if specified (following main_XAI_evaluation.py pattern)
    if model_path and model_path != '':
        if os.path.exists(model_path):
            # Load checkpoint
            if model_path.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    model_path, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Extract model state dict
            if 'model' in checkpoint:
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
            
            # Load with strict=False to handle potential mismatches
            model.load_state_dict(checkpoint_model, strict=False)
            print(f"Resume checkpoint {model_path} for {model_name} on {task}")
        else:
            print(f"Model path not found: {model_path}")
            print(f"Using pretrained weights for {model_name} on {task}")
    else:
        raise ValueError(f"No checkpoint specified for {model_name} on {task}, using pretrained weights")
    
    model.eval()
    return model, processor

# XAI Methods Implementation
class XAIGenerator:
    def __init__(self, model, model_name, input_size=224, batch_size=10, target_module=None, select_index=-1):
        self.model = model
        self.model_name = model_name
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.xai_list = ['Attention', 'RISE', 'GradCAM', 'ScoreCAM', 'HiResCAM', 'GradCAMPlusPlus']
        self.target_module = target_module
        self.select_index = select_index
        self.batch_size = batch_size
        # Initialize XAI methods
        self.init_xai_methods()
    
    def get_model_specific_config(self):
        """Get model-specific configuration for XAI methods"""
        config = {
            'patch_size': 14,
            'gpu_batch': self.batch_size,
            'attention_layers': 12
        }
        
        # Model-specific configurations
        if 'resnet' in self.model_name.lower():
            config.update({
                'patch_size': 7,  # ResNet has different spatial resolution
                'gpu_batch': self.batch_size,  # ResNet can handle larger batches
            })
        elif 'efficientnet' in self.model_name.lower():
            config.update({
                'patch_size': 7,  # EfficientNet spatial resolution
                'gpu_batch': self.batch_size,
            })
        elif 'vit' in self.model_name.lower():
            config.update({
                'patch_size': 16,  # ViT patch size
                'gpu_batch': self.batch_size,
                'attention_layers': 12,  # Standard ViT-Base layers
            })
        elif 'retfound' in self.model_name.lower():
            config.update({
                'patch_size': 16,  # RETFound uses ViT architecture
                'gpu_batch': self.batch_size,
                'attention_layers': 12,
            })
        
        return config
    
    def init_xai_methods(self):
        """Initialize all XAI methods with model-specific configurations"""
        config = self.get_model_specific_config()
        
        # Attention Maps (only for ViT-based models)
        if 'vit' in self.model_name.lower() or 'retfound' in self.model_name.lower():
            self.attention = Attention_Map(
                self.model, 
                self.model_name, 
                self.input_size, 
                N=config['attention_layers'],
                use_rollout=True,
                print_layers=False  # Disable layer printing to avoid issues
            )
            #print(f"✓ Attention initialized for {self.model_name} (layers: {config['attention_layers']})")
        else:
            self.attention = None
            #print(f"⚠ Attention skipped for {self.model_name} (not a transformer model)")
        
        # GradCAM with model-specific config
        self.gradcam = PytorchCAM(
            self.model, 
            self.model_name, 
            self.input_size, 
            patch_size=config['patch_size'],
            method=GradCAM,
            target_module=self.target_module,
            select_index=self.select_index
        )
        #print(f"✓ GradCAM initialized for {self.model_name} (patch_size: {config['patch_size']})")
        
        # ScoreCAM with model-specific config
        self.scorecam = PytorchCAM(
            self.model, 
            self.model_name, 
            self.input_size, 
            patch_size=config['patch_size'],
            method=ScoreCAM,
            target_module=self.target_module,
            select_index=self.select_index
        )
        #print(f"✓ ScoreCAM initialized for {self.model_name} (patch_size: {config['patch_size']})")
        
        # HiResCAM with model-specific config
        self.hirescam = PytorchCAM(
            self.model, 
            self.model_name, 
            self.input_size, 
            patch_size=config['patch_size'],
            method=HiResCAM,
            target_module=self.target_module,
            select_index=self.select_index
        )
        #print(f"✓ HiResCAM initialized for {self.model_name} (patch_size: {config['patch_size']})")
        
        # GradCAMPlusPlus with model-specific config
        self.gardcamplusplus = PytorchCAM(
            self.model, 
            self.model_name, 
            self.input_size, 
            patch_size=config['patch_size'],
            method=GradCAMPlusPlus,
            target_module=self.target_module,
            select_index=self.select_index
        )
        #print(f"✓ GradCAMPlusPlus initialized for {self.model_name} (patch_size: {config['patch_size']})")
        
        # RISE with model-specific batch size
        # Reduce batch for memory-heavy models
        self.rise = None
        rise_batch = config['gpu_batch']
        self.rise = RISEBatch(
            self.model, 
            input_size=(self.input_size, self.input_size), 
            gpu_batch=rise_batch,
            N=10,
            n_class = 2
        )
        #print(f"✓ RISE initialized for {self.model_name} (gpu_batch: {rise_batch})")
        
        self.lrp = None  # Will implement if model supports it
    
    def generate_gradcam(self, image_tensor, target_class=None):
        """Generate GradCAM heatmap"""
        if self.gradcam is None:
            return None
        self.model.zero_grad(set_to_none=True)
        # Detach tensor to prevent gradient contamination from previous computations
        image_tensor = image_tensor.detach().to(self.device)
        if target_class is None:
            # Get predicted class
            with torch.no_grad():
                outputs = self.model(image_tensor)
                target_class = outputs.argmax(dim=1).reshape(-1)
        image_tensor.requires_grad = True
        heatmap = self.gradcam(image_tensor, target_class)
        if hasattr(self.gradcam, "remove_hooks"):
            self.gradcam.remove_hooks()
        image_tensor.requires_grad = False
        return heatmap if len(heatmap) > 0 else None
    
    def generate_scorecam(self, image_tensor, target_class=None):
        """Generate ScoreCAM heatmap"""
        if self.scorecam is None:
            return None
        self.model.zero_grad(set_to_none=True)
        # Detach tensor to prevent gradient contamination
        image_tensor = image_tensor.detach().to(self.device)
        if target_class is None:
            # Get predicted class
            with torch.no_grad():
                outputs = self.model(image_tensor)
                target_class = outputs.argmax(dim=1).reshape(-1)

        heatmap = self.scorecam(image_tensor, target_class)
        if hasattr(self.scorecam, "remove_hooks"):
            self.scorecam.remove_hooks()
        image_tensor.requires_grad = False
        return heatmap if len(heatmap) > 0 else None
    
    def generate_gardcamplusplus(self, image_tensor, target_class=None):
        """Generate gardcamplusplus heatmap"""
        if self.gardcamplusplus is None:
            return None
        self.model.zero_grad(set_to_none=True)
        # Detach tensor to prevent gradient contamination
        image_tensor = image_tensor.detach().to(self.device)
        image_tensor.requires_grad = True
        if target_class is None:
            # Get predicted class
            with torch.no_grad():
                outputs = self.model(image_tensor)
                target_class = outputs.argmax(dim=1).reshape(-1)

        heatmap = self.gardcamplusplus(image_tensor, target_class)
        if hasattr(self.gardcamplusplus, "remove_hooks"):
            self.gardcamplusplus.remove_hooks()
        image_tensor.requires_grad = False
        return heatmap if len(heatmap) > 0 else None
    
    def generate_hirescam(self, image_tensor, target_class=None):
        """Generate hirescam heatmap"""
        if self.hirescam is None:
            return None
        self.model.zero_grad(set_to_none=True)
        # Detach tensor to prevent gradient contamination
        image_tensor = image_tensor.detach().to(self.device)
        image_tensor.requires_grad = True
        if target_class is None:
            # Get predicted class
            with torch.no_grad():
                outputs = self.model(image_tensor)
                target_class = outputs.argmax(dim=1).reshape(-1)

        heatmap = self.hirescam(image_tensor, target_class)
        if hasattr(self.hirescam, "remove_hooks"):
            self.hirescam.remove_hooks()
        image_tensor.requires_grad = False
        return heatmap if len(heatmap) > 0 else None
    
    def generate_rise(self, image_tensor, target_class=None):
        """Generate RISE heatmap"""
        if self.rise is None:
            return None
        image_tensor = image_tensor.to(self.device)

        if target_class is None:
            # Get predicted class
            with torch.no_grad():
                outputs = self.model(image_tensor)
                target_class = outputs.argmax(dim=1).reshape(-1)

        heatmap = self.rise(image_tensor,[target_class])
        image_tensor.requires_grad = False
        return heatmap if heatmap is not None else None

    def generate_attention(self, image_tensor, target_class=None):
        """Generate Attention heatmap"""
        if self.attention is None:
            return None
        image_tensor = image_tensor.to(self.device)
        attention_map = self.attention(image_tensor)
        return attention_map if attention_map is not None else None
    
    def generate_all_heatmaps(self, image_tensor, target_class=None, xai_name=None):
        """Generate requested heatmaps for an image (safe, no hook interference)
        
        Args:
            image_tensor (tensor): [B,3,H,W]
            target_class (np.array): [B]
            xai_name: str or list of str, default is all xai methods
        Returns:
            heatmaps: dict of heatmaps, key is xai name, value is heatmap [B,H,W]
        """
        if xai_name is None:
            xai_names = self.xai_list
        elif isinstance(xai_name, (list, tuple, set)):
            xai_names = list(xai_name)
        else:
            xai_names = [xai_name]
        heatmaps = {}
        attention_map, rise_map, gradcam_map, scorecam_map = None, None, None, None
        hirescam_map, gardcamplusplus_map = None, None
        # Attention
        if 'Attention' in xai_names:
            attention_map = self.generate_attention(image_tensor)
        if attention_map is not None:
            heatmaps['Attention'] = attention_map

        # RISE (chunk-based forward, 不依賴 hooks)
        if 'RISE' in xai_names:
            rise_map = self.generate_rise(image_tensor, target_class)
        if rise_map is not None:
            heatmaps['RISE'] = rise_map

        # GradCAM（需要梯度 + hooks）
        if 'GradCAM' in xai_names:
            gradcam_map = self.generate_gradcam(image_tensor, target_class)
        if gradcam_map is not None:
            heatmaps['GradCAM'] = gradcam_map

        # ScoreCAM
        if 'ScoreCAM' in xai_names:
            scorecam_map = self.generate_scorecam(image_tensor, target_class)
        if scorecam_map is not None:
            heatmaps['ScoreCAM'] = scorecam_map
        #['Attention', 'RISE', 'GradCAM', 'ScoreCAM', 'HiResCAM', 'GradCAMPlusPlus']
        # GradCAMPlusPlus
        if 'GradCAMPlusPlus' in xai_names:
            gardcamplusplus_map = self.generate_gardcamplusplus(image_tensor, target_class)
        if gardcamplusplus_map is not None:
            heatmaps['GradCAMPlusPlus'] = gardcamplusplus_map
        
        # HiResCAM
        if 'HiResCAM' in xai_names:
            hirescam_map = self.generate_hirescam(image_tensor, target_class)
        if hirescam_map is not None:
            heatmaps['HiResCAM'] = hirescam_map

        return heatmaps

# (removed) legacy multi-model test snippet

def masked_heatmap_func(heatmap, mask_slice):
    if heatmap is None:
        return None
    binary_mask = _build_binary_mask(mask_slice, heatmap)

    # 套用 mask (把 mask=0 的地方設為 0)
    masked_heatmap = heatmap.copy()
    masked_heatmap[binary_mask == 0] = 0

    return masked_heatmap

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

def normalize_heatmap(heatmap):
    """Normalize heatmap to 0-1 range"""
    if heatmap is None:
        return None
    
    heatmap = np.array(heatmap)
    if heatmap.max() == heatmap.min():
        return np.zeros_like(heatmap)
    
    return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

def overlay_heatmap_on_image(image, heatmap, mask_slice = None, alpha=0.4, colormap='jet'):
    """Overlay heatmap on original image"""
    
    if heatmap is None:
        return np.array(image)
    
    # Normalize heatmap
    heatmap_norm = normalize_heatmap(heatmap)
    
    # Resize heatmap to match image size
    if isinstance(image, Image.Image):
        image_array = np.array(image)
        image_size = image.size
    else:
        image_array = image
        image_size = (image.shape[1], image.shape[0])
    heatmap_resized = cv2.resize(heatmap_norm, image_size)
    
    if HEATMAP_MASK and mask_slice is not None:
        heatmap_resized = masked_heatmap_func(heatmap_resized, mask_slice)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized)
    #print(heatmap_colored.shape)
    heatmap_colored = heatmap_colored[:, :, :3]  # Remove alpha channel
    
    # Normalize image
    image_norm = image_array.astype(np.float32) / 255.0
    
    # Overlay
    overlay = alpha * heatmap_colored + (1 - alpha) * image_norm
    overlay = np.clip(overlay, 0, 1)
    
    return (overlay * 255).astype(np.uint8), heatmap_resized

from util.evaluation import RelevanceMetric
rel_metric = RelevanceMetric(pooling_type='sum,abs', output_type='mass')
rank_metric = RelevanceMetric(pooling_type='sum,abs', output_type='rank')

def _model_run_name(args) -> str:
    """Create a readable model name for output folders."""
    base = str(getattr(args, "model", "model"))
    if base == "SMP":
        mode = str(getattr(args, "SMPMode", "dec"))
        if mode == "fuse":
            return (
                f"SMP-{mode}-"
                f"{getattr(args,'smp_fuse_mode','weighted_sum')}-"
                f"align{getattr(args,'align','pre')}-"
                f"fus{int(getattr(args,'fusion_dim',0))}-"
                f"fea{int(getattr(args,'enc_idx',-1))}{int(getattr(args,'dec_idx',-1))}-"
                f"alpha{float(getattr(args,'smp_alpha',0.5))}-"
                f"{getattr(args,'smp_size_match','decoder_to_encoder')}-"
                f"{getattr(args,'smp_classifier','linear')}"
            )
        return (
            f"SMP-{mode}-"
            f"fea{int(getattr(args,'enc_idx',-1))}{int(getattr(args,'dec_idx',-1))}-"
            f"{getattr(args,'smp_classifier','linear')}"
        )
    else:
        pass
    return f"{args.task}-{base}"

def build_and_load_model_main_style(args):
    """Build model like main_XAI_evaluation.py and load --resume checkpoint if provided."""
    from main_XAI_evaluation import get_model as main_get_model

    device = torch.device(getattr(args, "device", "cuda"))
    model, processor, patch_size = main_get_model(args)
    print('Model:', model)
    # Load finetuned model checkpoint (main_XAI_evaluation.py pattern)
    resume = getattr(args, "resume", "0")
    if resume and str(resume).strip() and str(resume).strip() != "0":
        resume = str(resume).strip()
        if resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(resume, map_location='cpu', check_hash=True)
        else:
            with torch.serialization.safe_globals([argparse.Namespace]):
                checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
        checkpoint_model = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        model.load_state_dict(checkpoint_model, strict=True)
        print(f"Resume checkpoint {resume}")
    else:
        print('No checkpoint to resume')
        raise ValueError('No checkpoint to resume')

    model.to(torch.float32)
    model.to(device)
    model.eval()
    return model, processor, patch_size

module_select_dict_default = {'encoder':[10, 23, 42, 52],'decoder':[1,3,5,7,9]}

def get_args_parser():
    """Create argument parser (aligned to main_XAI_evaluation.py flags)"""
    parser = argparse.ArgumentParser(description='Generate XAI heatmaps with layer-wise analysis (main-style model build/load)')

    # ---- main_XAI_evaluation.py compatible core flags ----
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size. Here: XAI batch size per image.')
    parser.add_argument('--model', default='SMP', type=str, help='Model name (main-style). Use "SMP" for SMPClassifier.')
    parser.add_argument('--finetune', default='', type=str, help='Finetune from checkpoint (main-style). For SMP: pretrained seg ckpt.')
    parser.add_argument('--task', default='DME', type=str, help='Task name(s). Supports comma/space separated list.')
    parser.add_argument('--theme', default='', type=str, help='Theme name for wandb')
    parser.add_argument('--use_split', default='test', type=str, choices=['test', 'val', 'train'], help='(unused) main-style.')
    parser.add_argument('--input_size', default=512, type=int, help='Input image size.')
    parser.add_argument('--xai', default='gradcam', type=str, help='XAI method (main-style: attn/rise/gradcam/hirescam/scorecam/gradcam++/all).')
    parser.add_argument('--use_rollout', action='store_true', help='Use rollout (main-style).')
    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT', help='(unused) main-style.')
    parser.add_argument('--SMPMode', type=str, default='dec', help='SMP mode (fuse, enc, dec).')
    
    # fine-tuning parameters
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)

    # ---- main-style checkpoint/device ----
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--resume', default='0', type=str, help='Resume from checkpoint (classifier checkpoint).')
    parser.add_argument('--eval', action='store_true', help='(unused) main-style.')

    # ---- main-style SMP knobs ----
    parser.add_argument('--smp_fuse_mode', type=str, default='weighted_sum',
                        choices=["weighted_sum", "add", "channel_merge", "channel_multiply", "multiply"])
    parser.add_argument('--smp_learnable_alpha', action='store_true', default=False)
    parser.add_argument('--smp_alpha', type=float, default=0.5)
    parser.add_argument('--smp_size_match', type=str, default='decoder_to_encoder',
                        choices=["decoder_to_encoder", "encoder_to_decoder"])
    parser.add_argument('--seg_mask', action='store_true', default=False)
    parser.add_argument('--mask_softmax', action='store_true', default=False,
                        help='Softmax the segmentation mask output from SMP model')
    parser.add_argument('--ignore_background', action='store_true', default=False,
                        help='Ignore the background class in segmentation mask output from SMP model')
    parser.add_argument('--fusion_dim', type=int, default=0)
    parser.add_argument('--align', type=str, default='pre')
    parser.add_argument('--enc_idx', type=int, default=-1)
    parser.add_argument('--dec_idx', type=int, default=-1)
    parser.add_argument('--smp_classifier', type=str, default='linear', choices=["linear", "conv"])

    # ---- CAM selection (main-style) ----
    parser.add_argument('--target_module', type=str, nargs='+', default=['encoder', 'decoder', 'head'],
                        help='Target module(s): encoder/decoder/head/all')
    parser.add_argument('--select_index', type=int, default=-1, help='Select index for CAM methods (single index).')

    # ---- this script: dataset + sampling ----
    parser.add_argument('--dataset_dir', type=str, default=dataset_dir, help='Root directory for dataset.')
    parser.add_argument('--dataset_fname', type=str, default=dataset_fname, help='Dataset CSV filename.')
    parser.add_argument('--thickness_dir', type=str, default=Thickness_DIR, help='Directory for thickness mask data.')
    parser.add_argument('--thickness_csv', type=str, default=Thickness_CSV, help='CSV file for thickness mapping.')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to process. -1 for all.')
    parser.add_argument('--nb_classes', type=int, default=2, help='Number of classes for classification.')

    # ---- layer selection helpers ----
    parser.add_argument('--select_idx', type=int, nargs='+', default=None, help='Layer indices to analyze (list).')
    parser.add_argument('--encoder_idx', type=int, nargs='+', default=None, help='Encoder indices override.')
    parser.add_argument('--decoder_idx', type=int, nargs='+', default=None, help='Decoder indices override.')
    parser.add_argument('--choose_last_layer', action='store_true', default=False, help='Only analyze last layer.')
    parser.add_argument('--all_layers', action='store_true', default=False, help='Analyze all layers.')

    # ---- output + masks ----
    parser.add_argument('--output_dir', type=str, default='./heatmap_results_layerwise', help='Output directory.')
    parser.add_argument('--load_mask', action='store_true', default=True, help='Load thickness mask for images.')
    parser.add_argument('--no_load_mask', action='store_false', dest='load_mask', help='Do not load thickness mask.')
    parser.add_argument('--img_mask', action='store_true', default=False, help='Apply mask to input images.')
    parser.add_argument('--heatmap_mask', action='store_true', default=False, help='Apply mask to heatmaps.')
    parser.add_argument('--draw_layer', action='store_true', default=True, help='Draw layer boundaries on overlay.')
    parser.add_argument('--no_draw_layer', action='store_false', dest='draw_layer', help='Do not draw layer boundaries.')
    parser.add_argument('--save_mask', action='store_true', default=False, help='Save binary mask as numpy file.')
    parser.add_argument('--verbose', action='store_true', default=False)

    # ---- keep list_xai only ----
    parser.add_argument('--list_xai', action='store_true', default=False, help='List available XAI methods and exit.')
    return parser

def parse_args():
    """Parse command-line arguments"""
    parser = get_args_parser()
    args = parser.parse_args()

    # task: main-style string -> task_list; keep args.task as first task (main expects a string)
    task_list = [t for t in str(getattr(args, "task", "")).replace(",", " ").split() if t]
    if not task_list:
        task_list = ['DME']
    args.task_list = task_list
    args.task = task_list[0]

    # xai: main-style string -> xai_method_list (this script uses the full names)
    xai_parts = [p for p in str(getattr(args, "xai", "")).replace(",", " ").split() if p]
    mapping = {
        "attn": "Attention",
        "attention": "Attention",
        "rise": "RISE",
        "gradcam": "GradCAM",
        "gradcamv2": "GradCAM",
        "scorecam": "ScoreCAM",
        "hirescam": "HiResCAM",
        "gradcam++": "GradCAMPlusPlus",
        "gradcamplusplus": "GradCAMPlusPlus",
        "all": "all",
    }
    xai_method_list: List[str] = []
    for p in (xai_parts or ["gradcam"]):
        key = p.strip().lower()
        if key == "all":
            xai_method_list = ['Attention', 'RISE', 'GradCAM', 'ScoreCAM', 'HiResCAM', 'GradCAMPlusPlus']
            break
        if key not in mapping:
            parser.error(f"Unknown --xai '{p}'. Supported: {sorted([k for k in mapping.keys() if k != 'all'])} or 'all'.")
        xai_method_list.append(mapping[key])
    args.xai_method_list = xai_method_list

    # Normalize target_module to support quoted/space/comma separated values
    allowed_modules = {'encoder', 'decoder', 'head'}
    normalized_target_modules: List[str] = []
    for item in args.target_module or []:
        parts = str(item).replace(',', ' ').split()
        normalized_target_modules.extend([p for p in parts if p])
    if not normalized_target_modules:
        normalized_target_modules = ['encoder', 'decoder', 'head']
    if 'all' in normalized_target_modules:
        normalized_target_modules = ['encoder', 'decoder', 'head']
    invalid = [m for m in normalized_target_modules if m not in allowed_modules]
    if invalid:
        parser.error(f"--target_module accepts {sorted(allowed_modules)}, got {invalid}")
    args.target_module = normalized_target_modules

    # Map main-style --select_index (single) into list selection if user didn't provide list args.
    if args.select_idx is None and getattr(args, "select_index", None) is not None:
        args.select_idx = [int(args.select_index)]
    print('Parsed arguments:', args)
    return args

def build_module_select_dict(args):
    """Build module_select_dict based on arguments"""
    module_select_dict = {}
    
    if args.all_layers:
        # Return empty dict to trigger all layers mode
        return {}
    
    if args.encoder_idx is not None:
        module_select_dict['encoder'] = args.encoder_idx
    elif args.select_idx is not None:
        module_select_dict['encoder'] = args.select_idx
    else:
        module_select_dict['encoder'] = module_select_dict_default.get('encoder', [-1])
    
    if args.decoder_idx is not None:
        module_select_dict['decoder'] = args.decoder_idx
    elif args.select_idx is not None:
        module_select_dict['decoder'] = args.select_idx
    else:
        module_select_dict['decoder'] = module_select_dict_default.get('decoder', [-1])
    
    if args.select_idx is not None:
        module_select_dict['head'] = args.select_idx
    
    return module_select_dict

def _get(obj, name, default=None):
    return getattr(obj, name, default)
def get_all_conv_layers(model, module_name=None):
    """Get all Conv layers from the model"""
    seg_model = _get(model, "seg_model")
    encoder = _get(seg_model, "encoder")
    decoder = _get(seg_model, "decoder")
    head = _get(model, "head")
    conv_layers = []
    if module_name is None:
        return []
    elif hasattr(model, "mode") and (model.mode == 'fuse'  or model.mode == 'enc') and module_name=='encoder':
        module = encoder
    elif hasattr(model, "mode") and (model.mode == 'fuse' or model.mode == 'dec') and module_name=='decoder':
        module = decoder
    elif module_name=='head':
        module = head
    else:
        return []
    
    for name, module in module.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append(module)
    return conv_layers
# Updated function with new directory structure for heatmap saving and batch processing
def generate_comprehensive_heatmaps_v2(
    model,
    processor,
    model_name: str,
    input_size: int,
    num_samples=3,
    task_list=[],
    XAI_list=None,
    heatmap_dir="./heatmap_results",
    module_list=None,
    module_select_dict=None,
    choose_last_layer=True,
    batch_size=4,
    verbose=False,
    load_mask_flag=True,
    img_mask_flag=False,
    heatmap_mask_flag=False,
    draw_layer_flag=True,
    save_mask_flag=False,
    nb_classes=2,
):
    """Generate heatmaps for (tasks × modules × layers × XAI) for a single built model.

    This function assumes the caller already built the model (main_XAI_evaluation.py style)
    and loaded checkpoints if needed, then passes `model` and `processor` in.
    """
    global LOAD_MASK, IMG_MASK, HEATMAP_MASK, DRAW_LAYER
    
    # Set global flags based on arguments
    LOAD_MASK = load_mask_flag
    IMG_MASK = img_mask_flag
    HEATMAP_MASK = heatmap_mask_flag
    DRAW_LAYER = draw_layer_flag
    
    results = {}
    if XAI_list is None:
        XAI_list = ['Attention', 'RISE', 'GradCAM', 'ScoreCAM', 'HiResCAM', 'GradCAMPlusPlus']
    if module_select_dict is None:
        module_select_dict = {}
    
    print("=" * 60)
    print("Starting comprehensive heatmap generation...")
    print("=" * 60)
    print(f"Tasks: {task_list}")
    print(f"Model: {model_name}")
    print(f"Modules: {module_list}")
    print(f"XAI Methods: {XAI_list}")
    print(f"Samples per task: {num_samples if num_samples > 0 else 'all'}")
    print(f"XAI batch size: {batch_size}")
    print(f"Output directory: {heatmap_dir}")
    print("=" * 60)
    
    if module_list is None:
        module_list = [None]
    
    for task in task_list:
        print(f"\n{'='*20} Processing Task: {task} {'='*20}")
        results[task] = {}
        
        # Load sample data for this task (now returns filenames too)
        images, labels, filenames, mask_slices = load_sample_data(task, num_samples, save_mask=save_mask_flag)
        print(f"Loaded {len(images)} images for {task}")
        
        out_df = []
        # One model per run (built outside like main_XAI_evaluation.py)
        for module_name in module_list:
            # Skip modules that don't exist for the current model/mode
            conv_layers = get_all_conv_layers(model, module_name)
            if model_name=='SMP' and not conv_layers:
                if verbose:
                    print(f"  Skipping module '{module_name}' (no conv layers found)")
                continue

            # Get all Conv layers
            if module_select_dict.get(module_name, False):
                select_indexs = module_select_dict[module_name]
            elif choose_last_layer:
                select_indexs = [-1]
            else:
                select_indexs = list(range(len(conv_layers)))

            if verbose:
                print(f'  Module: {module_name}, Select Indices: {select_indexs}')

            for select_index in select_indexs:
                # Initialize XAI generator
                xai_generator = XAIGenerator(
                    model,
                    model_name,
                    input_size,
                    batch_size=batch_size,
                    target_module=module_name,
                    select_index=select_index,
                )

                results[task][model_name] = {
                    'images': filenames,
                    'labels': labels,
                    "mask_slices": mask_slices,
                    'module_name': module_name,
                    'select_index': select_index,
                }

                image_tensor_list, image_list, label_list, filename_list, mask_slice_list = [], [], [], [], []
                mass_acc, rank_acc = None, None
                for idx, (image, label, filename, mask_slice) in enumerate(zip(images, labels, filenames, mask_slices)):
                    print(f"Processing image {idx+1}/{len(images)} (Label: {label}, File: {filename})")
                    image_tensor = preprocess_image(image, processor, input_size) # [1,3,H,W]
                    image_list.append(image)
                    image_tensor_list.append(image_tensor)
                    label_list.append(label)
                    filename_list.append(filename)
                    mask_slice_list.append(mask_slice)

                    if idx % batch_size == batch_size - 1 or idx == len(images) - 1:
                        #first assert the length are the same
                        assert len(image_tensor_list) == len(image_list) == len(label_list) == len(filename_list) == len(mask_slice_list)
                        image_tensors = torch.concat(image_tensor_list) # [B,3,H,W]
                        labels_np = np.stack(label_list)
                        for xai_name in XAI_list:
                            heatmap_dict = xai_generator.generate_all_heatmaps(
                                image_tensors, target_class=labels_np, xai_name=xai_name
                            )
                            heatmaps = heatmap_dict.get(xai_name, None)
                            if heatmaps is None:
                                continue
                            for b_idx in range(len(heatmaps)):
                                if verbose:
                                    print(f"Processing heatmap {b_idx+1}/{len(heatmaps)} (Label: {label_list[b_idx]}, File: {filename_list[b_idx]})")
                                    print(f"Heatmap shape: {heatmaps[b_idx].shape}")
                                    print(f"Image shape: {image_list[b_idx].shape}")
                                    print(f"Label: {label_list[b_idx]}")
                                    print(f"Filename: {filename_list[b_idx]}")
                                    print(f"Mask slice: {mask_slice_list[b_idx]}")
                                heatmap = heatmaps[b_idx]
                                image_b = image_list[b_idx]
                                label_b = label_list[b_idx]
                                filename_b = filename_list[b_idx]
                                mask_slice_b = mask_slice_list[b_idx]
                                if heatmap is None:
                                    continue
                                heatmap = heatmap + 1e-9 # add small value to avoid all zeros
                                overlay, heatmap_resized = overlay_heatmap_on_image(image_b, heatmap, mask_slice_b)
                                binary_mask = _build_binary_mask(mask_slice_b, overlay) if mask_slice_b is not None else None
                                if binary_mask is not None:
                                    mass_acc = rel_metric(image_b, heatmap_resized, binary_mask)
                                    rank_acc = rank_metric(image_b, heatmap_resized, binary_mask)
                                if DRAW_LAYER and mask_slice_b is not None:
                                    overlay = add_layer_line(overlay, mask_slice_b)

                                module_idx = f'{module_name}_{select_index}' if module_name is not None else f'all_{select_index}'
                                img_dir = Path(heatmap_dir) / task / str(label_b) / filename_b / model_name
                                save_dir = img_dir / module_idx
                                save_dir.mkdir(parents=True, exist_ok=True)
                                out_path = save_dir / f"{xai_name}.jpg"
                                try:
                                    if not isinstance(overlay, Image.Image):
                                        overlay = Image.fromarray(overlay)
                                    overlay.save(out_path, format='JPEG', quality=95)
                                    np.save(save_dir / f"{xai_name}.npy", heatmap)
                                    image_b.save(img_dir / f"{xai_name}_image.jpg")
                                    if binary_mask is not None:
                                        plt.imshow(binary_mask, cmap='gray')
                                        plt.savefig(img_dir / f"{xai_name}_mask.jpg")
                                        plt.close()
                                except Exception as e:
                                    print(f"Failed to save {out_path}: {e}")
                                out_df.append({
                                    'task': task,
                                    'image_name': filename_b,
                                    'label': label_b,
                                    'output_path': str(out_path),
                                    'model_name': model_name,
                                    'xai_method': xai_name,
                                    'relevance_mass_accuracy': mass_acc,
                                    'relevance_rank_accuracy': rank_acc,
                                })
                        image_tensor_list, image_list, label_list, filename_list, mask_slice_list = [], [], [], [], []

        # Save out_df to CSV
        df = pd.DataFrame(out_df)
        df.to_csv(Path(heatmap_dir) / f"{task}_results.csv", index=False)
    return results

def list_available_xai():
    """Print available XAI methods"""
    print("\nAvailable XAI Methods:")
    print("-" * 40)
    xai_methods = ['Attention', 'RISE', 'GradCAM', 'ScoreCAM', 'HiResCAM', 'GradCAMPlusPlus']
    for method in xai_methods:
        print(f"  - {method}")
    print("-" * 40)

def main():
    """Main function with argument parsing"""
    args = parse_args()
    
    if args.list_xai:
        list_available_xai()
        return
    
    # Build up the common task name
    args.theme = _model_run_name(args)
    
    project_name = "RETFound_MAE_XAI_Case_Study"
    group_name = None
    model_add_dir = ""
    wandb.init(
        project=project_name,
        name=args.theme,
        group=group_name,
        config=args,
    )
    # Determine target modules
    target_modules = args.target_module if args.target_module else Module_list
    print('Target modules:', target_modules)
    # Build module select dict
    module_select_dict = build_module_select_dict(args)
    print('Module select dict:', module_select_dict)
    
    # Update global paths based on arguments
    global dataset_dir, dataset_fname, Thickness_DIR, Thickness_CSV, Model_root, Model_fname
    dataset_dir = args.dataset_dir
    dataset_fname = args.dataset_fname
    Thickness_DIR = args.thickness_dir
    Thickness_CSV = args.thickness_csv
    
    # Build model like main_XAI_evaluation.py and load --resume
    model_main, processor, patch_size = build_and_load_model_main_style(args)

    # Run heatmap generation (single model per run)
    heatmap_results = generate_comprehensive_heatmaps_v2(
        model=model_main,
        processor=processor,
        model_name=args.theme,
        input_size=int(args.input_size),
        num_samples=args.num_samples,
        task_list=args.task_list,
        XAI_list=args.xai_method_list,
        heatmap_dir=args.output_dir,
        module_list=target_modules,
        module_select_dict=module_select_dict,
        choose_last_layer=args.choose_last_layer,
        batch_size=args.batch_size,
        verbose=args.verbose,
        load_mask_flag=args.load_mask,
        img_mask_flag=args.img_mask,
        heatmap_mask_flag=args.heatmap_mask,
        draw_layer_flag=args.draw_layer,
        save_mask_flag=args.save_mask,
        nb_classes=args.nb_classes
    )
    
    print("\n" + "=" * 60)
    print("Heatmap generation completed!")
    print("=" * 60)
    wandb.finish()
    return heatmap_results


if __name__ == "__main__":
    main()
