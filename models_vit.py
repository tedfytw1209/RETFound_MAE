# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torchvision_models
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
from torch import Tensor

###!!! hard to change https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L676
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    #TODO: Decide on attention mask handling (timm==0.9.16 or timm==1.0.9)
    def forward_features(self, x, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1,keepdim=True)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

class DualViTClassifier(nn.Module):
    def __init__(self, vit_model_1, vit_model_2, num_classes):
        """
        初始化 DualViTClassifier 模型。

        Args:
            vit_model_1: 第一个 ViT
            vit_model_2: 第二个 ViT
            num_classes (int): 分类的类别数。
        """
        super(DualViTClassifier, self).__init__()
        self.num_classes = num_classes
        self.vit_model_1 = vit_model_1
        self.vit_model_2 = vit_model_2

    def forward(self, input_1, input_2):
        """
        前向传播。

        Args:
            input_1 (torch.Tensor): 输入到第一个 ViT 的数据 (batch_size, 3, H, W)。
            input_2 (torch.Tensor): 输入到第二个 ViT 的数据 (batch_size, 3, D, H, W)。

        Returns:
            torch.Tensor: 分类结果 (batch_size, num_classes)。
        """
        # 第一个 ViT 提取特征 (batch_size, num_classes)
        features_1 = self.vit_model_1(input_1)
        # 第二个 ViT 提取特征 (batch_size, num_classes)
        features_2 = self.vit_model_2(input_2)

        # output mean (batch_size, num_classes)
        combined_features = (features_1 + features_2) / 2

        return combined_features

class DinoV3Classifier(nn.Module):
    def __init__(self, backbone, num_labels):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size  # dinov3-vit-large = 1024
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, pixel_values=None, labels=None, **kwargs):
        out = self.backbone(pixel_values=pixel_values, **kwargs)  # BaseModelOutputWithPooling
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        return logits

def DualViT(**kwargs):
    model = DualViTClassifier(**kwargs)
    return model

def RETFound_mae(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def RETFound_dinov2(**kwargs):
    model = timm.create_model(
        'vit_large_patch14_dinov2.lvd142m',
        pretrained=True,
        **kwargs
    )
    return model

class DualCNN(nn.Module):
    """
    Flexible CNN model for MCI classification based on the paper description.
    
    The model can receive different combinations of inputs:
    1. GC-IPL thickness color maps (image input 1)
    2. OCTA SCP 6x6 mm en face images (image input 2) 
    3. Quantitative data (tabular input)
    
    Input modes:
    - 'all': Use all three inputs (GC-IPL + OCTA + quantitative)
    - 'images_only': Use only image inputs (GC-IPL + OCTA)
    - 'gc_ipl_only': Use only GC-IPL images
    - 'octa_only': Use only OCTA images
    - 'quantitative_only': Use only quantitative data
    - 'gc_ipl_quantitative': Use GC-IPL + quantitative data
    - 'octa_quantitative': Use OCTA + quantitative data
    
    Architecture:
    - Shared ResNet18 convolutional encoder for image inputs
    - Modality-specific feature transformations (fOCTA and fGC-IPL)
    - Prediction heads for all modalities (FCOCTA, FCGC-IPL, FCother)
    - Aggregation using simple averages + sigmoid activation
    - Separate dropout layers for image and quantitative features
    """
    def __init__(self, num_classes=1, quantitative_features=10, dropout_rate=0.5, 
                 pretrained=True, input_mode='all'):
        super(DualCNN, self).__init__()
        
        self.num_classes = num_classes
        self.quantitative_features = quantitative_features
        self.input_mode = input_mode
        
        # Validate input mode
        valid_modes = ['all', 'images_only', 'gc_ipl_only', 'octa_only', 
                      'quantitative_only', 'gc_ipl_quantitative', 'octa_quantitative']
        if input_mode not in valid_modes:
            raise ValueError(f"Invalid input_mode '{input_mode}'. Must be one of: {valid_modes}")
        
        # Determine which inputs are needed
        self.use_gc_ipl = input_mode in ['all', 'images_only', 'gc_ipl_only', 'gc_ipl_quantitative']
        self.use_octa = input_mode in ['all', 'images_only', 'octa_only', 'octa_quantitative']
        self.use_quantitative = input_mode in ['all', 'quantitative_only', 'gc_ipl_quantitative', 'octa_quantitative']
        
        # Shared ResNet18 encoder (only if using image inputs)
        if self.use_gc_ipl or self.use_octa:
            self.shared_encoder = torchvision_models.resnet18(pretrained=pretrained)
            # Remove the final classification layer
            self.shared_encoder = nn.Sequential(*list(self.shared_encoder.children())[:-1])
            
            # Get the output feature dimension from ResNet18 (512 for ResNet18)
            encoder_output_dim = 512
            
            # Modality-specific feature transformations (single fully connected layers)
            if self.use_gc_ipl:
                self.fGC_IPL = nn.Linear(encoder_output_dim, encoder_output_dim)  # fGC-IPL
                self.FCGC_IPL = nn.Linear(encoder_output_dim, num_classes)  # FCGC-IPL
            
            if self.use_octa:
                self.fOCTA = nn.Linear(encoder_output_dim, encoder_output_dim)  # fOCTA
                self.FCOCTA = nn.Linear(encoder_output_dim, num_classes)  # FCOCTA
            
            # Dropout layer for image features
            self.image_dropout = nn.Dropout(dropout_rate)
        
        # Quantitative data processing (only if using quantitative inputs)
        if self.use_quantitative:
            self.FCother = nn.Linear(quantitative_features, num_classes)  # FCother
            self.quantitative_dropout = nn.Dropout(dropout_rate)
        
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, gc_ipl_image=None, octa_image=None, quantitative_data=None):
        """
        Forward pass of the flexible CNN model.
        
        Args:
            gc_ipl_image (torch.Tensor, optional): GC-IPL thickness color maps (batch_size, 3, H, W)
            octa_image (torch.Tensor, optional): OCTA SCP en face images (batch_size, 3, H, W)
            quantitative_data (torch.Tensor, optional): Quantitative features (batch_size, quantitative_features)
        
        Returns:
            torch.Tensor: Probability scores for MCI classification (batch_size, num_classes)
        """
        predictions = []
        
        # Process GC-IPL image if needed
        if self.use_gc_ipl:
            if gc_ipl_image is None:
                raise ValueError("GC-IPL image is required for this input mode")
            
            # Process GC-IPL image through shared encoder
            gc_ipl_features = self.shared_encoder(gc_ipl_image)  # (batch_size, 512, 1, 1)
            gc_ipl_features = gc_ipl_features.view(gc_ipl_features.size(0), -1)  # (batch_size, 512)
            
            # Apply modality-specific feature transformation
            gc_ipl_transformed = self.fGC_IPL(gc_ipl_features)  # (batch_size, 512)
            
            # Apply image dropout
            gc_ipl_transformed = self.image_dropout(gc_ipl_transformed)
            
            # Get prediction from GC-IPL modality
            gc_ipl_pred = self.FCGC_IPL(gc_ipl_transformed)  # (batch_size, num_classes)
            predictions.append(gc_ipl_pred)
        
        # Process OCTA image if needed
        if self.use_octa:
            if octa_image is None:
                raise ValueError("OCTA image is required for this input mode")
            
            # Process OCTA image through shared encoder
            octa_features = self.shared_encoder(octa_image)  # (batch_size, 512, 1, 1)
            octa_features = octa_features.view(octa_features.size(0), -1)  # (batch_size, 512)
            
            # Apply modality-specific feature transformation
            octa_transformed = self.fOCTA(octa_features)  # (batch_size, 512)
            
            # Apply image dropout
            octa_transformed = self.image_dropout(octa_transformed)
            
            # Get prediction from OCTA modality
            octa_pred = self.FCOCTA(octa_transformed)  # (batch_size, num_classes)
            predictions.append(octa_pred)
        
        # Process quantitative data if needed
        if self.use_quantitative:
            if quantitative_data is None:
                raise ValueError("Quantitative data is required for this input mode")
            
            # Apply quantitative dropout
            quantitative_data = self.quantitative_dropout(quantitative_data)
            
            # Get prediction from quantitative modality
            quantitative_pred = self.FCother(quantitative_data)  # (batch_size, num_classes)
            predictions.append(quantitative_pred)
        
        # Aggregate predictions using simple averages
        if len(predictions) == 1:
            combined_pred = predictions[0]
        else:
            combined_pred = torch.stack(predictions, dim=0).mean(dim=0)
        
        # Apply sigmoid activation to produce probability scores
        output = self.sigmoid(combined_pred)
        
        return output

def DualInputCNN(**kwargs):
    model = DualCNN(**kwargs)
    return model

class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for image classification.
    Basic architecture with convolutional, pooling, and fully connected layers.
    """
    def __init__(self, num_classes=1, input_channels=3, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Activation function
        self.relu = nn.ReLU(inplace=True)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        # Assuming input size 224x224, after 4 pooling layers: 224/16 = 14x14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Convolutional layers with pooling
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 224x224 -> 112x112
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 112x112 -> 56x56
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # 56x56 -> 28x28
        x = self.pool(self.relu(self.bn4(self.conv4(x))))  # 28x28 -> 14x14
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class TransNetOCT(nn.Module):
    """
    TransNetOCT: Transformer-based network for OCT classification.
    Combines CNN feature extraction with transformer attention mechanisms.
    """
    def __init__(self, num_classes=1, input_channels=3, img_size=224, patch_size=16, 
                 embed_dim=768, num_heads=12, num_layers=12, dropout_rate=0.1):
        super(TransNetOCT, self).__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # CNN feature extractor (backbone)
        self.cnn_backbone = torchvision_models.resnet18(pretrained=True)
        # Remove the final layers
        self.cnn_backbone = nn.Sequential(*list(self.cnn_backbone.children())[:-2])
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(512, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]
        
        # CNN feature extraction
        x = self.cnn_backbone(x)  # (B, 512, H/32, W/32)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Classification
        x = self.norm(x[:, 0])  # Use class token
        x = self.head(x)
        
        return x

def SimpleCNN(**kwargs):
    model = SimpleCNN(**kwargs)
    return model

def TransNetOCT(**kwargs):
    model = TransNetOCT(**kwargs)
    return model

# Import AD-OCT models
try:
    from models_ad_oct import ADOCTModel as ADOCTModelClass, create_ad_oct_model, create_ad_oct_loss
    
    def ADOCTModel(**kwargs):
        """AD-OCT Model wrapper for compatibility with existing framework."""
        return create_ad_oct_model(**kwargs)
        
except ImportError:
    print("Warning: models_ad_oct module not found. ADOCTModel will not be available.")
    
    def ADOCTModel(**kwargs):
        """Placeholder for AD-OCT Model when module is not available."""
        raise ImportError("models_ad_oct module is required for ADOCTModel")


