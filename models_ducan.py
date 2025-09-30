"""
DuCAN (Dual Cross-Attention Network) Implementation
End-to-end framework for MCI detection based on OCT images and fundus photographs.

This module implements:
1. Modality-Feature-Extraction (MFE) modules
2. Cross-Modal-Fusion (CMF) units with self-attention and cross-attention
3. Position Attention Module (PAM) and Channel Attention Module (CAM)
4. Multi-modal classification with three classifiers

Based on the research paper methodology for dual-modal MCI detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class PositionAttentionModule(nn.Module):
    """
    Position Attention Module (PAM)
    
    Enhances discriminant abilities of feature representation to detect
    the most salient regions of objects received through spatial attention.
    """
    
    def __init__(self, in_channels: int):
        """
        Initialize PAM module.
        
        Args:
            in_channels: Number of input channels
        """
        super(PositionAttentionModule, self).__init__()
        
        self.in_channels = in_channels
        
        # Three convolutions to generate query, key, and value matrices
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=-1)
        
        # Learnable parameter for residual connection
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Forward pass of PAM.
        
        Args:
            x: Input feature map [B, C, H, W]
            
        Returns:
            Enhanced feature map with position attention
        """
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value matrices
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # [B, HW, C]
        key = self.key_conv(x).view(batch_size, -1, height * width)  # [B, C, HW]
        value = self.value_conv(x).view(batch_size, -1, height * width)  # [B, C, HW]
        
        # Compute attention weights
        attention = torch.bmm(query, key)  # [B, HW, HW]
        attention = self.softmax(attention)
        
        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(batch_size, channels, height, width)  # [B, C, H, W]
        
        # Residual connection with learnable parameter
        out = self.gamma * out + x
        
        return out


class ChannelAttentionModule(nn.Module):
    """
    Channel Attention Module (CAM)
    
    Pays attention to the channel information to enhance feature representation
    by modeling interdependencies between channels.
    """
    
    def __init__(self, in_channels: int):
        """
        Initialize CAM module.
        
        Args:
            in_channels: Number of input channels
        """
        super(ChannelAttentionModule, self).__init__()
        
        self.in_channels = in_channels
        
        # Channel attention components
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=-1)
        
        # Learnable parameter for residual connection
        self.beta = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Forward pass of CAM.
        
        Args:
            x: Input feature map [B, C, H, W]
            
        Returns:
            Enhanced feature map with channel attention
        """
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value matrices
        query = self.query_conv(x).view(batch_size, -1, height * width)  # [B, C, HW]
        key = self.key_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # [B, HW, C]
        value = self.value_conv(x).view(batch_size, -1, height * width)  # [B, C, HW]
        
        # Compute channel attention weights
        attention = torch.bmm(query, key)  # [B, C, C]
        attention = self.softmax(attention)
        
        # Apply attention to value
        out = torch.bmm(attention, value)  # [B, C, HW]
        out = out.view(batch_size, channels, height, width)  # [B, C, H, W]
        
        # Residual connection with learnable parameter
        out = self.beta * out + x
        
        return out


class CrossModalFusionUnit(nn.Module):
    """
    Cross-Modal-Fusion (CMF) Unit
    
    Composed of self-attention and cross-attention modules to integrate
    features from both modalities (OCT and fundus) by aggregating them
    with attention mechanisms.
    """
    
    def __init__(self, in_channels: int):
        """
        Initialize CMF unit.
        
        Args:
            in_channels: Number of input channels for each modality
        """
        super(CrossModalFusionUnit, self).__init__()
        
        self.in_channels = in_channels
        
        # Self-attention components for each modality
        self.pam_fundus = PositionAttentionModule(in_channels)
        self.cam_fundus = ChannelAttentionModule(in_channels)
        
        self.pam_oct = PositionAttentionModule(in_channels)
        self.cam_oct = ChannelAttentionModule(in_channels)
        
        # Cross-attention components
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Multi-layer perceptron for cross-attention weights
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, 2 * in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, fundus_features, oct_features):
        """
        Forward pass of CMF unit.
        
        Args:
            fundus_features: Fundus feature map [B, C, H, W]
            oct_features: OCT feature map [B, C, H, W]
            
        Returns:
            Tuple of (enhanced_fundus, enhanced_oct, cross_modal_features)
        """
        # Self-attention for fundus modality
        pam_fundus = self.pam_fundus(fundus_features)
        cam_fundus = self.cam_fundus(fundus_features)
        #sa_fundus = torch.cat([pam_fundus, cam_fundus], dim=1)  # [B, 2C, H, W]
        sa_fundus = pam_fundus + cam_fundus  # [B, C, H, W]
        
        # Self-attention for OCT modality  
        pam_oct = self.pam_oct(oct_features)
        cam_oct = self.cam_oct(oct_features)
        #sa_oct = torch.cat([pam_oct, cam_oct], dim=1)  # [B, 2C, H, W]
        sa_oct = pam_oct + cam_oct  # [B, C, H, W]
        
        # Combine self-attention features from both modalities
        #sa_combined = sa_fundus + sa_oct  # [B, 2C, H, W]
        sa_combined = torch.cat([sa_fundus, sa_oct], dim=1)  # [B, 2C, H, W]
        
        # Cross-attention mechanism
        # Global average pooling to get cross-modality descriptor
        gap_features = self.global_avg_pool(sa_combined)  # [B, 2C, 1, 1]
        gap_features = gap_features.view(gap_features.size(0), -1)  # [B, 2C]
        
        # Generate cross-attention weights
        ca_weights = self.mlp(gap_features)  # [B, 2C]
        ca_weights = ca_weights.view(ca_weights.size(0), -1, 1, 1)  # [B, 2C, 1, 1]
        
        # Apply cross-attention weights
        cross_modal_features = sa_combined * ca_weights  # [B, 2C, H, W]
        
        # Enhanced features for each modality (residual connections)
        enhanced_fundus = fundus_features + sa_fundus
        enhanced_oct = oct_features + sa_oct
        
        return enhanced_fundus, enhanced_oct, cross_modal_features


class ModalityFeatureExtraction(nn.Module):
    """
    Modality-Feature-Extraction (MFE) Module
    
    Extracts features from retinal images using a multi-scale approach.
    Uses 7x7 filters in the first layer to capture global context and
    3x3 filters in subsequent layers for fine-grained details.
    """
    
    def __init__(self, input_channels: int = 3):
        """
        Initialize MFE module.
        
        Args:
            input_channels: Number of input channels (3 for RGB images)
        """
        super(ModalityFeatureExtraction, self).__init__()
        
        # First convolutional layer with 7x7 filters for global context
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Subsequent layers with 3x3 filters - ResNet-like structure
        # Conv2: 3 layers
        self.conv2 = self._make_layer(64, 64, 3, stride=1)
        
        # Conv3: 4 layers  
        self.conv3 = self._make_layer(64, 128, 4, stride=2)
        
        # Conv4: 6 layers
        self.conv4 = self._make_layer(128, 256, 6, stride=2)
        
        # Conv5: 3 layers
        self.conv5 = self._make_layer(256, 512, 3, stride=2)
        
        # Global pooling and fully connected layers for attention features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Three fully connected layers as described in paper
        self.fc_attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128)
        )
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a layer with multiple convolutional blocks."""
        layers = []
        
        # First block may have stride > 1 for downsampling
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass of MFE module.
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            Dictionary containing features from different levels
        """
        # Extract features at different levels
        conv1_out = self.conv1(x)  # [B, 64, H/4, W/4]
        conv2_out = self.conv2(conv1_out)  # [B, 64, H/4, W/4]
        conv3_out = self.conv3(conv2_out)  # [B, 128, H/8, W/8]
        conv4_out = self.conv4(conv3_out)  # [B, 256, H/16, W/16]
        conv5_out = self.conv5(conv4_out)  # [B, 512, H/32, W/32]
        
        # Global pooling and attention features
        pooled = self.global_pool(conv5_out).view(conv5_out.size(0), -1)  # [B, 512]
        attention_features = self.fc_attention(pooled)  # [B, 128]
        
        return {
            'conv1': conv1_out,
            'conv2': conv2_out, 
            'conv3': conv3_out,
            'conv4': conv4_out,
            'conv5': conv5_out,
            'attention': attention_features
        }


class DuCAN(nn.Module):
    """
    DuCAN (Dual Cross-Attention Network)
    
    End-to-end framework for MCI detection based on OCT images and fundus photographs.
    The network consists of two MFE subnets and four CMF units, with three classifiers
    for multi-modal and single-modal classification.
    """
    
    def __init__(self, num_classes: int = 3, input_channels: int = 3):
        """
        Initialize DuCAN model.
        
        Args:
            num_classes: Number of classification classes (AD, MCI, CN)
            input_channels: Number of input channels (3 for RGB images)
        """
        super(DuCAN, self).__init__()
        
        self.num_classes = num_classes
        
        # Two MFE modules for fundus and OCT modalities
        self.mfe_fundus = ModalityFeatureExtraction(input_channels)
        self.mfe_oct = ModalityFeatureExtraction(input_channels)
        
        # Four CMF units following each conv layer as described
        self.cmf_conv2 = CrossModalFusionUnit(64)
        self.cmf_conv3 = CrossModalFusionUnit(128)
        self.cmf_conv4 = CrossModalFusionUnit(256)
        self.cmf_conv5 = CrossModalFusionUnit(512)
        
        # Multi-level fusion: subsample features to same size and concatenate
        # Target size for all feature maps (using conv5's spatial size as reference)
        self.target_size = 7  # Assuming input 224x224 -> conv5 is ~7x7
        
        # Adaptive pooling layers to resize features to same spatial size
        self.pool_conv2 = nn.AdaptiveAvgPool2d(self.target_size)
        self.pool_conv3 = nn.AdaptiveAvgPool2d(self.target_size)
        self.pool_conv4 = nn.AdaptiveAvgPool2d(self.target_size)
        self.pool_conv5 = nn.AdaptiveAvgPool2d(self.target_size)
        
        # Weighted addition parameters for multi-level fusion
        # Trainable weights for each level (conv2, conv3, conv4, conv5)
        self.fusion_weights = nn.Parameter(torch.ones(4))  # [w2, w3, w4, w5]
        
        # Channel dimensions for each level: [64, 128, 256, 512] -> need to match
        # Project all channels to same dimension for weighted addition
        self.channel_proj_conv2 = nn.Conv2d(64*2, 512, kernel_size=1)  # 2 modalities
        self.channel_proj_conv3 = nn.Conv2d(128*2, 512, kernel_size=1)
        self.channel_proj_conv4 = nn.Conv2d(256*2, 512, kernel_size=1)
        self.channel_proj_conv5 = nn.Conv2d(512*2, 512, kernel_size=1)
        
        # Auxiliary classifiers (inspired by GoogLeNet for training improvement)
        # 1. Fundus auxiliary classifier
        self.fundus_aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 2. OCT auxiliary classifier  
        self.oct_aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 3. Main multi-modal classifier (multi-level fused features)
        self.multimodal_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),  # After weighted addition, we have 512 channels
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, fundus_images, oct_images):
        """
        Forward pass of DuCAN model with proper multi-step propagation.
        
        Args:
            fundus_images: Fundus photographs [B, C, H, W]
            oct_images: OCT images [B, C, H, W]
            
        Returns:
            Dictionary containing predictions from all three classifiers
        """
        # Initial feature extraction through conv1
        fundus_conv1 = self.mfe_fundus.conv1(fundus_images)
        oct_conv1 = self.mfe_oct.conv1(oct_images)
        
        # Conv2 + CMF2: Multi-step propagation
        fundus_conv2 = self.mfe_fundus.conv2(fundus_conv1)
        oct_conv2 = self.mfe_oct.conv2(oct_conv1)
        enhanced_fundus_2, enhanced_oct_2, cross_modal_2 = self.cmf_conv2(
            fundus_conv2, oct_conv2
        )
        
        # Conv3 + CMF3: Use enhanced features from previous step
        fundus_conv3 = self.mfe_fundus.conv3(enhanced_fundus_2)
        oct_conv3 = self.mfe_oct.conv3(enhanced_oct_2)
        enhanced_fundus_3, enhanced_oct_3, cross_modal_3 = self.cmf_conv3(
            fundus_conv3, oct_conv3
        )
        
        # Conv4 + CMF4: Use enhanced features from previous step
        fundus_conv4 = self.mfe_fundus.conv4(enhanced_fundus_3)
        oct_conv4 = self.mfe_oct.conv4(enhanced_oct_3)
        enhanced_fundus_4, enhanced_oct_4, cross_modal_4 = self.cmf_conv4(
            fundus_conv4, oct_conv4
        )
        
        # Conv5 + CMF5: Use enhanced features from previous step
        fundus_conv5 = self.mfe_fundus.conv5(enhanced_fundus_4)
        oct_conv5 = self.mfe_oct.conv5(enhanced_oct_4)
        enhanced_fundus_5, enhanced_oct_5, cross_modal_5 = self.cmf_conv5(
            fundus_conv5, oct_conv5
        )
        
        # Multi-level fusion: subsample and combine features from all stages
        # Subsample all cross-modal features to same spatial size
        cross_modal_2_pooled = self.pool_conv2(cross_modal_2)  # [B, 128, target_size, target_size]
        cross_modal_3_pooled = self.pool_conv3(cross_modal_3)  # [B, 256, target_size, target_size]
        cross_modal_4_pooled = self.pool_conv4(cross_modal_4)  # [B, 512, target_size, target_size]
        cross_modal_5_pooled = self.pool_conv5(cross_modal_5)  # [B, 1024, target_size, target_size]
        
        # Project all features to same channel dimension (512) for weighted addition
        cross_modal_2_proj = self.channel_proj_conv2(cross_modal_2_pooled)  # [B, 512, target_size, target_size]
        cross_modal_3_proj = self.channel_proj_conv3(cross_modal_3_pooled)  # [B, 512, target_size, target_size]
        cross_modal_4_proj = self.channel_proj_conv4(cross_modal_4_pooled)  # [B, 512, target_size, target_size]
        cross_modal_5_proj = self.channel_proj_conv5(cross_modal_5_pooled)  # [B, 512, target_size, target_size]
        
        # Apply softmax to fusion weights to ensure they sum to 1
        fusion_weights_norm = F.softmax(self.fusion_weights, dim=0)
        
        # Weighted addition of multi-level features
        multi_level_fused = (
            fusion_weights_norm[0] * cross_modal_2_proj +
            fusion_weights_norm[1] * cross_modal_3_proj +
            fusion_weights_norm[2] * cross_modal_4_proj +
            fusion_weights_norm[3] * cross_modal_5_proj
        )
        
        # Auxiliary classifiers for training improvement (inspired by GoogLeNet)
        fundus_aux_pred = self.fundus_aux_classifier(enhanced_fundus_5)
        oct_aux_pred = self.oct_aux_classifier(enhanced_oct_5)
        
        # Main multi-modal classifier using multi-level fused features
        multimodal_pred = self.multimodal_classifier(multi_level_fused)
        
        return {
            'fundus': fundus_aux_pred,
            'oct': oct_aux_pred,
            'multimodal': multimodal_pred,
            'cross_modal_features': multi_level_fused,
            'fusion_weights': fusion_weights_norm  # For analysis
        }
    
    def forward_single_modality(self, images, modality='fundus'):
        """
        Forward pass for single modality inference.
        
        Args:
            images: Input images [B, C, H, W]
            modality: 'fundus' or 'oct'
            
        Returns:
            Single modality predictions
        """
        if modality == 'fundus':
            features = self.mfe_fundus(images)
            return self.fundus_classifier(features['conv5'])
        elif modality == 'oct':
            features = self.mfe_oct(images)
            return self.oct_classifier(features['conv5'])
        else:
            raise ValueError(f"Unknown modality: {modality}")


def create_ducan_model(num_classes: int = 3, input_channels: int = 3) -> DuCAN:
    """
    Create DuCAN model with specified parameters.
    
    Args:
        num_classes: Number of classification classes
        input_channels: Number of input channels
        
    Returns:
        DuCAN model instance
    """
    return DuCAN(num_classes=num_classes, input_channels=input_channels)


# Model registration for the main training script
def DuCANModel(num_classes: int = 3, input_channels: int = 3, **kwargs):
    """
    Factory function for DuCAN model to match the expected interface.
    """
    return create_ducan_model(num_classes=num_classes, input_channels=input_channels)
