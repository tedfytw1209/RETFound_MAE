"""
AD-OCT Model Implementation
Research methodology for Alzheimer's Disease detection using OCT images.

This module implements:
1. Integrated Residual Enhanced Attention Module (IREAM)
2. Clustered Feature Refinement (CFR)
3. Clustered Feature Fusion (C2F)
4. Prediction module with classification and localization losses

Based on the research paper methodology for OCT-based AD detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional, List


class IREAM(nn.Module):
    """
    Integrated Residual Enhanced Attention Module (IREAM)
    
    Combines residual learning and attention mechanisms to enhance the extraction
    of detailed features from OCT images. Includes local channel and global spatial
    attention operations to reduce background noise and enhance relevant information.
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        """
        Initialize IREAM module.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for channel attention
        """
        super(IREAM, self).__init__()
        
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        # Calculate adaptive kernel size for efficient channel attention (ECHA)
        # K = |log2(f) + ρ/σ|_odd where f is channel dimension
        self.adaptive_kernel_size = self._calculate_adaptive_kernel_size(in_channels)
        
        # Local Channel Attention (inspired by ECA)
        self.local_channel_attention = nn.Conv1d(
            1, 1, kernel_size=self.adaptive_kernel_size, 
            padding=(self.adaptive_kernel_size - 1) // 2, bias=False
        )
        
        # Global Spatial Attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
        # Residual connections
        self.residual_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(in_channels)
        
        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        
    def _calculate_adaptive_kernel_size(self, channels: int, rho: float = 2.0, sigma: float = 1.0) -> int:
        """
        Calculate adaptive kernel size based on channel dimension.
        K = |log2(f) + ρ/σ|_odd
        
        Args:
            channels: Number of channels (f)
            rho: Parameter ρ
            sigma: Parameter σ
            
        Returns:
            Odd kernel size
        """
        k = abs(math.log2(channels) + rho / sigma)
        k = int(k)
        # Ensure k is odd and at least 3
        if k % 2 == 0:
            k += 1
        return max(3, k)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of IREAM.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Enhanced feature tensor
        """
        batch_size, channels, height, width = x.size()
        
        # Store input for residual connection
        residual = x
        
        # Local Channel Attention
        # Global average pooling along spatial dimensions
        y = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        
        # Apply 1D convolution for local channel correlation
        y = self.local_channel_attention(y)  # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        
        # Apply channel attention
        channel_attention = self.sigmoid(y)
        x_channel = x * channel_attention
        
        # Global Spatial Attention
        # Compute spatial statistics
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)  # (B, 1, H, W)
        max_pool, _ = torch.max(x_channel, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate and apply spatial convolution
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, H, W)
        spatial_attention = self.sigmoid(self.spatial_conv(spatial_input))  # (B, 1, H, W)
        
        # Apply spatial attention
        x_spatial = x_channel * spatial_attention
        
        # Residual connection with batch normalization
        residual_processed = self.batch_norm(self.residual_conv(residual))
        output = self.relu(x_spatial + residual_processed)
        
        return output


class CFR(nn.Module):
    """
    Clustered Feature Refinement (CFR)
    
    Groups similar features and refines them to amplify discriminative capability.
    Enhances OCT polarization features (Depolarization, Birefringence, 
    Degree of Polarization Uniformity) by expanding channels 3-fold and 
    applying importance weighting.
    """
    
    def __init__(self, in_channels: int = 256, num_groups: int = 3):
        """
        Initialize CFR module.
        
        Args:
            in_channels: Number of input channels (R=256 in paper)
            num_groups: Number of polarization feature groups (3 in paper)
        """
        super(CFR, self).__init__()
        
        self.in_channels = in_channels
        self.num_groups = num_groups
        self.group_channels = in_channels  # Each group has same channels as input
        
        # Group convolutions to generate separate feature maps
        self.group_convs = nn.ModuleList([
            nn.Conv2d(in_channels, self.group_channels, kernel_size=3, padding=1, groups=1)
            for _ in range(num_groups)
        ])
        
        # Global pooling operations
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Batch normalization for each group
        self.group_norms = nn.ModuleList([
            nn.BatchNorm2d(self.group_channels) for _ in range(num_groups)
        ])
        
        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.epsilon = 1e-5  # ω for numerical stability
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass of CFR.
        
        Args:
            x: Input feature map (batch_size, channels, height, width)
            
        Returns:
            List of enhanced polarization feature groups
        """
        batch_size, channels, height, width = x.size()
        enhanced_groups = []
        
        for g in range(self.num_groups):
            # Generate g-th group feature map using group convolution
            Fg = self.group_convs[g](x)  # (B, group_channels, H, W)
            Fg = self.group_norms[g](Fg)
            
            # Extract global semantic feature Q using global pooling
            # Combine average and max pooling for richer representation
            Q_avg = self.global_avg_pool(Fg)  # (B, group_channels, 1, 1)
            Q_max = self.global_max_pool(Fg)  # (B, group_channels, 1, 1)
            Q = (Q_avg + Q_max) / 2  # Global semantic feature
            
            # Calculate importance coefficients for each spatial location
            # Reshape for batch matrix multiplication
            Fg_reshaped = Fg.view(batch_size, self.group_channels, -1)  # (B, C, H*W)
            Q_reshaped = Q.view(batch_size, self.group_channels, 1)  # (B, C, 1)
            
            # Compute dot product between global and local features
            zeta_l = torch.bmm(Q_reshaped.transpose(1, 2), Fg_reshaped)  # (B, 1, H*W)
            zeta_l = zeta_l.view(batch_size, 1, height, width)  # (B, 1, H, W)
            
            # Normalize importance coefficients
            # Calculate mean and variance across spatial dimensions
            mu_zeta = torch.mean(zeta_l, dim=[2, 3], keepdim=True)  # (B, 1, 1, 1)
            var_zeta = torch.var(zeta_l, dim=[2, 3], keepdim=True)  # (B, 1, 1, 1)
            std_zeta = torch.sqrt(var_zeta + self.epsilon)
            
            # Normalize using batch statistics
            zeta_l_normalized = (zeta_l - mu_zeta) / (std_zeta + self.epsilon)
            
            # Apply sigmoid activation to get importance weights
            importance_weights = self.sigmoid(zeta_l_normalized)  # (B, 1, H, W)
            
            # Apply importance weighting to original features
            Fg_enhanced = Fg * importance_weights  # (B, group_channels, H, W)
            
            enhanced_groups.append(Fg_enhanced)
        
        return enhanced_groups


class C2F(nn.Module):
    """
    Clustered Feature Fusion (C2F)
    
    Integrates features from various groups into a cohesive set, ensuring diverse
    aspects of the data are captured. Includes Pooling Channel Attention (PCA)
    mechanism and expands receptive field through 3x3 convolutions.
    """
    
    def __init__(self, group_channels: int = 256, num_groups: int = 3):
        """
        Initialize C2F module.
        
        Args:
            group_channels: Number of channels in each group
            num_groups: Number of feature groups to fuse
        """
        super(C2F, self).__init__()
        
        self.group_channels = group_channels
        self.num_groups = num_groups
        
        # 3x3 convolutions for each group to expand receptive field
        self.group_convs = nn.ModuleList([
            nn.Conv2d(group_channels, group_channels, kernel_size=3, padding=1)
            for _ in range(num_groups)
        ])
        
        # Batch normalization for each group
        self.group_norms = nn.ModuleList([
            nn.BatchNorm2d(group_channels) for _ in range(num_groups)
        ])
        
        # Pooling Channel Attention (PCA) mechanism
        self.pca_global_pool = nn.AdaptiveAvgPool2d(1)
        self.pca_fc = nn.Sequential(
            nn.Linear(group_channels * num_groups, group_channels * num_groups // 4),
            nn.ReLU(inplace=True),
            nn.Linear(group_channels * num_groups // 4, group_channels * num_groups),
            nn.Sigmoid()
        )
        
        # Final fusion convolution
        self.fusion_conv = nn.Conv2d(
            group_channels * num_groups, group_channels, 
            kernel_size=1, bias=False
        )
        self.fusion_norm = nn.BatchNorm2d(group_channels)
        
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, feature_groups: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of C2F.
        
        Args:
            feature_groups: List of enhanced polarization feature groups
            
        Returns:
            Fused multi-scale polarization features
        """
        if len(feature_groups) != self.num_groups:
            raise ValueError(f"Expected {self.num_groups} feature groups, got {len(feature_groups)}")
        
        batch_size = feature_groups[0].size(0)
        
        # Apply 3x3 convolutions to expand receptive field
        processed_groups = []
        for i, group_features in enumerate(feature_groups):
            # Apply group-specific convolution
            processed = self.group_convs[i](group_features)
            processed = self.group_norms[i](processed)
            processed = self.relu(processed)
            processed_groups.append(processed)
        
        # Concatenate all processed groups
        concatenated = torch.cat(processed_groups, dim=1)  # (B, group_channels*num_groups, H, W)
        
        # Apply Pooling Channel Attention (PCA)
        # Global pooling to get channel-wise statistics
        pca_pool = self.pca_global_pool(concatenated)  # (B, group_channels*num_groups, 1, 1)
        pca_pool = pca_pool.view(batch_size, -1)  # (B, group_channels*num_groups)
        
        # Generate attention weights
        attention_weights = self.pca_fc(pca_pool)  # (B, group_channels*num_groups)
        attention_weights = attention_weights.view(batch_size, -1, 1, 1)  # (B, group_channels*num_groups, 1, 1)
        
        # Apply attention weights
        attended_features = concatenated * attention_weights
        
        # Final fusion through 1x1 convolution
        fused_features = self.fusion_conv(attended_features)  # (B, group_channels, H, W)
        fused_features = self.fusion_norm(fused_features)
        fused_features = self.relu(fused_features)
        
        return fused_features


class ADOCTLoss(nn.Module):
    """
    AD-OCT Loss Module
    
    Implements the loss function combining classification and localization losses:
    LOSS = LO_Class + LO_LOC
    
    Uses BCE With Logits Loss for classification and Complete IoU (CIoU) for localization.
    """
    
    def __init__(self, classification_weight: float = 1.0, localization_weight: float = 1.0):
        """
        Initialize AD-OCT loss module.
        
        Args:
            classification_weight: Weight for classification loss
            localization_weight: Weight for localization loss
        """
        super(ADOCTLoss, self).__init__()
        
        self.classification_weight = classification_weight
        self.localization_weight = localization_weight
        
        # BCE with logits loss for classification
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                pred_boxes: Optional[torch.Tensor] = None, 
                target_boxes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of AD-OCT loss.
        
        Args:
            predictions: Classification predictions (batch_size, num_classes)
            targets: Classification targets (batch_size, num_classes)
            pred_boxes: Predicted bounding boxes (batch_size, 4) [x1, y1, x2, y2]
            target_boxes: Target bounding boxes (batch_size, 4) [x1, y1, x2, y2]
            
        Returns:
            Combined loss value
        """
        # Classification loss using BCE with logits
        classification_loss = self.bce_loss(predictions, targets.float())
        
        total_loss = self.classification_weight * classification_loss
        
        # Add localization loss if bounding boxes are provided
        if pred_boxes is not None and target_boxes is not None:
            localization_loss = self._compute_ciou_loss(pred_boxes, target_boxes)
            total_loss += self.localization_weight * localization_loss
        
        return total_loss
    
    def _compute_ciou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute Complete Intersection over Union (CIoU) loss.
        
        Args:
            pred_boxes: Predicted boxes (batch_size, 4) [x1, y1, x2, y2]
            target_boxes: Target boxes (batch_size, 4) [x1, y1, x2, y2]
            
        Returns:
            CIoU loss value
        """
        # Compute IoU
        iou = self._compute_iou(pred_boxes, target_boxes)
        
        # Compute center distance
        pred_centers = self._get_box_centers(pred_boxes)
        target_centers = self._get_box_centers(target_boxes)
        center_distance = torch.sum((pred_centers - target_centers) ** 2, dim=1)
        
        # Compute diagonal distance of enclosing box
        enclosing_box = self._get_enclosing_box(pred_boxes, target_boxes)
        diagonal_distance = self._get_diagonal_distance(enclosing_box)
        
        # Compute aspect ratio consistency
        pred_aspect_ratio = self._get_aspect_ratio(pred_boxes)
        target_aspect_ratio = self._get_aspect_ratio(target_boxes)
        
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(target_aspect_ratio) - torch.atan(pred_aspect_ratio), 2
        )
        
        # Compute alpha parameter
        alpha = v / (1 - iou + v + 1e-8)
        
        # Compute CIoU loss
        ciou_loss = 1 - iou + center_distance / (diagonal_distance + 1e-8) + alpha * v
        
        return torch.mean(ciou_loss)
    
    def _compute_iou(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute Intersection over Union."""
        # Get intersection coordinates
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        # Compute intersection area
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Compute union area
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union = pred_area + target_area - intersection
        
        # Compute IoU
        iou = intersection / (union + 1e-8)
        return iou
    
    def _get_box_centers(self, boxes: torch.Tensor) -> torch.Tensor:
        """Get box centers."""
        centers = torch.zeros_like(boxes[:, :2])
        centers[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x center
        centers[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y center
        return centers
    
    def _get_enclosing_box(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Get smallest enclosing box."""
        x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def _get_diagonal_distance(self, boxes: torch.Tensor) -> torch.Tensor:
        """Get diagonal distance of boxes."""
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        return torch.sqrt(width ** 2 + height ** 2)
    
    def _get_aspect_ratio(self, boxes: torch.Tensor) -> torch.Tensor:
        """Get aspect ratio of boxes."""
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        return width / (height + 1e-8)


class ADOCTModel(nn.Module):
    """
    Complete AD-OCT Model
    
    Integrates IREAM, CFR, and C2F modules for Alzheimer's Disease detection
    using OCT images. Supports both classification and localization tasks.
    """
    
    def __init__(self, num_classes: int = 2, input_channels: int = 3, 
                 feature_channels: int = 256, num_groups: int = 3,
                 include_localization: bool = False):
        """
        Initialize AD-OCT model.
        
        Args:
            num_classes: Number of classification classes
            input_channels: Number of input channels
            feature_channels: Number of feature channels
            num_groups: Number of polarization feature groups
            include_localization: Whether to include localization head
        """
        super(ADOCTModel, self).__init__()
        
        self.num_classes = num_classes
        self.include_localization = include_localization
        
        # Backbone CNN for initial feature extraction
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet-like blocks
            self._make_layer(64, 64, 2, stride=1),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
        )
        
        # IREAM module
        self.iream = IREAM(in_channels=feature_channels)
        
        # CFR module
        self.cfr = CFR(in_channels=feature_channels, num_groups=num_groups)
        
        # C2F module
        self.c2f = C2F(group_channels=feature_channels, num_groups=num_groups)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_channels, feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_channels // 2, num_classes)
        )
        
        # Localization head (optional)
        if include_localization:
            self.localizer = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(feature_channels, feature_channels // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(feature_channels // 2, 4)  # 4 coordinates for bounding box
            )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        """Create a residual layer."""
        layers = []
        layers.append(self._make_block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(self._make_block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _make_block(self, in_channels: int, out_channels: int, stride: int = 1):
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of AD-OCT model.
        
        Args:
            x: Input OCT images (batch_size, channels, height, width)
            
        Returns:
            Tuple of (classification_logits, localization_boxes)
        """
        # Backbone feature extraction
        features = self.backbone(x)  # (B, 256, H/8, W/8)
        
        # Apply IREAM for enhanced attention
        enhanced_features = self.iream(features)
        
        # Apply CFR for feature refinement
        refined_groups = self.cfr(enhanced_features)
        
        # Apply C2F for feature fusion
        fused_features = self.c2f(refined_groups)
        
        # Global pooling
        pooled_features = self.global_pool(fused_features)  # (B, 256, 1, 1)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # (B, 256)
        
        # Classification
        classification_logits = self.classifier(pooled_features)
        
        # Localization (optional)
        localization_boxes = None
        if self.include_localization:
            localization_boxes = self.localizer(pooled_features)
        
        return classification_logits, localization_boxes


def create_ad_oct_model(num_classes: int = 2, include_localization: bool = False, **kwargs) -> ADOCTModel:
    """
    Create AD-OCT model with specified configuration.
    
    Args:
        num_classes: Number of classification classes
        include_localization: Whether to include localization head
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized AD-OCT model
    """
    return ADOCTModel(
        num_classes=num_classes,
        include_localization=include_localization,
        **kwargs
    )


def create_ad_oct_loss(classification_weight: float = 1.0, 
                       localization_weight: float = 1.0) -> ADOCTLoss:
    """
    Create AD-OCT loss function.
    
    Args:
        classification_weight: Weight for classification loss
        localization_weight: Weight for localization loss
        
    Returns:
        AD-OCT loss function
    """
    return ADOCTLoss(
        classification_weight=classification_weight,
        localization_weight=localization_weight
    )


# Export functions for easy import
__all__ = [
    'IREAM', 'CFR', 'C2F', 'ADOCTLoss', 'ADOCTModel',
    'create_ad_oct_model', 'create_ad_oct_loss'
]
