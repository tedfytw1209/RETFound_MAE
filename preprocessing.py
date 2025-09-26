"""
Preprocessing module for DuCAN framework
Implements preprocessing methods for fundus photographs and OCT images.

This module includes:
1. ROI extraction for fundus photographs using image masks
2. Size standardization 
3. CLAHE (Contrast-Limited Adaptive Histogram Equalization) for fundus images
4. Otsu thresholding for OCT foreground masking
5. Hybrid filtering (bilateral + median) for OCT noise reduction
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
import torchvision.transforms as transforms
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt


class FundusPreprocessor:
    """
    Preprocessor for fundus photographs with ROI extraction, size standardization, and CLAHE.
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_grid_size: Tuple[int, int] = (8, 8),
                 roi_threshold: int = 10):
        """
        Initialize fundus preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            clahe_clip_limit: Clipping limit for CLAHE
            clahe_tile_grid_size: Size of the neighborhood for CLAHE
            roi_threshold: Threshold for ROI detection
        """
        self.target_size = target_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.roi_threshold = roi_threshold
        
        # Create CLAHE object
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid_size
        )
    
    def extract_roi(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract region of interest (ROI) from fundus photograph.
        Fundus photographs have black background with redundant information.
        
        Args:
            image: Input fundus image [H, W, C]
            
        Returns:
            Tuple of (roi_image, mask)
        """
        if len(image.shape) == 3:
            # Convert to grayscale for mask creation
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Create mask by thresholding to remove black background
        _, mask = cv2.threshold(gray, self.roi_threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find the largest contour (main fundus region)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.fillPoly(mask, [largest_contour], 255)
        
        # Apply mask to original image
        if len(image.shape) == 3:
            mask_3d = np.stack([mask] * 3, axis=-1) / 255.0
            roi_image = image * mask_3d
        else:
            roi_image = image * (mask / 255.0)
        
        return roi_image.astype(np.uint8), mask
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE).
        
        CLAHE operates on small regions individually with bilinear interpolation
        to avoid artifacts and create seamless appearance.
        
        Args:
            image: Input image [H, W, C] or [H, W]
            
        Returns:
            CLAHE-enhanced image
        """
        if len(image.shape) == 3:
            # Apply CLAHE to each channel separately
            enhanced_channels = []
            for i in range(image.shape[2]):
                channel = image[:, :, i]
                enhanced_channel = self.clahe.apply(channel)
                enhanced_channels.append(enhanced_channel)
            enhanced_image = np.stack(enhanced_channels, axis=-1)
        else:
            # Grayscale image
            enhanced_image = self.clahe.apply(image)
        
        return enhanced_image
    
    def standardize_size(self, image: np.ndarray) -> np.ndarray:
        """
        Standardize image size using high-quality interpolation.
        
        Args:
            image: Input image [H, W, C] or [H, W]
            
        Returns:
            Resized image
        """
        # Use INTER_CUBIC for high-quality upsampling, INTER_AREA for downsampling
        current_size = image.shape[:2]
        if current_size[0] * current_size[1] > self.target_size[0] * self.target_size[1]:
            interpolation = cv2.INTER_AREA  # Better for downsampling
        else:
            interpolation = cv2.INTER_CUBIC  # Better for upsampling
        
        resized_image = cv2.resize(
            image, 
            (self.target_size[1], self.target_size[0]), 
            interpolation=interpolation
        )
        
        return resized_image
    
    def preprocess(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Complete preprocessing pipeline for fundus photographs.
        
        Pipeline:
        1. Convert to numpy array if needed
        2. Extract ROI using image masks
        3. Standardize size
        4. Apply CLAHE for contrast enhancement
        
        Args:
            image: Input fundus image
            
        Returns:
            Preprocessed fundus image
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is in RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Already RGB
            pass
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA to RGB
            image = image[:, :, :3]
        
        # Step 1: Extract ROI to remove black background
        roi_image, mask = self.extract_roi(image)
        
        # Step 2: Standardize size
        standardized_image = self.standardize_size(roi_image)
        
        # Step 3: Apply CLAHE for contrast enhancement
        enhanced_image = self.apply_clahe(standardized_image)
        
        return enhanced_image


class OCTPreprocessor:
    """
    Preprocessor for OCT images with foreground masking, hybrid filtering, and size standardization.
    
    OCT images contain background regions with only indicative features in central retinal layers.
    Uses Otsu thresholding for foreground masking and hybrid filtering (bilateral + median)
    for noise reduction while preserving adjacent layer relationships.
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 use_otsu_masking: bool = True,
                 use_hybrid_filtering: bool = True,
                 bilateral_d: int = 9,
                 bilateral_sigma_color: float = 75,
                 bilateral_sigma_space: float = 75,
                 median_kernel_size: int = 5,
                 normalize_intensity: bool = True):
        """
        Initialize OCT preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            use_otsu_masking: Whether to apply Otsu thresholding for foreground masking
            use_hybrid_filtering: Whether to apply hybrid filtering (bilateral + median)
            bilateral_d: Diameter for bilateral filtering
            bilateral_sigma_color: Filter sigma in the color space for bilateral filtering
            bilateral_sigma_space: Filter sigma in the coordinate space for bilateral filtering
            median_kernel_size: Kernel size for median filtering
            normalize_intensity: Whether to normalize intensity values
        """
        self.target_size = target_size
        self.use_otsu_masking = use_otsu_masking
        self.use_hybrid_filtering = use_hybrid_filtering
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.median_kernel_size = median_kernel_size
        self.normalize_intensity = normalize_intensity
    
    def create_otsu_mask(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create foreground mask using Otsu method for threshold segmentation.
        
        OCT images contain background regions, but indicative features only appear
        in central retinal layers. This method highlights indicative features and
        avoids displaying irrelevant regions.
        
        Args:
            image: Input OCT image [H, W, C] or [H, W]
            
        Returns:
            Tuple of (masked_image, mask)
        """
        # Convert to grayscale for Otsu thresholding
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply Otsu's thresholding to find optimal threshold
        otsu_threshold, otsu_mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Clean up the mask using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel)
        otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply mask to original image
        if len(image.shape) == 3:
            mask_3d = np.stack([otsu_mask] * 3, axis=-1) / 255.0
            masked_image = image * mask_3d
        else:
            masked_image = image * (otsu_mask / 255.0)
        
        return masked_image.astype(np.uint8), otsu_mask
    
    def apply_hybrid_filtering(self, image: np.ndarray) -> np.ndarray:
        """
        Apply hybrid filtering combining bilateral and median filtering for noise reduction.
        
        OCT generates considerable noise during acquisition. Due to the particularity of
        OCT images, the relationship between adjacent layers must remain unchanged during
        noise removal. Hybrid filtering preserves layer relationships while reducing noise.
        
        Args:
            image: Input OCT image [H, W, C] or [H, W]
            
        Returns:
            Filtered image with preserved layer relationships
        """
        # Step 1: Apply bilateral filtering to preserve edges while smoothing
        # Bilateral filter preserves edges (layer boundaries) while reducing noise
        if len(image.shape) == 3:
            # Color image
            bilateral_filtered = cv2.bilateralFilter(
                image, 
                self.bilateral_d, 
                self.bilateral_sigma_color, 
                self.bilateral_sigma_space
            )
        else:
            # Grayscale image
            bilateral_filtered = cv2.bilateralFilter(
                image, 
                self.bilateral_d, 
                self.bilateral_sigma_color, 
                self.bilateral_sigma_space
            )
        
        # Step 2: Apply median filtering to remove salt-and-pepper noise
        # Median filter helps remove impulse noise while preserving layer structure
        median_filtered = cv2.medianBlur(bilateral_filtered, self.median_kernel_size)
        
        return median_filtered
    
    def normalize_intensity_values(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize intensity values to improve contrast.
        
        Args:
            image: Input image [H, W, C] or [H, W]
            
        Returns:
            Intensity-normalized image
        """
        # Normalize to 0-255 range
        image_min = image.min()
        image_max = image.max()
        
        if image_max > image_min:
            normalized = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
        else:
            normalized = image.astype(np.uint8)
        
        return normalized
    
    def standardize_size(self, image: np.ndarray) -> np.ndarray:
        """
        Standardize image size using high-quality interpolation.
        
        Args:
            image: Input image [H, W, C] or [H, W]
            
        Returns:
            Resized image
        """
        # Use INTER_CUBIC for high-quality upsampling, INTER_AREA for downsampling
        current_size = image.shape[:2]
        if current_size[0] * current_size[1] > self.target_size[0] * self.target_size[1]:
            interpolation = cv2.INTER_AREA  # Better for downsampling
        else:
            interpolation = cv2.INTER_CUBIC  # Better for upsampling
        
        resized_image = cv2.resize(
            image, 
            (self.target_size[1], self.target_size[0]), 
            interpolation=interpolation
        )
        
        return resized_image
    
    def preprocess(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Complete preprocessing pipeline for OCT images.
        
        Pipeline:
        1. Convert to numpy array if needed
        2. Create Otsu mask for foreground region (optional)
        3. Apply hybrid filtering for noise reduction (optional)
        4. Normalize intensity (optional)
        5. Standardize size
        
        Args:
            image: Input OCT image
            
        Returns:
            Preprocessed OCT image with highlighted indicative features
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is in proper format
        if len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA to RGB
            image = image[:, :, :3]
        
        # Step 1: Create Otsu mask for foreground region
        # Highlights indicative features in central retinal layers
        if self.use_otsu_masking:
            image, otsu_mask = self.create_otsu_mask(image)
        
        # Step 2: Apply hybrid filtering for noise reduction
        # Preserves adjacent layer relationships while reducing noise
        if self.use_hybrid_filtering:
            image = self.apply_hybrid_filtering(image)
        
        # Step 3: Normalize intensity if enabled
        if self.normalize_intensity:
            image = self.normalize_intensity_values(image)
        
        # Step 4: Standardize size
        standardized_image = self.standardize_size(image)
        
        return standardized_image


class DuCANPreprocessor:
    """
    Combined preprocessor for DuCAN framework handling both fundus and OCT images.
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 fundus_clahe_clip_limit: float = 2.0,
                 fundus_clahe_tile_grid_size: Tuple[int, int] = (8, 8)):
        """
        Initialize DuCAN preprocessor.
        
        Args:
            target_size: Target image size for both modalities
            fundus_clahe_clip_limit: CLAHE clip limit for fundus images
            fundus_clahe_tile_grid_size: CLAHE tile grid size for fundus images
        """
        self.fundus_preprocessor = FundusPreprocessor(
            target_size=target_size,
            clahe_clip_limit=fundus_clahe_clip_limit,
            clahe_tile_grid_size=fundus_clahe_tile_grid_size
        )
        
        self.oct_preprocessor = OCTPreprocessor(
            target_size=target_size,
            use_otsu_masking=True,
            use_hybrid_filtering=True,
            bilateral_d=9,
            bilateral_sigma_color=75,
            bilateral_sigma_space=75,
            median_kernel_size=5,
            normalize_intensity=True
        )
    
    def preprocess_fundus(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Preprocess fundus image."""
        return self.fundus_preprocessor.preprocess(image)
    
    def preprocess_oct(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Preprocess OCT image."""
        return self.oct_preprocessor.preprocess(image)
    
    def preprocess_batch(self, 
                        fundus_images: list, 
                        oct_images: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess a batch of fundus and OCT images.
        
        Args:
            fundus_images: List of fundus images
            oct_images: List of OCT images
            
        Returns:
            Tuple of (preprocessed_fundus_batch, preprocessed_oct_batch)
        """
        preprocessed_fundus = []
        preprocessed_oct = []
        
        for fundus_img in fundus_images:
            processed_fundus = self.preprocess_fundus(fundus_img)
            preprocessed_fundus.append(processed_fundus)
        
        for oct_img in oct_images:
            processed_oct = self.preprocess_oct(oct_img)
            preprocessed_oct.append(processed_oct)
        
        return np.array(preprocessed_fundus), np.array(preprocessed_oct)


def visualize_preprocessing(original_image: np.ndarray, 
                           preprocessed_image: np.ndarray,
                           title: str = "Preprocessing Comparison") -> None:
    """
    Visualize the before and after preprocessing results.
    
    Args:
        original_image: Original image
        preprocessed_image: Preprocessed image
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_image, cmap='gray' if len(original_image.shape) == 2 else None)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(preprocessed_image, cmap='gray' if len(preprocessed_image.shape) == 2 else None)
    axes[1].set_title('Preprocessed Image')
    axes[1].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Initialize preprocessors
    ducan_preprocessor = DuCANPreprocessor(target_size=(224, 224))
    
    # Example preprocessing (would work with actual images)
    print("DuCAN Preprocessor initialized successfully!")
    print("Features:")
    print("- Fundus: ROI extraction, size standardization, CLAHE enhancement")
    print("- OCT: Denoising, intensity normalization, size standardization")
