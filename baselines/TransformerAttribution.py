"""
Transformer Attribution (Chefer et al., 2021)
Generates class-specific saliency maps by combining attention with gradients.

Reference: Chefer, H., Gur, S., & Wolf, L. (2021).
Transformer Interpretability Beyond Attention Visualization. CVPR 2021.
"""

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import PatchEmbed
from torchvision.models.feature_extraction import create_feature_extractor


class TransformerAttribution(nn.Module):
    """
    Transformer Attribution for Vision Transformers.

    Combines attention weights with gradients to produce class-specific
    saliency maps that show which regions are relevant for a particular
    class prediction.
    """

    def __init__(self, model, model_name, input_size, N=12, device=None):
        """
        Args:
            model: The ViT model to explain
            model_name: Name of the model (e.g., 'RETFound_mae', 'vit-base-patch16-224')
            input_size: Input image size (e.g., 224)
            N: Number of transformer layers (default: 12)
            device: Device to run on
        """
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.input_size = input_size
        self.N = N
        self.device = device

        # Set up based on model type
        self._setup_model()

    def _setup_model(self):
        """Set up feature extractor based on model type."""
        # Detect HuggingFace models by presence of .config attribute
        is_hf = hasattr(self.model, 'config')

        if not is_hf:
            # timm / standard PyTorch ViT (RETFound, vit_base_patch16_224, etc.)
            self.return_attns = [f'blocks.{i}.attn.softmax' for i in range(self.N)]
            self.feature_extractor = create_feature_extractor(
                self.model,
                return_nodes=self.return_attns,
                tracer_kwargs={'leaf_modules': [PatchEmbed]}
            )
            self.use_timm = True
        else:
            # HuggingFace ViT / DINO
            self.feature_extractor = None
            self.use_timm = False

    def forward(self, inputs=None, targets=None, model=None, **kwargs):
        """
        Generate saliency maps for a batch of images.

        Args:
            inputs: Input images tensor (B, C, H, W)
            targets: Target class indices (B,) - required for gradient computation
            model: Optional model override
            **kwargs: Additional arguments (ignored)

        Returns:
            np.ndarray: Saliency maps of shape (B, input_size, input_size)
        """
        if model is None:
            model = self.model
        if inputs is None:
            raise ValueError("inputs parameter is required")
        if targets is None:
            raise ValueError("targets parameter is required for Transformer Attribution")

        # Ensure inputs are on correct device
        if self.device is not None:
            inputs = inputs.to(self.device)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(self.device)

        model.eval()
        B = inputs.shape[0]
        saliency_maps = []

        # Process each image individually (need per-sample gradients)
        for i in range(B):
            img = inputs[i:i+1].clone().detach().requires_grad_(True)
            target = targets[i] if isinstance(targets, torch.Tensor) else targets

            saliency = self._compute_single_attribution(img, target, model)
            saliency_maps.append(saliency)

        return np.stack(saliency_maps)

    def _compute_single_attribution(self, img, target, model):
        """
        Compute attribution for a single image.

        Args:
            img: Single image tensor (1, C, H, W) with requires_grad=True
            target: Target class index
            model: The model

        Returns:
            np.ndarray: Saliency map (input_size, input_size)
        """
        model.zero_grad()

        if self.use_timm:
            # timm models (RETFound)
            features = self.feature_extractor(img)
            attentions = [features[key] for key in self.return_attns]
            output = model(img)
            logits = output.logits if hasattr(output, 'logits') else output
        else:
            # HuggingFace model
            if hasattr(model, "config"):
                model.config.output_attentions = True
                model.config.return_dict = True
            output = model(pixel_values=img, output_attentions=True, return_dict=True)
            logits = output.logits if hasattr(output, 'logits') else output
            attentions = output.attentions
            if attentions is None and hasattr(model, 'vit'):
                output = model.vit(pixel_values=img, output_attentions=True, return_dict=True)
                attentions = output.attentions

        # Ensure all attention tensors require gradients
        attentions_with_grad = []
        for attn in attentions:
            if not attn.requires_grad:
                attn = attn.clone().detach().requires_grad_(True)
            attn.retain_grad()
            attentions_with_grad.append(attn)

        # Get target class score
        if isinstance(target, torch.Tensor):
            target_idx = target.item()
        else:
            target_idx = int(target)

        target_score = logits[0, target_idx]

        # Backward pass to get gradients
        target_score.backward(retain_graph=True)

        # Compute relevance: attention * gradient (positive only)
        relevance_maps = []
        for attn in attentions_with_grad:
            if attn.grad is not None:
                grad = attn.grad.detach()
            else:
                grad = torch.ones_like(attn)

            # attn shape: (1, num_heads, num_tokens, num_tokens)
            relevance = (attn.detach() * grad).clamp(min=0)
            # Average over heads
            relevance = relevance.mean(dim=1)  # (1, num_tokens, num_tokens)
            relevance_maps.append(relevance.cpu().numpy())

        # Aggregate using rollout
        saliency = self._rollout_relevance(relevance_maps)

        # Resize to input size
        saliency = self._resize_saliency(saliency)

        return saliency

    def _rollout_relevance(self, relevance_maps):
        """
        Aggregate relevance across layers using rollout.

        Args:
            relevance_maps: List of relevance matrices (1, num_tokens, num_tokens)

        Returns:
            np.ndarray: CLS token attention to patches (num_patches,)
        """
        num_tokens = relevance_maps[0].shape[-1]
        rollout = np.eye(num_tokens)

        for relevance in relevance_maps:
            rel = relevance.squeeze(0)  # (num_tokens, num_tokens)
            rel = rel + np.eye(num_tokens)  # Add residual
            rel = rel / (rel.sum(axis=-1, keepdims=True) + 1e-9)  # Normalize
            rollout = np.matmul(rollout, rel)

        # CLS token attention to patch tokens (exclude CLS)
        cls_attention = rollout[0, 1:]
        return cls_attention

    def _resize_saliency(self, saliency):
        """
        Reshape and resize saliency to image dimensions.

        Args:
            saliency: Flat saliency array (num_patches,)

        Returns:
            np.ndarray: Saliency map (input_size, input_size)
        """
        num_patches = saliency.shape[0]
        patch_grid = int(np.sqrt(num_patches))
        saliency_2d = saliency.reshape(patch_grid, patch_grid).astype(np.float32)

        saliency_resized = np.array(
            Image.fromarray(saliency_2d).resize(
                (self.input_size, self.input_size),
                resample=Image.BILINEAR
            )
        )

        # Normalize to [0, 1]
        saliency_min = saliency_resized.min()
        saliency_max = saliency_resized.max()
        return (saliency_resized - saliency_min) / (saliency_max - saliency_min + 1e-8)


if __name__ == "__main__":
    import timm

    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
    model.eval()

    tf_attr = TransformerAttribution(
        model=model,
        model_name='vit_base_patch16_224',
        input_size=224,
        N=12,
        device='cpu'
    )

    x = torch.randn(2, 3, 224, 224)
    targets = torch.tensor([0, 1])

    saliency = tf_attr(inputs=x, targets=targets)
    print("Saliency shape:", saliency.shape)
    print("Saliency range:", saliency.min(), "-", saliency.max())
