"""
Transformer Attribution (Chefer et al., 2021)
Generates class-specific saliency maps by combining attention with gradients.

Reference: Chefer, H., Gur, S., & Wolf, L. (2021).
Transformer Interpretability Beyond Attention Visualization. CVPR 2021.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # Detect model type once at init
        self.use_timm = not hasattr(self.model, 'config')

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

        if self.device is not None:
            inputs = inputs.to(self.device)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(self.device)

        model.eval()
        B = inputs.shape[0]
        saliency_maps = []

        for i in range(B):
            img = inputs[i:i+1].clone().detach().requires_grad_(True)
            target = targets[i] if isinstance(targets, torch.Tensor) else targets
            saliency = self._compute_single_attribution(img, target, model)
            saliency_maps.append(saliency)

        return np.stack(saliency_maps)

    def _compute_single_attribution(self, img, target, model):
        """
        Compute attribution for a single image using a single forward pass.

        Attention tensors are captured via forward hooks registered on each
        attention softmax layer. This ensures the captured tensors share the
        same computation graph as the output logits, so gradients flow back
        through them correctly.

        Args:
            img: Single image tensor (1, C, H, W) with requires_grad=True
            target: Target class index
            model: The model

        Returns:
            np.ndarray: Saliency map (input_size, input_size)
        """
        model.zero_grad()

        if self.use_timm:
            # Register hooks on attention softmax modules so attention tensors
            # are captured inside the same forward pass that produces logits.
            captured_attentions = []
            hooks = []

            def make_hook():
                def hook_fn(module, input, output):
                    # output is in the computation graph; retain grad so
                    # backward() can populate output.grad
                    output.retain_grad()
                    captured_attentions.append(output)
                return hook_fn

            for i in range(self.N):
                attn_module = model.blocks[i].attn
                # Older timm: self.softmax = nn.Softmax(dim=-1)
                # Newer timm: softmax is called inline, use attn_drop as proxy
                if hasattr(attn_module, 'softmax'):
                    hook_target = attn_module.softmax
                elif hasattr(attn_module, 'attn_drop'):
                    hook_target = attn_module.attn_drop
                else:
                    raise AttributeError(
                        f"Cannot find softmax or attn_drop on blocks[{i}].attn "
                        f"({type(attn_module).__name__}). "
                        "Please inspect the attention module and update hook_target."
                    )
                hooks.append(hook_target.register_forward_hook(make_hook()))

            try:
                output = model(img)
                logits = output.logits if hasattr(output, 'logits') else output
                attentions_with_grad = captured_attentions
            finally:
                for hook in hooks:
                    hook.remove()

        else:
            # HuggingFace model — single pass already returns attentions that
            # are part of the logits computation graph.
            if hasattr(model, "config"):
                model.config.output_attentions = True
                model.config.return_dict = True
            output = model(pixel_values=img, output_attentions=True, return_dict=True)
            logits = output.logits if hasattr(output, 'logits') else output
            attentions_with_grad = list(output.attentions or [])
            if not attentions_with_grad and hasattr(model, 'vit'):
                output2 = model.vit(pixel_values=img, output_attentions=True, return_dict=True)
                attentions_with_grad = list(output2.attentions)
            for attn in attentions_with_grad:
                if attn.requires_grad:
                    attn.retain_grad()

        # Backward pass through the shared computation graph
        target_idx = target.item() if isinstance(target, torch.Tensor) else int(target)
        logits[0, target_idx].backward(retain_graph=True)

        # Relevance = attention * gradient, positive contributions only
        relevance_maps = []
        for attn in attentions_with_grad:
            grad = attn.grad
            if grad is None:
                # Should not happen with the hook-based approach; warn if it does
                print(f"Warning: attn.grad is None for a layer — falling back to ones")
                grad = torch.ones_like(attn)
            else:
                grad = grad.detach()

            # attn shape: (1, num_heads, num_tokens, num_tokens)
            relevance = (attn.detach() * grad).clamp(min=0)
            relevance = relevance.mean(dim=1)  # average over heads → (1, num_tokens, num_tokens)
            relevance_maps.append(relevance.cpu().numpy())

        saliency = self._rollout_relevance(relevance_maps)
        return self._resize_saliency(saliency)

    def _rollout_relevance(self, relevance_maps):
        """
        Aggregate relevance across layers using rollout.

        Args:
            relevance_maps: List of (1, num_tokens, num_tokens) numpy arrays

        Returns:
            np.ndarray: CLS-to-patch relevance, shape (num_patches,)
        """
        num_tokens = relevance_maps[0].shape[-1]
        rollout = np.eye(num_tokens)

        for relevance in relevance_maps:
            rel = relevance.squeeze(0)            # (num_tokens, num_tokens)
            rel = rel + np.eye(num_tokens)        # residual connection
            rel = rel / (rel.sum(axis=-1, keepdims=True) + 1e-9)
            rollout = np.matmul(rel, rollout)     # rel @ rollout: last layer is left-most

        return rollout[0, 1:]  # CLS row, exclude CLS token itself

    def _resize_saliency(self, saliency):
        """
        Reshape flat patch saliency to (input_size, input_size).

        Args:
            saliency: np.ndarray of shape (num_patches,)

        Returns:
            np.ndarray: Normalised saliency map (input_size, input_size)
        """
        patch_grid = int(np.sqrt(saliency.shape[0]))
        saliency_t = torch.tensor(saliency, dtype=torch.float32).reshape(1, 1, patch_grid, patch_grid)
        saliency_t = F.interpolate(saliency_t, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        saliency_resized = saliency_t.squeeze().numpy()

        lo, hi = saliency_resized.min(), saliency_resized.max()
        return (saliency_resized - lo) / (hi - lo + 1e-8)


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
