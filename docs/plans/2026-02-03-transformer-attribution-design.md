# Transformer Attribution XAI Method

**Date**: 2026-02-03
**Status**: Approved for implementation

## Summary

Add Transformer Attribution (Chefer et al., 2021) as a new XAI method for generating class-specific saliency maps. This method combines gradient information with attention weights through relevance propagation, producing more faithful explanations than pure attention rollout.

## Motivation

Current attention rollout shows where the model "looks" but ignores the classification target. Transformer Attribution incorporates gradients to show which regions are relevant for a *specific class prediction*, making it more suitable for medical diagnosis explanations where clinicians need to understand why a particular condition was predicted.

## Design

### CLI Interface

```bash
python main_XAI_evaluation.py --xai tf_attr ...
```

### New File: `baselines/TransformerAttribution.py`

```python
class TransformerAttribution(torch.nn.Module):
    def __init__(self, model, model_name, input_size, N=12, device=None):
        # Store model, set up hooks for attention + gradients

    def forward(self, inputs=None, targets=None, model=None, **kwargs):
        # Returns: np.ndarray of shape (B, input_size, input_size)
```

### Algorithm

1. Register hooks to capture attention weights from all transformer layers
2. Forward pass to get logits
3. Backward pass on target class logit to get attention gradients
4. Compute relevance: `R = (attention * gradient).clamp(min=0)`
5. Propagate relevance through layers using rollout-style aggregation
6. Reshape from patch tokens to spatial map, resize to `input_size`
7. Normalize to [0, 1] range

### Model Support

- RETFound (timm ViT) - primary target
- HuggingFace ViT - secondary support

### Integration

Modify `main_XAI_evaluation.py`:

```python
from baselines.TransformerAttribution import TransformerAttribution

# In XAI method selection
elif args.xai == 'tf_attr':
    explain_func = TransformerAttribution(
        model=model,
        model_name=args.model,
        input_size=args.input_size,
        N=12,
        device=device
    )
```

### Evaluation Compatibility

- Works with insertion/deletion metrics (model-dependent)
- Works with relevance metrics (mass, rank)
- Works with OCT layer-importance analysis (if segmentation masks provided)
- No changes needed to `util/evaluation.py`

## Files to Change

| File | Change |
|------|--------|
| `baselines/TransformerAttribution.py` | New file (~100 lines) |
| `main_XAI_evaluation.py` | Add import + elif clause (~5 lines) |

## Dependencies

None - uses PyTorch hooks, same as existing methods.

## References

- Chefer, H., Gur, S., & Wolf, L. (2021). Transformer Interpretability Beyond Attention Visualization. CVPR 2021.