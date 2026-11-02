"""Standalone inference for the trained RETFound and SMP-fuse OCT classifiers:
class probabilities plus CAM saliency maps. Depends on nothing outside this folder."""

from .model import LoadedModel, list_bundled_models, load_model  # noqa: F401
from .pipeline import collect_items, run_inference  # noqa: F401
