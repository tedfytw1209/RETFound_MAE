# util/multiseed.py
"""Multi-seed run naming shared by main_finetune*.py, main_XAI_evaluation.py and aggregate_multiseed.py.

A suffix is added only when a seed differs from the original single-seed run, so the
original run name / output dir / wandb run name never change:
  split mode: --split_seed k (data from split/tune8-eval2-s{k})  -> <task>-s{k}
  train mode: --seed k (script default otherwise)                 -> <task>-seed{k}
"""
import re

ORIGINAL_SPLIT_SEED = 42
_SPLIT_TOKEN_RE = re.compile(r'-s\d+$')
_SEED_TOKEN_RE = re.compile(r'-seed\d+')


def seed_suffix(split_seed, seed, default_seed):
    suffix = ''
    if split_seed != ORIGINAL_SPLIT_SEED:
        suffix += f'-s{split_seed}'
    if seed != default_seed:
        suffix += f'-seed{seed}'
    return suffix


def apply_seed_suffix(task, split_seed, seed, default_seed):
    trailing = '/' if task.endswith('/') else ''
    return task.rstrip('/') + seed_suffix(split_seed, seed, default_seed) + trailing


def base_name(run_name):
    """Run name with the seed tokens removed: the key shared by all replicates of one configuration."""
    name = _SPLIT_TOKEN_RE.sub('', run_name.rstrip('/'))
    return _SEED_TOKEN_RE.sub('', name, count=1)
