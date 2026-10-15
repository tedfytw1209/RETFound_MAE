# util/multiseed.py
"""Multi-seed run naming shared by main_finetune*.py, main_XAI_evaluation.py and aggregate_multiseed.py.

A suffix is added only when a seed differs from the original single-seed run, so the
original run name / output dir / wandb run name never change:
  split mode: --split_seed k (data from split/tune8-eval2-s{k})  -> <task>-s{k}
  train mode: --seed k (script default otherwise)                 -> <task>-seed{k}
"""
import re

import pandas as pd

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


META_COLS = ['run', 'group', 'replicate', 'split_seed', 'seed']


def rows_from_runs(runs, mode, orig_seed):
    """One row of numeric test_* summary metrics per wandb run (latest run per name).

    mode='split': runs at the original training seed (replicate = split_seed)
    mode='train': runs on the original split (replicate = training seed)
    """
    latest = {}
    for run in runs:
        split_seed = int(run.config.get('split_seed', ORIGINAL_SPLIT_SEED))
        seed = int(run.config.get('seed', orig_seed))
        if mode == 'split' and seed != orig_seed:
            continue
        if mode == 'train' and split_seed != ORIGINAL_SPLIT_SEED:
            continue
        if run.name in latest and latest[run.name][0].created_at >= run.created_at:
            continue
        latest[run.name] = (run, split_seed, seed)
    rows = []
    for name, (run, split_seed, seed) in latest.items():
        summary = getattr(run.summary, '_json_dict', run.summary)
        row = {k: v for k, v in summary.items()
               if k.startswith('test_') and isinstance(v, (int, float)) and not isinstance(v, bool)}
        row.update(run=name, group=base_name(name), split_seed=split_seed, seed=seed,
                   replicate=split_seed if mode == 'split' else seed)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(rows, expect=0):
    metric_cols = [c for c in rows.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(rows[c])]
    grouped = rows.groupby('group')
    summary = grouped[metric_cols].agg(['mean', 'std'])
    summary.columns = [f'{m}_{s}' for m, s in summary.columns]
    summary.insert(0, 'n_seeds', grouped.size())
    summary.insert(1, 'replicates', grouped['replicate'].apply(lambda s: ' '.join(str(x) for x in sorted(s))))
    if expect:
        for group, n in summary['n_seeds'].items():
            if n < expect:
                print(f'[multiseed] {group}: {n}/{expect} seeds')
    return summary.reset_index()
