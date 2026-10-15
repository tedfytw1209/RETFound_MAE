# tests/test_multiseed_util.py
from util.multiseed import apply_seed_suffix, base_name, seed_suffix


def test_seed_suffix_empty_for_original_run():
    assert seed_suffix(split_seed=42, seed=0, default_seed=0) == ''
    assert seed_suffix(split_seed=42, seed=42, default_seed=42) == ''


def test_seed_suffix_split_and_train_modes():
    assert seed_suffix(split_seed=1952, seed=0, default_seed=0) == '-s1952'
    assert seed_suffix(split_seed=42, seed=1952, default_seed=0) == '-seed1952'


def test_apply_seed_suffix_keeps_trailing_slash():
    assert apply_seed_suffix('A-trsub0--/', 1952, 0, 0) == 'A-trsub0---s1952/'
    assert apply_seed_suffix('A-trsub0--/', 42, 0, 0) == 'A-trsub0--/'
    assert apply_seed_suffix('X-seed0', 1952, 0, 0) == 'X-seed0-s1952'


def test_base_name_strips_seed_tokens():
    assert base_name('A-trsub0--/') == 'A-trsub0--'
    assert base_name('A-trsub0---s1952/') == 'A-trsub0--'
    assert base_name('A-trsub0---seed1952/') == 'A-trsub0--'
    assert base_name('DME-v5-m-hirescam-224-blk-1-seed0-segmask-s1952') == 'DME-v5-m-hirescam-224-blk-1-segmask'
    assert base_name('DME-v5-m-hirescam-224-blk-1-seed1952-segmask') == 'DME-v5-m-hirescam-224-blk-1-segmask'


from types import SimpleNamespace

import pytest

from util.multiseed import rows_from_runs, summarize


def _run(name, auc, seed=0, split_seed=None, created='2026-10-01'):
    config = {'seed': seed} if split_seed is None else {'seed': seed, 'split_seed': split_seed}
    return SimpleNamespace(name=name, config=config, created_at=created,
                           summary={'test_roc_auc': auc, 'test_note': 'x', 'best_val_roc_auc': 0.1})


RUNS = [_run('A--/', 0.80),                                  # original (no split_seed in old config)
        _run('A---s1952/', 0.90, split_seed=1952),
        _run('A---s1952/', 0.10, split_seed=1952, created='2026-09-01'),  # older duplicate, dropped
        _run('A---seed1952/', 0.70, seed=1952, split_seed=42)]


def test_rows_from_runs_split_mode():
    rows = rows_from_runs(RUNS, mode='split', orig_seed=0)
    assert sorted(rows.replicate) == [42, 1952]
    assert set(rows.group) == {'A--'}
    assert sorted(rows.test_roc_auc) == [0.8, 0.9]
    assert 'test_note' not in rows and 'best_val_roc_auc' not in rows


def test_rows_from_runs_train_mode():
    rows = rows_from_runs(RUNS, mode='train', orig_seed=0)
    assert sorted(rows.replicate) == [0, 1952]
    assert sorted(rows.test_roc_auc) == [0.7, 0.8]


def test_summarize_mean_std():
    summary = summarize(rows_from_runs(RUNS, mode='split', orig_seed=0), expect=5)
    row = summary.iloc[0]
    assert row['n_seeds'] == 2 and row['replicates'] == '42 1952'
    assert row['test_roc_auc_mean'] == pytest.approx(0.85)
