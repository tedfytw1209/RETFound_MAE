# tests/test_uf_multiseed_split.py
import os

import pandas as pd
import pytest

from dataprocess.uf_multiseed_split import (FNAME_FMT, compare_all_split, make_all_split, run,
                                            split_dir_name, to_binary, tune_eval_split)


def make_df(n_per_label=40, labels=(0, 1), rows_per_patient=2):
    rows = []
    pid = 0
    for label in labels:
        for _ in range(n_per_label):
            for r in range(rows_per_patient):
                rows.append({'patient_id': f'P{pid:04d}', 'folder': f'P{pid:04d}_V{r}',
                             'eye': 'latR' if r == 0 else 'latL', 'label': label,
                             'oct_imgname': f'oct{r}', 'slice_num': 49})
            pid += 1
    return pd.DataFrame(rows)


def test_split_dir_name_keeps_original_for_seed_42():
    assert split_dir_name(0.8, 42) == 'tune8-eval2'
    assert split_dir_name(0.8, 1952) == 'tune8-eval2-s1952'
    assert split_dir_name(0.5, 7) == 'tune5-eval5-s7'


def test_tune_eval_split_is_patient_disjoint_and_stratified():
    df = make_df()
    ft, ev = tune_eval_split(df, 0.8, seed=3)
    assert not set(ft.patient_id) & set(ev.patient_id)
    assert len(ft) + len(ev) == len(df)
    assert ft.groupby('label').patient_id.nunique().to_dict() == {0: 32, 1: 32}


def test_tune_eval_split_depends_only_on_seed():
    df = make_df()
    a, _ = tune_eval_split(df, 0.8, seed=3)
    b, _ = tune_eval_split(df, 0.8, seed=3)
    c, _ = tune_eval_split(df, 0.8, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert set(a.patient_id) != set(c.patient_id)


def test_to_binary_maps_positive_grades_to_one():
    df = make_df(labels=(0, 1, 2, 3, 4))
    out = to_binary(df)
    assert sorted(out.label.unique()) == [0, 1]
    assert (out.label == (df.label > 0).astype(int)).all()


def test_make_all_split_has_disjoint_train_val_test():
    ft, ev = tune_eval_split(make_df(), 0.8, seed=3)
    full = make_all_split(ft, ev, seed=3)
    pats = {s: set(full[full.split == s].patient_id) for s in ('train', 'val', 'test')}
    assert pats['test'] == set(ev.patient_id)
    assert not pats['train'] & pats['val']
    assert len(pats['val']) == 8  # 12.5% of 64 finetune patients
    assert full.OCT.iloc[0].endswith('_25.jpg')


def test_run_writes_binary_variant_and_refuses_overwrite(tmp_path):
    make_df(labels=(0, 1, 2, 3, 4)).to_csv(tmp_path / FNAME_FMT.format('DME'), index=False)
    make_df().to_csv(tmp_path / FNAME_FMT.format('AMD'), index=False)
    written = run(str(tmp_path), seeds=[3], tasks=['DME', 'AMD'])
    assert sorted(os.path.basename(p) for p in written) == [
        'AMD_all_split.csv', 'DME_all_split.csv', 'DME_binary_all_split.csv']
    assert (tmp_path / 'split' / 'tune8-eval2-s3' / FNAME_FMT.format('DME_binary_finetune')).exists()
    with pytest.raises(FileExistsError):
        run(str(tmp_path), seeds=[3], tasks=['AMD'])


def test_compare_all_split_detects_identical_and_changed(tmp_path):
    make_df().to_csv(tmp_path / FNAME_FMT.format('AMD'), index=False)
    a = run(str(tmp_path), seeds=[3], tasks=['AMD'])[0]
    b = run(str(tmp_path), seeds=[4], tasks=['AMD'])[0]
    assert compare_all_split(a, a)['same_rows']
    assert not compare_all_split(a, b)['same_rows']
