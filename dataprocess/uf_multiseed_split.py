# dataprocess/uf_multiseed_split.py
"""Multi-seed patient-level splits for the UF IRB2024_v5_all benchmark.

Ports OphFoundation data_process/benchmark_split.py + benchmark_split_add.py (tune/eval split)
and the Label_process.ipynb merge step (train/val/test *_all_split.csv) with the seed exposed.
Seed 42 reproduces split/tune8-eval2; seed k writes split/tune8-eval2-s{k}.

  python dataprocess/uf_multiseed_split.py --seeds 1952 5235 8234 8386
  python dataprocess/uf_multiseed_split.py --seeds 42 --out_root /tmp/split_check \
      --verify_dir /orange/.../IRB2024_v5_all/split/tune8-eval2
"""
import argparse
import os
import random

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = '/orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/IRB2024_v5_all/'
FNAME_FMT = 'test_matched_oct_fundus_patient_{}.csv'
# (task, also write a label>0 binary copy): benchmark_split.py tasks, then benchmark_split_add.py tasks
TASKS = [('AMD', False), ('Cataract', False), ('DR', True), ('Glaucoma', True),
         ('DME', True), ('CSR', False), ('Drusen', False), ('ERM', False),
         ('MH', False), ('CRVO_CRAO', False), ('PVD', False), ('RNV', False)]
VAL_SIZE = 0.125
ORIGINAL_SEED = 42
DEFAULT_SEEDS = [1952, 5235, 8234, 8386]  # random.Random(2026).sample(range(1, 10000), 4)


def split_dir_name(train_ratio, seed):
    tune = int(train_ratio * 10)
    base = f'tune{tune}-eval{10 - tune}'
    return base if seed == ORIGINAL_SEED else f'{base}-s{seed}'


def tune_eval_split(df, train_ratio, seed):
    """Per-label patient split, identical to benchmark_split.analyze_and_split_csv."""
    patient_info = df.groupby('patient_id').first().reset_index()
    label_counts = patient_info['label'].value_counts().to_dict()
    train_ids, test_ids = [], []
    random.seed(seed)
    for label in label_counts:
        ids = patient_info[patient_info['label'] == label]['patient_id'].tolist()
        random.shuffle(ids)
        split_point = int(train_ratio * len(ids))
        train_ids.extend(ids[:split_point])
        test_ids.extend(ids[split_point:])
    train_records = df[df['patient_id'].isin(train_ids)].to_dict('records')
    test_records = df[df['patient_id'].isin(test_ids)].to_dict('records')
    random.Random(seed).shuffle(train_records)
    random.Random(seed).shuffle(test_records)
    return pd.DataFrame(train_records), pd.DataFrame(test_records)


def to_binary(df):
    out = df.copy()
    out['label'] = out['label'].apply(lambda x: 1 if int(x) > 0 else 0)
    return out


def _majority_label(g):
    nonzero = g[g['label'] != 0]['label']
    return nonzero.mode().iloc[0] if len(nonzero) > 0 else 0


def _add_oct_path(df):
    # Label_process.ipynb preprocess_columns; skipped when the source CSV lacks the raw columns
    if not {'oct_imgname', 'slice_num', 'eye', 'folder'}.issubset(df.columns):
        return df
    df['depth'] = df['slice_num']
    df['OCT'] = [f"{r.folder}/{r.oct_imgname}_{r.eye}_{(r.depth + 1) // 2}.jpg" for r in df.itertuples()]
    return df


def make_all_split(finetune_df, eval_df, seed):
    """Label_process.ipynb process_disease_data with random_state=seed."""
    patient_labels = finetune_df.groupby('patient_id').apply(_majority_label).reset_index()
    patient_labels.columns = ['patient_id', 'patient_label']
    patients = patient_labels['patient_id'].values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=seed)
    train_idx, val_idx = next(sss.split(patients, patient_labels['patient_label'].values))
    train_df = finetune_df[finetune_df['patient_id'].isin(set(patients[train_idx]))].copy()
    val_df = finetune_df[finetune_df['patient_id'].isin(set(patients[val_idx]))].copy()
    test_df = eval_df.copy()
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    return _add_oct_path(pd.concat([train_df, val_df, test_df], ignore_index=True))


def compare_all_split(new_path, ref_path, keys=('patient_id', 'folder', 'label', 'split')):
    new, ref = pd.read_csv(new_path), pd.read_csv(ref_path)
    cols = [c for c in keys if c in new.columns and c in ref.columns]
    a, b = new[cols].astype(str), ref[cols].astype(str)
    return {'rows_new': len(new), 'rows_ref': len(ref),
            'same_rows': a.sort_values(cols).reset_index(drop=True).equals(
                b.sort_values(cols).reset_index(drop=True)),
            'same_order': a.reset_index(drop=True).equals(b.reset_index(drop=True))}


def run(root, seeds, train_ratio=0.8, tasks=None, out_root=None, overwrite=False):
    out_root = out_root or os.path.join(root, 'split')
    written = []
    for seed in seeds:
        split_dir = os.path.join(out_root, split_dir_name(train_ratio, seed))
        os.makedirs(split_dir, exist_ok=True)
        for task, has_binary in TASKS:
            if tasks and task not in tasks:
                continue
            src = os.path.join(root, FNAME_FMT.format(task))
            if not os.path.exists(src):
                print(f'[skip] {src} not found')
                continue
            finetune_df, eval_df = tune_eval_split(pd.read_csv(src), train_ratio, seed)
            variants = [(task, finetune_df, eval_df)]
            if has_binary:
                variants.append((f'{task}_binary', to_binary(finetune_df), to_binary(eval_df)))
            for name, ft, ev in variants:
                out_path = os.path.join(split_dir, f'{name}_all_split.csv')
                if os.path.exists(out_path) and not overwrite:
                    raise FileExistsError(f'{out_path} exists; pass --overwrite to replace it')
                ft_path = os.path.join(split_dir, FNAME_FMT.format(f'{name}_finetune'))
                ev_path = os.path.join(split_dir, FNAME_FMT.format(f'{name}_evaluation'))
                ft.to_csv(ft_path, index=False)
                ev.to_csv(ev_path, index=False)
                # re-read like the notebook did, so dtypes match the original pipeline
                full = make_all_split(pd.read_csv(ft_path), pd.read_csv(ev_path), seed)
                full.to_csv(out_path, index=False)
                written.append(out_path)
                print(f'{out_path}: {full.groupby("split")["label"].value_counts().to_dict()}')
    return written


def get_args_parser():
    parser = argparse.ArgumentParser('UF IRB2024_v5_all multi-seed splits', add_help=True)
    parser.add_argument('--root', default=ROOT, type=str,
                        help='folder holding test_matched_oct_fundus_patient_<TASK>.csv')
    parser.add_argument('--out_root', default='', type=str, help='defaults to <root>/split')
    parser.add_argument('--seeds', default=DEFAULT_SEEDS, type=int, nargs='+')
    parser.add_argument('--train_ratio', default=0.8, type=float)
    parser.add_argument('--tasks', default=[], type=str, nargs='*')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verify_dir', default='', type=str,
                        help='compare outputs against an existing split dir, e.g. .../split/tune8-eval2')
    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    written = run(args.root, args.seeds, args.train_ratio, args.tasks, args.out_root or None, args.overwrite)
    if args.verify_dir:
        failed = 0
        for path in written:
            ref = os.path.join(args.verify_dir, os.path.basename(path))
            if not os.path.exists(ref):
                print(f'[verify] no reference {ref}')
                continue
            result = compare_all_split(path, ref)
            print(f'[verify] {os.path.basename(path)}: {result}')
            failed += not result['same_rows']
        if failed:
            raise SystemExit(f'{failed} file(s) differ from {args.verify_dir}')
