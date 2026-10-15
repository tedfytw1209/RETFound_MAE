# aggregate_multiseed.py
"""Mean/std across seeds from the wandb records of main_finetune.py, main_finetune_smp.py, main_XAI_evaluation.py.

  python aggregate_multiseed.py --kind baseline --mode split --name_regex '^DME_binary_all_split-IRB2024_v5_all' \
      --expect 5 --out results/multiseed_split_baseline_DME.csv
  python aggregate_multiseed.py --kind xai --mode split --name_regex '^DME_binary_all_split' \
      --expect 5 --out results/multiseed_split_xai_DME.csv
  python aggregate_multiseed.py --kind baseline --mode train --name_regex '^DME_all-OCTDL' --out results/multiseed_train_OCTDL.csv
"""
import argparse
import os

import wandb

from util.multiseed import rows_from_runs, summarize

# kind -> (wandb project, training seed the original runs used = the script's --seed default)
KINDS = {'baseline': ('RETFound_MAE', 0),
         'smp': ('RETFound_MAE_XAI', 42),
         'xai': ('RETFound_MAE_XAI_Evaluation', 0)}


def get_args_parser():
    parser = argparse.ArgumentParser('Multi-seed aggregation from wandb', add_help=True)
    parser.add_argument('--kind', choices=sorted(KINDS), required=True)
    parser.add_argument('--mode', choices=['split', 'train'], default='split')
    parser.add_argument('--name_regex', default='.*', type=str, help='wandb run-name filter')
    parser.add_argument('--entity', default='', type=str, help='wandb entity (default: logged-in user)')
    parser.add_argument('--expect', default=0, type=int, help='warn when a configuration has fewer seeds')
    parser.add_argument('--out', default='multiseed_summary.csv', type=str)
    return parser


def main(args):
    project, orig_seed = KINDS[args.kind]
    path = f'{args.entity}/{project}' if args.entity else project
    runs = wandb.Api().runs(path, filters={'display_name': {'$regex': args.name_regex}, 'state': 'finished'})
    rows = rows_from_runs(runs, args.mode, orig_seed)
    if rows.empty:
        raise SystemExit(f'no finished runs in {path} matching {args.name_regex}')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    runs_path = os.path.splitext(args.out)[0] + '_runs.csv'
    rows.to_csv(runs_path, index=False)
    summarize(rows, args.expect).to_csv(args.out, index=False)
    print(f'wrote {args.out} and {runs_path}')


if __name__ == '__main__':
    main(get_args_parser().parse_args())
