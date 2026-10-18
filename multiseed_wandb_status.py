"""Status of the multi-seed wandb runs, read by multiseed_run_all.sh to resume: one "<key>\t<status>" line per run key.

  python multiseed_wandb_status.py --project RETFound_MAE --kind train --regex='-s1952/?$'
  python multiseed_wandb_status.py --project RETFound_MAE_XAI_Evaluation --kind eval --regex='-s1952$'

status is 'finished' (finished with test_* metrics) or the state of the newest run (crashed, failed, killed, running).
kind=train keys by run name, kind=eval by '<resume checkpoint>|<xai>' (see util/multiseed.py status_rows).
Exit code 2 (message on stderr) when wandb is missing or the query fails: the caller then falls back to SLURM state.
"""
import argparse
import sys

from util.multiseed import status_rows


def get_args_parser():
    parser = argparse.ArgumentParser('Multi-seed run status from wandb', add_help=True)
    parser.add_argument('--project', required=True, type=str)
    parser.add_argument('--kind', choices=['train', 'eval'], required=True)
    parser.add_argument('--regex', default='.*', type=str, help='wandb run-name filter')
    parser.add_argument('--entity', default='', type=str, help='wandb entity (default: logged-in user)')
    return parser


def main(args):
    try:
        import wandb
    except ImportError:
        print('wandb is not installed in this python', file=sys.stderr)
        return 2
    path = f'{args.entity}/{args.project}' if args.entity else args.project
    try:
        runs = list(wandb.Api().runs(path, filters={'display_name': {'$regex': args.regex}}))
    except Exception as exc:  # no login, no network, unknown project...
        print(f'wandb query of {path} failed: {exc}', file=sys.stderr)
        return 2
    for key, status in sorted(status_rows(runs, args.kind).items()):
        print(f'{key}\t{status}')
    return 0


if __name__ == '__main__':
    sys.exit(main(get_args_parser().parse_args()))
