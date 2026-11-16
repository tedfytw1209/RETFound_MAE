"""Per-fold status of the OCT AD relative-model CV runs in wandb, read by adml_dl_run_all.sh to resume.

  python adml_dl_wandb_status.py --project OCTAD_Relatives --regex='_detect_data-IRB2024v5_ADCON_DL'

Prints one "<run name>\t<cv fold>\t<status>" line per (run name, fold). All folds of one CV job share a wandb run name:
main_finetune_Chua_Jacqueline.py appends _cv<k>fold<i> to args.task only after wandb.init, so it reaches the output dir
but not the run name; the fold is read from config.cv_fold instead.
status is 'finished' (ended normally with numeric test_* metrics, util.multiseed.is_finished) or the state of the
newest run of that fold (running, crashed, failed, killed, or no_test_metrics for a finished run without test_*).
Exit code 2 (message on stderr) when wandb is missing or the query fails: the caller then uses the disk/SLURM state.
"""
import argparse
import sys

from util.multiseed import is_finished


def get_args_parser():
    parser = argparse.ArgumentParser('OCT AD relative-model CV fold status from wandb', add_help=True)
    parser.add_argument('--project', default='OCTAD_Relatives', type=str)
    parser.add_argument('--regex', default='.*', type=str, help='wandb run-name filter')
    parser.add_argument('--entity', default='', type=str, help='wandb entity (default: logged-in user)')
    return parser


def fold_status_rows(runs):
    """(run name, fold) -> 'finished' | state of the newest run of that fold. Runs that are not CV folds are ignored."""
    newest, finished = {}, set()
    for run in runs:
        config = run.config
        if int(config.get('cv_folds', 0) or 0) <= 1 or config.get('cv_fold') is None:
            continue
        key = (run.name, int(config['cv_fold']))
        if is_finished(run):
            finished.add(key)
        if key not in newest or run.created_at > newest[key].created_at:
            newest[key] = run
    return {key: 'finished' if key in finished else ('no_test_metrics' if run.state == 'finished' else run.state)
            for key, run in newest.items()}


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
    for (name, fold), status in sorted(fold_status_rows(runs).items()):
        print(f'{name}\t{fold}\t{status}')
    return 0


if __name__ == '__main__':
    sys.exit(main(get_args_parser().parse_args()))
