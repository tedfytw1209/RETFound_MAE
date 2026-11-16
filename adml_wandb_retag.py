"""Fix the wandb tags of existing OCT AD runs: study2|study3, DL|ML|our_proposed, year<N> (study3), task.

  python adml_wandb_retag.py --project OCTAD_Relatives                     # dry run: print what would change
  python adml_wandb_retag.py --project OCTAD_Relatives --project <ML project> --apply

The tags follow util/adml_tags.py (the same rule main_finetune_Chua_Jacqueline.py applies to new DL runs). They are
derived from the run name, plus config.split_dir / config.data_path when the run logged them (the DL runs do):
  DL            name contains -Jacqueline- / -Wisely- / -Wisely2- / -Mahendran- / -Hebei-
  ML            name starts with WangXin / NunesAna / Mei_Xi / MeiXi / JinZ
  our_proposed  name starts with TY_ML
Other tags are kept; old period tags (1yr, 3yr, ...) become year1, year3, ...; a conflicting study/method/period/task
tag is replaced. Runs whose method cannot be told from the name are listed and left alone, as are runs whose study cannot
be told (only their method and old period tags are fixed). Without --apply nothing is written to wandb.
"""
import argparse
import sys
from collections import Counter

from util import adml_tags


def get_args_parser():
    parser = argparse.ArgumentParser('Fix the wandb tags of the OCT AD runs', add_help=True)
    parser.add_argument('--project', action='append', required=True, help='wandb project (repeat for several)')
    parser.add_argument('--entity', default='', type=str, help='wandb entity (default: logged-in user)')
    parser.add_argument('--regex', default='.*', type=str, help='only runs whose name matches')
    parser.add_argument('--apply', action='store_true', help='write the new tags (default: dry run)')
    return parser


def new_tags(name, config, tags):
    """(method, study, new tag list) of one run; method None = not an OCT AD run this tool knows."""
    method = adml_tags.method_of_name(name)
    if method is None:
        return None, None, list(tags)
    text = f"{config.get('split_dir') or ''} {config.get('data_path') or ''} {name}"
    # the run name / task names the task; data_path can name another (ad_rest reads the ad_mci_control CSV)
    task = adml_tags.task_of(config.get('task') or '') or adml_tags.task_of(name) or adml_tags.task_of(text)
    return method, adml_tags.study_of(text), adml_tags.tags_for(text, tags, method=method, task=task)


def main(args):
    try:
        import wandb
    except ImportError:
        print('wandb is not installed in this python', file=sys.stderr)
        return 2
    api = wandb.Api()
    counts = Counter()
    for project in args.project:
        path = f'{args.entity}/{project}' if args.entity else project
        try:
            runs = list(api.runs(path, filters={'display_name': {'$regex': args.regex}}))
        except Exception as exc:  # no login, no network, unknown project...
            print(f'wandb query of {path} failed: {exc}', file=sys.stderr)
            return 2
        for run in runs:
            config = {k: v for k, v in run.config.items() if isinstance(v, str)}
            method, study, tags = new_tags(run.name, config, list(run.tags))
            if method is None:
                counts['unrecognized'] += 1
                print(f'[unrecognized] {path}: {run.name} ({run.id}) tags={list(run.tags)}')
                continue
            if study is None:
                counts['no study'] += 1
                print(f'[no study] {path}: {run.name} ({run.id}): study2/study3 not in its name or paths')
            if tags == list(run.tags):
                counts['unchanged'] += 1
                continue
            counts['changed'] += 1
            print(f"[{'retag' if args.apply else 'would retag'}] {path}: {run.name} ({run.id}): {list(run.tags)} -> {tags}")
            if args.apply:
                run.tags = tags
                run.update()
    print(f"[adml_wandb_retag] {counts['changed']} {'retagged' if args.apply else 'to retag (dry run, --apply to write)'}, "
          f"{counts['unchanged']} already right, {counts['no study']} without a study, "
          f"{counts['unrecognized']} not recognized")
    return 0


if __name__ == '__main__':
    sys.exit(main(get_args_parser().parse_args()))
