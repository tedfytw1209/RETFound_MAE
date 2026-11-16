# util/adml_tags.py
"""wandb tags of the OCT AD ML experiments (OCT_ADML_experiment.xlsx), so runs can be filtered by study, method, period.

Each run carries, in this order:
  study2 | study3                                   study2 = AD/MCI detection, study3 = prediction 1/3/5/10 years ahead
  DL | ML | our_proposed                            relative DL baselines (Jacqueline, Wisely, Wisely2, Mahendran,
                                                    Hebei) | relative ML baselines (Wang, NunesAna, MeiXi, JinZ) |
                                                    our feature selection + ML
  year1 | year3 | year5 | year10 | ...              study3 only: the prediction period (old runs used 1yr, 3yr, ...)
  ad_control | mci_control | ad_mci_control | ad_rest   the task, when it can be told
followed by any other tag the run already had (class_weight, rerun, ...).
The study, period and task come from the data / split paths and the run name, e.g. IRB2024v5_ADCON_DL_study3_retinaf_1yr
or IRB2024_combined_study3_retinaf_diseaseclean_10yr/split_mcicon; when they can be told they replace a conflicting tag
the run was given by hand (a study2 run never keeps a period tag).
Used by main_finetune_Chua_Jacqueline.py (new DL runs) and adml_wandb_retag.py (existing DL / ML / our_proposed runs).
"""
import re

STUDIES = ('study2', 'study3')
METHODS = ('DL', 'ML', 'our_proposed')
TASKS = ('ad_rest', 'ad_mci_control', 'mci_control', 'ad_control')
# leftmost match wins, so ad_mci_control is found before the mci_control inside it
_TASK_RE = re.compile(r'(?<![a-z])(ad_rest|ad_mci_control|mci_control|ad_control)(?![a-z])')
_PERIOD_RE = re.compile(r'(?<![0-9a-z])(?:(\d+)yr|year(\d+))(?![0-9a-z])')
_PERIOD_TAG_RE = re.compile(r'^(?:(\d+)yr|year(\d+))$')
_DL_RE = re.compile(r'-(Jacqueline|Wisely2?|Mahendran|Hebei)-')
_ML_PREFIXES = ('WangXin', 'Wang_', 'NunesAna', 'Nunes', 'Mei_Xi', 'MeiXi', 'JinZ')
_OUR_PREFIXES = ('TY_ML',)


def study_of(text):
    """study3 / study2 from a path or run name (study2 DL data folders are IRB2024v5_ADCON_DL_data_*), else None."""
    if 'study3' in text:
        return 'study3'
    if 'study2' in text or 'ADCON_DL_data' in text:
        return 'study2'
    return None


def period_of(text):
    """'year<N>' from a ..._<N>yr... / year<N> token, else None."""
    m = _PERIOD_RE.search(text)
    return f'year{m.group(1) or m.group(2)}' if m else None


def task_of(text):
    m = _TASK_RE.search(text or '')
    return m.group(1) if m else None


def method_of_name(name):
    """DL / ML / our_proposed from a wandb run name, else None."""
    if name.startswith(_OUR_PREFIXES):
        return 'our_proposed'
    if name.startswith(_ML_PREFIXES):
        return 'ML'
    if _DL_RE.search(name):
        return 'DL'
    return None


def build_tags(existing, study=None, method=None, period=None, task=None):
    """[study, method, period, task] (the known ones) + the other existing tags; old '<N>yr' tags become 'year<N>'."""
    if study == 'study2':
        period = None
    rest = []
    for tag in existing or ():
        tag = str(tag).strip()
        if not tag:
            continue
        m = _PERIOD_TAG_RE.match(tag)
        if m:
            tag = f'year{m.group(1) or m.group(2)}'
            if period or study == 'study2':
                continue
        elif (tag in STUDIES and study) or (tag in METHODS and method) or (tag in TASKS and task):
            continue
        if tag not in rest:
            rest.append(tag)
    head = [t for t in (study, method, period, task) if t]
    return head + [t for t in rest if t not in head]


def tags_for(text, existing=(), method=None, task=None):
    """Tags of a run whose paths / name are in `text` (task: pass it when text could name another, e.g. ad_rest)."""
    return build_tags(existing, study_of(text), method, period_of(text), task or task_of(text))
