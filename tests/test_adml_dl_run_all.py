# tests/test_adml_dl_run_all.py
"""adml_dl_run_all.sh: dry runs with sbatch/torchrun stubbed out, and the wandb per-fold status of adml_dl_wandb_status.py."""
import os
import re
import shlex
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from adml_dl_wandb_status import fold_status_rows

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUBS = ('torchrun(){ echo "CMD torchrun $*"; }; python(){ echo "CMD python $*"; }; '
         'sbatch(){ echo "CMD sbatch $*"; }; module(){ :; }; conda(){ :; }; '
         'export -f torchrun python sbatch module conda; ')
needs_bash = pytest.mark.skipif(os.name == 'nt' or shutil.which('bash') is None,
                                reason='CRLF checkout on Windows; run on the cluster')


def _run(i, name, fold, state='finished', test=True, folds=10):
    summary = {'test_roc_auc': 0.6} if test else {}
    return SimpleNamespace(name=name, state=state, summary=summary, created_at=f'2026-01-{i:02d}',
                           config={'cv_folds': folds, 'cv_fold': fold})


def test_fold_status_rows():
    rows = fold_status_rows([
        _run(1, 'a', 0), _run(2, 'a', 0, state='crashed'),  # finished once stays finished
        _run(3, 'a', 1, state='crashed'), _run(4, 'a', 1, state='running'),  # newest state wins
        _run(5, 'a', 2, test=False),  # finished without test_* -> nothing to report
        _run(6, 'b', 0, folds=0),  # not a CV fold
    ])
    assert rows == {('a', 0): 'finished', ('a', 1): 'running', ('a', 2): 'no_test_metrics'}


def _env(tmp_path, **extra):
    env = {k: v for k, v in os.environ.items() if k not in ('PERIOD', 'CV_FOLDS_ONLY', 'MODELS', 'TASKS', 'VARIANTS')}
    env.update({'SLURM_JOBID': '1234', 'WANDB_CHECK': '0', 'CHECK_DATA': '0', 'DRY_RUN': '1',
                'RESULT_DIRS': str(tmp_path / 'results'), 'REF_FILE': str(tmp_path / 'ref.sh'), **extra})
    return env


def driver(tmp_path, stubs='', **extra):
    proc = subprocess.run(['bash', '-c', STUBS + stubs + 'bash adml_dl_run_all.sh'], cwd=REPO,
                          env=_env(tmp_path, **extra), capture_output=True, text=True, check=True)
    cmds = [line for line in proc.stdout.splitlines() if ' sbatch --job-name=' in line]
    return cmds, proc.stderr


def task_of(cmd_line, **env):
    """Run one printed sbatch line's training script with torchrun stubbed: the --task values and the env it got."""
    words = shlex.split(cmd_line)
    assigns = dict(w.split('=', 1) for w in words[:words.index('sbatch')])
    script_args = words[words.index('sbatch') + 2:]
    out = subprocess.run(['bash', '-c', STUBS + 'bash "$@"', 'x', *script_args], cwd=REPO,
                         env={**os.environ, 'SLURM_JOBID': '1234', **assigns, **env},
                         capture_output=True, text=True, check=True).stdout
    calls = [l for l in out.splitlines() if l.startswith('CMD torchrun')]
    return [re.search(r'--task (\S+)', c).group(1) for c in calls], [int(re.search(r'--cv_fold (\d+)', c).group(1)) for c in calls], out


@needs_bash
def test_todo_jobs_and_run_names_match_the_scripts(tmp_path):
    cmds, _ = driver(tmp_path, MAX_SLURM='0')
    assert len(cmds) == 3 + 4 * 5
    ref = (tmp_path / 'ref.sh').read_text()
    names = re.findall(r'^# adml-\S+ \[(\S+)\]', ref, flags=re.M)
    assert len(names) == len(cmds)
    for cmd, name in zip(cmds, names):
        tasks, folds, _ = task_of(cmd)
        assert folds == list(range(10))
        # the trainer appends _CW for --class_weight
        assert {t + ('_CW' if '--class_weight' in cmd else '') for t in tasks} == {name}


# the DL wandb names written in OCT_ADML_experiment (1).xlsx (Study3-disease N8:N12, without "  [10 fold runs]")
SHEET_NAMES = {
    'Jacqueline': 'ad_mci_control_detect_data-IRB2024v5_ADCON_DL_study3_retinaf_1yr-Jacqueline-convnext_tiny-Thickness-'
                  'bs64ep100lr1e-3wd5e-4-roc_auceval-subset0_CW',
    'Wisely': 'ad_mci_control_detect_data-IRB2024v5_ADCON_DL_study3_retinaf_1yr-Wisely-resnet18_paper-Thickness-'
              'roc_auceval-subset0_CW',
    'Mahendran': 'ad_mci_control_detect_data-IRB2024v5_ADCON_DL_study3_retinaf_1yr-Mahendran-ad_oct_model-OCT-'
                 'bs18ep100lr7e-5wd1e-2-roc_auceval-subset0_CW',
    'Wisely2': 'ad_mci_control_detect_data-IRB2024v5_ADCON_DL_study3_retinaf_1yr-Wisely2-dual_input_cnn-images_only-'
               'Thickness-roc_auceval-subset0_CW',
    'Hebei': 'ad_mci_control_detect_data-IRB2024v5_ADCON_DL_study3_retinaf_1yr-Hebei-ducan-dual-bs8ep400lr3e-4wd1e-2-'
             'accuracyeval-subset0_CW',
}


@needs_bash
@pytest.mark.parametrize('model', sorted(SHEET_NAMES))
def test_run_names_are_the_sheet_names(model):
    """The driver checks wandb by the names the runs already have: same pattern as the sheet, unchanged."""
    out = subprocess.run(['bash', '-c', 'eval "$(sed -n "/^run_name() {/,/^}/p" adml_dl_run_all.sh)"; '
                          f'run_name {model} ad_mci_control_detect_data IRB2024v5_ADCON_DL_study3_retinaf_1yr'],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    assert out + '_CW' == SHEET_NAMES[model]


def _wandb_rows(tmp_path, text):
    rows = tmp_path / 'wb.tsv'
    rows.write_text(text)
    status = tmp_path / 'status.sh'
    status.write_text(f'cat {rows.as_posix()}\n')
    return f'bash {status.as_posix()}'


@needs_bash
def test_wandb_unreadable_stops(tmp_path):
    proc = subprocess.run(['bash', '-c', STUBS + 'bash adml_dl_run_all.sh'], cwd=REPO,
                          env=_env(tmp_path, WANDB_CHECK='1', WANDB_STATUS_CMD='false'), capture_output=True, text=True)
    assert proc.returncode == 1 and 'cannot be read' in proc.stderr and 'sbatch' not in proc.stdout


@needs_bash
def test_wandb_name_decides_not_the_result_folders(tmp_path):
    name = ('ad_rest_detect_data-IRB2024v5_ADCON_DL_data_retinaf_diseaseclean-Wisely-resnet18_paper-Thickness-'
            'roc_auceval-subset0_CW')
    for k in range(10):  # folders say all done, wandb has folds 0-7 only
        d = tmp_path / 'results' / f'{name}_cv10fold{k}'
        d.mkdir(parents=True)
        (d / 'metrics_test.csv').write_text('x\n')
    cmd = _wandb_rows(tmp_path, ''.join(f'{name}\t{k}\tfinished\n' for k in range(8)) + f'{name}\t9\tfailed\n')
    cmds, err = driver(tmp_path, MODELS='Wisely', TASKS='ad_rest', WANDB_CHECK='1', WANDB_STATUS_CMD=cmd)
    assert len(cmds) == 1 and 'CV_FOLDS_ONLY="8 9"' in cmds[0]
    assert f'{name}: finished folds 0 1 2 3 4 5 6 7, not finished: 9=failed' in err
    cmds, err = driver(tmp_path, MODELS='Jacqueline', TASKS='ad_rest', WANDB_CHECK='1', WANDB_STATUS_CMD=cmd)
    assert len(cmds) == 1 and 'no run named ad_rest_detect_data-' in err


@needs_bash
def test_wandb_tags_follow_the_existing_runs(tmp_path, monkeypatch):
    """'<study>,<task>[,<period>]' as the earlier runs (e.g. study3,ad_mci_control,10yr); class_weight is the trainer's."""
    monkeypatch.setenv('PERIOD', '3yr')  # left in the shell: must not reach a study2 run
    cmds, _ = driver(tmp_path, MAX_SLURM='0')
    tags = {}
    for cmd in cmds:
        job = re.search(r'--job-name=(\S+)', cmd).group(1)
        _, _, out = task_of(cmd)
        tags[job] = set(re.findall(r'--wandb_tags (\S+)', out))
    assert tags['adml-s2-clean-ad_rest-Wisely-cw'] == {'study2,ad_rest'}
    assert tags['adml-s3-1yr-mci_control-Jacqueline-cw'] == {'study3,mci_control,1yr'}
    assert tags['adml-s3-10yr-mci_control-Hebei-cw'] == {'study3,mci_control,10yr'}
    assert all(len(t) == 1 for t in tags.values())


@needs_bash
def test_resume_runs_only_missing_folds(tmp_path):
    name = ('ad_rest_detect_data-IRB2024v5_ADCON_DL_data_retinaf_diseaseclean-Wisely-resnet18_paper-Thickness-'
            'roc_auceval-subset0_CW')
    for k in range(8):
        d = tmp_path / 'results' / f'{name}_cv10fold{k}'
        d.mkdir(parents=True)
        (d / 'metrics_test.csv').write_text('x\n')
    # fold 9: trained but crashed before the test eval (checkpoint still there) -> not done
    d = tmp_path / 'results' / f'{name}_cv10fold9'
    d.mkdir()
    (d / 'log.txt').write_text('{}\n')
    (d / 'checkpoint-best.pth').write_text('')
    cmds, err = driver(tmp_path, MODELS='Wisely', TASKS='ad_rest')
    assert len(cmds) == 1 and 'CV_FOLDS_ONLY="8 9"' in cmds[0]
    assert '[resume]' in err
    _, folds, _ = task_of(cmds[0])
    assert folds == [8, 9]


@needs_bash
def test_purged_checkpoint_counts_as_done(tmp_path):
    name = 'mci_control_detect_data-IRB2024v5_ADCON_DL_study3_retinaf_1yr-Hebei-ducan-dual-bs8ep400lr3e-4wd1e-2-accuracyeval-subset0_CW'
    for k in range(10):
        d = tmp_path / 'results' / f'{name}_cv10fold{k}'
        d.mkdir(parents=True)
        (d / 'log.txt').write_text('{}\n')
    cmds, err = driver(tmp_path, MODELS='Hebei', VARIANTS='1yr')
    assert cmds == [] and '[done]' in err


@needs_bash
def test_wandb_running_fold_is_skipped_and_finished_folds_count(tmp_path):
    name = 'mci_control_detect_data-IRB2024v5_ADCON_DL_study3_retinaf_3yr-Mahendran-ad_oct_model-OCT-bs18ep100lr7e-5wd1e-2-roc_auceval-subset0_CW'
    rows = tmp_path / 'wb.tsv'
    rows.write_text(''.join(f'{name}\t{k}\tfinished\n' for k in range(5)) + f'{name}\t5\trunning\n')
    status = tmp_path / 'status.sh'
    status.write_text(f'cat {rows.as_posix()}\n')
    cmds, err = driver(tmp_path, MODELS='Mahendran', VARIANTS='3yr', WANDB_CHECK='1', WANDB_STATUS_CMD=f'bash {status.as_posix()}')
    assert cmds == [] and 'running in wandb' in err
    rows.write_text(''.join(f'{name}\t{k}\tfinished\n' for k in range(5)) + f'{name}\t5\tcrashed\n')
    cmds, err = driver(tmp_path, MODELS='Mahendran', VARIANTS='3yr', WANDB_CHECK='1', WANDB_STATUS_CMD=f'bash {status.as_posix()}')
    assert len(cmds) == 1 and 'CV_FOLDS_ONLY="5 6 7 8 9"' in cmds[0] and '5=crashed' in err


@needs_bash
def test_limits(tmp_path):
    cmds, err = driver(tmp_path, MAX_SUBMIT='2')
    assert len(cmds) == 2 and '21 left by the limit' in err + (tmp_path / 'ref.sh').read_text()
    squeue = 'squeue(){ printf "a\\nb\\nc\\n"; }; sacct(){ :; }; export -f squeue sacct; '
    cmds, _ = driver(tmp_path, stubs=squeue, MAX_SLURM='5')
    assert len(cmds) == 2  # 3 jobs already queued + 2 = 5


@needs_bash
def test_cv_scripts_default_to_all_folds():
    _, folds, _ = task_of('PERIOD= CV_FOLDS_ONLY="" sbatch --job-name=x finetune_Hebei_admci_detection_cv.sh '
                          'ad_control_detect_data D /x/split ducan 0.7 0.7 1.0 3e-4 1e-2 2 0')
    assert folds == list(range(10))
