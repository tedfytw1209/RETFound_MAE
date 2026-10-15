# tests/test_multiseed_scripts.py
"""Dry-run the sbatch scripts with torchrun/python/sbatch stubbed out and check the seed wiring."""
import os
import shutil
import subprocess

import pytest

from util.multiseed import seed_suffix

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUBS = ('torchrun(){ echo "CMD torchrun $*"; }; python(){ echo "CMD python $*"; }; '
         'sbatch(){ echo "CMD sbatch $*"; }; module(){ :; }; conda(){ :; }; '
         'export -f torchrun python sbatch module conda; ')

pytestmark = pytest.mark.skipif(os.name == 'nt' or shutil.which('bash') is None,
                                reason='CRLF checkout on Windows; run on the cluster')


def _env(seed=None, mode=None, **extra):
    env = {k: v for k, v in os.environ.items() if k not in ('SEED', 'SEED_MODE')}
    env.update(SLURM_JOBID='1234', **extra)
    if seed is not None:
        env['SEED'] = str(seed)
    if mode is not None:
        env['SEED_MODE'] = mode
    return env


def dry_run(script, *args, seed=None, mode=None, **extra):
    out = subprocess.run(['bash', '-c', STUBS + 'bash "$@"', 'dry_run', script, *args],
                         cwd=REPO, env=_env(seed, mode, **extra), capture_output=True, text=True, check=True).stdout
    return [line[4:] for line in out.splitlines() if line.startswith('CMD ')]


def env_vars(seed=None, mode=None):
    return subprocess.run(['bash', '-c', 'source ./multiseed_env.sh >/dev/null && echo "$SPLIT_DIR|$SEED_ARGS|$RUN_SUFFIX"'],
                          cwd=REPO, env=_env(seed, mode), capture_output=True, text=True)


def test_env_defaults_to_original_run():
    assert env_vars().stdout.strip() == 'tune8-eval2||'
    assert env_vars(42).stdout.strip() == 'tune8-eval2||'


def test_env_split_mode():
    assert env_vars(1952).stdout.strip() == 'tune8-eval2-s1952|--split_seed 1952|-s1952'
    assert env_vars(1952).stdout.strip().endswith(seed_suffix(1952, 0, 0))


def test_env_train_mode():
    assert env_vars(1952, 'train').stdout.strip() == 'tune8-eval2|--seed 1952|-seed1952'
    assert env_vars(1952, 'train').stdout.strip().endswith(seed_suffix(42, 1952, 0))


@pytest.mark.parametrize('seed', [0, 42])
def test_env_train_mode_rejects_script_default_seeds(seed):
    assert env_vars(seed, 'train').returncode != 0


def test_submit_runs_command_once_per_seed():
    out = subprocess.run(['bash', 'multiseed_submit.sh', '1952 5235', 'bash', '-c', 'echo "CMD $SEED"'],
                         cwd=REPO, env=_env(), capture_output=True, text=True, check=True).stdout
    assert [l[4:] for l in out.splitlines() if l.startswith('CMD ')] == ['1952', '5235']


UF_TRAIN = ('finetune_retfound_UFbenchmark_irb2024v5_full2.sh', 'DME_binary_all_split', 'RETFound_mae',
            'RETFound_mae_natureOCT', '5e-4', '2', '0.05', 'default', 'OCT', '0')
TASK = ('DME_binary_all_split-IRB2024_v5_all-all-RETFound_mae_natureOCT-OCT-'
        'bs32ep50lr5e-4optadamw-defaulteval-trsub0--')
SMP_TRAIN = ('finetune_retfound_UFbenchmark_irb2024v5_smp_full.sh', 'DME_binary_all_split', 'SMP', '/blue/x/best.pth',
             '5e-4', '2', '1e-4', 'default', 'OCT', '0', 'enc', 'weighted_sum', '0.5', 'decoder_to_encoder',
             '0', 'pre', '-1', '-1', 'conv')


def test_uf_train_without_seed_is_unchanged():
    [cmd] = dry_run(*UF_TRAIN)
    assert 'split/tune8-eval2/DME_binary_all_split.csv' in cmd
    assert f'--task {TASK}/ ' in cmd  # the python side adds any seed suffix
    assert '--seed' not in cmd and '--split_seed' not in cmd


def test_uf_train_split_mode():
    [cmd] = dry_run(*UF_TRAIN, seed=1952)
    assert 'split/tune8-eval2-s1952/DME_binary_all_split.csv' in cmd
    assert '--split_seed 1952' in cmd and '--seed ' not in cmd


def test_uf_train_train_mode():
    [cmd] = dry_run(*UF_TRAIN, seed=1952, mode='train')
    assert 'split/tune8-eval2/DME_binary_all_split.csv' in cmd
    assert '--seed 1952' in cmd and '--split_seed' not in cmd


def test_smp_train_split_mode():
    [cmd] = dry_run(*SMP_TRAIN, seed=1952)
    assert 'split/tune8-eval2-s1952/' in cmd and '--split_seed 1952' in cmd


@pytest.mark.parametrize('script', ['finetune_retfound_OCTDL.sh', 'finetune_retfound_Celldata.sh',
                                    'finetune_retfound_OCTDL_smp.sh', 'finetune_retfound_Celldata_smp.sh'])
def test_public_train_always_varies_training_seed_only(script):
    [cmd] = dry_run(script, 'DME_all', seed=1952)  # SEED_MODE left at default: public scripts force train
    assert '--seed 1952' in cmd and '--split_seed' not in cmd
    [orig] = dry_run(script, 'DME_all')
    assert '--seed' not in orig


def test_train_driver_exports_seed_and_mode():
    cmds = dry_run('baseline_multirun_XAI_model_train.sh', 'finetune_retfound_UFbenchmark_irb2024v5_full2.sh',
                   'DME_binary_all_split', seed=1952, mode='train')
    assert len(cmds) == 4
    assert all(c.startswith('sbatch --export=ALL,SEED=1952,SEED_MODE=train finetune_retfound_UFbenchmark_irb2024v5_full2.sh')
               for c in cmds)


MODEL_DIR = '/orange/ruogu.fang/tienyuchang/RETfound_results'
EVAL_FULL = ('baseline_multirun_XAI_eval_full.sh', 'finetune_retfound_UFbenchmark_v5_eval_full2.sh',
             'DME_binary_all_split', '2', 'RETFound_mae', 'RETFound_mae_natureOCT', '224')
SMP_RESUME = '/orange/x/ERM_all_split-IRB2024_v5_all-all-/blue/y.pth-OCT-conv---/checkpoint-best.pth'
EVAL_SMP = ('baseline_multirun_XAI_eval_smp.sh', 'finetune_retfound_UFbenchmark_v5_eval_smp_full.sh',
            'ERM_all_split', '2', 'SMP', '/blue/y.pth', SMP_RESUME, '512', 'enc', 'weighted_sum', '0.5',
            'decoder_to_encoder', '0', 'pre', '-1', '-1', 'encoder', '-1', 'conv')


def test_eval_full_driver_original_uses_full2_checkpoint():
    cmds = dry_run(*EVAL_FULL)
    assert len(cmds) == 3  # hirescam, gradcam++, gradcamv2
    for cmd in cmds:
        assert f'--resume {MODEL_DIR}/{TASK}/checkpoint-best.pth' in cmd
        assert 'split/tune8-eval2/' in cmd and '--seed' not in cmd and '--split_seed' not in cmd


def test_eval_full_driver_split_mode():
    for cmd in dry_run(*EVAL_FULL, seed=1952):
        assert f'--resume {MODEL_DIR}/{TASK}-s1952/checkpoint-best.pth' in cmd
        assert 'split/tune8-eval2-s1952/' in cmd and '--split_seed 1952' in cmd


def test_eval_full_driver_train_mode():
    for cmd in dry_run(*EVAL_FULL, seed=1952, mode='train'):
        assert f'--resume {MODEL_DIR}/{TASK}-seed1952/checkpoint-best.pth' in cmd
        assert 'split/tune8-eval2/' in cmd and '--seed 1952' in cmd


@pytest.mark.parametrize('mode,suffix,split', [('split', '-s1952', 'tune8-eval2-s1952'),
                                               ('train', '-seed1952', 'tune8-eval2')])
def test_eval_smp_driver_rewrites_resume(mode, suffix, split):
    cmds = dry_run(*EVAL_SMP, seed=1952, mode=mode)
    assert len(cmds) == 3
    for cmd in cmds:
        assert f'--resume {SMP_RESUME[:-len("/checkpoint-best.pth")]}{suffix}/checkpoint-best.pth' in cmd
        assert f'split/{split}/' in cmd
