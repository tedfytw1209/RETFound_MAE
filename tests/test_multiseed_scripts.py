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
