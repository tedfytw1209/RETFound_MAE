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


SEG_CKPT = '/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth'
SMP_ARGS = ('fuse', 'weighted_sum', '0.5', 'decoder_to_encoder', '9', 'pre', '-2', '-1', 'conv', 'scalar',
            '--seg_mask', '--smp_learnable_alpha')
ENV_STUBS = STUBS.replace('sbatch(){ echo "CMD sbatch $*"; }',
                          'sbatch(){ echo "CMD sbatch [SEED=$SEED MODE=$SEED_MODE] $*"; }')


def run_all(*args, **extra):
    return subprocess.run(['bash', '-c', ENV_STUBS + 'bash "$@"', 'run_all', 'multiseed_run_all.sh', *args],
                          cwd=REPO, env=_env(**extra), capture_output=True, text=True)


def test_run_all_train_submits_every_job_with_the_seed():
    res = run_all('1952', 'train')
    cmds = [l[4:] for l in res.stdout.splitlines() if l.startswith('CMD ')]
    assert res.returncode == 0 and len(cmds) == 4 * 3 + 2  # 4 UF datasets x (4 baselines + SMP + sail-enc) + OCTDL + CellData
    assert all(c.startswith('sbatch [SEED=1952 MODE=split] ') for c in cmds)
    assert sum('baseline_multirun_XAI_model_train.sh' in c for c in cmds) == 4 + 2
    assert sum('irb2024v5_smp_full.sh' in c for c in cmds) == 4 + 4  # SAIL + sail-enc per UF dataset
    assert any('finetune_retfound_OCTDL.sh DME_all' in c for c in cmds)


def test_run_all_passes_seed_mode_through():
    cmds = [l for l in run_all('1952', 'train', SEED_MODE='train').stdout.splitlines() if l.startswith('CMD ')]
    assert cmds and all('[SEED=1952 MODE=train]' in c for c in cmds)


@pytest.mark.parametrize('args,mode', [(('42', 'train'), 'split'), (('0', 'train'), 'train'), (('42', 'eval'), 'split'),
                                       ((), 'split'), (('1952', 'bogus'), 'split')])
def test_run_all_refuses_original_seed_and_bad_args(args, mode):
    res = run_all(*args, SEED_MODE=mode)
    assert res.returncode != 0 and 'CMD ' not in res.stdout


def test_run_all_eval_skips_runs_without_checkpoints():
    res = run_all('1952', 'eval', DATASETS='NOPE_all_split')
    assert res.returncode == 0 and 'CMD ' not in res.stdout
    assert res.stderr.count('[skip] no checkpoint yet') == 6  # 4 baselines + SMP + sail-enc


def test_run_all_eval_submits_baselines_and_smp_resume_matches_training_run():
    cmds = [l[4:] for l in run_all('1952', 'eval', CHECK_CKPT='0', DATASETS='DME_binary_all_split').stdout.splitlines()
            if l.startswith('CMD ')]
    assert len(cmds) == 6
    assert all(c.startswith('sbatch [SEED=1952 MODE=split] ') for c in cmds)
    assert sum('baseline_multirun_XAI_eval_full.sh' in c for c in cmds) == 4
    assert any(c.endswith('DME_binary_all_split 2 timm_efficientnet-b4 timm_efficientnet-b4 380') for c in cmds)
    [smp] = [c for c in cmds if 'baseline_multirun_XAI_eval_smp.sh' in c]
    # the training run (SEED unset) writes <output_dir>/<task>checkpoint-best.pth: that is what the eval driver must be given
    [train] = dry_run('finetune_retfound_UFbenchmark_irb2024v5_smp_full.sh', 'DME_binary_all_split', 'SMP', SEG_CKPT,
                      '5e-4', '2', '1e-4', 'default', 'OCT', '0', *SMP_ARGS)
    tokens = train.split()
    resume = f'{tokens[tokens.index("--output_dir") + 1]}/{tokens[tokens.index("--task") + 1]}checkpoint-best.pth'
    assert f' {resume} 512 fuse weighted_sum 0.5 decoder_to_encoder 9 pre -2 -1 head -1 conv ' in smp


def test_run_all_dry_run_lists_driver_fanout_and_submits_nothing():
    res = run_all('1952', 'train', DRY_RUN='1')
    jobs = [l for l in res.stdout.splitlines() if l.startswith('    -> job: ')]
    assert res.returncode == 0 and 'CMD ' not in res.stdout  # nothing was handed to sbatch
    assert len(jobs) == (4 + 2) * 4  # 6 train drivers x 4 baselines (4 UF datasets + OCTDL + CellData)
    for model in ('timm_efficientnet-b4', 'resnet-50', 'vit-base-patch16-224', 'RETFound_mae'):
        assert sum(f' {model} ' in j for j in jobs) == 6
    assert '(dry run): 14 sbatch calls + 24 jobs spawned by train drivers' in res.stdout


ENC_ARGS = ('enc', 'weighted_sum', '0.5', 'decoder_to_encoder', '0', 'pre', '-1', '-1', 'conv', 'scalar')


def run_tmp(*args, **extra):
    return subprocess.run(['bash', '-c', ENV_STUBS + 'bash "$@"', 'tmp', 'multiseed_sail_enc_tmp.sh', *args],
                          cwd=REPO, env=_env(**extra), capture_output=True, text=True)


def _cmds(res):
    return [l[4:] for l in res.stdout.splitlines() if l.startswith('CMD ')]


def test_sail_enc_tmp_train_submits_only_sail_enc():
    res = run_tmp('1952', 'train')
    cmds = _cmds(res)
    assert res.returncode == 0 and len(cmds) == 4  # one training job per UF dataset, nothing else
    assert all(c.startswith('sbatch [SEED=1952 MODE=split] finetune_retfound_UFbenchmark_irb2024v5_smp_full.sh ')
               and f' {SEG_CKPT} 5e-4 2 1e-4 default OCT 0 ' + ' '.join(ENC_ARGS) in c and 'baseline_multirun' not in c
               for c in cmds)
    assert [c.split()[4] for c in cmds] == ['DME_binary_all_split', 'AMD_all_split', 'Glaucoma_binary_all_split', 'ERM_all_split']


def test_run_all_only_sail_enc_matches_tmp_script():
    assert _cmds(run_all('1952', 'train', ONLY='sail-enc')) == _cmds(run_tmp('1952', 'train'))
    assert len(_cmds(run_all('1952', 'train'))) == 14  # default still runs everything


def test_sail_enc_eval_resumes_the_training_run_dir_and_uses_rise():
    cmds = _cmds(run_tmp('1952', 'eval', CHECK_CKPT='0', DATASETS='DME_binary_all_split'))
    [cmd] = cmds
    assert cmd.startswith('sbatch [SEED=1952 MODE=split] finetune_retfound_UFbenchmark_v5_eval_smp_full.sh DME_binary_all_split SMP ')
    [train] = dry_run('finetune_retfound_UFbenchmark_irb2024v5_smp_full.sh', 'DME_binary_all_split', 'SMP', SEG_CKPT,
                      '5e-4', '2', '1e-4', 'default', 'OCT', '0', *ENC_ARGS, seed=1952)
    tokens = train.split()
    # main_finetune_smp.py appends the seed suffix to --task (util.multiseed), which names the output dir
    resume = f'{tokens[tokens.index("--output_dir") + 1]}/{tokens[tokens.index("--task") + 1].rstrip("/")}{seed_suffix(1952, 0, 0)}'
    # KDD_rise_enceval.sh argument order: RESUME 2 512 rise 1024 <thickness dir> enc ... head -1 conv
    assert f' {resume}/checkpoint-best.pth 2 512 rise 1024 /orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/ '            'enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 head -1 conv' in cmd
    assert cmd.endswith('head -1 conv')  # no extra flags: the training run carries none


def test_sail_enc_eval_one_job_per_xai_method_and_skips_missing_checkpoint():
    cmds = _cmds(run_tmp('1952', 'eval', CHECK_CKPT='0', DATASETS='DME_binary_all_split', SAIL_ENC_XAI='rise hirescam'))
    assert [c.split(' 2 512 ')[1].split()[0] for c in cmds] == ['rise', 'hirescam']
    res = run_tmp('1952', 'eval', DATASETS='NOPE_all_split')
    assert res.returncode == 0 and 'CMD ' not in res.stdout and res.stderr.count('[skip] no checkpoint yet') == 1


def test_run_all_rejects_unknown_only():
    res = run_all('1952', 'train', ONLY='bogus')
    assert res.returncode != 0 and 'CMD ' not in res.stdout
