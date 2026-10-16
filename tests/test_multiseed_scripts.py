# tests/test_multiseed_scripts.py
"""Dry-run the sbatch scripts with torchrun/python/sbatch stubbed out and check the seed wiring."""
import os
import re
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
                          r'sbatch(){ echo "CMD sbatch [SEED=$SEED MODE=$SEED_MODE] $*"; '
                          r'for a in "$@"; do case "$a" in --job-name=*) [ -n "$FAKE_STATE" ] && echo "${a#--job-name=}" >> "$FAKE_STATE";; esac; done; }')
# fake SLURM queue: squeue = $FAKE_SQUEUE + every job sbatch accepted (recorded in $FAKE_STATE); sacct = $FAKE_SACCT (COMPLETED)
SLURM_STUBS = (r'squeue(){ { printf "%s\n" "$FAKE_SQUEUE"; [ -f "$FAKE_STATE" ] && cat "$FAKE_STATE"; } | grep -v "^$"; return 0; }; '
               r'sacct(){ printf "%s\n" "$FAKE_SACCT" | grep -v "^$"; return 0; }; export -f squeue sacct; ')


def run_all(*args, slurm=False, **extra):
    return subprocess.run(['bash', '-c', ENV_STUBS + (SLURM_STUBS if slurm else '') + 'bash "$@"', 'run_all',
                           'multiseed_run_all.sh', *args],
                          cwd=REPO, env=_env(**extra), capture_output=True, text=True)


def _cmds(res):
    """sbatch calls the stub saw, without the --job-name option (see _names for that)"""
    return [re.sub(r' --job-name=\S+', '', l[4:]) for l in res.stdout.splitlines() if l.startswith('CMD ')]


def _names(res):
    return [m.group(1) for l in res.stdout.splitlines() if l.startswith('CMD ') for m in [re.search(r'--job-name=(\S+)', l)]]


def test_run_all_train_submits_every_job_with_the_seed():
    res = run_all('1952', 'train')
    cmds = _cmds(res)
    # 4 UF datasets x (4 baselines + SAIL + sail-enc) + OCTDL and CellData x 4 baselines
    assert res.returncode == 0 and len(cmds) == 4 * 6 + 2 * 4
    assert all(c.startswith('sbatch [SEED=1952 MODE=split] ') for c in cmds)
    assert not any('baseline_multirun_XAI_model_train.sh' in c for c in cmds)  # one job per model, no driver
    assert sum('irb2024v5_full2.sh' in c for c in cmds) == 4 * 4
    assert sum('irb2024v5_smp_full.sh' in c for c in cmds) == 4 + 4  # SAIL + sail-enc per UF dataset
    assert sum('finetune_retfound_OCTDL.sh DME_all' in c for c in cmds) == 4
    assert sum('finetune_retfound_Celldata.sh DME_all' in c for c in cmds) == 4
    for model in ('timm_efficientnet-b4', 'resnet-50', 'vit-base-patch16-224', 'RETFound_mae'):
        assert sum(f' DME_binary_all_split {model} ' in c for c in cmds) == 1  # the models the driver would have submitted
    assert any(c.endswith('DME_binary_all_split timm_efficientnet-b4 timm_efficientnet-b4 5e-4 2 0.05 default OCT 0') for c in cmds)


def test_run_all_passes_seed_mode_through():
    cmds = [l for l in run_all('1952', 'train', SEED_MODE='train').stdout.splitlines() if l.startswith('CMD ')]
    assert cmds and all('[SEED=1952 MODE=train]' in c for c in cmds)


def test_run_all_job_names_are_unique_and_encode_mode_seed_and_kind():
    names = _names(run_all('1952', 'train'))
    assert len(names) == len(set(names)) == 32
    assert 'ms-split-1952-tr-base-DME_binary_all_split-resnet-50' in names
    assert 'ms-split-1952-tr-sail-AMD_all_split' in names and 'ms-split-1952-tr-enc-ERM_all_split' in names
    assert 'ms-train-1952-tr-OCTDL-DME_all-vit-base-patch16-224' in names  # public data always varies the training seed
    # a UF split-mode run and a UF train-mode run of the same seed must not be mistaken for each other
    def uf(ns):
        return {n for n in ns if '-OCTDL-' not in n and '-Celldata-' not in n}
    train_mode = _names(run_all('1952', 'train', SEED_MODE='train'))
    assert uf(train_mode) and not uf(names) & uf(train_mode)
    assert all(n.startswith('ms-train-1952-') for n in train_mode if '-OCTDL-' in n or '-Celldata-' in n)


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
    cmds = _cmds(run_all('1952', 'eval', CHECK_CKPT='0', DATASETS='DME_binary_all_split'))
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


def test_run_all_dry_run_lists_every_job_and_submits_nothing():
    res = run_all('1952', 'train', DRY_RUN='1')
    jobs = [l for l in res.stdout.splitlines() if l.startswith('SEED=1952 SEED_MODE=split sbatch --job-name=ms-')]
    assert res.returncode == 0 and 'CMD ' not in res.stdout  # nothing was handed to sbatch
    assert len(jobs) == 32
    assert '32 submitted (dry run), 0 already done, 0 already queued, 0 not ready' in res.stdout


ENC_ARGS = ('enc', 'weighted_sum', '0.5', 'decoder_to_encoder', '0', 'pre', '-1', '-1', 'conv', 'scalar')


def run_tmp(*args, slurm=False, **extra):
    return subprocess.run(['bash', '-c', ENV_STUBS + (SLURM_STUBS if slurm else '') + 'bash "$@"', 'tmp',
                           'multiseed_sail_enc_tmp.sh', *args],
                          cwd=REPO, env=_env(**extra), capture_output=True, text=True)


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
    assert len(_cmds(run_all('1952', 'train'))) == 32  # default still runs everything


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


UF1 = dict(DATASETS='DME_binary_all_split', PUBLIC_DATASETS=' ')  # one UF dataset, no public data: 6 train jobs
TR = [f'ms-split-1952-tr-base-DME_binary_all_split-{m}' for m in
      ('RETFound_mae', 'resnet-50', 'vit-base-patch16-224', 'timm_efficientnet-b4')]
TR += ['ms-split-1952-tr-sail-DME_binary_all_split', 'ms-split-1952-tr-enc-DME_binary_all_split']


def slurm_env(tmp_path, squeue=(), sacct=()):
    return dict(FAKE_SQUEUE='\n'.join(squeue), FAKE_SACCT='\n'.join(sacct), FAKE_STATE=str(tmp_path / 'queue.txt'))


def test_run_all_resume_skips_completed_and_queued_jobs(tmp_path):
    env = slurm_env(tmp_path, squeue=[TR[5]], sacct=[TR[1], TR[4]])
    res = run_all('1952', 'train', slurm=True, **env, **UF1)
    assert res.returncode == 0 and _names(res) == [TR[0], TR[2], TR[3]]
    assert res.stderr.count('[skip] already done') == 2 and res.stderr.count('[skip] already queued/running') == 1
    assert '3 submitted, 2 already done, 1 already queued' in res.stdout
    # RERUN_DONE=1 resubmits the completed ones, but never what is still queued/running
    (tmp_path / 'again').mkdir()
    env = slurm_env(tmp_path / 'again', squeue=[TR[5]], sacct=[TR[1], TR[4]])
    res = run_all('1952', 'train', slurm=True, RERUN_DONE='1', **env, **UF1)
    assert _names(res) == TR[:5]


def test_run_all_resume_ignores_jobs_of_other_seeds_and_modes(tmp_path):
    other = [n.replace('1952', '5235') for n in TR] + [n.replace('split', 'train') for n in TR]
    res = run_all('1952', 'train', slurm=True, **slurm_env(tmp_path, squeue=other, sacct=other), **UF1)
    assert _names(res) == TR and 'already' not in res.stderr


def test_run_all_max_slurm_stops_then_a_rerun_finishes_without_duplicates(tmp_path):
    first = run_all('1952', 'train', slurm=True, MAX_SLURM='5', **slurm_env(tmp_path, squeue=['x1', 'x2', 'x3']), **UF1)
    assert _names(first) == TR[:2]  # 3 jobs already queued + 2 submitted = 5
    assert '[limit] 5 jobs in the queue >= MAX_SLURM=5' in first.stderr
    assert '2 submitted' in first.stdout and '4 left: MAX_SLURM=5 reached, rerun to continue' in first.stdout
    # later: the other jobs finished, the two submitted ones COMPLETED -> the rerun submits only the remaining four
    (tmp_path / 'second').mkdir()
    second = run_all('1952', 'train', slurm=True, MAX_SLURM='5', **slurm_env(tmp_path / 'second', sacct=TR[:2]), **UF1)
    assert _names(second) == TR[2:] and 'left' not in second.stdout
    assert sorted(_names(first) + _names(second)) == sorted(TR)


def test_run_all_max_slurm_counts_what_a_dry_run_would_submit(tmp_path):
    res = run_all('1952', 'train', slurm=True, DRY_RUN='1', MAX_SLURM='2', **slurm_env(tmp_path), **UF1)
    jobs = [l for l in res.stdout.splitlines() if l.startswith('SEED=1952 SEED_MODE=split sbatch --job-name=')]
    assert len(jobs) == 2 and '2 submitted (dry run)' in res.stdout and '4 left' in res.stdout
    assert not (tmp_path / 'queue.txt').exists()  # nothing really submitted
    unlimited = run_all('1952', 'train', slurm=True, DRY_RUN='1', **slurm_env(tmp_path, squeue=['x'] * 50), **UF1)
    assert '6 submitted (dry run)' in unlimited.stdout  # MAX_SLURM unset = no limit


def test_run_all_rejects_a_bad_max_slurm_and_warns_without_slurm():
    res = run_all('1952', 'train', MAX_SLURM='many')
    assert res.returncode != 0 and 'CMD ' not in res.stdout
    res = run_all('1952', 'train', MAX_SLURM='3', **UF1)  # no squeue/sacct on this machine: warn, do not guess
    assert res.returncode == 0 and 'squeue/sacct not found' in res.stderr and len(_cmds(res)) == 6


def _enc_checkpoint(root):
    """where the training run of sail-enc (seed 1952, split mode) leaves its checkpoint, under a fake results root"""
    [train] = dry_run('finetune_retfound_UFbenchmark_irb2024v5_smp_full.sh', 'DME_binary_all_split', 'SMP', SEG_CKPT,
                      '5e-4', '2', '1e-4', 'default', 'OCT', '0', *ENC_ARGS, seed=1952)
    tokens = train.split()
    run_dir = tokens[tokens.index('--task') + 1].rstrip('/') + seed_suffix(1952, 0, 0)
    path = os.path.join(root, run_dir, 'checkpoint-best.pth')
    os.makedirs(os.path.dirname(path))
    open(path, 'w').close()


def test_run_all_eval_waits_while_the_training_job_is_running(tmp_path):
    results = tmp_path / 'results'
    _enc_checkpoint(str(results))
    env = dict(MODEL_DIR=str(results), **UF1)
    running = run_tmp('1952', 'eval', slurm=True, **env, **slurm_env(tmp_path, squeue=[TR[5]]))
    assert not _cmds(running) and '[skip] training still queued/running: ' + TR[5] in running.stderr
    # checkpoint-best.pth exists after the first epoch, so the checkpoint alone must not be enough
    (tmp_path / 'idle').mkdir()
    finished = run_tmp('1952', 'eval', slurm=True, **env, **slurm_env(tmp_path / 'idle'))
    assert _names(finished) == ['ms-split-1952-ev-enc-DME_binary_all_split-rise']
    (tmp_path / 'forced').mkdir()
    forced = run_tmp('1952', 'eval', slurm=True, CHECK_CKPT='0', **env, **slurm_env(tmp_path / 'forced', squeue=[TR[5]]))
    assert _names(forced) == ['ms-split-1952-ev-enc-DME_binary_all_split-rise']


def test_run_all_eval_resume_skips_completed_eval_jobs(tmp_path):
    env = slurm_env(tmp_path, sacct=['ms-split-1952-ev-base-DME_binary_all_split-resnet-50'])
    res = run_all('1952', 'eval', slurm=True, CHECK_CKPT='0', **env, **UF1)
    assert len(_names(res)) == 5 and 'ms-split-1952-ev-base-DME_binary_all_split-resnet-50' not in _names(res)
    assert '[skip] already done: ms-split-1952-ev-base-DME_binary_all_split-resnet-50' in res.stderr
