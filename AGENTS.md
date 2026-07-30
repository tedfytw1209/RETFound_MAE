# Repository Guidelines

## Project Structure & Module Organization

RETFound_MAE is a PyTorch research repository for retinal-image pretraining, fine-tuning, segmentation, and explainability. Primary entry points live at the repository root: `main_pretrain.py`, `main_finetune.py`, `main_linear_probe.py`, and `main_XAI_evaluation.py`. Shared training, dataset, scheduling, loss, and evaluation helpers are in `util/` and `engine_finetune.py`. Model implementations are split across `models/`, `models_vit.py`, `SAM2UNet/`, `SMP/`, `sam2/`, `gcn_lib/`, and `baselines/`. Dataset preparation scripts live in `dataprocess/` and `baseline/`. Shell scripts at the root capture reproducible experiment configurations; notebooks are exploratory assets, not production entry points.

## Build, Test, and Development Commands

Create the documented environment before running experiments:

```bash
conda create -n retfound python=3.11 -y
conda activate retfound
pip install -r requirements.txt
```

Use `python -m compileall .` for a quick syntax smoke test. Start a single-GPU fine-tuning run with the command pattern documented in `README.md`, for example:

```bash
torchrun --nproc_per_node=1 main_finetune.py --data_path ./IDRiD \
  --model RETFound_mae --finetune RETFound_mae_meh --nb_classes 5
```

Use `--eval --resume PATH` to evaluate a checkpoint. Prefer adapting an existing `.sh` experiment script when reproducing a published configuration.

## Coding Style & Naming Conventions

Follow existing Python conventions: four-space indentation, `snake_case` functions and variables, `PascalCase` classes, and imports grouped as standard library, third-party, then local modules. Keep CLI options descriptive and underscore-separated, matching existing flags such as `--data_path` and `--batch_size`. No formatter or linter is configured; keep changes PEP 8-compatible and avoid unrelated notebook output or formatting churn.

## Testing Guidelines

There is currently no automated test suite or coverage threshold. For each change, run `python -m compileall .` and a focused smoke run of the affected training, inference, or preprocessing path. If adding unit tests, place them in `tests/`, name files `test_<module>.py`, and document any new test dependency in `requirements.txt`. GPU-dependent checks should state the hardware, dataset subset, seed, and checkpoint used.

## Commit & Pull Request Guidelines

Recent history uses short, imperative subjects such as `add retry` and `update workers`. Improve on that pattern with a specific scope, for example `finetune: fix checkpoint resume`. Keep commits focused and exclude datasets, checkpoints, `wandb/`, and generated heatmaps. Pull requests should explain the research goal, list commands and configurations tested, link related issues or experiments, and include metric tables or representative visualizations when results change.
