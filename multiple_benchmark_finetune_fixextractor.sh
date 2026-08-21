#!/bin/bash

# Same as multiple_benchmark_finetune.sh, but freezes the backbone and only trains the head
# (--fix_extractor, see main_finetune.py). Args match finetune_retfound_benchmark.sh's
# positional order: STUDY MODEL Num_CLASS FINETUNED_MODEL RESUME_DIR ADDCMD.
# RESUME_DIR is left as "" (fresh finetune runs, not --eval of an existing checkpoint) so
# ADDCMD lands in the 6th slot -- finetune_retfound_benchmark.sh both embeds ADDCMD in --task
# (which becomes the wandb run name) and appends it as a real CLI flag, so "--fix_extractor"
# shows up in the wandb name automatically, same as --eval does in multiple_benchmark_evaluate.sh.
sbatch finetune_retfound_benchmark.sh APTOS2019 RETFound_mae 5 RETFound_mae_natureCFP "" --fix_extractor
sbatch finetune_retfound_benchmark.sh MESSIDOR2 RETFound_mae 5 RETFound_mae_natureCFP "" --fix_extractor
sbatch finetune_retfound_benchmark.sh IDRiD_data RETFound_mae 5 RETFound_mae_natureCFP "" --fix_extractor
sbatch finetune_retfound_benchmark.sh PAPILA RETFound_mae 3 RETFound_mae_natureCFP "" --fix_extractor
sbatch finetune_retfound_benchmark.sh Glaucoma_fundus RETFound_mae 3 RETFound_mae_natureCFP "" --fix_extractor
sbatch finetune_retfound_benchmark.sh JSIEC RETFound_mae 39 RETFound_mae_natureCFP "" --fix_extractor
sbatch finetune_retfound_benchmark.sh Retina RETFound_mae 4 RETFound_mae_natureCFP "" --fix_extractor
#sbatch finetune_retfound_benchmark.sh OCTID RETFound_mae 5 RETFound_mae_natureOCT "" --fix_extractor
