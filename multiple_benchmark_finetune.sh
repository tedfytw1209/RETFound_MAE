#!/bin/bash

# Args match finetune_retfound_benchmark.sh's positional order: STUDY MODEL Num_CLASS FINETUNED_MODEL
# (no RESUME_DIR -- these are fresh finetune runs, not --eval of an existing checkpoint).
# STUDY must be the actual folder name under DIR_ROOT (see multiple_benchmark_evaluate.sh, which
# uses these same folder names) -- IDRiD_data and Glaucoma_fundus, not IDRID/Glaucoma.
sbatch finetune_retfound_benchmark.sh APTOS2019 RETFound_mae 5 RETFound_mae_natureCFP
sbatch finetune_retfound_benchmark.sh MESSIDOR2 RETFound_mae 5 RETFound_mae_natureCFP
sbatch finetune_retfound_benchmark.sh IDRiD_data RETFound_mae 5 RETFound_mae_natureCFP
sbatch finetune_retfound_benchmark.sh PAPILA RETFound_mae 3 RETFound_mae_natureCFP
sbatch finetune_retfound_benchmark.sh Glaucoma_fundus RETFound_mae 3 RETFound_mae_natureCFP
sbatch finetune_retfound_benchmark.sh JSIEC RETFound_mae 39 RETFound_mae_natureCFP
sbatch finetune_retfound_benchmark.sh Retina RETFound_mae 4 RETFound_mae_natureCFP
#sbatch finetune_retfound_benchmark.sh OCTID RETFound_mae 5 RETFound_mae_natureOCT
