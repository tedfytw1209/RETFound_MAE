#!/bin/bash
# multiseed_submit.sh — re-run any single-seed command once per seed, with SEED exported
# (SEED_MODE is inherited from the caller; default split).
#   bash multiseed_submit.sh "1952 5235 8234 8386" sbatch baseline_multirun_XAI_model_train.sh \
#       finetune_retfound_UFbenchmark_irb2024v5_full2.sh DME_binary_all_split RETFound_mae RETFound_mae_natureOCT 5e-4 2 0.05 default OCT
#   SEED_MODE=train bash multiseed_submit.sh "1952 5235 8234 8386" sbatch ...
SEEDS=$1
shift
for SEED in $SEEDS
do
    echo "SEED=$SEED SEED_MODE=${SEED_MODE:-split} $*"
    SEED=$SEED "$@"
done
