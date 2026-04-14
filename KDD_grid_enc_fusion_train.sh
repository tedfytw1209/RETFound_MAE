#!/bin/bash
# Grid training launcher — submits one SLURM job per (ENC_IDX × FUSION_DIM) combination.
# All other hyperparameters mirror KDD_alpha_gate_train.sh for direct comparison.
#
# Usage:
#   bash KDD_grid_enc_fusion_train.sh          # submit all 2×2 = 4 combinations
#
# Grid axes:
#   ENC_IDX    — encoder layer index used as cross-attention query: -1, -3
#   FUSION_DIM — fusion spatial dimension: 16, 32

BASE_CKPT=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth

ENC_IDXS=(-2)
#ENC_IDXS=(-1 -2 -3)
#FUSION_DIMS=(4 9 16)
FUSION_DIMS=(4 16 32)

for ENC_IDX in "${ENC_IDXS[@]}"; do
    for FUSION_DIM in "${FUSION_DIMS[@]}"; do
        echo "Submitting ENC_IDX=${ENC_IDX}, FUSION_DIM=${FUSION_DIM} ..."
        sbatch finetune_retfound_UFbenchmark_v5_eval_smp_full.sh \
            DME_binary_all_split \
            SMP \
            ${BASE_CKPT} \
            5e-4 \
            2 \
            1e-4 \
            default \
            OCT \
            0 \
            fuse \
            weighted_sum \
            0.5 \
            decoder_to_encoder \
            ${FUSION_DIM} \
            pre \
            ${ENC_IDX} \
            -1 \
            conv \
            scalar \
            "--seg_mask" \
            "--smp_learnable_alpha" \
            ""
    done
done
