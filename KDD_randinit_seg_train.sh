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


ENC_IDX=-2
FUSION_DIM=9


#echo "Submitting ENC_IDX=${ENC_IDX}, FUSION_DIM=${FUSION_DIM} ..."
sbatch finetune_retfound_UFbenchmark_irb2024v5_smp_full.sh \
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
    attn \
    "--seg_mask" \
    "--smp_learnable_alpha" \
    ""