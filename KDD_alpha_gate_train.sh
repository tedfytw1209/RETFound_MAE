#!/bin/bash
# Alpha-gate ablation launcher — submits one SLURM job per gate type.
# All other hyperparameters mirror KDD_gated_train.sh for direct comparison.
#
# Usage:
#   bash KDD_alpha_gate_train.sh                     # submit all 5 gate types
#   bash KDD_alpha_gate_train.sh scalar attn          # submit specific types only
#
# Gate types:
#   scalar  — single shared α (baseline)
#   channel — per-channel α vector  (C alphas)
#   spatial — simple per-pixel conv gate  (H×W alphas, no global context)
#   se      — SE global bias + CBAM spatial attention
#   attn    — cross-attention: encoder query × decoder spatial keys

BASE_CKPT=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth

if [ $# -eq 0 ]; then
    ALPHA_TYPES=(channel spatial se attn)
else
    ALPHA_TYPES=("$@")
fi

for ALPHA_TYPE in "${ALPHA_TYPES[@]}"; do
    echo "Submitting alpha_type=${ALPHA_TYPE} ..."
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
        9 \
        pre \
        -2 \
        -1 \
        conv \
        ${ALPHA_TYPE} \
        "--seg_mask" \
        "--smp_learnable_alpha" \
        ""
done
