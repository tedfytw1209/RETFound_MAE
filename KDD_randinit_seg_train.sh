#!/bin/bash
BASE_CKPT=na


ENC_IDX=-2
FUSION_DIM=9


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
    scalar \
    "--seg_mask" \
    "--smp_learnable_alpha" \
    ""

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
    enc \
    weighted_sum \
    0.5 \
    decoder_to_encoder \
    0 \
    pre \
    -1 \
    -1 \
    conv \
    scalar \
    "" \
    "" \
    ""