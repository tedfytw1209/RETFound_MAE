#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

# ─────────────────────────────────────────────────────────────────────────────
# UF -> public-dataset transfer eval, SMP variants (enc, fuse weighted_sum).
# One UF-trained SMP checkpoint per (task, SMP mode); XAI eval run on the
# equivalent-disease OCTDL split, and on CellData (OCT2017) where it exists
# (CellData only has a DME split).
#
# Table (Task / Source / Target / SMP(enc) / SMP(fuse,WS)):
#   DME  UF DME -> OCTDL DME
#   DME  UF DME -> OCT2017(CellData) DME
#   AMD  UF AMD -> OCTDL AMD
#   ERM  UF ERM -> OCTDL ERM
# SMP(fuse,WS) DME cells are already done -> commented out with #. Uncomment to rerun.
#
# Note: finetune_retfound_OCTDL_eval.sh / finetune_retfound_Celldata_eval.sh
# ignore the passed XAI arg and always loop hirescam/gradcamv2/gradcam++
# internally, so each call below already covers all three methods once —
# no need to wrap these in an outer XAI_METHODS loop.
# ─────────────────────────────────────────────────────────────────────────────

BASE_CKPT=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth
RESULTS_DIR=/orange/ruogu.fang/tienyuchang/RETfound_results
DATA_TYPE=IRB2024_v5_all
MODALITY=OCT
LR=5e-4
BATCH_SIZE=16
EPOCHS=100
ALPHA_TYPE=scalar
NUM_CLASS=2
INPUT_SIZE=512
STEP_PIXELS=1024
XAI_METHOD=${1:-"hirescam"}  # not used — see note above

OCTDL_MASK_DIR=/orange/ruogu.fang/tienyuchang/OCTDL_masks_multiclass_resnet50_new/
CELLDATA_MASK_DIR=/orange/ruogu.fang/tienyuchang/CellData_masks_multiclass_resnet50_new/

octdl_eval() {
    # $1=STUDY $2=RESUME $3=SMPMode $4=FUSION_DIM $5=ENC_IDX
    sbatch finetune_retfound_OCTDL_eval.sh "$1" SMP ${BASE_CKPT} "$2" ${NUM_CLASS} ${INPUT_SIZE} ${XAI_METHOD} ${STEP_PIXELS} ${OCTDL_MASK_DIR} "$3" weighted_sum 0.5 decoder_to_encoder "$4" pre "$5" -1 head -1 conv "${@:6}"
}

celldata_eval() {
    # $1=STUDY $2=RESUME $3=SMPMode $4=FUSION_DIM $5=ENC_IDX
    sbatch finetune_retfound_Celldata_eval.sh "$1" SMP ${BASE_CKPT} "$2" ${NUM_CLASS} ${INPUT_SIZE} ${XAI_METHOD} ${STEP_PIXELS} ${CELLDATA_MASK_DIR} "$3" weighted_sum 0.5 decoder_to_encoder "$4" pre "$5" -1 head -1 conv "${@:6}"
}

uf_resume() {
    # $1=DATASET(UF) $2=SUFFIX
    echo "${RESULTS_DIR}/$1-${DATA_TYPE}-all-${BASE_CKPT}-${MODALITY}-bs${BATCH_SIZE}ep${EPOCHS}lr${LR}optadamw-defaulteval-trsub0-$2/checkpoint-best.pth"
}

ENC_SUFFIX="enc-smpweighted_sum-pre-0-fea-1-1-0.5-decoder_to_encoder-conv---"
FUSE_WS_SUFFIX="fuse-smpweighted_sum-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask---smp_learnable_alpha-"

# ════════════════════════════════════════════════════════════════════════════
# Task: DME   (Source: UF DME  ->  Target: OCTDL DME / OCT2017(CellData) DME)
# ════════════════════════════════════════════════════════════════════════════
DATASET=DME_binary_all_split
OCTDL_STUDY=DME_all
CELLDATA_STUDY=DME_all

RESUME_ENC=$(uf_resume ${DATASET} "${ENC_SUFFIX}")
RESUME_FUSE=$(uf_resume ${DATASET} "${FUSE_WS_SUFFIX}")

octdl_eval ${OCTDL_STUDY} "${RESUME_ENC}" enc 0 -1
celldata_eval ${CELLDATA_STUDY} "${RESUME_ENC}" enc 0 -1

# SMP(fuse,WS) — done
# octdl_eval ${OCTDL_STUDY} "${RESUME_FUSE}" fuse 9 -2 "--seg_mask" "--smp_learnable_alpha" "--smp_alpha_type ${ALPHA_TYPE}"
# celldata_eval ${CELLDATA_STUDY} "${RESUME_FUSE}" fuse 9 -2 "--seg_mask" "--smp_learnable_alpha" "--smp_alpha_type ${ALPHA_TYPE}"

# ════════════════════════════════════════════════════════════════════════════
# Task: AMD   (Source: UF AMD  ->  Target: OCTDL AMD)
# ════════════════════════════════════════════════════════════════════════════
DATASET=AMD_all_split
OCTDL_STUDY=AMD_all

RESUME_ENC=$(uf_resume ${DATASET} "${ENC_SUFFIX}")
RESUME_FUSE=$(uf_resume ${DATASET} "${FUSE_WS_SUFFIX}")

octdl_eval ${OCTDL_STUDY} "${RESUME_ENC}" enc 0 -1
octdl_eval ${OCTDL_STUDY} "${RESUME_FUSE}" fuse 9 -2 "--seg_mask" "--smp_learnable_alpha" "--smp_alpha_type ${ALPHA_TYPE}"

# ════════════════════════════════════════════════════════════════════════════
# Task: ERM   (Source: UF ERM  ->  Target: OCTDL ERM)
# ════════════════════════════════════════════════════════════════════════════
DATASET=ERM_all_split
OCTDL_STUDY=ERM_all

RESUME_ENC=$(uf_resume ${DATASET} "${ENC_SUFFIX}")
RESUME_FUSE=$(uf_resume ${DATASET} "${FUSE_WS_SUFFIX}")

octdl_eval ${OCTDL_STUDY} "${RESUME_ENC}" enc 0 -1
octdl_eval ${OCTDL_STUDY} "${RESUME_FUSE}" fuse 9 -2 "--seg_mask" "--smp_learnable_alpha" "--smp_alpha_type ${ALPHA_TYPE}"
