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
# Reverse of KDD_UF_transfer_public_smp.sh: public-dataset -> UF transfer eval,
# SMP variants (enc, fuse weighted_sum). One OCTDL/CellData(OCT2017)-trained
# SMP checkpoint per (task, SMP mode); XAI eval run on the equivalent-disease
# UF split.
#
# finetune_retfound_UFbenchmark_v5_eval_smp_full.sh takes a single XAI method
# per call (unlike the OCTDL/Celldata eval scripts, which loop XAI internally),
# so each combination below is wrapped in an explicit XAI_METHODS loop instead.
#
# Table (Task / Source / Target / SMP(enc) / SMP(fuse,WS)):
#   DME  OCTDL DME          -> UF DME
#   DME  OCT2017(CellData) DME -> UF DME
#   AMD  OCTDL AMD          -> UF AMD
#   ERM  OCTDL ERM          -> UF ERM
# Nothing is marked done yet for this (reverse) direction — none commented out.
# ─────────────────────────────────────────────────────────────────────────────

BASE_CKPT=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth
RESULTS_DIR=/orange/ruogu.fang/tienyuchang/RETfound_results
MODALITY=OCT
ALPHA_TYPE=scalar
NUM_CLASS=2
INPUT_SIZE=512
STEP_PIXELS=1024
XAI_METHODS=("hirescam" "gradcam++" "gradcamv2")

UF_THICKNESS_DIR=/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/

uf_smp_eval() {
    # $1=STUDY(UF) $2=RESUME $3=SMPMode $4=FUSION_DIM $5=ENC_IDX
    for XAI in "${XAI_METHODS[@]}"
    do
        sbatch finetune_retfound_UFbenchmark_v5_eval_smp_full.sh "$1" SMP ${BASE_CKPT} "$2" ${NUM_CLASS} ${INPUT_SIZE} ${XAI} ${STEP_PIXELS} ${UF_THICKNESS_DIR} "$3" weighted_sum 0.5 decoder_to_encoder "$4" pre "$5" -1 head -1 conv "${@:6}"
    done
}

octdl_smp_resume() {
    # $1=STUDY(OCTDL) $2=SUFFIX
    echo "${RESULTS_DIR}/$1-OCTDL-all-${BASE_CKPT}-${MODALITY}-bs16ep50lr1e-4optadamw-defaulteval-trsub0-$2/checkpoint-best.pth"
}

celldata_smp_resume() {
    # $1=STUDY(CellData) $2=SUFFIX
    echo "${RESULTS_DIR}/$1-CellData-all-${BASE_CKPT}-${MODALITY}-bs16ep5lr1e-4optadamw-defaulteval-trsub0-$2/checkpoint-best.pth"
}

ENC_SUFFIX="enc-smpweighted_sum-pre-0-fea-1-1-0.5-decoder_to_encoder-conv---"
FUSE_WS_SUFFIX="fuse-smpweighted_sum-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask---smp_learnable_alpha-"

# ════════════════════════════════════════════════════════════════════════════
# Task: DME   (Source: OCTDL DME / OCT2017(CellData) DME  ->  Target: UF DME)
# ════════════════════════════════════════════════════════════════════════════
UF_DME=DME_binary_all_split
OCTDL_STUDY=DME_all
CELLDATA_STUDY=DME_all

RESUME_OCTDL_ENC=$(octdl_smp_resume ${OCTDL_STUDY} "${ENC_SUFFIX}")
RESUME_OCTDL_FUSE=$(octdl_smp_resume ${OCTDL_STUDY} "${FUSE_WS_SUFFIX}")
RESUME_CELL_ENC=$(celldata_smp_resume ${CELLDATA_STUDY} "${ENC_SUFFIX}")
RESUME_CELL_FUSE=$(celldata_smp_resume ${CELLDATA_STUDY} "${FUSE_WS_SUFFIX}")

uf_smp_eval ${UF_DME} "${RESUME_OCTDL_ENC}" enc 0 -1
uf_smp_eval ${UF_DME} "${RESUME_OCTDL_FUSE}" fuse 9 -2 "--seg_mask" "--smp_learnable_alpha" "--smp_alpha_type ${ALPHA_TYPE}"

uf_smp_eval ${UF_DME} "${RESUME_CELL_ENC}" enc 0 -1
uf_smp_eval ${UF_DME} "${RESUME_CELL_FUSE}" fuse 9 -2 "--seg_mask" "--smp_learnable_alpha" "--smp_alpha_type ${ALPHA_TYPE}"

# ════════════════════════════════════════════════════════════════════════════
# Task: AMD   (Source: OCTDL AMD  ->  Target: UF AMD)
# ════════════════════════════════════════════════════════════════════════════
UF_AMD=AMD_all_split
OCTDL_STUDY=AMD_all

RESUME_OCTDL_ENC=$(octdl_smp_resume ${OCTDL_STUDY} "${ENC_SUFFIX}")
RESUME_OCTDL_FUSE=$(octdl_smp_resume ${OCTDL_STUDY} "${FUSE_WS_SUFFIX}")

uf_smp_eval ${UF_AMD} "${RESUME_OCTDL_ENC}" enc 0 -1
uf_smp_eval ${UF_AMD} "${RESUME_OCTDL_FUSE}" fuse 9 -2 "--seg_mask" "--smp_learnable_alpha" "--smp_alpha_type ${ALPHA_TYPE}"

# ════════════════════════════════════════════════════════════════════════════
# Task: ERM   (Source: OCTDL ERM  ->  Target: UF ERM)
# ════════════════════════════════════════════════════════════════════════════
UF_ERM=ERM_all_split
OCTDL_STUDY=ERM_all

RESUME_OCTDL_ENC=$(octdl_smp_resume ${OCTDL_STUDY} "${ENC_SUFFIX}")
RESUME_OCTDL_FUSE=$(octdl_smp_resume ${OCTDL_STUDY} "${FUSE_WS_SUFFIX}")

uf_smp_eval ${UF_ERM} "${RESUME_OCTDL_ENC}" enc 0 -1
uf_smp_eval ${UF_ERM} "${RESUME_OCTDL_FUSE}" fuse 9 -2 "--seg_mask" "--smp_learnable_alpha" "--smp_alpha_type ${ALPHA_TYPE}"
