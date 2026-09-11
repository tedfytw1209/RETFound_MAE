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
# Reverse of KDD_UF_transfer_public_baselines.sh: public-dataset -> UF transfer
# eval, baseline architectures. One OCTDL/CellData(OCT2017)-trained checkpoint
# per (task, architecture); XAI eval run on the equivalent-disease UF split.
#
# finetune_retfound_UFbenchmark_v5_eval_full.sh takes a single XAI method per
# call (unlike the OCTDL/Celldata eval scripts, which loop XAI internally), so
# each combination below is wrapped in an explicit XAI_METHODS loop instead.
#
# Table (Task / Source / Target / RETFound / ViT / EfficientNet-b4 / Resnet-50):
#   DME  OCTDL DME          -> UF DME
#   DME  OCT2017(CellData) DME -> UF DME
#   AMD  OCTDL AMD          -> UF AMD
#   ERM  OCTDL ERM          -> UF ERM
# Nothing is marked done yet for this (reverse) direction — none commented out.
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR=/orange/ruogu.fang/tienyuchang/RETfound_results
MODALITY=OCT
NUM_CLASS=2
XAI_METHODS=("hirescam" "gradcam++" "gradcamv2")

# Baseline model pairs: MODEL (architecture) / FINETUNED_MODEL (checkpoint id) / INPUT_SIZE
# RETFound_mae         RETFound_mae_natureOCT              224
# vit-base-patch16-224 google/vit-base-patch16-224-in21k   224
# timm_efficientnet-b4 timm_efficientnet-b4                380
# resnet-50            microsoft/resnet-50                 224

uf_eval() {
    # $1=STUDY(UF) $2=MODEL $3=FINETUNED_MODEL $4=RESUME $5=INPUT_SIZE
    for XAI in "${XAI_METHODS[@]}"
    do
        sbatch finetune_retfound_UFbenchmark_v5_eval_full.sh "$1" "$2" "$3" "$4" ${NUM_CLASS} "$5" ${XAI} "$5"
    done
}

octdl_resume() {
    # $1=STUDY(OCTDL) $2=FINETUNED_MODEL
    echo "${RESULTS_DIR}/$1-OCTDL-all-$2-${MODALITY}-bs16ep50lr5e-4optadamw-defaulteval--/checkpoint-best.pth"
}

celldata_resume() {
    # $1=STUDY(CellData) $2=FINETUNED_MODEL
    echo "${RESULTS_DIR}/$1-CellData-all-$2-${MODALITY}-bs16ep3lr5e-4optadamw-defaulteval--/checkpoint-best.pth"
}

# ════════════════════════════════════════════════════════════════════════════
# Task: DME   (Source: OCTDL DME / OCT2017(CellData) DME  ->  Target: UF DME)
# ════════════════════════════════════════════════════════════════════════════
UF_DME=DME_binary_all_split
OCTDL_STUDY=DME_all
CELLDATA_STUDY=DME_all

RESUME_OCTDL_RETFound=$(octdl_resume ${OCTDL_STUDY} RETFound_mae_natureOCT)
RESUME_OCTDL_ViT=$(octdl_resume ${OCTDL_STUDY} google/vit-base-patch16-224-in21k)
RESUME_OCTDL_EffNet=$(octdl_resume ${OCTDL_STUDY} timm_efficientnet-b4)
RESUME_OCTDL_Resnet=$(octdl_resume ${OCTDL_STUDY} microsoft/resnet-50)

RESUME_CELL_RETFound=$(celldata_resume ${CELLDATA_STUDY} RETFound_mae_natureOCT)
RESUME_CELL_ViT=$(celldata_resume ${CELLDATA_STUDY} google/vit-base-patch16-224-in21k)
RESUME_CELL_EffNet=$(celldata_resume ${CELLDATA_STUDY} timm_efficientnet-b4)
RESUME_CELL_Resnet=$(celldata_resume ${CELLDATA_STUDY} microsoft/resnet-50)

uf_eval ${UF_DME} RETFound_mae RETFound_mae_natureOCT "${RESUME_OCTDL_RETFound}" 224
uf_eval ${UF_DME} vit-base-patch16-224 google/vit-base-patch16-224-in21k "${RESUME_OCTDL_ViT}" 224
uf_eval ${UF_DME} timm_efficientnet-b4 timm_efficientnet-b4 "${RESUME_OCTDL_EffNet}" 380
uf_eval ${UF_DME} resnet-50 microsoft/resnet-50 "${RESUME_OCTDL_Resnet}" 224

uf_eval ${UF_DME} RETFound_mae RETFound_mae_natureOCT "${RESUME_CELL_RETFound}" 224
uf_eval ${UF_DME} vit-base-patch16-224 google/vit-base-patch16-224-in21k "${RESUME_CELL_ViT}" 224
uf_eval ${UF_DME} timm_efficientnet-b4 timm_efficientnet-b4 "${RESUME_CELL_EffNet}" 380
uf_eval ${UF_DME} resnet-50 microsoft/resnet-50 "${RESUME_CELL_Resnet}" 224

# ════════════════════════════════════════════════════════════════════════════
# Task: AMD   (Source: OCTDL AMD  ->  Target: UF AMD)
# ════════════════════════════════════════════════════════════════════════════
UF_AMD=AMD_all_split
OCTDL_STUDY=AMD_all

RESUME_OCTDL_RETFound=$(octdl_resume ${OCTDL_STUDY} RETFound_mae_natureOCT)
RESUME_OCTDL_ViT=$(octdl_resume ${OCTDL_STUDY} google/vit-base-patch16-224-in21k)
RESUME_OCTDL_EffNet=$(octdl_resume ${OCTDL_STUDY} timm_efficientnet-b4)
RESUME_OCTDL_Resnet=$(octdl_resume ${OCTDL_STUDY} microsoft/resnet-50)

uf_eval ${UF_AMD} RETFound_mae RETFound_mae_natureOCT "${RESUME_OCTDL_RETFound}" 224
uf_eval ${UF_AMD} vit-base-patch16-224 google/vit-base-patch16-224-in21k "${RESUME_OCTDL_ViT}" 224
uf_eval ${UF_AMD} timm_efficientnet-b4 timm_efficientnet-b4 "${RESUME_OCTDL_EffNet}" 380
uf_eval ${UF_AMD} resnet-50 microsoft/resnet-50 "${RESUME_OCTDL_Resnet}" 224

# ════════════════════════════════════════════════════════════════════════════
# Task: ERM   (Source: OCTDL ERM  ->  Target: UF ERM)
# ════════════════════════════════════════════════════════════════════════════
UF_ERM=ERM_all_split
OCTDL_STUDY=ERM_all

RESUME_OCTDL_RETFound=$(octdl_resume ${OCTDL_STUDY} RETFound_mae_natureOCT)
RESUME_OCTDL_ViT=$(octdl_resume ${OCTDL_STUDY} google/vit-base-patch16-224-in21k)
RESUME_OCTDL_EffNet=$(octdl_resume ${OCTDL_STUDY} timm_efficientnet-b4)
RESUME_OCTDL_Resnet=$(octdl_resume ${OCTDL_STUDY} microsoft/resnet-50)

uf_eval ${UF_ERM} RETFound_mae RETFound_mae_natureOCT "${RESUME_OCTDL_RETFound}" 224
uf_eval ${UF_ERM} vit-base-patch16-224 google/vit-base-patch16-224-in21k "${RESUME_OCTDL_ViT}" 224
uf_eval ${UF_ERM} timm_efficientnet-b4 timm_efficientnet-b4 "${RESUME_OCTDL_EffNet}" 380
uf_eval ${UF_ERM} resnet-50 microsoft/resnet-50 "${RESUME_OCTDL_Resnet}" 224
