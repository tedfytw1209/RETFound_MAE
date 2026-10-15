#!/bin/bash
# multiseed_run_all.sh — submit every multi-seed experiment for ONE seed.
#   bash multiseed_run_all.sh 1952 train   # UF baselines + UF SMP + UF sail-enc + OCTDL/CellData baselines
#   bash multiseed_run_all.sh 1952 eval    # UF XAI eval (after the train jobs finish: it skips runs with no checkpoint yet)
#   DRY_RUN=1 bash multiseed_run_all.sh 1952 train   # print the commands, submit nothing
# env: SEED_MODE=split|train   UF seed mode (default split); OCTDL/CellData always vary the training seed only
#      DATASETS="..."          UF datasets (default: DME_binary AMD Glaucoma_binary ERM)
#      PUBLIC_DATASETS="..."   OCTDL/CellData studies (default: DME_all)
#      CHECK_CKPT=0            eval: submit even if the checkpoint is not there yet
#      ONLY=sail-enc           run only the sail-enc jobs (default: all; see multiseed_sail_enc_tmp.sh)
#      SAIL_ENC_XAI="rise"     XAI methods for the sail-enc eval (KDD_rise_enceval.sh uses rise)
# The original run (seed 42 in split mode) already exists, so it is refused here instead of overwriting it.
SEED=$1
STAGE=${2:-train}
if [ -z "$SEED" ]; then
    echo "usage: bash multiseed_run_all.sh <SEED> [train|eval]" >&2
    exit 1
fi
source ./multiseed_env.sh || exit 1
if [ -z "$RUN_SUFFIX" ]; then
    echo "[multiseed_run_all] SEED=$SEED SEED_MODE=$SEED_MODE is the original run; pick another seed" >&2
    exit 1
fi

DRY_RUN=${DRY_RUN:-0}
CHECK_CKPT=${CHECK_CKPT:-1}
DATASETS=${DATASETS:-"DME_binary_all_split AMD_all_split Glaucoma_binary_all_split ERM_all_split"}
PUBLIC_DATASETS=${PUBLIC_DATASETS:-"DME_all"}
ONLY=${ONLY:-all}
SAIL_ENC_XAI=${SAIL_ENC_XAI:-"rise"}
if [ "$ONLY" != "all" ] && [ "$ONLY" != "sail-enc" ]; then
    echo "unknown ONLY='$ONLY' (all|sail-enc)" >&2
    exit 1
fi
data_type="IRB2024_v5_all"
MODEL_DIR="/orange/ruogu.fang/tienyuchang/RETfound_results"
SEG_CKPT="/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth"

# UF SMP configuration (the single-seed example in README_OCTXAI.md / the driver comments: edit to match yours)
SMP_MODE="fuse"; SMP_FUSE="weighted_sum"; SMP_ALPHA="0.5"; SMP_SIZE_MATCH="decoder_to_encoder"
SMP_FUSION_DIM="9"; SMP_ALIGN="pre"; SMP_ENC_IDX="-2"; SMP_DEC_IDX="-1"; SMP_CLASSIFIER="conv"
SMP_ALPHA_TYPE="scalar"; SMP_ADD1="--seg_mask"; SMP_ADD2="--smp_learnable_alpha"
SMP_TARGET="head"; SMP_SELECT="-1" # XAI eval only

# sail-enc: SMP in enc mode, as in KDD_rise_enceval.sh (its RESUME dir carries no extra train flags, and its eval passes none)
SAIL_ENC_MODE="enc"; SAIL_ENC_FUSE="weighted_sum"; SAIL_ENC_ALPHA="0.5"; SAIL_ENC_SIZE_MATCH="decoder_to_encoder"
SAIL_ENC_FUSION_DIM="0"; SAIL_ENC_ALIGN="pre"; SAIL_ENC_ENC_IDX="-1"; SAIL_ENC_DEC_IDX="-1"; SAIL_ENC_CLASSIFIER="conv"
SAIL_ENC_ALPHA_TYPE="scalar"; SAIL_ENC_ADD1=""; SAIL_ENC_ADD2=""
SAIL_ENC_TARGET="head"; SAIL_ENC_SELECT="-1"; SAIL_ENC_STEP_PIXELS="1024" # XAI eval only
THICKNESS_DIR="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"

# the 4 baselines of baseline_multirun_XAI_model_train.sh: "MODEL FINETUNED_MODEL EVAL_INPUT_SIZE"
BASELINES=("RETFound_mae RETFound_mae_natureOCT 224" "resnet-50 microsoft/resnet-50 224"
           "vit-base-patch16-224 google/vit-base-patch16-224-in21k 224" "timm_efficientnet-b4 timm_efficientnet-b4 380")

SUBMITTED=0
SKIPPED=0
SPAWNED=0
# dry run only: run the train driver with sbatch stubbed out, to list the jobs it would submit
preview_driver() {
    local jobs
    jobs=$(
        sbatch() { shift; echo "    -> job: $*"; }
        export -f sbatch
        SEED=$SEED SEED_MODE=$SEED_MODE bash "$@" 2>/dev/null | grep '^    -> job: '
    )
    if [ -z "$jobs" ]; then
        echo "    -> (could not preview the jobs of $1)"
        return
    fi
    echo "$jobs"
    SPAWNED=$((SPAWNED + $(echo "$jobs" | wc -l)))
}
submit() {
    echo "SEED=$SEED SEED_MODE=$SEED_MODE $*"
    if [ "$DRY_RUN" != "1" ]; then
        SEED=$SEED SEED_MODE=$SEED_MODE "$@"
    elif [ "$2" = "baseline_multirun_XAI_model_train.sh" ]; then
        preview_driver "${@:2}"
    fi
    SUBMITTED=$((SUBMITTED + 1))
}
want() { [ "$ONLY" = "all" ] || [ "$ONLY" = "$1" ]; }
have_ckpt() {
    if [ "$CHECK_CKPT" = "0" ] || [ -f "$1" ]; then
        return 0
    fi
    echo "[skip] no checkpoint yet: $1" >&2
    SKIPPED=$((SKIPPED + 1))
    return 1
}

train_stage() {
    for D in $DATASETS; do
        if want all; then
            submit sbatch baseline_multirun_XAI_model_train.sh finetune_retfound_UFbenchmark_irb2024v5_full2.sh $D \
                RETFound_mae RETFound_mae_natureOCT 5e-4 2 0.05 default OCT
            submit sbatch finetune_retfound_UFbenchmark_irb2024v5_smp_full.sh $D SMP $SEG_CKPT 5e-4 2 1e-4 default OCT 0 \
                $SMP_MODE $SMP_FUSE $SMP_ALPHA $SMP_SIZE_MATCH $SMP_FUSION_DIM $SMP_ALIGN $SMP_ENC_IDX $SMP_DEC_IDX \
                $SMP_CLASSIFIER $SMP_ALPHA_TYPE $SMP_ADD1 $SMP_ADD2
        fi
        if want sail-enc; then
            submit sbatch finetune_retfound_UFbenchmark_irb2024v5_smp_full.sh $D SMP $SEG_CKPT 5e-4 2 1e-4 default OCT 0 \
                $SAIL_ENC_MODE $SAIL_ENC_FUSE $SAIL_ENC_ALPHA $SAIL_ENC_SIZE_MATCH $SAIL_ENC_FUSION_DIM $SAIL_ENC_ALIGN \
                $SAIL_ENC_ENC_IDX $SAIL_ENC_DEC_IDX $SAIL_ENC_CLASSIFIER $SAIL_ENC_ALPHA_TYPE $SAIL_ENC_ADD1 $SAIL_ENC_ADD2
        fi
    done
    want all || return 0
    for D in $PUBLIC_DATASETS; do
        for SCRIPT in finetune_retfound_OCTDL.sh finetune_retfound_Celldata.sh; do
            submit sbatch baseline_multirun_XAI_model_train.sh $SCRIPT $D \
                RETFound_mae RETFound_mae_natureOCT 5e-4 2 0.05 default OCT
        done
    done
}

eval_stage() {
    for D in $DATASETS; do
        if want all; then
            for B in "${BASELINES[@]}"; do
                read -r MODEL FT SIZE <<< "$B"
                have_ckpt "$MODEL_DIR/$D-$data_type-all-$FT-OCT-bs32ep50lr5e-4optadamw-defaulteval-trsub0--${RUN_SUFFIX}/checkpoint-best.pth" || continue
                submit sbatch baseline_multirun_XAI_eval_full.sh finetune_retfound_UFbenchmark_v5_eval_full2.sh $D 2 $MODEL $FT $SIZE
            done
            # original-run SMP dir: the eval driver appends the seed suffix to RESUME itself
            SMP_DIR="$MODEL_DIR/$D-$data_type-all-$SEG_CKPT-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-$SMP_MODE-smp$SMP_FUSE-$SMP_ALIGN-$SMP_FUSION_DIM-fea$SMP_ENC_IDX$SMP_DEC_IDX-$SMP_ALPHA-$SMP_SIZE_MATCH-$SMP_CLASSIFIER-alpha$SMP_ALPHA_TYPE-$SMP_ADD1-$SMP_ADD2-"
            if have_ckpt "${SMP_DIR}${RUN_SUFFIX}/checkpoint-best.pth"; then
                submit sbatch baseline_multirun_XAI_eval_smp.sh finetune_retfound_UFbenchmark_v5_eval_smp_full.sh $D 2 SMP $SEG_CKPT \
                    "$SMP_DIR/checkpoint-best.pth" 512 $SMP_MODE $SMP_FUSE $SMP_ALPHA $SMP_SIZE_MATCH $SMP_FUSION_DIM $SMP_ALIGN \
                    $SMP_ENC_IDX $SMP_DEC_IDX $SMP_TARGET $SMP_SELECT $SMP_CLASSIFIER $SMP_ADD1 $SMP_ADD2
            fi
        fi
        want sail-enc || continue
        # like KDD_rise_enceval.sh: the eval script is submitted directly (one job per XAI method), so RESUME carries the seed suffix here
        ENC_DIR="$MODEL_DIR/$D-$data_type-all-$SEG_CKPT-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-$SAIL_ENC_MODE-smp$SAIL_ENC_FUSE-$SAIL_ENC_ALIGN-$SAIL_ENC_FUSION_DIM-fea$SAIL_ENC_ENC_IDX$SAIL_ENC_DEC_IDX-$SAIL_ENC_ALPHA-$SAIL_ENC_SIZE_MATCH-$SAIL_ENC_CLASSIFIER-alpha$SAIL_ENC_ALPHA_TYPE-$SAIL_ENC_ADD1-$SAIL_ENC_ADD2-"
        have_ckpt "${ENC_DIR}${RUN_SUFFIX}/checkpoint-best.pth" || continue
        for X in $SAIL_ENC_XAI; do
            submit sbatch finetune_retfound_UFbenchmark_v5_eval_smp_full.sh $D SMP $SEG_CKPT \
                "${ENC_DIR}${RUN_SUFFIX}/checkpoint-best.pth" 2 512 $X $SAIL_ENC_STEP_PIXELS $THICKNESS_DIR \
                $SAIL_ENC_MODE $SAIL_ENC_FUSE $SAIL_ENC_ALPHA $SAIL_ENC_SIZE_MATCH $SAIL_ENC_FUSION_DIM $SAIL_ENC_ALIGN \
                $SAIL_ENC_ENC_IDX $SAIL_ENC_DEC_IDX $SAIL_ENC_TARGET $SAIL_ENC_SELECT $SAIL_ENC_CLASSIFIER $SAIL_ENC_ADD1 $SAIL_ENC_ADD2
        done
    done
}

case "$STAGE" in
    train) train_stage ;;
    eval) eval_stage ;;
    *) echo "unknown stage '$STAGE' (train|eval)" >&2; exit 1 ;;
esac
if [ "$DRY_RUN" = "1" ]; then
    echo "[multiseed_run_all] SEED=$SEED stage=$STAGE (dry run): $SUBMITTED sbatch calls + $SPAWNED jobs spawned by train drivers, $SKIPPED skipped"
else
    echo "[multiseed_run_all] SEED=$SEED stage=$STAGE: $SUBMITTED submitted, $SKIPPED skipped"
fi
