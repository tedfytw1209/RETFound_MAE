#!/bin/bash
# multiseed_run_all.sh — submit every multi-seed experiment for ONE seed.
#   bash multiseed_run_all.sh 1952 train   # UF baselines + UF SMP + UF sail-enc + OCTDL/CellData baselines
#   bash multiseed_run_all.sh 1952 eval    # UF XAI eval (after the train jobs finish: it skips runs that are not ready yet)
#   DRY_RUN=1 bash multiseed_run_all.sh 1952 train   # print the commands, submit nothing
# env: SEED_MODE=split|train   UF seed mode (default split); OCTDL/CellData always vary the training seed only
#      DATASETS="..."          UF datasets (default: DME_binary AMD Glaucoma_binary ERM)
#      PUBLIC_DATASETS="..."   OCTDL/CellData studies (default: DME_all)
#      MAX_SLURM=N             never let your queue (squeue, pending + running) go above N jobs: stop and report what is left
#      RERUN_DONE=1            resubmit jobs that already COMPLETED (default: skip them)
#      CHECK_CKPT=0            eval: submit even if the checkpoint is not there yet / the training job is still running
#      ONLY=sail-enc           run only the sail-enc jobs (default: all; see multiseed_sail_enc_tmp.sh)
#      SAIL_ENC_XAI="rise"     XAI methods for the sail-enc eval (KDD_rise_enceval.sh uses rise)
# Resume: every job is named ms-<mode>-<seed>-<kind>-..., and a job is skipped when that name is already in squeue
# (queued/running) or COMPLETED in sacct (last SACCT_DAYS days, default 90). So after a MAX_SLURM stop, a failure, or a
# timeout, rerunning the same command submits only what is missing. Failed/cancelled/timed-out jobs are resubmitted.
# The baselines are submitted one job per model (not through baseline_multirun_XAI_model_train.sh, which always submits
# all four and spawns its jobs later), so each model is resumed and counted on its own.
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
MAX_SLURM=${MAX_SLURM:-0}
RERUN_DONE=${RERUN_DONE:-0}
SACCT_DAYS=${SACCT_DAYS:-90}
DATASETS=${DATASETS:-"DME_binary_all_split AMD_all_split Glaucoma_binary_all_split ERM_all_split"}
PUBLIC_DATASETS=${PUBLIC_DATASETS:-"DME_all"}
ONLY=${ONLY:-all}
SAIL_ENC_XAI=${SAIL_ENC_XAI:-"rise"}
if [ "$ONLY" != "all" ] && [ "$ONLY" != "sail-enc" ]; then
    echo "unknown ONLY='$ONLY' (all|sail-enc)" >&2
    exit 1
fi
case "$MAX_SLURM" in
    ''|*[!0-9]*) echo "MAX_SLURM must be a non-negative integer (0 = no limit), got '$MAX_SLURM'" >&2; exit 1 ;;
esac
data_type="IRB2024_v5_all"
MODEL_DIR=${MODEL_DIR:-"/orange/ruogu.fang/tienyuchang/RETfound_results"}
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

# job names: ms-<mode>-<seed>-<kind>-<dataset>[-<model>]; the public-data jobs always vary the training seed only
TAG="$SEED_MODE-$SEED"
PUB_TAG="train-$SEED"

SUBMITTED=0
SKIPPED=0
SKIP_DONE=0
SKIP_QUEUED=0
LEFT=0
LIMIT_HIT=0
SLURM_USER=${USER:-$(whoami)}
HAVE_SLURM=0
if command -v squeue >/dev/null 2>&1 && command -v sacct >/dev/null 2>&1; then
    HAVE_SLURM=1
else
    echo "[multiseed_run_all] squeue/sacct not found: resume and MAX_SLURM are not checked" >&2
fi

in_list() { printf '%s\n' "$1" | grep -Fxq -- "$2"; }
queued_now() { if [ "$HAVE_SLURM" = "1" ]; then squeue -u "$SLURM_USER" -h -o %j 2>/dev/null; fi; return 0; }
DONE_NAMES=""
if [ "$HAVE_SLURM" = "1" ] && [ "$RERUN_DONE" != "1" ]; then
    DONE_NAMES=$(sacct -u "$SLURM_USER" -X -n -P -S "now-${SACCT_DAYS}days" --state=COMPLETED --format=JobName 2>/dev/null)
fi

# submit <job-name> <script> [args...]: skips what is queued/COMPLETED and stops at MAX_SLURM
submit() {
    local name=$1
    shift
    if [ "$LIMIT_HIT" = "1" ]; then
        LEFT=$((LEFT + 1))
        return
    fi
    local queued
    queued=$(queued_now)
    if in_list "$queued" "$name"; then
        echo "[skip] already queued/running: $name" >&2
        SKIP_QUEUED=$((SKIP_QUEUED + 1))
        return
    fi
    if in_list "$DONE_NAMES" "$name"; then
        echo "[skip] already done: $name" >&2
        SKIP_DONE=$((SKIP_DONE + 1))
        return
    fi
    if [ "$MAX_SLURM" -gt 0 ]; then
        local count
        count=$(printf '%s\n' "$queued" | grep -c .)
        if [ "$DRY_RUN" = "1" ]; then
            count=$((count + SUBMITTED)) # nothing was really submitted: count what this dry run would have added
        fi
        if [ "$count" -ge "$MAX_SLURM" ]; then
            echo "[limit] $count jobs in the queue >= MAX_SLURM=$MAX_SLURM: stopping here (rerun later; finished jobs are skipped)" >&2
            LIMIT_HIT=1
            LEFT=$((LEFT + 1))
            return
        fi
    fi
    echo "SEED=$SEED SEED_MODE=$SEED_MODE sbatch --job-name=$name $*"
    if [ "$DRY_RUN" != "1" ]; then
        SEED=$SEED SEED_MODE=$SEED_MODE sbatch --job-name="$name" "$@"
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
# true (and a note) while the training job behind a checkpoint is still queued/running: checkpoint-best.pth exists after epoch 1
train_running() {
    if [ "$CHECK_CKPT" = "0" ] || ! in_list "$(queued_now)" "$1"; then
        return 1
    fi
    echo "[skip] training still queued/running: $1" >&2
    SKIPPED=$((SKIPPED + 1))
    return 0
}

train_stage() {
    local B MODEL FT SIZE
    for D in $DATASETS; do
        if want all; then
            for B in "${BASELINES[@]}"; do
                read -r MODEL FT SIZE <<< "$B"
                submit "ms-$TAG-tr-base-$D-$MODEL" finetune_retfound_UFbenchmark_irb2024v5_full2.sh $D $MODEL $FT \
                    5e-4 2 0.05 default OCT 0
            done
            submit "ms-$TAG-tr-sail-$D" finetune_retfound_UFbenchmark_irb2024v5_smp_full.sh $D SMP $SEG_CKPT 5e-4 2 1e-4 default OCT 0 \
                $SMP_MODE $SMP_FUSE $SMP_ALPHA $SMP_SIZE_MATCH $SMP_FUSION_DIM $SMP_ALIGN $SMP_ENC_IDX $SMP_DEC_IDX \
                $SMP_CLASSIFIER $SMP_ALPHA_TYPE $SMP_ADD1 $SMP_ADD2
        fi
        if want sail-enc; then
            submit "ms-$TAG-tr-enc-$D" finetune_retfound_UFbenchmark_irb2024v5_smp_full.sh $D SMP $SEG_CKPT 5e-4 2 1e-4 default OCT 0 \
                $SAIL_ENC_MODE $SAIL_ENC_FUSE $SAIL_ENC_ALPHA $SAIL_ENC_SIZE_MATCH $SAIL_ENC_FUSION_DIM $SAIL_ENC_ALIGN \
                $SAIL_ENC_ENC_IDX $SAIL_ENC_DEC_IDX $SAIL_ENC_CLASSIFIER $SAIL_ENC_ALPHA_TYPE $SAIL_ENC_ADD1 $SAIL_ENC_ADD2
        fi
    done
    want all || return 0
    for D in $PUBLIC_DATASETS; do
        for PUB in OCTDL Celldata; do
            for B in "${BASELINES[@]}"; do
                read -r MODEL FT SIZE <<< "$B"
                submit "ms-$PUB_TAG-tr-$PUB-$D-$MODEL" finetune_retfound_$PUB.sh $D $MODEL $FT 5e-4 2 0.05 default OCT 0
            done
        done
    done
}

eval_stage() {
    local B MODEL FT SIZE
    for D in $DATASETS; do
        if want all; then
            for B in "${BASELINES[@]}"; do
                read -r MODEL FT SIZE <<< "$B"
                train_running "ms-$TAG-tr-base-$D-$MODEL" && continue
                have_ckpt "$MODEL_DIR/$D-$data_type-all-$FT-OCT-bs32ep50lr5e-4optadamw-defaulteval-trsub0--${RUN_SUFFIX}/checkpoint-best.pth" || continue
                submit "ms-$TAG-ev-base-$D-$MODEL" baseline_multirun_XAI_eval_full.sh finetune_retfound_UFbenchmark_v5_eval_full2.sh $D 2 $MODEL $FT $SIZE
            done
            # original-run SMP dir: the eval driver appends the seed suffix to RESUME itself
            SMP_DIR="$MODEL_DIR/$D-$data_type-all-$SEG_CKPT-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-$SMP_MODE-smp$SMP_FUSE-$SMP_ALIGN-$SMP_FUSION_DIM-fea$SMP_ENC_IDX$SMP_DEC_IDX-$SMP_ALPHA-$SMP_SIZE_MATCH-$SMP_CLASSIFIER-alpha$SMP_ALPHA_TYPE-$SMP_ADD1-$SMP_ADD2-"
            if ! train_running "ms-$TAG-tr-sail-$D" && have_ckpt "${SMP_DIR}${RUN_SUFFIX}/checkpoint-best.pth"; then
                submit "ms-$TAG-ev-sail-$D" baseline_multirun_XAI_eval_smp.sh finetune_retfound_UFbenchmark_v5_eval_smp_full.sh $D 2 SMP $SEG_CKPT \
                    "$SMP_DIR/checkpoint-best.pth" 512 $SMP_MODE $SMP_FUSE $SMP_ALPHA $SMP_SIZE_MATCH $SMP_FUSION_DIM $SMP_ALIGN \
                    $SMP_ENC_IDX $SMP_DEC_IDX $SMP_TARGET $SMP_SELECT $SMP_CLASSIFIER $SMP_ADD1 $SMP_ADD2
            fi
        fi
        want sail-enc || continue
        # like KDD_rise_enceval.sh: the eval script is submitted directly (one job per XAI method), so RESUME carries the seed suffix here
        ENC_DIR="$MODEL_DIR/$D-$data_type-all-$SEG_CKPT-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-$SAIL_ENC_MODE-smp$SAIL_ENC_FUSE-$SAIL_ENC_ALIGN-$SAIL_ENC_FUSION_DIM-fea$SAIL_ENC_ENC_IDX$SAIL_ENC_DEC_IDX-$SAIL_ENC_ALPHA-$SAIL_ENC_SIZE_MATCH-$SAIL_ENC_CLASSIFIER-alpha$SAIL_ENC_ALPHA_TYPE-$SAIL_ENC_ADD1-$SAIL_ENC_ADD2-"
        train_running "ms-$TAG-tr-enc-$D" && continue
        have_ckpt "${ENC_DIR}${RUN_SUFFIX}/checkpoint-best.pth" || continue
        for X in $SAIL_ENC_XAI; do
            submit "ms-$TAG-ev-enc-$D-$X" finetune_retfound_UFbenchmark_v5_eval_smp_full.sh $D SMP $SEG_CKPT \
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
SUMMARY="$SUBMITTED submitted$([ "$DRY_RUN" = "1" ] && echo ' (dry run)'), $SKIP_DONE already done, $SKIP_QUEUED already queued, $SKIPPED not ready"
if [ "$LIMIT_HIT" = "1" ]; then
    SUMMARY="$SUMMARY, $LEFT left: MAX_SLURM=$MAX_SLURM reached, rerun to continue"
fi
echo "[multiseed_run_all] SEED=$SEED stage=$STAGE: $SUMMARY"
