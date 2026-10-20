#!/bin/bash
# multiseed_run_all.sh — submit every multi-seed experiment for ONE seed.
#   baseline seeds: 1952 5235 8234 8386
#   bash multiseed_run_all.sh 1952 train   # UF baselines + UF SMP + UF sail-enc + OCTDL/CellData baselines
#   bash multiseed_run_all.sh 1952 eval    # UF XAI eval (after the train jobs finish: it skips runs that are not ready yet)
#   DRY_RUN=1 bash multiseed_run_all.sh 1952 train   # print the commands, submit nothing
# env: SEED_MODE=split|train   UF seed mode (default split); OCTDL/CellData always vary the training seed only
#      DATASETS="..."          UF datasets (default: DME_binary AMD Glaucoma_binary ERM)
#      PUBLIC_DATASETS="..."   OCTDL/CellData studies (default: DME_all)
#      MAX_SLURM=N             never let your queue (squeue, pending + running) go above N jobs: stop and report what is left
#      RERUN_DONE=1            resubmit jobs that already finished (default: skip them)
#      WANDB_CHECK=0           do not read wandb (default: read it when it can be queried; WANDB_ENTITY=<team> if needed)
#      CHECK_CKPT=0            eval: submit even if the checkpoint is not there yet / the training job is still running
#      ONLY=sail-enc           run only the sail-enc jobs (default: all; see multiseed_sail_enc_tmp.sh)
#      SAIL_ENC_XAI="rise"     XAI methods for the sail-enc eval (KDD_rise_enceval.sh uses rise)
# Resume: rerunning the same command submits only what is missing. wandb (multiseed_wandb_status.py) is read first,
# because that is what aggregate_multiseed.py aggregates, and it also recognizes runs submitted before the ms-... job
# names existed (the run names are deterministic; eval runs are matched by checkpoint + XAI method):
#   - every wandb run of the job FINISHED with test_* metrics  -> skipped
#   - a wandb run is still RUNNING (any of the job's runs)      -> skipped, with a "[skip] already running" note
#   - a wandb run exists but did not finish (crashed/failed/killed, or only some XAI methods) -> rerun, whatever
#     SLURM says, with a "[resume] <job>: wandb run(s) not finished (...)" note
#   - no wandb run (not run yet), or wandb not readable (then a note says so) -> the SLURM rules below decide
#     (never submitted -> submitted; a job queued in squeue is skipped)
# (a run whose job died is shown as running until wandb notices the lost heartbeat and marks it crashed: rerun later)
# SLURM: every job is named ms-<mode>-<seed>-<kind>-...:
#   - already in squeue (queued/running)            -> skipped
#   - COMPLETED in sacct (last SACCT_DAYS days)     -> skipped, but an eval job must also have logged every XAI
#     evaluation (the eval drivers run their methods inline, so a method killed for memory does not fail the job);
#     otherwise it is rerun and the note says "only n/m XAI evaluations finished" (the rerun reuses the cached heatmaps)
#   - last attempt CANCELLED/FAILED/OUT_OF_MEMORY/TIMEOUT/... -> rerun, with a "[resume] <job>: last attempt <STATE>" note
#   - never submitted                               -> submitted
# Training cannot continue mid-run (only checkpoint-best is saved), so a cancelled training restarts from epoch 0.
# eval waits for the training: it is only submitted once the training job COMPLETED (or, for a run made before these job
# names existed, once log.txt reached the last epoch). checkpoint-best.pth alone is not enough: it exists after epoch 1,
# also for a training that was cancelled half way.
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
WANDB_CHECK=${WANDB_CHECK:-1}
WANDB_STATUS_CMD=${WANDB_STATUS_CMD:-"python multiseed_wandb_status.py"}
WANDB_ENTITY=${WANDB_ENTITY:-}
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

# epochs of the training scripts (Epochs= in finetune_..._full2.sh / ..._smp_full.sh): the last epoch marks a finished run in log.txt
BASELINE_EPOCHS=50
SMP_EPOCHS=100
# XAI methods looped by baseline_multirun_XAI_eval_full.sh / _smp.sh: a finished driver job has one wandb run (and one log line) per method
EVAL_FULL_XAI="hirescam gradcam++ gradcamv2"
EVAL_SMP_XAI="hirescam gradcamv2 gradcam++"

# the 4 baselines of baseline_multirun_XAI_model_train.sh: "MODEL FINETUNED_MODEL EVAL_INPUT_SIZE"
BASELINES=("RETFound_mae RETFound_mae_natureOCT 224" "resnet-50 microsoft/resnet-50 224"
           "vit-base-patch16-224 google/vit-base-patch16-224-in21k 224" "timm_efficientnet-b4 timm_efficientnet-b4 380")

# job names: ms-<mode>-<seed>-<kind>-<dataset>[-<model>]; the public-data jobs always vary the training seed only
TAG="$SEED_MODE-$SEED"
PUB_TAG="train-$SEED"
PUB_SUFFIX="-seed$SEED"

# Run names = the training scripts' --task (the output dir name, and the wandb run name with a trailing "/"), without
# MODEL_DIR. The seed suffix is added by main_finetune*.py (util/multiseed.py), mirrored here by RUN_SUFFIX.
base_run() { echo "$1-$data_type-all-$2-OCT-bs32ep50lr5e-4optadamw-defaulteval-trsub0--${RUN_SUFFIX}"; } # <dataset> <finetuned model>
smp_run() { echo "$1-$data_type-all-$SEG_CKPT-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-$SMP_MODE-smp$SMP_FUSE-$SMP_ALIGN-$SMP_FUSION_DIM-fea$SMP_ENC_IDX$SMP_DEC_IDX-$SMP_ALPHA-$SMP_SIZE_MATCH-$SMP_CLASSIFIER-alpha$SMP_ALPHA_TYPE-$SMP_ADD1-$SMP_ADD2-"; } # <dataset>, no seed suffix
enc_run() { echo "$1-$data_type-all-$SEG_CKPT-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-$SAIL_ENC_MODE-smp$SAIL_ENC_FUSE-$SAIL_ENC_ALIGN-$SAIL_ENC_FUSION_DIM-fea$SAIL_ENC_ENC_IDX$SAIL_ENC_DEC_IDX-$SAIL_ENC_ALPHA-$SAIL_ENC_SIZE_MATCH-$SAIL_ENC_CLASSIFIER-alpha$SAIL_ENC_ALPHA_TYPE-$SAIL_ENC_ADD1-$SAIL_ENC_ADD2-"; } # <dataset>, no seed suffix
pub_run() { # <OCTDL|Celldata> <dataset> <finetuned model>
    case "$1" in
        OCTDL) echo "$2-OCTDL-all-$3-OCT-bs16ep50lr5e-4optadamw-defaulteval--$PUB_SUFFIX" ;;
        Celldata) echo "$2-CellData-all-$3-OCT-bs16ep3lr5e-4optadamw-defaulteval--$PUB_SUFFIX" ;;
    esac
}
# wandb keys of an eval job: one "<checkpoint>|<xai method>" per method (what the eval runs record as config.resume / config.xai)
eval_keys() { local x; for x in $2; do printf '%s|%s\n' "$1" "$x"; done; }
count_words() { echo $#; }

SUBMITTED=0
SKIPPED=0
SKIP_DONE=0
SKIP_QUEUED=0
RESUMED=0
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
# every attempt of our jobs, as name|state|jobid|workdir (sacct keeps cancelled/failed/timed-out ones too)
SACCT_ROWS=""
if [ "$HAVE_SLURM" = "1" ]; then
    SACCT_ROWS=$(sacct -u "$SLURM_USER" -X -n -P -S "now-${SACCT_DAYS}days" --format=JobName,State,JobID,WorkDir 2>/dev/null | grep '^ms-')
fi
# "completed" if any attempt of this job name COMPLETED, else the state of its newest attempt
# (CANCELLED, FAILED, OUT_OF_MEMORY, TIMEOUT, NODE_FAIL, ...), else "none"
job_state() {
    printf '%s\n' "$SACCT_ROWS" | awk -F'|' -v n="$1" '
        $1 == n { split($2, s, " "); if (s[1] == "COMPLETED") done = 1; if ($3 + 0 >= last) { last = $3 + 0; st = s[1] } }
        END { if (done) print "completed"; else if (st != "") print st; else print "none" }'
}
# A COMPLETED eval job is not necessarily a finished one: the eval drivers run their XAI methods inline, so a method
# that dies (e.g. killed for memory) does not fail the job when a later one succeeds. Each finished XAI evaluation prints
# "XAI Evaluation time" into the job log (<workdir>/<job-name>.<jobid>.out), so count those.
LAST_EVAL_NOTE=""
evals_finished() { # <job-name> <expected number of XAI evaluations>
    local rows id wd out n seen=0
    LAST_EVAL_NOTE=""
    rows=$(printf '%s\n' "$SACCT_ROWS" | awk -F'|' -v n="$1" '$1 == n && $2 ~ /^COMPLETED/' | sort -t'|' -k3,3nr)
    while IFS='|' read -r _ _ id wd; do
        out="$wd/$1.$id.out"
        [ -n "$id" ] && [ -f "$out" ] || continue
        seen=1
        n=$(grep -c 'XAI Evaluation time' "$out")
        if [ "$n" -ge "$2" ]; then
            return 0
        fi
        LAST_EVAL_NOTE="only $n/$2 XAI evaluations finished in $out"
    done <<< "$rows"
    if [ "$seen" = "1" ]; then
        return 1
    fi
    LAST_EVAL_NOTE="unverified: no job log found"
    return 0
}

# --- wandb: the record aggregate_multiseed.py reads (only runs that FINISHED with test_* metrics count there) -----------
# It also covers runs submitted before the ms-... job names existed, since the wandb run names are deterministic.
# When wandb cannot be read (not installed in this python, no login, no network) the SLURM state alone decides.
WB_P_TRAIN="RETFound_MAE"; WB_P_SMP="RETFound_MAE_XAI"; WB_P_EVAL="RETFound_MAE_XAI_Evaluation"
WB_REGEX="-s${SEED}/?\$|-seed${SEED}" # run names carry -s<seed> (split mode) or -seed<seed> (train mode)
# wb_load <project>: fetch the project's run table once (into $WBT_<project>) in THIS shell; fails when wandb is unavailable
wb_load() {
    local project=$1 var="WBT_$1" failed="WBF_$1" kind=train out err rc
    if [ "$WANDB_CHECK" != "1" ] || [ -n "${!failed}" ]; then
        return 1
    fi
    if [ -z "${!var+x}" ]; then
        if [ "$project" = "$WB_P_EVAL" ]; then
            kind=eval
        fi
        err=$(mktemp)
        out=$($WANDB_STATUS_CMD --project "$project" --kind "$kind" --regex="$WB_REGEX" ${WANDB_ENTITY:+--entity "$WANDB_ENTITY"} 2>"$err")
        rc=$?
        if [ "$rc" -ne 0 ]; then
            echo "[multiseed_run_all] wandb status of $project unavailable ($(head -n 1 "$err")): using SLURM state only" >&2
            printf -v "$failed" '%s' 1
            rm -f "$err"
            return 1
        fi
        rm -f "$err"
        printf -v "$var" '%s' "$out"
    fi
}
# wb_verdict <project> <newline-separated keys>: WB_VERDICT = done (every key finished) | running (not done, and some key is
# still running: the job is in flight) | partial (some run exists, none running, but not every key finished: crashed/failed/
# killed/...) | none (no run at all) | off (wandb unavailable); WB_NOTE says which key is in which state
WB_VERDICT=off
WB_NOTE=""
wb_verdict() {
    local project=$1 var="WBT_$1" key st label total=0 finished=0 existing=0 running=0
    WB_VERDICT=off
    WB_NOTE=""
    wb_load "$project" || return 0
    while IFS= read -r key; do
        [ -n "$key" ] || continue
        total=$((total + 1))
        st=$(printf '%s\n' "${!var}" | awk -F'\t' -v k="$key" '$1 == k { print $2; exit }')
        if [ "$st" = "finished" ]; then
            finished=$((finished + 1))
            continue
        fi
        label=""
        case "$key" in *'|'*) label="${key##*|}=" ;; esac
        WB_NOTE="${WB_NOTE:+$WB_NOTE, }$label${st:-no run}"
        if [ -n "$st" ]; then
            existing=$((existing + 1))
        fi
        if [ "$st" = "running" ]; then
            running=$((running + 1))
        fi
    done <<< "$2"
    if [ "$total" -gt 0 ] && [ "$finished" -eq "$total" ]; then
        WB_VERDICT=done
    elif [ "$running" -gt 0 ]; then
        WB_VERDICT=running
        WB_NOTE="run(s) still running ($WB_NOTE)"
    elif [ "$finished" -eq 0 ] && [ "$existing" -eq 0 ]; then
        WB_VERDICT=none
    else
        WB_VERDICT=partial
        WB_NOTE="run(s) not finished ($WB_NOTE)"
    fi
}

# submit <job-name> <script> [args...]: skips what is queued/finished and stops at MAX_SLURM.
# Optional env for one call: WB_PROJECT + WB_KEYS (wandb runs of this job), EXPECT_EVALS (XAI evaluations in its log).
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
    local state resume_msg="" no_wandb=""
    state=$(job_state "$name")
    LAST_EVAL_NOTE=""
    WB_VERDICT=off
    if [ -n "${WB_PROJECT:-}" ]; then
        wb_verdict "$WB_PROJECT" "$WB_KEYS"
    fi
    if [ "$WB_VERDICT" = "done" ] && [ "$RERUN_DONE" != "1" ]; then
        echo "[skip] already done (wandb: finished): $name" >&2
        SKIP_DONE=$((SKIP_DONE + 1))
        return
    fi
    if [ "$WB_VERDICT" = "running" ]; then
        echo "[skip] already running (wandb: $WB_NOTE): $name" >&2
        SKIP_QUEUED=$((SKIP_QUEUED + 1))
        return
    fi
    if [ "$WB_VERDICT" = "partial" ]; then
        # wandb is what gets aggregated: a run that crashed/failed there must be redone whatever SLURM says
        resume_msg="[resume] $name: wandb $WB_NOTE"
        if [ "$state" != "none" ]; then
            resume_msg="$resume_msg, SLURM job $state"
        fi
        resume_msg="$resume_msg -> resubmitting"
    else
        if [ "$WB_VERDICT" = "none" ]; then
            no_wandb=" [warning: no wandb run found, so aggregate_multiseed.py will not see it]"
        fi
        if [ "$state" = "completed" ] && [ "$RERUN_DONE" != "1" ]; then
            if [ "${EXPECT_EVALS:-0}" -eq 0 ] || evals_finished "$name" "$EXPECT_EVALS"; then
                echo "[skip] already done${LAST_EVAL_NOTE:+ ($LAST_EVAL_NOTE)}${no_wandb}: $name" >&2
                SKIP_DONE=$((SKIP_DONE + 1))
                return
            fi
            resume_msg="[resume] $name: job COMPLETED but $LAST_EVAL_NOTE -> resubmitting"
        elif [ "$state" != "none" ] && [ "$state" != "completed" ]; then
            resume_msg="[resume] $name: last attempt $state -> resubmitting"
        fi
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
    if [ -n "$resume_msg" ]; then
        echo "$resume_msg" >&2
        RESUMED=$((RESUMED + 1))
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
# Succeeds only when the training behind a checkpoint has really finished. checkpoint-best.pth exists after the first
# improving epoch, so it also exists for a training that was cancelled or failed half way: never evaluate that one.
#   queued/running            -> wait
#   a COMPLETED job record    -> finished
#   only failed/cancelled/... -> not finished (rerun the train stage; training restarts from epoch 0, it only keeps checkpoint-best)
#   no job record at all      -> a run made before these job names existed: finished if log.txt reached the last epoch
train_finished() { # <train job name> <run dir> <epochs> <wandb project> <wandb run name>
    local state last
    if [ "$CHECK_CKPT" = "0" ]; then
        return 0
    fi
    if in_list "$(queued_now)" "$1"; then
        echo "[skip] training still queued/running: $1" >&2
        SKIPPED=$((SKIPPED + 1))
        return 1
    fi
    wb_verdict "$4" "$5"
    case "$WB_VERDICT" in
        done) return 0 ;;
        running)
            echo "[skip] training still running: wandb $WB_NOTE for $1" >&2
            SKIPPED=$((SKIPPED + 1))
            return 1
            ;;
        partial)
            echo "[skip] training not finished: wandb $WB_NOTE for $1 (rerun the train stage)" >&2
            SKIPPED=$((SKIPPED + 1))
            return 1
            ;;
    esac
    state=$(job_state "$1")
    case "$state" in
        completed) return 0 ;;
        none)
            last=$(tail -n 1 "$2/log.txt" 2>/dev/null | sed -n 's/.*"epoch": \([0-9][0-9]*\).*/\1/p')
            if [ "$last" = "$(($3 - 1))" ]; then
                return 0
            fi
            if [ -n "$last" ]; then
                echo "[skip] training not finished: no job record for $1 and log.txt stops at epoch $last of $3" >&2
            else
                echo "[skip] training not finished: no job record for $1 and no log.txt in $2" >&2
            fi
            ;;
        *) echo "[skip] training not finished: last attempt of $1 was $state (rerun the train stage)" >&2 ;;
    esac
    SKIPPED=$((SKIPPED + 1))
    return 1
}

train_stage() {
    local B MODEL FT SIZE
    for D in $DATASETS; do
        if want all; then
            for B in "${BASELINES[@]}"; do
                read -r MODEL FT SIZE <<< "$B"
                WB_PROJECT=$WB_P_TRAIN WB_KEYS="$(base_run $D $FT)/" \
                    submit "ms-$TAG-tr-base-$D-$MODEL" finetune_retfound_UFbenchmark_irb2024v5_full2.sh $D $MODEL $FT \
                    5e-4 2 0.05 default OCT 0
            done
            WB_PROJECT=$WB_P_SMP WB_KEYS="$(smp_run $D)${RUN_SUFFIX}/" \
                submit "ms-$TAG-tr-sail-$D" finetune_retfound_UFbenchmark_irb2024v5_smp_full.sh $D SMP $SEG_CKPT 5e-4 2 1e-4 default OCT 0 \
                $SMP_MODE $SMP_FUSE $SMP_ALPHA $SMP_SIZE_MATCH $SMP_FUSION_DIM $SMP_ALIGN $SMP_ENC_IDX $SMP_DEC_IDX \
                $SMP_CLASSIFIER $SMP_ALPHA_TYPE $SMP_ADD1 $SMP_ADD2
        fi
        if want sail-enc; then
            WB_PROJECT=$WB_P_SMP WB_KEYS="$(enc_run $D)${RUN_SUFFIX}/" \
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
                WB_PROJECT=$WB_P_TRAIN WB_KEYS="$(pub_run $PUB $D $FT)/" \
                    submit "ms-$PUB_TAG-tr-$PUB-$D-$MODEL" finetune_retfound_$PUB.sh $D $MODEL $FT 5e-4 2 0.05 default OCT 0
            done
        done
    done
}

eval_stage() {
    local B MODEL FT SIZE RESUME_PATH
    for D in $DATASETS; do
        if want all; then
            for B in "${BASELINES[@]}"; do
                read -r MODEL FT SIZE <<< "$B"
                BASE_DIR="$MODEL_DIR/$(base_run $D $FT)"
                train_finished "ms-$TAG-tr-base-$D-$MODEL" "$BASE_DIR" $BASELINE_EPOCHS $WB_P_TRAIN "$(base_run $D $FT)/" || continue
                have_ckpt "$BASE_DIR/checkpoint-best.pth" || continue
                WB_PROJECT=$WB_P_EVAL WB_KEYS="$(eval_keys "$BASE_DIR/checkpoint-best.pth" "$EVAL_FULL_XAI")" EXPECT_EVALS=$(count_words $EVAL_FULL_XAI) \
                    submit "ms-$TAG-ev-base-$D-$MODEL" baseline_multirun_XAI_eval_full.sh finetune_retfound_UFbenchmark_v5_eval_full2.sh $D 2 $MODEL $FT $SIZE
            done
            # original-run SMP dir: the eval driver appends the seed suffix to RESUME itself
            SMP_DIR="$MODEL_DIR/$(smp_run $D)"
            if train_finished "ms-$TAG-tr-sail-$D" "${SMP_DIR}${RUN_SUFFIX}" $SMP_EPOCHS $WB_P_SMP "$(smp_run $D)${RUN_SUFFIX}/" \
                    && have_ckpt "${SMP_DIR}${RUN_SUFFIX}/checkpoint-best.pth"; then
                WB_PROJECT=$WB_P_EVAL WB_KEYS="$(eval_keys "${SMP_DIR}${RUN_SUFFIX}/checkpoint-best.pth" "$EVAL_SMP_XAI")" EXPECT_EVALS=$(count_words $EVAL_SMP_XAI) \
                    submit "ms-$TAG-ev-sail-$D" baseline_multirun_XAI_eval_smp.sh finetune_retfound_UFbenchmark_v5_eval_smp_full.sh $D 2 SMP $SEG_CKPT \
                    "$SMP_DIR/checkpoint-best.pth" 512 $SMP_MODE $SMP_FUSE $SMP_ALPHA $SMP_SIZE_MATCH $SMP_FUSION_DIM $SMP_ALIGN \
                    $SMP_ENC_IDX $SMP_DEC_IDX $SMP_TARGET $SMP_SELECT $SMP_CLASSIFIER $SMP_ADD1 $SMP_ADD2
            fi
        fi
        want sail-enc || continue
        # like KDD_rise_enceval.sh: the eval script is submitted directly (one job per XAI method), so RESUME carries the seed suffix here
        ENC_DIR="$MODEL_DIR/$(enc_run $D)"
        RESUME_PATH="${ENC_DIR}${RUN_SUFFIX}/checkpoint-best.pth"
        train_finished "ms-$TAG-tr-enc-$D" "${ENC_DIR}${RUN_SUFFIX}" $SMP_EPOCHS $WB_P_SMP "$(enc_run $D)${RUN_SUFFIX}/" || continue
        have_ckpt "$RESUME_PATH" || continue
        for X in $SAIL_ENC_XAI; do
            WB_PROJECT=$WB_P_EVAL WB_KEYS="$(eval_keys "$RESUME_PATH" "$X")" EXPECT_EVALS=1 \
                submit "ms-$TAG-ev-enc-$D-$X" finetune_retfound_UFbenchmark_v5_eval_smp_full.sh $D SMP $SEG_CKPT \
                "$RESUME_PATH" 2 512 $X $SAIL_ENC_STEP_PIXELS $THICKNESS_DIR \
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
SUMMARY="$SUBMITTED submitted$([ "$DRY_RUN" = "1" ] && echo ' (dry run)') ($RESUMED of them rerun after a failed/cancelled/incomplete attempt), $SKIP_DONE already done, $SKIP_QUEUED already queued/running, $SKIPPED not ready"
if [ "$LIMIT_HIT" = "1" ]; then
    SUMMARY="$SUMMARY, $LEFT left: MAX_SLURM=$MAX_SLURM reached, rerun to continue"
fi
echo "[multiseed_run_all] SEED=$SEED stage=$STAGE: $SUMMARY"
