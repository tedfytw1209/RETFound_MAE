#!/bin/bash
# adml_dl_run_all.sh — submit the deep-learning baselines of OCT_ADML_experiment.xlsx (Jacqueline, Wisely, Wisely2,
# Mahendran, Hebei; 10-fold CV, one SLURM job per model runs its folds in turn) that are still missing or died, and
# resume each one at the folds that did not finish. Same arguments as ADML_Relatives.sh (study2) / ADML_Relatives2.sh
# (study3); the job loop follows multiple_study2_multiimg_eval.sh (study2: all) / multiple_study3_multiimg_eval.sh
# (study3: 1 3 5 10 years).
#   DRY_RUN=1 bash adml_dl_run_all.sh          # per-fold wandb status + the commands, nothing submitted: run this first
#   bash adml_dl_run_all.sh                    # submit what OCT_ADML_experiment (1).xlsx lacks (JOBS below)
#   MAX_SLURM=4 bash adml_dl_run_all.sh        # keep at most 4 of your jobs in the queue; rerun later for the rest
# env: MODELS="..."       subset of: Jacqueline Wisely Wisely2 Mahendran Hebei   (default: as listed per job line)
#      TASKS="..."        subset of: ad_control mci_control ad_mci_control ad_rest
#      VARIANTS="..."     subset of: clean raw (study2) 1yr 3yr 5yr 10yr (study3)
#      MAX_SLURM=N        never let your queue (squeue, pending + running) go above N jobs (default 10; 0 = no limit)
#      MAX_SUBMIT=N       submit at most N jobs in this call (default 0 = no limit)
#      HEAVY_MODELS="..." models whose data loading is the bottleneck: CPUS_HEAVY CPUs (default: Jacqueline Wisely
#                         Wisely2 Hebei); the others get CPUS_LIGHT. Workers = CPUs - 1 (the *_cv.sh scripts)
#      CPUS_HEAVY=N CPUS_LIGHT=N   CPUs per job (default 16 / 8; sbatch --cpus-per-task)
#      MEM=N              memory per job in GB (default 4 per CPU: 64 / 32; sbatch --mem)
#      MEM_MAX=N          a job whose last attempt ran out of memory gets twice that attempt's memory, up to this
#                         (default 128 GB); out of memory = SLURM state OUT_OF_MEMORY or oom_kill in its log
#      RERUN_DONE=1       resubmit all 10 folds even when they finished
#      WANDB_ENTITY=team  wandb entity of the OCTAD_Relatives project (default: the logged-in user)
#      CHECK_DATA=0       submit even if the data CSV / split folder is not found
#      REF_FILE=path      rerun reference rewritten by every call (default adml_dl_rerun_ref.sh; REF_FILE= for none)
#      WANDB_CHECK=0      only when wandb cannot be read: decide from the result folders instead (see below)
#
# The jobs come only from OCT_ADML_experiment (1).xlsx, its DL rows (checked 2026-11-14):
#   Study3-disease MCI_Control, year1/3/5/10 (H8:K12, H33:K37, H59:K63, H84:K88): empty, marked "w"  -> 20 jobs
#   Study2-disease AD_REST (L58, L59, L61): Jacqueline "no matching rerun", Wisely "rerun incomplete
#                           (8/10)", Wisely2 "rerun incomplete (2/10)"                                    -> 3 jobs
#   Every other DL cell has its 10-fold result, so nothing else is submitted.
#
# Wandb run names are never changed: each job's name is the one its *_cv.sh script gives the run (--task, + _CW from
# the trainer for --class_weight), the same pattern as the names in the sheet (Study3-disease N8:N12, e.g.
# ad_mci_control_detect_data-IRB2024v5_ADCON_DL_study3_retinaf_1yr-Jacqueline-convnext_tiny-Thickness-bs64ep100lr1e-3wd5e-4-
# roc_auceval-subset0_CW; tests/test_adml_dl_run_all.py checks them). All folds of a job share that name.
#
# Resume, per job, by that wandb name (project OCTAD_Relatives; the fold is the run's config.cv_fold):
#   a fold exists when a run with this name and fold FINISHED with test_* metrics; crashed / failed / killed runs, or
#   no run, mean the fold is redone. Each job prints the name and which folds wandb has.
#   If wandb cannot be read the script stops; WANDB_CHECK=0 then decides from the result folders instead
#   (/blue or /orange RETfound_results/<name>_cv10fold<k>/ with metrics_test.csv, or log.txt with the checkpoint purged).
#   - all 10 folds done                         -> skipped
#   - job adml-... queued/running in squeue     -> skipped
#   - a fold still running in wandb             -> skipped (it may come from an ADML_Relatives*.sh job, whose SLURM
#                                                  job name is the script name; a dead job shows as running until
#                                                  wandb marks it crashed: rerun this later)
#   - otherwise                                 -> submitted for the missing folds only (CV_FOLDS_ONLY="..." is read
#                                                  by the *_cv.sh scripts; START_FOLD = the first missing fold), with
#                                                  the last SLURM state and error line of the previous attempt
# wandb tags follow the existing runs: the *_cv.sh scripts tag "<study>,<task>[,<period>]" (study from the split path,
# e.g. study3,mci_control,1yr / study2,ad_rest) and the trainer adds class_weight. PERIOD is set on every submission:
# the study3 period, and empty for study2, so a PERIOD left in your shell never tags a study2 run.
# The *_cv.sh scripts also resume by themselves (adml_cv_resume.sh): when a job starts it skips the folds whose wandb run
# already finished, so a job submitted by hand or by ADML_Relatives*.sh runs only the unfinished folds too.
# A fold cannot continue mid-run (only checkpoint-best is kept), so a missing fold restarts from epoch 0.
# Rerunning the same command submits only what is still missing. REF_FILE lists the command of every job that is
# not done, with why, for rerunning by hand.

DRY_RUN=${DRY_RUN:-0}
MAX_SLURM=${MAX_SLURM:-10}
MAX_SUBMIT=${MAX_SUBMIT:-0}
MEM=${MEM:-}
MEM_MAX=${MEM_MAX:-128}
HEAVY_MODELS=${HEAVY_MODELS-"Jacqueline Wisely Wisely2 Hebei"}
CPUS_HEAVY=${CPUS_HEAVY:-16}
CPUS_LIGHT=${CPUS_LIGHT:-8}
RERUN_DONE=${RERUN_DONE:-0}
WANDB_CHECK=${WANDB_CHECK:-1}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_PROJECT=${WANDB_PROJECT:-OCTAD_Relatives}
WANDB_STATUS_CMD=${WANDB_STATUS_CMD:-"python adml_dl_wandb_status.py"}
CHECK_DATA=${CHECK_DATA:-1}
SACCT_DAYS=${SACCT_DAYS:-120}
REF_FILE=${REF_FILE-adml_dl_rerun_ref.sh}
for v in MAX_SLURM MAX_SUBMIT MEM_MAX CPUS_HEAVY CPUS_LIGHT ${MEM:+MEM}; do
    case "${!v}" in ''|*[!0-9]*) echo "$v must be a non-negative integer (0 = no limit), got '${!v}'" >&2; exit 1 ;; esac
done

ALL_MODELS="Jacqueline Wisely Wisely2 Mahendran Hebei"
NFOLDS=10
DATA_ROOT=/blue/ruogu.fang/tienyuchang
SPLIT_ROOT=$DATA_ROOT/OCTAD_ML_pipeline/psrs_oct
# the *_cv.sh scripts write to /blue now (/orange was full); earlier folds are under /orange
RESULT_DIRS=${RESULT_DIRS:-"/blue/ruogu.fang/tienyuchang/RETfound_results /orange/ruogu.fang/tienyuchang/RETfound_results"}

# job lines, from the sheet only: "<study> <variant> <task> <cw|none> [models...]" (no models = all five)
JOBS=(
    "study2 clean ad_rest cw Jacqueline Wisely Wisely2"   # Study2-disease AD_REST, L58 L59 L61
    "study3 1yr mci_control cw"                           # Study3-disease MCI_Control year1  (rows 8-12)
    "study3 3yr mci_control cw"                           #                            year3  (rows 33-37)
    "study3 5yr mci_control cw"                           #                            year5  (rows 59-63)
    "study3 10yr mci_control cw"                          #                            year10 (rows 84-88)
)
# cw: the sheet's study3/study2-disease DL names end in _CW. The AD_REST rerun names are not in the sheet: if the dry run
# shows Wisely / Wisely2 with no wandb folds although the sheet says 8/10 and 2/10, those runs were not class-weighted:
# change cw to none on the first line.

# --- the five models: run name (= --task of the *_cv.sh script, the wandb run name) and sbatch arguments -----------------
# Arguments as in ADML_Relatives.sh / ADML_Relatives2.sh; tests/test_adml_dl_run_all.py checks the names against the scripts.
run_name() { # <model> <STUDY csv base> <DATA folder>
    case "$1" in
        Jacqueline) echo "$2-$3-Jacqueline-convnext_tiny-Thickness-bs64ep100lr1e-3wd5e-4-roc_auceval-subset0" ;;
        Wisely)     echo "$2-$3-Wisely-resnet18_paper-Thickness-roc_auceval-subset0" ;;
        Wisely2)    echo "$2-$3-Wisely2-dual_input_cnn-images_only-Thickness-roc_auceval-subset0" ;;
        Mahendran)  echo "$2-$3-Mahendran-ad_oct_model-OCT-bs18ep100lr7e-5wd1e-2-roc_auceval-subset0" ;;
        Hebei)      echo "$2-$3-Hebei-ducan-dual-bs8ep400lr3e-4wd1e-2-accuracyeval-subset0" ;;
    esac
}
model_cmd() { # <model> <STUDY> <DATA> <SPLIT_DIR> <NUM_CLASS> <START_FOLD>: script and its arguments, before the CW flag
    case "$1" in
        Jacqueline) echo "finetune_Jacqueline_adcon_irb2024_v5_cv.sh $2 $3 $4 convnext_tiny 64 1e-3 5e-4 $5 0 0.001 0.2 $6" ;;
        Wisely)     echo "finetune_Wisely_adcon_irb2024_v5_cv.sh $2 $3 $4 resnet18_paper 0.01 1e-3 0.01 $5 0 $6" ;;
        Wisely2)    echo "finetune_Wisely2_adcon_irb2024_v5_cv.sh $2 $3 $4 dual_input_cnn images_only 0.01 1e-4 0.01 $5 0 10 $6" ;;
        Mahendran)  echo "finetune_Mahendran_ad_oct_model_cv.sh $2 $3 $4 ad_oct_model 256 3 false 7e-5 1e-2 $5 $6" ;;
        Hebei)      echo "finetune_Hebei_admci_detection_cv.sh $2 $3 $4 ducan 0.7 0.7 1.0 3e-4 1e-2 $5 $6" ;;
    esac
}

in_words() { case " $1 " in *" $2 "*) return 0 ;; esac; return 1; } # <list> <word>
SUBMITTED=0; RESUMED=0; SKIP_DONE=0; SKIP_RUNNING=0; SKIP_DATA=0; LEFT=0; LIMIT_HIT=0

# --- SLURM ------------------------------------------------------------------------------------------------------------
SLURM_USER=${USER:-$(whoami)}
HAVE_SLURM=0
if command -v squeue >/dev/null 2>&1 && command -v sacct >/dev/null 2>&1; then
    HAVE_SLURM=1
else
    echo "[adml_dl_run_all] squeue/sacct not found: queue state and MAX_SLURM are not checked" >&2
fi
queued_now() { if [ "$HAVE_SLURM" = "1" ]; then squeue -u "$SLURM_USER" -h -o %j 2>/dev/null; fi; return 0; }
SACCT_ROWS=""
if [ "$HAVE_SLURM" = "1" ]; then # every attempt of our jobs: name|state|jobid|workdir
    SACCT_ROWS=$(sacct -u "$SLURM_USER" -X -n -P -S "now-${SACCT_DAYS}days" --format=JobName,State,JobID,WorkDir,ReqMem,AllocCPUS 2>/dev/null | grep '^adml-')
fi
last_attempt() { # <job name>: "STATE|jobid|workdir|ReqMem|AllocCPUS" of the newest attempt, or nothing
    printf '%s\n' "$SACCT_ROWS" | awk -F'|' -v n="$1" '$1 == n && $3 + 0 >= last { last = $3 + 0; split($2, s, " "); row = s[1] "|" $3 "|" $4 "|" $5 "|" $6 }
        END { if (row != "") print row }'
}
req_gb() { # <ReqMem> <AllocCPUS>: an attempt's memory in GB (sacct prints 8G, 8Gn, 4Gc = per CPU, 8000M), or nothing
    awk -v r="$1" -v c="${2:-1}" 'BEGIN { if (match(r, /^[0-9.]+/)) { n = substr(r, 1, RLENGTH); u = substr(r, RLENGTH + 1, 1)
        s = substr(r, RLENGTH + 2, 1); f = (u == "T") ? 1024 : (u == "M") ? 1 / 1024 : (u == "K") ? 1 / 1048576 : 1
        if (s == "c") n *= c; printf "%d\n", n * f + 0.5 } }'
}
ran_out_of_memory() { # <state> <job name> <jobid> <workdir>: SLURM says OUT_OF_MEMORY, or the log has oom_kill events
    # (a fold killed for memory does not end the job: the *_cv.sh loop goes on to the next fold, so check the log too)
    local out="$4/$2.$3.out"
    [ "$1" = "OUT_OF_MEMORY" ] && return 0
    [ -n "$3" ] && [ -f "$out" ] && grep -aqE 'oom_kill|oom-kill|OOM Killed' "$out"
}
last_error() { # <job name> <jobid> <workdir>: the last error-looking line of that attempt's log (%x.%j.out)
    local out="$3/$1.$2.out"
    [ -f "$out" ] || return 0
    grep -aE 'Error|Traceback|Killed|out of memory|CANCELLED|TIME LIMIT|No such file' "$out" | tail -n 1 | cut -c1-200
}

# --- wandb: one query for the whole project, kept in WB_ROWS as "name<TAB>fold<TAB>status" ---------------------------
WB_ROWS=""
WB_OK=0
if [ "$WANDB_CHECK" = "1" ]; then
    WB_ERR=$(mktemp)
    if WB_ROWS=$($WANDB_STATUS_CMD --project "$WANDB_PROJECT" --regex='_detect_data-IRB2024v5_ADCON_DL' \
            ${WANDB_ENTITY:+--entity "$WANDB_ENTITY"} 2>"$WB_ERR"); then
        WB_OK=1
    else
        echo "[adml_dl_run_all] wandb ($WANDB_PROJECT) cannot be read: $(head -n 1 "$WB_ERR")" >&2
        echo "  the runs are checked by their wandb name: fix it (wandb login, WANDB_ENTITY=...), or WANDB_CHECK=0 to" \
             "decide from the result folders instead" >&2
        rm -f "$WB_ERR"
        exit 1
    fi
    rm -f "$WB_ERR"
else
    echo "[adml_dl_run_all] WANDB_CHECK=0: folds are checked in the result folders ($RESULT_DIRS), not by wandb name" >&2
fi

disk_done() { # <run name> <fold>: only used with WANDB_CHECK=0
    local d dir
    [ "$WB_OK" = "1" ] && return 1
    for d in $RESULT_DIRS; do
        dir="$d/${1}_cv${NFOLDS}fold$2"
        if [ -f "$dir/metrics_test.csv" ] || { [ -f "$dir/log.txt" ] && [ ! -f "$dir/checkpoint-best.pth" ]; }; then
            return 0
        fi
    done
    return 1
}

# FOLDS_DONE / FOLDS_RUNNING / FOLDS_MISSING (space-separated) and FOLD_NOTE for one run name
fold_status() { # <run name>
    local rows k st
    FOLDS_DONE=""; FOLDS_RUNNING=""; FOLDS_MISSING=""; FOLD_NOTE=""
    rows=$(printf '%s\n' "$WB_ROWS" | awk -F'\t' -v n="$1" '$1 == n')
    for ((k = 0; k < NFOLDS; k++)); do
        st=$(printf '%s\n' "$rows" | awk -F'\t' -v f="$k" '$2 == f { print $3; exit }')
        if [ "$st" = "finished" ] || disk_done "$1" "$k"; then
            FOLDS_DONE="${FOLDS_DONE:+$FOLDS_DONE }$k"
        elif [ "$st" = "running" ]; then
            FOLDS_RUNNING="${FOLDS_RUNNING:+$FOLDS_RUNNING }$k"
        else
            FOLDS_MISSING="${FOLDS_MISSING:+$FOLDS_MISSING }$k"
            if [ -n "$st" ]; then
                FOLD_NOTE="${FOLD_NOTE:+$FOLD_NOTE, }$k=$st"
            fi
        fi
    done
}

# --- data -------------------------------------------------------------------------------------------------------------
DATA_ERR=""
check_data() { # <DATA> <STUDY> <SPLIT_DIR>
    local csv="$DATA_ROOT/$1/$2.csv"
    DATA_ERR=""
    [ "$CHECK_DATA" = "1" ] || return 0
    case "$2" in ad_rest_detect_data) csv="$DATA_ROOT/$1/ad_mci_control_detect_data.csv" ;; esac # redirected by the trainer
    if [ ! -f "$csv" ]; then
        DATA_ERR="data CSV not found: $csv"
    elif [ ! -f "$3/fold0-$NFOLDS.csv" ]; then
        DATA_ERR="split not found: $3/fold0-$NFOLDS.csv"
    fi
    [ -z "$DATA_ERR" ]
}

# --- rerun reference --------------------------------------------------------------------------------------------------
REF_TMP=""
if [ -n "$REF_FILE" ]; then
    REF_TMP=$(mktemp)
    {
        echo "# adml_dl_run_all.sh rerun reference, $(date '+%F %T')$([ "$DRY_RUN" = "1" ] && echo ' (dry run)')"
        echo "# One command per job that is not done. 'bash adml_dl_run_all.sh' resubmits all of them by itself (it skips"
        echo "# what is queued/running/done); copy a line to run one by hand. CV_FOLDS_ONLY = the folds still missing."
        echo "# folds checked by: $([ "$WB_OK" = "1" ] && echo "wandb run name ($WANDB_PROJECT)" || echo 'result folders (WANDB_CHECK=0)')"
    } > "$REF_TMP"
fi
ref() { [ -n "$REF_TMP" ] && printf '%s\n' "$@" >> "$REF_TMP"; return 0; }

# --- one job ----------------------------------------------------------------------------------------------------------
process() { # <study> <variant> <task> <cw> <model>
    local study=$1 variant=$2 task=$3 cw=$4 model=$5
    local num_class subdir split_base data period split_dir STUDY name job cw_flag="" start cmd state jobid wd err note queued count
    case "$task" in
        ad_control)     num_class=2; subdir=split ;;
        mci_control)    num_class=2; subdir=split_mcicon ;;
        ad_mci_control) num_class=3; subdir=split_admcicon ;;
        ad_rest)        num_class=2; subdir=split_adrest ;;
        *) echo "unknown task '$task'" >&2; return ;;
    esac
    case "$study-$variant" in
        study2-clean) split_base=$SPLIT_ROOT/IRB2024_prev_combined_study2_retinaf_diseaseclean; data=IRB2024v5_ADCON_DL_data_retinaf_diseaseclean; period="" ;;
        study2-raw)   split_base=$SPLIT_ROOT/IRB2024_prev_combined_study2_retinaf; data=IRB2024v5_ADCON_DL_data_retinaf; period="" ;;
        study3-*yr)   split_base=$SPLIT_ROOT/IRB2024_combined_study3_retinaf_diseaseclean_$variant; data=IRB2024v5_ADCON_DL_study3_retinaf_$variant; period=$variant ;;
        *) echo "unknown study/variant '$study $variant'" >&2; return ;;
    esac
    split_dir=$split_base/$subdir
    STUDY=${task}_detect_data
    name=$(run_name "$model" "$STUDY" "$data")
    if [ "$cw" = "cw" ]; then
        cw_flag="--class_weight"
        name="${name}_CW" # main_finetune_Chua_Jacqueline.py appends _CW for --class_weight
    fi
    job="adml-s${study#study}-$variant-$task-$model-$cw" # e.g. adml-s3-1yr-mci_control-Hebei-cw

    queued=$(queued_now)
    if printf '%s\n' "$queued" | grep -Fxq -- "$job"; then
        echo "[skip] queued/running: $job" >&2
        SKIP_RUNNING=$((SKIP_RUNNING + 1))
        return
    fi
    fold_status "$name"
    if [ "$WB_OK" = "1" ]; then # what wandb has under this exact name, for every job
        if [ -z "$FOLDS_DONE$FOLDS_RUNNING$FOLD_NOTE" ]; then
            echo "[wandb] $job: no run named $name" >&2
        else
            echo "[wandb] $job: $name: finished folds ${FOLDS_DONE:-none}${FOLDS_RUNNING:+, running $FOLDS_RUNNING}${FOLD_NOTE:+, not finished: $FOLD_NOTE}" >&2
        fi
    fi
    if [ "$RERUN_DONE" = "1" ]; then
        FOLDS_MISSING=$(seq -s ' ' 0 $((NFOLDS - 1))); FOLDS_DONE=""; FOLDS_RUNNING=""
    fi
    if [ -z "$FOLDS_MISSING" ] && [ -z "$FOLDS_RUNNING" ]; then
        echo "[done] $job: $NFOLDS/$NFOLDS folds" >&2
        SKIP_DONE=$((SKIP_DONE + 1))
        return
    fi
    if [ -n "$FOLDS_RUNNING" ]; then
        echo "[skip] running in wandb (folds $FOLDS_RUNNING; done: ${FOLDS_DONE:-none}): $job" >&2
        ref "# $job: folds $FOLDS_RUNNING running in wandb, missing: ${FOLDS_MISSING:-none}; rerun this script once it stops"
        SKIP_RUNNING=$((SKIP_RUNNING + 1))
        return
    fi

    start=${FOLDS_MISSING%% *}
    cmd="$(model_cmd "$model" "$STUDY" "$data" "$split_dir" "$num_class" "$start") $cw_flag"
    local folds_env=""
    if [ -n "$FOLDS_DONE" ]; then
        folds_env=$FOLDS_MISSING
    fi
    note="folds done: ${FOLDS_DONE:-none}; to run: $FOLDS_MISSING${FOLD_NOTE:+ (wandb: $FOLD_NOTE)}"
    local reqmem alloc last_gb cpus=$CPUS_LIGHT base_mem mem
    if in_words "$HEAVY_MODELS" "$model"; then
        cpus=$CPUS_HEAVY
    fi
    base_mem=${MEM:-$((cpus * 4))}
    mem=$base_mem
    IFS='|' read -r state jobid wd reqmem alloc <<< "$(last_attempt "$job")"
    if [ -n "$state" ]; then
        note="$note; last attempt $state (job $jobid)"
        err=$(last_error "$job" "$jobid" "$wd")
        if [ -n "$err" ]; then
            note="$note; last error: $err"
        fi
        if ran_out_of_memory "$state" "$job" "$jobid" "$wd"; then
            last_gb=$(req_gb "$reqmem" "$alloc")
            mem=$(( ${last_gb:-$base_mem} * 2 ))
            [ "$mem" -lt "$base_mem" ] && mem=$base_mem
            [ "$mem" -gt "$MEM_MAX" ] && mem=$MEM_MAX
            note="$note; ran out of memory at ${last_gb:-?} GB -> ${mem} GB"
            if [ "${last_gb:-0}" -ge "$MEM_MAX" ]; then
                note="$note (already at MEM_MAX=$MEM_MAX: raise MEM_MAX or lower the batch size / num_workers)"
            fi
        fi
    fi
    cmd="--cpus-per-task=$cpus --mem=${mem}gb $cmd"
    ref "# $job [$name]: $note" \
        "PERIOD=$period CV_FOLDS_ONLY=\"$folds_env\" sbatch --job-name=$job $cmd"

    if ! check_data "$data" "$STUDY" "$split_dir"; then
        echo "[error] $job: $DATA_ERR (CHECK_DATA=0 to submit anyway)" >&2
        ref "#   ^ not submitted: $DATA_ERR"
        SKIP_DATA=$((SKIP_DATA + 1))
        if [ "$DRY_RUN" != "1" ]; then
            return
        fi
    fi
    if [ "$LIMIT_HIT" = "1" ]; then
        LEFT=$((LEFT + 1))
        return
    fi
    if [ "$MAX_SUBMIT" -gt 0 ] && [ "$SUBMITTED" -ge "$MAX_SUBMIT" ]; then
        echo "[limit] MAX_SUBMIT=$MAX_SUBMIT jobs submitted: stopping here (rerun to continue)" >&2
        LIMIT_HIT=1; LEFT=$((LEFT + 1))
        return
    fi
    if [ "$MAX_SLURM" -gt 0 ] && [ "$HAVE_SLURM" = "1" ]; then
        count=$(printf '%s\n' "$queued" | grep -c .)
        if [ "$DRY_RUN" = "1" ]; then
            count=$((count + SUBMITTED)) # nothing was really submitted: count what this dry run would have added
        fi
        if [ "$count" -ge "$MAX_SLURM" ]; then
            echo "[limit] $count jobs in the queue >= MAX_SLURM=$MAX_SLURM: stopping here (rerun later; done folds are skipped)" >&2
            LIMIT_HIT=1; LEFT=$((LEFT + 1))
            return
        fi
    fi
    if [ -n "$FOLDS_DONE" ] || [ -n "$state" ]; then
        echo "[resume] $job: $note" >&2
        RESUMED=$((RESUMED + 1))
    else
        echo "[new] $job: $NFOLDS folds" >&2
    fi
    echo "PERIOD=$period CV_FOLDS_ONLY=\"$folds_env\" sbatch --job-name=$job $cmd"
    if [ "$DRY_RUN" != "1" ]; then
        # sbatch exports the environment: PERIOD tags the wandb run (study3), CV_FOLDS_ONLY picks the folds
        PERIOD=$period CV_FOLDS_ONLY=$folds_env sbatch --job-name="$job" $cmd
    fi
    SUBMITTED=$((SUBMITTED + 1))
}

for line in "${JOBS[@]}"; do
    read -r study variant task cw models <<< "$line"
    [ -n "$models" ] || models=$ALL_MODELS
    if [ -n "${TASKS:-}" ] && ! in_words "$TASKS" "$task"; then continue; fi
    if [ -n "${VARIANTS:-}" ] && ! in_words "$VARIANTS" "$variant"; then continue; fi
    for model in $models; do
        if [ -n "${MODELS:-}" ] && ! in_words "$MODELS" "$model"; then continue; fi
        process "$study" "$variant" "$task" "$cw" "$model"
    done
done

SUMMARY="$SUBMITTED submitted$([ "$DRY_RUN" = "1" ] && echo ' (dry run)') ($RESUMED of them resumed/rerun), $SKIP_DONE done, $SKIP_RUNNING queued/running, $SKIP_DATA data not found"
if [ "$LIMIT_HIT" = "1" ]; then
    SUMMARY="$SUMMARY, $LEFT left by the limit: rerun to continue"
fi
if [ -n "$REF_TMP" ]; then
    ref "# summary: $SUMMARY"
    if mv "$REF_TMP" "$REF_FILE"; then
        SUMMARY="$SUMMARY; rerun reference: $REF_FILE"
    fi
fi
echo "[adml_dl_run_all] $SUMMARY"
