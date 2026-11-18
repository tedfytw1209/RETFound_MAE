# adml_cv_resume.sh — sourced by the finetune_*_cv.sh scripts: resume a 10-fold CV job at its unfinished folds.
# When the job starts it asks wandb (project OCTAD_Relatives, adml_dl_wandb_status.py) which folds of its run already
# finished: a run with the job's wandb name (its --task, + _CW for --class_weight, as main_finetune_Chua_Jacqueline.py
# names it) and that config.cv_fold, in state finished with test_* metrics. Those folds are skipped; crashed / failed /
# killed / missing folds run. The wandb name is not changed.
# env: CV_RESUME=0        run every requested fold (default 1: skip the finished ones)
#      CV_STATUS_CMD=...  status command (default: python adml_dl_wandb_status.py); WANDB_ENTITY=<team> if needed
# If wandb cannot be read, a warning is printed and every requested fold runs, as before.
CV_DONE_FOLDS=""

cv_resume_init() { # <task> [extra trainer args...]: the wandb name is the task, + _CW if --class_weight is among the args
    local name=$1 regex rows err
    shift
    case " $* " in *" --class_weight "*) name="${name}_CW" ;; esac
    CV_DONE_FOLDS=""
    if [ "${CV_RESUME:-1}" != "1" ]; then
        echo "[cv-resume] CV_RESUME=0: every requested fold runs"
        return 0
    fi
    regex="^$(printf '%s' "$name" | sed 's/[][\.*^$()+?{}|]/\\&/g')\$"
    err=$(mktemp)
    if rows=$(${CV_STATUS_CMD:-python adml_dl_wandb_status.py} --project OCTAD_Relatives --regex="$regex" \
            ${WANDB_ENTITY:+--entity "$WANDB_ENTITY"} 2>"$err"); then
        CV_DONE_FOLDS=$(printf '%s\n' "$rows" | awk -F'\t' -v n="$name" '$1 == n && $3 == "finished" { printf "%s ", $2 }')
        echo "[cv-resume] $name: finished folds in wandb: ${CV_DONE_FOLDS:-none}"
    else
        echo "[cv-resume] wandb not readable ($(head -n 1 "$err")): every requested fold runs" >&2
    fi
    rm -f "$err"
}

cv_fold_done() { # <fold>: true when that fold already finished in wandb
    case " $CV_DONE_FOLDS " in
        *" $1 "*) echo "[cv-resume] fold $1 already finished in wandb: skipped"; return 0 ;;
    esac
    return 1
}
