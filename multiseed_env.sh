# multiseed_env.sh
# Shared multi-seed settings, sourced by the train / XAI-eval scripts and their drivers.
#   SEED unset             -> original single-seed run (split/${BASE_SPLIT}, no seed flags)
#   SEED=k SEED_MODE=split -> split/${BASE_SPLIT}-s{k}, --split_seed k (training seed unchanged)  [UF default]
#   SEED=k SEED_MODE=train -> split/${BASE_SPLIT},      --seed k       (split unchanged)          [public data]
# RUN_SUFFIX mirrors util/multiseed.py seed_suffix (used by the eval drivers to find checkpoints).
SEED=${SEED:-}
SEED_MODE=${SEED_MODE:-split}
BASE_SPLIT=${BASE_SPLIT:-tune8-eval2}
SPLIT_DIR=$BASE_SPLIT
SEED_ARGS=""
RUN_SUFFIX=""
if [ -n "$SEED" ] && [ "$SEED_MODE" = "split" ]; then
    if [ "$SEED" != "42" ]; then
        SPLIT_DIR=${BASE_SPLIT}-s${SEED}
        SEED_ARGS="--split_seed ${SEED}"
        RUN_SUFFIX="-s${SEED}"
    fi
elif [ -n "$SEED" ] && [ "$SEED_MODE" = "train" ]; then
    if [ "$SEED" = "0" ] || [ "$SEED" = "42" ]; then
        echo "[multiseed] SEED=$SEED is a script default seed; leave SEED unset for the original run" >&2
        return 1 2>/dev/null || exit 1
    fi
    SEED_ARGS="--seed ${SEED}"
    RUN_SUFFIX="-seed${SEED}"
elif [ -n "$SEED" ]; then
    echo "[multiseed] unknown SEED_MODE=$SEED_MODE (split|train)" >&2
    return 1 2>/dev/null || exit 1
fi
echo "[multiseed] SEED=${SEED:-original} SEED_MODE=$SEED_MODE SPLIT_DIR=$SPLIT_DIR SEED_ARGS=$SEED_ARGS"
