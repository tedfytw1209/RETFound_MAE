#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

# Segment the B-scans of a single OCT cube (folder of slice_XXXX.jpg files)
# and export each mask as both PNG (colorized) and NPY (raw class indices).
#
# Usage:
#   sbatch inference_mask_single.sh <CUBE_DIR> [SLICES] [CLASS_MODE] [IMG_SIZE] [OUTPUT_DIR]
#     SLICES: all (default) = every slice in the cube, middle = only the center slice
# Examples:
#   sbatch inference_mask_single.sh /orange/ruogu.fang/tienyuchang/OPT_Macular_Cube_512x128
#   sbatch inference_mask_single.sh /orange/ruogu.fang/tienyuchang/OPT_Macular_Cube_512x128 middle
#   sbatch inference_mask_single.sh /orange/ruogu.fang/tienyuchang/OPT_Macular_Cube_512x128 all binary 512

date;hostname;pwd

module load conda
conda activate retfound_new

USAGE="Usage: sbatch inference_mask_single.sh <CUBE_DIR> [all|middle] [CLASS_MODE] [IMG_SIZE] [OUTPUT_DIR]"
CUBE_DIR=${1:?$USAGE}
SLICES=${2:-all}
CLASS_MODE=${3:-multiclass_resnet50_new}
IMG_SIZE=${4:-512}
OUTPUT_DIR=${5:-${CUBE_DIR%/}_masks_${CLASS_MODE}}
CHECKPOINT=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_${CLASS_MODE}.pth

case "$SLICES" in
    all)    SLICE_ARGS=() ;;
    middle) SLICE_ARGS=(--middle-slice) ;;
    *)      echo "Invalid SLICES '$SLICES' (expected 'all' or 'middle'). $USAGE"; exit 1 ;;
esac

echo "Cube dir:   $CUBE_DIR"
echo "Slices:     $SLICES"
echo "Class mode: $CLASS_MODE"
echo "Checkpoint: $CHECKPOINT"
echo "Output dir: $OUTPUT_DIR"

python inference_general.py \
    --dataset custom \
    --input-dir "$CUBE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --class-mode "$CLASS_MODE" \
    --checkpoint "$CHECKPOINT" \
    --image-size "$IMG_SIZE" \
    --export-formats png npy \
    "${SLICE_ARGS[@]}"
