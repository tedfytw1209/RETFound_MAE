#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

# Segment the middle B-scan of a single OCT cube (folder of slice_XXXX.jpg files)
# and export the mask as both PNG (colorized) and NPY (raw class indices).
#
# Usage:
#   sbatch inference_mask_single.sh <CUBE_DIR> [CLASS_MODE] [IMG_SIZE] [OUTPUT_DIR]
# Examples:
#   sbatch inference_mask_single.sh /orange/ruogu.fang/tienyuchang/OPT_Macular_Cube_512x128
#   sbatch inference_mask_single.sh /orange/ruogu.fang/tienyuchang/OPT_Macular_Cube_512x128 multiclass_resnet50_new 512
#   sbatch inference_mask_single.sh /orange/ruogu.fang/tienyuchang/OPT_Macular_Cube_512x128 binary 512

date;hostname;pwd

module load conda
conda activate retfound_new

CUBE_DIR=${1:?Usage: sbatch inference_mask_single.sh <CUBE_DIR> [CLASS_MODE] [IMG_SIZE] [OUTPUT_DIR]}
CLASS_MODE=${2:-multiclass_resnet50_new}
IMG_SIZE=${3:-512}
OUTPUT_DIR=${4:-${CUBE_DIR%/}_masks_${CLASS_MODE}}
CHECKPOINT=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_${CLASS_MODE}.pth

echo "Cube dir:   $CUBE_DIR"
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
    --middle-slice \
    --export-formats png npy
