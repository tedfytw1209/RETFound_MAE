#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

date;hostname;pwd

module load conda
conda activate octxai

# ============================================================================
# Model Definitions
# ============================================================================


SCRIPT=$1
model_fname=$2
model=${3:-"SMP_enc"}
OUTPUT_DIR=${4:-"./heatmap_debug_batch"}
batch_size=${5:-2}

# ============================================================================
# Common parameters
# ============================================================================
DATASET_DIR="/blue/ruogu.fang/tienyuchang/OCT_EDA"
DATASET_FNAME="sampled_labels01.csv"
THICKNESS_DIR="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"
THICKNESS_CSV="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/thickness_map.csv"
MODEL_ROOT="/orange/ruogu.fang/tienyuchang/RETfound_results"


target_module="encoder decoder head"

# ============================================================================
# Run XAI for base models (DME_finetuned)
# ============================================================================
python "$SCRIPT" \
        --dataset_dir "$DATASET_DIR" \
        --dataset_fname "$DATASET_FNAME" \
        --thickness_dir "$THICKNESS_DIR" \
        --thickness_csv "$THICKNESS_CSV" \
        --model_root "$MODEL_ROOT" \
        --model_fname "$model_fname" \
        --model "$model" \
        --target_module "$target_module" \
        --task DME \
        --num_samples -1 \
        --xai_method GradCAM HiResCAM GradCAMPlusPlus \
        --batch_size $batch_size \
        --input_size 512 \
        --nb_classes 2 \
        --load_mask \
        --draw_layer \
        --output_dir "$OUTPUT_DIR" \
        --verbose