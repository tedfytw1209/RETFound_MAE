#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

date;hostname;pwd

module load conda
conda activate octxai

# ============================================================================
# Model Definitions
# ============================================================================

# Base model list
Model_list=(
    "SMP_enc"
    "SMP_dec"
    "SMP_enc_fix"
    "SMP_dec_fix"
)

# DME finetuned models
DME_finetuned=(
"DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-enc-smpweighted_sum-{0}-fea-1-1-0.5-decoder_to_encoder---/" 
"DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-dec-smpweighted_sum-{0}-fea-1-1-0.5-decoder_to_encoder---/" 
"DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-enc-smpweighted_sum-{0}-fea-1-1-0.5-decoder_to_encoder---fix_extractor--/" 
"DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-dec-smpweighted_sum-{0}-fea-1-1-0.5-decoder_to_encoder---fix_extractor--/"
)

# ============================================================================
# Common parameters
# ============================================================================
DATASET_DIR="/blue/ruogu.fang/tienyuchang/OCT_EDA"
DATASET_FNAME="sampled_labels01.csv"
THICKNESS_DIR="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"
THICKNESS_CSV="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/thickness_map.csv"
MODEL_ROOT="/orange/ruogu.fang/tienyuchang/RETfound_results"
OUTPUT_DIR="./heatmap_debug_batch2"
batch_size=6
target_module="encoder decoder head"

# ============================================================================
# Run XAI for base models (DME_finetuned)
# ============================================================================
echo "Running base models..."
for i in "${!Model_list[@]}"; do
    model="${Model_list[$i]}"
    model_fname="${DME_finetuned[$i]}checkpoint-best.pth"
    
    echo "Processing model: $model with checkpoint: $model_fname"
    
    python case_study_SMP_layermap.py \
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
done