#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12gb
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
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs4ep20lr1e-4optadamw-defaulteval-trsub0-enc--/"
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs4ep20lr1e-4optadamw-defaulteval-trsub0--/"
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr5e-4optadamw-defaulteval-trsub0-enc---fix_extractor--/"
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs4ep20lr1e-4optadamw-defaulteval-trsub0-dec---fix_extractor--/"
)

# DME finetuned masked models
DME_finetuned_masked=(
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs4ep20lr1e-4optadamw-defaulteval-trsub0-enc---add_mask---train_no_aug/"
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs4ep20lr1e-4optadamw-defaulteval-trsub0---add_mask---train_no_aug/"
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-2optadamw-defaulteval-trsub0-enc---add_mask---train_no_aug---fix_extractor/"
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs4ep20lr1e-4optadamw-defaulteval-trsub0-dec---add_mask---train_no_aug---fix_extractor/"
)

# Unmasked fuse models - paths
Fuse_models=(
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpmultiply-0.5-decoder_to_encoder---seg_mask--/"
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpchannel_merge-0.5-decoder_to_encoder---seg_mask--/"
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpchannel_multiply-0.5-decoder_to_encoder---seg_mask--/"
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpweighted_sum-0.5-decoder_to_encoder---smp_learnable_alpha---seg_mask-/"
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpchannel_merge-0.5-decoder_to_encoder---/"
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpadd-0.5-decoder_to_encoder---/"
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpmultiply-0.5-decoder_to_encoder---/"
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpweighted_sum-0.5-decoder_to_encoder---smp_learnable_alpha--/"
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpadd-fea-2-1-0.5-decoder_to_encoder---seg_mask--/"
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpchannel_merge-fea-2-1-0.5-decoder_to_encoder---seg_mask--/"
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpchannel_multiply-fea-2-1-0.5-decoder_to_encoder---seg_mask--/"
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpmultiply-fea-2-1-0.5-decoder_to_encoder---seg_mask--/"
    "DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpweighted_sum-fea-2-1-0.5-decoder_to_encoder---smp_learnable_alpha---seg_mask-/"
)

# Fuse model names following: SMP_fuse_{smp_fuse_mode}_fus{fusion_dim}enc{enc_idx}dec{dec_idx}_{seg|dec}
Fuse_model_names=(
    "SMP_fuse_multiply_fus0enc-1dec-1_seg"
    "SMP_fuse_channel_merge_fus0enc-1dec-1_seg"
    "SMP_fuse_channel_multiply_fus0enc-1dec-1_seg"
    "SMP_fuse_weighted_sum_fus0enc-1dec-1_seg"
    "SMP_fuse_channel_merge_fus0enc-1dec-1_dec"
    "SMP_fuse_add_fus0enc-1dec-1_dec"
    "SMP_fuse_multiply_fus0enc-1dec-1_dec"
    "SMP_fuse_weighted_sum_fus0enc-1dec-1_dec"
    "SMP_fuse_add_fus8enc-2dec-1_seg"
    "SMP_fuse_channel_merge_fus8enc-2dec-1_seg"
    "SMP_fuse_channel_multiply_fus0enc-2dec-1_seg"
    "SMP_fuse_multiply_fus8enc-2dec-1_seg"
    "SMP_fuse_weighted_sum_fus8enc-2dec-1_seg"
)

# ============================================================================
# Common parameters
# ============================================================================
DATASET_DIR="/blue/ruogu.fang/tienyuchang/OCT_EDA"
DATASET_FNAME="sampled_labels01.csv"
THICKNESS_DIR="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"
THICKNESS_CSV="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/thickness_map.csv"
MODEL_ROOT="/orange/ruogu.fang/tienyuchang/RETfound_results"
OUTPUT_DIR="./heatmap_results_production"
batch_size=4
#target_module="encoder decoder head"
target_module="head"

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

# ============================================================================
# Run XAI for masked models (DME_finetuned_masked)
# ============================================================================
echo "Running masked models..."
for i in "${!Model_list[@]}"; do
    model="${Model_list[$i]}"
    model_fname="${DME_finetuned_masked[$i]}checkpoint-best.pth"
    
    echo "Processing masked model: $model with checkpoint: $model_fname"
    
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

# ============================================================================
# Run XAI for fuse models
# ============================================================================
echo "Running fuse models..."
for i in "${!Fuse_models[@]}"; do
    model="${Fuse_model_names[$i]}"
    model_fname="${Fuse_models[$i]}checkpoint-best.pth"
    
    echo "Processing fuse model: $model with checkpoint: $model_fname"
    
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

echo "All models completed!"