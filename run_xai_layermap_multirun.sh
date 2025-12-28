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
batch_size=2
#target_module="encoder decoder head"
target_module="head"

# ============================================================================
# Run XAI for fuse models
# ============================================================================
echo "Running fuse models..."
for i in "${!Fuse_models[@]}"; do
    model="${Fuse_model_names[$i]}"
    model_fname="${DME_finetuned[$i]}checkpoint-best.pth"
    echo "sbatch run_xai_layermap_single.sh case_study_SMP_layermap_ori.py $model_fname $model ././heatmap_debug_batch 2"
    sbatch run_xai_layermap_single.sh case_study_SMP_layermap_ori.py "$model_fname" "$model" "././heatmap_debug_batch" 2
done