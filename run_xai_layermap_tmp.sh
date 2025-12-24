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

# Base model list
Model_list=(
#    "SMP_enc"
#    "SMP_dec"
    "SMP_enc_fix"
    "SMP_dec_fix"
)

# DME finetuned models
DME_finetuned=(
#"DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-enc-smpweighted_sum-{0}-fea-1-1-0.5-decoder_to_encoder---/" 
#"DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-dec-smpweighted_sum-{0}-fea-1-1-0.5-decoder_to_encoder---/" 
"DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-enc-smpweighted_sum-{0}-fea-1-1-0.5-decoder_to_encoder---fix_extractor--/" 
"DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-dec-smpweighted_sum-{0}-fea-1-1-0.5-decoder_to_encoder---fix_extractor--/"
)

# ============================================================================
# Run XAI for base models (DME_finetuned)
# ============================================================================
echo "Running base models..."
for i in "${!Model_list[@]}"; do
    model="${Model_list[$i]}"
    model_fname="${DME_finetuned[$i]}checkpoint-best.pth"
    sbatch run_xai_layermap_single.sh case_study_SMP_layermap_ori.py "$model_fname" "$model" "././heatmap_debug_batch" 2
done