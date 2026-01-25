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
STUDY=${1:-"DME"}
STUDY_NAME=${2:-"DME_binary_all_split"}

# sbatch run_xai_layermap_bs.sh DME DME_binary_all_split
# ============================================================================
# Baselines
#bash run_xai_layermap_bs.sh $STUDY RETFound_mae RETFound_mae_natureOCT /orange/ruogu.fang/tienyuchang/RETfound_results/${STUDY_NAME}-IRB2024_v5_all-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0--/checkpoint-best.pth 224 dec weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv
bash run_xai_layermap_bs.sh  $STUDY vit-base-patch16-224 google/vit-base-patch16-224-in21k /orange/ruogu.fang/tienyuchang/RETfound_results/${STUDY_NAME}-IRB2024_v5_all-all-google/vit-base-patch16-224-in21k-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0--/checkpoint-best.pth 224 dec weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv
bash run_xai_layermap_bs.sh  $STUDY resnet-50 microsoft/resnet-50 /orange/ruogu.fang/tienyuchang/RETfound_results/${STUDY_NAME}-IRB2024_v5_all-all-microsoft/resnet-50-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0--/checkpoint-best.pth 224 dec weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv
bash run_xai_layermap_bs.sh  $STUDY timm_efficientnet-b4 timm_efficientnet-b4 /orange/ruogu.fang/tienyuchang/RETfound_results/${STUDY_NAME}-IRB2024_v5_all-all-timm_efficientnet-b4-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0--/checkpoint-best.pth 224 dec weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv

#bash run_xai_layermap_bs.sh $STUDY SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth /orange/ruogu.fang/tienyuchang/RETfound_results/${STUDY_NAME}-IRB2024_v5_all-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-enc-smpweighted_sum-pre-0-fea-1-1-0.5-decoder_to_encoder-conv---/checkpoint-best.pth 512 enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv
#bash run_xai_layermap_bs.sh $STUDY SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth /orange/ruogu.fang/tienyuchang/RETfound_results/${STUDY_NAME}-IRB2024_v5_all-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-dec-smpweighted_sum-pre-0-fea-1-1-0.5-decoder_to_encoder-conv---/checkpoint-best.pth 512 dec weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 decoder -1 conv

#bash run_xai_layermap_bs.sh $STUDY SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth /orange/ruogu.fang/tienyuchang/RETfound_results/${STUDY_NAME}-IRB2024_v5_all-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask---smp_learnable_alpha-/checkpoint-best.pth 512 fuse weighted_sum 0.5 decoder_to_encoder 9 pre -2 -1 head -1 conv --seg_mask --smp_learnable_alpha
#bash run_xai_layermap_bs.sh $STUDY SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth /orange/ruogu.fang/tienyuchang/RETfound_results/${STUDY_NAME}-IRB2024_v5_all-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth-OCT-bs16ep100lr1e-4optadamw-defaulteval-trsub0-fuse-smpmultiply-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask--/checkpoint-best.pth 512 fuse multiply 0.5 decoder_to_encoder 9 pre -2 -1 head -1 conv --seg_mask