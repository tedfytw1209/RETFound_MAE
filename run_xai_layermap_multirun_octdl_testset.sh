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
# Full-test-set variant of run_xai_layermap_multirun_octdl.sh: calls
# run_xai_layermap_bs_octdl_testset.sh, which loads the full OCTDL test split via
# --data_path/--img_dir/--use_split test instead of the hand-picked "<task>_sampled" folder.
# ============================================================================
STUDY=${1:-"OCTDL_DME"}    # task label, used for --task / output dir naming
STUDY_NAME=${2:-"DME_all"} # full test-split CSV base filename (no .csv), e.g. DME_all, AMD_all

# bash run_xai_layermap_multirun_octdl_testset.sh OCTDL_DME DME_all
# ============================================================================
# Baselines (evaluated on the full OCTDL test split)
bash run_xai_layermap_bs_octdl_testset.sh "$STUDY" "$STUDY_NAME" RETFound_mae RETFound_mae_natureOCT /orange/ruogu.fang/tienyuchang/RETfound_results/${STUDY_NAME}-OCTDL-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-defaulteval--/checkpoint-best.pth 224 dec weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv
bash run_xai_layermap_bs_octdl_testset.sh "$STUDY" "$STUDY_NAME" vit-base-patch16-224 google/vit-base-patch16-224-in21k /orange/ruogu.fang/tienyuchang/RETfound_results/${STUDY_NAME}-OCTDL-all-google/vit-base-patch16-224-in21k-OCT-bs16ep50lr5e-4optadamw-defaulteval--/checkpoint-best.pth 224 dec weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv
bash run_xai_layermap_bs_octdl_testset.sh "$STUDY" "$STUDY_NAME" resnet-50 microsoft/resnet-50 /orange/ruogu.fang/tienyuchang/RETfound_results/${STUDY_NAME}-OCTDL-all-microsoft/resnet-50-OCT-bs16ep50lr5e-4optadamw-defaulteval--/checkpoint-best.pth 224 dec weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv
bash run_xai_layermap_bs_octdl_testset.sh "$STUDY" "$STUDY_NAME" timm_efficientnet-b4 timm_efficientnet-b4 /orange/ruogu.fang/tienyuchang/RETfound_results/${STUDY_NAME}-OCTDL-all-timm_efficientnet-b4-OCT-bs16ep50lr5e-4optadamw-defaulteval--/checkpoint-best.pth 380 dec weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv

#SMP runs
#bash run_xai_layermap_bs_octdl_testset.sh "$STUDY" "$STUDY_NAME" SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth /orange/ruogu.fang/tienyuchang/RETfound_results/${STUDY_NAME}-OCTDL-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs4ep20lr1e-4optadamw-defaulteval-enc--/checkpoint-best.pth 512 enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv
#bash run_xai_layermap_bs_octdl_testset.sh "$STUDY" "$STUDY_NAME" SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth /orange/ruogu.fang/tienyuchang/RETfound_results/${STUDY_NAME}-OCTDL-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs4ep20lr1e-4optadamw-defaulteval-dec--/checkpoint-best.pth 512 dec weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 decoder -1 conv

bash run_xai_layermap_bs_octdl_testset.sh "$STUDY" "$STUDY_NAME" SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth /orange/ruogu.fang/tienyuchang/RETfound_results/${STUDY_NAME}-OCTDL-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth-OCT-bs16ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask---smp_learnable_alpha-/checkpoint-best.pth 512 fuse weighted_sum 0.5 decoder_to_encoder 9 pre -2 -1 head -1 conv --seg_mask --smp_learnable_alpha
#bash run_xai_layermap_bs_octdl_testset.sh "$STUDY" "$STUDY_NAME" SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth /orange/ruogu.fang/tienyuchang/RETfound_results/${STUDY_NAME}-OCTDL-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth-OCT-bs16ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpmultiply-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask--/checkpoint-best.pth 512 fuse multiply 0.5 decoder_to_encoder 9 pre -2 -1 head -1 conv --seg_mask
