#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=144:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

SEG_PATH=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth
Thickness_DIR=/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/
#Datasets=(DME_binary_all_split AMD_all_split Glaucoma_binary_all_split ERM_all_split)
#Datasets=(DME_binary_all_split)
Datasets=(AMD_all_split Glaucoma_binary_all_split ERM_all_split)

for DATASET in "${Datasets[@]}"
do
    sbatch baseline_multirun_XAI_eval_full_risetmp.sh finetune_retfound_UFbenchmark_v5_eval_full2.sh $DATASET 2 RETFound_mae RETFound_mae_natureOCT 224
    sbatch baseline_multirun_XAI_eval_full_risetmp.sh finetune_retfound_UFbenchmark_v5_eval_full2.sh $DATASET 2 resnet-50 microsoft/resnet-50 224
    sbatch baseline_multirun_XAI_eval_full_risetmp.sh finetune_retfound_UFbenchmark_v5_eval_full2.sh $DATASET 2 vit-base-patch16-224 google/vit-base-patch16-224-in21k 224
    sbatch baseline_multirun_XAI_eval_full_risetmp.sh finetune_retfound_UFbenchmark_v5_eval_full2.sh $DATASET 2 timm_efficientnet-b4 timm_efficientnet-b4 380
done
