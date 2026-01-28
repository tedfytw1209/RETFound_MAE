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

#bash finetune_retfound_OCTDL.sh ERM_all vit-base-patch16-224 google/vit-base-patch16-224-in21k
#bash finetune_retfound_OCTDL.sh ERM_all RETFound_mae RETFound_mae_natureOCT
#bash finetune_retfound_OCTDL.sh ERM_all resnet-50 microsoft/resnet-50
#bash finetune_retfound_OCTDL.sh ERM_all timm_efficientnet-b4 timm_efficientnet-b4

bash finetune_retfound_OCTDL_smp.sh ERM_all SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth 1e-4 2 1e-4 default OCT 0 enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 conv
bash finetune_retfound_OCTDL_smp.sh ERM_all SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth 1e-4 2 1e-4 default OCT 0 dec weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 conv
bash finetune_retfound_OCTDL_smp.sh ERM_all SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth 1e-4 2 1e-4 default OCT 0 fuse weighted_sum 0.5 decoder_to_encoder 9 pre -2 -1 conv --seg_mask --smp_learnable_alpha
