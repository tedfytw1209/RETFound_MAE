#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=144:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

RESULTS_DIR=/orange/ruogu.fang/tienyuchang/RETfound_results
Thickness_DIR=/orange/ruogu.fang/tienyuchang/OCTDL_masks_multiclass_resnet50_new/
Datasets=(DME_all AMD_all ERM_all)
#resolution=(224 224 224 380)
#MODELS=(RETFound_mae resnet-50 vit-base-patch16-224 timm_efficientnet-b4)
#FINETUNED_MODELS=(RETFound_mae_natureOCT microsoft/resnet-50 google/vit-base-patch16-224-in21k timm_efficientnet-b4)
MODELS=(timm_efficientnet-b4)
resolution=(380)
FINETUNED_MODELS=(timm_efficientnet-b4)

for DATASET in "${Datasets[@]}"
do
    for i in "${!MODELS[@]}"
    do
        MODEL="${MODELS[$i]}"
        FINETUNED_MODEL="${FINETUNED_MODELS[$i]}"
        RESUME=${RESULTS_DIR}/${DATASET}-OCTDL-all-${FINETUNED_MODEL}-OCT-bs16ep50lr5e-4optadamw-defaulteval--/checkpoint-best.pth
        #echo $RESUME
        sbatch finetune_retfound_OCTDL_eval_rise.sh \
            $DATASET $MODEL $FINETUNED_MODEL $RESUME 2 ${resolution[$i]} \
            rise ${resolution[$i]} $Thickness_DIR \
            enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv
    done
done
