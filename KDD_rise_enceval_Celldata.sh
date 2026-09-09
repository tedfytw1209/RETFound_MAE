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

SEG_PATH=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth
Thickness_DIR=/orange/ruogu.fang/tienyuchang/CellData_masks_multiclass_resnet50_new/
Datasets=(DME_all)

#sbatch finetune_retfound_Celldata_smp.sh DME_all SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth 1e-4 2 1e-4 default OCT 0 enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 conv --add_mask --train_no_aug

for DATASET in "${Datasets[@]}"
do
    RESUME=/orange/ruogu.fang/tienyuchang/RETfound_results/${DATASET}-CellData-all-${SEG_PATH}-OCT-bs16ep5lr1e-4optadamw-defaulteval-trsub0-enc-smpweighted_sum-pre-0-fea-1-1-0.5-decoder_to_encoder-conv---/checkpoint-best.pth
    #echo $RESUME
    sbatch finetune_retfound_Celldata_eval_rise.sh \
        $DATASET SMP $SEG_PATH $RESUME 2 512 \
        rise 1024 $Thickness_DIR \
        enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 head -1 conv
done
