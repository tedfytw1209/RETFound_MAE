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
Thickness_DIR=/orange/ruogu.fang/tienyuchang/OCTDL_masks_multiclass_resnet50_new/
Datasets=(DME_all AMD_all ERM_all)

#sbatch finetune_retfound_OCTDL_smp.sh DME_all SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth 1e-4 2 1e-4 default OCT 0 fuse weighted_sum 0.5 decoder_to_encoder 9 pre -2 -1 conv --seg_mask --smp_learnable_alpha

for DATASET in "${Datasets[@]}"
do
    RESUME=/orange/ruogu.fang/tienyuchang/RETfound_results/${DATASET}-OCTDL-all-${SEG_PATH}-OCT-bs16ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask---smp_learnable_alpha-/checkpoint-best.pth
    #echo $RESUME
    sbatch finetune_retfound_OCTDL_eval_rise.sh \
        $DATASET SMP $SEG_PATH $RESUME 2 512 \
        rise 1024 $Thickness_DIR \
        fuse weighted_sum 0.5 decoder_to_encoder 9 pre -2 -1 head -1 conv \
        --seg_mask --smp_learnable_alpha
done
