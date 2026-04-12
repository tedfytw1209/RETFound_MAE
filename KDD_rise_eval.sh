#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

SEG_PATH=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth
Thickness_DIR=/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/
Datasets=(DME_binary_all_split AMD_all_split Glaucoma_binary_all_split ERM_all_split)

for DATASET in "${Datasets[@]}"
do
    RESUME=/orange/ruogu.fang/tienyuchang/RETfound_results/${DATASET}-IRB2024_v5_all-all-${SEG_PATH}-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask---smp_learnable_alpha-/checkpoint-best.pth
    sbatch finetune_retfound_UFbenchmark_v5_eval_smp_full.sh \
        $DATASET SMP $SEG_PATH $RESUME 2 512 \
        rise 1024 $Thickness_DIR \
        fuse weighted_sum 0.5 decoder_to_encoder 9 pre -2 -1 head -1 conv \
        --seg_mask --smp_learnable_alpha
done
