#! /bin/bash

SCRIPT=$1

NUM_K=0

#MODELS=(alexnet vgg11 resnet18 googlenet shufflenet convnext_tiny regnet_x_16gf efficientnet_b0 vit_b_16 swin_b)  # List of models
MODELS=(resnet18_paper)  # List of models
#LRS=(0.0001)
WDS=(0.005)
BSS=(32)
LRS=(0.001 0.005 0.01 0.05)
#WDS=(0.005 0.0005)
#BSS=(16 32 64)
REGULARIZATION=(0.01)

#bash baseline_multirun_relative.sh finetune_relative1_adcon_irb2024_v5.sh
DATASET="ad_control_detect_data"
NUM_CLASS=2
SUBSET_RATIO=1
# Nested loops to test all combinations
for MODEL in "${MODELS[@]}"
do
    for LR in "${LRS[@]}"
    do
        for WD in "${WDS[@]}"
        do
            for BS in "${BSS[@]}"
            do
                for REG in "${REGULARIZATION[@]}"
                do
                echo "Running MODEL: $MODEL, LR: $LR, WD: $WD, BS: $BS, REG: $REG"
                # Submit the job to Slurm
                echo "sbatch $SCRIPT $DATASET $MODEL $BS $LR $WD $NUM_CLASS $SUBSET_RATIO $REG"
                sbatch $SCRIPT $DATASET $MODEL $BS $LR $WD $NUM_CLASS $SUBSET_RATIO $REG
                done
            done
        done
    done
done