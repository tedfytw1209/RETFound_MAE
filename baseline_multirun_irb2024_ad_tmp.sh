#! /bin/bash

SCRIPT=$1

NUM_K=0

DATASETS=(ad_control_detect_data)  # List of datasets
LAYERS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14)
for i in "${!DATASETS[@]}"
do
    # Create a job name based on the variables
    DATASET="${DATASETS[$i]}"
    for LAYER in "${LAYERS[@]}"
    do
        # Submit the job to Slurm
        echo "sbatch $SCRIPT $DATASET $LAYER"
        sbatch $SCRIPT $DATASET $LAYER
    done
done