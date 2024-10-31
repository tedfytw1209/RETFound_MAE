FOLD_NUMS=(1)
MODEL_NAMES=(ad_mci_control ad_mci ad_control mci_control)
PROXIMALS=(normal balsam)

# Loop through all combinations of FOLD_NUM, MODEL_NAME, and PROXIMAL
for FOLD_NUM in "${FOLD_NUMS[@]}"
do
  for MODEL_NAME in "${MODEL_NAMES[@]}"
  do
    for PROXIMAL in "${PROXIMALS[@]}"
    do
      # Create a job name based on the variables
      JOB_NAME="f${FOLD_NUM}${MODEL_NAME}_${PROXIMAL}"

      # Submit the job to Slurm
      sbatch --job-name="$JOB_NAME" \
              finetune_retfound_study2.sh $MODEL_NAME $PROXIMAL study2
    done
  done
done