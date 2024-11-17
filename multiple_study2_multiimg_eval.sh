FOLD_NUMS=(1)
DIVIDES=(all) 
MODEL_NAMES=(ad_mci_control ad_mci ad_control mci_control)
PROXIMALS=("" "--bal_sampler")
num_classes=(3 2 2 2)
num_k=0.167

# Loop through all combinations of FOLD_NUM, MODEL_NAME, and PROXIMAL
for DIVIDE in "${DIVIDES[@]}"
do
  for i in "${!MODEL_NAMES[@]}"
  do
    MODEL_NAME=${MODEL_NAMES[$i]}
    NUM_CLASS=${num_classes[$i]}
    for PROXIMAL in "${PROXIMALS[@]}"
    do
      # Create a job name based on the variables
      JOB_NAME="study2_${DIVIDE}${MODEL_NAME}${NUM_CLASS}${PROXIMAL}_multiimg_eval"
      echo $JOB_NAME
      # Submit the job to Slurm
      sbatch --job-name="$JOB_NAME" finetune_retfound_multiimg_eval.sh study2 $MODEL_NAME $NUM_CLASS $DIVIDE $num_k $PROXIMAL
    done
  done
done