#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

date;hostname;pwd

module load conda
conda activate retfound_new
# Go to home directory
#cd $HOME
STUDY=$1
MODEL=${2:-"RETFound_mae"}
Num_CLASS=${3:-"2"}
FINETUNED_MODEL=${4:-"RETFound_mae_natureOCT"}
RESUME_DIR=${5:-""}
ADDCMD=${6:-""}

DIR_ROOT="/orange/ruogu.fang/tienyuchang/OCTRFF_Data/benchmark"
NUM_K=0
Eval_score="auc"
LR=1e-4
weight_decay=0.05
Modality="CFP"
Epochs=100
OPTIMIZER="adamw" # "adamw" or "sgd"
BATCH_SIZE=16

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $SUBSTUDY
echo $Num_CLASS

# --resume must be omitted entirely (not passed as an empty string) when there's no checkpoint
# to resume from -- main_finetune.py's argparse requires a value for --resume if the flag is
# present at all, so a fresh finetune run (RESUME_DIR unset) would otherwise fail to parse args.
RESUME_ARGS=()
if [ -n "$RESUME_DIR" ]; then
    RESUME_ARGS=(--resume "$RESUME_DIR")
fi

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune.py --savemodel --global_pool --batch_size $BATCH_SIZE --world_size 1 --model $MODEL --epochs $Epochs --lr $LR --optimizer $OPTIMIZER --layer_decay 0.65 --weight_decay $weight_decay --drop_path 0.0 --nb_classes $Num_CLASS --data_path $DIR_ROOT/$STUDY --task $STUDY-$FINETUNED_MODEL-${Modality}-${Eval_score}eval-$ADDCMD/ --finetune $FINETUNED_MODEL --num_workers 8 --input_size 224 --num_k $NUM_K --eval_score $Eval_score --modality $Modality "${RESUME_ARGS[@]}" $ADDCMD
