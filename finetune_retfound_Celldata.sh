#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4gb
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
STUDY=$1 #CNV_all 2, DME_all 2, DRUSEN_all 2
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
LR=${4:-"5e-4"}
Num_CLASS=${5:-"2"}
weight_decay=${6:-"0.05"}
Eval_score=${7:-"default"}
Modality=${8:-"OCT"} # OCT
ADDCMD=${9:-""}
ADDCMD2=${10:-""}

NUM_K=0
data_type="CellData"
IMG_Path="/orange/ruogu.fang/tienyuchang/"
Epochs=10
OPTIMIZER="adamw" # "adamw" or "sgd"
BATCH_SIZE=16

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $SUBSTUDY
echo $Num_CLASS

# Modify the path to your singularity container 
# bash baseline_multirun_irb2024.sh finetune_retfound_UFbenchmark_irb2024v4_tmp.sh vit-base-patch16-224 google/vit-base-patch16-224-in21k 5e-5 2 0.01 roc_auc OCT
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune.py --savemodel --global_pool --batch_size $BATCH_SIZE --world_size 1 --model $MODEL --epochs $Epochs --lr $LR --optimizer $OPTIMIZER --layer_decay 0.65 --weight_decay $weight_decay --drop_path 0.0 --nb_classes $Num_CLASS --data_path /orange/ruogu.fang/tienyuchang/${data_type}/OCT/${STUDY}.csv --task $STUDY-${data_type}-all-$FINETUNED_MODEL-${Modality}-bs${BATCH_SIZE}ep${Epochs}lr${LR}opt${OPTIMIZER}-${Eval_score}eval-$ADDCMD-$ADDCMD2/ --img_dir $IMG_Path --finetune $FINETUNED_MODEL --num_workers 8 --input_size 224 --num_k $NUM_K --eval_score $Eval_score --modality $Modality $ADDCMD $ADDCMD2
