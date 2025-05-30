#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module load conda
conda activate retfound_new
# Go to home directory
#cd $HOME
STUDY=$1
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
LR=${4:-"1e-3"}
Epochs=${5:-"100"}
Num_CLASS=${6:-"2"}
NUM_K=${7:-"0"}
ADDCMD=${8:-""}
ADDCMD2=${9:-""}

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $SUBSTUDY
echo $Num_CLASS

# Modify the path to your singularity container 

torchrun --nproc_per_node=1 --master_port=48798 main_finetune.py --savemodel --global_pool    --batch_size 16     --world_size 1     --model $MODEL     --epochs 200 --lr $LR --layer_decay 0.65     --weight_decay 0.05 --drop_path 0.2     --nb_classes $Num_CLASS     --data_path /blue/ruogu.fang/tienyuchang/IRB2024_DL_data/${STUDY}.csv     --task $STUDY-all-$MODEL-$ADDCMD-$ADDCMD2/ --img_dir /orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/ --finetune $FINETUNED_MODEL --num_workers 8 --input_size 224 --num_k $NUM_K $ADDCMD $ADDCMD2
