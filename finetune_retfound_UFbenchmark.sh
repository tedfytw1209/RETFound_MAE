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
STUDY=$1 #AMD_all_split 2, Cataract_all_split 2, DR_all_split 5, Glaucoma_all_split 5
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
LR=${4:-"5e-4"}
Num_CLASS=${5:-"2"}
weight_decay=${6:-"0.05"}
NUM_K=${7:-"0"}
ADDCMD=${8:-""}
ADDCMD2=${9:-""}

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $SUBSTUDY
echo $Num_CLASS

# Modify the path to your singularity container 
# sbatch finetune_retfound_UFbenchmark.sh DR_all_split RETFound_mae RETFound_mae_natureOCT 1e-3 5
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune.py --savemodel --global_pool    --batch_size 16     --world_size 1     --model $MODEL     --epochs 100 --lr $LR --layer_decay 0.65     --weight_decay $weight_decay --drop_path 0.2     --nb_classes $Num_CLASS     --data_path /orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/new/split/tune5-eval5/${STUDY}.csv     --task $STUDY-all-$MODEL-$ADDCMD-$ADDCMD2/ --img_dir /orange/ruogu.fang/tienyuchang/all_imgs_paired/ --finetune $FINETUNED_MODEL --num_workers 8 --input_size 224 --num_k $NUM_K $ADDCMD $ADDCMD2
