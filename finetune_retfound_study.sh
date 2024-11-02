#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=32gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out


date;hostname;pwd

module load conda
conda activate retfound
# Go to home directory
#cd $HOME
STUDY=$1
SUBSTUDY=$2
Num_CLASS=$3
DIVIDE=${4:-"all"}
ADDCMD=${5:-""}

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $SUBSTUDY
echo $Num_CLASS

# Modify the path to your singularity container 

python -m torch.distributed.launch --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune.py     --batch_size 16     --world_size 1     --model vit_large_patch16     --epochs 50 \
    --blr 5e-3 --layer_decay 0.65     --weight_decay 0.05 --drop_path 0.2     --nb_classes $Num_CLASS     --data_path ./dataset/ADMCI_study/${DIVIDE}_lists_data-$SUBSTUDY.csv     --task $STUDY-$DIVIDE-$SUBSTUDY-$ADDCMD/ \
    --finetune ./pretrain_OCT/RETFound_oct_weights.pth     --input_size 224 $ADDCMD
