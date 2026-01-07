#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

date;hostname;pwd

module load conda
conda activate retfound_new
# Go to home directory
#cd $HOME
DATASET=$1
STUDY=$2
CLASS_MODE=$3
MAX_IMAGES=$4

# sbatch inference_mask.sh uf DME_all_split binary 500
# sbatch inference_mask.sh uf DME_all_split multiclass 500
# sbatch inference_mask.sh celldata DME_all multiclass_resnet50 500
python inference_general.py --dataset $DATASET --study $STUDY --class-mode $CLASS_MODE --max-images $MAX_IMAGES --save-composite --checkpoint /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_${CLASS_MODE}.pth