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
STUDY=$1 #AMD_all 2, DME_all 2
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
LR=${4:-"5e-4"}
Num_CLASS=${5:-"2"}
weight_decay=${6:-"0.05"}
Eval_score=${7:-"default"}
Modality=${8:-"OCT"} # OCT
SUBSETNUM=${9:-0} # 0, 500, 1000
SMPMode=${10:-"dec"} # dec, enc, fuse
ADDCMD=${11:-""}
ADDCMD2=${12:-""}

NUM_K=0
data_type="OCTDL"
IMG_Path="/orange/ruogu.fang/tienyuchang/OCTDL/"
Epochs=20
OPTIMIZER="adamw" # "adamw" or "sgd"
BATCH_SIZE=4

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $SUBSTUDY
echo $Num_CLASS

# Modify the path to your singularity container 
# sbatch finetune_retfound_OCTDL_smp.sh DME_all SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth 1e-4 2 1e-4 default OCT 0 dec --add_mask --train_no_aug
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune.py --savemodel --global_pool --batch_size $BATCH_SIZE --world_size 1 --model $MODEL --epochs $Epochs --lr $LR --optimizer $OPTIMIZER --layer_decay 0.65 --weight_decay $weight_decay --drop_path 0.0 --nb_classes $Num_CLASS --data_path /orange/ruogu.fang/tienyuchang/${data_type}/${STUDY}.csv --task $STUDY-${data_type}-all-$FINETUNED_MODEL-${Modality}-bs${BATCH_SIZE}ep${Epochs}lr${LR}opt${OPTIMIZER}-${Eval_score}eval-${SMPMode}-$ADDCMD-$ADDCMD2/ --img_dir $IMG_Path --log_dir /orange/ruogu.fang/tienyuchang/RETfound_results --output_dir /orange/ruogu.fang/tienyuchang/RETfound_results --finetune $FINETUNED_MODEL --num_workers 8 --input_size 512 --num_k $NUM_K --eval_score $Eval_score --modality $Modality --new_subset_num $SUBSETNUM --SMPMode $SMPMode $ADDCMD $ADDCMD2
