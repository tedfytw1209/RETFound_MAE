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
STUDY=$1 #AMD_all_split 2, Cataract_all_split 2, DR_all_split 6, Glaucoma_all_split 6, DR_binary_all_split 2, Glaucoma_binary_all_split 2
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
LR=${4:-"5e-3"}
Num_CLASS=${5:-"2"}
weight_decay=${6:-"0.05"}
Eval_score=${7:-"default"}
Modality=${8:-"OCT"} # CFP, OCT, OCT_CFP
SUBSETNUM=${9:-0} # 0, 500, 1000
ADDCMD=${10:-""}
ADDCMD2=${11:-""}

NUM_K=0
data_type="IRB2024_v5"
IMG_Path="/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/"
Epochs=200
OPTIMIZER="sgd" # "adamw" or "sgd"
BATCH_SIZE=16
SMOOTH=0
WARMUP_EPOCHS=20

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $SUBSTUDY
echo $Num_CLASS

# Modify the path to your singularity container 
# sbatch finetune_retfound_UFbenchmark_pytorchvit.sh DR_binary_all_split pytorchvit B_16_imagenet1k 5e-3 2 0 mcc OCT --testval
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune.py --savemodel --global_pool --batch_size $BATCH_SIZE --world_size 1 --model $MODEL --epochs $Epochs --optimizer $OPTIMIZER --lr $LR --layer_decay 0.65 --weight_decay $weight_decay --lr_scheduler cosine --schedule_step 20 --schedule_gamma 0.5 --drop_path 0.2 --smoothing $SMOOTH --warmup_epochs $WARMUP_EPOCHS --nb_classes $Num_CLASS --data_path /orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/${data_type}/split/tune5-eval5/${STUDY}.csv --task $STUDY-${data_type}-all-$FINETUNED_MODEL-${Modality}-bs${BATCH_SIZE}ep${Epochs}lr${LR}opt${OPTIMIZER}-${Eval_score}eval-trsub${SUBSETNUM}-$ADDCMD-$ADDCMD2/ --img_dir $IMG_Path --finetune $FINETUNED_MODEL --num_workers 8 --input_size 224 --num_k $NUM_K --eval_score $Eval_score --modality $Modality --subset_num $SUBSETNUM $ADDCMD $ADDCMD2
