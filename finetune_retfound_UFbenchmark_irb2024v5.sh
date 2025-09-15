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
LR=${4:-"5e-4"}
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
Epochs=50
OPTIMIZER="adamw" # "adamw" or "sgd"
BATCH_SIZE=16

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $SUBSTUDY
echo $Num_CLASS

# Modify the path to your singularity container 
# sbatch finetune_retfound_UFbenchmark_irb2024v5.sh DR_all_split RETFound_mae RETFound_mae_natureCFP 5e-4 6 0.05 default CFP
# sbatch finetune_retfound_UFbenchmark_irb2024v5.sh DR_binary_all_split vig_b_224_gelu vig_b_82.6 5e-4 2 0.05 default OCT
# sbatch finetune_retfound_UFbenchmark_irb2024v5.sh Glaucoma_all_split RETFound_mae RETFound_mae_natureOCT 5e-4 6 0.05 default OCT
# sbatch finetune_retfound_UFbenchmark_irb2024v5.sh AMD_all_split RETFound_dinov2 RETFound_dinov2_meh 5e-4 2
# sbatch finetune_retfound_UFbenchmark_irb2024v5.sh Cataract_all_split pytorchvit B_16_imagenet1k 5e-4 6 0.05 mcc OCT --testval
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune.py --savemodel --global_pool --batch_size $BATCH_SIZE --world_size 1 --model $MODEL --epochs $Epochs --lr $LR --optimizer $OPTIMIZER --layer_decay 0.65 --weight_decay $weight_decay --drop_path 0.2 --nb_classes $Num_CLASS --data_path /orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/${data_type}/split/tune5-eval5/${STUDY}.csv --task $STUDY-${data_type}-all-$FINETUNED_MODEL-${Modality}-bs${BATCH_SIZE}ep${Epochs}lr${LR}opt${OPTIMIZER}-${Eval_score}eval-trsub${SUBSETNUM}-$ADDCMD-$ADDCMD2/ --img_dir $IMG_Path --log_dir /orange/ruogu.fang/tienyuchang/RETfound_results --finetune $FINETUNED_MODEL --num_workers 8 --input_size 224 --num_k $NUM_K --eval_score $Eval_score --modality $Modality --subset_num $SUBSETNUM $ADDCMD $ADDCMD2
