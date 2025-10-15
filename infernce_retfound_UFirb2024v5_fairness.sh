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
RESUME_MODEL=${4:-"RETFound_mae_natureOCT"}
LR=${5:-"5e-4"}
Num_CLASS=${6:-"2"}
weight_decay=${7:-"0.05"}
Eval_score=${8:-"default"}
Modality=${9:-"OCT"} # CFP, OCT, OCT_CFP
SUBSETNUM=${10:-"500"} # 0, 500, 1000
SUBGROUP_COL=${11:-"race_ethnicity"} # age_group, race_ethnicity, gender_source_value
PROTECTED_VALUE=${12:-"NHB"}
PRIVALENT_VALUE=${13:-"NHW"}
# NHB-NHW, HISPANIC-NHW
# FEMALE-MALE

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
# sbatch infernce_retfound_UFirb2024v5_fairness.sh AMD_all_split RETFound_mae RETFound_mae_natureOCT output_dir/AMD_all_split-IRB2024_v5-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-roc_auceval--/checkpoint-best.pth 5e-4 2 0.05 auc OCT 0 race_ethnicity NHB NHW
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_evaluate_fairness.py --savemodel --global_pool --batch_size $BATCH_SIZE --world_size 1 --model $MODEL --epochs $Epochs --lr $LR --optimizer $OPTIMIZER --layer_decay 0.65 --weight_decay $weight_decay --drop_path 0.2 --nb_classes $Num_CLASS --data_path /orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/${data_type}/split/tune5-eval5/${STUDY}.csv --task $STUDY-${data_type}-all-$FINETUNED_MODEL-${Modality}-bs${BATCH_SIZE}ep${Epochs}lr${LR}opt${OPTIMIZER}-${Eval_score}eval-${SUBGROUP_COL}-${PROTECTED_VALUE}_${PRIVALENT_VALUE}-fairness/ --img_dir $IMG_Path --log_dir /orange/ruogu.fang/tienyuchang/RETfound_results --output_dir /orange/ruogu.fang/tienyuchang/RETfound_results --finetune $FINETUNED_MODEL --num_workers 8 --input_size 224 --num_k $NUM_K --eval_score $Eval_score --modality $Modality --resume $RESUME_MODEL --subgroup_path "/blue/ruogu.fang/tienyuchang/OCT_EDA/Subgroup_Ana/subgroup_oct" --subgroup_col $SUBGROUP_COL --protect_value $PROTECTED_VALUE --prevalent_value $PRIVALENT_VALUE --new_subset_num $SUBSETNUM
