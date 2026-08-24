#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=144:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

date;hostname;pwd

module load conda
conda activate retfound_new
# Go to home directory
#cd $HOME
STUDY=$1
data_type=${2:-"IRB2024v5_ADCON_DL_data"}
SPLIT_DIR=${3:-"/blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split"}
# --- Task-aware split-subdir resolution (study3: ad_control / mci_control / ad_mci_control) ---
# TASK is inferred from the STUDY csv name, e.g. ad_mci_control_detect_data -> ad_mci_control.
# If SPLIT_DIR is a study base dir (leaf is not split*/), the matching subdir is appended:
#   ad_control -> split | mci_control -> split_mcicon | ad_mci_control -> split_admcicon
TASK="${STUDY%_detect_data}"
case "$(basename "$SPLIT_DIR")" in
    split|split_*) : ;;                       # already a resolved split dir; keep as-is
    *)
        case "$TASK" in
            ad_mci_control) SPLIT_DIR="$SPLIT_DIR/split_admcicon" ;;
            mci_control)    SPLIT_DIR="$SPLIT_DIR/split_mcicon" ;;
            *)              SPLIT_DIR="$SPLIT_DIR/split" ;;
        esac ;;
esac
echo "Resolved SPLIT_DIR: $SPLIT_DIR  (TASK=$TASK)"
# wandb tags (reference format "study3,<task>,<period>"): study id inferred from split path + task, period (1yr/3yr/5yr/10yr) from launcher
case "$SPLIT_DIR" in
    *study3*) STUDY_TAG=study3 ;;
    *study2*) STUDY_TAG=study2 ;;
    *)        STUDY_TAG="" ;;
esac
WANDB_TAGS="${STUDY_TAG:+$STUDY_TAG,}${TASK}${PERIOD:+,$PERIOD}${EXTRA_TAGS:+,$EXTRA_TAGS}"
echo "wandb tags: $WANDB_TAGS"
MODEL=${4:-"dual_input_cnn"}
INPUT_MODE=${5:-"images_only"} # all, images_only, gc_ipl_only, octa_only, quantitative_only, gc_ipl_quantitative, octa_quantitative
Regularization=${6:-"0.01"} # 0.001 to 10 for regularisation loss
BS=128 #32*4=128
LR=${7:-"1e-4"} # Adam optimizer learning rate (paper uses adaptive moment estimation)
wd=${8:-"0.01"} # Weight decay 0.01 as specified in paper
#Epochs="100"
Epochs="50"
Num_CLASS=${9:-"2"}
SUBSET_RATIO=${10:-"1.3"}
Eval_score="roc_auc" # AUC as primary performance metric
Modality="Thickness"
IMG_Path="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"
#SPLIT_DIR="/blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split"
Scheduler_step=10
Scheduler_gamma=0.5
Quantitative_Features=${11:-"10"} # Number of quantitative features
START_FOLD=${12:-0} # Start fold index for resuming CV
ADDCMD=${13:-""} # Additional command line arguments
Relative="Wisely2"
NFOLDS=10
FOLDS=(0 1 2 3 4 5 6 7 8 9) # CV folds for training

#data_type="IRB2024v5_Wisely_ADCON_DL_data"
#data_type="IRB2024v5_ADCON_DL_data"

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
#MASTER_PORT=29501

echo $SUBSTUDY
echo $Num_CLASS

# Paper-specified hyperparameters:
# - 100 epochs
# - Adam optimizer (adaptive moment estimation)
# - Weight decay 0.01
# - Binary cross entropy loss
# - AUC as evaluation metric
# - Youden index optimization for thresholding

# Usage examples:
# sbatch finetune_Wisely2_adcon_irb2024_v5_cv.sh mci_control_detect_data IRB2024v5_ADCON_DL_data /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split dual_input_cnn images_only 0.01 1e-4 0.01 2 1.3 10 --use_img_per_patient
for fold in ${FOLDS[@]}; do
    [ "$fold" -lt "$START_FOLD" ] && continue
    torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune_Chua_Jacqueline.py \
        --savemodel \
        --global_pool \
        --batch_size $BS \
        --world_size 1 \
        --model $MODEL \
        --input_mode $INPUT_MODE \
        --quantitative_features $Quantitative_Features \
        --epochs $Epochs \
        --lr $LR \
        --weight_decay $wd \
        --nb_classes $Num_CLASS \
        --data_path /blue/ruogu.fang/tienyuchang/${data_type}/${STUDY}.csv \
        --task $STUDY-${data_type}-${Relative}-$MODEL-${INPUT_MODE}-${Modality}-${Eval_score}eval-subset${SUBSET_RATIO} \
        --output_dir /orange/ruogu.fang/tienyuchang/RETfound_results \
        --eval_score $Eval_score \
        --modality $Modality \
        --img_dir $IMG_Path \
        --finetune $MODEL \
        --num_workers 16 \
        --input_size 224 \
        --num_k 0 \
        --optimizer adam \
        --momentum 0.9 \
        --lr_scheduler step \
        --schedule_step $Scheduler_step \
        --schedule_gamma $Scheduler_gamma \
        --subset_ratio $SUBSET_RATIO \
        --l1_reg $Regularization \
        --l2_reg $Regularization \
        --transform 2 \
        --split_dir $SPLIT_DIR \
        --cv_folds $NFOLDS --cv_fold $fold \
        --wandb_tags "$WANDB_TAGS" \
        $ADDCMD
done