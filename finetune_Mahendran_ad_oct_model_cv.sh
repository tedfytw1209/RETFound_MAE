#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

date;hostname;pwd

module load conda
conda activate retfound_new

# AD-OCT Model Training Script
# Based on research paper specifications for Alzheimer's Disease detection using OCT images

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
WANDB_TAGS="${STUDY_TAG:+$STUDY_TAG,}${TASK}${PERIOD:+,$PERIOD}"
echo "wandb tags: $WANDB_TAGS"
MODEL=${4:-"ad_oct_model"}
FEATURE_CHANNELS=${5:-"256"}  # Number of feature channels (default: 256)
NUM_GROUPS=${6:-"3"}          # Number of polarization feature groups (default: 3)
INCLUDE_LOCALIZATION=${7:-"false"}  # Enable localization head (true/false)

# Training hyperparameters based on research paper
BS=18                         # Batch size as specified in paper
LR=${8:-"7e-5"}              # Learning rate: 7e-5 as specified in paper  
WD=${9:-"1e-2"}              # Weight decay: 1e-2 as specified in paper
EPOCHS="100"                  # Number of training epochs
Num_CLASS=${10:-"2"}          # Number of classes (AD vs Control)
START_FOLD=${11:-0} # Start fold index for resuming CV
ADDCMD=${12:-""} # Additional command line arguments
Eval_score="roc_auc"         # Evaluation metric
Modality="OCT"               # Modality type
OPTIMIZER="adabelief"        # AdaBelief optimizer as specified in paper
TRANSFORM="3"                # AD-OCT specific data augmentation
#SUBSET_RATIO=1
#SUBSET_RATIO=4
SUBSET_RATIO=0
Relative="Mahendran"
NFOLDS=10
FOLDS=(0 1 2 3 4 5 6 7 8 9) # CV folds for training

# Data paths
IMG_Path="/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/"
#SPLIT_DIR="/blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split"
#data_type="IRB2024v5_ADCON_DL_data"

# Scheduler parameters
Scheduler_step=20
Scheduler_gamma=0.5

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# Construct localization flag
LOCALIZATION_FLAG=""
if [ "$INCLUDE_LOCALIZATION" = "true" ]; then
    LOCALIZATION_FLAG="--include_localization"
fi

# Usage examples:
# sbatch finetune_Mahendran_ad_oct_model_cv.sh ad_control_detect_data IRB2024v5_ADCON_DL_data /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split ad_oct_model 256 3 false 7e-5 1e-2 2 --use_img_per_patient
for fold in ${FOLDS[@]}; do
    [ "$fold" -lt "$START_FOLD" ] && continue
    torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune_Chua_Jacqueline.py \
        --savemodel \
        --global_pool \
        --batch_size $BS \
        --world_size 1 \
        --model $MODEL \
        --feature_channels $FEATURE_CHANNELS \
        --num_groups $NUM_GROUPS \
        $LOCALIZATION_FLAG \
        --epochs $EPOCHS \
        --lr $LR \
        --weight_decay $WD \
        --nb_classes $Num_CLASS \
        --data_path /blue/ruogu.fang/tienyuchang/${data_type}/${STUDY}.csv \
        --task $STUDY-${data_type}-${Relative}-$MODEL-${Modality}-bs${BS}ep${EPOCHS}lr${LR}wd${WD}-${Eval_score}eval-subset${SUBSET_RATIO} \
        --output_dir /orange/ruogu.fang/tienyuchang/RETfound_results \
        --eval_score $Eval_score \
        --modality $Modality \
        --img_dir $IMG_Path \
        --finetune $MODEL \
        --split_dir $SPLIT_DIR \
        --num_workers 16 \
        --input_size 224 \
        --num_k 0 \
        --optimizer $OPTIMIZER \
        --momentum 0.9 \
        --lr_scheduler step \
        --schedule_step $Scheduler_step \
        --schedule_gamma $Scheduler_gamma \
        --subset_ratio $SUBSET_RATIO \
        --transform $TRANSFORM \
        --use_focal_loss \
        --focal_gamma 2.0 \
        --early_stopping \
        --patience 15 \
        --cv_folds $NFOLDS --cv_fold $fold \
        --visualize_samples \
        --wandb_tags "$WANDB_TAGS" \
        $ADDCMD
done