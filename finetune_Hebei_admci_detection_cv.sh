#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=96:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

date;hostname;pwd

module load conda
conda activate retfound_new

# DuCAN Model Training Script
# End-to-end framework for MCI detection based on OCT images and fundus photographs
# Based on research paper specifications with dual-modal cross-attention network

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
MODEL=${4:-"ducan"}
FUNDUS_WEIGHT=${5:-"0.7"}      # α weight for fundus loss (paper specifies 0.7)
OCT_WEIGHT=${6:-"0.7"}         # β weight for OCT loss (paper specifies 0.7)
MULTIMODAL_WEIGHT=${7:-"1.0"}  # Weight for fusion loss (normalized to 1.0)
Relative="Hebei"
NFOLDS=10
FOLDS=(0 1 2 3 4 5 6 7 8 9) # CV folds for training

# Training hyperparameters based on research paper specifications
BS=8                           # Batch size: 8 as specified in paper
LR=${8:-"3e-4"}               # Initial learning rate: 0.0003 as specified in paper
WD=${9:-"1e-2"}              # Weight decay: 0.01 as specified in paper (initial decay factor)
EPOCHS="400"                   # Number of training epochs: 400 as specified in paper
Num_CLASS=${10:-"3"}          # Number of classes (AD, MCI, CN)
START_FOLD=${11:-0} # Start fold index for resuming CV
# Everything after the 11 named positional args is forwarded to the python script.
# This lets you append arbitrary flags, e.g. `... 3 0 --use_img_per_patient --foo bar`.
if [ "$#" -gt 11 ]; then
    shift 11
    ADDCMD="$@"
else
    ADDCMD=""
fi
#SUBSET_RATIO=1.3        # Subset ratio for dataset sampling
SUBSET_RATIO=0        # Subset ratio for dataset sampling
Eval_score="accuracy"         # Evaluation metric
Modality="dual"               # Dual modality (fundus + OCT)
OPTIMIZER="sgd"               # Stochastic Gradient Descent as specified in paper
TRANSFORM="3"                 # Data augmentation with random cropping and reversing

# Data paths for dual-modal training
IMG_Path="/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/"
#data_type="IRB2024v5_ADCON_DL_data"
#SPLIT_DIR="/blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split"

# Scheduler parameters - not specified in paper, using reasonable defaults
Scheduler_step=50
Scheduler_gamma=0.5

# Early stopping parameters based on paper specifications
PATIENCE=50                    # Stop if validation doesn't improve within 50 epochs
EARLY_STOPPING="true"

# Loss function weights based on paper equation (13)
# Lfinal = αLOCT + βLfundus + Lfusion where α=β=0.7
FUNDUS_LOSS_WEIGHT=$FUNDUS_WEIGHT
OCT_LOSS_WEIGHT=$OCT_WEIGHT
MULTIMODAL_LOSS_WEIGHT=$MULTIMODAL_WEIGHT

MASTER_PORT=$(expr 10000 + $(echo -n "${SLURM_JOBID:-0}" | tail -c 4))

# Usage: sbatch finetune_Hebei_admci_detection_cv.sh STUDY data_type SPLIT_DIR MODEL FUNDUS_WEIGHT OCT_WEIGHT MULTIMODAL_WEIGHT LR WD Num_CLASS START_FOLD [ADDCMD...]
# Note: START_FOLD is positional arg 11; any extra flags (ADDCMD) must come AFTER it.
# Example: sbatch finetune_Hebei_admci_detection_cv.sh ad_mci_control_detect_data IRB2024v5_ADCON_DL_data /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split ducan 0.7 0.7 1.0 3e-4 1e-2 3 0 --use_img_per_patient
for fold in ${FOLDS[@]}; do
    [ "$fold" -lt "$START_FOLD" ] && continue
    torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune_Chua_Jacqueline.py \
        --savemodel \
        --global_pool \
        --batch_size $BS \
        --world_size 1 \
        --model $MODEL \
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
        --fundus_loss_weight $FUNDUS_LOSS_WEIGHT \
        --oct_loss_weight $OCT_LOSS_WEIGHT \
        --multimodal_loss_weight $MULTIMODAL_LOSS_WEIGHT \
        --early_stopping \
        --patience $PATIENCE \
        --visualize_samples \
        --use_ducan_preprocessing \
        --warmup_epochs 10 \
        --min_lr 1e-6 \
        --clip_grad 1.0 \
        --cv_folds $NFOLDS --cv_fold $fold \
        --split_dir $SPLIT_DIR \
        --wandb_tags "$WANDB_TAGS" \
        $ADDCMD
done
