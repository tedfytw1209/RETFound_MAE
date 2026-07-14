#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=72:00:00
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
WANDB_TAGS="${STUDY_TAG:+$STUDY_TAG,}${TASK}${PERIOD:+,$PERIOD}"
echo "wandb tags: $WANDB_TAGS"
MODEL=${4:-"RETFound_mae"}
FINETUNED_MODEL=$MODEL
BS=${5:-"32"} # 16,32,64
LR=${6:-"1e-3"} # 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05
wd=${7:-"0.005"} # 0.005, 0.0005
Epochs="100"
Num_CLASS=${8:-"2"}
SUBSET_RATIO=${9:-"0"}
Regularization=${10:-"0.01"} # 0.001 to 1 for regularisation loss
DROP_PATH=${11:-"0.2"} # drop path rate, e.g. 0.1, 0.2, 0.7
START_FOLD=${12:-0} # Start fold index for resuming CV
ADDCMD=${13:-""} # Additional command line arguments
Eval_score="roc_auc"
Modality="Thickness"
IMG_Path="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"
#SPLIT_DIR="/blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split"
Patience="10"
Relative="Jacqueline"
NFOLDS=10
FOLDS=(0 1 2 3 4 5 6 7 8 9) # CV folds for training
#FOLDS=(1 2 3 4 5 6 7 8 9) # CV folds for training
#FOLDS=(0) # CV folds for training

#data_type="IRB2024v5_Jacqueline_ADCON_DL_data"
#data_type="IRB2024v5_ADCON_DL_data"

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $SUBSTUDY
echo $Num_CLASS

# Usage: sbatch finetune_Jacqueline_adcon_irb2024_v5_cv.sh STUDY data_type SPLIT_DIR MODEL BS LR wd Num_CLASS SUBSET_RATIO Regularization DROP_PATH [ADDCMD...]
# Example: sbatch finetune_Jacqueline_adcon_irb2024_v5_cv.sh ad_control_detect_data IRB2024v5_ADCON_DL_data /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split convnext_tiny 128 1e-3 5e-4 2 0 0.001 0.2 --use_img_per_patient
for fold in ${FOLDS[@]}; do
    [ "$fold" -lt "$START_FOLD" ] && continue
    torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune_Chua_Jacqueline.py --savemodel --global_pool --batch_size $BS --world_size 1 --model $MODEL --drop_path $DROP_PATH --epochs $Epochs --lr $LR --weight_decay $wd --nb_classes $Num_CLASS --data_path /blue/ruogu.fang/tienyuchang/${data_type}/${STUDY}.csv --task $STUDY-${data_type}-${Relative}-$MODEL-${Modality}-bs${BS}ep${Epochs}lr${LR}wd${wd}-${Eval_score}eval-subset${SUBSET_RATIO} --output_dir /orange/ruogu.fang/tienyuchang/RETfound_results --eval_score $Eval_score --modality $Modality --img_dir $IMG_Path --finetune $FINETUNED_MODEL --num_workers 16 --input_size 224 --num_k 0 --optimizer sgd --momentum 0.9 --lr_scheduler false --early_stopping --patience $Patience --subset_ratio $SUBSET_RATIO --visualize_samples --transform 1 --l1_reg $Regularization --l2_reg $Regularization --cv_folds $NFOLDS --cv_fold $fold --split_dir $SPLIT_DIR --wandb_tags "$WANDB_TAGS" $ADDCMD
done