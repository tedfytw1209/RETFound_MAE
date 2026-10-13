#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

# ─────────────────────────────────────────────────────────────────────────────
# NCex E5 step 1/3: plant the white-square marker into UF DME (train/val/test).
# Stage-I retina mask -> vitreous position (seeded) -> PNG trees + e5_data.csv + markers.csv.
#
#   sbatch NCex_E5_make_data.sh <RUN_ID>          # e.g. sbatch NCex_E5_make_data.sh E5_v1
# Output: ${RESULTS_ROOT}/NCex/E5/<RUN_ID>/{data/{trainval,T2_all,T3_healthy},e5_data.csv,markers.csv,examples/}
# Then:   bash NCex_E5_train.sh <RUN_ID>   and afterwards   bash NCex_E5_eval.sh <RUN_ID>
# ─────────────────────────────────────────────────────────────────────────────

RUN_ID=${1:-E5_v1}
RESULTS_ROOT=/orange/ruogu.fang/tienyuchang/RETfound_XAI_results
SEG_CKPT=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth
DATA_CSV=/orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/IRB2024_v5_all/split/tune8-eval2/DME_binary_all_split.csv
IMG_DIR=/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/

date;hostname;pwd
module load conda
conda activate octxai

python NCex_E5_make_data.py --run_id ${RUN_ID} --results_root ${RESULTS_ROOT} \
    --data_path ${DATA_CSV} --img_dir ${IMG_DIR} --modality OCT --nb_classes 2 --disease_class 1 \
    --seg_ckpt ${SEG_CKPT} --seg_size 512 --mark_frac 0.8 --marker_px 10 --margin_px 5 --seed 2026
