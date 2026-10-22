#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3gb
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

# ─────────────────────────────────────────────────────────────────────────────
# NCex E5 step 1/3: plant the white-square marker into UF DME (train/val/test), inside the retina (default)
# or in the background (vitreous, above the retina).
# OCTExplorer layer mask (loaded as in main_XAI_evaluation.py --output_mask) -> seeded position in the region
# -> PNG trees + e5_data.csv + markers.csv + test layer labels and marker regions.
#
#   sbatch NCex_E5_make_data.sh <RUN_ID> [retina|background]
#   e.g. sbatch NCex_E5_make_data.sh E5_v3              # marker inside the retina
#        sbatch NCex_E5_make_data.sh E5_v3_bg background # marker in the vitreous
# Use one RUN_ID per region: train and eval read the region from <RUN_ID>/markers.csv.
# Output: ${RESULTS_ROOT}/NCex/E5/<RUN_ID>/{data/{trainval,T2_all,T3_healthy,retina_labels,marker_region},e5_data.csv,markers.csv,examples/}
# E5_v1 used the earlier vitreous placement (Stage-I mask); do not reuse its RUN_ID.
# Then:   bash NCex_E5_train.sh <RUN_ID>   and afterwards   bash NCex_E5_eval.sh <RUN_ID>
# ─────────────────────────────────────────────────────────────────────────────

RUN_ID=${1:-E5_v3}
MARKER_REGION=${2:-retina}   # retina | background
case "${MARKER_REGION}" in
    retina)     MARGIN_PX=2 ;;
    background) MARGIN_PX=5 ;;   # E5 v1 margin from the retina
    *) echo "Unknown marker region ${MARKER_REGION} (retina|background)"; exit 1 ;;
esac
RESULTS_ROOT=/orange/ruogu.fang/tienyuchang/RETfound_XAI_results
SEG_CKPT=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth
DATA_CSV=/orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/IRB2024_v5_all/split/tune8-eval2/DME_binary_all_split.csv
IMG_DIR=/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/
THICKNESS_DIR=/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/
THICKNESS_CSV=/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/thickness_map.csv

date;hostname;pwd
module load conda
conda activate octxai

python NCex_E5_make_data.py --run_id ${RUN_ID} --results_root ${RESULTS_ROOT} \
    --data_path ${DATA_CSV} --img_dir ${IMG_DIR} --modality OCT --nb_classes 2 --disease_class 1 \
    --marker_region ${MARKER_REGION} --retina_source layers --thickness_dir ${THICKNESS_DIR} --thickness_csv ${THICKNESS_CSV} \
    --seg_ckpt ${SEG_CKPT} --seg_size 512 --mark_frac 0.8 --marker_px 10 --margin_px ${MARGIN_PX} --seed 2026
