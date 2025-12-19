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
conda activate octxai

# ============================================================================
# Example 1: List available models and XAI methods
# ============================================================================
echo "=== Listing available models ==="
python case_study_SMP_layermap.py --list_models

echo "=== Listing available XAI methods ==="
python case_study_SMP_layermap.py --list_xai

python case_study_SMP_layermap.py \
    --dataset_dir /blue/ruogu.fang/tienyuchang/OCT_EDA \
    --dataset_fname sampled_labels01.csv \
    --thickness_dir /orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/ \
    --thickness_csv /orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/thickness_map.csv \
    --model_root /orange/ruogu.fang/tienyuchang/RETfound_results \
    --model_fname checkpoint-best.pth \
    --model SMP_enc \
    --target_module encoder \
    --encoder_idx 10 23 42 52 \
    --task DME \
    --num_samples -1 \
    --xai_method GradCAM HiResCAM GradCAMPlusPlus \
    --batch_size 8 \
    --input_size 512 \
    --nb_classes 2 \
    --load_mask \
    --draw_layer \
    --output_dir ./heatmap_results_production \
    --verbose