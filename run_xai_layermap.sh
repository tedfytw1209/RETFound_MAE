#!/bin/bash
# ============================================================================
# Example bash script for running case_study_SMP_layermap.py
# XAI Heatmap Generation for SMP Models with Layer-wise Analysis
# ============================================================================

# Set environment
export CUDA_VISIBLE_DEVICES=0

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
    --model SMP_enc SMP_dec \
    --target_module encoder decoder \
    --encoder_idx 10 23 42 52 \
    --decoder_idx 1 3 5 7 9 \
    --task DME \
    --num_samples -1 \
    --xai_method GradCAM HiResCAM GradCAMPlusPlus \
    --batch_size 4 \
    --input_size 512 \
    --nb_classes 2 \
    --load_mask \
    --draw_layer \
    --output_dir ./heatmap_results_production \
    --verbose