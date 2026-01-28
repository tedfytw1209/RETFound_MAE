#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

NUM_K=0
MODEL_DIR="/orange/ruogu.fang/tienyuchang/RETfound_results"
#microsoft/resnet-50, timm_efficientnet-b4, google/vit-base-patch16-224-in21k, RETFound_mae_natureOCT
#DATASETS=(AMD_all_split DR_binary_all_split DME_binary_all_split)  # List of datasets
#CLASSES=(2 2 2)  # Number of classes for each dataset
#DATASETS=(DME_binary_all_split)  # List of datasets
#CLASSES=(2)  # Number of classes for each dataset
STEP_PIXELS=1024
Thickness_DIR="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"

python /blue/ruogu.fang/tienyuchang/RETFound_MAE/main_XAI_evaluation.py --batch_size 4 --model RETFound_mae --nb_classes 2 --data_path /orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/IRB2024_v5_all/split/tune8-eval2/AMD_all_split.csv --task AMD_all_split-IRB2024_v5_all-RETFound_mae_natureOCT--XAIgradcamv2-EVAL/ --img_dir /orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/ --finetune RETFound_mae_natureOCT --num_workers 8 --input_size 224 --num_k 0 --resume /orange/ruogu.fang/tienyuchang/RETfound_results/AMD_all_split-IRB2024_v5_all-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0--/checkpoint-best.pth --xai gradcamv2 --step_pixels 224 --output_mask --skip_model_dependent_metrics

python /blue/ruogu.fang/tienyuchang/RETFound_MAE/main_XAI_evaluation.py --batch_size 4 --model RETFound_mae --nb_classes 2 --data_path /orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/IRB2024_v5_all/split/tune8-eval2/ERM_all_split.csv --task ERM_all_split-IRB2024_v5_all-RETFound_mae_natureOCT--XAIgradcamv2-EVAL/ --img_dir /orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/ --finetune RETFound_mae_natureOCT --num_workers 8 --input_size 224 --num_k 0 --resume /orange/ruogu.fang/tienyuchang/RETfound_results/ERM_all_split-IRB2024_v5_all-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0--/checkpoint-best.pth --xai gradcamv2 --step_pixels 224 --output_mask --skip_model_dependent_metrics

python /blue/ruogu.fang/tienyuchang/RETFound_MAE/main_XAI_evaluation.py --batch_size 4 --model RETFound_mae --nb_classes 2 --data_path /orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/IRB2024_v5_all/split/tune8-eval2/Glaucoma_binary_all_split.csv --task Glaucoma_binary_all_split-IRB2024_v5_all-RETFound_mae_natureOCT--XAIgradcamv2-EVAL/ --img_dir /orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/ --finetune RETFound_mae_natureOCT --num_workers 8 --input_size 224 --num_k 0 --resume /orange/ruogu.fang/tienyuchang/RETfound_results/Glaucoma_binary_all_split-IRB2024_v5_all-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0--/checkpoint-best.pth --xai gradcamv2 --step_pixels 224 --output_mask --skip_model_dependent_metrics

python /blue/ruogu.fang/tienyuchang/RETFound_MAE/main_XAI_evaluation.py --batch_size 4 --model RETFound_mae --nb_classes 2 --data_path /orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/IRB2024_v5_all/split/tune8-eval2/DME_binary_all_split.csv --task DME_binary_all_split-IRB2024_v5_all-RETFound_mae_natureOCT--XAIgradcamv2-EVAL/ --img_dir /orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/ --finetune RETFound_mae_natureOCT --num_workers 8 --input_size 224 --num_k 0 --resume /orange/ruogu.fang/tienyuchang/RETfound_results/DME_binary_all_split-IRB2024_v5_all-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0--/checkpoint-best.pth --xai gradcamv2 --step_pixels 224 --output_mask --skip_model_dependent_metrics

python /blue/ruogu.fang/tienyuchang/RETFound_MAE/main_XAI_evaluation.py --batch_size 2 --model RETFound_mae --nb_classes 2 --data_path /orange/ruogu.fang/tienyuchang/OCTDL/AMD_all.csv --task AMD_all-OCTDL-all-RETFound_mae_natureOCT-XAIgradcamv2-EVAL/ --img_dir /orange/ruogu.fang/tienyuchang/OCTDL/ --thickness_dir /orange/ruogu.fang/tienyuchang/OCTDL_masks_multiclass_resnet50_new/ --finetune RETFound_mae_natureOCT --num_workers 8 --input_size 224 --num_k 0 --resume output_dir/AMD_all-OCTDL-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-defaulteval--/checkpoint-best.pth --xai gradcamv2 --step_pixels 224 --SMPMode enc --output_mask --target_module encoder --select_index -1 --smp_fuse_mode weighted_sum --smp_alpha 0.5 --smp_size_match decoder_to_encoder --fusion_dim 0 --enc_idx -1 --dec_idx -1 --smp_classifier conv --skip_model_dependent_metrics

python /blue/ruogu.fang/tienyuchang/RETFound_MAE/main_XAI_evaluation.py --batch_size 2 --model RETFound_mae --nb_classes 2 --data_path /orange/ruogu.fang/tienyuchang/OCTDL/DME_all.csv --task DME_all-OCTDL-all-RETFound_mae_natureOCT-XAIgradcamv2-EVAL/ --img_dir /orange/ruogu.fang/tienyuchang/OCTDL/ --thickness_dir /orange/ruogu.fang/tienyuchang/OCTDL_masks_multiclass_resnet50_new/ --finetune RETFound_mae_natureOCT --num_workers 8 --input_size 224 --num_k 0 --resume output_dir/DME_all-OCTDL-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-defaulteval--/checkpoint-best.pth --xai gradcamv2 --step_pixels 224 --SMPMode enc --output_mask --target_module encoder --select_index -1 --smp_fuse_mode weighted_sum --smp_alpha 0.5 --smp_size_match decoder_to_encoder --fusion_dim 0 --enc_idx -1 --dec_idx -1 --smp_classifier conv --skip_model_dependent_metrics

python /blue/ruogu.fang/tienyuchang/RETFound_MAE/main_XAI_evaluation.py --batch_size 2 --model RETFound_mae --nb_classes 2 --data_path /orange/ruogu.fang/tienyuchang/CellData/OCT/DME_all.csv --thickness_dir /orange/ruogu.fang/tienyuchang/CellData_masks_multiclass_resnet50_new/ --task DME_all-CellData-all-RETFound_mae_natureOCT-XAIgradcamv2-EVAL/ --img_dir /orange/ruogu.fang/tienyuchang/CellData/ --finetune RETFound_mae_natureOCT --num_workers 8 --input_size 224 --num_k 0 --resume output_dir/DME_all-CellData-all-RETFound_mae_natureOCT-OCT-bs16ep3lr5e-4optadamw-defaulteval--/checkpoint-best.pth --xai gradcamv2 --step_pixels 224 --SMPMode enc --output_mask --target_module encoder --select_index -1 --smp_fuse_mode weighted_sum --smp_alpha 0.5 --smp_size_match decoder_to_encoder --fusion_dim 0 --enc_idx -1 --dec_idx -1 --smp_classifier conv --skip_model_dependent_metrics
