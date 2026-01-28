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

bash finetune_retfound_OCTDL.sh ERM_all vit-base-patch16-224 google/vit-base-patch16-224-in21k
bash finetune_retfound_OCTDL.sh ERM_all RETFound_mae RETFound_mae_natureOCT
bash finetune_retfound_OCTDL.sh ERM_all resnet-50 microsoft/resnet-50
bash finetune_retfound_OCTDL.sh ERM_all timm_efficientnet-b4 timm_efficientnet-b4

python /blue/ruogu.fang/tienyuchang/RETFound_MAE/main_finetune_smp.py --savemodel --global_pool --batch_size 16 --world_size 1 --model SMP --epochs 50 --lr 1e-4 --optimizer adamw --layer_decay 0.65 --weight_decay 1e-4 --drop_path 0.0 --nb_classes 2 --data_path /orange/ruogu.fang/tienyuchang/OCTDL/ERM_all.csv --task ERM_all-OCTDL-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth-OCT-bs16ep50lr1e-4optadamw-defaulteval-trsub0-dec-smpweighted_sum-pre-0-fea-1-1-0.5-decoder_to_encoder-conv---/ --img_dir /orange/ruogu.fang/tienyuchang/OCTDL/ --log_dir /orange/ruogu.fang/tienyuchang/RETfound_results --output_dir /orange/ruogu.fang/tienyuchang/RETfound_results --finetune /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth --num_workers 8 --input_size 512 --num_k 0 --eval_score default --modality OCT --visualize_samples --new_subset_num 0 --SMPMode dec --smp_fuse_mode weighted_sum --fusion_dim 0 --align pre --smp_alpha 0.5 --smp_size_match decoder_to_encoder --enc_idx -1 --dec_idx -1 --smp_classifier conv

python /blue/ruogu.fang/tienyuchang/RETFound_MAE/main_finetune_smp.py --savemodel --global_pool --batch_size 16 --world_size 1 --model SMP --epochs 50 --lr 1e-4 --optimizer adamw --layer_decay 0.65 --weight_decay 1e-4 --drop_path 0.0 --nb_classes 2 --data_path /orange/ruogu.fang/tienyuchang/OCTDL/ERM_all.csv --task ERM_all-OCTDL-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth-OCT-bs16ep50lr1e-4optadamw-defaulteval-trsub0-enc-smpweighted_sum-pre-0-fea-1-1-0.5-decoder_to_encoder-conv---/ --img_dir /orange/ruogu.fang/tienyuchang/OCTDL/ --log_dir /orange/ruogu.fang/tienyuchang/RETfound_results --output_dir /orange/ruogu.fang/tienyuchang/RETfound_results --finetune /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth --num_workers 8 --input_size 512 --num_k 0 --eval_score default --modality OCT --visualize_samples --new_subset_num 0 --SMPMode enc --smp_fuse_mode weighted_sum --fusion_dim 0 --align pre --smp_alpha 0.5 --smp_size_match decoder_to_encoder --enc_idx -1 --dec_idx -1 --smp_classifier conv

python /blue/ruogu.fang/tienyuchang/RETFound_MAE/main_finetune_smp.py --savemodel --global_pool --batch_size 16 --world_size 1 --model SMP --epochs 50 --lr 1e-4 --optimizer adamw --layer_decay 0.65 --weight_decay 1e-4 --drop_path 0.0 --nb_classes 2 --data_path /orange/ruogu.fang/tienyuchang/OCTDL/ERM_all.csv --task ERM_all-OCTDL-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth-OCT-bs16ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask---smp_learnable_alpha-/ --img_dir /orange/ruogu.fang/tienyuchang/OCTDL/ --log_dir /orange/ruogu.fang/tienyuchang/RETfound_results --output_dir /orange/ruogu.fang/tienyuchang/RETfound_results --finetune /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth --num_workers 8 --input_size 512 --num_k 0 --eval_score default --modality OCT --visualize_samples --new_subset_num 0 --SMPMode fuse --smp_fuse_mode weighted_sum --fusion_dim 9 --align pre --smp_alpha 0.5 --smp_size_match decoder_to_encoder --enc_idx -2 --dec_idx -1 --smp_classifier conv --seg_mask --smp_learnable_alpha
