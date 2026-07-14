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

MASTER_PORT=$(expr 10000 + $(echo -n "${SLURM_JOBID:-0}" | tail -c 4))

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune_Chua_Jacqueline.py --savemodel --global_pool --batch_size 8 --world_size 1 --model ducan --epochs 400 --lr 3e-4 --weight_decay 1e-2 --nb_classes 2 --data_path /blue/ruogu.fang/tienyuchang/IRB2024v5_ADCON_DL_data_retinaf_diseaseclean/ad_control_detect_data.csv --task ad_control_detect_data-IRB2024v5_ADCON_DL_data_retinaf_diseaseclean-Hebei-ducan-dual-bs8ep400lr3e-4wd1e-2-accuracyeval-subset0 --output_dir /orange/ruogu.fang/tienyuchang/RETfound_results --eval_score accuracy --modality dual --img_dir /blue/ruogu.fang/tienyuchang/IRB2024_imgs_paired/ --finetune ducan --num_workers 16 --input_size 224 --num_k 0 --optimizer sgd --momentum 0.9 --lr_scheduler step --schedule_step 50 --schedule_gamma 0.5 --subset_ratio 0 --transform 3 --fundus_loss_weight 0.7 --oct_loss_weight 0.7 --multimodal_loss_weight 0 --early_stopping --patience 50 --visualize_samples --use_ducan_preprocessing --warmup_epochs 10 --min_lr 1e-6 --clip_grad 1.0 --cv_folds 10 --cv_fold 9 --split_dir /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf_diseaseclean/split --class_weight