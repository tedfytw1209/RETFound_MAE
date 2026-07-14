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

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune_Chua_Jacqueline.py --savemodel --global_pool --batch_size 128 --world_size 1 --model dual_input_cnn --input_mode images_only --quantitative_features 10 --epochs 50 --lr 1e-4 --weight_decay 0.01 --nb_classes 2 --data_path /blue/ruogu.fang/tienyuchang/IRB2024v5_ADCON_DL_data_retinaf/ad_control_detect_data.csv --task ad_control_detect_data-IRB2024v5_ADCON_DL_data_retinaf-Wisely2-dual_input_cnn-images_only-Thickness-roc_auceval-subset0 --output_dir /orange/ruogu.fang/tienyuchang/RETfound_results --eval_score roc_auc --modality Thickness --img_dir /orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/ --finetune dual_input_cnn --num_workers 16 --input_size 224 --num_k 0 --optimizer adam --momentum 0.9 --lr_scheduler step --schedule_step 10 --schedule_gamma 0.5 --subset_ratio 0 --l1_reg 0.01 --l2_reg 0.01 --transform 2 --split_dir /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split --cv_folds 10 --cv_fold 8 --class_weight