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

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune_Chua_Jacqueline.py --savemodel --purge_checkpoint --global_pool --batch_size 256 --world_size 1 --model resnet18_paper --epochs 100 --lr 1e-3 --weight_decay 0.01 --nb_classes 3 --data_path /blue/ruogu.fang/tienyuchang/IRB2024v5_ADCON_DL_study3_retinaf_10yr/ad_mci_control_detect_data.csv --task ad_mci_control_detect_data-IRB2024v5_ADCON_DL_study3_retinaf_10yr-Wisely-resnet18_paper-Thickness-roc_auceval-subset0 --output_dir /blue/ruogu.fang/tienyuchang/RETfound_results --eval_score roc_auc --modality Thickness --img_dir /orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/ --finetune resnet18_paper --num_workers 16 --input_size 128 --num_k 0 --optimizer adamw --momentum 0.9 --lr_scheduler step --schedule_step 10 --schedule_gamma 0.5 --subset_ratio 0 --l1_reg 0.01 --l2_reg 0.01 --transform 2 --split_dir /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_combined_study3_retinaf_diseaseclean_10yr/split_admcicon --cv_folds 10 --cv_fold 3 --wandb_tags study3,ad_mci_control,10yr --class_weight


    