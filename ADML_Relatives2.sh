#! /bin/bash


#Jacq
#sbatch finetune_Jacqueline_adcon_irb2024_v5_cv.sh ad_control_detect_data IRB2024v5_ADCON_DL_data_retinaf /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split convnext_tiny 128 1e-3 5e-4 2 0 0.001 0.2 0

#hebei
sbatch finetune_Hebei_admci_detection_cv.sh ad_control_detect_data IRB2024v5_ADCON_DL_data_retinaf /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split ducan 0.7 0.7 0 3e-4 1e-2 2 2

#Mahendran
sbatch finetune_Mahendran_ad_oct_model_cv.sh ad_control_detect_data IRB2024v5_ADCON_DL_data_retinaf /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split ad_oct_model 256 3 false 7e-5 1e-2 2 5

#Wisely
#sbatch finetune_Wisely_adcon_irb2024_v5_cv.sh ad_control_detect_data IRB2024v5_ADCON_DL_data_retinaf /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split resnet18_paper 0.01 1e-3 0.01 2 0 2

#Wisely2
#sbatch finetune_Wisely2_adcon_irb2024_v5_cv.sh ad_control_detect_data IRB2024v5_ADCON_DL_data_retinaf /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split dual_input_cnn images_only 0.01 1e-4 0.01 2 0 10 3