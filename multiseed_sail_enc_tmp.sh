#!/bin/bash
# tmp: run ONLY the sail-enc experiments (SMP enc mode, UF datasets) for one seed; see multiseed_run_all.sh for the options.
#   bash multiseed_sail_enc_tmp.sh 1952 train   # sail-enc training, one job per UF dataset
#   bash multiseed_sail_enc_tmp.sh 1952 eval    # sail-enc XAI eval (rise, as in KDD_rise_enceval.sh) once the training has finished
#   DRY_RUN=1 bash multiseed_sail_enc_tmp.sh 1952 train
#   SAIL_ENC_XAI="rise hirescam gradcamv2" bash multiseed_sail_enc_tmp.sh 1952 eval
ONLY=sail-enc exec bash multiseed_run_all.sh "$@"
