#!/bin/bash
# Submits every RISE XAI-evaluation experiment (baseline architectures, enc SMP,
# and fuse SMP) across all three datasets: IRB2024 (UF), OCTDL, and CellData.
# Each line below submits its own batch of sbatch jobs — this script itself is
# meant to be run directly, not sbatch'd.
#
# Usage: bash KDD_rise_multirun_all.sh

set -e

#echo "=== IRB2024 (UF) ==="
#bash KDD_rise_baseline_eval.sh
#bash KDD_rise_enceval.sh
#bash KDD_rise_eval.sh

echo "=== OCTDL ==="
bash KDD_rise_baseline_eval_OCTDL.sh
bash KDD_rise_enceval_OCTDL.sh
bash KDD_rise_eval_OCTDL.sh

echo "=== CellData ==="
bash KDD_rise_baseline_eval_Celldata.sh
bash KDD_rise_enceval_Celldata.sh
bash KDD_rise_eval_Celldata.sh
