#!/usr/bin/env bash
set -e

source venv/bin/activate

MODULES=(
  "src.01_export_fif_file"
  "src.02_compute_sleep_features"
  "src.util.run_eeg_summary_template"
)

for MOD in "${MODULES[@]}"; do
    echo "########## Running $MOD ##########"
    python3 -m "$MOD"
done
