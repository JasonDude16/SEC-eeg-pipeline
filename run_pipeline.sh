#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ ! -f venv/bin/activate ]; then
    echo "Missing venv. Follow the setup steps in README.md." >&2
    exit 1
fi
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
