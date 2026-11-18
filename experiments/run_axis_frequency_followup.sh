#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${AXIS_FREQUENCY_PYTHON:-.venv/bin/python}"
mkdir -p runs/axis-frequency-followup
"$PYTHON" -m experiments.low_frequency_grid.check_axis_frequency \
  > runs/axis-frequency-followup/preflight.log 2>&1
for architecture in axis_frequency axis_frequency_attn; do
  "$PYTHON" -u -m experiments.low_frequency_grid.run \
    --architecture "$architecture" --frontend standard --folds spk03 \
    --objective paired_supcon --positive-mode same_command \
    --supcon-weight 0.1 --temperature 0.1 \
    --epochs 1000 --schedule-epochs 1000 --stop-after 500 \
    --output "runs/lowfreq-${architecture}-standard-spk03-e1000" \
    > "runs/axis-frequency-followup/${architecture}.log" 2>&1
done
"$PYTHON" -m experiments.report_axis_frequency
