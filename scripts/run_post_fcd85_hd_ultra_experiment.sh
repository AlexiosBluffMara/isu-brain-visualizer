#!/usr/bin/env bash
set -u

ROOT="/home/soumitty/isu_brain_mri_run"
DATASET="$ROOT/data/openneuro/ds004199"
MAX_WATCHER_PID_FILE="$ROOT/outputs/fcd85_hd_max_experiments.pid"
LOG_DIR="$ROOT/outputs/logs/fcd85_hd_ultra_experiments"
mkdir -p "$LOG_DIR"
cd "$ROOT" || exit 1
echo "$$" > "$ROOT/outputs/fcd85_hd_ultra_experiments.pid"

export TORCH_EXTENSIONS_DIR=/home/soumitty/torch_extensions
export CUDA_HOME=/usr/local/cuda

wait_for_pid_file() {
  local label="$1"
  local pid_file="$2"
  local self_pid="$$"
  local pid=""

  if [ ! -f "$pid_file" ]; then
    echo "[$(date -Is)] no $label pid file at $pid_file" >> "$LOG_DIR/watcher.log"
    return 0
  fi

  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [ -z "$pid" ] || [ "$pid" = "$self_pid" ]; then
    echo "[$(date -Is)] no active external $label pid found" >> "$LOG_DIR/watcher.log"
    return 0
  fi

  while ps -p "$pid" >/dev/null 2>&1; do
    echo "[$(date -Is)] waiting for $label pid=$pid" >> "$LOG_DIR/watcher.log"
    sleep 300
  done
}

run_exp() {
  local name="$1"
  local out="$ROOT/outputs/experiments_hd/$name"
  local log="$LOG_DIR/$name.log"

  echo "[$(date -Is)] START $name points=5000000 gaussians=200000 steps=900 image_size=288" | tee "$log"
  "$ROOT/.venv/bin/python" -m fcd_prototype.end_to_end \
    --dataset-dir "$DATASET" \
    --output-dir "$out" \
    --subjects sub-00058 \
    --synthetic-mode placeholder \
    --roi-strategy provided \
    --force \
    --paper-3dgs-mode \
    --gaussian-device cuda \
    --optimized-gaussian-field-audit-device cuda \
    --max-splat-points 250000 \
    --splat-sampling-mode volume \
    --presentation-splat-points 5000000 \
    --presentation-splat-sampling-mode volume \
    --splat-reconstruction-audit-max-points 200000 \
    --gaussian-num-gaussians 200000 \
    --gaussian-steps 900 \
    --gaussian-image-size 288 \
    --gaussian-volume-sample-count 98304 \
    --optimized-gaussian-field-audit-sample-count 98304 \
    --gaussian-volume-loss-weight 0.50 >> "$log" 2>&1
  echo "[$(date -Is)] END $name exit=$?" >> "$log"
}

wait_for_pid_file "max-detail experiments" "$MAX_WATCHER_PID_FILE"
run_exp "sub-00058_5p0m_200k_ultra_detail"
echo "[$(date -Is)] ULTRA-DETAIL EXPERIMENTS COMPLETE" >> "$LOG_DIR/watcher.log"
