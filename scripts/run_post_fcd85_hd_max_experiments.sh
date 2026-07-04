#!/usr/bin/env bash
set -u

ROOT="/home/soumitty/isu_brain_mri_run"
DATASET="$ROOT/data/openneuro/ds004199"
BATCH_PID_FILE="$ROOT/outputs/fcd85_hd_batch.pid"
STANDARD_WATCHER_PID_FILE="$ROOT/outputs/fcd85_hd_experiments.pid"
LOG_DIR="$ROOT/outputs/logs/fcd85_hd_max_experiments"
mkdir -p "$LOG_DIR"
cd "$ROOT" || exit 1
echo "$$" > "$ROOT/outputs/fcd85_hd_max_experiments.pid"

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
  local points="$2"
  local gaussians="$3"
  local steps="$4"
  local image_size="$5"
  local volume_samples="$6"
  local volume_weight="$7"
  local out="$ROOT/outputs/experiments_hd/$name"
  local log="$LOG_DIR/$name.log"

  echo "[$(date -Is)] START $name points=$points gaussians=$gaussians steps=$steps image_size=$image_size" | tee "$log"
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
    --max-splat-points 200000 \
    --splat-sampling-mode volume \
    --presentation-splat-points "$points" \
    --presentation-splat-sampling-mode volume \
    --splat-reconstruction-audit-max-points 160000 \
    --gaussian-num-gaussians "$gaussians" \
    --gaussian-steps "$steps" \
    --gaussian-image-size "$image_size" \
    --gaussian-volume-sample-count "$volume_samples" \
    --optimized-gaussian-field-audit-sample-count "$volume_samples" \
    --gaussian-volume-loss-weight "$volume_weight" >> "$log" 2>&1
  echo "[$(date -Is)] END $name exit=$?" >> "$log"
}

wait_for_pid_file "FCD85 batch" "$BATCH_PID_FILE"
wait_for_pid_file "standard post-batch experiments" "$STANDARD_WATCHER_PID_FILE"

run_exp "sub-00058_3p0m_100k_detail" 3000000 100000 500 224 32768 0.40
run_exp "sub-00058_4p0m_150k_max_detail" 4000000 150000 750 256 65536 0.45

echo "[$(date -Is)] MAX-DETAIL EXPERIMENTS COMPLETE" >> "$LOG_DIR/watcher.log"
