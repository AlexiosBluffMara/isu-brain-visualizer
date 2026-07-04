#!/usr/bin/env bash
set -u

ROOT="/home/soumitty/isu_brain_mri_run"
DATASET="$ROOT/data/openneuro/ds004199"
PID_FILE="$ROOT/outputs/fcd85_hd_batch.pid"
LOG_DIR="$ROOT/outputs/logs/fcd85_hd_experiments"
mkdir -p "$LOG_DIR"
cd "$ROOT" || exit 1
echo "$$" > "$ROOT/outputs/fcd85_hd_experiments.pid"

export TORCH_EXTENSIONS_DIR=/home/soumitty/torch_extensions
export CUDA_HOME=/usr/local/cuda

if [ -f "$PID_FILE" ]; then
  batch_pid="$(cat "$PID_FILE")"
  while [ -n "$batch_pid" ] && ps -p "$batch_pid" >/dev/null 2>&1; do
    echo "[$(date -Is)] waiting for FCD85 batch pid=$batch_pid" >> "$LOG_DIR/watcher.log"
    sleep 300
  done
fi

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

  echo "[$(date -Is)] START $name points=$points gaussians=$gaussians" | tee "$log"
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
    --splat-reconstruction-audit-max-points 140000 \
    --gaussian-num-gaussians "$gaussians" \
    --gaussian-steps "$steps" \
    --gaussian-image-size "$image_size" \
    --gaussian-volume-sample-count "$volume_samples" \
    --optimized-gaussian-field-audit-sample-count "$volume_samples" \
    --gaussian-volume-loss-weight "$volume_weight" >> "$log" 2>&1
  echo "[$(date -Is)] END $name exit=$?" >> "$log"
}

run_exp "sub-00058_1p2m_55k" 1200000 55000 260 160 16384 0.35
run_exp "sub-00058_1p5m_45k" 1500000 45000 240 144 12288 0.30
run_exp "sub-00058_2p0m_45k_visual_density" 2000000 45000 260 160 16384 0.30
run_exp "sub-00058_2p0m_60k_balanced" 2000000 60000 320 176 24576 0.35
run_exp "sub-00058_2p5m_45k_visual_density" 2500000 45000 280 160 16384 0.30
run_exp "sub-00058_3p0m_100k_detail" 3000000 100000 500 224 32768 0.40
run_exp "sub-00058_4p0m_150k_max_detail" 4000000 150000 750 256 65536 0.45

echo "[$(date -Is)] POST-BATCH EXPERIMENTS COMPLETE" >> "$LOG_DIR/watcher.log"
