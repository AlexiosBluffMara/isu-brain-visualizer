#!/usr/bin/env bash
set -u

ROOT="/home/soumitty/isu_brain_mri_run"
DATASET="$ROOT/data/openneuro/ds004199"
LOG_DIR="$ROOT/outputs/logs/fcd85_hd_fit_experiments"
mkdir -p "$LOG_DIR"
cd "$ROOT" || exit 1
echo "$$" > "$ROOT/outputs/fcd85_hd_fit_experiments.pid"

export TORCH_EXTENSIONS_DIR=/home/soumitty/torch_extensions
export CUDA_HOME=/usr/local/cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_exp() {
  local name="$1"
  local points="$2"
  local gaussians="$3"
  local steps="$4"
  local image_size="$5"
  local samples="$6"
  local weight="$7"
  local audit_points="$8"
  local out="$ROOT/outputs/experiments_hd/$name"
  local log="$LOG_DIR/$name.log"

  echo "[$(date -Is)] START $name points=$points gaussians=$gaussians steps=$steps image_size=$image_size samples=$samples" | tee "$log"
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
    --splat-reconstruction-audit-max-points "$audit_points" \
    --gaussian-num-gaussians "$gaussians" \
    --gaussian-steps "$steps" \
    --gaussian-image-size "$image_size" \
    --gaussian-volume-sample-count "$samples" \
    --optimized-gaussian-field-audit-sample-count "$samples" \
    --gaussian-volume-loss-weight "$weight" >> "$log" 2>&1
  echo "[$(date -Is)] END $name exit=$?" >> "$log"
}

run_exp "sub-00058_3p0m_60k_fit_boundary" 3000000 60000 320 176 24576 0.35 160000
run_exp "sub-00058_3p5m_45k_visual_density_fit" 3500000 45000 280 160 16384 0.30 180000
run_exp "sub-00058_2p0m_65k_gaussian_probe" 2000000 65000 340 176 24576 0.35 160000

echo "[$(date -Is)] FIT EXPERIMENTS COMPLETE" >> "$LOG_DIR/watcher.log"
