#!/usr/bin/env bash
set -u

ROOT="/home/soumitty/isu_brain_mri_run"
DATASET="$ROOT/data/openneuro/ds004199"
OUT="$ROOT/outputs/end_to_end_fcd85_hd"
SUBJECTS_FILE="$ROOT/outputs/fcd85_subjects.txt"
LOG_DIR="$ROOT/outputs/logs/fcd85_hd"
SUMMARY="$ROOT/outputs/fcd85_hd_progress.tsv"

mkdir -p "$OUT" "$LOG_DIR" "$(dirname "$SUMMARY")"
cd "$ROOT" || exit 1

export TORCH_EXTENSIONS_DIR=/home/soumitty/torch_extensions
export CUDA_HOME=/usr/local/cuda

printf "timestamp\tsubject\tstatus\tseconds\n" > "$SUMMARY"

is_done() {
  local subject="$1"
  "$ROOT/.venv/bin/python" - "$OUT/$subject/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
try:
    manifest = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
artifacts = manifest.get("artifacts", {})
status = manifest.get("stage_status", {})
if (
    artifacts.get("patient_splat_lesion_highlighted_ply")
    and status.get("splat_ply") == "ok"
    and status.get("gaussian_field") == "ok"
    and str(status.get("optimized_gaussian_field_audit", "")).startswith("ok")
):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

while IFS= read -r subject; do
  [ -n "$subject" ] || continue

  if is_done "$subject"; then
    printf "%s\t%s\t%s\t%s\n" "$(date -Is)" "$subject" "skipped_complete" "0" >> "$SUMMARY"
    continue
  fi

  start=$(date +%s)
  log="$LOG_DIR/${subject}.log"
  echo "[$(date -Is)] START $subject" | tee "$log"

  if "$ROOT/.venv/bin/python" -m fcd_prototype.end_to_end \
    --dataset-dir "$DATASET" \
    --output-dir "$OUT" \
    --subjects "$subject" \
    --synthetic-mode placeholder \
    --roi-strategy provided \
    --force \
    --paper-3dgs-mode \
    --gaussian-device cuda \
    --optimized-gaussian-field-audit-device cuda \
    --max-splat-points 200000 \
    --splat-sampling-mode volume \
    --presentation-splat-points 900000 \
    --presentation-splat-sampling-mode volume \
    --splat-reconstruction-audit-max-points 120000 \
    --gaussian-num-gaussians 45000 \
    --gaussian-steps 220 \
    --gaussian-image-size 144 \
    --gaussian-volume-sample-count 12288 \
    --optimized-gaussian-field-audit-sample-count 12288 \
    --gaussian-volume-loss-weight 0.30 >> "$log" 2>&1; then
    status="ok"
  else
    status="failed_process"
  fi

  end=$(date +%s)
  seconds=$((end - start))
  if is_done "$subject"; then
    status="ok_verified"
  elif [ "$status" = "ok" ]; then
    status="finished_but_not_verified"
  fi
  printf "%s\t%s\t%s\t%s\n" "$(date -Is)" "$subject" "$status" "$seconds" >> "$SUMMARY"
  echo "[$(date -Is)] END $subject $status ${seconds}s" >> "$log"
done < "$SUBJECTS_FILE"

"$ROOT/.venv/bin/python" -m fcd_prototype.cohort_evaluation \
  "$OUT" \
  --output-json "$ROOT/outputs/fcd85_hd_cohort_evaluation.json" \
  --output-csv "$ROOT/outputs/fcd85_hd_cohort_evaluation.csv" \
  >> "$LOG_DIR/cohort_evaluation.log" 2>&1 || true

echo "[$(date -Is)] BATCH COMPLETE" >> "$LOG_DIR/batch.log"
