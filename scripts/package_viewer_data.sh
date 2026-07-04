#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-outputs/end_to_end_fcd85_hd}"
OUT_DIR="${2:-dist/viewer-data}"
PART_SIZE="${PART_SIZE:-1900M}"

if [[ ! -d "$ROOT" ]]; then
  echo "Data root not found: $ROOT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

base_name="isu-brain-visualizer-viewer-data"
manifest="$OUT_DIR/${base_name}.manifest.txt"

{
  echo "source=$ROOT"
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "part_size=$PART_SIZE"
  echo "files=$(find "$ROOT" -type f | wc -l)"
  du -sh "$ROOT"
} > "$manifest"

tar -C "$(dirname "$ROOT")" -czf - "$(basename "$ROOT")" \
  | split -b "$PART_SIZE" - "$OUT_DIR/${base_name}.tar.gz.part-"

sha256sum "$OUT_DIR"/${base_name}.tar.gz.part-* > "$OUT_DIR/${base_name}.sha256"

cat <<EOF
Wrote split archive parts to: $OUT_DIR

To restore:
  cat ${base_name}.tar.gz.part-* > ${base_name}.tar.gz
  tar -xzf ${base_name}.tar.gz

Checksum file:
  ${base_name}.sha256
EOF
