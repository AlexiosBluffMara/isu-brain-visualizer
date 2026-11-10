#!/usr/bin/env bash
# Bootstrap lightweight venv and run all tests + generate demo previews.
# Run from repo root: bash scripts/setup_and_test.sh
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_ROOT/.venv"

if [ ! -d "$VENV" ]; then
    echo "Creating .venv ..."
    python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet nibabel numpy scipy matplotlib pytest

echo ""
echo "=== Running pytest ==="
pytest "$REPO_ROOT/tests/" -v

echo ""
echo "=== Generating synthetic demo previews ==="
python3 "$REPO_ROOT/scripts/_gen_demo_previews.py"
