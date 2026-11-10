#!/usr/bin/env python3
"""CLI: print NIfTI header + statistics.

Usage:
    python scripts/inspect_nifti.py path/to/volume.nii.gz
"""

import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fcd_prototype.nifti_inspect import inspect_nifti, print_inspection


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <volume.nii|.nii.gz>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    stats = inspect_nifti(path)
    print_inspection(stats)


if __name__ == "__main__":
    main()
