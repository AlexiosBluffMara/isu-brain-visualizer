#!/usr/bin/env python3
"""CLI: save axial/coronal/sagittal PNG previews for a NIfTI file.

Usage:
    python scripts/preview_nifti.py path/to/volume.nii.gz
    python scripts/preview_nifti.py path/to/volume.nii.gz --output-dir outputs/previews
    python scripts/preview_nifti.py path/to/volume.nii.gz --prefix my_scan
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fcd_prototype.nifti_preview import save_previews


def main():
    parser = argparse.ArgumentParser(description="Save orthogonal NIfTI slice previews.")
    parser.add_argument("nifti", help="Path to .nii or .nii.gz file")
    parser.add_argument("--output-dir", default="outputs/previews", help="Output directory")
    parser.add_argument("--prefix", default=None, help="File name prefix (default: volume stem)")
    args = parser.parse_args()

    saved = save_previews(args.nifti, output_dir=args.output_dir, prefix=args.prefix)
    for plane, path in saved.items():
        print(f"{plane:10s}: {path}")


if __name__ == "__main__":
    main()
