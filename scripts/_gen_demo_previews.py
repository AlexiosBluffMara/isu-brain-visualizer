#!/usr/bin/env python3
"""Generate synthetic demo preview PNGs under outputs/previews/."""

import sys
import tempfile
import os
import numpy as np
import nibabel as nib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from fcd_prototype.nifti_preview import save_previews

repo_root = Path(__file__).parent.parent
os.chdir(repo_root)

rng = np.random.default_rng(0)
data = (rng.standard_normal((64, 64, 48)) * 300 + 800).astype(np.float32)
img = nib.Nifti1Image(data, np.eye(4))

tmp = tempfile.mktemp(suffix=".nii.gz")
nib.save(img, tmp)
try:
    saved = save_previews(tmp, output_dir="outputs/previews", prefix="demo_synthetic")
    for plane, path in saved.items():
        print(f"  {plane}: {path}")
finally:
    os.unlink(tmp)
