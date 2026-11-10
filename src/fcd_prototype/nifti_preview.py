"""NIfTI orthogonal slice preview utilities."""

from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe in headless/WSL
import matplotlib.pyplot as plt


def _robust_normalize(data: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Clip to percentile range and scale to [0, 1]."""
    p_low = np.percentile(data, low)
    p_high = np.percentile(data, high)
    if p_high == p_low:
        return np.zeros_like(data, dtype=np.float32)
    clipped = np.clip(data, p_low, p_high)
    return ((clipped - p_low) / (p_high - p_low)).astype(np.float32)


def save_previews(
    nifti_path: str,
    output_dir: str = "outputs/previews",
    prefix: str | None = None,
) -> dict[str, str]:
    """
    Load a NIfTI file and save axial, coronal, and sagittal PNG previews.

    Returns a dict mapping plane name -> saved PNG path.
    """
    img = nib.load(nifti_path)
    # Reorient to RAS so slice directions are consistent
    img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float64)

    # Collapse any 4D volume to 3D by taking the first time point
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume after squeeze, got shape {data.shape}")

    norm = _robust_normalize(data)
    cx, cy, cz = [s // 2 for s in norm.shape]

    slices = {
        "axial": norm[:, :, cz].T,        # z-centre, view from above
        "coronal": norm[:, cy, :].T,       # y-centre, view from front
        "sagittal": norm[cx, :, :].T,      # x-centre, view from side
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = prefix or Path(nifti_path).name.replace(".nii.gz", "").replace(".nii", "")
    saved = {}

    for plane, arr in slices.items():
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(arr, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.set_title(f"{stem} — {plane}")
        ax.axis("off")
        out_path = out_dir / f"{stem}_{plane}.png"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        saved[plane] = str(out_path)

    return saved
