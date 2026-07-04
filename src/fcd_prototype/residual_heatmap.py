"""Patient - synthetic_normal residual heatmap.

Computes |patient - synthetic_normal| and writes a normalized NIfTI.
For FCD subjects the residual concentrates near the ROI.
For controls (no ROI) the residual is expected to be near-zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


def _load_canonical(path: Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data, img.affine.copy(), img.header.copy()


def compute_residual(
    patient_volume: Path,
    synthetic_normal: Path,
    output_path: Path,
    roi_mask: Optional[Path] = None,
    smooth_sigma: float = 1.0,
) -> dict:
    """Compute and save |patient - synthetic_normal| residual heatmap.

    Parameters
    ----------
    patient_volume:
        Normalized patient NIfTI (fuzzed or original).
    synthetic_normal:
        Synthetic normal NIfTI in the same space.
    output_path:
        Output residual NIfTI path.
    roi_mask:
        Optional ROI mask used to compute how much residual falls inside ROI.
    smooth_sigma:
        Light smoothing on the residual (voxels).  0 to disable.

    Returns
    -------
    dict with residual statistics and ROI concentration metric.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    p_data, affine, header = _load_canonical(Path(patient_volume))
    s_data, _, _ = _load_canonical(Path(synthetic_normal))

    if p_data.shape != s_data.shape:
        raise ValueError(
            f"Shape mismatch: patient {p_data.shape} vs synthetic {s_data.shape}. "
            "Resample to the same space before calling compute_residual."
        )

    residual = np.abs(p_data - s_data).astype(np.float32)
    if smooth_sigma > 0:
        residual = gaussian_filter(residual, sigma=smooth_sigma).astype(np.float32)

    r_max = float(residual.max())
    if r_max > 0:
        residual = residual / r_max

    img_out = nib.Nifti1Image(residual, affine, header)
    img_out.header.set_data_dtype(np.float32)
    nib.save(img_out, str(output_path))

    stats: dict = {
        "residual_max_before_norm": r_max,
        "residual_mean": float(residual.mean()),
        "residual_nonzero_fraction": float((residual > 0).mean()),
    }

    roi_concentration: Optional[float] = None
    roi_voxels = 0
    if roi_mask is not None and Path(roi_mask).is_file():
        roi_data, _, _ = _load_canonical(Path(roi_mask))
        mask = (roi_data > 0) & (roi_data.shape == p_data.shape)
        if roi_data.shape == p_data.shape:
            mask = roi_data > 0
            roi_voxels = int(mask.sum())
            if roi_voxels > 0 and residual.sum() > 0:
                roi_mass = float(residual[mask].sum())
                total_mass = float(residual.sum())
                roi_concentration = roi_mass / total_mass
                stats["roi_concentration"] = roi_concentration
                stats["roi_voxels"] = roi_voxels

    notes = []
    if roi_concentration is not None:
        notes.append(
            f"Residual ROI concentration: {roi_concentration:.2%} of total signal inside ROI "
            f"({roi_voxels} voxels). For FCD placeholder mode, expect ≥20% with ROI."
        )
    else:
        notes.append(
            "No ROI mask used for concentration metric. "
            "For controls, residual should be near-zero (mean ≈ 0)."
        )

    stats["notes"] = notes
    return stats
