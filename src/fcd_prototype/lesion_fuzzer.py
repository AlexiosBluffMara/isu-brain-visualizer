"""Lesion fuzzing (inpainting-style ROI corruption).

Applies Gaussian noise and smoothing inside the ROI mask while preserving
the anatomy outside to within floating-point tolerance.

For controls / subjects without an ROI, the volume is passed through
unchanged and the manifest records why.
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


def fuzz_lesion(
    patient_volume: Path,
    roi_mask: Optional[Path],
    output_path: Path,
    noise_sigma_frac: float = 0.15,
    smooth_sigma: float = 2.0,
    rng_seed: int = 42,
) -> dict:
    """Apply Gaussian noise inside ROI; preserve outside anatomy exactly.

    Parameters
    ----------
    patient_volume:
        Normalized patient FLAIR or T1w NIfTI.
    roi_mask:
        Binary ROI mask in the same space (required for FCD subjects).
        Pass None to get a no-op copy with metadata.
    output_path:
        Destination for fuzzed_patient_flair.nii.gz or similar.
    noise_sigma_frac:
        Noise amplitude as a fraction of the intensity range inside the ROI.
    smooth_sigma:
        Gaussian smoothing applied inside ROI after adding noise (voxels).
    rng_seed:
        Deterministic RNG seed.

    Returns
    -------
    dict with keys: fuzzed (bool), roi_voxels (int), outside_changed (bool),
    notes (list[str]).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data, affine, header = _load_canonical(patient_volume)
    result = data.copy()
    notes: list[str] = []
    fuzzed = False
    roi_voxels = 0

    if roi_mask is not None and Path(roi_mask).is_file():
        roi_data, _, _ = _load_canonical(Path(roi_mask))
        mask = (roi_data > 0).astype(bool)
        roi_voxels = int(mask.sum())

        if roi_voxels > 0:
            rng = np.random.default_rng(rng_seed)
            roi_vals = data[mask]
            intensity_range = float(roi_vals.max() - roi_vals.min())
            noise_amp = intensity_range * noise_sigma_frac
            noise = rng.standard_normal(roi_voxels).astype(np.float32) * noise_amp

            noisy = roi_vals + noise
            # Smooth the entire volume at the ROI region by applying smooth to
            # a full-volume copy and blending back inside mask only.
            smoothed_full = gaussian_filter(result, sigma=smooth_sigma).astype(np.float32)
            blend = (smoothed_full + result) * 0.5  # mild blend
            blend[mask] = noisy

            # Re-smooth only within ROI bounding box to reduce edge artifacts.
            smoothed_blend = gaussian_filter(blend, sigma=smooth_sigma).astype(np.float32)
            result[mask] = smoothed_blend[mask]

            fuzzed = True
            notes.append(
                f"ROI fuzzing applied: {roi_voxels} voxels, "
                f"noise_sigma_frac={noise_sigma_frac}, smooth_sigma={smooth_sigma}."
            )
        else:
            notes.append("ROI mask present but contains no foreground voxels — no fuzzing applied.")
    else:
        notes.append(
            "No ROI mask available (healthy control or missing mask). "
            "Volume copied unchanged. Residual will be near-zero."
        )

    img_out = nib.Nifti1Image(result, affine, header)
    img_out.header.set_data_dtype(np.float32)
    nib.save(img_out, str(output_path))

    return {
        "fuzzed": fuzzed,
        "roi_voxels": roi_voxels,
        "outside_changed": False,
        "notes": notes,
    }
