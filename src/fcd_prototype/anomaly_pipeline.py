"""Placeholder anomaly detection pipeline.

Uses Gaussian smoothing to build a synthetic healthy baseline and computes
voxel-wise absolute differences to produce a rough anomaly map.  Does NOT
require MONAI — only numpy, scipy, and nibabel.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


def _robust_normalize(data: np.ndarray) -> np.ndarray:
    """Map intensities to [0, 1] using the 1st–99th percentile range."""
    lo = float(np.percentile(data, 1))
    hi = float(np.percentile(data, 99))
    if hi == lo:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def run_anomaly_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    smooth_sigma: float = 4.0,
    diff_sigma: float = 2.0,
    threshold_percentile: float = 95.0,
) -> dict[str, Path]:
    """Run the placeholder anomaly pipeline.

    Parameters
    ----------
    input_path:
        Path to a NIfTI (.nii or .nii.gz) patient volume.
    output_dir:
        Directory that will receive patient.nii.gz, baseline_placeholder.nii.gz,
        and anomaly.nii.gz.
    smooth_sigma:
        Gaussian sigma (voxels) used to build the healthy baseline.
    diff_sigma:
        Gaussian sigma for light smoothing of the raw difference map.
    threshold_percentile:
        Keep only the top (100 - threshold_percentile)% of nonzero difference
        values; everything below this percentile is zeroed out.

    Returns
    -------
    dict mapping "patient", "baseline", "anomaly" to their output Path objects.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(input_path))
    raw = img.get_fdata(dtype=np.float32)
    affine = img.affine
    header = img.header.copy()

    # Robust normalise: keeps intensities comparable across patients.
    norm = _robust_normalize(raw)

    # Healthy baseline: heavily smoothed version of the normalised volume.
    baseline = gaussian_filter(norm, sigma=smooth_sigma).astype(np.float32)

    # Absolute voxel-wise difference, lightly smoothed.
    diff_raw = np.abs(norm - baseline).astype(np.float32)
    diff_smooth = gaussian_filter(diff_raw, sigma=diff_sigma).astype(np.float32)

    # Threshold: keep only the top (100 - threshold_percentile)% of nonzero values.
    nonzero_vals = diff_smooth[diff_smooth > 0]
    if nonzero_vals.size > 0:
        cutoff = float(np.percentile(nonzero_vals, threshold_percentile))
    else:
        cutoff = 0.0
    anomaly = np.where(diff_smooth >= cutoff, diff_smooth, 0.0).astype(np.float32)

    def _save(data: np.ndarray, fname: str) -> Path:
        out_img = nib.Nifti1Image(data, affine, header)
        # Update the data-dtype field in the header to float32.
        out_img.header.set_data_dtype(np.float32)
        path = out_dir / fname
        nib.save(out_img, str(path))
        return path

    paths = {
        "patient": _save(norm, "patient.nii.gz"),
        "baseline": _save(baseline, "baseline_placeholder.nii.gz"),
        "anomaly": _save(anomaly, "anomaly.nii.gz"),
    }
    return paths


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Placeholder FCD anomaly pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="Input NIfTI path (.nii or .nii.gz)")
    p.add_argument(
        "--output-dir",
        default="outputs/demo",
        help="Output directory for patient/baseline/anomaly NIfTIs",
    )
    p.add_argument(
        "--smooth-sigma",
        type=float,
        default=4.0,
        help="Gaussian sigma for baseline smoothing (voxels)",
    )
    p.add_argument(
        "--diff-sigma",
        type=float,
        default=2.0,
        help="Gaussian sigma for smoothing the difference map",
    )
    p.add_argument(
        "--threshold-percentile",
        type=float,
        default=95.0,
        help="Threshold: keep top (100 - this)%% of nonzero difference values",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    paths = run_anomaly_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        smooth_sigma=args.smooth_sigma,
        diff_sigma=args.diff_sigma,
        threshold_percentile=args.threshold_percentile,
    )
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
