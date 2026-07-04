"""Tests for the placeholder anomaly pipeline using synthetic data."""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fcd_prototype.anomaly_pipeline import _robust_normalize, run_anomaly_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_volume_with_lesion(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """Create a small synthetic NIfTI with one bright ellipsoid lesion."""
    rng = np.random.default_rng(0)
    shape = (32, 32, 24)
    background = gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma=4) * 200 + 500

    # Embed a bright 4×4×3 ellipsoid near the centre.
    cx, cy, cz = 16, 16, 12
    rx, ry, rz = 3, 3, 2
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    mask = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2 <= 1.0
    background[mask] = background[mask].max() * 1.6

    affine = np.diag([1.0, 1.0, 1.2, 1.0])
    img = nib.Nifti1Image(background, affine)
    img.header.set_zooms((1.0, 1.0, 1.2))
    path = tmp_path / "patient.nii.gz"
    nib.save(img, str(path))
    return path, mask


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_robust_normalize_range():
    data = np.linspace(0, 1000, 100).astype(np.float32)
    n = _robust_normalize(data)
    assert n.min() >= 0.0
    assert n.max() <= 1.0


def test_robust_normalize_flat():
    data = np.ones((5, 5, 5), dtype=np.float32) * 42.0
    n = _robust_normalize(data)
    assert np.allclose(n, 0.0)


def test_pipeline_output_files_exist(tmp_path):
    input_path, _ = _make_volume_with_lesion(tmp_path)
    out_dir = tmp_path / "out"
    paths = run_anomaly_pipeline(input_path, out_dir)
    for key in ("patient", "baseline", "anomaly"):
        assert paths[key].exists(), f"{key} output missing"


def test_pipeline_preserves_affine(tmp_path):
    input_path, _ = _make_volume_with_lesion(tmp_path)
    affine_orig = nib.load(str(input_path)).affine
    out_dir = tmp_path / "out"
    paths = run_anomaly_pipeline(input_path, out_dir)
    for key in ("patient", "baseline", "anomaly"):
        affine_out = nib.load(str(paths[key])).affine
        np.testing.assert_allclose(affine_out, affine_orig, atol=1e-5)


def test_pipeline_preserves_shape(tmp_path):
    input_path, _ = _make_volume_with_lesion(tmp_path)
    orig_shape = nib.load(str(input_path)).shape
    out_dir = tmp_path / "out"
    paths = run_anomaly_pipeline(input_path, out_dir)
    for key in ("patient", "baseline", "anomaly"):
        assert nib.load(str(paths[key])).shape == orig_shape


def test_anomaly_map_nonzero_near_lesion(tmp_path):
    """Top-5% anomaly voxels should overlap the artificial lesion."""
    input_path, lesion_mask = _make_volume_with_lesion(tmp_path)
    out_dir = tmp_path / "out"
    paths = run_anomaly_pipeline(input_path, out_dir, threshold_percentile=95.0)
    anomaly_data = nib.load(str(paths["anomaly"])).get_fdata(dtype=np.float32)
    anomaly_binary = anomaly_data > 0
    # At least one anomaly voxel should fall inside the lesion region.
    overlap = int((anomaly_binary & lesion_mask).sum())
    assert overlap > 0, "Anomaly map missed the synthetic lesion entirely"


def test_patient_output_is_normalised(tmp_path):
    """patient.nii.gz should be in [0, 1] range (after robust normalisation)."""
    input_path, _ = _make_volume_with_lesion(tmp_path)
    out_dir = tmp_path / "out"
    paths = run_anomaly_pipeline(input_path, out_dir)
    data = nib.load(str(paths["patient"])).get_fdata(dtype=np.float32)
    assert data.min() >= -1e-6
    assert data.max() <= 1.0 + 1e-6


def test_custom_threshold_percentile(tmp_path):
    """Lower threshold percentile → more nonzero anomaly voxels."""
    input_path, _ = _make_volume_with_lesion(tmp_path)
    out95 = tmp_path / "out95"
    out50 = tmp_path / "out50"
    paths95 = run_anomaly_pipeline(input_path, out95, threshold_percentile=95.0)
    paths50 = run_anomaly_pipeline(input_path, out50, threshold_percentile=50.0)
    n95 = int((nib.load(str(paths95["anomaly"])).get_fdata() > 0).sum())
    n50 = int((nib.load(str(paths50["anomaly"])).get_fdata() > 0).sum())
    assert n50 > n95, "Lower percentile should yield more anomaly voxels"
