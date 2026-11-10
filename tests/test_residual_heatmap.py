"""Tests for residual_heatmap."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fcd_prototype.residual_heatmap import compute_residual


def _make_volume(path: Path, shape=(24, 24, 20), val: float = 0.5, seed=0) -> None:
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal(shape).astype(np.float32) * 0.1 + val).clip(0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.5, 1.0])), str(path))


def _make_roi(path: Path, shape=(24, 24, 20)) -> None:
    data = np.zeros(shape, dtype=np.uint8)
    data[8:14, 8:14, 7:13] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.5, 1.0])), str(path))


# ---------------------------------------------------------------------------

def test_residual_creates_output(tmp_path):
    p = tmp_path / "patient.nii.gz"
    s = tmp_path / "synth.nii.gz"
    out = tmp_path / "residual.nii.gz"
    _make_volume(p, seed=1)
    _make_volume(s, seed=2)
    compute_residual(p, s, out)
    assert out.exists()


def test_residual_normalized_to_unit(tmp_path):
    p = tmp_path / "patient.nii.gz"
    s = tmp_path / "synth.nii.gz"
    out = tmp_path / "residual.nii.gz"
    _make_volume(p, seed=3)
    _make_volume(s, seed=4)
    compute_residual(p, s, out)
    data = nib.load(str(out)).get_fdata(dtype=np.float32)
    assert float(data.max()) <= 1.0 + 1e-5
    assert float(data.min()) >= 0.0 - 1e-5


def test_identical_volumes_give_near_zero_residual(tmp_path):
    p = tmp_path / "vol.nii.gz"
    out = tmp_path / "residual.nii.gz"
    _make_volume(p, seed=5)
    compute_residual(p, p, out)
    data = nib.load(str(out)).get_fdata(dtype=np.float32)
    assert float(data.max()) < 1e-4, "Identical input should yield near-zero residual"


def test_residual_concentrates_near_roi(tmp_path):
    """For FCD-like case: patient has extra signal in ROI; residual should be ROI-concentrated."""
    shape = (24, 24, 20)
    affine = np.diag([1.0, 1.0, 1.5, 1.0])
    rng = np.random.default_rng(7)
    base = (rng.standard_normal(shape).astype(np.float32) * 0.1 + 0.5).clip(0.0, 1.0)

    # Patient has extra intensity inside ROI
    patient_data = base.copy()
    roi_mask_arr = np.zeros(shape, dtype=bool)
    roi_mask_arr[8:14, 8:14, 7:13] = True
    patient_data[roi_mask_arr] += 0.3  # strong signal boost in ROI

    p = tmp_path / "patient.nii.gz"
    s = tmp_path / "synth.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    out = tmp_path / "residual.nii.gz"

    nib.save(nib.Nifti1Image(patient_data, affine), str(p))
    nib.save(nib.Nifti1Image(base, affine), str(s))
    roi_data = roi_mask_arr.astype(np.uint8)
    nib.save(nib.Nifti1Image(roi_data, affine), str(roi))

    stats = compute_residual(p, s, out, roi_mask=roi)
    assert "roi_concentration" in stats
    assert stats["roi_concentration"] >= 0.20, (
        f"ROI concentration {stats['roi_concentration']:.2%} too low; "
        "expected >=20% for FCD-like phantom"
    )


def test_control_no_roi_residual_near_zero(tmp_path):
    """Controls (identical base): residual should be near-zero (mean < 0.01)."""
    p = tmp_path / "vol.nii.gz"
    out = tmp_path / "residual.nii.gz"
    _make_volume(p, seed=8)
    stats = compute_residual(p, p, out)
    data = nib.load(str(out)).get_fdata(dtype=np.float32)
    assert float(data.mean()) < 0.01


def test_residual_shape_matches_input(tmp_path):
    p = tmp_path / "patient.nii.gz"
    s = tmp_path / "synth.nii.gz"
    out = tmp_path / "residual.nii.gz"
    _make_volume(p, shape=(20, 18, 14), seed=9)
    _make_volume(s, shape=(20, 18, 14), seed=10)
    compute_residual(p, s, out)
    orig = nib.load(str(p))
    res = nib.load(str(out))
    assert orig.shape == res.shape


def test_shape_mismatch_raises(tmp_path):
    p = tmp_path / "patient.nii.gz"
    s = tmp_path / "synth_wrong.nii.gz"
    out = tmp_path / "residual.nii.gz"
    _make_volume(p, shape=(20, 20, 16), seed=11)
    _make_volume(s, shape=(16, 16, 12), seed=12)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_residual(p, s, out)


def test_stats_keys_present(tmp_path):
    p = tmp_path / "patient.nii.gz"
    s = tmp_path / "synth.nii.gz"
    out = tmp_path / "residual.nii.gz"
    _make_volume(p, seed=13)
    _make_volume(s, seed=14)
    stats = compute_residual(p, s, out)
    for key in ("residual_max_before_norm", "residual_mean", "residual_nonzero_fraction", "notes"):
        assert key in stats
