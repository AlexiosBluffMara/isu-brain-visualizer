"""Tests for lesion_fuzzer."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fcd_prototype.lesion_fuzzer import fuzz_lesion


def _make_volume(path: Path, shape=(24, 24, 20), seed=0) -> None:
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal(shape).astype(np.float32) * 0.2 + 0.6).clip(0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.5, 1.0])), str(path))


def _make_roi(path: Path, shape=(24, 24, 20), roi_slice=None) -> None:
    data = np.zeros(shape, dtype=np.uint8)
    if roi_slice is None:
        data[8:14, 8:14, 7:13] = 1
    else:
        data[roi_slice] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.5, 1.0])), str(path))


# ---------------------------------------------------------------------------

def test_fuzz_creates_output(tmp_path):
    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    out = tmp_path / "fuzzed.nii.gz"
    _make_volume(vol)
    _make_roi(roi)
    fuzz_lesion(vol, roi, out)
    assert out.exists()


def test_inside_roi_changes(tmp_path):
    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    out = tmp_path / "fuzzed.nii.gz"
    _make_volume(vol)
    _make_roi(roi)
    info = fuzz_lesion(vol, roi, out, rng_seed=0)

    orig = nib.load(str(vol)).get_fdata(dtype=np.float32)
    fuzzed = nib.load(str(out)).get_fdata(dtype=np.float32)
    roi_data = nib.load(str(roi)).get_fdata() > 0

    # Values inside ROI must change
    assert not np.allclose(orig[roi_data], fuzzed[roi_data]), \
        "Fuzzed values inside ROI must differ from original"
    assert info["fuzzed"] is True
    assert info["roi_voxels"] > 0


def test_outside_roi_preserved(tmp_path):
    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    out = tmp_path / "fuzzed.nii.gz"
    _make_volume(vol)
    _make_roi(roi)
    fuzz_lesion(vol, roi, out, rng_seed=0)

    orig = nib.load(str(vol)).get_fdata(dtype=np.float32)
    fuzzed = nib.load(str(out)).get_fdata(dtype=np.float32)
    roi_data = nib.load(str(roi)).get_fdata() > 0
    outside = ~roi_data

    # Values outside ROI must be identical (outside_changed=False)
    assert info_outside_unchanged(orig, fuzzed, outside)


def info_outside_unchanged(orig, fuzzed, outside_mask) -> bool:
    return np.allclose(orig[outside_mask], fuzzed[outside_mask])


def test_no_roi_copy_unchanged(tmp_path):
    vol = tmp_path / "vol.nii.gz"
    out = tmp_path / "fuzzed.nii.gz"
    _make_volume(vol)
    info = fuzz_lesion(vol, None, out)

    orig = nib.load(str(vol)).get_fdata(dtype=np.float32)
    fuzzed = nib.load(str(out)).get_fdata(dtype=np.float32)

    assert np.allclose(orig, fuzzed), "No-ROI path must copy volume unchanged"
    assert info["fuzzed"] is False
    assert "no ROI mask" in " ".join(info["notes"]).lower() or "no roi" in " ".join(info["notes"]).lower()


def test_missing_roi_file_treated_as_no_roi(tmp_path):
    vol = tmp_path / "vol.nii.gz"
    out = tmp_path / "fuzzed.nii.gz"
    nonexistent_roi = tmp_path / "does_not_exist.nii.gz"
    _make_volume(vol)
    info = fuzz_lesion(vol, nonexistent_roi, out)
    assert info["fuzzed"] is False


def test_fuzz_deterministic(tmp_path):
    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    out1 = tmp_path / "f1.nii.gz"
    out2 = tmp_path / "f2.nii.gz"
    _make_volume(vol)
    _make_roi(roi)
    fuzz_lesion(vol, roi, out1, rng_seed=99)
    fuzz_lesion(vol, roi, out2, rng_seed=99)
    d1 = nib.load(str(out1)).get_fdata()
    d2 = nib.load(str(out2)).get_fdata()
    assert np.allclose(d1, d2)


def test_fuzz_notes_non_empty(tmp_path):
    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    out = tmp_path / "fuzzed.nii.gz"
    _make_volume(vol)
    _make_roi(roi)
    info = fuzz_lesion(vol, roi, out)
    assert isinstance(info["notes"], list)
    assert len(info["notes"]) > 0
