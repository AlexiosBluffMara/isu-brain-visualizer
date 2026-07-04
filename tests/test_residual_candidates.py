"""Tests for residual-derived lesion candidates."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fcd_prototype.residual_candidates import extract_residual_candidates


def _save(path: Path, data: np.ndarray, affine=None) -> None:
    if affine is None:
        affine = np.diag([1.0, 1.0, 1.5, 1.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), str(path))


def test_residual_candidates_detects_blob(tmp_path):
    residual = np.zeros((24, 24, 20), dtype=np.float32)
    residual[10:14, 9:13, 7:10] = 0.9
    residual[3:5, 3:5, 3:5] = 0.3
    path = tmp_path / "residual.nii.gz"
    out = tmp_path / "candidates"
    _save(path, residual)

    result = extract_residual_candidates(
        residual_heatmap=path,
        output_dir=out,
        max_candidates=1,
        percentile=95.0,
        min_threshold=0.2,
        min_component_voxels=4,
        dilate_iterations=0,
    )

    assert result["selected_count"] == 1
    assert (out / "residual_candidate_mask.nii.gz").exists()
    assert (out / "residual_candidate_00.nii.gz").exists()
    top = result["selected_components"][0]
    assert top["score_max"] == pytest.approx(0.9)
    assert top["voxel_count"] == 48


def test_residual_candidates_empty_heatmap_writes_empty_mask(tmp_path):
    residual = np.zeros((12, 12, 8), dtype=np.float32)
    path = tmp_path / "residual.nii.gz"
    out = tmp_path / "candidates"
    _save(path, residual)

    result = extract_residual_candidates(
        residual_heatmap=path,
        output_dir=out,
        percentile=99.0,
        min_threshold=0.25,
    )

    assert result["selected_count"] == 0
    mask = nib.load(str(out / "residual_candidate_mask.nii.gz")).get_fdata(dtype=np.float32)
    assert float(mask.sum()) == 0.0


def test_residual_candidates_manual_overlap_metrics(tmp_path):
    residual = np.zeros((20, 20, 16), dtype=np.float32)
    residual[8:12, 8:12, 6:10] = 1.0
    roi = np.zeros_like(residual)
    roi[8:12, 8:12, 6:10] = 1.0
    residual_path = tmp_path / "residual.nii.gz"
    roi_path = tmp_path / "roi.nii.gz"
    out = tmp_path / "candidates"
    _save(residual_path, residual)
    _save(roi_path, roi)

    result = extract_residual_candidates(
        residual_heatmap=residual_path,
        output_dir=out,
        manual_roi=roi_path,
        max_candidates=1,
        percentile=90.0,
        min_threshold=0.5,
        min_component_voxels=4,
        dilate_iterations=0,
    )

    overlap = result["manual_overlap"]
    assert overlap["manual_roi_available"] is True
    assert overlap["manual_coverage"] == pytest.approx(1.0)
    assert overlap["candidate_precision"] == pytest.approx(1.0)
    assert overlap["dice"] == pytest.approx(1.0)


def test_residual_candidates_physical_dilation_expands_mask(tmp_path):
    residual = np.zeros((16, 16, 16), dtype=np.float32)
    residual[8, 8, 8] = 1.0
    path = tmp_path / "residual.nii.gz"
    out = tmp_path / "candidates"
    _save(path, residual, np.diag([1.0, 1.0, 2.0, 1.0]))

    result = extract_residual_candidates(
        residual_heatmap=path,
        output_dir=out,
        max_candidates=1,
        percentile=90.0,
        min_threshold=0.5,
        min_component_voxels=1,
        dilate_mm=2.1,
    )

    assert result["dilation_mode"] == "physical_mm"
    assert result["dilate_mm"] == pytest.approx(2.1)
    assert result["candidate_voxels"] > 1


def test_residual_candidates_resamples_to_reference_volume(tmp_path):
    residual = np.ones((10, 10, 8), dtype=np.float32)
    ref = np.ones((20, 20, 16), dtype=np.float32)
    residual_path = tmp_path / "residual.nii.gz"
    ref_path = tmp_path / "ref.nii.gz"
    out = tmp_path / "candidates"
    _save(residual_path, residual, np.diag([2.0, 2.0, 3.0, 1.0]))
    _save(ref_path, ref, np.diag([1.0, 1.0, 1.5, 1.0]))

    result = extract_residual_candidates(
        residual_heatmap=residual_path,
        output_dir=out,
        reference_volume=ref_path,
        percentile=90.0,
        min_threshold=0.5,
        min_component_voxels=4,
        dilate_iterations=0,
    )

    mask_img = nib.load(str(out / "residual_candidate_mask.nii.gz"))
    assert mask_img.shape == ref.shape
    assert np.allclose(mask_img.affine, np.diag([1.0, 1.0, 1.5, 1.0]))
    assert result["selected_count"] == 1
