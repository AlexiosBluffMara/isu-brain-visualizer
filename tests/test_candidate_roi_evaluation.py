"""Tests for candidate ROI recall evaluation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fcd_prototype.candidate_roi_evaluation import (
    dice_score,
    evaluate_candidate_recall,
    mask_coverage,
    sweep_candidate_recall,
)


# ---------------------------------------------------------------------------
# Test dice and coverage metrics
# ---------------------------------------------------------------------------


def test_dice_score_identical():
    """Identical masks should give dice = 1.0."""
    a = np.ones((10, 10, 10), dtype=np.float32)
    b = np.ones((10, 10, 10), dtype=np.float32)
    assert dice_score(a, b) == 1.0


def test_dice_score_disjoint():
    """Disjoint masks should give dice = 0.0."""
    a = np.zeros((10, 10, 10), dtype=np.float32)
    a[:5, :, :] = 1.0
    b = np.zeros((10, 10, 10), dtype=np.float32)
    b[5:, :, :] = 1.0
    assert dice_score(a, b) == 0.0


def test_dice_score_partial_overlap():
    """Partial overlap should give dice in (0, 1)."""
    a = np.zeros((10, 10, 10), dtype=np.float32)
    a[:8, :, :] = 1.0
    b = np.zeros((10, 10, 10), dtype=np.float32)
    b[2:, :, :] = 1.0
    dice = dice_score(a, b)
    assert 0.0 < dice < 1.0
    assert abs(dice - 0.75) < 0.01


def test_dice_score_empty():
    """Both empty should give dice = 1.0."""
    a = np.zeros((10, 10, 10), dtype=np.float32)
    b = np.zeros((10, 10, 10), dtype=np.float32)
    assert dice_score(a, b) == 1.0


def test_mask_coverage_full():
    """Candidate fully covering reference should give 1.0."""
    reference = np.ones((10, 10, 10), dtype=np.float32)
    candidate = np.ones((10, 10, 10), dtype=np.float32)
    assert mask_coverage(candidate, reference) == 1.0


def test_mask_coverage_zero():
    """Disjoint candidate and reference should give 0.0."""
    reference = np.zeros((10, 10, 10), dtype=np.float32)
    reference[:5, :, :] = 1.0
    candidate = np.zeros((10, 10, 10), dtype=np.float32)
    candidate[5:, :, :] = 1.0
    assert mask_coverage(candidate, reference) == 0.0


def test_mask_coverage_partial():
    """Partial coverage should give intermediate value."""
    reference = np.ones((10, 10, 10), dtype=np.float32)
    candidate = np.zeros((10, 10, 10), dtype=np.float32)
    candidate[:5, :, :] = 1.0
    coverage = mask_coverage(candidate, reference)
    assert abs(coverage - 0.5) < 0.01


def test_mask_coverage_empty_reference():
    """Empty reference should give 1.0."""
    reference = np.zeros((10, 10, 10), dtype=np.float32)
    candidate = np.zeros((10, 10, 10), dtype=np.float32)
    assert mask_coverage(candidate, reference) == 1.0


# ---------------------------------------------------------------------------
# Test evaluation with synthetic data
# ---------------------------------------------------------------------------


def _create_synthetic_nifti(
    shape: tuple[int, int, int] = (64, 64, 64),
    data_pattern: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic NIfTI data and affine.

    Returns (data, affine).
    """
    affine = np.eye(4)
    affine[:3, :3] *= 2.0
    if data_pattern == "uniform":
        data = np.ones(shape, dtype=np.float32)
    elif data_pattern == "zeros":
        data = np.zeros(shape, dtype=np.float32)
    else:
        data = np.random.rand(*shape).astype(np.float32)
    return data, affine


def _write_nifti(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """Write data to a NIfTI file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))


@pytest.fixture
def temp_dataset():
    """Create a temporary synthetic dataset with FCD subjects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create participants.tsv
        participants_file = tmpdir_path / "participants.tsv"
        with participants_file.open("w") as f:
            f.write("participant_id\tgroup\n")
            f.write("sub-00001\tfcd\n")
            f.write("sub-00002\tfcd\n")

        # Create subject 1: with FLAIR and ROI
        anat_dir_1 = tmpdir_path / "sub-00001" / "anat"
        anat_dir_1.mkdir(parents=True)

        data, affine = _create_synthetic_nifti((64, 64, 64), "uniform")
        _write_nifti(data, affine, anat_dir_1 / "sub-00001_FLAIR.nii.gz")

        roi_data = np.zeros((64, 64, 64), dtype=np.float32)
        roi_data[20:30, 20:30, 20:30] = 1.0
        _write_nifti(roi_data, affine, anat_dir_1 / "sub-00001_FLAIR_roi.nii.gz")

        # Create subject 2: with FLAIR and ROI
        anat_dir_2 = tmpdir_path / "sub-00002" / "anat"
        anat_dir_2.mkdir(parents=True)

        data2, affine2 = _create_synthetic_nifti((64, 64, 64), "uniform")
        _write_nifti(data2, affine2, anat_dir_2 / "sub-00002_FLAIR.nii.gz")

        roi_data2 = np.zeros((64, 64, 64), dtype=np.float32)
        roi_data2[30:40, 30:40, 30:40] = 1.0
        _write_nifti(roi_data2, affine2, anat_dir_2 / "sub-00002_FLAIR_roi.nii.gz")

        yield tmpdir_path


def test_evaluate_candidate_recall_basic(temp_dataset, monkeypatch):
    """Test basic evaluation pipeline."""
    with tempfile.TemporaryDirectory() as output_dir:
        output_path = Path(output_dir)

        # Mock generate_candidate_rois to return predictable result
        def mock_generate_candidate_rois(volume_path, output_dir, **kwargs):
            # Create a union mask that partially covers the ROI
            candidate_mask = np.zeros((64, 64, 64), dtype=np.float32)
            candidate_mask[18:32, 18:32, 18:32] = 1.0
            affine = np.eye(4)
            affine[:3, :3] *= 2.0
            _write_nifti(candidate_mask, affine, Path(output_dir) / "candidate_roi_union.nii.gz")
            return {"selected_count": 2, "volume_path": str(volume_path)}

        monkeypatch.setattr(
            "fcd_prototype.candidate_roi_evaluation.generate_candidate_rois",
            mock_generate_candidate_rois,
        )

        summary = evaluate_candidate_recall(
            dataset_dir=temp_dataset,
            output_dir=output_path,
            max_candidates=5,
            radius_mm=8.0,
            min_separation_mm=16.0,
            percentile=99.5,
        )

        assert summary["config_id"] == "mc5_r8.0_ms16.0_p99.5"
        assert summary["n_subjects"] == 2
        assert len(summary["subject_metrics"]) == 2

        # Check JSON was written
        summary_path = output_path / "mc5_r8.0_ms16.0_p99.5" / "summary.json"
        assert summary_path.exists()
        with summary_path.open() as f:
            loaded_summary = json.load(f)
            assert loaded_summary["n_subjects"] == 2

        # Check CSV was written
        csv_path = output_path / "mc5_r8.0_ms16.0_p99.5" / "subjects.csv"
        assert csv_path.exists()


def test_evaluate_candidate_recall_metrics(temp_dataset, monkeypatch):
    """Test that metrics are computed correctly."""
    with tempfile.TemporaryDirectory() as output_dir:
        output_path = Path(output_dir)

        # Mock to return candidates that perfectly overlap ROI
        def mock_generate_candidate_rois(volume_path, output_dir, **kwargs):
            candidate_mask = np.zeros((64, 64, 64), dtype=np.float32)
            candidate_mask[20:30, 20:30, 20:30] = 1.0
            affine = np.eye(4)
            affine[:3, :3] *= 2.0
            _write_nifti(candidate_mask, affine, Path(output_dir) / "candidate_roi_union.nii.gz")
            return {"selected_count": 1}

        monkeypatch.setattr(
            "fcd_prototype.candidate_roi_evaluation.generate_candidate_rois",
            mock_generate_candidate_rois,
        )

        summary = evaluate_candidate_recall(
            dataset_dir=temp_dataset,
            output_dir=output_path,
            subjects=["sub-00001"],
            max_candidates=5,
            radius_mm=8.0,
            min_separation_mm=16.0,
            percentile=99.5,
        )

        assert summary["n_subjects"] == 1
        metric = summary["subject_metrics"][0]
        assert metric["subject_id"] == "sub-00001"
        assert metric["dice"] == 1.0
        assert metric["manual_coverage"] == 1.0
        assert metric["candidate_precision"] == 1.0


def test_sweep_candidate_recall(temp_dataset, monkeypatch):
    """Test parameter sweep."""
    with tempfile.TemporaryDirectory() as output_dir:
        output_path = Path(output_dir)

        def mock_generate_candidate_rois(volume_path, output_dir, **kwargs):
            candidate_mask = np.zeros((64, 64, 64), dtype=np.float32)
            candidate_mask[20:30, 20:30, 20:30] = 1.0
            affine = np.eye(4)
            affine[:3, :3] *= 2.0
            _write_nifti(candidate_mask, affine, Path(output_dir) / "candidate_roi_union.nii.gz")
            return {"selected_count": 1}

        monkeypatch.setattr(
            "fcd_prototype.candidate_roi_evaluation.generate_candidate_rois",
            mock_generate_candidate_rois,
        )

        sweep_result = sweep_candidate_recall(
            dataset_dir=temp_dataset,
            output_dir=output_path,
            max_candidates_values=(3, 5),
            radius_mm_values=(8.0,),
            percentile_values=(99.5,),
            min_separation_mm=16.0,
        )

        assert "configs" in sweep_result
        assert len(sweep_result["configs"]) == 2
        assert sweep_result["best_by_mean_coverage"] is not None
        assert sweep_result["best_by_fraction_ge_0_25"] is not None


def test_evaluate_candidate_recall_no_subjects(temp_dataset, monkeypatch):
    """Test with no matching subjects."""
    with tempfile.TemporaryDirectory() as output_dir:
        output_path = Path(output_dir)

        summary = evaluate_candidate_recall(
            dataset_dir=temp_dataset,
            output_dir=output_path,
            subjects=["sub-nonexistent"],
            max_candidates=5,
            radius_mm=8.0,
            min_separation_mm=16.0,
            percentile=99.5,
        )

        assert summary["n_subjects"] == 0
        assert summary["mean_dice"] is None
        assert summary["recommendation"] == "No subjects with FCD + FLAIR ROI found."


def test_resample_different_affines_same_world_lesion(temp_dataset, monkeypatch):
    """Regression: resampling should work when candidate and manual have different affines but same world-space lesion."""
    with tempfile.TemporaryDirectory() as output_dir:
        output_path = Path(output_dir)

        def mock_generate_candidate_rois(volume_path, output_dir, **kwargs):
            candidate_mask = np.zeros((32, 32, 32), dtype=np.float32)
            candidate_mask[10:20, 10:20, 10:20] = 1.0
            affine_candidate = np.eye(4)
            affine_candidate[:3, :3] *= 4.0
            _write_nifti(candidate_mask, affine_candidate, Path(output_dir) / "candidate_roi_union.nii.gz")
            return {"selected_count": 1}

        monkeypatch.setattr(
            "fcd_prototype.candidate_roi_evaluation.generate_candidate_rois",
            mock_generate_candidate_rois,
        )

        summary = evaluate_candidate_recall(
            dataset_dir=temp_dataset,
            output_dir=output_path,
            subjects=["sub-00001"],
            max_candidates=5,
            radius_mm=8.0,
            min_separation_mm=16.0,
            percentile=99.5,
        )

        assert summary["n_subjects"] == 1
        metric = summary["subject_metrics"][0]
        assert metric["subject_id"] == "sub-00001"
        assert metric["dice"] > 0.0
