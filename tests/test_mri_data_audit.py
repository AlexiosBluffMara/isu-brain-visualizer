"""Tests for MRI/NIfTI data-quality audit helpers."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from fcd_prototype.mri_data_audit import (
    audit_dataset,
    build_subject_data_quality,
    compare_canonical_geometry,
    inspect_nifti,
    record_roi_processing,
    roi_stats,
)


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def test_inspect_nifti_reports_orientation_spacing_and_nonfinite(tmp_path):
    data = np.zeros((4, 5, 6), dtype=np.float32)
    data[0, 0, 0] = np.nan
    data[1, 1, 1] = np.inf
    affine = np.diag([-1.0, 1.0, 3.5, 1.0])
    path = _write_nifti(tmp_path / "vol.nii.gz", data, affine)

    stats = inspect_nifti(path)

    assert stats["shape"] == [4, 5, 6]
    assert stats["voxel_spacing_mm"] == [1.0, 1.0, 3.5]
    assert stats["finite"]["nan_count"] == 1
    assert stats["finite"]["inf_count"] == 1
    assert "thick_slice_spacing_gt_3mm" in stats["warnings"]
    assert "nonfinite_voxels_present" in stats["warnings"]
    assert "orientation_reordered_to_canonical" in stats["warnings"]


def test_inspect_nifti_flags_qform_sform_mismatch_and_shear(tmp_path):
    data = np.ones((4, 4, 4), dtype=np.float32)
    qform = np.eye(4)
    sform = np.array([
        [1.0, 0.2, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    img = nib.Nifti1Image(data, sform)
    img.set_qform(qform, code=1)
    img.set_sform(sform, code=1)
    path = tmp_path / "mismatch.nii.gz"
    nib.save(img, str(path))

    stats = inspect_nifti(path)

    assert stats["qform_sform_max_abs_diff"] > 0.0
    assert stats["affine_orthogonality_error"] > 0.0
    assert "qform_sform_affine_mismatch" in stats["warnings"]
    assert "affine_axes_not_orthogonal_or_sheared" in stats["warnings"]


def test_compare_canonical_geometry_flags_resampling_need(tmp_path):
    ref = _write_nifti(tmp_path / "flair.nii.gz", np.ones((8, 8, 8), dtype=np.float32), np.eye(4))
    moving = _write_nifti(
        tmp_path / "t1w.nii.gz",
        np.ones((10, 10, 10), dtype=np.float32),
        np.diag([0.8, 0.8, 0.8, 1.0]),
    )

    comparison = compare_canonical_geometry(ref, moving, "flair", "t1w")

    assert comparison["shape_match"] is False
    assert comparison["needs_resample_to_reference"] is True
    assert comparison["spacing_max_abs_diff"] > 0.0


def test_roi_stats_flags_empty_tiny_and_huge_masks(tmp_path):
    empty = _write_nifti(tmp_path / "empty_roi.nii.gz", np.zeros((5, 5, 5), dtype=np.uint8), np.eye(4))
    tiny_data = np.zeros((5, 5, 5), dtype=np.uint8)
    tiny_data[1, 1, 1] = 1
    tiny = _write_nifti(tmp_path / "tiny_roi.nii.gz", tiny_data, np.eye(4))
    huge_data = np.ones((5, 5, 5), dtype=np.uint8)
    huge = _write_nifti(tmp_path / "huge_roi.nii.gz", huge_data, np.eye(4))

    assert "roi_empty" in roi_stats(empty)["warnings"]
    assert "roi_tiny_lt_5_voxels" in roi_stats(tiny)["warnings"]
    assert "roi_fraction_gt_10_percent" in roi_stats(huge)["warnings"]


def test_build_subject_data_quality_collects_modality_and_roi_warnings(tmp_path):
    flair_data = np.linspace(0.0, 1.0, 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
    t2w_data = np.linspace(0.0, 1.0, 9 * 9 * 9, dtype=np.float32).reshape(9, 9, 9)
    t1w_data = np.linspace(0.0, 1.0, 10 * 10 * 10, dtype=np.float32).reshape(10, 10, 10)
    flair = _write_nifti(tmp_path / "flair.nii.gz", flair_data, np.eye(4))
    t2w = _write_nifti(
        tmp_path / "t2w.nii.gz",
        t2w_data,
        np.diag([0.9, 0.9, 0.9, 1.0]),
    )
    t1w = _write_nifti(
        tmp_path / "t1w.nii.gz",
        t1w_data,
        np.diag([0.8, 0.8, 0.8, 1.0]),
    )
    roi_data = np.zeros((8, 8, 8), dtype=np.uint8)
    roi_data[2, 2, 2] = 1
    roi = _write_nifti(tmp_path / "roi.nii.gz", roi_data, np.eye(4))

    audit = build_subject_data_quality(flair_path=flair, t2w_path=t2w, t1w_path=t1w, roi_path=roi)

    assert audit["status"] == "warning"
    assert audit["warning_count"] > 0
    assert audit["comparisons"]["t2w_to_flair"]["needs_resample_to_reference"] is True
    assert audit["comparisons"]["t1w_to_flair"]["needs_resample_to_reference"] is True
    assert "t2w_to_flair:geometry_mismatch_requires_resampling" in audit["warnings"]
    assert "t1w_to_flair:geometry_mismatch_requires_resampling" in audit["warnings"]
    assert "flair_roi:roi_tiny_lt_5_voxels" in audit["warnings"]
    readiness = audit["upload_readiness"]
    assert readiness["status"] == "review"
    assert readiness["can_run_3dgs"] is True
    assert "t2w_to_flair:geometry_mismatch_requires_resampling" in readiness["review_items"]
    assert "t1w_to_flair:geometry_mismatch_requires_resampling" in readiness["review_items"]


def test_build_subject_data_quality_accepts_t2w_primary_without_flair(tmp_path):
    data = np.linspace(0.0, 1.0, 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
    t2w = _write_nifti(tmp_path / "patient_T2w.nii.gz", data, np.eye(4))

    audit = build_subject_data_quality(flair_path=None, t2w_path=t2w, t1w_path=None, roi_path=None)

    readiness = audit["upload_readiness"]
    assert readiness["status"] in {"pass", "review"}
    assert readiness["primary_modalities"] == ["t2w"]
    assert readiness["can_run_3dgs"] is True


def test_upload_readiness_blocks_nonfinite_primary_mri(tmp_path):
    data = np.ones((8, 8, 8), dtype=np.float32)
    data[0, 0, 0] = np.nan
    flair = _write_nifti(tmp_path / "flair.nii.gz", data, np.eye(4))

    audit = build_subject_data_quality(flair_path=flair, t1w_path=None, roi_path=None)

    readiness = audit["upload_readiness"]
    assert readiness["status"] == "fail"
    assert readiness["can_run_3dgs"] is False
    assert readiness["can_subtract_patient_normal"] is False
    assert "flair:nonfinite_voxels_present" in readiness["blockers"]
    assert readiness["interpretation"] == "not_safe_for_paper_grade_3dgs_subtraction"


def test_upload_readiness_blocks_empty_provided_roi(tmp_path):
    flair = _write_nifti(tmp_path / "flair.nii.gz", np.ones((8, 8, 8), dtype=np.float32), np.eye(4))
    roi = _write_nifti(tmp_path / "roi.nii.gz", np.zeros((8, 8, 8), dtype=np.uint8), np.eye(4))

    audit = build_subject_data_quality(flair_path=flair, t1w_path=None, roi_path=roi)

    readiness = audit["upload_readiness"]
    assert readiness["status"] == "fail"
    assert "flair_roi:roi_empty" in readiness["blockers"]
    assert readiness["can_use_provided_roi_for_fuzzing"] is False


def test_build_subject_data_quality_reports_explicit_missing_path(tmp_path):
    missing = tmp_path / "missing_flair.nii.gz"

    audit = build_subject_data_quality(flair_path=missing, t1w_path=None, roi_path=None)

    assert audit["status"] == "warning"
    assert audit["volumes"]["flair"]["exists"] is False
    assert "flair:missing_file" in audit["warnings"]


def test_record_roi_processing_records_actual_resample_outcome():
    audit = {"warnings": [], "warning_count": 0, "status": "ok"}
    record_roi_processing(
        audit,
        source="provided",
        source_shape=(5, 5, 5),
        source_affine=np.diag([2.0, 2.0, 2.0, 1.0]),
        target_shape=(10, 10, 10),
        target_affine=np.eye(4),
        resampled=True,
    )

    assert audit["roi_processing"]["resampled_to_base"] is True
    assert audit["roi_processing"]["target_shape"] == [10, 10, 10]
    assert audit["status"] == "warning"
    assert "provided:roi_resampled_to_base_grid" in audit["warnings"]


def test_audit_dataset_summarizes_geometry_risks_and_writes_outputs(tmp_path):
    ds = tmp_path / "ds"
    (ds / "sub-00001" / "anat").mkdir(parents=True)
    (ds / "sub-00002" / "anat").mkdir(parents=True)
    (ds / "participants.tsv").write_text(
        "participant_id\tgroup\nsub-00001\tfcd\nsub-00002\thc\n",
        encoding="utf-8",
    )
    _write_nifti(
        ds / "sub-00001" / "anat" / "sub-00001_FLAIR.nii.gz",
        np.ones((8, 8, 8), dtype=np.float32),
        np.eye(4),
    )
    _write_nifti(
        ds / "sub-00001" / "anat" / "sub-00001_T1w.nii.gz",
        np.ones((10, 10, 10), dtype=np.float32),
        np.diag([0.8, 0.8, 0.8, 1.0]),
    )
    _write_nifti(
        ds / "sub-00002" / "anat" / "sub-00002_T2w.nii.gz",
        np.ones((8, 8, 8), dtype=np.float32),
        np.eye(4),
    )
    roi = np.zeros((8, 8, 8), dtype=np.uint8)
    roi[2:4, 2:4, 2:4] = 1
    _write_nifti(ds / "sub-00001" / "anat" / "sub-00001_FLAIR_roi.nii.gz", roi, np.eye(4))
    _write_nifti(
        ds / "sub-00002" / "anat" / "sub-00002_FLAIR.nii.gz",
        np.ones((8, 8, 8), dtype=np.float32),
        np.eye(4),
    )
    output_json = tmp_path / "audit.json"
    output_csv = tmp_path / "audit.csv"

    result = audit_dataset(ds, output_json=output_json, output_csv=output_csv)

    summary = result["summary"]
    assert summary["n_total"] == 2
    assert summary["group_counts"]["fcd"] == 1
    assert summary["group_counts"]["hc"] == 1
    assert summary["modality_counts"]["flair"] == 2
    assert summary["modality_counts"]["t2w"] == 1
    assert summary["modality_counts"]["flair_roi"] == 1
    assert summary["geometry_mismatch_counts"]["t1w_to_flair"] == 1
    assert "t1w_to_flair:geometry_mismatch_requires_resampling" in summary["warning_counts"]
    assert output_json.exists()
    assert output_csv.exists()
    csv_header = output_csv.read_text(encoding="utf-8").splitlines()[0]
    assert "t2w_spacing_mm" in csv_header
    assert "t1w_to_flair_needs_resample" in csv_header
