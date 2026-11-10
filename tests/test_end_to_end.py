"""Tests for end_to_end pipeline - dataset-wide processing."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from unittest.mock import patch

from fcd_prototype.end_to_end import (
    build_end_to_end,
    build_paper_method_feasibility,
    build_product_readiness,
    main,
    process_subject,
    _roi_localization_evidence_gate,
    _nvidia_stages_ok,
)
from fcd_prototype.bids_discovery import discover_subject


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_volume(path: Path, shape=(20, 20, 16), affine=None, seed=0) -> None:
    if affine is None:
        affine = np.diag([1.0, 1.0, 1.5, 1.0])
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal(shape).astype(np.float32) * 0.2 + 0.6).clip(0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _make_roi(path: Path, shape=(20, 20, 16)) -> None:
    data = np.zeros(shape, dtype=np.uint8)
    data[6:10, 6:10, 6:10] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.5, 1.0])), str(path))


def _fake_dataset(tmp_path: Path, subjects: list[dict]) -> Path:
    """subjects = [{"id": ..., "group": ..., "has_roi": bool}, ...]"""
    ds = tmp_path / "ds"
    tsv_lines = ["participant_id\tgroup"]
    for s in subjects:
        sub = s["id"]
        anat = ds / sub / "anat"
        anat.mkdir(parents=True, exist_ok=True)
        _make_volume(anat / f"{sub}_FLAIR.nii.gz")
        _make_volume(anat / f"{sub}_T1w.nii.gz")
        if s.get("has_roi"):
            _make_roi(anat / f"{sub}_FLAIR_roi.nii.gz")
        tsv_lines.append(f"{sub}\t{s['group']}")
    (ds / "participants.tsv").write_text("\n".join(tsv_lines))
    return ds


# ---------------------------------------------------------------------------
# Per-subject tests
# ---------------------------------------------------------------------------

def test_process_fcd_subject_manifest(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(rec, tmp_path / "out", force=True)
    assert manifest["subject"] == "sub-00001"
    assert manifest["group"] == "fcd"
    assert manifest["roi_available"] is True


def test_process_fcd_subject_artifacts_exist(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"
    manifest = process_subject(rec, out, force=True)
    for key, fname in manifest["artifacts"].items():
        if not fname.startswith("0."):  # skip roi_concentration float string
            p = out / "sub-00001" / fname
            assert p.exists(), f"Missing artifact {key}: {fname}"


def test_process_fcd_subject_has_splat_ply(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(rec, tmp_path / "out", force=True)
    assert "patient_splat_ply" in manifest["artifacts"]


def test_process_fcd_subject_dense_splat_defaults_to_volume_mode(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"
    manifest = process_subject(rec, out, force=True)
    assert manifest["artifacts"]["patient_splat_dense_ply"] == "patient_splat_dense.ply"
    dense_meta = json.loads((out / "sub-00001" / "patient_splat_dense_meta.json").read_text())
    assert dense_meta["sampling_mode"] == "volume"
    assert dense_meta["effective_percentile_threshold"] >= 20.0
    assert dense_meta["foreground_mask_model"] == "largest_connected_tissue_component"
    assert dense_meta["gaussian_scale_model"] == "coverage_balanced_density_adaptive_volume_kernel"
    assert dense_meta["density_scale_factor"] >= 1.35
    assert dense_meta["density_scale_factor"] <= 2.25


def test_process_fcd_subject_has_splat_reconstruction_audit(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(rec, tmp_path / "out", force=True)
    assert manifest["stage_status"]["splat_reconstruction_audit"] == "ok"
    assert "splat_reconstruction_audit" in manifest["artifacts"]
    assert "splat_reconstruction" in manifest["artifacts"]
    meta = manifest["splat_reconstruction_audit_meta"]
    assert meta["foreground_coverage"] is not None
    assert meta["audited_points"] > 0
    assert meta["kernel_mode"] == "gaussian"
    assert meta["gaussian_sigma_extent"] == pytest.approx(2.5)
    assert meta["mri_to_3dgs_readiness_status"] in {"pass", "review", "fail"}
    assert isinstance(meta["mri_to_3dgs_claim_allowed"], bool)
    assert "mri_to_3dgs_claim_class" in meta
    assert "representation_readiness_gate" in meta


def test_process_fcd_subject_has_residual(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(rec, tmp_path / "out", force=True)
    assert "residual_heatmap" in manifest["artifacts"]


def test_process_fcd_subject_has_paired_gaussian_residual(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(rec, tmp_path / "out", force=True)
    assert manifest["stage_status"]["paired_gaussian_residual"] == "ok"
    assert manifest["paired_gaussian_residual_meta"]["same_gaussian_support"] is True
    assert manifest["paired_gaussian_residual_meta"]["sampling_mode"] == "volume"
    assert manifest["paired_gaussian_residual_meta"]["support_basis_mode"] == "union"
    assert manifest["paired_gaussian_residual_meta"]["support_basis"] == (
        "max(patient, synthetic_normal)"
    )
    assert manifest["paired_gaussian_residual_meta"]["heatmap_method"] == "gaussian"
    assert "heatmap_backend" in manifest["paired_gaussian_residual_meta"]
    assert "heatmap_active_gaussians" in manifest["paired_gaussian_residual_meta"]
    assert manifest["paired_gaussian_residual_meta"]["gaussian_orientation"] == "local_gradient_tangent_frame"
    assert manifest["paired_gaussian_residual_meta"]["gaussian_scale_model"] == (
        "coverage_balanced_density_adaptive_volume_kernel"
    )
    assert manifest["paired_gaussian_residual_meta"]["foreground_mask_model"] == (
        "largest_connected_tissue_component"
    )
    assert manifest["paired_gaussian_residual_meta"]["density_scale_factor"] >= 1.35
    assert manifest["paired_gaussian_residual_meta"]["density_scale_factor"] <= 2.25
    assert manifest["paired_gaussian_residual_meta"]["density_voxels_per_splat"] >= 1.0
    for key in (
        "paired_patient_splat",
        "paired_normal_splat",
        "paired_gaussian_residual_splat",
        "paired_gaussian_residual_heatmap",
        "paired_gaussian_residual_metrics",
    ):
        assert key in manifest["artifacts"]


def test_process_subject_can_use_patient_paired_gaussian_support(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(
        rec,
        tmp_path / "out",
        force=True,
        max_splat_points=50,
        presentation_splat_points=50,
        paired_gaussian_support_basis="patient",
    )

    assert manifest["stage_status"]["paired_gaussian_residual"] == "ok"
    assert manifest["paired_gaussian_residual_meta"]["support_basis_mode"] == "patient"
    assert manifest["paired_gaussian_residual_meta"]["support_basis"] == "patient"


def test_process_subject_control_gaussian_prior_residual_outputs(tmp_path):
    ds = _fake_dataset(tmp_path, [
        {"id": "sub-00001", "group": "fcd", "has_roi": True},
        {"id": "sub-00002", "group": "hc", "has_roi": False},
        {"id": "sub-00003", "group": "hc", "has_roi": False},
    ])
    control_1 = ds / "sub-00002" / "anat" / "sub-00002_FLAIR.nii.gz"
    control_2 = ds / "sub-00003" / "anat" / "sub-00003_FLAIR.nii.gz"
    _make_volume(control_1, seed=2)
    _make_volume(control_2, seed=3)

    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"
    manifest = process_subject(
        rec,
        out,
        force=True,
        max_splat_points=60,
        presentation_splat_points=60,
        compute_control_gaussian_prior_residual=True,
        control_prior_volumes=[control_1, control_2],
        control_gaussian_prior_max_points=60,
        control_gaussian_prior_heatmap_method="scatter",
    )

    assert manifest["stage_status"]["control_gaussian_prior"] == "ok"
    meta = manifest["control_gaussian_prior_meta"]
    assert meta["method"] == "patient_support_control_prior_zscore"
    assert meta["gaussian_state_schema"] == "patient_control_gaussian_prior_v1"
    assert meta["same_gaussian_support"] is True
    assert meta["n_controls"] == 2
    assert meta["control_prior_subjects"] == ["sub-00002", "sub-00003"]
    assert meta["roi_metrics_independent_validation"] is False
    assert meta["roi_metrics_independence_reason"] == "manual_roi_reused_as_fuzzing_roi"
    assert manifest["residual_candidate_meta"]["source_heatmap"] == (
        "control_gaussian_prior/control_prior_z_residual_heatmap.nii.gz"
    )

    for key in (
        "control_prior_patient_splat",
        "control_prior_mean_splat",
        "control_prior_z_residual_splat",
        "control_prior_z_residual_heatmap",
        "control_prior_z_scores",
        "control_gaussian_prior_state",
        "control_gaussian_prior_metrics",
    ):
        rel = manifest["artifacts"][key]
        assert (out / "sub-00001" / rel).exists(), f"missing {key}: {rel}"


def test_process_fcd_subject_has_residual_candidates(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(rec, tmp_path / "out", force=True)
    assert manifest["stage_status"]["residual_candidates"].startswith("ok")
    assert "residual_candidate_mask" in manifest["artifacts"]
    assert "residual_candidate_manifest" in manifest["artifacts"]
    assert manifest["residual_candidate_meta"]["source_heatmap"] == (
        "paired_gaussian_residual/paired_gaussian_residual_heatmap.nii.gz"
    )
    assert manifest["residual_candidate_meta"]["selected_roi_source"] == "provided"
    assert manifest["residual_candidate_meta"]["manual_roi_source"] == "provided"
    assert manifest["residual_candidate_meta"]["manual_roi_available"] is True
    assert (
        manifest["residual_candidate_meta"][
            "manual_overlap_independent_from_fuzzing_roi"
        ]
        is False
    )
    assert (
        manifest["residual_candidate_meta"]["manual_overlap_independence_reason"]
        == "manual_roi_reused_as_fuzzing_roi"
    )


def test_process_fcd_subject_has_synthetic_normal_audit(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(rec, tmp_path / "out", force=True)
    assert manifest["stage_status"]["synthetic_normal_audit"] == "ok"
    assert manifest["artifacts"]["synthetic_normal_audit"] == "synthetic_normal_audit.json"
    audit_meta = manifest["synthetic_normal_audit_meta"]
    assert audit_meta["patient_conditioned_inpainting"] is False
    assert audit_meta["claim_boundary"] == "not_patient_conditioned_inpainting"
    assert audit_meta["subtraction_interpretation"] == "roi_hybrid_same_patient_grid_residual"


def test_nvidia_mode_writes_preflight_when_dependency_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_GENERATE_CTMR_ROOT", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_ROOT", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_WEIGHTS", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_DIM", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_SPACING", raising=False)

    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"

    manifest = process_subject(rec, out, synthetic_mode="nvidia", force=True)

    assert manifest["stage_status"]["synthetic_normal"] == "failed: NVIDIA not available"
    assert manifest["stage_status"]["nvidia_preflight"].startswith("failed:")
    assert manifest["artifacts"]["nvidia_preflight"] == "nvidia_preflight.json"
    preflight_path = out / "sub-00001" / "nvidia_preflight.json"
    preflight = json.loads(preflight_path.read_text())
    assert preflight["ready_for_inference"] is False
    assert preflight["patient_conditioned_inpainting_supported"] is False
    assert preflight["ready_for_patient_conditioned_counterfactual"] is False
    assert manifest["nvidia_preflight_meta"]["ready_for_inference"] is False
    assert (
        manifest["nvidia_preflight_meta"]["ready_for_patient_conditioned_counterfactual"]
        is False
    )
    claim_gate = manifest["nvidia_preflight_meta"]["paper_claim_readiness"]
    assert claim_gate["nvidia_inference_claim_allowed"] is False
    assert claim_gate["patient_specific_healthy_counterfactual_claim_allowed"] is False
    adapter = manifest["nvidia_preflight_meta"]["local_adapter_contract"]
    assert adapter["passes_patient_mri_to_nvidia_model"] is False
    assert adapter["passes_roi_mask_to_nvidia_model"] is False
    assert manifest["nvidia_preflight_meta"]["blocker_count"] > 0


def test_process_control_subject_no_roi(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00002", "group": "hc", "has_roi": False}])
    rec = discover_subject(ds, "sub-00002", group="hc")
    manifest = process_subject(rec, tmp_path / "out", force=True)
    assert manifest["roi_available"] is False
    assert "lesion_roi" not in manifest["artifacts"]


def test_process_control_has_residual_stage(tmp_path):
    """Controls should still generate residual (should be near-zero)."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00002", "group": "hc", "has_roi": False}])
    rec = discover_subject(ds, "sub-00002", group="hc")
    manifest = process_subject(rec, tmp_path / "out", force=True)
    assert "residual_heatmap" in manifest["artifacts"]


def test_manifest_has_stage_status(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(rec, tmp_path / "out", force=True)
    assert "stage_status" in manifest
    for key in (
        "data_quality", "patient_volumes", "splat_ply", "synthetic_normal",
        "splat_reconstruction_audit", "synthetic_normal_audit", "residual",
        "residual_candidates",
    ):
        assert key in manifest["stage_status"]


def test_manifest_has_data_quality_audit(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(rec, tmp_path / "out", force=True)

    assert "data_quality" in manifest
    assert "volumes" in manifest["data_quality"]
    assert "flair" in manifest["data_quality"]["volumes"]
    assert manifest["data_quality"]["volumes"]["flair"]["canonical_orientation"] == ["R", "A", "S"]
    assert manifest["data_quality"]["upload_readiness"]["status"] in {"pass", "review"}
    assert manifest["data_quality"]["upload_readiness"]["can_run_3dgs"] is True


def test_data_quality_flags_t1w_to_flair_geometry_mismatch(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    anat = ds / "sub-00001" / "anat"
    _make_volume(
        anat / "sub-00001_T1w.nii.gz",
        shape=(24, 24, 20),
        affine=np.diag([0.8, 0.8, 0.8, 1.0]),
        seed=11,
    )
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(rec, tmp_path / "out", force=True)

    assert manifest["data_quality"]["comparisons"]["t1w_to_flair"]["needs_resample_to_reference"] is True
    assert "t1w_to_flair:geometry_mismatch_requires_resampling" in manifest["data_quality"]["warnings"]
    assert manifest["stage_status"]["data_quality"].startswith("warning")
    assert manifest["data_quality"]["upload_readiness"]["status"] == "review"


def test_data_quality_stage_fails_for_nonfinite_primary_mri(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": False}])
    anat = ds / "sub-00001" / "anat"
    data = np.ones((20, 20, 16), dtype=np.float32)
    data[0, 0, 0] = np.nan
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.5, 1.0])), str(anat / "sub-00001_FLAIR.nii.gz"))
    rec = discover_subject(ds, "sub-00001", group="fcd")

    manifest = process_subject(
        rec,
        tmp_path / "out",
        force=True,
        synthetic_mode="placeholder",
        roi_strategy="none",
    )

    assert manifest["stage_status"]["data_quality"].startswith("failed:")
    assert manifest["data_quality"]["upload_readiness"]["status"] == "fail"
    assert "flair:nonfinite_voxels_present" in manifest["data_quality"]["upload_readiness"]["blockers"]


def test_data_quality_records_provided_roi_resampling(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    anat = ds / "sub-00001" / "anat"
    roi_data = np.zeros((10, 10, 8), dtype=np.uint8)
    roi_data[3:5, 3:5, 3:5] = 1
    nib.save(
        nib.Nifti1Image(roi_data, np.diag([2.0, 2.0, 3.0, 1.0])),
        str(anat / "sub-00001_FLAIR_roi.nii.gz"),
    )
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(rec, tmp_path / "out", force=True)

    roi_processing = manifest["data_quality"]["roi_processing"]
    assert roi_processing["source"] == "provided"
    assert roi_processing["resampled_to_base"] is True
    assert roi_processing["target_shape"] == [20, 20, 16]
    assert "provided:roi_resampled_to_base_grid" in manifest["data_quality"]["warnings"]


def test_manifest_synthetic_normal_source_placeholder(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(rec, tmp_path / "out", force=True, synthetic_mode="placeholder")
    assert manifest["synthetic_normal_source"] == "placeholder"
    assert manifest["synthetic_normal_is_diagnostic"] is False


def test_process_control_prior_records_meta(tmp_path):
    ds = _fake_dataset(tmp_path, [
        {"id": "sub-00001", "group": "fcd", "has_roi": True},
        {"id": "sub-00002", "group": "hc", "has_roi": False},
    ])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    control = ds / "sub-00002" / "anat" / "sub-00002_FLAIR.nii.gz"

    manifest = process_subject(
        rec,
        tmp_path / "out",
        force=True,
        synthetic_mode="control-prior",
        control_prior_volumes=[control],
    )

    assert manifest["synthetic_normal_source"] == "control_prior_roi_hybrid"
    assert manifest["control_prior_meta"]["control_prior_count"] == 1
    assert manifest["control_prior_meta"]["control_prior_subjects"] == ["sub-00002"]
    assert "control_prior_quality_gate" in manifest["control_prior_meta"]
    assert "control_prior_reference_similarity_1_minus_mae" in manifest["control_prior_meta"]
    assert "synthetic_normal" in manifest["artifacts"]


def test_manifest_has_notes_and_limitations(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(rec, tmp_path / "out", force=True)
    assert isinstance(manifest["notes"], list)
    assert isinstance(manifest["limitations"], list)


def test_skip_without_force(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"
    process_subject(rec, out, force=True)
    mtime1 = (out / "sub-00001" / "manifest.json").stat().st_mtime
    process_subject(rec, out, force=False)
    mtime2 = (out / "sub-00001" / "manifest.json").stat().st_mtime
    assert mtime1 == mtime2


# ---------------------------------------------------------------------------
# Dataset-wide index tests
# ---------------------------------------------------------------------------

def test_build_end_to_end_index_created(tmp_path):
    ds = _fake_dataset(tmp_path, [
        {"id": "sub-00001", "group": "fcd", "has_roi": True},
        {"id": "sub-00002", "group": "hc", "has_roi": False},
    ])
    out = tmp_path / "out"
    index_path = build_end_to_end(ds, out, force=True)
    assert index_path.exists()
    with index_path.open() as fh:
        index = json.load(fh)
    assert len(index["subjects"]) == 2


def test_build_end_to_end_subject_filter(tmp_path):
    ds = _fake_dataset(tmp_path, [
        {"id": "sub-00001", "group": "fcd", "has_roi": True},
        {"id": "sub-00002", "group": "hc", "has_roi": False},
    ])
    out = tmp_path / "out"
    index_path = build_end_to_end(ds, out, subjects=["sub-00001"], force=True)
    with index_path.open() as fh:
        index = json.load(fh)
    assert len(index["subjects"]) == 1
    assert index["subjects"][0]["subject"] == "sub-00001"


def test_build_control_prior_subject_filter_still_uses_dataset_controls(tmp_path):
    ds = _fake_dataset(tmp_path, [
        {"id": "sub-00001", "group": "fcd", "has_roi": True},
        {"id": "sub-00002", "group": "hc", "has_roi": False},
    ])
    out = tmp_path / "out"
    index_path = build_end_to_end(
        ds,
        out,
        subjects=["sub-00001"],
        synthetic_mode="control-prior",
        force=True,
    )
    with index_path.open() as fh:
        index = json.load(fh)
    assert len(index["subjects"]) == 1
    manifest_path = out / "sub-00001" / "manifest.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert manifest["synthetic_normal_source"] == "control_prior_roi_hybrid"
    assert manifest["control_prior_meta"]["control_prior_count"] == 1
    assert manifest["control_prior_meta"]["control_prior_subjects"] == ["sub-00002"]


def test_build_control_gaussian_prior_uses_dataset_controls_without_control_prior_mode(tmp_path):
    ds = _fake_dataset(tmp_path, [
        {"id": "sub-00001", "group": "fcd", "has_roi": True},
        {"id": "sub-00002", "group": "hc", "has_roi": False},
        {"id": "sub-00003", "group": "hc", "has_roi": False},
    ])
    _make_volume(ds / "sub-00002" / "anat" / "sub-00002_FLAIR.nii.gz", seed=2)
    _make_volume(ds / "sub-00003" / "anat" / "sub-00003_FLAIR.nii.gz", seed=3)
    out = tmp_path / "out"

    build_end_to_end(
        ds,
        out,
        subjects=["sub-00001"],
        max_splat_points=50,
        presentation_splat_points=50,
        compute_control_gaussian_prior_residual=True,
        control_gaussian_prior_max_points=50,
        control_gaussian_prior_heatmap_method="scatter",
        force=True,
    )

    manifest_path = out / "sub-00001" / "manifest.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert manifest["synthetic_normal_source"] == "placeholder"
    assert manifest["stage_status"]["control_gaussian_prior"] == "ok"
    assert manifest["control_gaussian_prior_meta"]["n_controls"] == 2
    assert manifest["residual_candidate_meta"]["source_heatmap"] == (
        "control_gaussian_prior/control_prior_z_residual_heatmap.nii.gz"
    )


def test_build_control_prior_can_cap_controls_for_pilot_runs(tmp_path):
    ds = _fake_dataset(tmp_path, [
        {"id": "sub-00001", "group": "fcd", "has_roi": True},
        {"id": "sub-00002", "group": "hc", "has_roi": False},
        {"id": "sub-00003", "group": "hc", "has_roi": False},
        {"id": "sub-00004", "group": "hc", "has_roi": False},
    ])
    out = tmp_path / "out"

    build_end_to_end(
        ds,
        out,
        subjects=["sub-00001"],
        synthetic_mode="control-prior",
        control_prior_max_controls=2,
        force=True,
    )

    manifest_path = out / "sub-00001" / "manifest.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    control_subjects = manifest["control_prior_meta"]["control_prior_subjects"]
    assert manifest["control_prior_meta"]["control_prior_count"] == 2
    assert len(control_subjects) == 2
    assert set(control_subjects) <= {"sub-00002", "sub-00003", "sub-00004"}
    assert "sub-00001" not in control_subjects


def test_build_control_prior_no_roi_uses_whole_brain_prior(tmp_path):
    ds = _fake_dataset(tmp_path, [
        {"id": "sub-00001", "group": "fcd", "has_roi": False},
        {"id": "sub-00002", "group": "hc", "has_roi": False},
    ])
    out = tmp_path / "out"

    build_end_to_end(
        ds,
        out,
        subjects=["sub-00001"],
        synthetic_mode="control-prior",
        roi_strategy="none",
        max_splat_points=50,
        presentation_splat_points=50,
        force=True,
    )

    manifest_path = out / "sub-00001" / "manifest.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert manifest["synthetic_normal_source"] == "control_prior_whole_brain_prior"
    assert manifest["control_prior_meta"]["control_prior_scope"] == "whole_brain_no_roi"
    assert manifest["control_prior_meta"]["control_prior_count"] == 1
    assert manifest["stage_status"]["paired_gaussian_residual"] == "ok"
    assert manifest["paired_gaussian_residual_meta"]["residual_mean"] > 0


def test_index_has_dataset_field(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    out = tmp_path / "out"
    index_path = build_end_to_end(ds, out, force=True)
    with index_path.open() as fh:
        index = json.load(fh)
    assert "dataset" in index
    assert "ds004199" in index["dataset"]


# ---------------------------------------------------------------------------
# CLI exit-code tests for --synthetic-mode nvidia
# ---------------------------------------------------------------------------

def test_nvidia_stages_ok_all_good():
    manifests = [
        {"stage_status": {"synthetic_normal": "ok", "residual": "ok"}},
        {"stage_status": {"synthetic_normal": "ok", "residual": "ok"}},
    ]
    assert _nvidia_stages_ok(manifests) is True


def test_nvidia_stages_ok_failed_synthetic():
    manifests = [
        {"stage_status": {"synthetic_normal": "failed: NVIDIA not available", "residual": "skipped"}},
    ]
    assert _nvidia_stages_ok(manifests) is False


def test_nvidia_stages_ok_failed_residual():
    manifests = [
        {"stage_status": {"synthetic_normal": "ok", "residual": "failed: some error"}},
    ]
    assert _nvidia_stages_ok(manifests) is False


def test_main_nvidia_exits_nonzero_on_stage_failure(tmp_path):
    """main() must sys.exit(1) when nvidia mode and a subject fails synthetic_normal."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    out = tmp_path / "out"

    # Patch generate_synthetic_normal to raise NvidiaDependencyError so stage fails.
    from fcd_prototype.synthetic_normal import NvidiaDependencyError
    with patch(
        "fcd_prototype.end_to_end.generate_synthetic_normal",
        side_effect=NvidiaDependencyError("no nvidia"),
    ):
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--dataset-dir", str(ds),
                "--output-dir", str(out),
                "--synthetic-mode", "nvidia",
                "--force",
            ])
    assert exc_info.value.code == 1


def test_main_placeholder_exits_zero(tmp_path):
    """main() must NOT sys.exit when placeholder mode regardless of nvidia stage."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    out = tmp_path / "out"
    # Should not raise SystemExit
    main([
        "--dataset-dir", str(ds),
        "--output-dir", str(out),
        "--synthetic-mode", "placeholder",
        "--force",
    ])


def test_main_control_prior_cli_accepts_mode(tmp_path):
    ds = _fake_dataset(tmp_path, [
        {"id": "sub-00001", "group": "fcd", "has_roi": True},
        {"id": "sub-00002", "group": "hc", "has_roi": False},
        {"id": "sub-00003", "group": "hc", "has_roi": False},
    ])
    out = tmp_path / "out"

    main([
        "--dataset-dir", str(ds),
        "--output-dir", str(out),
        "--subjects", "sub-00001",
        "--synthetic-mode", "control-prior",
        "--control-prior-max-controls", "1",
        "--force",
    ])

    manifest_path = out / "sub-00001" / "manifest.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert manifest["synthetic_normal_source"] == "control_prior_roi_hybrid"
    assert manifest["control_prior_meta"]["control_prior_count"] == 1


# ---------------------------------------------------------------------------
# Dense splat / sampling mode tests
# ---------------------------------------------------------------------------

def test_dense_splat_artifact_created_when_presentation_exceeds_max(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"
    manifest = process_subject(
        rec, out, force=True,
        max_splat_points=100,
        presentation_splat_points=200,
    )
    assert "patient_splat_dense_ply" in manifest["artifacts"]
    assert "patient_splat_dense_meta" in manifest["artifacts"]
    assert "patient_splat_state" in manifest["artifacts"]
    assert "patient_splat_dense_state" in manifest["artifacts"]
    assert "paired_gaussian_state" in manifest["artifacts"]
    dense_ply = out / "sub-00001" / manifest["artifacts"]["patient_splat_dense_ply"]
    assert dense_ply.exists()
    dense_state = out / "sub-00001" / manifest["artifacts"]["patient_splat_dense_state"]
    assert dense_state.exists()
    paired_state = out / "sub-00001" / manifest["artifacts"]["paired_gaussian_state"]
    assert paired_state.exists()
    assert manifest["paired_gaussian_residual_meta"]["gaussian_state_schema"] == (
        "paired_gaussian_residual_v1"
    )
    assert manifest["paired_gaussian_residual_meta"]["max_points"] == 200
    assert manifest["paired_gaussian_residual_meta"]["n_gaussians"] <= 200
    assert manifest["paired_gaussian_residual_meta"]["n_gaussians"] == min(
        200,
        manifest["paired_gaussian_residual_meta"]["foreground_voxels_total"],
    )
    assert any(
        "presentation splat density" in note
        for note in manifest["notes"]
    )


def test_paired_gaussian_explicit_max_points_overrides_presentation_default(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(
        rec,
        tmp_path / "out",
        force=True,
        max_splat_points=100,
        presentation_splat_points=200,
        paired_gaussian_max_points=75,
    )

    assert manifest["paired_gaussian_residual_meta"]["max_points"] == 75
    assert manifest["paired_gaussian_residual_meta"]["n_gaussians"] == 75


def test_dense_splat_not_created_when_presentation_le_max(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"
    manifest = process_subject(
        rec, out, force=True,
        max_splat_points=200,
        presentation_splat_points=200,
    )
    assert "patient_splat_dense_ply" not in manifest["artifacts"]


def test_splat_sampling_mode_surface_default(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"
    manifest = process_subject(rec, out, force=True)
    # Standard splat must exist
    assert "patient_splat_ply" in manifest["artifacts"]


def test_splat_sampling_mode_cli_surface(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    out = tmp_path / "out"
    main([
        "--dataset-dir", str(ds),
        "--output-dir", str(out),
        "--synthetic-mode", "placeholder",
        "--splat-sampling-mode", "surface",
        "--max-splat-points", "100",
        "--presentation-splat-points", "200",
        "--force",
    ])
    manifest_path = out / "sub-00001" / "manifest.json"
    assert manifest_path.exists()
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert "patient_splat_dense_ply" in manifest["artifacts"]


def test_splat_sampling_mode_cli_uniform(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    out = tmp_path / "out"
    main([
        "--dataset-dir", str(ds),
        "--output-dir", str(out),
        "--synthetic-mode", "placeholder",
        "--splat-sampling-mode", "uniform",
        "--max-splat-points", "100",
        "--presentation-splat-points", "50",  # <= max -> no dense artifact
        "--force",
    ])
    manifest_path = out / "sub-00001" / "manifest.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert "patient_splat_ply" in manifest["artifacts"]
    assert "patient_splat_dense_ply" not in manifest["artifacts"]


# ---------------------------------------------------------------------------
# Gaussian field optional stage tests
# ---------------------------------------------------------------------------

def _fake_train_gaussian_field(
    volume_path, output_dir, roi_path=None, num_gaussians=20000,
    steps=300, image_size=128, views="all", device="auto", seed=42,
    learning_rate=1e-3, preview_every=0, volume_loss_weight=0.0,
    volume_sample_count=4096,
):
    """Fake train_gaussian_field that writes expected files without training."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "optimized_gaussian_splat.ply").write_bytes(b"PLY\n")
    (output_dir / "gaussian_field.pt").write_bytes(b"PT\n")
    metrics = {
        "backend": "fallback",
        "device": "cpu",
        "training_objective": (
            "mip_projection_mse_plus_volume_sample_mse"
            if volume_loss_weight > 0 else "mip_projection_mse"
        ),
        "final_loss": 0.1,
        "final_mip_loss": 0.08,
        "final_volume_loss": 0.02 if volume_loss_weight > 0 else None,
        "volume_loss_weight": volume_loss_weight,
        "volume_sample_count": volume_sample_count if volume_loss_weight > 0 else 0,
        "num_gaussians": 30,
        "steps": 2,
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics))
    (output_dir / "reconstruction_preview.png").write_bytes(b"PNG\n")
    if roi_path is not None and Path(roi_path).is_file():
        (output_dir / "lesion_gaussian_mask.npy").write_bytes(b"NPY\n")
    return metrics


def _fake_project_gaussian_support(
    support_field, volume_path, output_dir, preview_image_size=64, copy_opacity=True,
):
    """Fake same-support projection for synthetic normal Gaussian field tests."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "optimized_gaussian_splat.ply").write_bytes(b"PLY\n")
    (output_dir / "gaussian_field.pt").write_bytes(b"PT\n")
    metrics = {
        "backend": "support_projection",
        "device": "cpu",
        "training_objective": "projected_volume_on_existing_gaussian_support",
        "final_loss": None,
        "final_mip_loss": None,
        "final_volume_loss": None,
        "support_projection": True,
        "same_gaussian_support_as": str(support_field),
        "support_points_in_volume_bounds_fraction": 1.0,
        "num_gaussians": 30,
        "steps": 0,
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics))
    (output_dir / "reconstruction_preview.png").write_bytes(b"PNG\n")
    return metrics


def _fake_splat_reconstruction_audit(
    reference_volume,
    splat_ply,
    output_json,
    output_nifti,
    max_points=100_000,
    smooth_sigma_vox=1.0,
    kernel_mode="gaussian",
    gaussian_sigma_extent=2.5,
    rng_seed=42,
):
    """Fake reconstruction audit that marks browser splats as claim-ready."""
    ref = nib.load(str(reference_volume))
    zeros = np.zeros(ref.shape[:3], dtype=np.float32)
    nib.save(nib.Nifti1Image(zeros, ref.affine, ref.header), str(output_nifti))
    audit = {
        "method": "fake_splat_reconstruction_audit",
        "total_points": 30,
        "audited_points": 30,
        "subsampled": False,
        "kernel_mode": kernel_mode,
        "gaussian_sigma_extent": gaussian_sigma_extent,
        "foreground_coverage": 0.95,
        "all_foreground_metrics": {
            "mae": 0.01,
            "rmse": 0.02,
            "psnr": 34.0,
        },
        "represented_foreground_metrics": {
            "mae": 0.01,
            "psnr": 34.0,
        },
        "mip_metrics": {"mean_mae": 0.01},
        "mri_to_3dgs_readiness_status": "pass",
        "mri_to_3dgs_readiness_blocker_count": 0,
        "mri_to_3dgs_readiness_warning_count": 0,
        "mri_to_3dgs_claim_allowed": True,
        "mri_to_3dgs_claim_class": "volumetric_mri_gaussian_representation",
        "representation_readiness_gate": {
            "status": "pass",
            "blockers": [],
            "warnings": [],
        },
    }
    Path(output_json).write_text(json.dumps(audit))
    return audit


def _fake_optimized_gaussian_field_audit(
    gaussian_field_path,
    reference_volume_path,
    output_json,
    sample_count=8192,
    seed=1049,
    device="cpu",
):
    """Fake held-out optimized field audit that marks the field claim-ready."""
    audit = {
        "schema": "gaussian_field_fidelity_audit_v1",
        "checkpoint_path": str(gaussian_field_path),
        "reference_volume_path": str(reference_volume_path),
        "reference_shape": [20, 20, 16],
        "sample_count": sample_count,
        "seed": seed,
        "device": device,
        "affine_mismatch": False,
        "sample_counts": {"total": sample_count, "foreground": sample_count // 2, "background": sample_count // 2},
        "metrics": {
            "all": {"count": sample_count, "mae": 0.01, "rmse": 0.02, "psnr": 34.0, "correlation": 0.95},
            "foreground": {"count": sample_count // 2, "mae": 0.01, "rmse": 0.02, "psnr": 34.0, "correlation": 0.95},
            "background": {"count": sample_count // 2, "mae": 0.01, "rmse": 0.02, "psnr": 34.0, "correlation": 0.95},
        },
        "checkpoint_metadata": {
            "training_objective": "mip_projection_mse_plus_volume_sample_mse",
            "volume_loss_weight": 0.25,
            "volume_sample_count": 8192,
            "volume_path": str(reference_volume_path),
        },
        "readiness_gate": {
            "status": "pass",
            "optimized_mri_to_3dgs_claim_allowed": True,
            "blockers": [],
            "warnings": [],
            "thresholds": {
                "foreground_mae_pass": 0.25,
                "foreground_mae_review": 0.35,
                "all_mae_pass": 0.20,
                "all_mae_review": 0.30,
            },
        },
    }
    Path(output_json).write_text(json.dumps(audit))
    return audit


def _passing_optimized_gaussian_field_audit_meta() -> dict:
    return {
        "schema": "gaussian_field_fidelity_audit_v1",
        "sample_count": 8192,
        "seed": 1051,
        "device": "cpu",
        "affine_mismatch": False,
        "sample_counts": {"total": 8192, "foreground": 6144, "background": 2048},
        "metrics": {
            "all": {"count": 8192, "mae": 0.01, "rmse": 0.02, "psnr": 34.0, "correlation": 0.95},
            "foreground": {"count": 6144, "mae": 0.01, "rmse": 0.02, "psnr": 34.0, "correlation": 0.95},
            "background": {"count": 2048, "mae": 0.01, "rmse": 0.02, "psnr": 34.0, "correlation": 0.95},
        },
        "checkpoint_metadata": {
            "training_objective": "mip_projection_mse_plus_volume_sample_mse",
            "volume_loss_weight": 0.25,
            "volume_sample_count": 8192,
            "volume_path": "patient_flair.nii.gz",
        },
        "readiness_gate": {
            "status": "pass",
            "optimized_mri_to_3dgs_claim_allowed": True,
            "blockers": [],
            "warnings": [],
            "thresholds": {
                "foreground_mae_pass": 0.25,
                "foreground_mae_review": 0.35,
                "all_mae_pass": 0.20,
                "all_mae_review": 0.30,
            },
        },
    }


def _passing_supervised_roi_operating_point_meta() -> dict:
    return {
        "source_kind": "threshold_sweep",
        "threshold": 0.8,
        "selection_mode": "threshold",
        "selected_operating_point": True,
        "meets_operating_constraints": True,
        "n_fcd": 5,
        "n_hc": 5,
        "fcd_mean_manual_coverage": 0.74,
        "fcd_coverage_ge_0_50_rate": 0.8,
        "hc_mean_predicted_foreground_fraction": 0.006,
        "constraints": {
            "min_coverage_ge_0_50_rate": 0.5,
            "max_hc_mean_predicted_foreground_fraction": 0.01,
            "min_component_voxels": 64,
        },
        "min_coverage_ge_0_50_rate": 0.5,
        "max_hc_mean_predicted_foreground_fraction": 0.01,
    }


def test_gaussian_field_skipped_by_default(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(rec, tmp_path / "out", force=True)
    assert manifest["stage_status"]["gaussian_field"].startswith("skipped")
    assert "optimized_gaussian_splat_ply" not in manifest["artifacts"]
    assert manifest["gaussian_field_meta"] == {}


def test_gaussian_field_artifacts_recorded(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"

    with patch("fcd_prototype.gaussian_field.train_gaussian_field", _fake_train_gaussian_field):
        manifest = process_subject(
            rec, out, force=True,
            train_gaussian_field=True,
            gaussian_num_gaussians=30,
            gaussian_steps=2,
            gaussian_image_size=16,
            gaussian_views="axial",
            gaussian_device="cpu",
        )

    assert manifest["stage_status"]["gaussian_field"] == "ok"
    assert "optimized_gaussian_splat_ply" in manifest["artifacts"]
    assert "gaussian_field_checkpoint" in manifest["artifacts"]
    assert "gaussian_field_metrics" in manifest["artifacts"]
    assert "gaussian_field_preview" in manifest["artifacts"]
    assert "lesion_gaussian_mask" in manifest["artifacts"]
    assert manifest["gaussian_field_meta"]["backend"] == "fallback"
    assert manifest["gaussian_field_meta"]["final_loss"] == pytest.approx(0.1)


def test_gaussian_field_no_roi_no_mask(tmp_path):
    ds = _fake_dataset(tmp_path, [{"id": "sub-00002", "group": "hc", "has_roi": False}])
    rec = discover_subject(ds, "sub-00002", group="hc")
    out = tmp_path / "out"

    with patch("fcd_prototype.gaussian_field.train_gaussian_field", _fake_train_gaussian_field):
        manifest = process_subject(
            rec, out, force=True,
            train_gaussian_field=True,
            gaussian_steps=2,
            gaussian_image_size=16,
            gaussian_views="axial",
            gaussian_device="cpu",
        )

    assert manifest["stage_status"]["gaussian_field"] == "ok"
    assert "lesion_gaussian_mask" not in manifest["artifacts"]


def test_gaussian_field_failure_non_fatal(tmp_path):
    """A crash in the Gaussian stage must not prevent rest of manifest from being written."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"

    def _explode(volume_path, output_dir, **kwargs):
        raise RuntimeError("simulated GPU OOM")

    with patch("fcd_prototype.gaussian_field.train_gaussian_field", _explode):
        manifest = process_subject(
            rec, out, force=True,
            train_gaussian_field=True,
            gaussian_device="cpu",
        )

    assert manifest["stage_status"]["gaussian_field"].startswith("failed:")
    # Core stages still ran
    assert manifest["stage_status"]["patient_volumes"] == "ok"
    assert "patient_splat_ply" in manifest["artifacts"]
    assert any("non-fatal" in n for n in manifest["notes"])
    assert any("fall back" in lim for lim in manifest["limitations"])


def test_gaussian_field_cli_flag_skipped_without_flag(tmp_path):
    """Running CLI without --train-gaussian-field must set stage to skipped."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    out = tmp_path / "out"
    main([
        "--dataset-dir", str(ds),
        "--output-dir", str(out),
        "--synthetic-mode", "placeholder",
        "--force",
    ])
    manifest_path = out / "sub-00001" / "manifest.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert manifest["stage_status"]["gaussian_field"].startswith("skipped")


def test_gaussian_field_manifest_has_meta_key(tmp_path):
    """manifest.json must always have gaussian_field_meta (empty dict when skipped)."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(rec, tmp_path / "out", force=True)
    assert "gaussian_field_meta" in manifest
    assert isinstance(manifest["gaussian_field_meta"], dict)


# ---------------------------------------------------------------------------
# Synthetic Gaussian field and Gaussian residual tests
# ---------------------------------------------------------------------------

def _fake_compare_gaussian_fields(
    patient_field, normal_field, reference_volume, output_dir, roi=None,
    k_neighbors=4, max_points=200_000, heatmap_percentile=99.0,
    heatmap_method="field", field_chunk_size=128,
):
    """Fake compare_gaussian_fields that writes expected residual artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gaussian_residual_splat.ply").write_bytes(b"PLY\n")
    (output_dir / "gaussian_residual_heatmap.nii.gz").write_bytes(b"GZIP\n")
    (output_dir / "gaussian_residual_metrics.json").write_text(
        '{"k_neighbors":4,"n_patient":30,"n_normal":30,'
        '"heatmap_method":"field","residual_score_method":"indexwise_same_support_intensity",'
        '"same_gaussian_support":true,"gaussian_subtraction_claim_allowed":true,'
        '"support_fingerprint_schema":"gaussian_support_v1",'
        '"support_fingerprint_match":true,'
        '"patient_support_fingerprint_sha256":"' + ("b" * 64) + '",'
        '"normal_support_fingerprint_sha256":"' + ("b" * 64) + '",'
        '"residual_mean":0.05,"residual_p95":0.15,"residual_p99":0.2,'
        '"field_residual_mean":0.04,"field_residual_p95":0.12,'
        '"field_residual_p99":0.18,"field_chunk_size":128,'
        '"roi_metrics_valid":false,"roi_to_non_roi_ratio":0.0}'
    )
    (output_dir / "gaussian_residual_scores.npy").write_bytes(b"NPY\n")
    return {
        "k_neighbors": 4, "n_patient": 30, "n_normal": 30,
        "heatmap_method": heatmap_method,
        "residual_score_method": "indexwise_same_support_intensity",
        "same_gaussian_support": True,
        "support_fingerprint_schema": "gaussian_support_v1",
        "support_fingerprint_match": True,
        "patient_support_fingerprint_sha256": "b" * 64,
        "normal_support_fingerprint_sha256": "b" * 64,
        "gaussian_subtraction_claim_allowed": True,
        "residual_mean": 0.05, "residual_p95": 0.15, "residual_p99": 0.2,
        "field_residual_mean": 0.04, "field_residual_p95": 0.12,
        "field_residual_p99": 0.18, "field_chunk_size": field_chunk_size,
        "roi_metrics_valid": False, "roi_to_non_roi_ratio": 0.0,
    }


def _fake_predict_lesion_roi(
    checkpoint_path, volume_path, output_path,
    threshold=0.5, min_component_voxels=0, device="auto",
    selection_mode="threshold", foreground_fraction=None,
):
    """Fake supervised ROI predictor that writes dummy lesion mask."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shape = (20, 20, 16)
    data = np.zeros(shape, dtype=np.float32)
    data[8:12, 8:12, 6:10] = 1.0
    nib.save(
        nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.5, 1.0])),
        str(output_path),
    )
    return {
        "predicted_voxels": 64,
        "predicted_foreground_fraction": 64 / (20 * 20 * 16),
        "probability_mean": 0.75,
        "probability_max": 0.95,
        "threshold": threshold,
        "selection_mode": selection_mode,
        "foreground_fraction": foreground_fraction,
        "spatial_size": (64, 64, 64),
    }


def _fake_overlarge_predict_lesion_roi(
    checkpoint_path, volume_path, output_path,
    threshold=0.5, min_component_voxels=0, device="auto",
    selection_mode="threshold", foreground_fraction=None,
):
    """Fake supervised ROI predictor that oversegments most of the brain."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shape = (20, 20, 16)
    data = np.zeros(shape, dtype=np.float32)
    data[:16, :16, :12] = 1.0
    nib.save(
        nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.5, 1.0])),
        str(output_path),
    )
    voxels = int(data.sum())
    return {
        "predicted_voxels": voxels,
        "predicted_foreground_fraction": voxels / data.size,
        "probability_mean": 0.80,
        "probability_max": 0.99,
        "threshold": threshold,
        "selection_mode": selection_mode,
        "foreground_fraction": foreground_fraction,
        "spatial_size": (64, 64, 64),
    }


def test_synthetic_gaussian_field_skipped_by_default(tmp_path):
    """Default manifest has synthetic_gaussian_field and gaussian_residual skipped."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(rec, tmp_path / "out", force=True)
    assert manifest["stage_status"]["synthetic_gaussian_field"].startswith("skipped")
    assert manifest["stage_status"]["gaussian_residual"].startswith("skipped")
    assert manifest["synth_gaussian_field_meta"] == {}
    assert manifest["gaussian_residual_meta"] == {}


def test_synthetic_gaussian_field_training_records_artifacts(tmp_path):
    """Enabling train_synthetic_gaussian_field records artifacts and stage status."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"

    with patch("fcd_prototype.gaussian_field.train_gaussian_field", _fake_train_gaussian_field):
        with patch(
            "fcd_prototype.gaussian_field.project_volume_to_gaussian_support",
            _fake_project_gaussian_support,
        ):
            manifest = process_subject(
                rec, out, force=True,
                train_gaussian_field=True,
                train_synthetic_gaussian_field=True,
                gaussian_steps=2,
                gaussian_image_size=16,
                gaussian_views="axial",
                gaussian_device="cpu",
            )

    assert manifest["stage_status"]["synthetic_gaussian_field"] == "ok"
    assert "synthetic_normal_gaussian_splat_ply" in manifest["artifacts"]
    assert "synthetic_normal_gaussian_checkpoint" in manifest["artifacts"]
    assert "synthetic_normal_gaussian_metrics" in manifest["artifacts"]
    assert "synthetic_normal_gaussian_preview" in manifest["artifacts"]
    assert manifest["synth_gaussian_field_meta"]["backend"] == "support_projection"
    assert manifest["synth_gaussian_field_meta"]["support_projection"] is True


def test_gaussian_residual_skipped_without_checkpoint(tmp_path):
    """Enabling compute_gaussian_residual without checkpoints skips gracefully."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"

    # Don't train Gaussian fields; just try compute_gaussian_residual
    manifest = process_subject(
        rec, out, force=True,
        train_gaussian_field=False,
        train_synthetic_gaussian_field=False,
        compute_gaussian_residual=True,
    )

    assert "gaussian_field.pt not found" in manifest["stage_status"]["gaussian_residual"]
    assert "gaussian_residual_splat" not in manifest["artifacts"]


def test_gaussian_residual_with_both_training_and_compare(tmp_path):
    """Enabling both gaussian training flags and compute_gaussian_residual."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"

    with patch("fcd_prototype.gaussian_field.train_gaussian_field", _fake_train_gaussian_field):
        with patch(
            "fcd_prototype.gaussian_field.project_volume_to_gaussian_support",
            _fake_project_gaussian_support,
        ):
            with patch("fcd_prototype.gaussian_residual.compare_gaussian_fields", _fake_compare_gaussian_fields):
                manifest = process_subject(
                    rec, out, force=True,
                    train_gaussian_field=True,
                    train_synthetic_gaussian_field=True,
                    compute_gaussian_residual=True,
                    gaussian_steps=2,
                    gaussian_image_size=16,
                    gaussian_views="axial",
                    gaussian_device="cpu",
                )

    assert manifest["stage_status"]["gaussian_field"] == "ok"
    assert manifest["stage_status"]["synthetic_gaussian_field"] == "ok"
    assert manifest["stage_status"]["gaussian_residual"] == "ok"
    assert "gaussian_residual_splat" in manifest["artifacts"]
    assert "gaussian_residual_heatmap" in manifest["artifacts"]
    assert "gaussian_residual_metrics" in manifest["artifacts"]
    assert "gaussian_residual_scores" in manifest["artifacts"]
    assert manifest["gaussian_residual_meta"]["residual_mean"] == pytest.approx(0.05)
    assert manifest["gaussian_residual_meta"]["heatmap_method"] == "field"
    assert manifest["gaussian_residual_meta"]["field_residual_p99"] == pytest.approx(0.18)
    assert manifest["gaussian_residual_meta"]["support_fingerprint_match"] is True
    assert manifest["gaussian_residual_meta"]["patient_support_fingerprint_sha256"] == "b" * 64
    assert manifest["gaussian_residual_meta"]["normal_support_fingerprint_sha256"] == "b" * 64


def test_gaussian_residual_passes_field_method_options(tmp_path):
    """End-to-end wrapper passes field residual method settings into comparator."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"
    captured = {}

    def _capture_compare(
        patient_field, normal_field, reference_volume, output_dir, roi=None,
        k_neighbors=4, max_points=200_000, heatmap_percentile=99.0,
        heatmap_method="field", field_chunk_size=128,
    ):
        captured["heatmap_method"] = heatmap_method
        captured["field_chunk_size"] = field_chunk_size
        return _fake_compare_gaussian_fields(
            patient_field, normal_field, reference_volume, output_dir, roi=roi,
            k_neighbors=k_neighbors, max_points=max_points,
            heatmap_percentile=heatmap_percentile,
            heatmap_method=heatmap_method,
            field_chunk_size=field_chunk_size,
        )

    with patch("fcd_prototype.gaussian_field.train_gaussian_field", _fake_train_gaussian_field):
        with patch(
            "fcd_prototype.gaussian_field.project_volume_to_gaussian_support",
            _fake_project_gaussian_support,
        ):
            with patch("fcd_prototype.gaussian_residual.compare_gaussian_fields", _capture_compare):
                manifest = process_subject(
                    rec, out, force=True,
                    train_gaussian_field=True,
                    train_synthetic_gaussian_field=True,
                    compute_gaussian_residual=True,
                    gaussian_residual_heatmap_method="scatter",
                    gaussian_residual_field_chunk_size=33,
                    gaussian_steps=2,
                    gaussian_image_size=16,
                    gaussian_views="axial",
                    gaussian_device="cpu",
                )

    assert manifest["stage_status"]["gaussian_residual"] == "ok"
    assert captured == {"heatmap_method": "scatter", "field_chunk_size": 33}
    assert manifest["gaussian_residual_meta"]["heatmap_method"] == "scatter"
    assert manifest["gaussian_residual_meta"]["field_chunk_size"] == 33


# ---------------------------------------------------------------------------
# ROI strategy tests (auto, provided, provided-or-auto, none)
# ---------------------------------------------------------------------------

def _fake_generate_candidate_rois(volume_path, output_dir, **kwargs):
    """Fake candidate ROI generator for testing ROI strategy logic."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Create dummy union mask
    shape = (20, 20, 16)
    data = np.zeros(shape, dtype=np.float32)
    data[8:12, 8:12, 6:10] = 1.0
    union_path = output_dir / "candidate_roi_union.nii.gz"
    nib.save(
        nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.5, 1.0])),
        str(union_path),
    )
    # Write individual candidate
    cand_path = output_dir / "candidate_roi_00.nii.gz"
    nib.save(
        nib.Nifti1Image(data.copy(), np.diag([1.0, 1.0, 1.5, 1.0])),
        str(cand_path),
    )
    return {
        "selected_count": 1,
        "centers_voxel": [[10.0, 10.0, 8.0]],
        "centers_world": [[10.0, 10.0, 12.0]],
        "individual_masks": ["candidate_roi_00.nii.gz"],
        "union_mask": "candidate_roi_union.nii.gz",
    }


def test_roi_strategy_auto_with_candidate(tmp_path):
    """roi_strategy='auto' should use candidate ROI when available."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"

    with patch(
        "fcd_prototype.end_to_end.generate_candidate_rois",
        _fake_generate_candidate_rois,
    ):
        manifest = process_subject(rec, out, roi_strategy="auto", force=True)

    # Should use candidate, not provided ROI
    assert manifest["roi_source"] == "candidate"
    assert manifest["stage_status"]["candidate_roi"].startswith("ok")
    assert "lesion_roi" in manifest["artifacts"]
    assert "manual_lesion_roi" in manifest["artifacts"]
    assert manifest["residual_candidate_meta"]["selected_roi_source"] == "candidate"
    assert manifest["residual_candidate_meta"]["manual_roi_source"] == "provided"
    assert manifest["residual_candidate_meta"]["manual_roi_available"] is True
    assert (
        manifest["residual_candidate_meta"][
            "manual_overlap_independent_from_fuzzing_roi"
        ]
        is True
    )
    assert (
        manifest["residual_candidate_meta"]["manual_overlap_independence_reason"]
        == "independent_manual_roi_not_used_for_fuzzing"
    )


def test_roi_strategy_auto_ignores_provided(tmp_path):
    """roi_strategy='auto' should ignore provided ROI and use candidate instead."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"

    with patch(
        "fcd_prototype.end_to_end.generate_candidate_rois",
        _fake_generate_candidate_rois,
    ):
        manifest = process_subject(rec, out, roi_strategy="auto", force=True)

    # roi_source should be "candidate", not "provided"
    assert manifest["roi_source"] != "provided"
    assert manifest["roi_source"] == "candidate"


def test_roi_strategy_provided_uses_provided(tmp_path):
    """roi_strategy='provided' should use provided ROI."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"

    with patch(
        "fcd_prototype.end_to_end.generate_candidate_rois",
        _fake_generate_candidate_rois,
    ):
        manifest = process_subject(rec, out, roi_strategy="provided", force=True)

    assert manifest["roi_source"] == "provided"
    assert manifest["stage_status"]["roi_mask"] == "ok"


def test_roi_strategy_provided_or_auto_uses_provided_when_available(tmp_path):
    """roi_strategy='provided-or-auto' should prefer provided ROI and not generate candidates."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"

    def should_not_be_called(*args, **kwargs):
        raise AssertionError("generate_candidate_rois should not be called when provided ROI exists")

    with patch(
        "fcd_prototype.end_to_end.generate_candidate_rois",
        should_not_be_called,
    ):
        manifest = process_subject(rec, out, roi_strategy="provided-or-auto", force=True)

    assert manifest["roi_source"] == "provided"
    assert "lesion_roi" in manifest["artifacts"]
    assert manifest["stage_status"]["candidate_roi"] == "skipped: provided ROI available"


def test_roi_strategy_provided_or_auto_fallback_to_candidate(tmp_path):
    """roi_strategy='provided-or-auto' should fallback to candidate when no provided ROI."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00002", "group": "hc", "has_roi": False}])
    rec = discover_subject(ds, "sub-00002", group="hc")
    out = tmp_path / "out"

    with patch(
        "fcd_prototype.end_to_end.generate_candidate_rois",
        _fake_generate_candidate_rois,
    ):
        manifest = process_subject(rec, out, roi_strategy="provided-or-auto", force=True)

    assert manifest["roi_source"] == "candidate"
    assert manifest["stage_status"]["candidate_roi"].startswith("ok")


def test_roi_strategy_none_no_roi(tmp_path):
    """roi_strategy='none' should never create ROI mask."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"

    with patch(
        "fcd_prototype.end_to_end.generate_candidate_rois",
        _fake_generate_candidate_rois,
    ):
        manifest = process_subject(rec, out, roi_strategy="none", force=True)

    assert manifest["roi_source"] == "none"
    assert "lesion_roi" not in manifest["artifacts"]
    assert manifest["stage_status"]["roi_mask"] == "skipped: roi_strategy=none"
    # Should not attempt candidate generation
    assert manifest["stage_status"]["candidate_roi"] == "skipped: roi_strategy=none"


def test_roi_strategy_none_no_candidate_generation(tmp_path):
    """roi_strategy='none' should skip candidate ROI generation entirely."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"

    # If generate_candidate_rois is called, it will raise an error
    def should_not_be_called(*args, **kwargs):
        raise AssertionError("generate_candidate_rois should not be called for roi_strategy='none'")

    with patch(
        "fcd_prototype.end_to_end.generate_candidate_rois",
        should_not_be_called,
    ):
        manifest = process_subject(rec, out, roi_strategy="none", force=True)

    # Manifest was created without calling generate_candidate_rois
    assert manifest is not None


def test_roi_strategy_artifact_and_stage_status_clear(tmp_path):
    """Artifacts and stage_status should remain clear for skipped stages."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": False}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"

    manifest = process_subject(rec, out, roi_strategy="none", force=True)

    # No ROI-related artifacts should exist
    for key in manifest["artifacts"]:
        assert not key.startswith("candidate_roi"), f"Unexpected artifact: {key}"
        assert key != "lesion_roi", "lesion_roi should not exist"
    # Stage status should have explicit skip messages, not empty
    assert "candidate_roi" in manifest["stage_status"]
    assert "roi_mask" in manifest["stage_status"]
    assert manifest["stage_status"]["candidate_roi"] != ""
    assert manifest["stage_status"]["roi_mask"] != ""


# ---------------------------------------------------------------------------
# Supervised ROI strategy tests
# ---------------------------------------------------------------------------

def test_supervised_roi_strategy_with_checkpoint(tmp_path):
    """roi_strategy='supervised' with checkpoint should predict and use supervised ROI."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": False}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"
    ckpt = tmp_path / "fake_checkpoint.pt"
    ckpt.write_bytes(b"PT")

    with patch(
        "fcd_prototype.supervised_lesion_proposal.predict_lesion_roi",
        _fake_predict_lesion_roi,
    ):
        manifest = process_subject(
            rec, out, roi_strategy="supervised", force=True,
            supervised_roi_checkpoint=ckpt,
        )

    assert manifest["roi_source"] == "supervised"
    assert "lesion_roi" in manifest["artifacts"]
    assert manifest["stage_status"]["supervised_roi"] == "ok"
    assert manifest["supervised_roi_meta"]["accepted"] is True
    assert manifest["supervised_roi_meta"]["predicted_foreground_fraction"] == pytest.approx(0.01)
    assert any("Supervised ROI accepted" in n for n in manifest["notes"])



def test_supervised_roi_strategy_missing_checkpoint(tmp_path):
    """roi_strategy='supervised' with missing checkpoint should skip gracefully."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": False}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"

    manifest = process_subject(
        rec, out, roi_strategy="supervised", force=True,
        supervised_roi_checkpoint=None,
    )

    assert manifest["stage_status"]["supervised_roi"] == "skipped: checkpoint not provided or missing"
    assert "lesion_roi" not in manifest["artifacts"]


def test_provided_or_supervised_prefers_provided(tmp_path):
    """roi_strategy='provided-or-supervised' with provided ROI should not call supervised."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"
    ckpt = tmp_path / "fake_checkpoint.pt"
    ckpt.write_bytes(b"PT")

    def should_not_be_called(*args, **kwargs):
        raise AssertionError("predict_lesion_roi should not be called when provided ROI exists")

    with patch(
        "fcd_prototype.supervised_lesion_proposal.predict_lesion_roi",
        should_not_be_called,
    ):
        manifest = process_subject(
            rec, out, roi_strategy="provided-or-supervised", force=True,
            supervised_roi_checkpoint=ckpt,
        )

    assert manifest["roi_source"] == "provided"
    assert manifest["stage_status"]["supervised_roi"] == "skipped: provided ROI available"
    assert "lesion_roi" in manifest["artifacts"]



def test_provided_or_supervised_fallback_to_supervised(tmp_path):
    """roi_strategy='provided-or-supervised' without provided ROI should use supervised."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": False}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"
    ckpt = tmp_path / "fake_checkpoint.pt"
    ckpt.write_bytes(b"PT")

    with patch(
        "fcd_prototype.supervised_lesion_proposal.predict_lesion_roi",
        _fake_predict_lesion_roi,
    ):
        manifest = process_subject(
            rec, out, roi_strategy="provided-or-supervised", force=True,
            supervised_roi_checkpoint=ckpt,
        )

    assert manifest["roi_source"] == "supervised"
    assert manifest["stage_status"]["supervised_roi"] == "ok"
    assert "lesion_roi" in manifest["artifacts"]


def test_supervised_roi_rejects_overlarge_prediction(tmp_path):
    """Overlarge supervised masks should not be allowed to drive fuzz/subtract."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": False}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"
    ckpt = tmp_path / "fake_checkpoint.pt"
    ckpt.write_bytes(b"PT")

    with patch(
        "fcd_prototype.supervised_lesion_proposal.predict_lesion_roi",
        _fake_overlarge_predict_lesion_roi,
    ):
        manifest = process_subject(
            rec, out, roi_strategy="supervised", force=True,
            supervised_roi_checkpoint=ckpt,
            supervised_roi_max_foreground_fraction=0.01,
        )

    assert manifest["roi_source"] == "none"
    assert manifest["stage_status"]["supervised_roi"] == "rejected: above_max_foreground_fraction"
    assert manifest["supervised_roi_meta"]["accepted"] is False
    assert manifest["supervised_roi_meta"]["rejection_reason"] == "above_max_foreground_fraction"
    assert "supervised_roi" in manifest["artifacts"]
    assert "lesion_roi" not in manifest["artifacts"]
    assert manifest["stage_status"]["roi_mask"] == "skipped: no ROI mask available"



def test_supervised_roi_cli_with_checkpoint(tmp_path):
    """CLI with --roi-strategy supervised and --supervised-roi-checkpoint should work."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": False}])
    out = tmp_path / "out"
    ckpt = tmp_path / "fake_checkpoint.pt"
    ckpt.write_bytes(b"PT")

    with patch(
        "fcd_prototype.supervised_lesion_proposal.predict_lesion_roi",
        _fake_predict_lesion_roi,
    ):
        main([
            "--dataset-dir", str(ds),
            "--output-dir", str(out),
            "--roi-strategy", "supervised",
            "--supervised-roi-checkpoint", str(ckpt),
            "--force",
        ])

    manifest_path = out / "sub-00001" / "manifest.json"
    assert manifest_path.exists()
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert manifest["roi_source"] == "supervised"



def test_supervised_roi_manifest_always_has_status(tmp_path):
    """manifest.json must always have supervised_roi in stage_status."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(rec, tmp_path / "out", force=True)
    assert "supervised_roi" in manifest["stage_status"]


def test_supervised_roi_threshold_parameter(tmp_path):
    """Supervised ROI should pass threshold parameter through."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": False}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"
    ckpt = tmp_path / "fake_checkpoint.pt"
    ckpt.write_bytes(b"PT")

    call_args = []

    def capture_call(*args, **kwargs):
        call_args.append(kwargs)
        return _fake_predict_lesion_roi(*args, **kwargs)

    with patch(
        "fcd_prototype.supervised_lesion_proposal.predict_lesion_roi",
        side_effect=capture_call,
    ):
        manifest = process_subject(
            rec, out, roi_strategy="supervised", force=True,
            supervised_roi_checkpoint=ckpt,
            supervised_roi_threshold=0.7,
        )

    assert len(call_args) > 0
    assert call_args[0]["threshold"] == 0.7


def test_supervised_roi_selection_mode_parameters(tmp_path):
    """Supervised ROI should pass budgeted selection settings through."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": False}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"
    ckpt = tmp_path / "fake_checkpoint.pt"
    ckpt.write_bytes(b"PT")

    call_args = []

    def capture_call(*args, **kwargs):
        call_args.append(kwargs)
        return _fake_predict_lesion_roi(*args, **kwargs)

    with patch(
        "fcd_prototype.supervised_lesion_proposal.predict_lesion_roi",
        side_effect=capture_call,
    ):
        manifest = process_subject(
            rec, out, roi_strategy="supervised", force=True,
            supervised_roi_checkpoint=ckpt,
            supervised_roi_selection_mode="top-fraction",
            supervised_roi_foreground_fraction=0.015,
        )

    assert len(call_args) > 0
    assert call_args[0]["selection_mode"] == "top-fraction"
    assert call_args[0]["foreground_fraction"] == pytest.approx(0.015)
    assert manifest["supervised_roi_meta"]["selection_mode"] == "top-fraction"
    assert manifest["supervised_roi_meta"]["foreground_fraction"] == pytest.approx(0.015)


def test_supervised_roi_operating_point_overrides_runtime_settings(tmp_path):
    """Calibrated operating-point JSON should drive supervised ROI prediction."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": False}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"
    ckpt = tmp_path / "fake_checkpoint.pt"
    ckpt.write_bytes(b"PT")
    op = tmp_path / "threshold_sweep.json"
    op.write_text(json.dumps({
        "method": "supervised_lesion_threshold_sweep",
        "selected_operating_point": {
            "threshold": 0.8,
            "selection_mode": "threshold-and-top-fraction",
            "foreground_fraction": 0.012,
        },
        "constraints": {
            "min_component_voxels": 9,
            "max_hc_mean_predicted_foreground_fraction": 0.02,
        },
    }))

    call_args = []

    def capture_call(*args, **kwargs):
        call_args.append(kwargs)
        return _fake_predict_lesion_roi(*args, **kwargs)

    with patch(
        "fcd_prototype.supervised_lesion_proposal.predict_lesion_roi",
        side_effect=capture_call,
    ):
        manifest = process_subject(
            rec, out, roi_strategy="supervised", force=True,
            supervised_roi_checkpoint=ckpt,
            supervised_roi_operating_point=op,
            supervised_roi_threshold=0.2,
            supervised_roi_selection_mode="threshold",
            supervised_roi_foreground_fraction=None,
            supervised_roi_max_foreground_fraction=0.5,
        )

    assert call_args[0]["threshold"] == pytest.approx(0.8)
    assert call_args[0]["selection_mode"] == "threshold-and-top-fraction"
    assert call_args[0]["foreground_fraction"] == pytest.approx(0.012)
    assert call_args[0]["min_component_voxels"] == 9
    assert manifest["stage_status"]["supervised_roi_operating_point"] == "ok"
    assert manifest["supervised_roi_meta"]["max_predicted_foreground_fraction"] == pytest.approx(0.02)
    assert manifest["supervised_roi_meta"]["operating_point"]["source_kind"] == "threshold_sweep"
    assert manifest["supervised_roi_resolved_params"]["threshold"] == pytest.approx(0.8)
    assert (
        manifest["supervised_roi_resolved_params"]["max_foreground_fraction"]
        == pytest.approx(0.02)
    )


# ---------------------------------------------------------------------------
# --paper-3dgs-mode preset tests
# ---------------------------------------------------------------------------

def test_paper_3dgs_mode_sets_training_flags_and_pipeline_preset(tmp_path):
    """--paper-3dgs-mode enables both Gaussian stages and optimized residuals."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    out = tmp_path / "out"

    with patch("fcd_prototype.gaussian_field.train_gaussian_field", _fake_train_gaussian_field):
        with patch(
            "fcd_prototype.gaussian_field.project_volume_to_gaussian_support",
            _fake_project_gaussian_support,
        ):
            with patch(
                "fcd_prototype.gaussian_residual.compare_gaussian_fields",
                _fake_compare_gaussian_fields,
            ):
                main([
                    "--dataset-dir", str(ds),
                    "--output-dir", str(out),
                    "--synthetic-mode", "placeholder",
                    "--paper-3dgs-mode",
                    "--skip-paired-gaussian-residual",
                    "--force",
                ])

    manifest_path = out / "sub-00001" / "manifest.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)

    assert manifest["pipeline_preset"] == "paper_3dgs"
    assert manifest["paper_3dgs_mode_meta"]["enabled"] is True
    assert manifest["paper_3dgs_mode_meta"]["train_gaussian_field"] is True
    assert manifest["paper_3dgs_mode_meta"]["train_synthetic_gaussian_field"] is True
    assert manifest["paper_3dgs_mode_meta"]["compute_paired_gaussian_residual"] is True
    assert manifest["paper_3dgs_mode_meta"]["compute_gaussian_residual"] is True
    assert manifest["stage_status"]["gaussian_field"] == "ok"
    assert manifest["stage_status"]["synthetic_gaussian_field"] == "ok"
    assert manifest["stage_status"]["paired_gaussian_residual"] == "ok"
    assert manifest["stage_status"]["gaussian_residual"] == "ok"
    assert "gaussian_residual_splat" in manifest["artifacts"]


def test_paper_3dgs_mode_default_volume_loss_weight(tmp_path):
    """--paper-3dgs-mode applies 0.25 volume loss weight when not explicitly set."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    out = tmp_path / "out"
    captured = {}

    def capture_train(volume_path, output_dir, **kwargs):
        captured.update({
            "volume_loss_weight": kwargs.get("volume_loss_weight"),
            "volume_sample_count": kwargs.get("volume_sample_count"),
        })
        return _fake_train_gaussian_field(volume_path, output_dir, **kwargs)

    with patch("fcd_prototype.gaussian_field.train_gaussian_field", capture_train):
        with patch(
            "fcd_prototype.gaussian_field.project_volume_to_gaussian_support",
            _fake_project_gaussian_support,
        ):
            with patch(
                "fcd_prototype.gaussian_residual.compare_gaussian_fields",
                _fake_compare_gaussian_fields,
            ):
                main([
                    "--dataset-dir", str(ds),
                    "--output-dir", str(out),
                    "--synthetic-mode", "placeholder",
                    "--paper-3dgs-mode",
                    "--force",
                ])

    assert captured["volume_loss_weight"] == pytest.approx(0.25)
    assert captured["volume_sample_count"] == 8192


def test_paper_3dgs_mode_respects_explicit_volume_loss_weight(tmp_path):
    """--paper-3dgs-mode respects --gaussian-volume-loss-weight when explicitly passed."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    out = tmp_path / "out"
    captured = {}

    def capture_train(volume_path, output_dir, **kwargs):
        captured.update({
            "volume_loss_weight": kwargs.get("volume_loss_weight"),
        })
        return _fake_train_gaussian_field(volume_path, output_dir, **kwargs)

    with patch("fcd_prototype.gaussian_field.train_gaussian_field", capture_train):
        with patch(
            "fcd_prototype.gaussian_field.project_volume_to_gaussian_support",
            _fake_project_gaussian_support,
        ):
            with patch(
                "fcd_prototype.gaussian_residual.compare_gaussian_fields",
                _fake_compare_gaussian_fields,
            ):
                main([
                    "--dataset-dir", str(ds),
                    "--output-dir", str(out),
                    "--synthetic-mode", "placeholder",
                    "--paper-3dgs-mode",
                    "--gaussian-volume-loss-weight", "0.5",
                    "--force",
                ])

    assert captured["volume_loss_weight"] == pytest.approx(0.5)


def test_paper_3dgs_mode_respects_explicit_zero_volume_loss_weight(tmp_path):
    """--paper-3dgs-mode allows explicit 0 to disable volume loss."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    out = tmp_path / "out"
    captured = {}

    def capture_train(volume_path, output_dir, **kwargs):
        captured.update({
            "volume_loss_weight": kwargs.get("volume_loss_weight"),
        })
        return _fake_train_gaussian_field(volume_path, output_dir, **kwargs)

    with patch("fcd_prototype.gaussian_field.train_gaussian_field", capture_train):
        with patch(
            "fcd_prototype.gaussian_field.project_volume_to_gaussian_support",
            _fake_project_gaussian_support,
        ):
            with patch(
                "fcd_prototype.gaussian_residual.compare_gaussian_fields",
                _fake_compare_gaussian_fields,
            ):
                main([
                    "--dataset-dir", str(ds),
                    "--output-dir", str(out),
                    "--synthetic-mode", "placeholder",
                    "--paper-3dgs-mode",
                    "--gaussian-volume-loss-weight", "0",
                    "--force",
                ])

    assert captured["volume_loss_weight"] == pytest.approx(0.0)


def test_paper_3dgs_mode_respects_explicit_sample_count(tmp_path):
    """--paper-3dgs-mode respects --gaussian-volume-sample-count when explicitly passed."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    out = tmp_path / "out"
    captured = {}

    def capture_train(volume_path, output_dir, **kwargs):
        captured.update({
            "volume_sample_count": kwargs.get("volume_sample_count"),
        })
        return _fake_train_gaussian_field(volume_path, output_dir, **kwargs)

    with patch("fcd_prototype.gaussian_field.train_gaussian_field", capture_train):
        with patch(
            "fcd_prototype.gaussian_field.project_volume_to_gaussian_support",
            _fake_project_gaussian_support,
        ):
            with patch(
                "fcd_prototype.gaussian_residual.compare_gaussian_fields",
                _fake_compare_gaussian_fields,
            ):
                main([
                    "--dataset-dir", str(ds),
                    "--output-dir", str(out),
                    "--synthetic-mode", "placeholder",
                    "--paper-3dgs-mode",
                    "--gaussian-volume-sample-count", "2048",
                    "--force",
                ])

    assert captured["volume_sample_count"] == 2048


def test_no_preset_records_none_in_manifest(tmp_path):
    """Without --paper-3dgs-mode, pipeline_preset is None in the manifest."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    out = tmp_path / "out"
    main([
        "--dataset-dir", str(ds),
        "--output-dir", str(out),
        "--synthetic-mode", "placeholder",
        "--force",
    ])
    manifest_path = out / "sub-00001" / "manifest.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert manifest["pipeline_preset"] is None
    assert manifest["paper_3dgs_mode_meta"] == {}


def test_no_preset_uses_default_volume_loss_zero(tmp_path):
    """Without --paper-3dgs-mode, volume loss weight defaults to 0.0."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    out = tmp_path / "out"
    captured = {}

    def capture_train(volume_path, output_dir, **kwargs):
        captured.update({
            "volume_loss_weight": kwargs.get("volume_loss_weight"),
            "volume_sample_count": kwargs.get("volume_sample_count"),
        })
        return _fake_train_gaussian_field(volume_path, output_dir, **kwargs)

    with patch("fcd_prototype.gaussian_field.train_gaussian_field", capture_train):
        with patch(
            "fcd_prototype.gaussian_field.project_volume_to_gaussian_support",
            _fake_project_gaussian_support,
        ):
            main([
                "--dataset-dir", str(ds),
                "--output-dir", str(out),
                "--synthetic-mode", "placeholder",
                "--train-gaussian-field",
                "--force",
            ])

    assert captured["volume_loss_weight"] == pytest.approx(0.0)
    assert captured["volume_sample_count"] == 4096


# ---------------------------------------------------------------------------
# Product readiness tests
# ---------------------------------------------------------------------------

def test_product_readiness_not_ready_by_default(tmp_path):
    """Default manifest without optimized Gaussian field/residual is not_ready."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(rec, tmp_path / "out", force=True)

    assert "product_readiness" in manifest
    pr = manifest["product_readiness"]
    assert pr["overall_status"] == "not_ready"
    assert isinstance(pr["blockers"], list)
    assert any("Optimized Gaussian" in b for b in pr["blockers"])
    assert "gates" in pr
    assert pr["gates"]["optimized_gaussian_field"]["status"] == "fail"
    assert pr["gates"]["optimized_gaussian_subtraction"]["status"] == "fail"


def test_product_readiness_candidate_roi_not_ready_without_localization_evidence(tmp_path):
    """Candidate ROI artifacts alone should not make the product detection-ready."""
    ds = _fake_dataset(tmp_path, [
        {"id": "sub-00001", "group": "fcd", "has_roi": True},
        {"id": "sub-00002", "group": "hc", "has_roi": False},
    ])
    out = tmp_path / "out"

    with patch(
        "fcd_prototype.end_to_end.generate_candidate_rois",
        _fake_generate_candidate_rois,
    ):
        with patch(
            "fcd_prototype.splat_reconstruction_audit.audit_splat_reconstruction",
            _fake_splat_reconstruction_audit,
        ):
            with patch("fcd_prototype.gaussian_field.train_gaussian_field", _fake_train_gaussian_field):
                with patch(
                    "fcd_prototype.gaussian_field.project_volume_to_gaussian_support",
                    _fake_project_gaussian_support,
                ):
                    with patch(
                        "fcd_prototype.gaussian_field_audit.audit_gaussian_field_fidelity",
                        _fake_optimized_gaussian_field_audit,
                    ):
                        with patch(
                            "fcd_prototype.gaussian_residual.compare_gaussian_fields",
                            _fake_compare_gaussian_fields,
                        ):
                            main([
                                "--dataset-dir", str(ds),
                                "--output-dir", str(out),
                                "--subjects", "sub-00001",
                                "--synthetic-mode", "control-prior",
                                "--roi-strategy", "auto",
                                "--paper-3dgs-mode",
                                "--skip-paired-gaussian-residual",
                                "--force",
                            ])

    manifest_path = out / "sub-00001" / "manifest.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)

    assert "product_readiness" in manifest
    pr = manifest["product_readiness"]
    assert pr["overall_status"] == "not_ready"
    assert pr["gates"]["roi_provenance"]["status"] == "pass"
    assert pr["gates"]["roi_localization_evidence"]["status"] == "fail"
    assert (
        pr["gates"]["roi_localization_evidence"]["evidence_reason"]
        == "heuristic_candidate_roi_unvalidated"
    )
    assert any("roi_localization_evidence" in b for b in pr["blockers"])
    assert pr["gates"]["healthy_reference"]["synthetic_normal_source"] == (
        "control_prior_roi_hybrid"
    )
    assert pr["gates"]["optimized_gaussian_field"]["status"] == "pass"
    assert pr["gates"]["optimized_gaussian_subtraction"]["status"] == "pass"
    assert pr["gates"]["direct_browser_splat"]["status"] == "pass"
    assert pr["gates"]["healthy_reference"]["status"] == "pass"


def test_product_readiness_placeholder_not_ready_even_with_3dgs_core(tmp_path):
    """Placeholder healthy references remain engineering-only, not product-ready."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    out = tmp_path / "out"

    with patch("fcd_prototype.gaussian_field.train_gaussian_field", _fake_train_gaussian_field):
        with patch(
            "fcd_prototype.gaussian_field.project_volume_to_gaussian_support",
            _fake_project_gaussian_support,
        ):
            with patch(
                "fcd_prototype.gaussian_field_audit.audit_gaussian_field_fidelity",
                _fake_optimized_gaussian_field_audit,
            ):
                with patch(
                    "fcd_prototype.gaussian_residual.compare_gaussian_fields",
                    _fake_compare_gaussian_fields,
                ):
                    main([
                        "--dataset-dir", str(ds),
                        "--output-dir", str(out),
                        "--synthetic-mode", "placeholder",
                        "--paper-3dgs-mode",
                        "--skip-paired-gaussian-residual",
                        "--force",
                    ])

    manifest_path = out / "sub-00001" / "manifest.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)

    assert "product_readiness" in manifest
    pr = manifest["product_readiness"]
    assert pr["overall_status"] == "not_ready"
    assert pr["gates"]["healthy_reference"]["status"] == "fail"
    assert any("synthetic_normal_source=placeholder" in b for b in pr["blockers"])


def test_product_readiness_roi_provenance_provided_blocks_uploaded_mri(tmp_path):
    """Provided ROI should block roi_provenance gate for uploaded-MRI detection."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"
    manifest = process_subject(rec, out, roi_strategy="provided", force=True)

    pr = manifest["product_readiness"]
    roi_gate = pr["gates"]["roi_provenance"]
    assert roi_gate["roi_source"] == "provided"
    # Check if warning is in blockers (for provided ROI case)
    assert any("roi_source=provided" in b for b in pr["blockers"])


def test_product_readiness_roi_provenance_candidate_ok(tmp_path):
    """Candidate ROI should pass roi_provenance gate."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    out = tmp_path / "out"

    with patch(
        "fcd_prototype.end_to_end.generate_candidate_rois",
        _fake_generate_candidate_rois,
    ):
        manifest = process_subject(rec, out, roi_strategy="auto", force=True)

    pr = manifest["product_readiness"]
    roi_gate = pr["gates"]["roi_provenance"]
    assert roi_gate["roi_source"] == "candidate"
    assert roi_gate["status"] == "pass"
    evidence_gate = pr["gates"]["roi_localization_evidence"]
    assert evidence_gate["status"] == "fail"
    assert evidence_gate["evidence_reason"] == "heuristic_candidate_roi_unvalidated"
    assert not any("roi_source=provided" in b for b in pr["blockers"])


def test_roi_localization_evidence_rejects_metricless_threshold_sweep():
    manifest = {
        "group": "fcd",
        "roi_source": "supervised",
        "supervised_roi_meta": {
            "accepted": True,
            "operating_point": {
                "source_kind": "threshold_sweep",
                "threshold": 0.8,
            },
        },
    }

    gate = _roi_localization_evidence_gate(manifest)

    assert gate["status"] == "fail"
    assert (
        gate["evidence_reason"]
        == "supervised_roi_threshold_sweep_missing_population_metrics"
    )
    assert gate["fcd_coverage_ge_0_50_rate"] is None
    assert gate["hc_mean_predicted_foreground_fraction"] is None


def test_roi_localization_evidence_rejects_weak_threshold_sweep():
    op = _passing_supervised_roi_operating_point_meta()
    op["fcd_coverage_ge_0_50_rate"] = 0.2
    op["hc_mean_predicted_foreground_fraction"] = 0.03
    manifest = {
        "group": "fcd",
        "roi_source": "supervised",
        "supervised_roi_meta": {
            "accepted": True,
            "operating_point": op,
        },
    }

    gate = _roi_localization_evidence_gate(manifest)

    assert gate["status"] == "fail"
    assert (
        gate["evidence_reason"]
        == "supervised_roi_threshold_sweep_failed_population_constraints"
    )
    assert gate["required_min_fcd_coverage_ge_0_50_rate"] == pytest.approx(0.5)
    assert (
        gate["required_max_hc_mean_predicted_foreground_fraction"]
        == pytest.approx(0.01)
    )


def test_roi_localization_evidence_accepts_population_validated_threshold_sweep():
    manifest = {
        "group": "fcd",
        "roi_source": "supervised",
        "supervised_roi_meta": {
            "accepted": True,
            "operating_point": _passing_supervised_roi_operating_point_meta(),
        },
    }

    gate = _roi_localization_evidence_gate(manifest)

    assert gate["status"] == "pass"
    assert (
        gate["evidence_reason"]
        == "calibrated_supervised_roi_threshold_sweep_population_validated"
    )
    assert gate["fcd_coverage_ge_0_50_rate"] == pytest.approx(0.8)
    assert gate["hc_mean_predicted_foreground_fraction"] == pytest.approx(0.006)


def test_product_readiness_gates_all_present(tmp_path):
    """All expected gates should be present in product_readiness."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(rec, tmp_path / "out", force=True)

    pr = manifest["product_readiness"]
    gates = pr["gates"]
    expected_gates = {
        "data_quality",
        "direct_browser_splat",
        "optimized_gaussian_field",
        "healthy_reference",
        "optimized_gaussian_subtraction",
        "niivue_residual_overlay",
        "lesion_candidate_overlay",
        "roi_provenance",
        "roi_localization_evidence",
    }
    assert set(gates.keys()) == expected_gates
    for gate_name in expected_gates:
        gate = gates[gate_name]
        assert "status" in gate
        assert "interpretation" in gate
        assert gate["status"] in ("pass", "fail", "warning")


def test_paper_method_feasibility_manifest_blocks_original_pitch_by_default(tmp_path):
    """Engineering artifacts do not imply the original NVIDIA counterfactual claim."""
    ds = _fake_dataset(tmp_path, [{"id": "sub-00001", "group": "fcd", "has_roi": True}])
    rec = discover_subject(ds, "sub-00001", group="fcd")
    manifest = process_subject(rec, tmp_path / "out", force=True)

    assert "paper_method_feasibility" in manifest
    feasibility = manifest["paper_method_feasibility"]
    assert feasibility["overall_status"] == "not_feasible_for_original_pitch"
    assert "nvidia_patient_conditioned_counterfactual" in feasibility["blocking_gates"]
    assert "optimized_mri_native_3dgs" in feasibility["blocking_gates"]
    assert feasibility["claim_boundaries"]["fallback_is_not_original_pitch"] is True
    assert (
        feasibility["gates"]["nvidia_patient_conditioned_counterfactual"]["status"]
        == "fail"
    )
    assert (
        feasibility["gates"]["nvidia_patient_conditioned_counterfactual"][
            "model_card_supports_patient_conditioned_counterfactual"
        ]
        is False
    )
    assert (
        feasibility["claim_boundaries"][
            "nvidia_model_card_supports_patient_conditioned_counterfactual"
        ]
        is False
    )


def test_paper_method_feasibility_blocks_control_prior_candidate_fallback(tmp_path):
    """Control-prior candidate fallback can be inspectable while not paper-ready."""
    ds = _fake_dataset(tmp_path, [
        {"id": "sub-00001", "group": "fcd", "has_roi": True},
        {"id": "sub-00002", "group": "hc", "has_roi": False},
    ])
    out = tmp_path / "out"

    with patch(
        "fcd_prototype.end_to_end.generate_candidate_rois",
        _fake_generate_candidate_rois,
    ):
        with patch(
            "fcd_prototype.splat_reconstruction_audit.audit_splat_reconstruction",
            _fake_splat_reconstruction_audit,
        ):
            with patch("fcd_prototype.gaussian_field.train_gaussian_field", _fake_train_gaussian_field):
                with patch(
                    "fcd_prototype.gaussian_field.project_volume_to_gaussian_support",
                    _fake_project_gaussian_support,
                ):
                    with patch(
                        "fcd_prototype.gaussian_field_audit.audit_gaussian_field_fidelity",
                        _fake_optimized_gaussian_field_audit,
                    ):
                        with patch(
                            "fcd_prototype.gaussian_residual.compare_gaussian_fields",
                            _fake_compare_gaussian_fields,
                        ):
                            main([
                                "--dataset-dir", str(ds),
                                "--output-dir", str(out),
                                "--subjects", "sub-00001",
                                "--synthetic-mode", "control-prior",
                                "--roi-strategy", "auto",
                                "--paper-3dgs-mode",
                                "--skip-paired-gaussian-residual",
                                "--force",
                            ])

    manifest = json.loads((out / "sub-00001" / "manifest.json").read_text())
    assert manifest["product_readiness"]["overall_status"] == "not_ready"
    assert (
        manifest["product_readiness"]["gates"]["roi_localization_evidence"]["status"]
        == "fail"
    )

    feasibility = manifest["paper_method_feasibility"]
    assert feasibility["overall_status"] == "not_feasible_for_original_pitch"
    assert feasibility["fallback_status"] == "fallback_research_workflow_possible"
    assert "nvidia_patient_conditioned_counterfactual" in feasibility["blocking_gates"]
    assert (
        feasibility["gates"]["nvidia_patient_conditioned_counterfactual"][
            "synthetic_normal_source"
        ]
        == "control_prior_roi_hybrid"
    )


def test_paper_method_feasibility_can_pass_future_patient_conditioned_nvidia():
    """Future patient-conditioned NVIDIA support should be the explicit pass path."""
    manifest = {
        "group": "fcd",
        "roi_source": "supervised",
        "synthetic_normal_source": "nvidia_patient_conditioned_inpainting",
        "product_readiness": {"overall_status": "ready"},
        "artifacts": {
            "patient_flair": "patient_flair.nii.gz",
            "patient_splat_dense_ply": "patient_splat_dense.ply",
            "optimized_gaussian_splat_ply": "optimized_gaussian_splat.ply",
            "gaussian_field_checkpoint": "gaussian_field.pt",
            "synthetic_normal": "synthetic_normal.nii.gz",
            "fuzzed_patient_flair": "fuzzed_patient_flair.nii.gz",
            "gaussian_residual_splat": "gaussian_residual/gaussian_residual_splat.ply",
            "gaussian_residual_heatmap": "gaussian_residual/gaussian_residual_heatmap.nii.gz",
            "gaussian_residual_metrics": "gaussian_residual/gaussian_residual_metrics.json",
            "gaussian_residual_scores": "gaussian_residual/gaussian_residual_scores.npy",
        },
        "stage_status": {
            "gaussian_field": "ok",
            "lesion_fuzzing": "ok",
        },
        "supervised_roi_meta": {
            "accepted": True,
            "operating_point": _passing_supervised_roi_operating_point_meta(),
        },
        "supervised_roi_operating_point_meta": (
            _passing_supervised_roi_operating_point_meta()
        ),
        "data_quality": {
            "upload_readiness": {
                "status": "pass",
                "blocking_issue_count": 0,
                "can_run_3dgs": True,
                "can_subtract_patient_normal": True,
            },
        },
        "splat_reconstruction_audit_meta": {
            "mri_to_3dgs_claim_allowed": True,
            "mri_to_3dgs_readiness_status": "pass",
            "foreground_coverage": 0.8,
            "mip_mean_mae": 0.04,
        },
        "gaussian_field_meta": {
            "backend": "gsplat",
            "training_objective": "mip_projection_mse_plus_volume_sample_mse",
            "volume_loss_weight": 0.25,
            "volume_sample_count": 8192,
        },
        "optimized_gaussian_field_audit_meta": _passing_optimized_gaussian_field_audit_meta(),
        "synthetic_normal_audit_meta": {
            "patient_conditioned_inpainting": True,
            "patient_conditioned_counterfactual_claim_allowed": True,
            "claim_boundary": "patient_conditioned_inpainting",
        },
        "nvidia_preflight_meta": {
            "ready_for_inference": True,
            "ready_for_patient_conditioned_counterfactual": True,
            "patient_conditioned_inpainting_supported": True,
            "paper_claim_readiness": {
                "claim_boundary": "patient_conditioned_counterfactual",
                "model_card_supports_original_pitch": True,
                "model_card_supports_patient_conditioned_counterfactual": True,
                "model_card_supports_fcd_mask_guided_generation": True,
                "released_workflow_accepts_patient_mri_or_roi_mask": True,
                "paper_claim_requires_new_conditioning_method": False,
            },
        },
        "gaussian_residual_meta": {
            "same_gaussian_support": True,
            "support_fingerprint_match": True,
            "gaussian_subtraction_claim_allowed": True,
            "residual_score_method": "indexwise_same_support_intensity",
            "heatmap_method": "field",
        },
    }

    feasibility = build_paper_method_feasibility(manifest)

    assert feasibility["overall_status"] == "feasible_for_original_pitch"
    assert feasibility["blocking_gates"] == []
    assert feasibility["claim_boundaries"]["original_pitch_feasible"] is True
    assert (
        feasibility["gates"]["nvidia_patient_conditioned_counterfactual"]["status"]
        == "pass"
    )
    assert (
        feasibility["claim_boundaries"][
            "nvidia_released_workflow_accepts_patient_mri_or_roi_mask"
        ]
        is True
    )


def test_product_readiness_accepts_gsplat_backend():
    """Production CUDA/gsplat Gaussian fields should pass readiness."""
    manifest = {
        "group": "fcd",
        "roi_source": "supervised",
        "artifacts": {
            "patient_flair": "patient_flair.nii.gz",
            "patient_splat_dense_ply": "patient_splat_dense.ply",
            "optimized_gaussian_splat_ply": "optimized_gaussian_splat.ply",
            "gaussian_field_checkpoint": "gaussian_field.pt",
            "synthetic_normal": "synthetic_normal.nii.gz",
            "gaussian_residual_splat": "gaussian_residual/gaussian_residual_splat.ply",
            "gaussian_residual_heatmap": "gaussian_residual/gaussian_residual_heatmap.nii.gz",
            "gaussian_residual_metrics": "gaussian_residual/gaussian_residual_metrics.json",
            "gaussian_residual_scores": "gaussian_residual/gaussian_residual_scores.npy",
            "residual_candidate_mask": "residual_candidates/residual_candidate_mask.nii.gz",
        },
        "stage_status": {"gaussian_field": "ok"},
        "supervised_roi_meta": {
            "accepted": True,
            "operating_point": _passing_supervised_roi_operating_point_meta(),
        },
        "supervised_roi_operating_point_meta": (
            _passing_supervised_roi_operating_point_meta()
        ),
        "data_quality": {
            "upload_readiness": {
                "status": "pass",
                "blocking_issue_count": 0,
                "review_issue_count": 0,
                "can_run_3dgs": True,
                "can_subtract_patient_normal": True,
                "blockers": [],
            },
        },
        "splat_reconstruction_audit_meta": {
            "mri_to_3dgs_claim_allowed": True,
        },
        "gaussian_field_meta": {"backend": "gsplat"},
        "optimized_gaussian_field_audit_meta": _passing_optimized_gaussian_field_audit_meta(),
        "synthetic_normal_source": "nvidia_rflow-mr-brain_roi_hybrid",
        "synthetic_normal_audit_meta": {
            "subtraction_readiness_status": "pass",
            "gaussian_subtraction_claim_allowed": True,
        },
        "gaussian_residual_meta": {
            "same_gaussian_support": True,
            "support_fingerprint_match": True,
            "patient_support_fingerprint_sha256": "a" * 64,
            "normal_support_fingerprint_sha256": "a" * 64,
            "gaussian_subtraction_claim_allowed": True,
            "residual_score_method": "indexwise_same_support_intensity",
            "heatmap_method": "field",
        },
    }

    readiness = build_product_readiness(manifest)

    assert readiness["overall_status"] == "ready"
    assert readiness["gates"]["optimized_gaussian_field"]["status"] == "pass"
    assert readiness["gates"]["roi_localization_evidence"]["status"] == "pass"
    assert readiness["gates"]["optimized_gaussian_field"]["gaussian_field_backend"] == "gsplat"
    residual_gate = readiness["gates"]["optimized_gaussian_subtraction"]
    assert residual_gate["support_fingerprint_match"] is True
    assert residual_gate["patient_support_fingerprint_sha256"] == "a" * 64
    assert not any("backend=gsplat" in item for item in readiness["blockers"])


def test_product_readiness_requires_optimized_gaussian_field_audit():
    """Optimized field artifacts do not pass without held-out fidelity evidence."""
    manifest = {
        "artifacts": {
            "patient_splat_dense_ply": "patient_splat_dense.ply",
            "optimized_gaussian_splat_ply": "optimized_gaussian_splat.ply",
            "gaussian_field_checkpoint": "gaussian_field.pt",
        },
        "stage_status": {"gaussian_field": "ok"},
        "data_quality": {
            "upload_readiness": {
                "status": "pass",
                "blocking_issue_count": 0,
                "can_run_3dgs": True,
                "can_subtract_patient_normal": True,
            },
        },
        "splat_reconstruction_audit_meta": {
            "mri_to_3dgs_claim_allowed": True,
        },
        "gaussian_field_meta": {"backend": "gsplat"},
    }

    readiness = build_product_readiness(manifest)

    gate = readiness["gates"]["optimized_gaussian_field"]
    assert gate["status"] == "fail"
    assert gate["optimized_gaussian_field_audit_status"] == "missing"
    assert not gate["optimized_mri_to_3dgs_claim_allowed"]
    assert "optimized_gaussian_field_fidelity_not_proven" in readiness["blockers"]


def test_paper_method_feasibility_requires_passing_optimized_field_audit():
    """Original pitch cannot pass optimized MRI-native 3DGS without the audit gate."""
    audit_meta = json.loads(json.dumps(_passing_optimized_gaussian_field_audit_meta()))
    audit_meta["readiness_gate"]["status"] = "fail"
    audit_meta["readiness_gate"]["optimized_mri_to_3dgs_claim_allowed"] = False
    audit_meta["readiness_gate"]["blockers"] = ["heldout_error_too_high"]
    manifest = {
        "artifacts": {
            "patient_splat_dense_ply": "patient_splat_dense.ply",
            "optimized_gaussian_splat_ply": "optimized_gaussian_splat.ply",
            "gaussian_field_checkpoint": "gaussian_field.pt",
        },
        "stage_status": {"gaussian_field": "ok"},
        "data_quality": {
            "upload_readiness": {
                "status": "pass",
                "blocking_issue_count": 0,
                "can_run_3dgs": True,
                "can_subtract_patient_normal": True,
            },
        },
        "splat_reconstruction_audit_meta": {
            "mri_to_3dgs_claim_allowed": True,
        },
        "gaussian_field_meta": {
            "backend": "gsplat",
            "training_objective": "mip_projection_mse_plus_volume_sample_mse",
            "volume_loss_weight": 0.25,
            "volume_sample_count": 8192,
        },
        "optimized_gaussian_field_audit_meta": audit_meta,
    }

    feasibility = build_paper_method_feasibility(manifest)

    gate = feasibility["gates"]["optimized_mri_native_3dgs"]
    assert gate["status"] == "fail"
    assert gate["optimized_gaussian_field_audit_status"] == "fail"
    assert not gate["optimized_mri_to_3dgs_claim_allowed"]
    assert "optimized_mri_native_3dgs" in feasibility["blocking_gates"]
