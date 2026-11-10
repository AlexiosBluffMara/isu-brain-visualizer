"""Tests for synthetic-normal geometry and claim audit."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from fcd_prototype.synthetic_normal_audit import audit_synthetic_normal


def _save(path, data, affine=None):
    if affine is None:
        affine = np.diag([1.0, 1.0, 1.5, 1.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), str(path))


def test_audit_classifies_nvidia_as_not_patient_conditioned(tmp_path):
    rng = np.random.default_rng(7)
    patient = rng.random((20, 20, 16), dtype=np.float32)
    roi = np.zeros_like(patient, dtype=np.float32)
    roi[8:12, 8:12, 6:10] = 1.0
    synthetic = patient.copy()
    synthetic[roi > 0] = 0.25

    patient_path = tmp_path / "patient.nii.gz"
    synth_path = tmp_path / "synthetic_normal.nii.gz"
    roi_path = tmp_path / "roi.nii.gz"
    out = tmp_path / "audit.json"
    _save(patient_path, patient)
    _save(synth_path, synthetic)
    _save(roi_path, roi)

    audit = audit_synthetic_normal(
        patient_volume=patient_path,
        synthetic_normal=synth_path,
        roi_mask=roi_path,
        output_json=out,
        synthetic_normal_source="nvidia_rflow-mr-brain_roi_hybrid",
        source_metadata={
            "nvidia": {
                "nvidia_steps": 30,
                "nvidia_generated_shape_matches_config": True,
            }
        },
    )

    assert out.exists()
    assert audit["counterfactual_class"] == "unconditional_nvidia_prior_roi_hybrid"
    assert audit["patient_conditioned_inpainting"] is False
    assert audit["claim_boundary"] == "not_patient_conditioned_inpainting"
    assert audit["subtraction_interpretation"] == "roi_hybrid_same_patient_grid_residual"
    gate = audit["subtraction_readiness_gate"]
    assert gate["status"] == "pass"
    assert gate["gaussian_subtraction_claim_allowed"] is True
    assert gate["patient_conditioned_counterfactual_claim_allowed"] is False
    assert gate["claim_class"] == "research_prior_roi_hybrid_not_patient_conditioned"
    assert gate["nvidia_steps"] == 30
    assert gate["nvidia_min_inference_steps"] == 30
    assert gate["nvidia_generated_shape_matches_config"] is True
    assert audit["outside_roi"]["mae"] == pytest.approx(0.0)
    assert audit["inside_roi"]["mae"] > 0.0


def test_audit_fails_nvidia_smoke_with_too_few_steps(tmp_path):
    patient = np.full((20, 20, 16), 0.5, dtype=np.float32)
    roi = np.zeros_like(patient, dtype=np.float32)
    roi[8:12, 8:12, 6:10] = 1.0
    synthetic = patient.copy()
    synthetic[roi > 0] = 0.25

    patient_path = tmp_path / "patient.nii.gz"
    synth_path = tmp_path / "synthetic_normal.nii.gz"
    roi_path = tmp_path / "roi.nii.gz"
    _save(patient_path, patient)
    _save(synth_path, synthetic)
    _save(roi_path, roi)

    audit = audit_synthetic_normal(
        patient_volume=patient_path,
        synthetic_normal=synth_path,
        roi_mask=roi_path,
        synthetic_normal_source="nvidia_rflow-mr-brain_roi_hybrid",
        source_metadata={
            "nvidia": {
                "nvidia_steps": 1,
                "nvidia_generated_shape_matches_config": True,
            }
        },
    )

    gate = audit["subtraction_readiness_gate"]
    assert gate["status"] == "fail"
    assert gate["gaussian_subtraction_claim_allowed"] is False
    assert gate["nvidia_steps"] == 1
    assert any("below the required minimum" in blocker for blocker in gate["blockers"])


def test_audit_fails_nvidia_when_steps_missing(tmp_path):
    patient = np.full((20, 20, 16), 0.5, dtype=np.float32)
    roi = np.zeros_like(patient, dtype=np.float32)
    roi[8:12, 8:12, 6:10] = 1.0
    synthetic = patient.copy()
    synthetic[roi > 0] = 0.25

    patient_path = tmp_path / "patient.nii.gz"
    synth_path = tmp_path / "synthetic_normal.nii.gz"
    roi_path = tmp_path / "roi.nii.gz"
    _save(patient_path, patient)
    _save(synth_path, synthetic)
    _save(roi_path, roi)

    audit = audit_synthetic_normal(
        patient_volume=patient_path,
        synthetic_normal=synth_path,
        roi_mask=roi_path,
        synthetic_normal_source="nvidia_rflow-mr-brain_roi_hybrid",
        source_metadata={
            "nvidia": {
                "nvidia_generated_shape_matches_config": True,
            }
        },
    )

    gate = audit["subtraction_readiness_gate"]
    assert gate["status"] == "fail"
    assert gate["gaussian_subtraction_claim_allowed"] is False
    assert gate["nvidia_steps"] is None
    assert any("missing nvidia_steps" in blocker for blocker in gate["blockers"])


def test_audit_fails_nvidia_when_generated_shape_mismatch(tmp_path):
    patient = np.full((20, 20, 16), 0.5, dtype=np.float32)
    roi = np.zeros_like(patient, dtype=np.float32)
    roi[8:12, 8:12, 6:10] = 1.0
    synthetic = patient.copy()
    synthetic[roi > 0] = 0.25

    patient_path = tmp_path / "patient.nii.gz"
    synth_path = tmp_path / "synthetic_normal.nii.gz"
    roi_path = tmp_path / "roi.nii.gz"
    _save(patient_path, patient)
    _save(synth_path, synthetic)
    _save(roi_path, roi)

    audit = audit_synthetic_normal(
        patient_volume=patient_path,
        synthetic_normal=synth_path,
        roi_mask=roi_path,
        synthetic_normal_source="nvidia_rflow-mr-brain_roi_hybrid",
        source_metadata={
            "nvidia_steps": 30,
            "nvidia_generated_shape_matches_config": False,
        },
    )

    gate = audit["subtraction_readiness_gate"]
    assert gate["status"] == "fail"
    assert gate["gaussian_subtraction_claim_allowed"] is False
    assert gate["nvidia_steps"] == 30
    assert gate["nvidia_generated_shape_matches_config"] is False
    assert any("shape_matches_config" in blocker for blocker in gate["blockers"])


def test_audit_resamples_synthetic_to_patient_grid(tmp_path):
    patient = np.ones((16, 16, 12), dtype=np.float32) * 0.5
    synthetic = np.ones((8, 8, 6), dtype=np.float32) * 0.4
    patient_affine = np.diag([1.0, 1.0, 1.5, 1.0])
    synthetic_affine = np.diag([2.0, 2.0, 3.0, 1.0])

    patient_path = tmp_path / "patient.nii.gz"
    synth_path = tmp_path / "synthetic_normal.nii.gz"
    _save(patient_path, patient, patient_affine)
    _save(synth_path, synthetic, synthetic_affine)

    audit = audit_synthetic_normal(
        patient_volume=patient_path,
        synthetic_normal=synth_path,
        synthetic_normal_source="control_prior_whole_brain_prior",
    )

    assert audit["same_grid_before_resample"] is False
    assert audit["synthetic_resampled_to_patient_grid"] is True
    assert audit["outside_roi"]["voxel_count"] == patient.size
    assert audit["outside_roi"]["mae"] == pytest.approx(0.1)
    assert audit["subtraction_readiness_gate"]["status"] == "review"


def test_audit_fails_patient_copy_degenerate_reference(tmp_path):
    patient = np.ones((16, 16, 12), dtype=np.float32) * 0.5
    patient_path = tmp_path / "patient.nii.gz"
    synth_path = tmp_path / "synthetic_normal.nii.gz"
    _save(patient_path, patient)
    _save(synth_path, patient)

    audit = audit_synthetic_normal(
        patient_volume=patient_path,
        synthetic_normal=synth_path,
        synthetic_normal_source="control_prior_no_roi_patient_copy",
    )

    gate = audit["subtraction_readiness_gate"]
    assert gate["status"] == "fail"
    assert gate["gaussian_subtraction_claim_allowed"] is False
    assert any("patient-copy" in blocker for blocker in gate["blockers"])
