"""Tests for NVIDIA MR-Brain readiness preflight."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from fcd_prototype.nvidia_preflight import preflight_nvidia_mr_brain


def _make_volume(path: Path, shape=(52, 80, 80), affine=None) -> Path:
    if affine is None:
        affine = np.diag([3.2, 3.2, 3.2, 1.0])
    data = np.zeros(shape, dtype=np.float32)
    data[20:-20, 20:-20, 20:-20] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def _make_roi(path: Path, shape=(52, 80, 80), affine=None, positive=True) -> Path:
    if affine is None:
        affine = np.diag([3.2, 3.2, 3.2, 1.0])
    data = np.zeros(shape, dtype=np.uint8)
    if positive:
        data[23:29, 37:43, 37:43] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def _make_fake_nvidia_root(root: Path, *, with_weights: bool = True) -> Path:
    for rel in (
        "scripts/diff_model_infer.py",
        "configs/config_network_rflow.json",
        "configs/environment_maisi_diff_model_rflow-mr-brain.json",
        "configs/config_maisi_diff_model_rflow-mr-brain.json",
    ):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    if with_weights:
        for rel in (
            "models/autoencoder_v1.pt",
            "models/diff_unet_3d_rflow-mr-brain_v0.pt",
        ):
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"fake")
    return root


def test_preflight_reports_ready_with_fake_complete_root(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_MAISI_DIM", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_SPACING", raising=False)
    root = _make_fake_nvidia_root(tmp_path / "nvidia")
    vol = _make_volume(tmp_path / "patient.nii.gz")
    roi = _make_roi(tmp_path / "roi.nii.gz")

    report = preflight_nvidia_mr_brain(
        patient_volume=vol,
        roi_mask=roi,
        modality_hint="FLAIR",
        nvidia_root=root,
    )

    assert report["ready_for_inference"] is True
    assert report["root"]["available"] is True
    assert report["modality_code"] == 11
    assert report["patient"]["inferred_plane"] == "sagittal"
    assert report["generation_dim"] == [128, 256, 256]
    assert report["recommended_fov_mm"] == [176.0, 250.0, 250.0]
    assert report["geometry_checks"]["dim_divisible_by_16"] is True
    assert report["geometry_checks"]["dim_supported_by_documented_constraints"] is True
    assert report["geometry_checks"]["dim_preferred_layout"] is True
    assert report["geometry_checks"]["spacing_within_documented_minimum"] is True
    assert report["supported_modalities_documented"] == ["T1", "FLAIR", "T2", "SWI"]
    assert report["multi_modality_generation_supported"] is True
    assert report["cross_modality_synthesis_supported"] is False
    assert report["cross_modality_synthesis_status"] == "coming_soon_on_model_card"
    assert report["patient_mri_conditioning_supported"] is False
    assert report["lesion_mask_inpainting_supported"] is False
    assert report["patient_conditioned_inpainting_supported"] is False
    assert report["ready_for_patient_conditioned_counterfactual"] is False
    assert report["capability_boundary"] == "image_only_unconditional_generation_plus_roi_hybrid"
    assert report["model_card_url"] == "https://huggingface.co/nvidia/NV-Generate-MR-Brain"
    assert report["model_card_task"] == "unconditional-image-generation"
    assert report["fcd_mask_guided_generation_supported"] is False
    assert report["clinical_diagnostic_use_allowed"] is False
    assert report["paper_claim_requires_new_conditioning_method"] is True
    assert report["model_card_contract"]["input_parameters_documented"] == [
        "number_of_output_samples",
        "modality_label",
        "output_size",
        "spacing",
    ]
    assert report["model_card_contract"]["model_card_task"] == (
        "unconditional-image-generation"
    )
    assert report["model_card_contract"]["training_dataset_documented"] == (
        "Forithmus/MR-RATE"
    )
    assert report["model_card_contract"]["training_scan_count_approx"] == 28_000
    assert report["model_card_contract"]["testing_scan_count_approx"] == 8_000
    assert report["model_card_contract"]["evaluation_scan_count_approx"] == 4_000
    assert report["model_card_contract"]["hf_inference_provider_deployed"] is False
    assert report["model_card_contract"]["local_repo_weights_required"] is True
    assert report["model_card_contract"]["clinical_validation_documented"] is False
    assert report["model_card_contract"]["clinical_diagnostic_use_allowed"] is False
    assert (
        report["model_card_contract"]["cross_modality_synthesis_documented_status"]
        == "coming_soon"
    )
    assert report["model_card_contract"]["patient_mri_input_documented"] is False
    assert report["model_card_contract"]["roi_mask_input_documented"] is False
    assert (
        report["model_card_contract"]["patient_conditioned_counterfactual_documented"]
        is False
    )
    assert (
        report["model_card_contract"]["fcd_mask_guided_generation_documented"]
        is False
    )
    assert report["local_adapter_contract"]["uses_official_diff_model_infer"] is True
    assert report["local_adapter_contract"]["passes_patient_mri_to_nvidia_model"] is False
    assert report["local_adapter_contract"]["passes_roi_mask_to_nvidia_model"] is False
    assert (
        report["local_adapter_contract"]["released_workflow_accepts_patient_mri_or_roi_mask"]
        is False
    )
    claim_gate = report["paper_claim_readiness"]
    assert claim_gate["nvidia_inference_claim_allowed"] is True
    assert claim_gate["population_prior_residual_claim_allowed"] is True
    assert claim_gate["model_card_task"] == "unconditional-image-generation"
    assert claim_gate["model_card_supports_original_pitch"] is False
    assert claim_gate["model_card_supports_patient_conditioned_counterfactual"] is False
    assert claim_gate["model_card_supports_fcd_mask_guided_generation"] is False
    assert claim_gate["released_workflow_accepts_patient_mri_or_roi_mask"] is False
    assert claim_gate["paper_claim_requires_new_conditioning_method"] is True
    assert claim_gate["patient_specific_healthy_counterfactual_claim_allowed"] is False
    assert claim_gate["patient_conditioned_lesion_inpainting_claim_allowed"] is False
    assert claim_gate["fcd_mask_guided_generation_claim_allowed"] is False
    assert "parameter-conditioned synthetic MRI generation" in (
        claim_gate["patient_conditioned_blocker_reason"]
    )
    assert claim_gate["claim_boundary"] == (
        "nvidia_image_only_prior_not_patient_conditioned_counterfactual"
    )
    assert report["roi"]["positive_voxels"] > 0
    assert not report["blockers"]
    assert not report["warnings"]


def test_preflight_blocks_missing_weights(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_MAISI_DIM", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_SPACING", raising=False)
    root = _make_fake_nvidia_root(tmp_path / "nvidia", with_weights=False)
    vol = _make_volume(tmp_path / "patient.nii.gz")

    report = preflight_nvidia_mr_brain(
        patient_volume=vol,
        modality_hint="FLAIR",
        nvidia_root=root,
    )

    assert report["ready_for_inference"] is False
    assert report["root"]["available"] is False
    assert "models/autoencoder_v1.pt" in report["root"]["missing_files"]
    assert any("repo/weights" in blocker for blocker in report["blockers"])


def test_preflight_blocks_invalid_modality(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_MAISI_DIM", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_SPACING", raising=False)
    root = _make_fake_nvidia_root(tmp_path / "nvidia")
    vol = _make_volume(tmp_path / "patient.nii.gz")

    report = preflight_nvidia_mr_brain(
        patient_volume=vol,
        modality_hint="DWI",
        nvidia_root=root,
    )

    assert report["modality_supported"] is False
    assert report["ready_for_inference"] is False
    assert any("Unsupported modality" in blocker for blocker in report["blockers"])


def test_preflight_blocks_bad_env_geometry(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_MAISI_DIM", "130,256,256")
    monkeypatch.setenv("NVIDIA_MAISI_SPACING", "0.3,1,1")
    root = _make_fake_nvidia_root(tmp_path / "nvidia")
    vol = _make_volume(tmp_path / "patient.nii.gz")

    report = preflight_nvidia_mr_brain(
        patient_volume=vol,
        modality_hint="FLAIR",
        nvidia_root=root,
    )

    assert report["ready_for_inference"] is False
    assert report["geometry_checks"]["dim_divisible_by_16"] is False
    assert report["geometry_checks"]["dim_supported_by_documented_constraints"] is False
    assert report["geometry_checks"]["dim_preferred_layout"] is False
    assert report["geometry_checks"]["spacing_within_documented_minimum"] is False
    assert any("dim/spacing" in blocker for blocker in report["blockers"])


def test_preflight_warns_on_empty_roi(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_MAISI_DIM", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_SPACING", raising=False)
    root = _make_fake_nvidia_root(tmp_path / "nvidia")
    vol = _make_volume(tmp_path / "patient.nii.gz")
    roi = _make_roi(tmp_path / "roi.nii.gz", positive=False)

    report = preflight_nvidia_mr_brain(
        patient_volume=vol,
        roi_mask=roi,
        modality_hint="FLAIR",
        nvidia_root=root,
    )

    assert report["ready_for_inference"] is True
    assert report["roi"]["positive_voxels"] == 0
    assert any("contains no positive voxels" in warning for warning in report["warnings"])


def test_preflight_report_is_json_serializable(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_MAISI_DIM", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_SPACING", raising=False)
    root = _make_fake_nvidia_root(tmp_path / "nvidia")
    vol = _make_volume(tmp_path / "patient.nii.gz")

    report = preflight_nvidia_mr_brain(
        patient_volume=vol,
        modality_hint="T1w",
        nvidia_root=root,
    )

    encoded = json.dumps(report)
    assert "nvidia_mr_brain_preflight" in encoded


def test_preflight_allows_documented_max_geometry(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_MAISI_DIM", "512,512,256")
    monkeypatch.setenv("NVIDIA_MAISI_SPACING", "0.5,0.5,0.8")
    root = _make_fake_nvidia_root(tmp_path / "nvidia")
    vol = _make_volume(tmp_path / "patient.nii.gz")

    report = preflight_nvidia_mr_brain(
        patient_volume=vol,
        modality_hint="SWI",
        nvidia_root=root,
    )

    assert report["ready_for_inference"] is True
    assert report["modality_supported"] is True
    assert report["modality_code"] == 20
    assert report["generation_dim"] == [512, 512, 256]
    assert report["geometry_checks"]["dim_supported_by_documented_constraints"] is True
    assert report["geometry_checks"]["dim_preferred_layout"] is True
