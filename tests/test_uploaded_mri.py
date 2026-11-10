"""Tests for single-scan uploaded MRI intake."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import fcd_prototype.supervised_lesion_proposal as slp
import fcd_prototype.uploaded_mri as uploaded_mri
from fcd_prototype.uploaded_mri import (
    DicomConversionError,
    build_upload_input_compatibility,
    build_uploaded_mri,
    main,
)


def _make_volume(path: Path, shape=(20, 20, 16), affine=None, seed=0) -> None:
    if affine is None:
        affine = np.diag([1.0, 1.0, 1.5, 1.0])
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal(shape).astype(np.float32) * 0.2 + 0.6).clip(0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _make_roi(path: Path, shape=(20, 20, 16), affine=None) -> None:
    if affine is None:
        affine = np.diag([1.0, 1.0, 1.5, 1.0])
    data = np.zeros(shape, dtype=np.uint8)
    data[6:10, 6:10, 6:10] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _fake_control_dataset(tmp_path: Path) -> Path:
    ds = tmp_path / "controls"
    ds.mkdir(parents=True, exist_ok=True)
    (ds / "participants.tsv").write_text(
        "participant_id\tgroup\n"
        "sub-ctrl01\thc\n"
        "sub-ctrl02\thc\n"
    )
    for idx, sub in enumerate(("sub-ctrl01", "sub-ctrl02"), start=1):
        anat = ds / sub / "anat"
        _make_volume(anat / f"{sub}_FLAIR.nii.gz", seed=idx)
        _make_volume(anat / f"{sub}_T2w.nii.gz", seed=idx + 20)
        _make_volume(anat / f"{sub}_T1w.nii.gz", seed=idx + 10)
    return ds


def _fake_predict_uploaded_roi(
    checkpoint_path, volume_path, output_path,
    threshold=0.5, min_component_voxels=0, device="auto",
    selection_mode="threshold", foreground_fraction=None,
):
    shape = (20, 20, 16)
    data = np.zeros(shape, dtype=np.float32)
    data[8:12, 8:12, 6:10] = 1.0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.5, 1.0])),
        str(output_path),
    )
    voxels = int(data.sum())
    return {
        "predicted_voxels": voxels,
        "predicted_foreground_fraction": voxels / data.size,
        "probability_mean": 0.7,
        "probability_max": 0.95,
        "threshold": threshold,
        "selection_mode": selection_mode,
        "foreground_fraction": foreground_fraction,
        "spatial_size": (64, 64, 64),
    }


def _fake_train_gaussian_field(
    volume_path, output_dir, roi_path=None, num_gaussians=20000,
    steps=300, image_size=128, views="all", device="auto", seed=42,
    learning_rate=1e-3, preview_every=0, volume_loss_weight=0.0,
    volume_sample_count=4096,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "optimized_gaussian_splat.ply").write_bytes(b"PLY\n")
    (output_dir / "gaussian_field.pt").write_bytes(b"PT\n")
    (output_dir / "reconstruction_preview.png").write_bytes(b"PNG\n")
    if roi_path is not None and Path(roi_path).is_file():
        (output_dir / "lesion_gaussian_mask.npy").write_bytes(b"NPY\n")
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
        "num_gaussians": num_gaussians,
        "steps": steps,
        "support_fingerprint_schema": "gaussian_support_v1",
        "support_fingerprint_sha256": "b" * 64,
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics))
    return metrics


def _fake_project_gaussian_support(
    support_field, volume_path, output_dir, preview_image_size=64, copy_opacity=True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "optimized_gaussian_splat.ply").write_bytes(b"PLY\n")
    (output_dir / "gaussian_field.pt").write_bytes(b"PT\n")
    (output_dir / "reconstruction_preview.png").write_bytes(b"PNG\n")
    metrics = {
        "backend": "support_projection",
        "device": "cpu",
        "training_objective": "projected_volume_on_existing_gaussian_support",
        "final_loss": None,
        "final_mip_loss": None,
        "final_volume_loss": None,
        "support_projection": True,
        "same_gaussian_support_as": str(support_field),
        "same_gaussian_support_as_fingerprint": "b" * 64,
        "support_fingerprint_schema": "gaussian_support_v1",
        "support_fingerprint_sha256": "b" * 64,
        "support_points_in_volume_bounds_fraction": 1.0,
        "num_gaussians": 30,
        "steps": 0,
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics))
    return metrics


def _fake_compare_gaussian_fields(
    patient_field, normal_field, reference_volume, output_dir, roi=None,
    k_neighbors=4, max_points=200_000, heatmap_percentile=99.0,
    heatmap_method="field", field_chunk_size=128,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gaussian_residual_splat.ply").write_bytes(b"PLY\n")
    (output_dir / "gaussian_residual_heatmap.nii.gz").write_bytes(b"GZIP\n")
    (output_dir / "gaussian_residual_scores.npy").write_bytes(b"NPY\n")
    metrics = {
        "k_neighbors": k_neighbors,
        "n_patient": 30,
        "n_normal": 30,
        "heatmap_method": heatmap_method,
        "residual_score_method": "indexwise_same_support_intensity",
        "same_gaussian_support": True,
        "support_fingerprint_schema": "gaussian_support_v1",
        "support_fingerprint_match": True,
        "patient_support_fingerprint_sha256": "b" * 64,
        "normal_support_fingerprint_sha256": "b" * 64,
        "gaussian_subtraction_claim_allowed": True,
        "residual_mean": 0.05,
        "residual_p95": 0.15,
        "residual_p99": 0.2,
        "field_residual_mean": 0.04,
        "field_residual_p95": 0.12,
        "field_residual_p99": 0.18,
        "field_chunk_size": field_chunk_size,
        "roi_metrics_valid": False,
        "roi_to_non_roi_ratio": 0.0,
    }
    (output_dir / "gaussian_residual_metrics.json").write_text(json.dumps(metrics))
    return metrics


def test_uploaded_mri_requires_modality(tmp_path):
    with pytest.raises(ValueError, match="At least one"):
        build_uploaded_mri(
            flair=None,
            t1w=None,
            roi=None,
            output_root=tmp_path / "out",
        )


def test_uploaded_mri_rejects_non_nifti(tmp_path):
    txt = tmp_path / "scan.txt"
    txt.write_text("not a nifti")
    with pytest.raises(ValueError, match=".nii"):
        build_uploaded_mri(
            flair=txt,
            t1w=None,
            roi=None,
            output_root=tmp_path / "out",
        )


def test_uploaded_mri_rejects_nifti_and_dicom_for_same_modality(tmp_path):
    flair = tmp_path / "upload" / "flair.nii.gz"
    dicom_dir = tmp_path / "dicom-flair"
    _make_volume(flair)
    dicom_dir.mkdir()
    (dicom_dir / "IM0001.dcm").write_bytes(b"DICOM")

    with pytest.raises(ValueError, match="either --flair or --flair-dicom-dir"):
        build_uploaded_mri(
            flair=flair,
            flair_dicom_dir=dicom_dir,
            t1w=None,
            roi=None,
            output_root=tmp_path / "out",
        )


def test_uploaded_mri_rejects_auto_scan_with_explicit_modality(tmp_path):
    scan = tmp_path / "upload" / "patient_FLAIR.nii.gz"
    t1w = tmp_path / "upload" / "patient_T1w.nii.gz"
    _make_volume(scan)
    _make_volume(t1w)

    with pytest.raises(ValueError, match="cannot be combined"):
        build_uploaded_mri(
            scan=scan,
            flair=None,
            t2w=None,
            t1w=t1w,
            roi=None,
            output_root=tmp_path / "out",
        )


def test_upload_input_compatibility_accepts_labeled_nifti(tmp_path):
    flair = tmp_path / "patient_FLAIR.nii.gz"
    _make_volume(flair)

    report = build_upload_input_compatibility(
        flair=flair,
        t1w=None,
        roi=None,
    )

    assert report["status"] == "pass"
    assert report["can_attempt_processing"] is True
    assert report["can_support_paper_claim"] is True
    assert report["inputs"]["FLAIR"]["readable"] is True
    assert "flair" in report["inputs"]["FLAIR"]["filename_terms"]


def test_upload_input_compatibility_accepts_t2w_nifti(tmp_path):
    t2w = tmp_path / "patient_T2w.nii.gz"
    _make_volume(t2w)

    report = build_upload_input_compatibility(
        flair=None,
        t2w=t2w,
        t1w=None,
        roi=None,
    )

    assert report["status"] == "pass"
    assert report["primary_modalities"] == ["T2w"]
    assert report["inputs"]["T2w"]["readable"] is True
    assert "t2" in report["inputs"]["T2w"]["filename_terms"]


def test_upload_input_compatibility_auto_scan_infers_flair(tmp_path):
    scan = tmp_path / "patient_T2sel_FLAIR.nii.gz"
    _make_volume(scan)

    report = build_upload_input_compatibility(
        scan=scan,
        flair=None,
        t2w=None,
        t1w=None,
        roi=None,
    )

    assert report["status"] == "pass"
    assert report["auto_resolution"]["status"] == "pass"
    assert report["auto_resolution"]["inferred_label"] == "FLAIR"
    assert report["primary_modalities"] == ["FLAIR"]
    assert report["inputs"]["FLAIR"]["readable"] is True


def test_upload_input_compatibility_auto_scan_uses_nifti_sidecar(tmp_path):
    scan = tmp_path / "scan.nii.gz"
    sidecar = tmp_path / "scan.json"
    _make_volume(scan)
    sidecar.write_text(
        json.dumps(
            {
                "Modality": "MR",
                "SeriesDescription": "3D T2 FLAIR structural",
                "ProtocolName": "FLAIR",
            }
        ),
        encoding="utf-8",
    )

    report = build_upload_input_compatibility(
        scan=scan,
        flair=None,
        t2w=None,
        t1w=None,
        roi=None,
    )

    auto = report["auto_resolution"]
    assert auto["status"] == "pass"
    assert auto["inference_source"] == "sidecar"
    assert auto["inferred_label"] == "FLAIR"
    assert auto["sidecar_exists"] is True
    assert auto["sidecar_json_parse_ok"] is True
    assert auto["sidecar_path"] == str(sidecar)
    assert "flair" in auto["sidecar_terms"]
    assert report["primary_modalities"] == ["FLAIR"]
    assert report["can_attempt_processing"] is True


def test_upload_input_compatibility_auto_scan_rejects_filename_sidecar_conflict(tmp_path):
    scan = tmp_path / "patient_T1w.nii.gz"
    _make_volume(scan)
    (tmp_path / "patient_T1w.json").write_text(
        json.dumps(
            {
                "Modality": "MR",
                "SeriesDescription": "Coronal T2 FLAIR",
                "ProtocolName": "FLAIR",
            }
        ),
        encoding="utf-8",
    )

    report = build_upload_input_compatibility(
        scan=scan,
        flair=None,
        t2w=None,
        t1w=None,
        roi=None,
    )

    auto = report["auto_resolution"]
    assert report["status"] == "fail"
    assert auto["status"] == "fail"
    assert auto["filename_inferred_label"] == "T1w"
    assert auto["sidecar_inferred_label"] == "FLAIR"
    assert auto["inference_source"] == "conflict"
    assert "auto_scan:auto_scan_modality_conflict" in report["blockers"]


def test_upload_input_compatibility_auto_scan_ambiguous_requires_explicit(tmp_path):
    scan = tmp_path / "patient_structural_scan.nii.gz"
    _make_volume(scan)

    report = build_upload_input_compatibility(
        scan=scan,
        flair=None,
        t2w=None,
        t1w=None,
        roi=None,
    )

    assert report["status"] == "fail"
    assert "auto_scan:auto_scan_modality_not_inferred" in report["blockers"]
    assert report["auto_resolution"]["status"] == "fail"


def test_upload_input_compatibility_auto_scan_dicom_dir_infers_t2w(tmp_path, monkeypatch):
    dicom_dir = tmp_path / "patient_T2w_dicom"
    dicom_dir.mkdir()
    (dicom_dir / "IM0001.dcm").write_bytes(b"DICOM")
    monkeypatch.setattr(uploaded_mri.shutil, "which", lambda _name: "dcm2niix")

    report = build_upload_input_compatibility(
        scan_dicom_dir=dicom_dir,
        flair=None,
        t2w=None,
        t1w=None,
        roi=None,
    )

    assert report["status"] == "pass"
    assert report["auto_resolution"]["inferred_label"] == "T2w"
    assert report["inputs"]["T2w"]["requires_conversion"] is True


def test_upload_input_compatibility_flags_modality_label_mismatch(tmp_path):
    flair = tmp_path / "patient_T1w.nii.gz"
    _make_volume(flair)

    report = build_upload_input_compatibility(
        flair=flair,
        t1w=None,
        roi=None,
    )

    assert report["status"] == "review"
    assert report["can_attempt_processing"] is True
    assert "FLAIR:filename_expected_flair_term_not_detected" in report["review_items"]
    assert "FLAIR:filename_suggests_t1_for_flair_input" in report["review_items"]


def test_upload_input_compatibility_reviews_generic_filename(tmp_path):
    flair = tmp_path / "scan.nii.gz"
    _make_volume(flair)

    report = build_upload_input_compatibility(
        flair=flair,
        t1w=None,
        roi=None,
    )

    assert report["status"] == "review"
    assert "FLAIR:filename_expected_flair_term_not_detected" in report["review_items"]


def test_upload_input_compatibility_rejects_unsupported_file(tmp_path):
    txt = tmp_path / "patient_FLAIR.txt"
    txt.write_text("not a nifti", encoding="utf-8")

    report = build_upload_input_compatibility(
        flair=txt,
        t1w=None,
        roi=None,
    )

    assert report["status"] == "fail"
    assert report["can_attempt_processing"] is False
    assert "FLAIR:unsupported_file_type" in report["blockers"]


def test_upload_input_compatibility_reports_dicom_inventory(tmp_path, monkeypatch):
    dicom_dir = tmp_path / "dicom-flair"
    dicom_dir.mkdir()
    (dicom_dir / "IM0001.dcm").write_bytes(b"DICOM")
    monkeypatch.setattr(uploaded_mri.shutil, "which", lambda _name: "dcm2niix")

    report = build_upload_input_compatibility(
        flair=None,
        flair_dicom_dir=dicom_dir,
        t1w=None,
        roi=None,
    )

    flair_info = report["inputs"]["FLAIR"]
    assert report["status"] == "pass"
    assert report["can_attempt_processing"] is True
    assert report["can_support_paper_claim"] is False
    assert flair_info["requires_conversion"] is True
    assert flair_info["converter_available"] is True
    assert flair_info["input_inventory"]["file_count"] == 1
    assert flair_info["paper_claim_requires"].startswith("successful dcm2niix")


def test_upload_preflight_only_cli_writes_json_without_processing(tmp_path, capsys):
    flair = tmp_path / "upload" / "patient_FLAIR.nii.gz"
    out_json = tmp_path / "preflight" / "report.json"
    out_dir = tmp_path / "out"
    _make_volume(flair)

    main([
        "--flair", str(flair),
        "--output-dir", str(out_dir),
        "--preflight-only",
        "--preflight-output-json", str(out_json),
    ])

    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert printed["status"] == "pass"
    assert saved == printed
    assert not (out_dir / "index.json").exists()


def test_uploaded_mri_dicom_missing_converter_raises(tmp_path, monkeypatch):
    dicom_dir = tmp_path / "dicom-flair"
    dicom_dir.mkdir()
    (dicom_dir / "IM0001.dcm").write_bytes(b"DICOM")
    monkeypatch.setattr(uploaded_mri.shutil, "which", lambda _name: None)

    with pytest.raises(DicomConversionError, match="dcm2niix"):
        build_uploaded_mri(
            flair=None,
            flair_dicom_dir=dicom_dir,
            t1w=None,
            roi=None,
            output_root=tmp_path / "out",
            synthetic_mode="placeholder",
            roi_strategy="none",
        )


def test_uploaded_mri_dicom_empty_dir_raises(tmp_path, monkeypatch):
    dicom_dir = tmp_path / "empty-dicom"
    dicom_dir.mkdir()
    monkeypatch.setattr(uploaded_mri.shutil, "which", lambda _name: "dcm2niix")

    with pytest.raises(DicomConversionError, match="contains no files"):
        build_uploaded_mri(
            flair=None,
            flair_dicom_dir=dicom_dir,
            t1w=None,
            roi=None,
            output_root=tmp_path / "out",
            synthetic_mode="placeholder",
            roi_strategy="none",
        )


def test_uploaded_mri_dicom_converter_failure_raises(tmp_path, monkeypatch):
    dicom_dir = tmp_path / "dicom-flair"
    dicom_dir.mkdir()
    (dicom_dir / "IM0001.dcm").write_bytes(b"DICOM")
    monkeypatch.setattr(uploaded_mri.shutil, "which", lambda _name: "dcm2niix")

    def fake_run(command, capture_output, text, timeout):
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="bad series")

    monkeypatch.setattr(uploaded_mri.subprocess, "run", fake_run)

    with pytest.raises(DicomConversionError, match="bad series"):
        build_uploaded_mri(
            flair=None,
            flair_dicom_dir=dicom_dir,
            t1w=None,
            roi=None,
            output_root=tmp_path / "out",
            synthetic_mode="placeholder",
            roi_strategy="none",
        )


def test_uploaded_mri_dicom_no_new_nifti_raises(tmp_path, monkeypatch):
    dicom_dir = tmp_path / "dicom-flair"
    converted_dir = tmp_path / "converted"
    stale_dir = converted_dir / "DICOM_Upload_01" / "flair"
    dicom_dir.mkdir()
    stale_dir.mkdir(parents=True)
    (dicom_dir / "IM0001.dcm").write_bytes(b"DICOM")
    _make_volume(stale_dir / "stale.nii.gz", seed=45)
    monkeypatch.setattr(uploaded_mri.shutil, "which", lambda _name: "dcm2niix")

    def fake_run(command, capture_output, text, timeout):
        return subprocess.CompletedProcess(command, 0, stdout="nothing new", stderr="")

    monkeypatch.setattr(uploaded_mri.subprocess, "run", fake_run)

    with pytest.raises(DicomConversionError, match="produced no new"):
        build_uploaded_mri(
            flair=None,
            flair_dicom_dir=dicom_dir,
            t1w=None,
            roi=None,
            output_root=tmp_path / "out",
            subject_id="DICOM Upload 01",
            converted_dicom_dir=converted_dir,
            synthetic_mode="placeholder",
            roi_strategy="none",
        )


def test_uploaded_mri_dicom_multiple_outputs_raises(tmp_path, monkeypatch):
    dicom_dir = tmp_path / "dicom-flair"
    dicom_dir.mkdir()
    (dicom_dir / "IM0001.dcm").write_bytes(b"DICOM")
    monkeypatch.setattr(uploaded_mri.shutil, "which", lambda _name: "dcm2niix")

    def fake_run(command, capture_output, text, timeout):
        out_dir = Path(command[command.index("-o") + 1])
        _make_volume(out_dir / "flair_series_a.nii.gz", seed=51)
        _make_volume(out_dir / "flair_series_b.nii.gz", seed=52)
        return subprocess.CompletedProcess(command, 0, stdout="converted two", stderr="")

    monkeypatch.setattr(uploaded_mri.subprocess, "run", fake_run)

    with pytest.raises(DicomConversionError, match="multiple NIfTI candidates"):
        build_uploaded_mri(
            flair=None,
            flair_dicom_dir=dicom_dir,
            t1w=None,
            roi=None,
            output_root=tmp_path / "out",
            synthetic_mode="placeholder",
            roi_strategy="none",
        )


def test_uploaded_mri_dicom_success_feeds_pipeline(tmp_path, monkeypatch):
    dicom_dir = tmp_path / "dicom-flair"
    out = tmp_path / "out"
    dicom_dir.mkdir()
    (dicom_dir / "IM0001.dcm").write_bytes(b"DICOM")
    monkeypatch.setattr(uploaded_mri.shutil, "which", lambda _name: "dcm2niix")

    def fake_run(command, capture_output, text, timeout):
        out_dir = Path(command[command.index("-o") + 1])
        _make_volume(out_dir / "flair.nii.gz", seed=50)
        return subprocess.CompletedProcess(command, 0, stdout="converted", stderr="")

    monkeypatch.setattr(uploaded_mri.subprocess, "run", fake_run)

    index_path = build_uploaded_mri(
        flair=None,
        flair_dicom_dir=dicom_dir,
        t1w=None,
        roi=None,
        output_root=out,
        subject_id="DICOM Upload 01",
        synthetic_mode="placeholder",
        roi_strategy="none",
        max_splat_points=50,
        presentation_splat_points=50,
        force=True,
    )

    with index_path.open() as fh:
        index = json.load(fh)
    manifest_path = out / index["subjects"][0]["manifest"]
    with manifest_path.open() as fh:
        manifest = json.load(fh)

    conversion = manifest["upload_meta"]["dicom_conversion"]["FLAIR"]
    assert manifest["input_mode"] == "uploaded_mri"
    assert manifest["modalities"] == ["FLAIR"]
    assert conversion["input_dir"] == str(dicom_dir)
    assert conversion["input_inventory"]["file_count"] == 1
    assert conversion["input_inventory"]["suffix_counts"] == {".dcm": 1}
    assert conversion["output_nifti"].endswith(".nii.gz")
    assert conversion["selected_output_nifti"] == conversion["output_nifti"]
    assert conversion["returncode"] == 0
    assert conversion["sidecar"]["exists"] is False
    assert "sidecar_missing" in conversion["conversion_warnings"]
    assert conversion["conversion_readiness"]["status"] == "review"
    assert manifest["upload_meta"]["dicom_conversion_safety"]["status"] == "review"
    assert "FLAIR:sidecar_missing" in manifest["upload_meta"]["dicom_conversion_safety"]["review_items"]
    assert manifest["data_quality"]["dicom_conversion_safety"]["status"] == "review"
    assert "patient_splat_ply" in manifest["artifacts"]


def test_uploaded_mri_dicom_rejects_localizer_sidecar(tmp_path, monkeypatch):
    dicom_dir = tmp_path / "dicom-flair"
    out = tmp_path / "out"
    dicom_dir.mkdir()
    (dicom_dir / "IM0001.dcm").write_bytes(b"DICOM")
    monkeypatch.setattr(uploaded_mri.shutil, "which", lambda _name: "dcm2niix")

    def fake_run(command, capture_output, text, timeout):
        out_dir = Path(command[command.index("-o") + 1])
        _make_volume(out_dir / "flair.nii.gz", seed=50)
        (out_dir / "flair.json").write_text(
            json.dumps(
                {
                    "Modality": "MR",
                    "SeriesDescription": "3-plane localizer scout",
                    "ProtocolName": "localizer",
                    "ImageType": ["ORIGINAL", "PRIMARY"],
                    "ImageOrientationPatientDICOM": [1, 0, 0, 0, 1, 0],
                }
            )
        )
        return subprocess.CompletedProcess(command, 0, stdout="converted", stderr="")

    monkeypatch.setattr(uploaded_mri.subprocess, "run", fake_run)

    with pytest.raises(DicomConversionError, match="unsafe for structural FCD processing"):
        build_uploaded_mri(
            flair=None,
            flair_dicom_dir=dicom_dir,
            t1w=None,
            roi=None,
            output_root=out,
            subject_id="DICOM Upload 01",
            synthetic_mode="placeholder",
            roi_strategy="none",
            max_splat_points=50,
            presentation_splat_points=50,
            force=True,
        )


def test_uploaded_mri_dicom_sidecar_structural_pass(tmp_path, monkeypatch):
    dicom_dir = tmp_path / "dicom-flair"
    out = tmp_path / "out"
    dicom_dir.mkdir()
    (dicom_dir / "IM0001.dcm").write_bytes(b"DICOM")
    monkeypatch.setattr(uploaded_mri.shutil, "which", lambda _name: "dcm2niix")

    def fake_run(command, capture_output, text, timeout):
        out_dir = Path(command[command.index("-o") + 1])
        _make_volume(out_dir / "flair.nii.gz", seed=51)
        (out_dir / "flair.json").write_text(
            json.dumps(
                {
                    "Modality": "MR",
                    "SeriesDescription": "3D FLAIR structural",
                    "ProtocolName": "FLAIR",
                    "ImageType": ["ORIGINAL", "PRIMARY"],
                    "MRAcquisitionType": "3D",
                    "ImageOrientationPatientDICOM": [1, 0, 0, 0, 1, 0],
                }
            )
        )
        return subprocess.CompletedProcess(command, 0, stdout="converted", stderr="")

    monkeypatch.setattr(uploaded_mri.subprocess, "run", fake_run)

    index_path = build_uploaded_mri(
        flair=None,
        flair_dicom_dir=dicom_dir,
        t1w=None,
        roi=None,
        output_root=out,
        subject_id="DICOM Upload 02",
        synthetic_mode="placeholder",
        roi_strategy="none",
        max_splat_points=50,
        presentation_splat_points=50,
        force=True,
    )

    with index_path.open() as fh:
        index = json.load(fh)
    manifest_path = out / index["subjects"][0]["manifest"]
    with manifest_path.open() as fh:
        manifest = json.load(fh)

    conversion = manifest["upload_meta"]["dicom_conversion"]["FLAIR"]
    assert conversion["sidecar"]["exists"] is True
    assert conversion["sidecar"]["json_parse_ok"] is True
    assert conversion["sidecar"]["series_descriptor_terms"] == ["flair"]
    assert conversion["conversion_warnings"] == []
    assert conversion["conversion_readiness"]["status"] == "pass"
    assert manifest["upload_meta"]["dicom_conversion_safety"]["status"] == "pass"


def test_uploaded_mri_missing_control_dataset_raises(tmp_path):
    flair = tmp_path / "upload" / "flair.nii.gz"
    _make_volume(flair)

    with pytest.raises(FileNotFoundError, match="control dataset"):
        build_uploaded_mri(
            flair=flair,
            t1w=None,
            roi=None,
            output_root=tmp_path / "out",
            synthetic_mode="control-prior",
            control_dataset_dir=tmp_path / "missing-controls",
        )


def test_uploaded_mri_with_explicit_control_volume(tmp_path):
    flair = tmp_path / "upload" / "flair.nii.gz"
    roi = tmp_path / "upload" / "roi.nii.gz"
    control = tmp_path / "controls" / "sub-control" / "anat" / "sub-control_FLAIR.nii.gz"
    out = tmp_path / "out"
    _make_volume(flair, seed=20)
    _make_roi(roi)
    _make_volume(control, seed=21)

    index_path = build_uploaded_mri(
        flair=flair,
        t1w=None,
        roi=roi,
        output_root=out,
        subject_id="Patient Upload 01",
        synthetic_mode="control-prior",
        roi_strategy="provided",
        control_volumes=[control],
        max_splat_points=50,
        presentation_splat_points=50,
        force=True,
    )

    assert index_path.exists()
    with index_path.open() as fh:
        index = json.load(fh)
    assert index["dataset"] == "Uploaded MRI"
    subject = index["subjects"][0]["subject"]
    assert subject == "Patient_Upload_01"

    manifest_path = out / subject / "manifest.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert manifest["input_mode"] == "uploaded_mri"
    assert manifest["synthetic_normal_source"] == "control_prior_roi_hybrid"
    assert manifest["control_prior_meta"]["control_prior_count"] == 1
    assert manifest["upload_meta"]["resolved_control_volume_count"] == 1
    assert manifest["upload_meta"]["upload_input_compatibility"]["status"] == "pass"
    assert manifest["data_quality"]["upload_input_compatibility"]["status"] == "pass"
    assert manifest["upload_meta"]["data_quality_gate"]["status"] in {"pass", "review"}
    assert manifest["upload_meta"]["data_quality_gate"]["can_run_3dgs"] is True
    assert "patient_splat_ply" in manifest["artifacts"]
    assert "residual_heatmap" in manifest["artifacts"]


def test_uploaded_mri_control_gaussian_prior_residual_with_placeholder_mode(tmp_path):
    flair = tmp_path / "upload" / "flair.nii.gz"
    roi = tmp_path / "upload" / "roi.nii.gz"
    control_1 = tmp_path / "controls" / "sub-control01" / "anat" / "sub-control01_FLAIR.nii.gz"
    control_2 = tmp_path / "controls" / "sub-control02" / "anat" / "sub-control02_FLAIR.nii.gz"
    out = tmp_path / "out"
    _make_volume(flair, seed=22)
    _make_roi(roi)
    _make_volume(control_1, seed=23)
    _make_volume(control_2, seed=24)

    index_path = build_uploaded_mri(
        flair=flair,
        t1w=None,
        roi=roi,
        output_root=out,
        subject_id="external-control-gp",
        synthetic_mode="placeholder",
        roi_strategy="provided",
        control_volumes=[control_1, control_2],
        compute_control_gaussian_prior_residual=True,
        control_gaussian_prior_max_points=50,
        control_gaussian_prior_heatmap_method="scatter",
        max_splat_points=50,
        presentation_splat_points=50,
        force=True,
    )

    with index_path.open() as fh:
        index = json.load(fh)
    manifest_path = out / index["subjects"][0]["manifest"]
    with manifest_path.open() as fh:
        manifest = json.load(fh)

    assert manifest["synthetic_normal_source"] == "placeholder"
    assert manifest["stage_status"]["control_gaussian_prior"] == "ok"
    assert manifest["upload_meta"]["compute_control_gaussian_prior_residual"] is True
    assert manifest["upload_meta"]["resolved_control_volume_count"] == 2
    assert manifest["control_gaussian_prior_meta"]["n_controls"] == 2
    assert manifest["control_gaussian_prior_meta"]["gaussian_state_schema"] == (
        "patient_control_gaussian_prior_v1"
    )
    assert manifest["residual_candidate_meta"]["source_heatmap"] == (
        "control_gaussian_prior/control_prior_z_residual_heatmap.nii.gz"
    )
    for key in (
        "control_prior_patient_splat",
        "control_prior_mean_splat",
        "control_prior_z_residual_splat",
        "control_gaussian_prior_metrics",
    ):
        rel = manifest["artifacts"][key]
        assert (out / "external-control-gp" / rel).exists(), f"missing {key}: {rel}"


def test_uploaded_mri_uses_control_dataset_pool(tmp_path):
    flair = tmp_path / "upload" / "flair.nii.gz"
    roi = tmp_path / "upload" / "roi.nii.gz"
    out = tmp_path / "out"
    controls = _fake_control_dataset(tmp_path)
    _make_volume(flair, seed=30)
    _make_roi(roi)

    index_path = build_uploaded_mri(
        flair=flair,
        t1w=None,
        roi=roi,
        output_root=out,
        subject_id="sub-uploaded",
        synthetic_mode="control-prior",
        roi_strategy="provided",
        control_dataset_dir=controls,
        max_splat_points=50,
        presentation_splat_points=50,
        force=True,
    )

    with index_path.open() as fh:
        index = json.load(fh)
    manifest_path = out / index["subjects"][0]["manifest"]
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert manifest["synthetic_normal_source"] == "control_prior_roi_hybrid"
    assert manifest["control_prior_meta"]["control_prior_count"] == 2
    assert manifest["control_prior_meta"]["control_prior_subjects"] == [
        "sub-ctrl01",
        "sub-ctrl02",
    ]


def test_uploaded_mri_excludes_same_subject_from_control_dataset(tmp_path):
    flair = tmp_path / "upload" / "flair.nii.gz"
    roi = tmp_path / "upload" / "roi.nii.gz"
    out = tmp_path / "out"
    controls = _fake_control_dataset(tmp_path)
    _make_volume(flair, seed=31)
    _make_roi(roi)

    index_path = build_uploaded_mri(
        flair=flair,
        t1w=None,
        roi=roi,
        output_root=out,
        subject_id="sub-ctrl01",
        synthetic_mode="control-prior",
        roi_strategy="provided",
        control_dataset_dir=controls,
        max_splat_points=50,
        presentation_splat_points=50,
        force=True,
    )

    with index_path.open() as fh:
        index = json.load(fh)
    manifest_path = out / index["subjects"][0]["manifest"]
    with manifest_path.open() as fh:
        manifest = json.load(fh)

    assert manifest["control_prior_meta"]["control_prior_count"] == 1
    assert manifest["control_prior_meta"]["control_prior_subjects"] == ["sub-ctrl02"]
    pool_meta = manifest["upload_meta"]["control_pool_meta"]
    assert pool_meta["leakage_guard_applied"] is True
    assert pool_meta["excluded_control_subjects"] == ["sub-ctrl01"]
    assert pool_meta["excluded_control_count"] == 1
    assert pool_meta["excluded_control_volumes"][0]["reason"] == "same_subject_as_upload"


def test_uploaded_mri_excludes_uploaded_file_from_explicit_controls(tmp_path):
    flair = tmp_path / "upload" / "flair.nii.gz"
    roi = tmp_path / "upload" / "roi.nii.gz"
    control = tmp_path / "controls" / "sub-control" / "anat" / "sub-control_FLAIR.nii.gz"
    out = tmp_path / "out"
    _make_volume(flair, seed=32)
    _make_roi(roi)
    _make_volume(control, seed=33)

    index_path = build_uploaded_mri(
        flair=flair,
        t1w=None,
        roi=roi,
        output_root=out,
        subject_id="external-explicit-guard",
        synthetic_mode="control-prior",
        roi_strategy="provided",
        control_volumes=[flair, control],
        max_splat_points=50,
        presentation_splat_points=50,
        force=True,
    )

    with index_path.open() as fh:
        index = json.load(fh)
    manifest_path = out / index["subjects"][0]["manifest"]
    with manifest_path.open() as fh:
        manifest = json.load(fh)

    assert manifest["control_prior_meta"]["control_prior_count"] == 1
    assert manifest["control_prior_meta"]["control_prior_subjects"] == ["sub-control"]
    pool_meta = manifest["upload_meta"]["control_pool_meta"]
    assert pool_meta["excluded_control_count"] == 1
    assert pool_meta["excluded_control_volumes"][0]["reason"] == "same_as_uploaded_volume"


def test_uploaded_mri_without_roi_uses_whole_brain_control_prior(tmp_path):
    flair = tmp_path / "upload" / "flair.nii.gz"
    out = tmp_path / "out"
    controls = _fake_control_dataset(tmp_path)
    _make_volume(flair, seed=33)

    index_path = build_uploaded_mri(
        flair=flair,
        t1w=None,
        roi=None,
        output_root=out,
        subject_id="external-no-roi",
        synthetic_mode="control-prior",
        roi_strategy="none",
        control_dataset_dir=controls,
        max_splat_points=50,
        presentation_splat_points=50,
        force=True,
    )

    with index_path.open() as fh:
        index = json.load(fh)
    manifest_path = out / index["subjects"][0]["manifest"]
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert manifest["input_mode"] == "uploaded_mri"
    assert manifest["synthetic_normal_source"] == "control_prior_whole_brain_prior"
    assert manifest["control_prior_meta"]["control_prior_scope"] == "whole_brain_no_roi"
    assert manifest["upload_meta"]["resolved_control_volume_count"] == 2
    assert manifest["paired_gaussian_residual_meta"]["residual_mean"] > 0


def test_uploaded_mri_cli_placeholder(tmp_path):
    t1w = tmp_path / "upload" / "t1w.nii.gz"
    out = tmp_path / "out"
    _make_volume(t1w, seed=40)

    main([
        "--t1w", str(t1w),
        "--subject-id", "external-01",
        "--output-dir", str(out),
        "--synthetic-mode", "placeholder",
        "--roi-strategy", "none",
        "--max-splat-points", "50",
        "--presentation-splat-points", "50",
        "--paired-gaussian-support-basis", "patient",
        "--force",
    ])

    manifest_path = out / "external-01" / "manifest.json"
    assert manifest_path.exists()
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert manifest["input_mode"] == "uploaded_mri"
    assert manifest["synthetic_normal_source"] == "placeholder"
    assert manifest["modalities"] == ["T1w"]
    assert manifest["upload_meta"]["paired_gaussian_support_basis"] == "patient"
    assert manifest["paired_gaussian_residual_meta"]["support_basis_mode"] == "patient"
    assert manifest["paired_gaussian_residual_meta"]["support_basis"] == "patient"


def test_uploaded_mri_cli_control_gaussian_prior_flags(tmp_path):
    flair = tmp_path / "upload" / "flair.nii.gz"
    roi = tmp_path / "upload" / "roi.nii.gz"
    control_1 = tmp_path / "controls" / "sub-c01" / "anat" / "sub-c01_FLAIR.nii.gz"
    control_2 = tmp_path / "controls" / "sub-c02" / "anat" / "sub-c02_FLAIR.nii.gz"
    out = tmp_path / "out"
    _make_volume(flair, seed=44)
    _make_roi(roi)
    _make_volume(control_1, seed=45)
    _make_volume(control_2, seed=46)

    main([
        "--flair", str(flair),
        "--roi", str(roi),
        "--subject-id", "external-control-cli",
        "--output-dir", str(out),
        "--synthetic-mode", "placeholder",
        "--roi-strategy", "provided",
        "--control-volume", str(control_1),
        "--control-volume", str(control_2),
        "--compute-control-gaussian-prior-residual",
        "--control-gaussian-prior-max-points", "50",
        "--control-gaussian-prior-heatmap-method", "scatter",
        "--max-splat-points", "50",
        "--presentation-splat-points", "50",
        "--force",
    ])

    manifest_path = out / "external-control-cli" / "manifest.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert manifest["upload_meta"]["compute_control_gaussian_prior_residual"] is True
    assert manifest["upload_meta"]["control_gaussian_prior_max_points"] == 50
    assert manifest["upload_meta"]["control_gaussian_prior_heatmap_method"] == "scatter"
    assert manifest["stage_status"]["control_gaussian_prior"] == "ok"
    assert "control_prior_z_residual_splat" in manifest["artifacts"]
    assert manifest["residual_candidate_meta"]["source_heatmap"] == (
        "control_gaussian_prior/control_prior_z_residual_heatmap.nii.gz"
    )


def test_uploaded_mri_cli_placeholder_t2w(tmp_path):
    t2w = tmp_path / "upload" / "patient_T2w.nii.gz"
    out = tmp_path / "out"
    _make_volume(t2w, seed=41)

    main([
        "--t2w", str(t2w),
        "--subject-id", "external-t2",
        "--output-dir", str(out),
        "--synthetic-mode", "placeholder",
        "--roi-strategy", "none",
        "--max-splat-points", "50",
        "--presentation-splat-points", "50",
        "--force",
    ])

    manifest_path = out / "external-t2" / "manifest.json"
    assert manifest_path.exists()
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert manifest["modalities"] == ["T2w"]
    assert manifest["source_files"]["t2w"] == str(t2w)
    assert manifest["upload_meta"]["t2w"] == str(t2w)
    assert manifest["upload_meta"]["preferred_structural_modality"] == "T2w"
    assert "patient_t2w" in manifest["artifacts"]


def test_uploaded_mri_cli_auto_scan_placeholder(tmp_path):
    scan = tmp_path / "upload" / "external_FLAIR.nii.gz"
    out = tmp_path / "out"
    _make_volume(scan, seed=44)

    main([
        "--scan", str(scan),
        "--subject-id", "external-auto",
        "--output-dir", str(out),
        "--synthetic-mode", "placeholder",
        "--roi-strategy", "none",
        "--max-splat-points", "50",
        "--presentation-splat-points", "50",
        "--force",
    ])

    manifest_path = out / "external-auto" / "manifest.json"
    assert manifest_path.exists()
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert manifest["modalities"] == ["FLAIR"]
    assert manifest["source_files"]["flair"] == str(scan)
    auto = manifest["upload_meta"]["auto_structural_resolution"]
    assert auto["status"] == "pass"
    assert auto["kind"] == "nifti"
    assert auto["inferred_label"] == "FLAIR"
    assert manifest["upload_meta"]["upload_input_compatibility"]["auto_resolution"] == auto


def _patch_uploaded_gaussian_pipeline(monkeypatch, train_func=_fake_train_gaussian_field):
    monkeypatch.setattr(
        "fcd_prototype.gaussian_field.train_gaussian_field",
        train_func,
    )
    monkeypatch.setattr(
        "fcd_prototype.gaussian_field.project_volume_to_gaussian_support",
        _fake_project_gaussian_support,
    )
    monkeypatch.setattr(
        "fcd_prototype.gaussian_residual.compare_gaussian_fields",
        _fake_compare_gaussian_fields,
    )


def _run_uploaded_paper_3dgs_cli(tmp_path, extra_args=None):
    flair = tmp_path / "upload" / "patient_FLAIR.nii.gz"
    roi = tmp_path / "upload" / "patient_roi.nii.gz"
    out = tmp_path / "out"
    _make_volume(flair, seed=42)
    _make_roi(roi)

    args = [
        "--flair", str(flair),
        "--roi", str(roi),
        "--subject-id", "external-paper",
        "--output-dir", str(out),
        "--synthetic-mode", "placeholder",
        "--roi-strategy", "provided",
        "--paper-3dgs-mode",
        "--skip-splat-reconstruction-audit",
        "--max-splat-points", "50",
        "--presentation-splat-points", "50",
        "--force",
    ]
    if extra_args:
        args.extend(extra_args)
    main(args)

    manifest_path = out / "external-paper" / "manifest.json"
    with manifest_path.open() as fh:
        return json.load(fh)


def test_uploaded_mri_paper_3dgs_mode_sets_training_flags_and_manifest(
    tmp_path, monkeypatch,
):
    _patch_uploaded_gaussian_pipeline(monkeypatch)

    manifest = _run_uploaded_paper_3dgs_cli(tmp_path)

    assert manifest["input_mode"] == "uploaded_mri"
    assert manifest["pipeline_preset"] == "paper_3dgs"
    assert manifest["upload_meta"]["pipeline_preset"] == "paper_3dgs"
    assert manifest["upload_meta"]["paper_3dgs_mode"] is True
    assert manifest["paper_3dgs_mode_meta"]["enabled"] is True
    assert manifest["paper_3dgs_mode_meta"]["train_gaussian_field"] is True
    assert manifest["paper_3dgs_mode_meta"]["train_synthetic_gaussian_field"] is True
    assert manifest["paper_3dgs_mode_meta"]["compute_control_gaussian_prior_residual"] is True
    assert manifest["paper_3dgs_mode_meta"]["compute_gaussian_residual"] is True
    assert manifest["upload_meta"]["compute_control_gaussian_prior_residual"] is True
    assert manifest["stage_status"]["gaussian_field"] == "ok"
    assert manifest["stage_status"]["synthetic_gaussian_field"] == "ok"
    assert manifest["stage_status"]["gaussian_residual"] == "ok"
    assert manifest["stage_status"]["control_gaussian_prior"].startswith("skipped:")
    assert "optimized_gaussian_splat_ply" in manifest["artifacts"]
    assert "synthetic_normal_gaussian_splat_ply" in manifest["artifacts"]
    assert "gaussian_residual_splat" in manifest["artifacts"]
    assert manifest["gaussian_residual_meta"]["support_fingerprint_match"] is True


def test_uploaded_mri_paper_3dgs_mode_default_volume_settings(
    tmp_path, monkeypatch,
):
    captured = {}

    def capture_train(volume_path, output_dir, **kwargs):
        captured["volume_loss_weight"] = kwargs.get("volume_loss_weight")
        captured["volume_sample_count"] = kwargs.get("volume_sample_count")
        return _fake_train_gaussian_field(volume_path, output_dir, **kwargs)

    _patch_uploaded_gaussian_pipeline(monkeypatch, train_func=capture_train)

    manifest = _run_uploaded_paper_3dgs_cli(tmp_path)

    assert captured["volume_loss_weight"] == pytest.approx(0.25)
    assert captured["volume_sample_count"] == 8192
    assert manifest["upload_meta"]["gaussian_volume_loss_weight"] == pytest.approx(0.25)
    assert manifest["upload_meta"]["gaussian_volume_sample_count"] == 8192
    assert manifest["paper_3dgs_mode_meta"]["gaussian_volume_loss_weight"] == pytest.approx(0.25)
    assert manifest["paper_3dgs_mode_meta"]["gaussian_volume_sample_count"] == 8192


def test_uploaded_mri_paper_3dgs_mode_respects_explicit_zero_volume_loss(
    tmp_path, monkeypatch,
):
    captured = {}

    def capture_train(volume_path, output_dir, **kwargs):
        captured["volume_loss_weight"] = kwargs.get("volume_loss_weight")
        return _fake_train_gaussian_field(volume_path, output_dir, **kwargs)

    _patch_uploaded_gaussian_pipeline(monkeypatch, train_func=capture_train)

    manifest = _run_uploaded_paper_3dgs_cli(
        tmp_path,
        extra_args=["--gaussian-volume-loss-weight", "0"],
    )

    assert captured["volume_loss_weight"] == pytest.approx(0.0)
    assert manifest["upload_meta"]["gaussian_volume_loss_weight"] == pytest.approx(0.0)
    assert manifest["paper_3dgs_mode_meta"]["gaussian_volume_loss_weight"] == pytest.approx(0.0)


def test_uploaded_mri_paper_3dgs_mode_respects_explicit_sample_count(
    tmp_path, monkeypatch,
):
    captured = {}

    def capture_train(volume_path, output_dir, **kwargs):
        captured["volume_sample_count"] = kwargs.get("volume_sample_count")
        return _fake_train_gaussian_field(volume_path, output_dir, **kwargs)

    _patch_uploaded_gaussian_pipeline(monkeypatch, train_func=capture_train)

    manifest = _run_uploaded_paper_3dgs_cli(
        tmp_path,
        extra_args=["--gaussian-volume-sample-count", "2048"],
    )

    assert captured["volume_sample_count"] == 2048
    assert manifest["upload_meta"]["gaussian_volume_sample_count"] == 2048
    assert manifest["paper_3dgs_mode_meta"]["gaussian_volume_sample_count"] == 2048


def test_uploaded_mri_cli_supervised_roi_checkpoint(tmp_path, monkeypatch):
    flair = tmp_path / "upload" / "flair.nii.gz"
    out = tmp_path / "out"
    ckpt = tmp_path / "lesion_proposer.pt"
    _make_volume(flair, seed=41)
    ckpt.write_bytes(b"PT")
    monkeypatch.setattr(slp, "predict_lesion_roi", _fake_predict_uploaded_roi)

    main([
        "--flair", str(flair),
        "--subject-id", "external-supervised",
        "--output-dir", str(out),
        "--synthetic-mode", "placeholder",
        "--roi-strategy", "supervised",
        "--supervised-roi-checkpoint", str(ckpt),
        "--supervised-roi-threshold", "0.7",
        "--supervised-roi-selection-mode", "top-fraction",
        "--supervised-roi-foreground-fraction", "0.015",
        "--supervised-roi-max-foreground-fraction", "0.02",
        "--max-splat-points", "50",
        "--presentation-splat-points", "50",
        "--force",
    ])

    manifest_path = out / "external-supervised" / "manifest.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert manifest["input_mode"] == "uploaded_mri"
    assert manifest["roi_source"] == "supervised"
    assert manifest["stage_status"]["supervised_roi"] == "ok"
    assert manifest["supervised_roi_meta"]["accepted"] is True
    assert manifest["upload_meta"]["supervised_roi_checkpoint"] == str(ckpt)
    assert manifest["upload_meta"]["supervised_roi_threshold"] == pytest.approx(0.7)
    assert manifest["upload_meta"]["supervised_roi_selection_mode"] == "top-fraction"
    assert manifest["upload_meta"]["supervised_roi_foreground_fraction"] == pytest.approx(0.015)
    assert manifest["supervised_roi_meta"]["selection_mode"] == "top-fraction"
    assert manifest["supervised_roi_meta"]["foreground_fraction"] == pytest.approx(0.015)


def test_uploaded_mri_cli_supervised_roi_operating_point(tmp_path, monkeypatch):
    flair = tmp_path / "upload" / "flair.nii.gz"
    out = tmp_path / "out"
    ckpt = tmp_path / "lesion_proposer.pt"
    op = tmp_path / "threshold_sweep.json"
    _make_volume(flair, seed=42)
    ckpt.write_bytes(b"PT")
    op.write_text(json.dumps({
        "method": "supervised_lesion_threshold_sweep",
        "selected_operating_point": {
            "threshold": 0.8,
            "selection_mode": "threshold-and-top-fraction",
            "foreground_fraction": 0.012,
        },
        "constraints": {
            "min_component_voxels": 11,
            "max_hc_mean_predicted_foreground_fraction": 0.02,
        },
    }))

    calls = []

    def capture_predict(*args, **kwargs):
        calls.append(kwargs)
        return _fake_predict_uploaded_roi(*args, **kwargs)

    monkeypatch.setattr(slp, "predict_lesion_roi", capture_predict)

    main([
        "--flair", str(flair),
        "--subject-id", "external-supervised-op",
        "--output-dir", str(out),
        "--synthetic-mode", "placeholder",
        "--roi-strategy", "supervised",
        "--supervised-roi-checkpoint", str(ckpt),
        "--supervised-roi-operating-point", str(op),
        "--supervised-roi-threshold", "0.2",
        "--supervised-roi-max-foreground-fraction", "0.5",
        "--max-splat-points", "50",
        "--presentation-splat-points", "50",
        "--force",
    ])

    manifest_path = out / "external-supervised-op" / "manifest.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert calls[0]["threshold"] == pytest.approx(0.8)
    assert calls[0]["selection_mode"] == "threshold-and-top-fraction"
    assert calls[0]["foreground_fraction"] == pytest.approx(0.012)
    assert calls[0]["min_component_voxels"] == 11
    assert manifest["upload_meta"]["supervised_roi_operating_point"] == str(op)
    assert manifest["upload_meta"]["supervised_roi_threshold"] == pytest.approx(0.8)
    assert manifest["upload_meta"]["supervised_roi_selection_mode"] == (
        "threshold-and-top-fraction"
    )
    assert manifest["upload_meta"]["supervised_roi_foreground_fraction"] == pytest.approx(0.012)
    assert manifest["upload_meta"]["supervised_roi_max_foreground_fraction"] == pytest.approx(0.02)
    assert manifest["supervised_roi_meta"]["operating_point"]["source_kind"] == "threshold_sweep"
