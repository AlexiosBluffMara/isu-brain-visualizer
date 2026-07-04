"""Tests for viewer metadata helpers: load_sidecar and build_viewer_metadata."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fcd_prototype.bids_discovery import load_sidecar, SubjectRecord
from fcd_prototype.mri_data_audit import build_viewer_metadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_nifti(path: Path, shape=(10, 12, 8), seed=0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape).astype(np.float32) * 0.3 + 0.5
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.5, 1.0])), str(path))
    return path


def _write_roi(path: Path, shape=(10, 12, 8)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros(shape, dtype=np.uint8)
    data[3:6, 3:6, 3:6] = 1
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.5, 1.0])), str(path))
    return path


def _write_sidecar(path: Path, fields: dict) -> Path:
    path.write_text(json.dumps(fields), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_sidecar tests
# ---------------------------------------------------------------------------

def test_load_sidecar_returns_known_fields(tmp_path):
    nii = tmp_path / "sub-00001_acq-tse_FLAIR.nii.gz"
    _write_nifti(nii)
    sidecar = tmp_path / "sub-00001_acq-tse_FLAIR.json"
    _write_sidecar(sidecar, {
        "Manufacturer": "Siemens",
        "MagneticFieldStrength": 3.0,
        "EchoTime": 0.372,
        "SomeOtherField": "ignored",
    })
    result = load_sidecar(nii)
    assert result["Manufacturer"] == "Siemens"
    assert result["MagneticFieldStrength"] == 3.0
    assert result["EchoTime"] == 0.372
    assert "SomeOtherField" not in result


def test_load_sidecar_missing_returns_empty(tmp_path):
    nii = tmp_path / "sub-00001_T1w.nii.gz"
    _write_nifti(nii)
    result = load_sidecar(nii)
    assert result == {}


def test_load_sidecar_corrupt_json_returns_empty(tmp_path):
    nii = tmp_path / "sub-00001_FLAIR.nii.gz"
    _write_nifti(nii)
    (tmp_path / "sub-00001_FLAIR.json").write_text("{invalid json", encoding="utf-8")
    result = load_sidecar(nii)
    assert result == {}


def test_load_sidecar_partial_fields(tmp_path):
    nii = tmp_path / "sub-00002_T1w.nii.gz"
    _write_nifti(nii)
    _write_sidecar(tmp_path / "sub-00002_T1w.json", {"Manufacturer": "GE"})
    result = load_sidecar(nii)
    assert result == {"Manufacturer": "GE"}


# ---------------------------------------------------------------------------
# build_viewer_metadata tests
# ---------------------------------------------------------------------------

def _make_record(subject_id="sub-00001", group="fcd", metadata=None):
    return SubjectRecord(
        subject_id=subject_id,
        group=group,
        metadata=metadata or {"sex": "M", "age_scan": "32", "split": "train"},
    )


def test_build_viewer_metadata_has_required_sections(tmp_path):
    nii = _write_nifti(tmp_path / "flair.nii.gz")
    rec = _make_record()
    result = build_viewer_metadata(
        subject_record=rec,
        base_nifti_path=nii,
        base_modality="FLAIR",
    )
    assert "participant" in result
    assert "scan" in result
    assert "geometry" in result


def test_build_viewer_metadata_participant_fields(tmp_path):
    nii = _write_nifti(tmp_path / "flair.nii.gz")
    rec = _make_record(group="fcd", metadata={"sex": "F", "age_scan": "28", "lobe": "FL"})
    result = build_viewer_metadata(
        subject_record=rec,
        base_nifti_path=nii,
        base_modality="FLAIR",
    )
    p = result["participant"]
    assert p["subject_id"] == "sub-00001"
    assert p["group"] == "fcd"
    assert p["sex"] == "F"
    assert p["age_scan"] == "28"
    assert p["lobe"] == "FL"


def test_build_viewer_metadata_geometry_populated(tmp_path):
    nii = _write_nifti(tmp_path / "flair.nii.gz", shape=(10, 12, 8))
    rec = _make_record()
    result = build_viewer_metadata(
        subject_record=rec,
        base_nifti_path=nii,
        base_modality="FLAIR",
    )
    geo = result["geometry"]
    assert geo["shape"] == [10, 12, 8]
    assert len(geo["voxel_spacing_mm"]) == 3
    assert isinstance(geo["orientation"], list)
    assert "intensity_p1" in geo
    assert "intensity_p99" in geo


def test_build_viewer_metadata_sidecar_injected(tmp_path):
    nii = _write_nifti(tmp_path / "flair.nii.gz")
    rec = _make_record()
    sidecar = {"Manufacturer": "Philips", "MagneticFieldStrength": 1.5, "EchoTime": 0.1}
    result = build_viewer_metadata(
        subject_record=rec,
        base_nifti_path=nii,
        base_modality="FLAIR",
        sidecar_fields=sidecar,
    )
    scan = result["scan"]
    assert scan["Manufacturer"] == "Philips"
    assert scan["MagneticFieldStrength"] == 1.5
    assert scan["modality"] == "FLAIR"


def test_build_viewer_metadata_roi_section_present(tmp_path):
    nii = _write_nifti(tmp_path / "flair.nii.gz")
    roi = _write_roi(tmp_path / "roi.nii.gz")
    rec = _make_record()
    result = build_viewer_metadata(
        subject_record=rec,
        base_nifti_path=nii,
        base_modality="FLAIR",
        roi_nifti_path=roi,
    )
    assert "roi" in result
    r = result["roi"]
    assert r["voxel_count"] > 0
    assert r["volume_mm3"] > 0
    assert r["center_ijk"] is not None
    assert len(r["center_ijk"]) == 3
    assert r["bounds_min_ijk"] is not None
    assert r["bounds_max_ijk"] is not None


def test_build_viewer_metadata_no_roi_no_roi_section(tmp_path):
    nii = _write_nifti(tmp_path / "flair.nii.gz")
    rec = _make_record()
    result = build_viewer_metadata(
        subject_record=rec,
        base_nifti_path=nii,
        base_modality="FLAIR",
    )
    assert "roi" not in result


def test_build_viewer_metadata_missing_nifti_no_crash(tmp_path):
    rec = _make_record()
    result = build_viewer_metadata(
        subject_record=rec,
        base_nifti_path=tmp_path / "nonexistent.nii.gz",
        base_modality="T1w",
    )
    assert "participant" in result
    assert result["geometry"] == {}


def test_build_viewer_metadata_hc_subject(tmp_path):
    nii = _write_nifti(tmp_path / "flair.nii.gz")
    rec = _make_record(group="hc", metadata={"sex": "M", "split": "test"})
    result = build_viewer_metadata(
        subject_record=rec,
        base_nifti_path=nii,
        base_modality="FLAIR",
    )
    assert result["participant"]["group"] == "hc"
    assert "roi" not in result


# ---------------------------------------------------------------------------
# Integration: manifest round-trip check
# ---------------------------------------------------------------------------

def test_viewer_meta_in_manifest_is_serializable(tmp_path):
    nii = _write_nifti(tmp_path / "flair.nii.gz")
    roi = _write_roi(tmp_path / "roi.nii.gz")
    rec = _make_record(metadata={"sex": "F", "age_scan": "25", "hemisphere": "L", "lobe": "FL"})
    result = build_viewer_metadata(
        subject_record=rec,
        base_nifti_path=nii,
        base_modality="FLAIR",
        roi_nifti_path=roi,
        sidecar_fields={"Manufacturer": "Siemens", "MagneticFieldStrength": 3.0},
    )
    # Should round-trip through JSON without error
    serialized = json.dumps(result)
    parsed = json.loads(serialized)
    assert parsed["participant"]["subject_id"] == "sub-00001"
    assert parsed["roi"]["voxel_count"] > 0
    assert parsed["scan"]["Manufacturer"] == "Siemens"
