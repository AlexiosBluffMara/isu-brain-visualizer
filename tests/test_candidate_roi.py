"""Tests for candidate ROI generation."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fcd_prototype.candidate_roi import generate_candidate_rois


def _make_nifti_with_blob(path: Path, shape=(40, 40, 32), blob_center=(20, 20, 16),
                          blob_radius_mm=8.0, affine=None, seed=0):
    """Create NIfTI with a bright blob at blob_center."""
    if affine is None:
        affine = np.diag([1.0, 1.0, 1.5, 1.0])

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal(shape).astype(np.float32) * 0.1 + 0.5).clip(0.0, 1.0)

    # Create bright blob in voxel space
    i, j, k = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt(
        ((i - blob_center[0]) ** 2) +
        ((j - blob_center[1]) ** 2) +
        ((k - blob_center[2]) ** 2)
    )
    blob_vox_radius = blob_radius_mm / 1.0  # voxel size ~ 1mm in first two dims
    blob_mask = dist <= blob_vox_radius
    data[blob_mask] = 0.95

    img = nib.Nifti1Image(data.astype(np.float32), affine)
    img.header.set_data_dtype(np.float32)
    nib.save(img, str(path))
    return path


def test_generate_candidate_rois_creates_output_dir(tmp_path):
    volume = _make_nifti_with_blob(tmp_path / "volume.nii.gz")
    output_dir = tmp_path / "output"
    assert not output_dir.exists()

    generate_candidate_rois(volume, output_dir)
    assert output_dir.exists()


def test_generate_candidate_rois_writes_union(tmp_path):
    volume = _make_nifti_with_blob(tmp_path / "volume.nii.gz")
    output_dir = tmp_path / "output"

    manifest = generate_candidate_rois(volume, output_dir)
    union_path = output_dir / "candidate_roi_union.nii.gz"
    assert union_path.exists()
    assert manifest["union_mask"] == "candidate_roi_union.nii.gz"


def test_generate_candidate_rois_writes_manifest(tmp_path):
    volume = _make_nifti_with_blob(tmp_path / "volume.nii.gz")
    output_dir = tmp_path / "output"

    manifest = generate_candidate_rois(volume, output_dir)
    manifest_path = output_dir / "candidate_roi_manifest.json"
    assert manifest_path.exists()

    with manifest_path.open() as fh:
        loaded = json.load(fh)
    assert loaded == manifest


def test_generate_candidate_rois_detects_blob(tmp_path):
    """Candidate generator should detect the bright blob; at least 1 candidate selected."""
    volume = _make_nifti_with_blob(tmp_path / "volume.nii.gz")
    output_dir = tmp_path / "output"

    manifest = generate_candidate_rois(volume, output_dir, max_candidates=5, percentile=95.0)
    assert manifest["selected_count"] >= 1, "Should detect at least one bright region"


def test_generate_candidate_rois_centers_near_blob(tmp_path):
    """First candidate center should be near the blob center."""
    blob_center_mm = (20.0, 20.0, 24.0)  # voxel (20,20,16) in mm with affine diag(1,1,1.5,1)
    volume = _make_nifti_with_blob(
        tmp_path / "volume.nii.gz",
        blob_center=(20, 20, 16),
        affine=np.diag([1.0, 1.0, 1.5, 1.0]),
    )
    output_dir = tmp_path / "output"

    manifest = generate_candidate_rois(volume, output_dir, max_candidates=5, percentile=95.0)
    if manifest["selected_count"] >= 1:
        center_world = np.array(manifest["centers_world"][0])
        # Allow 5mm tolerance due to heuristic proposal
        dist = np.linalg.norm(center_world - np.array(blob_center_mm))
        assert dist < 5.0, f"First candidate center {center_world} too far from blob {blob_center_mm}"


def test_generate_candidate_rois_asymmetry_hybrid_records_and_selects_unilateral_blob(tmp_path):
    shape = (41, 40, 32)
    affine = np.diag([1.0, 1.0, 1.5, 1.0])
    rng = np.random.default_rng(23)
    data = (rng.normal(loc=0.50, scale=0.04, size=shape)).astype(np.float32)
    data = np.clip(data, 0.30, 0.70)
    blob_center = np.array([10, 20, 16])
    i, j, k = np.ogrid[: shape[0], : shape[1], : shape[2]]
    blob = (
        (i - blob_center[0]) ** 2
        + (j - blob_center[1]) ** 2
        + (k - blob_center[2]) ** 2
    ) <= 4**2
    data[blob] = 0.95

    volume = tmp_path / "asymmetric_flair.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(volume))

    manifest = generate_candidate_rois(
        volume,
        tmp_path / "out",
        max_candidates=1,
        radius_mm=5.0,
        percentile=99.0,
        proposal_mode="asymmetry-hybrid",
    )

    assert manifest["proposal_mode"] == "asymmetry-hybrid"
    weights = manifest["proposal_feature_weights"]
    assert weights["left_right_asymmetry"] > 0.0
    assert manifest["proposal_feature_summary"]["left_right_asymmetry_max"] > 0.0
    assert manifest["selected_count"] == 1
    center_vox = np.array(manifest["centers_voxel"][0])
    assert np.linalg.norm(center_vox - blob_center.astype(float)) < 6.0


def test_generate_candidate_rois_individual_masks(tmp_path):
    volume = _make_nifti_with_blob(tmp_path / "volume.nii.gz")
    output_dir = tmp_path / "output"

    manifest = generate_candidate_rois(volume, output_dir, max_candidates=5)
    for i in range(manifest["selected_count"]):
        fname = f"candidate_roi_{i:02d}.nii.gz"
        assert (output_dir / fname).exists()
        assert fname in manifest["individual_masks"]


def test_generate_candidate_rois_separation_constraint(tmp_path):
    """Multiple candidates should be separated by at least min_separation_mm."""
    volume = _make_nifti_with_blob(tmp_path / "volume.nii.gz")
    output_dir = tmp_path / "output"

    min_sep = 30.0
    manifest = generate_candidate_rois(
        volume, output_dir, max_candidates=10, percentile=90.0, min_separation_mm=min_sep
    )

    centers = np.array(manifest["centers_world"])
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = np.linalg.norm(centers[i] - centers[j])
            assert dist >= min_sep * 0.95, f"Candidates {i} and {j} too close: {dist}mm"


def test_generate_candidate_rois_manifest_has_limitations(tmp_path):
    volume = _make_nifti_with_blob(tmp_path / "volume.nii.gz")
    output_dir = tmp_path / "output"

    manifest = generate_candidate_rois(volume, output_dir)
    assert "limitations" in manifest
    assert len(manifest["limitations"]) > 0
    # Check that the first limitation warns it's heuristic, not validated
    assert "heuristic" in manifest["limitations"][0].lower()


def test_generate_candidate_rois_all_zero_volume(tmp_path):
    """All-zero volume should produce empty manifest with selected_count=0, no individual masks, and empty union."""
    affine = np.diag([1.0, 1.0, 1.5, 1.0])
    path = tmp_path / "volume.nii.gz"
    path.parent.mkdir(parents=True, exist_ok=True)

    data = np.zeros((40, 40, 32), dtype=np.float32)
    img = nib.Nifti1Image(data, affine)
    img.header.set_data_dtype(np.float32)
    nib.save(img, str(path))

    output_dir = tmp_path / "output"
    manifest = generate_candidate_rois(path, output_dir)

    # Should have no candidates
    assert manifest["selected_count"] == 0
    assert len(manifest["individual_masks"]) == 0
    assert manifest["centers_voxel"] == []
    assert manifest["centers_world"] == []

    # Union should exist but be empty
    union_path = output_dir / "candidate_roi_union.nii.gz"
    assert union_path.exists()
    union_img = nib.load(str(union_path))
    union_data = union_img.get_fdata(dtype=np.float32)
    assert np.allclose(union_data, 0.0), "Union mask should be all zeros"


def test_candidate_roi_main_wires_cli_arguments(monkeypatch, tmp_path, capsys):
    from fcd_prototype import candidate_roi as mod

    calls = []

    def fake_generate_candidate_rois(**kwargs):
        calls.append(kwargs)
        return {"selected_count": 3, "centers_world": []}

    monkeypatch.setattr(mod, "generate_candidate_rois", fake_generate_candidate_rois)

    mod.main([
        str(tmp_path / "volume.nii.gz"),
        "--output-dir", str(tmp_path / "candidates"),
        "--max-candidates", "3",
        "--radius-mm", "6.5",
        "--min-separation-mm", "11.0",
        "--percentile", "98.0",
        "--proposal-mode", "asymmetry-hybrid",
    ])

    assert calls == [{
        "volume_path": tmp_path / "volume.nii.gz",
        "output_dir": tmp_path / "candidates",
        "max_candidates": 3,
        "radius_mm": 6.5,
        "min_separation_mm": 11.0,
        "percentile": 98.0,
        "proposal_mode": "asymmetry-hybrid",
    }]
    out = capsys.readouterr().out
    assert "Selected 3 candidate(s)" in out
    assert '"selected_count": 3' in out
