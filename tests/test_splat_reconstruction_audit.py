"""Tests for MRI Gaussian PLY reconstruction audit."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fcd_prototype.nifti_preview import _robust_normalize
from fcd_prototype.splat_ply_export import _PLY_DTYPE, _PLY_PROPERTIES, export_splat_ply
from fcd_prototype.splat_reconstruction_audit import audit_splat_reconstruction


def _save_nifti(path: Path, data: np.ndarray, affine=None) -> Path:
    if affine is None:
        affine = np.diag([1.0, 1.0, 1.5, 1.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), str(path))
    return path


def _write_ply(
    path: Path,
    voxels: np.ndarray,
    intensities: np.ndarray,
    affine: np.ndarray,
    *,
    scale: float = 1.0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    hom = np.concatenate([voxels.astype(np.float64), np.ones((len(voxels), 1))], axis=1)
    world = (affine @ hom.T).T[:, :3]
    arr = np.zeros(len(voxels), dtype=_PLY_DTYPE)
    gray = np.clip(intensities * 255, 0, 255).astype(np.uint8)
    arr["x"] = world[:, 0].astype(np.float32)
    arr["y"] = world[:, 1].astype(np.float32)
    arr["z"] = world[:, 2].astype(np.float32)
    arr["red"] = gray
    arr["green"] = gray
    arr["blue"] = gray
    arr["opacity"] = 1.0
    arr["scale_x"] = scale
    arr["scale_y"] = scale
    arr["scale_z"] = scale
    arr["rot_w"] = 1.0
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(voxels)}\n"
        + _PLY_PROPERTIES
        + "end_header\n"
    ).encode("ascii")
    with path.open("wb") as fh:
        fh.write(header)
        fh.write(arr.tobytes())
    return path


def test_splat_reconstruction_audit_exact_voxel_samples(tmp_path):
    affine = np.diag([1.0, 1.0, 1.5, 1.0])
    data = np.arange(1, 65, dtype=np.float32).reshape(4, 4, 4)
    volume = _save_nifti(tmp_path / "vol.nii.gz", data, affine)
    norm = _robust_normalize(data)
    mask = norm > 0.01
    voxels = np.argwhere(mask)
    ply = _write_ply(tmp_path / "vol.ply", voxels, norm[mask], affine)

    result = audit_splat_reconstruction(
        reference_volume=volume,
        splat_ply=ply,
        output_json=tmp_path / "audit.json",
        output_nifti=tmp_path / "recon.nii.gz",
        kernel_mode="nearest",
        smooth_sigma_vox=0.0,
        max_points=None,
    )

    assert result["foreground_coverage"] == pytest.approx(1.0)
    assert result["all_foreground_metrics"]["mae"] < 1 / 255
    assert result["represented_foreground_metrics"]["mae"] < 1 / 255
    assert result["mri_to_3dgs_readiness_status"] == "review"
    assert result["mri_to_3dgs_claim_allowed"] is False
    assert (
        "audit_did_not_use_stored_gaussian_kernel"
        in result["representation_readiness_gate"]["warnings"]
    )
    assert (
        "coordinate_contract_state_missing"
        in result["representation_readiness_gate"]["warnings"]
    )
    assert (tmp_path / "audit.json").exists()
    assert (tmp_path / "recon.nii.gz").exists()


def test_splat_reconstruction_audit_reports_partial_coverage(tmp_path):
    affine = np.eye(4)
    data = np.arange(1, 65, dtype=np.float32).reshape(4, 4, 4)
    volume = _save_nifti(tmp_path / "vol.nii.gz", data, affine)
    norm = _robust_normalize(data)
    voxels = np.array([[3, 3, 3]], dtype=np.int64)
    ply = _write_ply(tmp_path / "one.ply", voxels, np.array([norm[3, 3, 3]]), affine)

    result = audit_splat_reconstruction(
        reference_volume=volume,
        splat_ply=ply,
        kernel_mode="nearest",
        smooth_sigma_vox=0.0,
        max_points=None,
    )

    assert 0.0 < result["foreground_coverage"] < 1.0
    assert result["represented_foreground_metrics"]["voxel_count"] == 1
    assert result["all_foreground_metrics"]["mae"] > 0.0
    assert result["mri_to_3dgs_readiness_status"] == "fail"
    assert result["mri_to_3dgs_claim_allowed"] is False
    assert (
        "foreground_coverage_below_review_floor"
        in result["representation_readiness_gate"]["blockers"]
    )


def test_splat_reconstruction_audit_gaussian_kernel_uses_scale(tmp_path):
    affine = np.eye(4)
    data = np.zeros((7, 7, 7), dtype=np.float32)
    data[3, 3, 3] = 10.0
    data[3, 3, 4] = 5.0
    volume = _save_nifti(tmp_path / "vol.nii.gz", data, affine)
    voxels = np.array([[3, 3, 3]], dtype=np.int64)

    small = _write_ply(
        tmp_path / "small.ply",
        voxels,
        np.array([1.0], dtype=np.float32),
        affine,
        scale=0.25,
    )
    large = _write_ply(
        tmp_path / "large.ply",
        voxels,
        np.array([1.0], dtype=np.float32),
        affine,
        scale=1.5,
    )

    small_result = audit_splat_reconstruction(
        reference_volume=volume,
        splat_ply=small,
        kernel_mode="gaussian",
        gaussian_sigma_extent=2.0,
        smooth_sigma_vox=0.0,
        max_points=None,
    )
    large_result = audit_splat_reconstruction(
        reference_volume=volume,
        splat_ply=large,
        kernel_mode="gaussian",
        gaussian_sigma_extent=2.0,
        smooth_sigma_vox=0.0,
        max_points=None,
    )

    assert small_result["kernel_mode"] == "gaussian"
    assert large_result["kernel_mode"] == "gaussian"
    assert large_result["represented_voxels"] > small_result["represented_voxels"]
    assert large_result["gaussian_sigma_extent"] == pytest.approx(2.0)


def test_splat_reconstruction_audit_passes_dense_gaussian_representation(tmp_path):
    affine = np.eye(4)
    x, y, z = np.meshgrid(
        np.arange(16),
        np.arange(16),
        np.arange(16),
        indexing="ij",
    )
    center = np.array([7.5, 7.5, 7.5])
    radius = 6.1
    distance = np.sqrt(
        (x - center[0]) ** 2
        + (y - center[1]) ** 2
        + (z - center[2]) ** 2
    )
    data = (0.15 + 0.85 * np.clip(1.0 - distance / radius, 0.0, 1.0)).astype(
        np.float32
    )
    data[distance > radius] = 0.0
    volume = _save_nifti(tmp_path / "vol.nii.gz", data, affine)
    ply = tmp_path / "dense.ply"
    export_splat_ply(
        volume,
        ply,
        max_points=10_000,
        percentile_threshold=0.0,
        sampling_mode="volume",
        rng_seed=1,
    )

    result = audit_splat_reconstruction(
        reference_volume=volume,
        splat_ply=ply,
        kernel_mode="gaussian",
        gaussian_sigma_extent=2.0,
        smooth_sigma_vox=0.0,
        max_points=None,
    )

    assert result["foreground_coverage"] == pytest.approx(1.0)
    assert result["mri_to_3dgs_readiness_status"] == "pass"
    assert result["mri_to_3dgs_claim_allowed"] is True
    assert result["mri_to_3dgs_claim_class"] == "volumetric_mri_gaussian_representation"
    assert result["coordinate_contract"]["status"] == "pass"
    observed = result["coordinate_contract"]["observed"]
    assert observed["affine_max_abs_diff_mm"] <= 1e-5
    assert observed["voxel_to_world_max_abs_diff_mm"] <= 1e-5
    assert observed["state_voxel_positions_in_reference_bounds_fraction"] == pytest.approx(1.0)
    gate = result["representation_readiness_gate"]
    assert gate["blocker_count"] == 0
    assert gate["warning_count"] == 0


def test_splat_reconstruction_audit_blocks_coordinate_contract_mismatch(tmp_path):
    affine = np.eye(4)
    x, y, z = np.meshgrid(
        np.arange(16),
        np.arange(16),
        np.arange(16),
        indexing="ij",
    )
    center = np.array([7.5, 7.5, 7.5])
    radius = 6.1
    distance = np.sqrt(
        (x - center[0]) ** 2
        + (y - center[1]) ** 2
        + (z - center[2]) ** 2
    )
    data = (0.15 + 0.85 * np.clip(1.0 - distance / radius, 0.0, 1.0)).astype(
        np.float32
    )
    data[distance > radius] = 0.0
    volume = _save_nifti(tmp_path / "vol.nii.gz", data, affine)
    ply = tmp_path / "dense.ply"
    export_splat_ply(
        volume,
        ply,
        max_points=10_000,
        percentile_threshold=0.0,
        sampling_mode="volume",
        rng_seed=1,
    )

    state_path = tmp_path / "dense_state.npz"
    with np.load(state_path, allow_pickle=False) as loaded:
        state = {key: loaded[key] for key in loaded.files}
    shifted_affine = np.array(state["affine"], copy=True)
    shifted_affine[0, 3] += 4.0
    state["affine"] = shifted_affine
    np.savez_compressed(state_path, **state)

    result = audit_splat_reconstruction(
        reference_volume=volume,
        splat_ply=ply,
        kernel_mode="gaussian",
        gaussian_sigma_extent=2.0,
        smooth_sigma_vox=0.0,
        max_points=None,
    )

    assert result["foreground_coverage"] == pytest.approx(1.0)
    assert result["coordinate_contract"]["status"] == "fail"
    assert (
        "coordinate_contract_affine_mismatch"
        in result["coordinate_contract"]["blockers"]
    )
    assert result["mri_to_3dgs_readiness_status"] == "fail"
    assert result["mri_to_3dgs_claim_allowed"] is False
