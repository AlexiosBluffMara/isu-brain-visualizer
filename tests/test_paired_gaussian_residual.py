"""Tests for paired-support Gaussian splat residuals."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from fcd_prototype.paired_gaussian_residual import build_paired_gaussian_residual
from fcd_prototype.splat_ply_export import _PLY_DTYPE


def _make_pair_volumes(tmp_path: Path, shape=(18, 18, 14)) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(123)
    base = (rng.standard_normal(shape).astype(np.float32) * 0.05 + 0.5).clip(0.0, 1.0)
    mask = np.zeros(shape, dtype=bool)
    mask[3:15, 3:15, 3:11] = True
    normal = np.where(mask, base, 0.0).astype(np.float32)
    patient = normal.copy()
    roi = np.zeros(shape, dtype=np.uint8)
    roi[7:10, 7:10, 6:9] = 1
    patient[roi > 0] = np.clip(patient[roi > 0] + 0.45, 0.0, 1.0)

    affine = np.diag([1.0, 1.0, 1.5, 1.0])
    patient_path = tmp_path / "patient.nii.gz"
    normal_path = tmp_path / "normal.nii.gz"
    roi_path = tmp_path / "roi.nii.gz"
    nib.save(nib.Nifti1Image(patient, affine), str(patient_path))
    nib.save(nib.Nifti1Image(normal, affine), str(normal_path))
    nib.save(nib.Nifti1Image(roi, affine), str(roi_path))
    return patient_path, normal_path, roi_path


def _make_support_basis_volumes(tmp_path: Path, shape=(16, 16, 12)) -> tuple[Path, Path]:
    rng = np.random.default_rng(456)
    patient = np.zeros(shape, dtype=np.float32)
    normal = np.zeros(shape, dtype=np.float32)
    shared = (0.55 + rng.random((6, 6, 5), dtype=np.float32) * 0.25).astype(np.float32)
    patient[3:9, 3:9, 3:8] = shared
    normal[3:9, 3:9, 3:8] = shared
    normal_extra = (0.75 + rng.random((4, 4, 4), dtype=np.float32) * 0.20).astype(np.float32)
    normal[10:14, 10:14, 4:8] = normal_extra

    affine = np.eye(4, dtype=np.float32)
    patient_path = tmp_path / "support_patient.nii.gz"
    normal_path = tmp_path / "support_normal.nii.gz"
    nib.save(nib.Nifti1Image(patient, affine), str(patient_path))
    nib.save(nib.Nifti1Image(normal, affine), str(normal_path))
    return patient_path, normal_path


def _read_ply(path: Path) -> np.ndarray:
    with path.open("rb") as fh:
        header = []
        while True:
            line = fh.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        n = next(int(line.split()[-1]) for line in header if line.startswith("element vertex"))
        return np.frombuffer(fh.read(), dtype=_PLY_DTYPE, count=n)


def test_paired_residual_outputs_share_positions(tmp_path):
    patient, normal, roi = _make_pair_volumes(tmp_path)
    out = tmp_path / "out"

    metrics = build_paired_gaussian_residual(
        patient_volume=patient,
        normal_volume=normal,
        output_dir=out,
        roi=roi,
        max_points=300,
        sampling_mode="uniform",
        rng_seed=7,
    )

    assert metrics["method"] == "paired_correspondence"
    assert metrics["same_gaussian_support"] is True
    assert metrics["n_gaussians"] == 300
    assert metrics["heatmap_method"] == "gaussian"
    assert metrics["heatmap_backend"] == "exact_anisotropic_gaussian"
    for name in (
        "paired_patient_splat.ply",
        "paired_normal_splat.ply",
        "paired_gaussian_residual_splat.ply",
        "paired_gaussian_residual_heatmap.nii.gz",
        "paired_gaussian_residual_scores.npy",
        "paired_gaussian_state.npz",
        "paired_gaussian_residual_metrics.json",
    ):
        assert (out / name).exists()

    patient_arr = _read_ply(out / "paired_patient_splat.ply")
    normal_arr = _read_ply(out / "paired_normal_splat.ply")
    residual_arr = _read_ply(out / "paired_gaussian_residual_splat.ply")
    for field in (
        "x", "y", "z",
        "scale_x", "scale_y", "scale_z",
        "rot_w", "rot_x", "rot_y", "rot_z",
    ):
        np.testing.assert_allclose(patient_arr[field], normal_arr[field])
        np.testing.assert_allclose(patient_arr[field], residual_arr[field])

    q = np.stack(
        [
            patient_arr["rot_w"],
            patient_arr["rot_x"],
            patient_arr["rot_y"],
            patient_arr["rot_z"],
        ],
        axis=1,
    )
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)
    assert metrics["gaussian_orientation"] == "local_gradient_tangent_frame"
    assert metrics["gaussian_scale_model"] == "gradient_oriented_voxel"
    assert metrics["gaussian_state_schema"] == "paired_gaussian_residual_v1"
    assert metrics["artifacts"]["paired_gaussian_state"] == "paired_gaussian_state.npz"
    assert metrics["support_basis_mode"] == "union"
    assert metrics["support_basis"] == "max(patient, synthetic_normal)"

    with np.load(out / "paired_gaussian_state.npz") as state:
        assert str(state["state_schema"]) == "paired_gaussian_residual_v1"
        assert str(state["method"]) == "paired_correspondence"
        assert str(state["support_basis_mode"]) == "union"
        assert str(state["support_basis"]) == "max(patient, synthetic_normal)"
        assert state["positions_world"].shape == (metrics["n_gaussians"], 3)
        assert state["positions_voxel"].shape == (metrics["n_gaussians"], 3)
        assert state["scales"].shape == (metrics["n_gaussians"], 3)
        assert state["rotations_quat"].shape == (metrics["n_gaussians"], 4)
        np.testing.assert_allclose(
            np.linalg.norm(state["rotations_quat"], axis=1),
            1.0,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            state["residual_raw"],
            np.abs(state["patient_intensity"] - state["normal_intensity"]),
            atol=1e-6,
        )
        assert int(state["roi_at_support"].sum()) == metrics["roi_gaussian_count"]


def test_support_basis_controls_shared_gaussian_support(tmp_path):
    patient, normal = _make_support_basis_volumes(tmp_path)

    patient_metrics = build_paired_gaussian_residual(
        patient_volume=patient,
        normal_volume=normal,
        output_dir=tmp_path / "patient_support",
        max_points=10_000,
        percentile_threshold=0.0,
        sampling_mode="uniform",
        support_basis="patient",
        rng_seed=3,
    )
    union_metrics = build_paired_gaussian_residual(
        patient_volume=patient,
        normal_volume=normal,
        output_dir=tmp_path / "union_support",
        max_points=10_000,
        percentile_threshold=0.0,
        sampling_mode="uniform",
        support_basis="union",
        rng_seed=3,
    )

    assert patient_metrics["support_basis_mode"] == "patient"
    assert patient_metrics["support_basis"] == "patient"
    assert union_metrics["support_basis_mode"] == "union"
    assert union_metrics["support_basis"] == "max(patient, synthetic_normal)"
    assert union_metrics["foreground_voxels_total"] > patient_metrics["foreground_voxels_total"]

    with np.load(tmp_path / "patient_support" / "paired_gaussian_state.npz") as state:
        assert not np.any(state["positions_voxel"][:, 0] >= 10)
    with np.load(tmp_path / "union_support" / "paired_gaussian_state.npz") as state:
        assert np.any(state["positions_voxel"][:, 0] >= 10)


def test_identical_volumes_have_zero_paired_residual(tmp_path):
    patient, _normal, roi = _make_pair_volumes(tmp_path)
    out = tmp_path / "identical"

    metrics = build_paired_gaussian_residual(
        patient_volume=patient,
        normal_volume=patient,
        output_dir=out,
        roi=roi,
        max_points=250,
        sampling_mode="uniform",
        rng_seed=8,
    )

    assert metrics["residual_max"] == 0.0
    assert metrics["residual_mean"] == 0.0
    assert metrics["residual_normalization_value"] == 0.0
    assert metrics["residual_nonzero_fraction"] == 0.0
    heatmap = nib.load(str(out / "paired_gaussian_residual_heatmap.nii.gz")).get_fdata()
    assert float(heatmap.max()) == 0.0


def test_volume_mode_builds_full_foreground_paired_support(tmp_path):
    patient, normal, roi = _make_pair_volumes(tmp_path)
    out = tmp_path / "volume"

    metrics = build_paired_gaussian_residual(
        patient_volume=patient,
        normal_volume=normal,
        output_dir=out,
        roi=roi,
        max_points=500,
        sampling_mode="volume",
        rng_seed=12,
    )

    assert metrics["sampling_mode"] == "volume"
    assert metrics["effective_percentile_threshold"] >= 20.0
    assert metrics["gaussian_scale_model"] == "coverage_balanced_density_adaptive_volume_kernel"
    assert metrics["sampling_weight_model"] == "full_foreground_base_plus_intensity_gradient"
    assert metrics["volume_sampling_strategy"] == "foreground_grid_stratified_weighted"
    assert metrics["stratified_occupied_cells_total"] >= metrics["stratified_cells_sampled"] > 0
    assert metrics["foreground_mask_model"] == "largest_connected_tissue_component"
    assert metrics["cleaned_foreground_voxels_total"] == metrics["foreground_voxels_total"]
    assert metrics["volume_voxels_total"] == metrics["foreground_voxels_total"]
    assert metrics["density_scale_factor"] >= 1.35
    assert metrics["density_scale_factor"] <= 2.25
    assert metrics["density_voxels_per_splat"] >= 1.0

    patient_arr = _read_ply(out / "paired_patient_splat.ply")
    assert float(patient_arr["opacity"].min()) >= 0.005
    assert float(patient_arr["opacity"].max()) <= 0.55


def test_roi_patch_has_higher_paired_residual(tmp_path):
    patient, normal, roi = _make_pair_volumes(tmp_path)
    out = tmp_path / "roi"

    metrics = build_paired_gaussian_residual(
        patient_volume=patient,
        normal_volume=normal,
        output_dir=out,
        roi=roi,
        max_points=800,
        sampling_mode="uniform",
        rng_seed=9,
    )

    assert metrics["roi_metrics_valid"] is True
    assert metrics["roi_residual_mean"] > metrics["non_roi_residual_mean"]
    assert metrics["roi_to_non_roi_ratio"] > 1.0
    assert metrics["residual_p99"] > 0.0
    assert metrics["residual_normalization_value"] > 0.0
    assert metrics["residual_nonzero_fraction"] > 0.0
    assert metrics["heatmap_method"] == "gaussian"

    on_disk = json.loads((out / "paired_gaussian_residual_metrics.json").read_text())
    assert on_disk["roi_residual_concentration"] == metrics["roi_residual_concentration"]

    heatmap = nib.load(str(out / "paired_gaussian_residual_heatmap.nii.gz")).get_fdata(
        dtype=np.float32
    )
    roi_data = nib.load(str(roi)).get_fdata(dtype=np.float32) > 0
    assert float(heatmap[roi_data].mean()) > float(heatmap[~roi_data].mean())


def test_paired_gaussian_heatmap_falls_back_for_large_active_support(tmp_path):
    patient, normal, roi = _make_pair_volumes(tmp_path)
    out = tmp_path / "fallback"

    metrics = build_paired_gaussian_residual(
        patient_volume=patient,
        normal_volume=normal,
        output_dir=out,
        roi=roi,
        max_points=500,
        sampling_mode="volume",
        rng_seed=12,
        heatmap_gaussian_exact_max_points=0,
    )

    assert metrics["heatmap_method"] == "gaussian"
    assert metrics["heatmap_backend"] == "isotropic_convolution"
    assert metrics["heatmap_active_gaussians"] > 0


def test_normal_volume_is_resampled_to_patient_grid(tmp_path):
    patient, normal, _roi = _make_pair_volumes(tmp_path)
    normal_img = nib.load(str(normal))
    shifted_normal = tmp_path / "normal_shifted.nii.gz"
    shifted_affine = normal_img.affine.copy()
    shifted_affine[0, 3] = 2.0
    nib.save(nib.Nifti1Image(normal_img.get_fdata(dtype=np.float32), shifted_affine), str(shifted_normal))

    metrics = build_paired_gaussian_residual(
        patient_volume=patient,
        normal_volume=shifted_normal,
        output_dir=tmp_path / "resampled",
        max_points=100,
        sampling_mode="uniform",
        rng_seed=10,
    )

    assert metrics["normal_resampled_to_patient_grid"] is True
