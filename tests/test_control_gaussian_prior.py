"""Tests for patient-support healthy-control Gaussian prior residuals."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fcd_prototype.control_gaussian_prior import (
    _load_control_list,
    build_control_gaussian_prior_residual,
)
from fcd_prototype.splat_ply_export import _PLY_DTYPE


def _make_patient_controls(
    tmp_path: Path,
    *,
    lesion: bool = False,
    shape: tuple[int, int, int] = (18, 18, 14),
) -> tuple[Path, list[Path], Path]:
    rng = np.random.default_rng(910)
    base = np.zeros(shape, dtype=np.float32)
    foreground = np.zeros(shape, dtype=bool)
    foreground[3:15, 3:15, 3:11] = True
    texture = (
        rng.standard_normal(shape).astype(np.float32) * 0.02
        + 0.50
        + np.linspace(0.0, 0.12, shape[0], dtype=np.float32)[:, None, None]
    )
    base[foreground] = texture[foreground]
    base = np.clip(base, 0.0, 1.0)

    roi = np.zeros(shape, dtype=np.uint8)
    roi[7:10, 7:10, 6:9] = 1
    patient = base.copy()
    if lesion:
        patient[roi > 0] = np.clip(patient[roi > 0] + 0.35, 0.0, 1.0)

    affine = np.diag([1.0, 1.0, 1.5, 1.0]).astype(np.float32)
    patient_path = tmp_path / "patient.nii.gz"
    roi_path = tmp_path / "roi.nii.gz"
    nib.save(nib.Nifti1Image(patient, affine), str(patient_path))
    nib.save(nib.Nifti1Image(roi, affine), str(roi_path))

    controls: list[Path] = []
    for idx, offset in enumerate((-0.015, 0.015)):
        control = np.clip(base + offset, 0.0, 1.0).astype(np.float32)
        path = tmp_path / f"control_{idx}.nii.gz"
        nib.save(nib.Nifti1Image(control, affine), str(path))
        controls.append(path)
    return patient_path, controls, roi_path


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


def test_identical_patient_control_prior_has_zero_residual(tmp_path):
    patient, _controls, roi = _make_patient_controls(tmp_path, lesion=False)
    controls = [patient, patient]

    metrics = build_control_gaussian_prior_residual(
        patient_volume=patient,
        control_volumes=controls,
        output_dir=tmp_path / "identity",
        roi=roi,
        max_points=300,
        sampling_mode="uniform",
        rng_seed=3,
    )

    assert metrics["method"] == "patient_support_control_prior_zscore"
    assert metrics["gaussian_state_schema"] == "patient_control_gaussian_prior_v1"
    assert metrics["same_gaussian_support"] is True
    assert metrics["n_controls"] == 2
    assert metrics["residual_z_max"] == pytest.approx(0.0)
    assert metrics["residual_z_mean"] == pytest.approx(0.0)
    assert metrics["residual_z_normalization_value"] == pytest.approx(0.0)
    heatmap = nib.load(
        str(tmp_path / "identity" / "control_prior_z_residual_heatmap.nii.gz")
    ).get_fdata(dtype=np.float32)
    assert float(heatmap.max()) == pytest.approx(0.0)


def test_control_prior_residual_localizes_synthetic_lesion(tmp_path):
    patient, controls, roi = _make_patient_controls(tmp_path, lesion=True)
    out = tmp_path / "lesion"

    metrics = build_control_gaussian_prior_residual(
        patient_volume=patient,
        control_volumes=controls,
        output_dir=out,
        roi=roi,
        max_points=600,
        sampling_mode="uniform",
        rng_seed=4,
    )

    for name in (
        "control_prior_patient_splat.ply",
        "control_prior_mean_splat.ply",
        "control_prior_z_residual_splat.ply",
        "control_prior_z_residual_heatmap.nii.gz",
        "control_prior_z_scores.npy",
        "control_gaussian_prior_state.npz",
        "control_gaussian_prior_metrics.json",
    ):
        assert (out / name).exists()

    patient_arr = _read_ply(out / "control_prior_patient_splat.ply")
    prior_arr = _read_ply(out / "control_prior_mean_splat.ply")
    residual_arr = _read_ply(out / "control_prior_z_residual_splat.ply")
    for field in (
        "x", "y", "z",
        "scale_x", "scale_y", "scale_z",
        "rot_w", "rot_x", "rot_y", "rot_z",
    ):
        np.testing.assert_allclose(patient_arr[field], prior_arr[field])
        np.testing.assert_allclose(patient_arr[field], residual_arr[field])

    assert metrics["roi_metrics_valid"] is True
    assert metrics["roi_residual_mean"] > metrics["non_roi_residual_mean"]
    assert metrics["roi_to_non_roi_ratio"] > 1.0
    assert metrics["residual_z_p99"] > 0.0

    heatmap = nib.load(
        str(out / "control_prior_z_residual_heatmap.nii.gz")
    ).get_fdata(dtype=np.float32)
    roi_data = nib.load(str(roi)).get_fdata(dtype=np.float32) > 0
    assert float(heatmap[roi_data].mean()) > float(heatmap[~roi_data].mean())

    with np.load(out / "control_gaussian_prior_state.npz") as state:
        assert str(state["state_schema"]) == "patient_control_gaussian_prior_v1"
        assert state["positions_world"].shape == (metrics["n_gaussians"], 3)
        assert state["control_prior_mean"].shape == (metrics["n_gaussians"],)
        assert state["residual_z"].shape == (metrics["n_gaussians"],)


def test_control_prior_requires_multiple_controls(tmp_path):
    patient, controls, _roi = _make_patient_controls(tmp_path, lesion=False)

    with pytest.raises(ValueError, match="at least 2 existing controls"):
        build_control_gaussian_prior_residual(
            patient_volume=patient,
            control_volumes=[controls[0]],
            output_dir=tmp_path / "single",
            max_points=100,
            sampling_mode="uniform",
        )


def test_shifted_control_is_resampled_to_patient_grid(tmp_path):
    patient, controls, _roi = _make_patient_controls(tmp_path, lesion=False)
    shifted_img = nib.load(str(controls[0]))
    shifted_path = tmp_path / "control_shifted.nii.gz"
    shifted_affine = shifted_img.affine.copy()
    shifted_affine[0, 3] = 2.0
    nib.save(
        nib.Nifti1Image(
            shifted_img.get_fdata(dtype=np.float32),
            shifted_affine,
        ),
        str(shifted_path),
    )

    metrics = build_control_gaussian_prior_residual(
        patient_volume=patient,
        control_volumes=[shifted_path, controls[1]],
        output_dir=tmp_path / "shifted",
        max_points=150,
        sampling_mode="uniform",
        rng_seed=5,
    )

    assert metrics["control_resampled_to_patient_grid_count"] == 1
    assert any(
        item["resampled_to_patient_grid"]
        for item in metrics["control_metadata"]
    )


def test_control_list_resolves_relative_to_list_file(tmp_path):
    controls_dir = tmp_path / "controls"
    controls_dir.mkdir()
    list_path = controls_dir / "controls.txt"
    list_path.write_text("# comment\ncontrol_a.nii.gz\n")

    controls = _load_control_list(list_path)

    assert controls == [controls_dir / "control_a.nii.gz"]
