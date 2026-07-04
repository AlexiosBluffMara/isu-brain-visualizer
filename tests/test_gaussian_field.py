"""Tests for gaussian_field — MRI → optimized Gaussian field training."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_volume(path: Path, shape=(16, 16, 12), seed=0) -> None:
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal(shape).astype(np.float32) * 0.2 + 0.6).clip(0.0, 1.0)
    affine = np.diag([1.0, 1.0, 1.5, 1.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _make_roi(path: Path, shape=(16, 16, 12)) -> None:
    data = np.zeros(shape, dtype=np.uint8)
    data[4:8, 4:8, 4:8] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.5, 1.0])), str(path))


def _read_binary_ply(path: Path, dtype: np.dtype) -> np.ndarray:
    with open(path, "rb") as fh:
        header = b""
        while True:
            line = fh.readline()
            header += line
            if line.strip() == b"end_header":
                break
        raw = fh.read()
    header_text = header.decode("ascii")
    n = 0
    for line in header_text.splitlines():
        if line.startswith("element vertex"):
            n = int(line.split()[2])
            break
    return np.frombuffer(raw, dtype=dtype, count=n)


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_torch(),
    reason="PyTorch not installed; skipping gaussian_field tests",
)


# ---------------------------------------------------------------------------
# Initialization tests (no training needed)
# ---------------------------------------------------------------------------

def test_init_gaussians_returns_expected_keys(tmp_path):
    from fcd_prototype.gaussian_field import _init_gaussians, _load_canonical, _robust_normalize

    vol = tmp_path / "vol.nii.gz"
    _make_volume(vol)
    data, affine = _load_canonical(vol)
    norm = _robust_normalize(data)

    result = _init_gaussians(norm, affine, None, num_gaussians=50, seed=42)
    for key in ("means_norm", "log_scales", "quat_wxyz", "opacity_logit",
                "colors_raw", "is_roi", "norm_center", "norm_scale", "voxel_indices"):
        assert key in result, f"Missing key: {key}"


def test_init_gaussians_shapes(tmp_path):
    from fcd_prototype.gaussian_field import _init_gaussians, _load_canonical, _robust_normalize

    vol = tmp_path / "vol.nii.gz"
    _make_volume(vol)
    data, affine = _load_canonical(vol)
    norm = _robust_normalize(data)

    n = 50
    result = _init_gaussians(norm, affine, None, num_gaussians=n, seed=7)
    N = len(result["means_norm"])
    assert N <= n
    assert result["means_norm"].shape == (N, 3)
    assert result["log_scales"].shape == (N, 3)
    assert result["quat_wxyz"].shape == (N, 4)
    assert result["opacity_logit"].shape == (N,)
    assert result["colors_raw"].shape == (N, 3)
    assert result["is_roi"].shape == (N,)


def test_init_gaussians_identity_quaternions(tmp_path):
    from fcd_prototype.gaussian_field import _init_gaussians, _load_canonical, _robust_normalize

    vol = tmp_path / "vol.nii.gz"
    _make_volume(vol)
    data, affine = _load_canonical(vol)
    norm = _robust_normalize(data)

    result = _init_gaussians(norm, affine, None, num_gaussians=100, seed=0)
    q = result["quat_wxyz"]
    np.testing.assert_allclose(q[:, 0], np.ones(len(q)), atol=1e-6)
    np.testing.assert_allclose(q[:, 1:], np.zeros((len(q), 3)), atol=1e-6)


def test_init_gaussians_means_bounded(tmp_path):
    from fcd_prototype.gaussian_field import _init_gaussians, _load_canonical, _robust_normalize

    vol = tmp_path / "vol.nii.gz"
    _make_volume(vol)
    data, affine = _load_canonical(vol)
    norm = _robust_normalize(data)

    result = _init_gaussians(norm, affine, None, num_gaussians=200, seed=1)
    means = result["means_norm"]
    assert float(np.abs(means).max()) <= 1.5, "means_norm should be approximately within [-1, 1]^3"


def test_init_gaussians_deterministic(tmp_path):
    from fcd_prototype.gaussian_field import _init_gaussians, _load_canonical, _robust_normalize

    vol = tmp_path / "vol.nii.gz"
    _make_volume(vol)
    data, affine = _load_canonical(vol)
    norm = _robust_normalize(data)

    r1 = _init_gaussians(norm, affine, None, num_gaussians=80, seed=99)
    r2 = _init_gaussians(norm, affine, None, num_gaussians=80, seed=99)
    np.testing.assert_array_equal(r1["means_norm"], r2["means_norm"])
    np.testing.assert_array_equal(r1["voxel_indices"], r2["voxel_indices"])


def test_init_gaussians_jitter_moves_means(tmp_path):
    """With jitter enabled, means should differ from the no-jitter lattice."""
    from fcd_prototype.gaussian_field import _init_gaussians, _load_canonical, _robust_normalize

    vol = tmp_path / "vol.nii.gz"
    _make_volume(vol)
    data, affine = _load_canonical(vol)
    norm = _robust_normalize(data)

    r_jitter = _init_gaussians(norm, affine, None, num_gaussians=80, seed=42, jitter_voxels=True)
    r_no_jitter = _init_gaussians(norm, affine, None, num_gaussians=80, seed=42, jitter_voxels=False)
    # Same integer voxels, different world positions
    np.testing.assert_array_equal(r_jitter["voxel_indices"], r_no_jitter["voxel_indices"])
    assert not np.array_equal(r_jitter["means_norm"], r_no_jitter["means_norm"])


def test_init_gaussians_jitter_deterministic(tmp_path):
    """Same seed produces identical jittered means on repeated calls."""
    from fcd_prototype.gaussian_field import _init_gaussians, _load_canonical, _robust_normalize

    vol = tmp_path / "vol.nii.gz"
    _make_volume(vol)
    data, affine = _load_canonical(vol)
    norm = _robust_normalize(data)

    r1 = _init_gaussians(norm, affine, None, num_gaussians=80, seed=42, jitter_voxels=True)
    r2 = _init_gaussians(norm, affine, None, num_gaussians=80, seed=42, jitter_voxels=True)
    np.testing.assert_array_equal(r1["means_norm"], r2["means_norm"])


def test_init_gaussians_no_jitter_exact_voxel_centers(tmp_path):
    """Without jitter, means must map exactly from integer voxel centers."""
    from fcd_prototype.gaussian_field import _init_gaussians, _load_canonical, _robust_normalize

    vol = tmp_path / "vol.nii.gz"
    _make_volume(vol)
    data, affine = _load_canonical(vol)
    norm = _robust_normalize(data)

    r = _init_gaussians(norm, affine, None, num_gaussians=50, seed=7, jitter_voxels=False)
    # Reconstruct expected world coords from integer voxel indices
    vox = r["voxel_indices"].astype(np.float64)
    ones = np.ones((len(vox), 1))
    expected_world = (affine @ np.hstack([vox, ones]).T).T[:, :3].astype(np.float32)
    center = r["norm_center"]
    scale = float(r["norm_scale"])
    expected_norm = (expected_world - center) / scale
    np.testing.assert_allclose(r["means_norm"], expected_norm, atol=1e-5)


def test_init_gaussians_jitter_recorded(tmp_path):
    """jitter_voxels key is present in the returned dict."""
    from fcd_prototype.gaussian_field import _init_gaussians, _load_canonical, _robust_normalize

    vol = tmp_path / "vol.nii.gz"
    _make_volume(vol)
    data, affine = _load_canonical(vol)
    norm = _robust_normalize(data)

    for flag in (True, False):
        r = _init_gaussians(norm, affine, None, num_gaussians=30, seed=0, jitter_voxels=flag)
        assert "jitter_voxels" in r
        assert r["jitter_voxels"] is flag


def test_init_gaussians_different_seeds_differ(tmp_path):
    from fcd_prototype.gaussian_field import _init_gaussians, _load_canonical, _robust_normalize

    vol = tmp_path / "vol.nii.gz"
    _make_volume(vol)
    data, affine = _load_canonical(vol)
    norm = _robust_normalize(data)

    r1 = _init_gaussians(norm, affine, None, num_gaussians=80, seed=1)
    r2 = _init_gaussians(norm, affine, None, num_gaussians=80, seed=2)
    assert not np.array_equal(r1["means_norm"], r2["means_norm"])


def test_init_gaussians_roi_mapped(tmp_path):
    from fcd_prototype.gaussian_field import _init_gaussians, _load_canonical, _robust_normalize

    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    _make_volume(vol)
    _make_roi(roi)

    data, affine = _load_canonical(vol)
    norm = _robust_normalize(data)
    roi_data, _ = _load_canonical(roi)

    result = _init_gaussians(norm, affine, roi_data, num_gaussians=200, seed=42)
    # Some gaussians should be marked as ROI
    assert result["is_roi"].sum() > 0
    # ROI fraction should not exceed ~50% (anti-collapse guard)
    roi_frac = result["is_roi"].mean()
    assert roi_frac <= 0.6, f"ROI fraction too high: {roi_frac:.2f}"


def test_init_gaussians_roi_not_all(tmp_path):
    """Even with ROI, not all gaussians should collapse into ROI."""
    from fcd_prototype.gaussian_field import _init_gaussians, _load_canonical, _robust_normalize

    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    _make_volume(vol)
    _make_roi(roi)

    data, affine = _load_canonical(vol)
    norm = _robust_normalize(data)
    roi_data, _ = _load_canonical(roi)

    result = _init_gaussians(norm, affine, roi_data, num_gaussians=100, seed=0)
    assert result["is_roi"].sum() < len(result["is_roi"])


# ---------------------------------------------------------------------------
# Target projection tests
# ---------------------------------------------------------------------------

def test_build_targets_returns_all_views(tmp_path):
    from fcd_prototype.gaussian_field import _load_canonical, _robust_normalize, _build_targets

    vol = tmp_path / "vol.nii.gz"
    _make_volume(vol)
    data, _ = _load_canonical(vol)
    norm = _robust_normalize(data)

    targets = _build_targets(norm, 32, ["axial", "coronal", "sagittal"])
    assert set(targets.keys()) == {"axial", "coronal", "sagittal"}
    for v, arr in targets.items():
        assert arr.shape == (32, 32), f"{v} has wrong shape {arr.shape}"
        assert arr.dtype == np.float32


def test_build_targets_values_bounded(tmp_path):
    from fcd_prototype.gaussian_field import _load_canonical, _robust_normalize, _build_targets

    vol = tmp_path / "vol.nii.gz"
    _make_volume(vol)
    data, _ = _load_canonical(vol)
    norm = _robust_normalize(data)

    targets = _build_targets(norm, 32, ["axial"])
    assert float(targets["axial"].min()) >= 0.0
    assert float(targets["axial"].max()) <= 1.0


# ---------------------------------------------------------------------------
# Volume-space sampled reconstruction objective tests
# ---------------------------------------------------------------------------

def test_build_volume_samples_deterministic_and_mixed(tmp_path):
    from fcd_prototype.gaussian_field import (
        _build_volume_samples,
        _init_gaussians,
        _load_canonical,
        _robust_normalize,
    )

    vol = tmp_path / "vol.nii.gz"
    _make_volume(vol)
    data, affine = _load_canonical(vol)
    norm = _robust_normalize(data)
    init = _init_gaussians(norm, affine, None, num_gaussians=40, seed=3)

    kwargs = dict(
        norm=norm,
        affine=affine,
        norm_center=init["norm_center"],
        norm_scale=float(init["norm_scale"]),
        sample_count=64,
        seed=99,
    )
    s1 = _build_volume_samples(**kwargs)
    s2 = _build_volume_samples(**kwargs)

    assert s1["sample_points_norm"].shape == (64, 3)
    assert s1["targets"].shape == (64,)
    assert s1["foreground_count"] > 0
    assert s1["background_count"] > 0
    assert float(s1["targets"].min()) >= 0.0
    assert float(s1["targets"].max()) <= 1.0
    np.testing.assert_array_equal(s1["sample_points_norm"], s2["sample_points_norm"])
    np.testing.assert_array_equal(s1["targets"], s2["targets"])


def test_render_volume_samples_output_shape_and_bounds():
    import torch
    from fcd_prototype.gaussian_field import _render_volume_samples

    means = torch.zeros(8, 3)
    log_scales = torch.full((8, 3), -1.5)
    opacity_logit = torch.zeros(8)
    colors_raw = torch.zeros(8, 3)
    sample_points = torch.zeros(11, 3)

    out = _render_volume_samples(
        means, log_scales, opacity_logit, colors_raw, sample_points
    )
    assert out.shape == (11,)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


# ---------------------------------------------------------------------------
# Full training smoke test (CPU, tiny volume, few steps)
# ---------------------------------------------------------------------------

def _smoke_train(tmp_path: Path, use_roi: bool = False) -> dict:
    from fcd_prototype.gaussian_field import train_gaussian_field

    vol = tmp_path / "vol.nii.gz"
    _make_volume(vol, shape=(12, 12, 10), seed=3)
    roi = None
    if use_roi:
        roi = tmp_path / "roi.nii.gz"
        _make_roi(roi, shape=(12, 12, 10))

    out = tmp_path / "out"
    metrics = train_gaussian_field(
        volume_path=vol,
        output_dir=out,
        roi_path=roi,
        num_gaussians=30,
        steps=3,
        image_size=16,
        views="axial",
        device="cpu",
        seed=42,
        learning_rate=1e-3,
    )
    return metrics


def test_smoke_outputs_created(tmp_path):
    _smoke_train(tmp_path)
    out = tmp_path / "out"
    assert (out / "gaussian_field.pt").exists()
    assert (out / "optimized_gaussian_splat.ply").exists()
    assert (out / "training_metrics.json").exists()
    assert (out / "reconstruction_preview.png").exists()


def test_smoke_metrics_structure(tmp_path):
    metrics = _smoke_train(tmp_path)
    for key in ("volume_path", "num_gaussians", "steps", "backend", "device",
                "seed", "jitter_voxels", "loss_history", "mip_loss_history",
                "final_loss", "final_mip_loss", "training_objective"):
        assert key in metrics, f"Missing key: {key}"
    assert metrics["training_objective"] == "mip_projection_mse"
    assert metrics["volume_sample_count"] == 0
    assert metrics["final_volume_loss"] is None


def test_smoke_backend_is_fallback_on_cpu(tmp_path):
    metrics = _smoke_train(tmp_path)
    assert metrics["backend"] == "fallback"


def test_smoke_loss_history_length(tmp_path):
    metrics = _smoke_train(tmp_path)
    assert len(metrics["loss_history"]) == metrics["steps"]


def test_smoke_loss_history_finite(tmp_path):
    metrics = _smoke_train(tmp_path)
    for v in metrics["loss_history"]:
        assert np.isfinite(v), f"Non-finite loss value: {v}"


def test_smoke_roi_outputs(tmp_path):
    metrics = _smoke_train(tmp_path, use_roi=True)
    assert (tmp_path / "out" / "lesion_gaussian_mask.npy").exists()
    assert "lesion_gaussian_count" in metrics
    assert "lesion_gaussian_fraction" in metrics
    assert 0.0 <= metrics["lesion_gaussian_fraction"] <= 1.0


def test_smoke_metrics_json_on_disk(tmp_path):
    _smoke_train(tmp_path)
    metrics_path = tmp_path / "out" / "training_metrics.json"
    with metrics_path.open() as fh:
        on_disk = json.load(fh)
    assert "loss_history" in on_disk
    assert "backend" in on_disk


def test_smoke_deterministic(tmp_path):
    """Two runs with same seed must produce identical loss histories."""
    from fcd_prototype.gaussian_field import train_gaussian_field

    vol = tmp_path / "vol.nii.gz"
    _make_volume(vol, shape=(10, 10, 8), seed=5)

    def run(out_name):
        return train_gaussian_field(
            volume_path=vol,
            output_dir=tmp_path / out_name,
            num_gaussians=20,
            steps=2,
            image_size=16,
            views="axial",
            device="cpu",
            seed=11,
            learning_rate=1e-3,
        )

    m1 = run("out1")
    m2 = run("out2")
    assert m1["loss_history"] == m2["loss_history"]


def test_smoke_volume_loss_metrics(tmp_path):
    from fcd_prototype.gaussian_field import train_gaussian_field

    vol = tmp_path / "vol.nii.gz"
    _make_volume(vol, shape=(10, 10, 8), seed=6)
    metrics = train_gaussian_field(
        volume_path=vol,
        output_dir=tmp_path / "out",
        num_gaussians=20,
        steps=2,
        image_size=16,
        views="axial",
        device="cpu",
        seed=12,
        learning_rate=1e-3,
        volume_loss_weight=0.25,
        volume_sample_count=32,
    )

    assert metrics["training_objective"] == "mip_projection_mse_plus_volume_sample_mse"
    assert metrics["volume_loss_weight"] == pytest.approx(0.25)
    assert metrics["volume_sample_count"] == 32
    assert metrics["volume_sample_foreground_count"] > 0
    assert metrics["volume_sample_background_count"] > 0
    assert len(metrics["mip_loss_history"]) == metrics["steps"]
    assert len(metrics["volume_loss_history"]) == metrics["steps"]
    assert np.isfinite(metrics["final_mip_loss"])
    assert np.isfinite(metrics["final_volume_loss"])


def test_project_volume_to_gaussian_support_preserves_geometry(tmp_path):
    import torch
    from fcd_prototype.gaussian_field import (
        project_volume_to_gaussian_support,
        train_gaussian_field,
    )

    patient_vol = tmp_path / "patient.nii.gz"
    normal_vol = tmp_path / "normal.nii.gz"
    _make_volume(patient_vol, shape=(10, 10, 8), seed=7)
    _make_volume(normal_vol, shape=(10, 10, 8), seed=8)

    patient_out = tmp_path / "patient_field"
    train_gaussian_field(
        volume_path=patient_vol,
        output_dir=patient_out,
        num_gaussians=20,
        steps=1,
        image_size=16,
        views="axial",
        device="cpu",
        seed=21,
    )

    normal_out = tmp_path / "normal_projected"
    metrics = project_volume_to_gaussian_support(
        support_field=patient_out / "gaussian_field.pt",
        volume_path=normal_vol,
        output_dir=normal_out,
        preview_image_size=16,
    )

    assert metrics["backend"] == "support_projection"
    assert metrics["training_objective"] == "projected_volume_on_existing_gaussian_support"
    assert metrics["support_projection"] is True
    assert metrics["support_fingerprint_schema"] == "gaussian_support_v1"
    assert len(metrics["support_fingerprint_sha256"]) == 64
    assert metrics["support_fingerprint_sha256"] == metrics["same_gaussian_support_as_fingerprint"]
    assert metrics["support_points_in_volume_bounds_fraction"] > 0.0
    assert (normal_out / "gaussian_field.pt").exists()
    assert (normal_out / "optimized_gaussian_splat.ply").exists()
    assert (normal_out / "training_metrics.json").exists()

    patient_state = torch.load(str(patient_out / "gaussian_field.pt"), weights_only=False)
    normal_state = torch.load(str(normal_out / "gaussian_field.pt"), weights_only=False)
    for key in ("means_norm", "log_scales", "quat_wxyz"):
        np.testing.assert_allclose(
            patient_state[key].detach().cpu().numpy(),
            normal_state[key].detach().cpu().numpy(),
            atol=1e-6,
        )
    assert normal_state["support_projection"] is True
    assert patient_state["support_fingerprint_schema"] == "gaussian_support_v1"
    assert normal_state["support_fingerprint_schema"] == "gaussian_support_v1"
    assert patient_state["support_fingerprint_sha256"] == normal_state["support_fingerprint_sha256"]
    assert normal_state["same_gaussian_support_as_fingerprint"] == patient_state["support_fingerprint_sha256"]


def test_smoke_ply_header(tmp_path):
    """PLY output must be binary_little_endian with required properties."""
    _smoke_train(tmp_path)
    ply = tmp_path / "out" / "optimized_gaussian_splat.ply"
    with open(ply, "rb") as fh:
        header = b""
        while True:
            line = fh.readline()
            header += line
            if line.strip() == b"end_header":
                break
    hstr = header.decode("ascii")
    assert "binary_little_endian" in hstr
    for prop in (
        "x", "y", "z", "red", "green", "blue", "opacity",
        "scale_x", "scale_y", "scale_z",
        "rot_w", "rot_x", "rot_y", "rot_z",
    ):
        assert prop in hstr


def test_smoke_ply_exports_normalized_quaternions(tmp_path):
    from fcd_prototype.gaussian_field import _PLY_DTYPE

    _smoke_train(tmp_path)
    arr = _read_binary_ply(tmp_path / "out" / "optimized_gaussian_splat.ply", _PLY_DTYPE)
    for prop in ("rot_w", "rot_x", "rot_y", "rot_z"):
        assert prop in arr.dtype.names
    q = np.stack([arr["rot_w"], arr["rot_x"], arr["rot_y"], arr["rot_z"]], axis=1)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)


def test_smoke_pt_loadable(tmp_path):
    import torch
    _smoke_train(tmp_path)
    state = torch.load(str(tmp_path / "out" / "gaussian_field.pt"), weights_only=False)
    for key in ("means_norm", "log_scales", "quat_wxyz", "opacity_logit", "colors_raw",
                "backend", "device", "seed", "jitter_voxels"):
        assert key in state


def test_smoke_multiple_views(tmp_path):
    from fcd_prototype.gaussian_field import train_gaussian_field

    vol = tmp_path / "vol.nii.gz"
    _make_volume(vol, shape=(10, 10, 8), seed=2)
    metrics = train_gaussian_field(
        volume_path=vol,
        output_dir=tmp_path / "out",
        num_gaussians=20,
        steps=2,
        image_size=16,
        views="axial,coronal,sagittal",
        device="cpu",
        seed=42,
    )
    assert set(metrics["views"]) == {"axial", "coronal", "sagittal"}


def test_smoke_no_roi_no_mask_file(tmp_path):
    _smoke_train(tmp_path, use_roi=False)
    assert not (tmp_path / "out" / "lesion_gaussian_mask.npy").exists()
    metrics_path = tmp_path / "out" / "training_metrics.json"
    with metrics_path.open() as fh:
        on_disk = json.load(fh)
    assert "lesion_gaussian_count" not in on_disk


# ---------------------------------------------------------------------------
# Fallback renderer unit test
# ---------------------------------------------------------------------------

def test_fallback_render_output_shape():
    import torch
    from fcd_prototype.gaussian_field import _fallback_render

    N = 10
    means = torch.zeros(N, 3)
    log_scales = torch.full((N, 3), -2.0)
    opacity_logit = torch.zeros(N)
    colors_raw = torch.zeros(N, 3)

    out = _fallback_render(means, log_scales, opacity_logit, colors_raw, 16, "axial")
    assert out.shape == (16, 16, 3)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_fallback_render_all_views():
    import torch
    from fcd_prototype.gaussian_field import _fallback_render

    N = 5
    means = torch.randn(N, 3) * 0.3
    log_scales = torch.full((N, 3), -2.0)
    opacity_logit = torch.zeros(N)
    colors_raw = torch.zeros(N, 3)

    for view in ("axial", "coronal", "sagittal"):
        out = _fallback_render(means, log_scales, opacity_logit, colors_raw, 8, view)
        assert out.shape == (8, 8, 3), f"{view} wrong shape"


# ---------------------------------------------------------------------------
# Dependency / error path tests
# ---------------------------------------------------------------------------

def test_check_torch_passes_for_cpu():
    from fcd_prototype.gaussian_field import _check_torch
    _check_torch("cpu")  # should not raise


def test_check_torch_raises_for_missing_torch(monkeypatch):
    import sys
    import importlib
    # Simulate missing torch by blocking the import
    original = sys.modules.get("torch")
    sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        import importlib
        import fcd_prototype.gaussian_field as gf
        importlib.reload(gf)
        with pytest.raises((ImportError, TypeError)):
            gf._check_torch("cpu")
    finally:
        if original is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original


def test_gsplat_available_is_bool():
    from fcd_prototype.gaussian_field import _gsplat_available
    result = _gsplat_available()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Scale clamp pixel-unit tests
# ---------------------------------------------------------------------------

def test_fallback_render_tiny_scales_no_huge_splat():
    """Very small normalized scales must not produce a near-uniform image.

    Bug: clamping normalized scales to 0.5 before pixel conversion makes every
    splat cover W/2 = huge.  After the fix (clamp in px space) tiny scales stay
    tiny.
    """
    import torch
    from fcd_prototype.gaussian_field import _fallback_render

    N = 5
    H = W = 32
    means = torch.zeros(N, 3)
    # Extremely small normalized scale: exp(-10) ≈ 4.5e-5
    log_scales = torch.full((N, 3), -10.0)
    opacity_logit = torch.ones(N) * 2.0  # high opacity so splats are visible
    # White Gaussians on black background
    colors_raw = torch.ones(N, 3) * 3.0

    out = _fallback_render(means, log_scales, opacity_logit, colors_raw, W, "axial")
    assert out.shape == (H, W, 3)

    # With tiny scales, only the center pixel(s) should be lit; the image
    # should NOT be nearly uniform white (which would indicate huge splats).
    center = out[H // 2, W // 2].mean().item()
    corner = out[0, 0].mean().item()
    # Center should be brighter than corner
    assert center > corner, (
        f"Center brightness {center:.4f} should exceed corner {corner:.4f}; "
        "tiny scales may be getting clamped in normalized space (pre-fix bug)"
    )


def test_fallback_render_large_scales_produce_bright_image():
    """Large scales should fill the image (regression guard for the pixel clamp)."""
    import torch
    from fcd_prototype.gaussian_field import _fallback_render

    N = 3
    means = torch.zeros(N, 3)
    log_scales = torch.full((N, 3), 2.0)   # exp(2) ≈ 7.4 normalized → >W/2 px
    opacity_logit = torch.ones(N) * 3.0
    colors_raw = torch.ones(N, 3) * 3.0

    out = _fallback_render(means, log_scales, opacity_logit, colors_raw, 16, "axial")
    # Large Gaussians should produce non-trivial output
    assert float(out.mean()) > 0.1


# ---------------------------------------------------------------------------
# gsplat viewmat construction (no CUDA required)
# ---------------------------------------------------------------------------

def test_gsplat_viewmat_translation_is_camera_z():
    """All three views must put translation on the camera-z component (index 3, row 2)."""
    import torch
    from fcd_prototype.gaussian_field import _gsplat_viewmat

    for view in ("axial", "coronal", "sagittal"):
        vm = _gsplat_viewmat(view, torch.device("cpu"))  # (1, 4, 4)
        assert vm.shape == (1, 4, 4), f"{view}: wrong shape {vm.shape}"
        t = vm[0, :3, 3]
        # x and y translation must be zero; z must be positive (scene in front of camera)
        assert abs(float(t[0])) < 1e-6, f"{view}: t_x={float(t[0])} should be 0"
        assert abs(float(t[1])) < 1e-6, f"{view}: t_y={float(t[1])} should be 0"
        assert float(t[2]) > 0, f"{view}: t_z={float(t[2])} should be positive"


def test_gsplat_viewmat_rotation_is_orthonormal():
    """R must be orthonormal for all views (no shear / scale artifacts)."""
    import torch
    from fcd_prototype.gaussian_field import _gsplat_viewmat

    for view in ("axial", "coronal", "sagittal"):
        vm = _gsplat_viewmat(view, torch.device("cpu"))
        R = vm[0, :3, :3]
        eye = torch.mm(R, R.t())
        np.testing.assert_allclose(
            eye.numpy(), np.eye(3, dtype=np.float32), atol=1e-5,
            err_msg=f"{view}: R is not orthonormal",
        )


def test_gsplat_viewmat_world_axes_mapped_to_camera_z():
    """Each view must map the correct world axis to camera-z (the depth axis).

    axial:    world-z  → cam-z  (R row 2 = [0,0,1])
    coronal:  world-y  → cam-z  (R row 2 = [0,1,0])
    sagittal: world-x  → cam-z  (R row 2 = [1,0,0])
    """
    import torch
    from fcd_prototype.gaussian_field import _gsplat_viewmat

    expected_cam_z = {
        "axial":    [0, 0, 1],
        "coronal":  [0, 1, 0],
        "sagittal": [1, 0, 0],
    }
    for view, expected in expected_cam_z.items():
        vm = _gsplat_viewmat(view, torch.device("cpu"))
        cam_z_row = vm[0, 2, :3].numpy()
        np.testing.assert_allclose(
            cam_z_row, np.array(expected, dtype=np.float32), atol=1e-5,
            err_msg=f"{view}: camera-z row {cam_z_row} != expected {expected}",
        )


# ---------------------------------------------------------------------------
# View validation tests
# ---------------------------------------------------------------------------

def test_unknown_view_raises_before_training(tmp_path):
    from fcd_prototype.gaussian_field import train_gaussian_field

    vol = tmp_path / "vol.nii.gz"
    _make_volume(vol)
    with pytest.raises(ValueError, match="Unknown view"):
        train_gaussian_field(
            volume_path=vol,
            output_dir=tmp_path / "out",
            num_gaussians=10,
            steps=1,
            image_size=8,
            views="front",   # invalid
            device="cpu",
            seed=0,
        )


def test_mixed_valid_invalid_views_raises(tmp_path):
    from fcd_prototype.gaussian_field import train_gaussian_field

    vol = tmp_path / "vol.nii.gz"
    _make_volume(vol)
    with pytest.raises(ValueError, match="Unknown view"):
        train_gaussian_field(
            volume_path=vol,
            output_dir=tmp_path / "out",
            num_gaussians=10,
            steps=1,
            image_size=8,
            views="axial,top",
            device="cpu",
            seed=0,
        )
