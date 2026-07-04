"""Tests for gaussian_residual — Gaussian-space nearest-neighbour residual."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nifti(path: Path, shape=(12, 12, 10), seed=0) -> None:
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal(shape).astype(np.float32) * 0.2 + 0.6).clip(0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.5, 1.0])), str(path))


def _make_roi(path: Path, shape=(12, 12, 10)) -> None:
    data = np.zeros(shape, dtype=np.uint8)
    data[2:5, 2:5, 2:5] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.5, 1.0])), str(path))


def _make_field_pt(
    path: Path,
    n: int = 40,
    seed: int = 0,
    intensity_bias: float = 0.0,
    roi_intensity_bias: float = 0.0,
    cluster_shift: float = 0.0,
    force_roi_gaussians: int = 0,
    quat_wxyz: np.ndarray | None = None,
    support_fingerprint_sha256: str | None = None,
) -> None:
    """Save a minimal synthetic gaussian_field.pt for testing.

    intensity_bias     — add to all opacity_logits (shifts scalar intensity up/down).
    cluster_shift      — displace a subset of means by this amount in the x-axis (world mm).
    force_roi_gaussians — if > 0, pin that many means_norm to land inside the test ROI
                          (world voxels [2:5,2:5,2:5] with affine diag([1,1,1.5,1])).
                          ROI world bbox: x∈[2,4], y∈[2,4], z∈[3,6] mm;
                          with norm_scale=50, norm_center=0 → norm coords ≈ [0.04,0.08].
    """
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed")
    from fcd_prototype.gaussian_field import (
        GAUSSIAN_SUPPORT_FINGERPRINT_SCHEMA,
        gaussian_support_fingerprint,
    )

    rng = np.random.default_rng(seed)
    means_norm    = rng.uniform(-0.8, 0.8, (n, 3)).astype(np.float32)
    log_scales    = rng.uniform(-3.0, -1.0, (n, 3)).astype(np.float32)
    if quat_wxyz is None:
        quat_arr = np.zeros((n, 4), dtype=np.float32)
        quat_arr[:, 0] = 1.0
    else:
        quat_arr = np.asarray(quat_wxyz, dtype=np.float32)
        if quat_arr.shape == (4,):
            quat_arr = np.tile(quat_arr[None, :], (n, 1))
        assert quat_arr.shape == (n, 4)
    opacity_logit = rng.uniform(-1.0, 1.0, (n,)).astype(np.float32) + intensity_bias
    colors_raw    = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)

    norm_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    norm_scale  = np.float32(50.0)

    # Optionally shift the first 10 Gaussians in world space by perturbing means_norm
    if cluster_shift != 0.0:
        means_norm[:10, 0] += cluster_shift / float(norm_scale)

    # Pin a deterministic subset of means to lie inside the test ROI.
    # ROI voxel range [2:5,2:5,2:5], affine diag([1,1,1.5,1]) → world [2,4],[2,4],[3,6].
    # In normalised coords (norm_scale=50, norm_center=0): x∈[0.04,0.08],y∈[0.04,0.08],z∈[0.06,0.12].
    k = min(force_roi_gaussians, n)
    if k > 0:
        roi_norm_lo = np.array([0.04, 0.04, 0.06], dtype=np.float32)
        roi_norm_hi = np.array([0.08, 0.08, 0.12], dtype=np.float32)
        rng2 = np.random.default_rng(seed + 9999)
        means_norm[:k] = rng2.uniform(roi_norm_lo, roi_norm_hi, (k, 3)).astype(np.float32)
        opacity_logit[:k] += roi_intensity_bias

    state = {
        "means_norm":    torch.tensor(means_norm),
        "log_scales":    torch.tensor(log_scales),
        "quat_wxyz":     torch.tensor(quat_arr),
        "opacity_logit": torch.tensor(opacity_logit),
        "colors_raw":    torch.tensor(colors_raw),
        "norm_center":   norm_center,
        "norm_scale":    float(norm_scale),
        "affine":        np.diag([1.0, 1.0, 1.5, 1.0]).astype(np.float64),
        "num_gaussians": n,
        "backend": "fallback",
        "device": "cpu",
        "seed": seed,
        "jitter_voxels": False,
        "volume_path": "",
        "roi_path": None,
        "support_fingerprint_schema": GAUSSIAN_SUPPORT_FINGERPRINT_SCHEMA,
        "support_fingerprint_sha256": (
            support_fingerprint_sha256
            or gaussian_support_fingerprint(
                means_norm,
                log_scales,
                quat_arr,
                norm_center,
                float(norm_scale),
            )
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _read_binary_ply(path: Path) -> np.ndarray:
    from fcd_prototype.gaussian_residual import _PLY_DTYPE

    with open(path, "rb") as fh:
        while True:
            line = fh.readline()
            if line.strip() == b"end_header":
                break
        payload = fh.read()
    return np.frombuffer(payload, dtype=_PLY_DTYPE)


pytestmark = pytest.mark.skipif(
    not _has_torch(),
    reason="PyTorch not installed; skipping gaussian_residual tests",
)


# ---------------------------------------------------------------------------
# Unit: world coordinate denormalisation
# ---------------------------------------------------------------------------

def test_denormalize_roundtrip():
    from fcd_prototype.gaussian_residual import _denormalize
    center     = np.array([10.0, 20.0, -5.0], dtype=np.float32)
    scale      = 50.0
    means_norm = np.array([[0.0, 0.5, -1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    world      = _denormalize(means_norm, center, scale)
    expected   = means_norm * scale + center
    np.testing.assert_allclose(world, expected, atol=1e-5)


def test_denormalize_zeros_give_center():
    from fcd_prototype.gaussian_residual import _denormalize
    center = np.array([3.0, 7.0, -2.0], dtype=np.float32)
    world  = _denormalize(np.zeros((5, 3), dtype=np.float32), center, 30.0)
    np.testing.assert_allclose(world, np.tile(center, (5, 1)), atol=1e-5)


# ---------------------------------------------------------------------------
# Unit: scalar intensity
# ---------------------------------------------------------------------------

def test_scalar_intensity_range():
    from fcd_prototype.gaussian_residual import _scalar_intensity
    rng         = np.random.default_rng(7)
    logit       = rng.uniform(-3.0, 3.0, 100).astype(np.float32)
    colors_raw  = rng.uniform(-2.0, 2.0, (100, 3)).astype(np.float32)
    intensity   = _scalar_intensity(logit, colors_raw)
    assert intensity.shape == (100,)
    assert float(intensity.min()) >= 0.0
    assert float(intensity.max()) <= 1.0


# ---------------------------------------------------------------------------
# Unit: residual is near-zero for identical fields
# ---------------------------------------------------------------------------

def test_residual_near_zero_identical(tmp_path):
    from fcd_prototype.gaussian_residual import compute_residual, _denormalize
    from fcd_prototype.gaussian_residual import _scalar_intensity, _mean_scale_world
    from fcd_prototype.gaussian_residual import _load_state

    pt = tmp_path / "field.pt"
    _make_field_pt(pt, n=40, seed=1)

    s = _load_state(pt)
    world = _denormalize(s["means_norm"], s["norm_center"], s["norm_scale"])
    intensity = _scalar_intensity(s["opacity_logit"], s["colors_raw"])
    scale     = _mean_scale_world(s["log_scales"], s["norm_scale"])

    residual = compute_residual(world, intensity, scale, world, intensity, scale, k_neighbors=4)
    assert residual.shape == (40,)
    # Identical fields — every Gaussian finds itself at distance 0, residual should be ~0
    assert float(residual.max()) < 1e-5


# ---------------------------------------------------------------------------
# Unit: residual detects a deliberately changed cluster
# ---------------------------------------------------------------------------

def test_residual_detects_changed_cluster(tmp_path):
    from fcd_prototype.gaussian_residual import _load_state, _denormalize
    from fcd_prototype.gaussian_residual import _scalar_intensity, _mean_scale_world
    from fcd_prototype.gaussian_residual import compute_residual

    normal_pt  = tmp_path / "normal.pt"
    patient_pt = tmp_path / "patient.pt"
    _make_field_pt(normal_pt,  n=40, seed=2)
    # Patient has first 10 Gaussians shifted in intensity and world-space
    _make_field_pt(patient_pt, n=40, seed=2, intensity_bias=5.0, cluster_shift=30.0)

    p = _load_state(patient_pt)
    n = _load_state(normal_pt)
    p_world = _denormalize(p["means_norm"], p["norm_center"], p["norm_scale"])
    n_world = _denormalize(n["means_norm"], n["norm_center"], n["norm_scale"])
    p_int   = _scalar_intensity(p["opacity_logit"], p["colors_raw"])
    n_int   = _scalar_intensity(n["opacity_logit"], n["colors_raw"])
    p_sc    = _mean_scale_world(p["log_scales"], p["norm_scale"])
    n_sc    = _mean_scale_world(n["log_scales"], n["norm_scale"])

    residual = compute_residual(p_world, p_int, p_sc, n_world, n_int, n_sc, k_neighbors=4)
    # Mean residual of changed cluster (first 10) should exceed that of the rest
    assert float(residual[:10].mean()) > float(residual[10:].mean())


# ---------------------------------------------------------------------------
# Full pipeline smoke: outputs exist
# ---------------------------------------------------------------------------

def test_outputs_created(tmp_path):
    from fcd_prototype.gaussian_residual import compare_gaussian_fields

    p_pt = tmp_path / "patient" / "gaussian_field.pt"
    n_pt = tmp_path / "normal"  / "gaussian_field.pt"
    vol  = tmp_path / "ref.nii.gz"
    _make_field_pt(p_pt, n=30, seed=0)
    _make_field_pt(n_pt, n=30, seed=1)
    _make_nifti(vol)

    out = tmp_path / "out"
    compare_gaussian_fields(p_pt, n_pt, vol, out, k_neighbors=2)

    assert (out / "gaussian_residual_scores.npy").exists()
    assert (out / "gaussian_residual_splat.ply").exists()
    assert (out / "gaussian_residual_heatmap.nii.gz").exists()
    assert (out / "gaussian_residual_metrics.json").exists()


# ---------------------------------------------------------------------------
# PLY header validity
# ---------------------------------------------------------------------------

def test_ply_header(tmp_path):
    from fcd_prototype.gaussian_residual import compare_gaussian_fields

    p_pt = tmp_path / "patient" / "gaussian_field.pt"
    n_pt = tmp_path / "normal"  / "gaussian_field.pt"
    vol  = tmp_path / "ref.nii.gz"
    _make_field_pt(p_pt, n=25, seed=3)
    _make_field_pt(n_pt, n=25, seed=4)
    _make_nifti(vol)

    out = tmp_path / "out"
    compare_gaussian_fields(p_pt, n_pt, vol, out)

    ply = out / "gaussian_residual_splat.ply"
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
        assert prop in hstr, f"PLY header missing property: {prop}"


# ---------------------------------------------------------------------------
# Metrics JSON structure
# ---------------------------------------------------------------------------

def test_metrics_json_structure(tmp_path):
    from fcd_prototype.gaussian_residual import compare_gaussian_fields

    p_pt = tmp_path / "patient" / "gaussian_field.pt"
    n_pt = tmp_path / "normal"  / "gaussian_field.pt"
    vol  = tmp_path / "ref.nii.gz"
    _make_field_pt(p_pt, n=20, seed=5)
    _make_field_pt(n_pt, n=20, seed=6)
    _make_nifti(vol)

    out = tmp_path / "out"
    metrics = compare_gaussian_fields(p_pt, n_pt, vol, out)

    for key in ("k_neighbors", "n_patient", "n_normal", "residual_min",
                "residual_max", "residual_mean", "residual_p95", "residual_p99",
                "heatmap_percentile"):
        assert key in metrics, f"Missing key in metrics: {key}"

    on_disk = json.loads((out / "gaussian_residual_metrics.json").read_text())
    assert on_disk["n_patient"] == 20


def test_same_support_uses_indexwise_residual(tmp_path):
    from fcd_prototype.gaussian_residual import compare_gaussian_fields

    p_pt = tmp_path / "patient" / "gaussian_field.pt"
    n_pt = tmp_path / "normal" / "gaussian_field.pt"
    vol = tmp_path / "ref.nii.gz"
    _make_field_pt(n_pt, n=24, seed=13)
    _make_field_pt(p_pt, n=24, seed=13, intensity_bias=4.0)
    _make_nifti(vol)

    metrics = compare_gaussian_fields(p_pt, n_pt, vol, tmp_path / "out")

    assert metrics["same_gaussian_support"] is True
    assert metrics["residual_score_method"] == "indexwise_same_support_intensity"
    assert metrics["gaussian_subtraction_claim_allowed"] is True
    assert metrics["means_norm_max_abs_diff"] == pytest.approx(0.0)
    assert metrics["support_fingerprint_schema"] == "gaussian_support_v1"
    assert metrics["support_fingerprint_match"] is True
    assert len(metrics["patient_support_fingerprint_sha256"]) == 64
    assert metrics["patient_support_fingerprint_sha256"] == metrics["normal_support_fingerprint_sha256"]
    assert metrics["patient_support_fingerprint_matches_geometry"] is True
    assert metrics["normal_support_fingerprint_matches_geometry"] is True
    assert metrics["patient_support_fingerprint_stale"] is False
    assert metrics["normal_support_fingerprint_stale"] is False
    assert metrics["residual_mean"] > 0.0


def test_independent_support_blocks_indexwise_claim(tmp_path):
    from fcd_prototype.gaussian_residual import compare_gaussian_fields

    p_pt = tmp_path / "patient" / "gaussian_field.pt"
    n_pt = tmp_path / "normal" / "gaussian_field.pt"
    vol = tmp_path / "ref.nii.gz"
    _make_field_pt(p_pt, n=24, seed=14)
    _make_field_pt(n_pt, n=24, seed=15)
    _make_nifti(vol)

    metrics = compare_gaussian_fields(p_pt, n_pt, vol, tmp_path / "out")

    assert metrics["same_gaussian_support"] is False
    assert metrics["residual_score_method"] == "nearest_neighbor_common_space"
    assert metrics["gaussian_subtraction_claim_allowed"] is False
    assert metrics["means_norm_max_abs_diff"] is not None
    assert metrics["means_norm_max_abs_diff"] > 0.0
    assert metrics["support_fingerprint_match"] is False
    assert metrics["patient_support_fingerprint_sha256"] != metrics["normal_support_fingerprint_sha256"]
    assert metrics["patient_support_fingerprint_matches_geometry"] is True
    assert metrics["normal_support_fingerprint_matches_geometry"] is True


def test_stale_matching_stored_fingerprints_do_not_fake_support_match(tmp_path):
    from fcd_prototype.gaussian_residual import compare_gaussian_fields

    p_pt = tmp_path / "patient" / "gaussian_field.pt"
    n_pt = tmp_path / "normal" / "gaussian_field.pt"
    vol = tmp_path / "ref.nii.gz"
    stale = "f" * 64
    _make_field_pt(p_pt, n=24, seed=14, support_fingerprint_sha256=stale)
    _make_field_pt(n_pt, n=24, seed=15, support_fingerprint_sha256=stale)
    _make_nifti(vol)

    metrics = compare_gaussian_fields(p_pt, n_pt, vol, tmp_path / "out")

    assert metrics["same_gaussian_support"] is False
    assert metrics["support_fingerprint_match"] is False
    assert metrics["patient_stored_support_fingerprint_sha256"] == stale
    assert metrics["normal_stored_support_fingerprint_sha256"] == stale
    assert metrics["patient_support_fingerprint_sha256"] != stale
    assert metrics["normal_support_fingerprint_sha256"] != stale
    assert metrics["patient_support_fingerprint_stale"] is True
    assert metrics["normal_support_fingerprint_stale"] is True
    assert metrics["gaussian_subtraction_claim_allowed"] is False


# ---------------------------------------------------------------------------
# ROI metrics: changed cluster in ROI increases ratio
# ---------------------------------------------------------------------------

def test_roi_metrics_higher_when_cluster_in_roi(tmp_path):
    from fcd_prototype.gaussian_residual import compare_gaussian_fields

    n_pt   = tmp_path / "normal"  / "gaussian_field.pt"
    p_no   = tmp_path / "patient_no"  / "gaussian_field.pt"
    p_yes  = tmp_path / "patient_yes" / "gaussian_field.pt"
    vol    = tmp_path / "ref.nii.gz"
    roi    = tmp_path / "roi.nii.gz"

    # Place 8 of 30 Gaussians deterministically inside the test ROI for both fields.
    # p_no is identical to normal → residual ≈ 0 everywhere.
    # p_yes has a large intensity_bias → residual is elevated, including in the ROI.
    _make_field_pt(n_pt,  n=30, seed=10, force_roi_gaussians=8)
    _make_field_pt(p_no,  n=30, seed=10, force_roi_gaussians=8)
    _make_field_pt(p_yes, n=30, seed=10, force_roi_gaussians=8, intensity_bias=6.0)

    _make_nifti(vol)
    _make_roi(roi)

    out_no  = tmp_path / "out_no"
    out_yes = tmp_path / "out_yes"
    m_no  = compare_gaussian_fields(n_pt, n_pt, vol, out_no,  roi=roi, k_neighbors=2)
    m_yes = compare_gaussian_fields(p_yes, n_pt, vol, out_yes, roi=roi, k_neighbors=2)

    # All 7 ROI keys must always be present when roi is supplied.
    for key in ("roi_gaussian_count", "non_roi_gaussian_count", "roi_metrics_valid",
                "roi_residual_mean", "non_roi_residual_mean",
                "roi_to_non_roi_ratio", "roi_residual_concentration"):
        assert key in m_no,  f"Missing key in m_no: {key}"
        assert key in m_yes, f"Missing key in m_yes: {key}"

    # Identical fields → residual ≈ 0 everywhere including ROI.
    assert m_no["roi_metrics_valid"] is True
    assert m_yes["roi_metrics_valid"] is True

    # Boosted patient field must have higher ROI residual than the identical one.
    assert m_yes["roi_residual_mean"] > m_no["roi_residual_mean"]
    assert m_yes["roi_gaussian_count"] == m_no["roi_gaussian_count"]


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def test_cli_smoke(tmp_path):
    from fcd_prototype.gaussian_residual import main

    p_pt = tmp_path / "patient" / "gaussian_field.pt"
    n_pt = tmp_path / "normal"  / "gaussian_field.pt"
    vol  = tmp_path / "ref.nii.gz"
    _make_field_pt(p_pt, n=20, seed=7)
    _make_field_pt(n_pt, n=20, seed=8)
    _make_nifti(vol)

    out = tmp_path / "cli_out"
    main([
        "--patient-field",    str(p_pt),
        "--normal-field",     str(n_pt),
        "--reference-volume", str(vol),
        "--output-dir",       str(out),
        "--k-neighbors",      "2",
    ])

    assert (out / "gaussian_residual_metrics.json").exists()
    assert (out / "gaussian_residual_splat.ply").exists()


# ---------------------------------------------------------------------------
# Heatmap NIfTI shape matches reference
# ---------------------------------------------------------------------------

def test_heatmap_shape_matches_reference(tmp_path):
    from fcd_prototype.gaussian_residual import compare_gaussian_fields

    p_pt = tmp_path / "patient" / "gaussian_field.pt"
    n_pt = tmp_path / "normal"  / "gaussian_field.pt"
    vol  = tmp_path / "ref.nii.gz"
    _make_field_pt(p_pt, n=20, seed=9)
    _make_field_pt(n_pt, n=20, seed=11)
    ref_shape = (12, 12, 10)
    _make_nifti(vol, shape=ref_shape)

    out = tmp_path / "out"
    compare_gaussian_fields(p_pt, n_pt, vol, out)

    hm = nib.load(str(out / "gaussian_residual_heatmap.nii.gz"))
    assert hm.shape[:3] == ref_shape


# ---------------------------------------------------------------------------
# Field-mode: identical fields produce near-zero heatmap
# ---------------------------------------------------------------------------

def test_field_mode_identical_near_zero(tmp_path):
    from fcd_prototype.gaussian_residual import compare_gaussian_fields

    p_pt = tmp_path / "patient" / "gaussian_field.pt"
    n_pt = tmp_path / "normal"  / "gaussian_field.pt"
    vol  = tmp_path / "ref.nii.gz"
    # Use identical fields
    _make_field_pt(p_pt, n=30, seed=42)
    _make_field_pt(n_pt, n=30, seed=42)
    _make_nifti(vol)

    out = tmp_path / "out"
    metrics = compare_gaussian_fields(
        p_pt, n_pt, vol, out,
        heatmap_method="field",
        field_chunk_size=16,
    )

    assert metrics["heatmap_method"] == "field"
    assert "field_residual_mean" in metrics
    # Identical fields should have very small residual
    assert metrics["field_residual_mean"] < 0.01

    hm = nib.load(str(out / "gaussian_residual_heatmap.nii.gz"))
    assert hm.shape[:3] == (12, 12, 10)
    assert float(hm.get_fdata().max()) < 0.01


# ---------------------------------------------------------------------------
# Field-mode: changed cluster produces elevated residual in that region
# ---------------------------------------------------------------------------

def test_field_mode_detects_roi_cluster_above_background(tmp_path):
    from fcd_prototype.gaussian_residual import compare_gaussian_fields

    n_pt = tmp_path / "normal" / "gaussian_field.pt"
    p_pt = tmp_path / "patient" / "gaussian_field.pt"
    vol  = tmp_path / "ref.nii.gz"
    roi  = tmp_path / "roi.nii.gz"

    _make_field_pt(n_pt, n=40, seed=50, force_roi_gaussians=8)
    _make_field_pt(
        p_pt,
        n=40,
        seed=50,
        force_roi_gaussians=8,
        roi_intensity_bias=6.0,
    )
    _make_nifti(vol)
    _make_roi(roi)

    out = tmp_path / "out"
    metrics = compare_gaussian_fields(
        p_pt, n_pt, vol, out, roi=roi,
        heatmap_method="field",
        field_chunk_size=32,
    )

    assert metrics["heatmap_method"] == "field"
    # Boosted intensity should cause non-zero residual
    assert metrics["field_residual_mean"] > 0.0
    assert metrics["field_residual_max"] > 0.001

    hm = nib.load(str(out / "gaussian_residual_heatmap.nii.gz"))
    hm_data = hm.get_fdata(dtype=np.float32)
    roi_mask = nib.load(str(roi)).get_fdata(dtype=np.float32) > 0
    assert hm_data[roi_mask].mean() > hm_data[~roi_mask].mean()
    assert hm_data[roi_mask].max() > 0.01


def test_field_mode_respects_quaternion_rotation():
    from fcd_prototype.gaussian_residual import _evaluate_gaussian_field_on_grid

    means = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
    # A clearly anisotropic Gaussian: long axis is local x.
    log_scales = np.log(np.array([[3.0, 1.0, 1.0]], dtype=np.float32))
    intensity = np.array([1.0], dtype=np.float32)
    ref_affine = np.eye(4, dtype=np.float64)
    ref_shape = (21, 21, 21)

    identity = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    theta = np.pi / 2.0
    rot_z_90 = np.array([[np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)]], dtype=np.float32)

    field_identity, _ = _evaluate_gaussian_field_on_grid(
        means, log_scales, identity, intensity, 1.0, ref_affine, ref_shape, chunk_size=1
    )
    field_rotated, _ = _evaluate_gaussian_field_on_grid(
        means, log_scales, rot_z_90, intensity, 1.0, ref_affine, ref_shape, chunk_size=1
    )

    # Identity: the long response axis points along world x.
    assert field_identity[13, 10, 10] > field_identity[10, 13, 10]
    # 90-degree z rotation: the same long axis now points along world y.
    assert field_rotated[10, 13, 10] > field_rotated[13, 10, 10]


def test_residual_ply_preserves_patient_quaternions(tmp_path):
    from fcd_prototype.gaussian_residual import compare_gaussian_fields

    theta = np.pi / 3.0
    quat = np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)], dtype=np.float32)
    p_pt = tmp_path / "patient" / "gaussian_field.pt"
    n_pt = tmp_path / "normal" / "gaussian_field.pt"
    vol = tmp_path / "ref.nii.gz"
    _make_field_pt(p_pt, n=12, seed=90, quat_wxyz=quat)
    _make_field_pt(n_pt, n=12, seed=91)
    _make_nifti(vol)

    out = tmp_path / "out"
    compare_gaussian_fields(p_pt, n_pt, vol, out, max_points=12)

    arr = _read_binary_ply(out / "gaussian_residual_splat.ply")
    np.testing.assert_allclose(arr["rot_w"], quat[0], atol=1e-5)
    np.testing.assert_allclose(arr["rot_x"], quat[1], atol=1e-5)
    np.testing.assert_allclose(arr["rot_y"], quat[2], atol=1e-5)
    np.testing.assert_allclose(arr["rot_z"], quat[3], atol=1e-5)


# ---------------------------------------------------------------------------
# Both scatter and field methods produce outputs
# ---------------------------------------------------------------------------

def test_scatter_mode_still_available(tmp_path):
    from fcd_prototype.gaussian_residual import compare_gaussian_fields

    p_pt = tmp_path / "patient" / "gaussian_field.pt"
    n_pt = tmp_path / "normal"  / "gaussian_field.pt"
    vol  = tmp_path / "ref.nii.gz"
    _make_field_pt(p_pt, n=25, seed=60)
    _make_field_pt(n_pt, n=25, seed=61)
    _make_nifti(vol)

    out = tmp_path / "out"
    metrics = compare_gaussian_fields(
        p_pt, n_pt, vol, out,
        heatmap_method="scatter",
    )

    assert metrics["heatmap_method"] == "scatter"
    assert "residual_p99" in metrics
    # scatter should NOT have field_* keys
    assert "field_residual_mean" not in metrics


# ---------------------------------------------------------------------------
# Field metrics include heatmap_method and field_* keys
# ---------------------------------------------------------------------------

def test_field_metrics_complete(tmp_path):
    from fcd_prototype.gaussian_residual import compare_gaussian_fields

    p_pt = tmp_path / "patient" / "gaussian_field.pt"
    n_pt = tmp_path / "normal"  / "gaussian_field.pt"
    vol  = tmp_path / "ref.nii.gz"
    _make_field_pt(p_pt, n=20, seed=70)
    _make_field_pt(n_pt, n=20, seed=71)
    _make_nifti(vol)

    out = tmp_path / "out"
    metrics = compare_gaussian_fields(
        p_pt, n_pt, vol, out,
        heatmap_method="field",
        field_chunk_size=64,
    )

    required_field_keys = {
        "heatmap_method",
        "field_patient_sum",
        "field_normal_sum",
        "field_residual_mean",
        "field_residual_min",
        "field_residual_max",
        "field_residual_p95",
        "field_residual_p99",
        "field_chunk_size",
        "field_covariance_mode",
    }
    assert required_field_keys.issubset(set(metrics.keys()))
    assert metrics["heatmap_method"] == "field"
    assert metrics["field_chunk_size"] == 64
    assert metrics["field_covariance_mode"] == "rotated_anisotropic"

    on_disk = json.loads((out / "gaussian_residual_metrics.json").read_text())
    assert required_field_keys.issubset(set(on_disk.keys()))


# ---------------------------------------------------------------------------
# CLI passes heatmap_method and field_chunk_size
# ---------------------------------------------------------------------------

def test_cli_field_mode(tmp_path):
    from fcd_prototype.gaussian_residual import main

    p_pt = tmp_path / "patient" / "gaussian_field.pt"
    n_pt = tmp_path / "normal"  / "gaussian_field.pt"
    vol  = tmp_path / "ref.nii.gz"
    _make_field_pt(p_pt, n=20, seed=80)
    _make_field_pt(n_pt, n=20, seed=81)
    _make_nifti(vol)

    out = tmp_path / "cli_field_out"
    main([
        "--patient-field",    str(p_pt),
        "--normal-field",     str(n_pt),
        "--reference-volume", str(vol),
        "--output-dir",       str(out),
        "--heatmap-method",   "field",
        "--field-chunk-size", "32",
    ])

    metrics = json.loads((out / "gaussian_residual_metrics.json").read_text())
    assert metrics["heatmap_method"] == "field"
    assert metrics["field_chunk_size"] == 32
    assert "field_residual_mean" in metrics
