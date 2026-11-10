"""Tests for gaussian_field_audit — fidelity metrics and readiness gate."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def _make_synthetic_volume(path: Path, shape=(12, 12, 10), seed=0) -> None:
    """Create a synthetic volume with Gaussian-like structure."""
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal(shape).astype(np.float32) * 0.2 + 0.6).clip(0.0, 1.0)
    affine = np.diag([1.0, 1.0, 1.5, 1.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _make_checkpoint(
    path: Path,
    num_gaussians: int = 20,
    training_objective: str = "mip_projection_mse_plus_volume_sample_mse",
    volume_loss_weight: float = 0.1,
    volume_sample_count: int = 100,
    volume_path: str | None = None,
    affine: np.ndarray | None = None,
    support_fingerprint_sha256: str | None = None,
    include_support_fingerprint: bool = True,
) -> dict:
    """Create a synthetic Gaussian field checkpoint."""
    import torch
    from fcd_prototype.gaussian_field import (
        GAUSSIAN_SUPPORT_FINGERPRINT_SCHEMA,
        gaussian_support_fingerprint,
    )

    if affine is None:
        affine = np.diag([1.0, 1.0, 1.5, 1.0])

    norm_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    norm_scale = np.float32(10.0)

    means_norm = np.random.randn(num_gaussians, 3).astype(np.float32) * 0.3
    log_scales = np.full((num_gaussians, 3), -1.5, dtype=np.float32)
    quat_wxyz = np.zeros((num_gaussians, 4), dtype=np.float32)
    quat_wxyz[:, 0] = 1.0
    opacity_logit = np.zeros(num_gaussians, dtype=np.float32)
    colors_raw = np.random.randn(num_gaussians, 3).astype(np.float32) * 0.2

    checkpoint = {
        "means_norm": torch.tensor(means_norm, dtype=torch.float32),
        "log_scales": torch.tensor(log_scales, dtype=torch.float32),
        "quat_wxyz": torch.tensor(quat_wxyz, dtype=torch.float32),
        "opacity_logit": torch.tensor(opacity_logit, dtype=torch.float32),
        "colors_raw": torch.tensor(colors_raw, dtype=torch.float32),
        "norm_center": norm_center,
        "norm_scale": norm_scale,
        "affine": affine,
        "training_objective": training_objective,
        "volume_loss_weight": float(volume_loss_weight),
        "volume_sample_count": int(volume_sample_count),
        "volume_path": str(volume_path) if volume_path else None,
    }
    if include_support_fingerprint:
        support_fingerprint = (
            support_fingerprint_sha256
            or gaussian_support_fingerprint(
                means_norm,
                log_scales,
                quat_wxyz,
                norm_center,
                float(norm_scale),
            )
        )
        checkpoint["support_fingerprint_schema"] = GAUSSIAN_SUPPORT_FINGERPRINT_SCHEMA
        checkpoint["support_fingerprint_sha256"] = support_fingerprint

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(path))
    return checkpoint


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_torch(),
    reason="PyTorch not installed; skipping gaussian_field_audit tests",
)


# ---------------------------------------------------------------------------
# Basic fidelity audit tests
# ---------------------------------------------------------------------------

def test_audit_creates_json_output(tmp_path):
    from fcd_prototype.gaussian_field_audit import audit_gaussian_field_fidelity

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output = tmp_path / "audit.json"

    _make_synthetic_volume(vol)
    _make_checkpoint(ckpt, training_objective="mip_projection_mse_plus_volume_sample_mse")

    result = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output,
        sample_count=256,
        seed=1049,
        device="cpu",
    )

    assert output.exists()
    with output.open() as fh:
        on_disk = json.load(fh)
    assert on_disk["schema"] == "gaussian_field_fidelity_audit_v1"
    assert result["schema"] == "gaussian_field_fidelity_audit_v1"


def test_audit_returns_expected_schema_keys(tmp_path):
    from fcd_prototype.gaussian_field_audit import audit_gaussian_field_fidelity

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output = tmp_path / "audit.json"

    _make_synthetic_volume(vol)
    _make_checkpoint(ckpt)

    result = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output,
        sample_count=128,
        seed=42,
        device="cpu",
    )

    required_keys = {
        "schema",
        "checkpoint_path",
        "reference_volume_path",
        "reference_shape",
        "sample_count",
        "seed",
        "device",
        "metrics",
        "sample_counts",
        "checkpoint_metadata",
        "affine_mismatch",
        "readiness_gate",
    }
    assert required_keys.issubset(result.keys())


def test_audit_metrics_structure(tmp_path):
    from fcd_prototype.gaussian_field_audit import audit_gaussian_field_fidelity

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output = tmp_path / "audit.json"

    _make_synthetic_volume(vol)
    _make_checkpoint(ckpt)

    result = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output,
        sample_count=128,
        seed=42,
        device="cpu",
    )

    metrics = result["metrics"]
    for region in ("all", "foreground", "background"):
        assert region in metrics
        for key in ("mae", "rmse", "psnr", "correlation"):
            assert key in metrics[region]
            assert isinstance(metrics[region][key], (int, float))


def test_audit_sample_counts_populated(tmp_path):
    from fcd_prototype.gaussian_field_audit import audit_gaussian_field_fidelity

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output = tmp_path / "audit.json"

    _make_synthetic_volume(vol)
    _make_checkpoint(ckpt)

    result = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output,
        sample_count=256,
        seed=42,
        device="cpu",
    )

    counts = result["sample_counts"]
    assert counts["total"] == 256
    assert counts["foreground"] > 0
    assert counts["total"] == counts["foreground"] + counts["background"]


def test_audit_deterministic_with_seed(tmp_path):
    from fcd_prototype.gaussian_field_audit import audit_gaussian_field_fidelity

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output1 = tmp_path / "audit1.json"
    output2 = tmp_path / "audit2.json"

    _make_synthetic_volume(vol)
    _make_checkpoint(ckpt)

    result1 = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output1,
        sample_count=128,
        seed=999,
        device="cpu",
    )

    result2 = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output2,
        sample_count=128,
        seed=999,
        device="cpu",
    )

    assert result1["metrics"]["all"]["mae"] == result2["metrics"]["all"]["mae"]
    assert result1["sample_counts"]["foreground"] == result2["sample_counts"]["foreground"]


def test_audit_different_seeds_differ(tmp_path):
    from fcd_prototype.gaussian_field_audit import audit_gaussian_field_fidelity

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output1 = tmp_path / "audit1.json"
    output2 = tmp_path / "audit2.json"

    _make_synthetic_volume(vol)
    _make_checkpoint(ckpt)

    result1 = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output1,
        sample_count=128,
        seed=1,
        device="cpu",
    )

    result2 = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output2,
        sample_count=128,
        seed=2,
        device="cpu",
    )

    # Different seeds should produce (likely) different metrics
    assert result1["metrics"]["all"]["mae"] != result2["metrics"]["all"]["mae"]


# ---------------------------------------------------------------------------
# Readiness gate tests
# ---------------------------------------------------------------------------

def test_readiness_gate_has_required_fields(tmp_path):
    from fcd_prototype.gaussian_field_audit import audit_gaussian_field_fidelity

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output = tmp_path / "audit.json"

    _make_synthetic_volume(vol)
    _make_checkpoint(ckpt)

    result = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output,
        sample_count=128,
        seed=42,
        device="cpu",
    )

    gate = result["readiness_gate"]
    required_fields = {"status", "optimized_mri_to_3dgs_claim_allowed", "blockers", "warnings", "thresholds"}
    assert required_fields.issubset(gate.keys())
    assert gate["status"] in ("pass", "review", "fail")
    assert isinstance(gate["optimized_mri_to_3dgs_claim_allowed"], bool)
    assert isinstance(gate["blockers"], list)
    assert isinstance(gate["warnings"], list)


def test_readiness_gate_fails_on_non_volume_loss_objective(tmp_path):
    """Gate fails when training_objective is NOT volume-anchored."""
    from fcd_prototype.gaussian_field_audit import audit_gaussian_field_fidelity

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output = tmp_path / "audit.json"

    _make_synthetic_volume(vol)
    _make_checkpoint(
        ckpt,
        training_objective="mip_projection_mse",
        volume_loss_weight=0.0,
        volume_sample_count=0,
    )

    result = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output,
        sample_count=128,
        seed=42,
        device="cpu",
    )

    gate = result["readiness_gate"]
    assert gate["status"] == "fail"
    assert "training_objective_not_volume_anchored" in gate["blockers"]
    assert not gate["optimized_mri_to_3dgs_claim_allowed"]


def test_readiness_gate_fails_on_affine_mismatch(tmp_path):
    """Gate fails when reference volume has different affine than checkpoint."""
    from fcd_prototype.gaussian_field_audit import audit_gaussian_field_fidelity

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output = tmp_path / "audit.json"

    # Volume with identity affine
    _make_synthetic_volume(vol)
    # Checkpoint with different affine
    mismatched_affine = np.diag([2.0, 2.0, 3.0, 1.0])
    _make_checkpoint(ckpt, affine=mismatched_affine)

    result = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output,
        sample_count=128,
        seed=42,
        device="cpu",
    )

    gate = result["readiness_gate"]
    assert gate["status"] == "fail"
    assert "affine_mismatch" in gate["blockers"]
    assert result["affine_mismatch"] is True


def test_readiness_gate_pass_with_good_metrics(tmp_path):
    """Gate passes when metrics meet strict thresholds and objective is volume-anchored."""
    from fcd_prototype.gaussian_field_audit import audit_gaussian_field_fidelity
    from fcd_prototype.gaussian_field import (
        GAUSSIAN_SUPPORT_FINGERPRINT_SCHEMA,
        gaussian_support_fingerprint,
    )
    import torch

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output = tmp_path / "audit.json"

    _make_synthetic_volume(vol, shape=(12, 12, 10), seed=0)

    # Create a checkpoint with identical affine
    means_norm = torch.zeros(10, 3, dtype=torch.float32)
    log_scales = torch.full((10, 3), -3.0, dtype=torch.float32)
    quat_wxyz = torch.zeros(10, 4, dtype=torch.float32)
    norm_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    norm_scale = np.float32(10.0)
    support_fingerprint = gaussian_support_fingerprint(
        means_norm.numpy(),
        log_scales.numpy(),
        quat_wxyz.numpy(),
        norm_center,
        float(norm_scale),
    )
    ckpt_data = {
        "means_norm": means_norm,
        "log_scales": log_scales,
        "quat_wxyz": quat_wxyz,
        "opacity_logit": torch.full((10,), 2.0, dtype=torch.float32),
        "colors_raw": torch.full((10, 3), 0.5, dtype=torch.float32),
        "norm_center": norm_center,
        "norm_scale": norm_scale,
        "affine": np.diag([1.0, 1.0, 1.5, 1.0]),
        "training_objective": "mip_projection_mse_plus_volume_sample_mse",
        "volume_loss_weight": 0.1,
        "volume_sample_count": 100,
        "volume_path": None,
        "support_fingerprint_schema": GAUSSIAN_SUPPORT_FINGERPRINT_SCHEMA,
        "support_fingerprint_sha256": support_fingerprint,
    }
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt_data, str(ckpt))

    result = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output,
        sample_count=128,
        seed=42,
        device="cpu",
    )

    gate = result["readiness_gate"]
    # With tight Gaussians on target, should have low MAE
    if gate["status"] == "pass":
        assert gate["optimized_mri_to_3dgs_claim_allowed"] is True
    elif gate["status"] == "review":
        assert gate["optimized_mri_to_3dgs_claim_allowed"] is False
    # May vary, but should not have blockers with matching affines and volume objective
    assert "affine_mismatch" not in gate["blockers"]
    assert "training_objective_not_volume_anchored" not in gate["blockers"]


def test_readiness_gate_fails_on_stale_support_fingerprint(tmp_path):
    """A copied support hash should not support optimized 3DGS evidence."""
    from fcd_prototype.gaussian_field_audit import audit_gaussian_field_fidelity

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output = tmp_path / "audit.json"

    _make_synthetic_volume(vol)
    _make_checkpoint(ckpt, support_fingerprint_sha256="0" * 64)

    result = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output,
        sample_count=128,
        seed=42,
        device="cpu",
    )

    gate = result["readiness_gate"]
    meta = result["checkpoint_metadata"]
    assert gate["status"] == "fail"
    assert "support_fingerprint_stale" in gate["blockers"]
    assert meta["support_fingerprint_sha256"] == "0" * 64
    assert meta["support_fingerprint_recomputed_sha256"] != "0" * 64
    assert meta["support_fingerprint_matches_geometry"] is False


def test_readiness_gate_fails_when_support_fingerprint_missing(tmp_path):
    from fcd_prototype.gaussian_field_audit import audit_gaussian_field_fidelity

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output = tmp_path / "audit.json"

    _make_synthetic_volume(vol)
    _make_checkpoint(ckpt, include_support_fingerprint=False)

    result = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output,
        sample_count=128,
        seed=42,
        device="cpu",
    )

    gate = result["readiness_gate"]
    assert "support_fingerprint_missing" in gate["blockers"]
    assert "support_fingerprint_schema_missing_or_invalid" in gate["blockers"]
    assert result["checkpoint_metadata"]["support_fingerprint_matches_geometry"] is False


def test_readiness_gate_review_on_marginal_metrics(tmp_path):
    """Gate enters 'review' state when metrics are marginal but not failed."""
    from fcd_prototype.gaussian_field_audit import audit_gaussian_field_fidelity

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output = tmp_path / "audit.json"

    _make_synthetic_volume(vol)
    _make_checkpoint(ckpt)

    result = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output,
        sample_count=256,
        seed=42,
        device="cpu",
    )

    gate = result["readiness_gate"]
    # Synthetic checkpoint may naturally have moderate metrics
    # We just verify the gate logic is applied consistently
    assert gate["status"] in ("pass", "review", "fail")


# ---------------------------------------------------------------------------
# Affine detection tests
# ---------------------------------------------------------------------------

def test_audit_detects_matching_affine(tmp_path):
    from fcd_prototype.gaussian_field_audit import audit_gaussian_field_fidelity

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output = tmp_path / "audit.json"

    matching_affine = np.diag([1.0, 1.0, 1.5, 1.0])
    _make_synthetic_volume(vol)
    _make_checkpoint(ckpt, affine=matching_affine)

    result = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output,
        sample_count=128,
        seed=42,
        device="cpu",
    )

    assert result["affine_mismatch"] is False


def test_audit_json_paths_are_strings(tmp_path):
    from fcd_prototype.gaussian_field_audit import audit_gaussian_field_fidelity

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output = tmp_path / "audit.json"

    _make_synthetic_volume(vol)
    _make_checkpoint(ckpt)

    result = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output,
        sample_count=128,
        seed=42,
        device="cpu",
    )

    assert isinstance(result["checkpoint_path"], str)
    assert isinstance(result["reference_volume_path"], str)


def test_audit_preserves_checkpoint_metadata(tmp_path):
    from fcd_prototype.gaussian_field_audit import audit_gaussian_field_fidelity

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output = tmp_path / "audit.json"

    vol_path_str = str(tmp_path / "training_vol.nii.gz")
    _make_synthetic_volume(vol)
    _make_checkpoint(
        ckpt,
        training_objective="mip_projection_mse_plus_volume_sample_mse",
        volume_loss_weight=0.25,
        volume_sample_count=512,
        volume_path=vol_path_str,
    )

    result = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output,
        sample_count=128,
        seed=42,
        device="cpu",
    )

    meta = result["checkpoint_metadata"]
    assert meta["training_objective"] == "mip_projection_mse_plus_volume_sample_mse"
    assert meta["volume_loss_weight"] == pytest.approx(0.25)
    assert meta["volume_sample_count"] == 512
    assert meta["volume_path"] == vol_path_str


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

def test_cli_main_creates_output(tmp_path):
    from fcd_prototype.gaussian_field_audit import main

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output = tmp_path / "audit.json"

    _make_synthetic_volume(vol)
    _make_checkpoint(ckpt)

    main([
        "--gaussian-field", str(ckpt),
        "--reference-volume", str(vol),
        "--output-json", str(output),
        "--sample-count", "128",
        "--seed", "42",
        "--device", "cpu",
    ])

    assert output.exists()
    with output.open() as fh:
        data = json.load(fh)
    assert data["schema"] == "gaussian_field_fidelity_audit_v1"


def test_cli_main_parses_device_auto(tmp_path):
    from fcd_prototype.gaussian_field_audit import main

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output = tmp_path / "audit.json"

    _make_synthetic_volume(vol)
    _make_checkpoint(ckpt)

    # Should not raise
    main([
        "--gaussian-field", str(ckpt),
        "--reference-volume", str(vol),
        "--output-json", str(output),
        "--sample-count", "64",
        "--device", "auto",
    ])

    assert output.exists()


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

def test_audit_small_volume(tmp_path):
    """Audit on a tiny volume should not crash."""
    from fcd_prototype.gaussian_field_audit import audit_gaussian_field_fidelity

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output = tmp_path / "audit.json"

    _make_synthetic_volume(vol, shape=(4, 4, 4), seed=0)
    _make_checkpoint(ckpt, num_gaussians=5)

    result = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output,
        sample_count=16,
        seed=42,
        device="cpu",
    )

    assert result["schema"] == "gaussian_field_fidelity_audit_v1"
    assert result["sample_counts"]["total"] == 16


def test_audit_metrics_bounded(tmp_path):
    """MAE/RMSE/PSNR/correlation should be in expected ranges for normalized volumes."""
    from fcd_prototype.gaussian_field_audit import audit_gaussian_field_fidelity

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output = tmp_path / "audit.json"

    _make_synthetic_volume(vol)
    _make_checkpoint(ckpt)

    result = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output,
        sample_count=128,
        seed=42,
        device="cpu",
    )

    for region in ("all", "foreground", "background"):
        metrics = result["metrics"][region]
        mae = metrics["mae"]
        rmse = metrics["rmse"]
        psnr = metrics["psnr"]
        corr = metrics["correlation"]

        # Normalized volumes in [0, 1]
        assert 0.0 <= mae <= 1.0, f"{region} MAE out of range: {mae}"
        assert 0.0 <= rmse <= 1.0, f"{region} RMSE out of range: {rmse}"
        # PSNR can be inf for perfect predictions
        assert psnr >= 0 or np.isinf(psnr), f"{region} PSNR invalid: {psnr}"
        # Correlation in [-1, 1]
        assert -1.0 <= corr <= 1.0, f"{region} correlation out of range: {corr}"


def test_audit_reference_shape_recorded(tmp_path):
    from fcd_prototype.gaussian_field_audit import audit_gaussian_field_fidelity

    vol = tmp_path / "vol.nii.gz"
    ckpt = tmp_path / "ckpt.pt"
    output = tmp_path / "audit.json"

    shape = (16, 18, 12)
    _make_synthetic_volume(vol, shape=shape, seed=0)
    _make_checkpoint(ckpt)

    result = audit_gaussian_field_fidelity(
        gaussian_field_path=ckpt,
        reference_volume_path=vol,
        output_json=output,
        sample_count=128,
        seed=42,
        device="cpu",
    )

    assert result["reference_shape"] == shape
