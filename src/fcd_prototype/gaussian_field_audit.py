"""Gaussian field fidelity audit: compare reconstructed vs reference volumes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _check_torch(device: str) -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for gaussian_field audit. "
            "Install with: pip install torch"
        ) from exc
    if device == "cuda":
        import torch as _torch
        if not _torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is False. "
                "Check your CUDA installation or use --device cpu."
            )


def audit_gaussian_field_fidelity(
    gaussian_field_path: Path,
    reference_volume_path: Path,
    output_json: Path,
    sample_count: int = 8192,
    seed: int = 1049,
    device: str = "cpu",
) -> dict:
    """
    Audit Gaussian field reconstruction fidelity against a reference volume.

    Loads checkpoint, samples points from reference volume, renders predictions,
    computes MAE/RMSE/PSNR/correlation across all/foreground/background regions,
    and applies readiness gate.

    Returns audit dict with schema gaussian_field_fidelity_audit_v1.
    """
    import torch
    from fcd_prototype.gaussian_field import (
        GAUSSIAN_SUPPORT_FINGERPRINT_SCHEMA,
        _load_canonical,
        _as_normalized_volume,
        _build_volume_samples,
        _render_volume_samples,
        gaussian_support_fingerprint,
    )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _check_torch(device)
    dev = torch.device(device)

    gaussian_field_path = Path(gaussian_field_path)
    reference_volume_path = Path(reference_volume_path)
    output_json = Path(output_json)

    # Load checkpoint
    checkpoint = torch.load(str(gaussian_field_path), weights_only=False)
    means_norm = checkpoint["means_norm"]
    log_scales = checkpoint["log_scales"]
    quat_wxyz = checkpoint.get("quat_wxyz")
    opacity_logit = checkpoint["opacity_logit"]
    colors_raw = checkpoint["colors_raw"]
    norm_center = np.asarray(checkpoint["norm_center"], dtype=np.float32)
    norm_scale = float(checkpoint["norm_scale"])
    checkpoint_affine = checkpoint.get("affine")
    affine = (
        np.asarray(checkpoint_affine, dtype=np.float64)
        if checkpoint_affine is not None else None
    )

    training_objective = checkpoint.get("training_objective", "unknown")
    volume_loss_weight = float(checkpoint.get("volume_loss_weight", 0.0))
    volume_sample_count = int(checkpoint.get("volume_sample_count", 0))
    volume_path = checkpoint.get("volume_path")
    stored_support_fingerprint = (
        checkpoint.get("support_fingerprint_sha256")
        or checkpoint.get("support_fingerprint")
    )
    support_fingerprint_schema = checkpoint.get("support_fingerprint_schema")
    if quat_wxyz is None:
        if hasattr(means_norm, "shape"):
            n_gaussians = int(means_norm.shape[0])
        else:
            n_gaussians = int(np.asarray(means_norm).shape[0])
        quat_np = np.zeros((n_gaussians, 4), dtype=np.float32)
        quat_np[:, 0] = 1.0
    elif hasattr(quat_wxyz, "detach"):
        quat_np = quat_wxyz.detach().cpu().numpy().astype(np.float32)
    else:
        quat_np = np.asarray(quat_wxyz, dtype=np.float32)
    support_fingerprint_live = gaussian_support_fingerprint(
        means_norm.detach().cpu().numpy() if hasattr(means_norm, "detach") else means_norm,
        log_scales.detach().cpu().numpy() if hasattr(log_scales, "detach") else log_scales,
        quat_np,
        norm_center,
        norm_scale,
    )
    support_fingerprint_stale = (
        stored_support_fingerprint is not None
        and str(stored_support_fingerprint) != support_fingerprint_live
    )

    # Detect affine mismatch
    ref_data, ref_affine = _load_canonical(reference_volume_path)
    ref_norm = _as_normalized_volume(ref_data)
    affine_missing = affine is None
    affine_mismatch = (
        True if affine_missing else not np.allclose(affine, ref_affine, atol=1e-3)
    )

    # Build deterministic samples from reference using audit seed
    volume_samples = _build_volume_samples(
        norm=ref_norm,
        affine=ref_affine,
        norm_center=norm_center,
        norm_scale=norm_scale,
        sample_count=sample_count,
        seed=seed,
        foreground_fraction=0.75,
    )

    # Move tensors to device
    def _to_tensor(v, device):
        if hasattr(v, 'to'):
            return v.to(device=device, dtype=torch.float32)
        return torch.tensor(v, dtype=torch.float32, device=device)

    means_norm_t = _to_tensor(means_norm, dev)
    log_scales_t = _to_tensor(log_scales, dev)
    opacity_logit_t = _to_tensor(opacity_logit, dev)
    colors_raw_t = _to_tensor(colors_raw, dev)
    sample_points_t = _to_tensor(volume_samples["sample_points_norm"], dev)

    # Render predictions
    with torch.no_grad():
        predictions = _render_volume_samples(
            means_norm_t,
            log_scales_t,
            opacity_logit_t,
            colors_raw_t,
            sample_points_t,
            gaussian_chunk_size=512,
        )

    predictions_np = predictions.detach().cpu().numpy().astype(np.float32)
    targets_np = volume_samples["targets"].astype(np.float32)
    is_foreground = volume_samples["is_foreground"].astype(bool)

    # Compute metrics
    def compute_metrics(pred: np.ndarray, tgt: np.ndarray) -> dict:
        """Compute MAE, RMSE, PSNR, correlation for predictions vs targets.

        Returns count plus metrics as null when count is 0; otherwise valid finite JSON.
        PSNR may be null/infinite (converted safely).
        """
        count = len(pred)
        if count == 0:
            return {"count": 0, "mae": None, "rmse": None, "psnr": None, "correlation": None}

        mae = float(np.mean(np.abs(pred - tgt)))
        rmse = float(np.sqrt(np.mean((pred - tgt) ** 2)))
        mse = rmse ** 2
        if mse > 0:
            psnr = float(10.0 * np.log10(1.0 / mse))
        else:
            psnr = float("inf")
        if not np.isfinite(psnr):
            psnr = None
        if len(pred) > 1 and pred.std() > 1e-8 and tgt.std() > 1e-8:
            corr = float(np.corrcoef(pred, tgt)[0, 1])
        else:
            corr = 0.0 if len(pred) > 1 else 1.0
        return {"count": count, "mae": mae, "rmse": rmse, "psnr": psnr, "correlation": corr}

    metrics_all = compute_metrics(predictions_np, targets_np)
    metrics_fg = compute_metrics(predictions_np[is_foreground], targets_np[is_foreground])
    metrics_bg = compute_metrics(predictions_np[~is_foreground], targets_np[~is_foreground])

    # Readiness gate thresholds
    fg_mae_threshold_pass = 0.25
    fg_mae_threshold_review = 0.35
    all_mae_threshold_pass = 0.20
    all_mae_threshold_review = 0.30

    foreground_count = int(is_foreground.sum())
    background_count = int((~is_foreground).sum())

    blockers = []
    warnings = []

    # Check volume-anchored objective
    has_volume_loss = (
        training_objective == "mip_projection_mse_plus_volume_sample_mse"
        or (volume_loss_weight > 0 and volume_sample_count > 0)
    )
    if not has_volume_loss:
        blockers.append("training_objective_not_volume_anchored")

    # Check optimized support fingerprint provenance
    if support_fingerprint_schema != GAUSSIAN_SUPPORT_FINGERPRINT_SCHEMA:
        blockers.append("support_fingerprint_schema_missing_or_invalid")
    if not stored_support_fingerprint:
        blockers.append("support_fingerprint_missing")
    elif support_fingerprint_stale:
        blockers.append("support_fingerprint_stale")

    # Check affine mismatch
    if affine_mismatch:
        blockers.append("affine_mismatch")
    if affine_missing:
        blockers.append("checkpoint_affine_missing")

    # Check foreground samples exist
    if foreground_count == 0:
        blockers.append("no_foreground_samples")

    # Check background samples (warning only)
    if background_count == 0:
        warnings.append("no_background_samples")

    # Determine gate status
    if blockers:
        gate_status = "fail"
        optimized_claim_allowed = False
    elif (
        metrics_fg.get("mae") is not None
        and metrics_all.get("mae") is not None
        and metrics_fg["mae"] <= fg_mae_threshold_pass
        and metrics_all["mae"] <= all_mae_threshold_pass
    ):
        gate_status = "pass"
        optimized_claim_allowed = True
    elif (
        metrics_fg.get("mae") is not None
        and metrics_all.get("mae") is not None
        and metrics_fg["mae"] <= fg_mae_threshold_review
        and metrics_all["mae"] <= all_mae_threshold_review
    ):
        gate_status = "review"
        optimized_claim_allowed = False
        warnings.append("marginal_fidelity_requires_review")
    else:
        gate_status = "fail"
        optimized_claim_allowed = False

    audit_output = {
        "schema": "gaussian_field_fidelity_audit_v1",
        "checkpoint_path": str(gaussian_field_path),
        "reference_volume_path": str(reference_volume_path),
        "reference_shape": tuple(ref_norm.shape),
        "sample_count": sample_count,
        "seed": seed,
        "device": str(dev),
        "metrics": {
            "all": metrics_all,
            "foreground": metrics_fg,
            "background": metrics_bg,
        },
        "sample_counts": {
            "total": int(len(targets_np)),
            "foreground": foreground_count,
            "background": background_count,
        },
        "checkpoint_metadata": {
            "training_objective": training_objective,
            "volume_loss_weight": float(volume_loss_weight),
            "volume_sample_count": int(volume_sample_count),
            "volume_path": str(volume_path) if volume_path else None,
            "support_fingerprint_schema": support_fingerprint_schema,
            "support_fingerprint_sha256": (
                str(stored_support_fingerprint)
                if stored_support_fingerprint else None
            ),
            "support_fingerprint_recomputed_sha256": support_fingerprint_live,
            "support_fingerprint_matches_geometry": (
                bool(
                    stored_support_fingerprint
                    and not support_fingerprint_stale
                    and support_fingerprint_schema == GAUSSIAN_SUPPORT_FINGERPRINT_SCHEMA
                )
            ),
        },
        "affine_mismatch": affine_mismatch,
        "affine_missing": affine_missing,
        "readiness_gate": {
            "status": gate_status,
            "optimized_mri_to_3dgs_claim_allowed": optimized_claim_allowed,
            "blockers": blockers,
            "warnings": warnings,
            "thresholds": {
                "foreground_mae_pass": fg_mae_threshold_pass,
                "foreground_mae_review": fg_mae_threshold_review,
                "all_mae_pass": all_mae_threshold_pass,
                "all_mae_review": all_mae_threshold_review,
            },
        },
    }

    # Write JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit_output, indent=2))

    return audit_output


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Audit Gaussian field fidelity: compare reconstructed volume "
            "against reference NIfTI."
        )
    )
    p.add_argument("--gaussian-field", type=Path, required=True,
                   help="Path to gaussian_field.pt checkpoint")
    p.add_argument("--reference-volume", type=Path, required=True,
                   help="Reference NIfTI volume (.nii.gz)")
    p.add_argument("--output-json", type=Path, required=True,
                   help="Output JSON audit report")
    p.add_argument("--sample-count", type=int, default=8192,
                   help="Number of sample points (default: 8192)")
    p.add_argument("--seed", type=int, default=1049,
                   help="RNG seed for deterministic sampling (default: 1049)")
    p.add_argument("--device", type=str, default="cpu",
                   choices=["auto", "cuda", "cpu"],
                   help="Device for rendering (default: cpu)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    audit_output = audit_gaussian_field_fidelity(
        gaussian_field_path=args.gaussian_field,
        reference_volume_path=args.reference_volume,
        output_json=args.output_json,
        sample_count=args.sample_count,
        seed=args.seed,
        device=device,
    )

    gate_status = audit_output["readiness_gate"]["status"]
    fg_mae = audit_output["metrics"]["foreground"]["mae"]
    all_mae = audit_output["metrics"]["all"]["mae"]
    print(f"Audit complete. Gate={gate_status}  FG MAE={fg_mae:.4f}  All MAE={all_mae:.4f}")


if __name__ == "__main__":
    main()
