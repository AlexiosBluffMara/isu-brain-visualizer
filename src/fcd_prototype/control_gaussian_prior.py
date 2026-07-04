"""Patient-support Gaussian residual against a healthy-control prior.

This module is the paper-path fallback when a released generative model is not
patient-conditioned.  It samples Gaussians from the patient MRI, resamples
healthy controls into the same patient grid, estimates a per-Gaussian healthy
mean/std, and writes a z-score residual splat on the patient's own support.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

from fcd_prototype.paired_gaussian_residual import (
    _gaussian_scores_to_grid,
    _load_canonical,
    _maybe_resample_to_reference,
    _normalize_for_pairing,
    _roi_metrics,
    _sample_support,
    _scatter_scores,
    _write_ply,
)
from fcd_prototype.splat_ply_export import _voxel_to_world, oriented_gaussian_parameters


_CONTROL_GAUSSIAN_PRIOR_SCHEMA = "patient_control_gaussian_prior_v1"


def _load_control_list(path: Optional[Path]) -> list[Path]:
    if path is None:
        return []
    list_path = Path(path)
    base_dir = list_path.parent
    controls: list[Path] = []
    for raw in list_path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            control_path = Path(line)
            controls.append(
                control_path if control_path.is_absolute() else base_dir / control_path
            )
    return controls


def _reference_mask(
    patient_norm: np.ndarray,
    roi_mask: np.ndarray,
    *,
    roi_dilation_iterations: int,
) -> np.ndarray:
    mask = np.isfinite(patient_norm) & (patient_norm > 0.05)
    if roi_mask.any():
        excluded = binary_dilation(
            roi_mask.astype(bool),
            iterations=max(0, int(roi_dilation_iterations)),
        )
        candidate = mask & ~excluded
        if int(candidate.sum()) >= 32:
            mask = candidate
    if int(mask.sum()) < 32:
        mask = np.isfinite(patient_norm)
    return mask.astype(bool)


def _match_control_to_patient(
    control_norm: np.ndarray,
    patient_norm: np.ndarray,
    reference_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    mask = reference_mask & np.isfinite(control_norm) & np.isfinite(patient_norm)
    if int(mask.sum()) < 32:
        return control_norm.astype(np.float32), {
            "reference_voxels": int(mask.sum()),
            "matched": False,
            "reason": "insufficient_reference_voxels",
        }

    patient_values = patient_norm[mask].astype(np.float64)
    control_values = control_norm[mask].astype(np.float64)
    if np.allclose(control_values, patient_values, atol=1e-7, rtol=1e-7):
        return control_norm.astype(np.float32), {
            "reference_voxels": int(mask.sum()),
            "matched": True,
            "reason": "already_matched_to_patient",
        }
    patient_mean = float(patient_values.mean())
    patient_std = float(patient_values.std())
    control_mean = float(control_values.mean())
    control_std = float(control_values.std())
    if patient_std <= 1e-8 or control_std <= 1e-8:
        return control_norm.astype(np.float32), {
            "reference_voxels": int(mask.sum()),
            "matched": False,
            "reason": "near_zero_reference_std",
            "patient_reference_std": patient_std,
            "control_reference_std": control_std,
        }

    matched = (control_norm.astype(np.float32) - control_mean) / (
        control_std + 1e-8
    )
    matched = matched * patient_std + patient_mean
    return matched.clip(0.0, 1.0).astype(np.float32), {
        "reference_voxels": int(mask.sum()),
        "matched": True,
        "patient_reference_mean": patient_mean,
        "patient_reference_std": patient_std,
        "control_reference_mean": control_mean,
        "control_reference_std": control_std,
    }


def _load_roi_on_patient_grid(
    roi: Optional[Path],
    ref_shape: tuple[int, int, int],
    ref_affine: np.ndarray,
) -> tuple[np.ndarray, bool]:
    if roi is None or not Path(roi).is_file():
        return np.zeros(ref_shape, dtype=bool), False
    roi_data, roi_affine, _ = _load_canonical(Path(roi))
    roi_data, roi_resampled = _maybe_resample_to_reference(
        (roi_data > 0).astype(np.float32),
        roi_affine,
        ref_shape,
        ref_affine,
        nearest=True,
    )
    return roi_data > 0.5, bool(roi_resampled)


def build_control_gaussian_prior_residual(
    *,
    patient_volume: Path,
    control_volumes: list[Path],
    output_dir: Path,
    roi: Optional[Path] = None,
    min_controls: int = 2,
    max_points: int = 50_000,
    percentile_threshold: float = 5.0,
    surface_percentile_threshold: float = 70.0,
    sampling_mode: str = "volume",
    rng_seed: int = 42,
    heatmap_percentile: float = 99.0,
    smooth_sigma: float = 1.0,
    heatmap_method: str = "gaussian",
    heatmap_gaussian_sigma_extent: float = 2.5,
    heatmap_gaussian_exact_max_points: int = 50_000,
    std_floor_fraction: float = 0.05,
    std_floor_absolute: float = 1e-3,
    roi_dilation_iterations: int = 3,
    subject: str = "",
    group: str = "",
) -> dict[str, object]:
    """Write control-prior patient/mean/z-residual Gaussian splats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    controls = [Path(p) for p in control_volumes if Path(p).is_file()]
    if len(controls) < int(min_controls):
        raise ValueError(
            "control Gaussian prior requires at least "
            f"{min_controls} existing controls, got {len(controls)}"
        )

    patient_data, patient_affine, patient_header = _load_canonical(Path(patient_volume))
    ref_shape = tuple(int(x) for x in patient_data.shape[:3])
    patient_norm = _normalize_for_pairing(patient_data)
    roi_mask, roi_resampled = _load_roi_on_patient_grid(roi, ref_shape, patient_affine)
    reference_mask = _reference_mask(
        patient_norm,
        roi_mask,
        roi_dilation_iterations=roi_dilation_iterations,
    )

    vox_idx, support_meta = _sample_support(
        patient_norm,
        support_basis_label="patient",
        max_points=max_points,
        percentile_threshold=percentile_threshold,
        surface_percentile_threshold=surface_percentile_threshold,
        sampling_mode=sampling_mode,
        rng_seed=rng_seed,
    )
    patient_i = patient_norm[
        vox_idx[:, 0],
        vox_idx[:, 1],
        vox_idx[:, 2],
    ].astype(np.float32)

    control_values: list[np.ndarray] = []
    control_meta: list[dict[str, object]] = []
    resampled_count = 0
    for control_path in controls:
        control_data, control_affine, _ = _load_canonical(control_path)
        control_grid, resampled = _maybe_resample_to_reference(
            control_data,
            control_affine,
            ref_shape,
            patient_affine,
        )
        resampled_count += int(resampled)
        control_norm = _normalize_for_pairing(control_grid)
        matched, match_meta = _match_control_to_patient(
            control_norm,
            patient_norm,
            reference_mask,
        )
        values = matched[
            vox_idx[:, 0],
            vox_idx[:, 1],
            vox_idx[:, 2],
        ].astype(np.float32)
        control_values.append(values)
        control_meta.append({
            "path": str(control_path),
            "resampled_to_patient_grid": bool(resampled),
            "intensity_match": match_meta,
        })

    control_matrix = np.stack(control_values, axis=0).astype(np.float32)
    prior_mean = control_matrix.mean(axis=0).astype(np.float32)
    prior_std = control_matrix.std(axis=0, ddof=1).astype(np.float32)
    global_std = float(np.std(control_matrix))
    std_floor = max(
        float(std_floor_absolute),
        float(std_floor_fraction) * max(global_std, 1e-8),
    )
    denom = np.maximum(prior_std, std_floor).astype(np.float32)
    std_clamped = prior_std < std_floor
    residual_z = (np.abs(patient_i - prior_mean) / denom).astype(np.float32)

    percentile_value = float(np.percentile(residual_z, heatmap_percentile))
    max_value = float(residual_z.max())
    normalization_value = percentile_value if percentile_value > 1e-8 else max_value
    if normalization_value > 1e-8:
        residual_norm = np.clip(residual_z / normalization_value, 0.0, 1.0).astype(np.float32)
    else:
        residual_norm = np.zeros_like(residual_z, dtype=np.float32)

    positions = _voxel_to_world(vox_idx.astype(np.float32), patient_affine)
    scales, quats, orientation_meta = oriented_gaussian_parameters(
        patient_norm,
        vox_idx.astype(np.float32),
        patient_affine,
        sampling_mode=sampling_mode,
        support_voxel_count=int(support_meta["foreground_voxels_total"]),
        sampled_point_count=int(support_meta["n_gaussians"]),
    )

    patient_ply = output_dir / "control_prior_patient_splat.ply"
    prior_ply = output_dir / "control_prior_mean_splat.ply"
    residual_ply = output_dir / "control_prior_z_residual_splat.ply"
    _write_ply(patient_ply, positions, patient_i, scales, quats,
               heatmap=False, sampling_mode=sampling_mode)
    _write_ply(prior_ply, positions, np.clip(prior_mean, 0.0, 1.0), scales, quats,
               heatmap=False, sampling_mode=sampling_mode)
    _write_ply(residual_ply, positions, residual_norm, scales, quats,
               heatmap=True, sampling_mode=sampling_mode)

    if heatmap_method == "scatter":
        heatmap = _scatter_scores(vox_idx, residual_norm, ref_shape, smooth_sigma)
        heatmap_meta: dict[str, object] = {
            "heatmap_method": "scatter",
            "heatmap_backend": "max_scatter_plus_gaussian_filter",
            "heatmap_effective_sigma_vox": float(smooth_sigma),
        }
    elif heatmap_method == "gaussian":
        heatmap, heatmap_meta = _gaussian_scores_to_grid(
            vox_idx,
            residual_norm,
            scales,
            quats,
            ref_shape,
            patient_affine,
            smooth_sigma=smooth_sigma,
            gaussian_sigma_extent=heatmap_gaussian_sigma_extent,
            exact_max_points=heatmap_gaussian_exact_max_points,
        )
        heatmap_meta["heatmap_method"] = "gaussian"
    else:
        raise ValueError(
            "heatmap_method must be 'scatter' or 'gaussian', "
            f"got {heatmap_method!r}"
        )

    heatmap_path = output_dir / "control_prior_z_residual_heatmap.nii.gz"
    heatmap_img = nib.Nifti1Image(heatmap, patient_affine, patient_header)
    heatmap_img.header.set_data_dtype(np.float32)
    nib.save(heatmap_img, str(heatmap_path))

    scores_path = output_dir / "control_prior_z_scores.npy"
    np.save(str(scores_path), residual_z)

    roi_at_support_state = np.zeros(len(vox_idx), dtype=np.uint8)
    roi_metric_values: dict[str, object] = {}
    if roi_mask.any():
        roi_at_support = roi_mask[
            vox_idx[:, 0],
            vox_idx[:, 1],
            vox_idx[:, 2],
        ] > 0
        roi_at_support_state = roi_at_support.astype(np.uint8)
        roi_metric_values.update(_roi_metrics(residual_norm, roi_at_support))
        roi_metric_values["roi_path"] = str(roi) if roi is not None else None
        roi_metric_values["roi_resampled_to_patient_grid"] = roi_resampled

    state_path = output_dir / "control_gaussian_prior_state.npz"
    np.savez_compressed(
        state_path,
        state_schema=np.array(_CONTROL_GAUSSIAN_PRIOR_SCHEMA),
        method=np.array("patient_support_control_prior_zscore"),
        patient_volume=np.array(str(patient_volume)),
        control_volumes=np.array([str(p) for p in controls]),
        subject=np.array(subject),
        group=np.array(group),
        sampling_mode=np.array(sampling_mode),
        positions_world=positions.astype(np.float32),
        positions_voxel=vox_idx.astype(np.int32),
        scales=scales.astype(np.float32),
        rotations_quat=quats.astype(np.float32),
        patient_intensity=patient_i.astype(np.float32),
        control_prior_mean=prior_mean.astype(np.float32),
        control_prior_std=prior_std.astype(np.float32),
        control_prior_std_floor=np.array(float(std_floor), dtype=np.float32),
        residual_z=residual_z.astype(np.float32),
        residual_normalized=residual_norm.astype(np.float32),
        roi_at_support=roi_at_support_state,
        affine=patient_affine.astype(np.float64),
        source_shape=np.array(ref_shape, dtype=np.int64),
        rng_seed=np.array(int(rng_seed), dtype=np.int64),
    )

    metrics: dict[str, object] = {
        "method": "patient_support_control_prior_zscore",
        "gaussian_state_schema": _CONTROL_GAUSSIAN_PRIOR_SCHEMA,
        "same_gaussian_support": True,
        "patient_volume": str(patient_volume),
        "subject": subject,
        "group": group,
        "healthy_reference": "affine_resampled_control_gaussian_prior",
        "registration_method": "affine_header_resample_to_patient_grid",
        "intensity_matching": "zscore_match_over_patient_foreground_excluding_roi",
        "n_controls": len(controls),
        "min_controls": int(min_controls),
        "control_volumes": [str(p) for p in controls],
        "control_resampled_to_patient_grid_count": int(resampled_count),
        "reference_shape": list(ref_shape),
        "reference_mask_voxels": int(reference_mask.sum()),
        "roi_dilation_iterations": int(roi_dilation_iterations),
        "std_floor_fraction": float(std_floor_fraction),
        "std_floor_absolute": float(std_floor_absolute),
        "std_floor_value": float(std_floor),
        "std_clamped_count": int(std_clamped.sum()),
        "std_clamped_fraction": float(std_clamped.mean()),
        "heatmap_percentile": float(heatmap_percentile),
        **heatmap_meta,
        "residual_z_normalization_percentile_value": percentile_value,
        "residual_z_normalization_value": normalization_value,
        "residual_z_min": float(residual_z.min()),
        "residual_z_max": float(residual_z.max()),
        "residual_z_mean": float(residual_z.mean()),
        "residual_z_p95": float(np.percentile(residual_z, 95)),
        "residual_z_p99": float(np.percentile(residual_z, 99)),
        "residual_norm_mean": float(residual_norm.mean()),
        "residual_norm_p99": float(np.percentile(residual_norm, 99)),
        "world_bounds_min": positions.min(axis=0).tolist(),
        "world_bounds_max": positions.max(axis=0).tolist(),
        "control_metadata": control_meta,
        "artifacts": {
            "control_prior_patient_splat": patient_ply.name,
            "control_prior_mean_splat": prior_ply.name,
            "control_prior_z_residual_splat": residual_ply.name,
            "control_prior_z_residual_heatmap": heatmap_path.name,
            "control_prior_z_scores": scores_path.name,
            "control_gaussian_prior_state": state_path.name,
        },
        **support_meta,
        **orientation_meta,
        **roi_metric_values,
    }

    metrics_path = output_dir / "control_gaussian_prior_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build patient-support Gaussian residuals against healthy controls."
    )
    p.add_argument("--patient-volume", type=Path, required=True)
    p.add_argument("--control-volume", type=Path, action="append", default=[])
    p.add_argument("--control-list", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--roi", type=Path, default=None)
    p.add_argument("--min-controls", type=int, default=2)
    p.add_argument("--max-points", type=int, default=50_000)
    p.add_argument("--percentile-threshold", type=float, default=5.0)
    p.add_argument("--surface-percentile-threshold", type=float, default=70.0)
    p.add_argument(
        "--sampling-mode",
        choices=["surface", "volume", "uniform"],
        default="volume",
    )
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--heatmap-percentile", type=float, default=99.0)
    p.add_argument("--smooth-sigma", type=float, default=1.0)
    p.add_argument("--heatmap-method", choices=["scatter", "gaussian"], default="gaussian")
    p.add_argument("--heatmap-gaussian-sigma-extent", type=float, default=2.5)
    p.add_argument("--heatmap-gaussian-exact-max-points", type=int, default=50_000)
    p.add_argument("--std-floor-fraction", type=float, default=0.05)
    p.add_argument("--std-floor-absolute", type=float, default=1e-3)
    p.add_argument("--roi-dilation-iterations", type=int, default=3)
    p.add_argument("--subject", default="")
    p.add_argument("--group", default="")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    control_volumes = [Path(p) for p in args.control_volume] + _load_control_list(
        args.control_list
    )
    metrics = build_control_gaussian_prior_residual(
        patient_volume=args.patient_volume,
        control_volumes=control_volumes,
        output_dir=args.output_dir,
        roi=args.roi,
        min_controls=args.min_controls,
        max_points=args.max_points,
        percentile_threshold=args.percentile_threshold,
        surface_percentile_threshold=args.surface_percentile_threshold,
        sampling_mode=args.sampling_mode,
        rng_seed=args.rng_seed,
        heatmap_percentile=args.heatmap_percentile,
        smooth_sigma=args.smooth_sigma,
        heatmap_method=args.heatmap_method,
        heatmap_gaussian_sigma_extent=args.heatmap_gaussian_sigma_extent,
        heatmap_gaussian_exact_max_points=args.heatmap_gaussian_exact_max_points,
        std_floor_fraction=args.std_floor_fraction,
        std_floor_absolute=args.std_floor_absolute,
        roi_dilation_iterations=args.roi_dilation_iterations,
        subject=args.subject,
        group=args.group,
    )
    print(
        "Control Gaussian prior complete: "
        f"N={metrics['n_gaussians']} controls={metrics['n_controls']} "
        f"z_p99={metrics['residual_z_p99']:.4f}"
    )


if __name__ == "__main__":
    main()
