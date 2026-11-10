"""Audit how well a browser-visible Gaussian PLY preserves its source MRI.

The direct MRI splat path is intentionally not a trained neural 3DGS.  This
module makes that explicit by scattering a PLY back into the reference NIfTI
grid and reporting coverage and reconstruction metrics.  It is a paper-facing
quality gate for the MRI -> 3D Gaussian representation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Literal, Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from fcd_prototype.nifti_preview import _robust_normalize
from fcd_prototype.splat_ply_export import _COORDINATE_FRAME_SCHEMA, _PLY_DTYPE


_READINESS_FOREGROUND_COVERAGE_PASS = 0.50
_READINESS_FOREGROUND_COVERAGE_REVIEW = 0.25
_READINESS_MIP_MEAN_MAE_PASS = 0.10
_READINESS_MIP_MEAN_MAE_REVIEW = 0.20
_READINESS_REPRESENTED_FOREGROUND_MAE_REVIEW = 0.20
_COORDINATE_AFFINE_ATOL_MM = 1e-3
_COORDINATE_WORLD_ATOL_MM = 1e-3
_COORDINATE_PLY_STATE_ATOL_MM = 1e-4


def _load_reference(path: Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return _robust_normalize(data).astype(np.float32), img.affine.astype(np.float64), img.header.copy()


def _read_binary_ply(path: Path) -> np.ndarray:
    with Path(path).open("rb") as fh:
        header_lines: list[str] = []
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"PLY header ended unexpectedly: {path}")
            text = line.decode("ascii", errors="replace").strip()
            header_lines.append(text)
            if text == "end_header":
                break
        if not any(line == "format binary_little_endian 1.0" for line in header_lines):
            raise ValueError(f"Only binary_little_endian PLY is supported: {path}")
        n_vertices = next(
            int(line.split()[-1])
            for line in header_lines
            if line.startswith("element vertex")
        )
        payload = fh.read()
    arr = np.frombuffer(payload, dtype=_PLY_DTYPE, count=n_vertices)
    if len(arr) != n_vertices:
        raise ValueError(f"Expected {n_vertices} vertices, read {len(arr)} from {path}")
    return arr


def _world_to_voxel(world_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    hom = np.concatenate([world_xyz.astype(np.float64), np.ones((len(world_xyz), 1))], axis=1)
    return (np.linalg.inv(affine) @ hom.T).T[:, :3]


def _voxel_to_world(vox_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    hom = np.concatenate([vox_xyz.astype(np.float64), np.ones((len(vox_xyz), 1))], axis=1)
    return (affine @ hom.T).T[:, :3]


def _audit_coordinate_contract(
    arr: np.ndarray,
    *,
    reference_shape: tuple[int, int, int],
    reference_affine: np.ndarray,
    splat_ply: Path,
) -> dict[str, object]:
    """Validate the exported PLY against its direct-splat Gaussian state sidecar."""
    state_path = Path(splat_ply).parent / (Path(splat_ply).stem + "_state.npz")
    warnings: list[str] = []
    blockers: list[str] = []
    observed: dict[str, object] = {
        "state_path": str(state_path),
        "state_exists": state_path.is_file(),
    }
    if not state_path.is_file():
        warnings.append("coordinate_contract_state_missing")
        return {
            "status": "warning",
            "blockers": blockers,
            "warnings": warnings,
            "blocker_count": 0,
            "warning_count": len(warnings),
            "observed": observed,
        }

    try:
        with np.load(state_path, allow_pickle=False) as state:
            schema = str(state["coordinate_frame_schema"]) if "coordinate_frame_schema" in state else None
            positions_world = np.asarray(state["positions_world"], dtype=np.float64)
            positions_voxel = np.asarray(state["positions_voxel"], dtype=np.float64)
            state_affine = np.asarray(state["affine"], dtype=np.float64)
            source_shape = tuple(int(v) for v in np.asarray(state["source_shape"]).tolist())
    except Exception as exc:
        blockers.append("coordinate_contract_state_unreadable")
        observed["error"] = str(exc)
        return {
            "status": "fail",
            "blockers": blockers,
            "warnings": warnings,
            "blocker_count": len(blockers),
            "warning_count": len(warnings),
            "observed": observed,
        }

    ply_world = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    ply_state_max_abs = (
        float(np.abs(ply_world - positions_world).max())
        if len(ply_world) and ply_world.shape == positions_world.shape else None
    )
    affine_max_abs = float(np.abs(state_affine - reference_affine).max())
    expected_world = _voxel_to_world(positions_voxel, reference_affine)
    voxel_to_world_max_abs = (
        float(np.abs(expected_world - ply_world).max())
        if len(ply_world) and expected_world.shape == ply_world.shape else None
    )
    in_bounds = (
        (positions_voxel[:, 0] >= 0) & (positions_voxel[:, 0] < reference_shape[0]) &
        (positions_voxel[:, 1] >= 0) & (positions_voxel[:, 1] < reference_shape[1]) &
        (positions_voxel[:, 2] >= 0) & (positions_voxel[:, 2] < reference_shape[2])
    )
    in_bounds_fraction = float(in_bounds.mean()) if in_bounds.size else 0.0

    observed.update({
        "schema": schema,
        "n_ply_points": int(len(ply_world)),
        "n_state_points": int(len(positions_world)),
        "source_shape": list(source_shape),
        "reference_shape": [int(v) for v in reference_shape],
        "affine_max_abs_diff_mm": affine_max_abs,
        "ply_state_world_max_abs_diff_mm": ply_state_max_abs,
        "voxel_to_world_max_abs_diff_mm": voxel_to_world_max_abs,
        "state_voxel_positions_in_reference_bounds_fraction": in_bounds_fraction,
    })

    if schema != _COORDINATE_FRAME_SCHEMA:
        blockers.append("coordinate_contract_schema_mismatch")
    if positions_world.shape != ply_world.shape or positions_voxel.shape != ply_world.shape:
        blockers.append("coordinate_contract_point_count_or_shape_mismatch")
    if source_shape != tuple(int(v) for v in reference_shape):
        blockers.append("coordinate_contract_source_shape_mismatch")
    if affine_max_abs > _COORDINATE_AFFINE_ATOL_MM:
        blockers.append("coordinate_contract_affine_mismatch")
    if ply_state_max_abs is None or ply_state_max_abs > _COORDINATE_PLY_STATE_ATOL_MM:
        blockers.append("coordinate_contract_ply_state_world_mismatch")
    if (
        voxel_to_world_max_abs is None
        or voxel_to_world_max_abs > _COORDINATE_WORLD_ATOL_MM
    ):
        blockers.append("coordinate_contract_voxel_to_world_mismatch")
    if in_bounds_fraction < 1.0:
        blockers.append("coordinate_contract_voxel_positions_out_of_bounds")

    status = "fail" if blockers else ("warning" if warnings else "pass")
    return {
        "status": status,
        "blockers": blockers,
        "warnings": warnings,
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "thresholds": {
            "affine_max_abs_diff_mm": _COORDINATE_AFFINE_ATOL_MM,
            "voxel_to_world_max_abs_diff_mm": _COORDINATE_WORLD_ATOL_MM,
            "ply_state_world_max_abs_diff_mm": _COORDINATE_PLY_STATE_ATOL_MM,
        },
        "observed": observed,
    }


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert one normalized wxyz quaternion to a 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12 or not np.isfinite(norm):
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _safe_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size < 2 or b.size < 2:
        return None
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std <= 1e-8 or b_std <= 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _metrics(reference: np.ndarray, recon: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    n = int(mask.sum())
    if n == 0:
        return {
            "voxel_count": 0,
            "mae": None,
            "rmse": None,
            "psnr": None,
            "correlation": None,
        }
    ref = reference[mask].astype(np.float64)
    pred = recon[mask].astype(np.float64)
    diff = pred - ref
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(float(np.mean(diff * diff))))
    psnr = None if rmse <= 1e-12 else float(20.0 * math.log10(1.0 / rmse))
    return {
        "voxel_count": n,
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "correlation": _safe_corr(ref, pred),
    }


def _mip_metrics(reference: np.ndarray, recon: np.ndarray) -> dict[str, object]:
    views = {
        "axial": (reference.max(axis=2), recon.max(axis=2)),
        "coronal": (reference.max(axis=1), recon.max(axis=1)),
        "sagittal": (reference.max(axis=0), recon.max(axis=0)),
    }
    out: dict[str, object] = {}
    maes: list[float] = []
    rmses: list[float] = []
    for view, (ref, pred) in views.items():
        mask = np.ones(ref.shape, dtype=bool)
        values = _metrics(ref, pred, mask)
        out[view] = values
        if values["mae"] is not None:
            maes.append(float(values["mae"]))
        if values["rmse"] is not None:
            rmses.append(float(values["rmse"]))
    out["mean_mae"] = float(np.mean(maes)) if maes else None
    out["mean_rmse"] = float(np.mean(rmses)) if rmses else None
    return out


def _build_representation_readiness_gate(
    *,
    foreground_coverage: float,
    all_foreground_metrics: dict[str, object],
    represented_foreground_metrics: dict[str, object],
    mip_metrics: dict[str, object],
    point_info: dict[str, object],
    coordinate_contract: dict[str, object],
) -> dict[str, object]:
    """Classify whether a PLY supports a paper-facing MRI-to-3DGS claim."""
    blockers: list[str] = []
    warnings: list[str] = []

    total_points = int(point_info.get("total_points") or 0)
    in_bounds = int(point_info.get("points_in_reference_bounds") or 0)
    kernel_mode = str(point_info.get("kernel_mode") or "unknown")
    mip_mae = mip_metrics.get("mean_mae")
    represented_mae = represented_foreground_metrics.get("mae")
    all_foreground_mae = all_foreground_metrics.get("mae")

    if total_points <= 0:
        blockers.append("splat_has_no_gaussians")
    if in_bounds <= 0:
        blockers.append("no_gaussians_in_reference_grid")

    if foreground_coverage < _READINESS_FOREGROUND_COVERAGE_REVIEW:
        blockers.append("foreground_coverage_below_review_floor")
    elif foreground_coverage < _READINESS_FOREGROUND_COVERAGE_PASS:
        warnings.append("foreground_coverage_below_pass_threshold")

    if mip_mae is None:
        blockers.append("missing_mip_reconstruction_error")
    elif float(mip_mae) > _READINESS_MIP_MEAN_MAE_REVIEW:
        blockers.append("mip_reconstruction_error_above_review_ceiling")
    elif float(mip_mae) > _READINESS_MIP_MEAN_MAE_PASS:
        warnings.append("mip_reconstruction_error_above_pass_threshold")

    if represented_mae is None:
        blockers.append("missing_represented_foreground_error")
    elif float(represented_mae) > _READINESS_REPRESENTED_FOREGROUND_MAE_REVIEW:
        warnings.append("represented_foreground_error_high")

    if kernel_mode != "gaussian":
        warnings.append("audit_did_not_use_stored_gaussian_kernel")

    coordinate_status = str(coordinate_contract.get("status") or "unknown")
    for blocker in coordinate_contract.get("blockers", []):
        blockers.append(str(blocker))
    for warning in coordinate_contract.get("warnings", []):
        warnings.append(str(warning))
    if coordinate_status not in {"pass", "warning", "fail"}:
        warnings.append("coordinate_contract_status_unknown")

    if blockers:
        status = "fail"
    elif warnings:
        status = "review"
    else:
        status = "pass"

    claim_allowed = status == "pass"
    if claim_allowed:
        claim_class = "volumetric_mri_gaussian_representation"
    elif status == "review":
        claim_class = "review_required_before_3dgs_claim"
    else:
        claim_class = "sparse_or_low_fidelity_visualization_only"

    return {
        "status": status,
        "blockers": blockers,
        "warnings": warnings,
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "thresholds": {
            "foreground_coverage_pass": _READINESS_FOREGROUND_COVERAGE_PASS,
            "foreground_coverage_review_floor": _READINESS_FOREGROUND_COVERAGE_REVIEW,
            "mip_mean_mae_pass": _READINESS_MIP_MEAN_MAE_PASS,
            "mip_mean_mae_review_ceiling": _READINESS_MIP_MEAN_MAE_REVIEW,
            "represented_foreground_mae_review_ceiling": (
                _READINESS_REPRESENTED_FOREGROUND_MAE_REVIEW
            ),
            "gaussian_kernel_audit_required_for_clean_pass": True,
            "coordinate_contract_required_for_clean_pass": True,
        },
        "observed": {
            "foreground_coverage": float(foreground_coverage),
            "mip_mean_mae": float(mip_mae) if mip_mae is not None else None,
            "represented_foreground_mae": (
                float(represented_mae) if represented_mae is not None else None
            ),
            "all_foreground_mae": (
                float(all_foreground_mae) if all_foreground_mae is not None else None
            ),
            "total_points": total_points,
            "points_in_reference_bounds": in_bounds,
            "kernel_mode": kernel_mode,
            "coordinate_contract_status": coordinate_status,
        },
        "mri_to_3dgs_claim_allowed": claim_allowed,
        "claim_class": claim_class,
    }


def _scatter_to_grid(
    arr: np.ndarray,
    reference_shape: tuple[int, int, int],
    reference_affine: np.ndarray,
    *,
    kernel_mode: Literal["nearest", "gaussian"],
    gaussian_sigma_extent: float,
    smooth_sigma_vox: float,
    max_points: Optional[int],
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    total_points = int(len(arr))
    if max_points is not None and total_points > max_points:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(total_points, size=int(max_points), replace=False)
        arr = arr[idx]
        subsampled = True
    else:
        subsampled = False

    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    vox_float = _world_to_voxel(xyz, reference_affine)
    vox = np.rint(vox_float).astype(np.int64)
    in_bounds = (
        (vox[:, 0] >= 0) & (vox[:, 0] < reference_shape[0]) &
        (vox[:, 1] >= 0) & (vox[:, 1] < reference_shape[1]) &
        (vox[:, 2] >= 0) & (vox[:, 2] < reference_shape[2])
    )
    vox = vox[in_bounds]
    vox_float = vox_float[in_bounds]
    valid_arr = arr[in_bounds]

    intensities = (
        valid_arr["red"].astype(np.float32)
        + valid_arr["green"].astype(np.float32)
        + valid_arr["blue"].astype(np.float32)
    ) / (3.0 * 255.0)
    weights = np.clip(valid_arr["opacity"].astype(np.float32), 1e-4, None)

    numerator = np.zeros(reference_shape, dtype=np.float32)
    denominator = np.zeros(reference_shape, dtype=np.float32)
    if kernel_mode == "nearest":
        np.add.at(numerator, (vox[:, 0], vox[:, 1], vox[:, 2]), intensities * weights)
        np.add.at(denominator, (vox[:, 0], vox[:, 1], vox[:, 2]), weights)
    elif kernel_mode == "gaussian":
        if gaussian_sigma_extent <= 0:
            raise ValueError("gaussian_sigma_extent must be > 0 for gaussian kernel_mode")
        spacing = np.sqrt((reference_affine[:3, :3] ** 2).sum(axis=0)).astype(np.float64)
        spacing = np.clip(spacing, 1e-6, None)
        linear = reference_affine[:3, :3].astype(np.float64)
        origin = reference_affine[:3, 3].astype(np.float64)
        extent_sq = float(gaussian_sigma_extent * gaussian_sigma_extent)

        for rec, center_world, center_vox, intensity, opacity in zip(
            valid_arr, xyz[in_bounds], vox_float, intensities, weights
        ):
            scales = np.array(
                [rec["scale_x"], rec["scale_y"], rec["scale_z"]],
                dtype=np.float64,
            )
            scales = np.clip(np.abs(scales), 1e-3, None)
            world_radius = float(gaussian_sigma_extent * scales.max())
            radius_vox = np.ceil(world_radius / spacing).astype(np.int64) + 1
            lo = np.maximum(np.floor(center_vox).astype(np.int64) - radius_vox, 0)
            hi = np.minimum(np.ceil(center_vox).astype(np.int64) + radius_vox + 1, reference_shape)
            if np.any(hi <= lo):
                continue

            ii, jj, kk = np.meshgrid(
                np.arange(lo[0], hi[0]),
                np.arange(lo[1], hi[1]),
                np.arange(lo[2], hi[2]),
                indexing="ij",
            )
            ijk = np.stack([ii, jj, kk], axis=-1).reshape(-1, 3).astype(np.float64)
            world = ijk @ linear.T + origin
            delta = world - center_world
            rot = _quat_wxyz_to_matrix(
                np.array([rec["rot_w"], rec["rot_x"], rec["rot_y"], rec["rot_z"]])
            )
            # Rotation columns are the local Gaussian axes in world space.
            local = delta @ rot
            mahal = np.sum((local / scales) ** 2, axis=1)
            keep = mahal <= extent_sq
            if not np.any(keep):
                continue
            kept_ijk = ijk[keep].astype(np.int64)
            kernel = np.exp(-0.5 * mahal[keep]).astype(np.float32) * float(opacity)
            np.add.at(
                numerator,
                (kept_ijk[:, 0], kept_ijk[:, 1], kept_ijk[:, 2]),
                float(intensity) * kernel,
            )
            np.add.at(
                denominator,
                (kept_ijk[:, 0], kept_ijk[:, 1], kept_ijk[:, 2]),
                kernel,
            )
    else:
        raise ValueError(f"kernel_mode must be 'nearest' or 'gaussian', got {kernel_mode!r}")

    if smooth_sigma_vox > 0:
        numerator = gaussian_filter(numerator, sigma=float(smooth_sigma_vox))
        denominator = gaussian_filter(denominator, sigma=float(smooth_sigma_vox))

    recon = np.zeros(reference_shape, dtype=np.float32)
    mask = denominator > 1e-8
    recon[mask] = numerator[mask] / denominator[mask]
    recon = np.clip(recon, 0.0, 1.0).astype(np.float32)
    info = {
        "total_points": total_points,
        "audited_points": int(len(arr)),
        "points_in_reference_bounds": int(len(valid_arr)),
        "subsampled": subsampled,
        "smooth_sigma_vox": float(smooth_sigma_vox),
        "kernel_mode": kernel_mode,
        "gaussian_sigma_extent": float(gaussian_sigma_extent) if kernel_mode == "gaussian" else None,
    }
    return recon, mask, info


def audit_splat_reconstruction(
    *,
    reference_volume: Path,
    splat_ply: Path,
    output_json: Optional[Path] = None,
    output_nifti: Optional[Path] = None,
    foreground_threshold: float = 0.01,
    kernel_mode: Literal["nearest", "gaussian"] = "gaussian",
    gaussian_sigma_extent: float = 2.5,
    smooth_sigma_vox: float = 1.0,
    max_points: Optional[int] = 100_000,
    rng_seed: int = 42,
) -> dict[str, object]:
    """Scatter a PLY back to NIfTI space and report reconstruction metrics."""
    reference, affine, header = _load_reference(Path(reference_volume))
    arr = _read_binary_ply(Path(splat_ply))
    coordinate_contract = _audit_coordinate_contract(
        arr,
        reference_shape=tuple(int(v) for v in reference.shape[:3]),
        reference_affine=affine,
        splat_ply=Path(splat_ply),
    )
    recon, represented_mask, point_info = _scatter_to_grid(
        arr,
        tuple(int(v) for v in reference.shape[:3]),
        affine,
        kernel_mode=kernel_mode,
        gaussian_sigma_extent=gaussian_sigma_extent,
        smooth_sigma_vox=smooth_sigma_vox,
        max_points=max_points,
        rng_seed=rng_seed,
    )

    foreground_mask = reference > float(foreground_threshold)
    represented_foreground = represented_mask & foreground_mask
    coverage = (
        float(represented_foreground.sum() / foreground_mask.sum())
        if int(foreground_mask.sum()) else 0.0
    )
    all_foreground_metrics = _metrics(reference, recon, foreground_mask)
    represented_foreground_metrics = _metrics(reference, recon, represented_foreground)
    mip_metrics = _mip_metrics(reference, recon)
    readiness_gate = _build_representation_readiness_gate(
        foreground_coverage=coverage,
        all_foreground_metrics=all_foreground_metrics,
        represented_foreground_metrics=represented_foreground_metrics,
        mip_metrics=mip_metrics,
        point_info=point_info,
        coordinate_contract=coordinate_contract,
    )

    result = {
        "method": "splat_to_reference_grid_reconstruction_audit",
        "reference_volume": str(reference_volume),
        "splat_ply": str(splat_ply),
        "foreground_threshold": float(foreground_threshold),
        "reference_shape": [int(v) for v in reference.shape[:3]],
        "reference_foreground_voxels": int(foreground_mask.sum()),
        "represented_voxels": int(represented_mask.sum()),
        "represented_foreground_voxels": int(represented_foreground.sum()),
        "foreground_coverage": coverage,
        **point_info,
        "coordinate_contract": coordinate_contract,
        "all_foreground_metrics": all_foreground_metrics,
        "represented_foreground_metrics": represented_foreground_metrics,
        "mip_metrics": mip_metrics,
        "mri_to_3dgs_readiness_status": readiness_gate["status"],
        "mri_to_3dgs_readiness_blocker_count": readiness_gate["blocker_count"],
        "mri_to_3dgs_readiness_warning_count": readiness_gate["warning_count"],
        "mri_to_3dgs_claim_allowed": readiness_gate["mri_to_3dgs_claim_allowed"],
        "mri_to_3dgs_claim_class": readiness_gate["claim_class"],
        "representation_readiness_gate": readiness_gate,
        "limitations": [
            "This reconstructs the browser PLY back to voxel space with the stored Gaussian scale, rotation, opacity, and color parameters.",
            "The gaussian audit mode is still a CPU approximation of browser splat accumulation, not a differentiable training objective.",
            "Direct sampled PLY splats are not optimized neural 3DGS fields; low coverage or high error is expected for sparse/surface-only exports.",
            "Use this as a representation-quality gate before interpreting patient-vs-normal splat subtraction.",
        ],
    }

    if output_nifti is not None:
        output_nifti = Path(output_nifti)
        output_nifti.parent.mkdir(parents=True, exist_ok=True)
        img = nib.Nifti1Image(recon.astype(np.float32), affine, header)
        img.header.set_data_dtype(np.float32)
        nib.save(img, str(output_nifti))
        result["reconstruction_nifti"] = str(output_nifti)

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2))

    return result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit MRI Gaussian PLY reconstruction quality.")
    p.add_argument("--reference-volume", type=Path, required=True)
    p.add_argument("--splat-ply", type=Path, required=True)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-nifti", type=Path, default=None)
    p.add_argument("--foreground-threshold", type=float, default=0.01)
    p.add_argument("--kernel-mode", choices=["nearest", "gaussian"], default="gaussian")
    p.add_argument("--gaussian-sigma-extent", type=float, default=2.5)
    p.add_argument("--smooth-sigma-vox", type=float, default=1.0)
    p.add_argument("--max-points", type=int, default=100_000)
    p.add_argument("--rng-seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = audit_splat_reconstruction(
        reference_volume=args.reference_volume,
        splat_ply=args.splat_ply,
        output_json=args.output_json,
        output_nifti=args.output_nifti,
        foreground_threshold=args.foreground_threshold,
        kernel_mode=args.kernel_mode,
        gaussian_sigma_extent=args.gaussian_sigma_extent,
        smooth_sigma_vox=args.smooth_sigma_vox,
        max_points=args.max_points,
        rng_seed=args.rng_seed,
    )
    print(json.dumps({
        "foreground_coverage": result["foreground_coverage"],
        "all_foreground_mae": result["all_foreground_metrics"]["mae"],
        "all_foreground_psnr": result["all_foreground_metrics"]["psnr"],
        "represented_foreground_mae": result["represented_foreground_metrics"]["mae"],
    }, indent=2))


if __name__ == "__main__":
    main()
