"""Paired-support Gaussian splat residual for aligned MRI volumes.

This module builds patient and synthetic-normal Gaussian splats on the exact
same world-space support, making index-wise splat subtraction meaningful. It is
the direct CPU path for the paper claim: MRI -> 3D Gaussian representation ->
healthy counterfactual -> Gaussian-space residual.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from fcd_prototype.splat_ply_export import (
    _PLY_DTYPE,
    _PLY_PROPERTIES,
    _build_surface_weights,
    _build_volume_weights,
    _foreground_mask_for_sampling,
    _robust_normalize,
    _stratified_weighted_sample_indices,
    _volume_opacity_from_intensity,
    _voxel_to_world,
    oriented_gaussian_parameters,
)


_PAIRED_GAUSSIAN_STATE_SCHEMA = "paired_gaussian_residual_v1"
_SUPPORT_BASIS_CHOICES = ("union", "patient", "normal", "intersection")


def _support_signal_for_basis(
    patient_norm: np.ndarray,
    normal_norm: np.ndarray,
    support_basis: str,
) -> tuple[np.ndarray, str]:
    if support_basis == "union":
        return np.maximum(patient_norm, normal_norm), "max(patient, synthetic_normal)"
    if support_basis == "patient":
        return patient_norm, "patient"
    if support_basis == "normal":
        return normal_norm, "synthetic_normal"
    if support_basis == "intersection":
        return np.minimum(patient_norm, normal_norm), "min(patient, synthetic_normal)"
    raise ValueError(
        "support_basis must be one of "
        f"{', '.join(_SUPPORT_BASIS_CHOICES)}, got {support_basis!r}"
    )


def _load_canonical(path: Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data.astype(np.float32), img.affine.astype(np.float64), img.header.copy()


def _resample_linear(
    data: np.ndarray,
    src_affine: np.ndarray,
    ref_shape: tuple[int, int, int],
    ref_affine: np.ndarray,
) -> np.ndarray:
    inv_src = np.linalg.inv(src_affine)
    out = np.zeros(ref_shape, dtype=np.float32)
    for z0 in range(0, ref_shape[2], 16):
        z1 = min(ref_shape[2], z0 + 16)
        i, j, k = np.meshgrid(
            np.arange(ref_shape[0]),
            np.arange(ref_shape[1]),
            np.arange(z0, z1),
            indexing="ij",
        )
        ref_vox = np.stack([i, j, k, np.ones_like(i)], axis=0).reshape(4, -1)
        world = ref_affine @ ref_vox
        src_vox = inv_src @ world
        sampled = map_coordinates(
            data,
            [src_vox[0], src_vox[1], src_vox[2]],
            order=1,
            mode="nearest",
        ).reshape((ref_shape[0], ref_shape[1], z1 - z0))
        out[:, :, z0:z1] = sampled.astype(np.float32)
    return out


def _resample_nearest(
    data: np.ndarray,
    src_affine: np.ndarray,
    ref_shape: tuple[int, int, int],
    ref_affine: np.ndarray,
) -> np.ndarray:
    inv_src = np.linalg.inv(src_affine)
    out = np.zeros(ref_shape, dtype=np.float32)
    for z0 in range(0, ref_shape[2], 16):
        z1 = min(ref_shape[2], z0 + 16)
        i, j, k = np.meshgrid(
            np.arange(ref_shape[0]),
            np.arange(ref_shape[1]),
            np.arange(z0, z1),
            indexing="ij",
        )
        ref_vox = np.stack([i, j, k, np.ones_like(i)], axis=0).reshape(4, -1)
        world = ref_affine @ ref_vox
        src_vox = inv_src @ world
        sampled = map_coordinates(
            data,
            [src_vox[0], src_vox[1], src_vox[2]],
            order=0,
            mode="nearest",
        ).reshape((ref_shape[0], ref_shape[1], z1 - z0))
        out[:, :, z0:z1] = sampled.astype(np.float32)
    return out


def _maybe_resample_to_reference(
    data: np.ndarray,
    affine: np.ndarray,
    ref_shape: tuple[int, int, int],
    ref_affine: np.ndarray,
    *,
    nearest: bool = False,
) -> tuple[np.ndarray, bool]:
    if data.shape[:3] == ref_shape and np.allclose(affine, ref_affine):
        return data.astype(np.float32), False
    if nearest:
        return _resample_nearest(data, affine, ref_shape, ref_affine), True
    return _resample_linear(data, affine, ref_shape, ref_affine), True


def _normalize_for_pairing(data: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
        return np.clip(finite, 0.0, 1.0).astype(np.float32)
    return _robust_normalize(finite)


def _sample_support(
    support_signal: np.ndarray,
    *,
    support_basis_label: str,
    max_points: int,
    percentile_threshold: float,
    surface_percentile_threshold: float,
    sampling_mode: str,
    rng_seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if sampling_mode not in ("surface", "volume", "uniform"):
        raise ValueError(
            "sampling_mode must be 'surface', 'volume', or 'uniform', "
            f"got {sampling_mode!r}"
        )
    if max_points <= 0:
        raise ValueError(f"max_points must be > 0, got {max_points}")

    try:
        fg_mask, effective, _threshold, foreground_meta = _foreground_mask_for_sampling(
            support_signal,
            percentile_threshold=percentile_threshold,
            sampling_mode=sampling_mode,
            surface_percentile_threshold=surface_percentile_threshold,
        )
    except ValueError as exc:
        raise ValueError("No foreground voxels found for paired Gaussian support.") from exc
    fg_idx = np.argwhere(fg_mask)
    if len(fg_idx) == 0:
        raise ValueError("Foreground threshold removed all paired Gaussian support voxels.")

    rng = np.random.default_rng(rng_seed)
    n = min(max_points, len(fg_idx))
    surface_voxels_total: Optional[int] = None

    if sampling_mode == "surface":
        weights, surface_shell = _build_surface_weights(support_signal, fg_mask)
        surface_voxels_total = int(surface_shell.sum())
        sel = rng.choice(len(fg_idx), size=n, replace=False, p=weights) if n < len(fg_idx) else np.arange(len(fg_idx))
        sampling_meta: dict[str, object] = {}
    elif sampling_mode == "volume":
        weights = _build_volume_weights(support_signal, fg_mask)
        sel, sampling_meta = _stratified_weighted_sample_indices(
            fg_idx,
            weights,
            n=n,
            rng=rng,
        )
    else:
        sel = rng.choice(len(fg_idx), size=n, replace=False) if n < len(fg_idx) else np.arange(len(fg_idx))
        sampling_meta = {}

    meta: dict[str, object] = {
        "support_basis": support_basis_label,
        "sampling_mode": sampling_mode,
        "max_points": max_points,
        "n_gaussians": int(n),
        "foreground_voxels_total": int(len(fg_idx)),
        "percentile_threshold": percentile_threshold,
        "effective_percentile_threshold": effective,
        "rng_seed": rng_seed,
        **foreground_meta,
        **sampling_meta,
    }
    if surface_voxels_total is not None:
        meta["surface_voxels_total"] = surface_voxels_total
    if sampling_mode == "volume":
        meta["volume_voxels_total"] = int(len(fg_idx))
        meta["sampling_weight_model"] = "full_foreground_base_plus_intensity_gradient"
    return fg_idx[sel].astype(np.int32), meta


def _heatmap_colors(t: np.ndarray) -> np.ndarray:
    stops = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float32)
    colors = np.array([
        [0.05, 0.05, 0.30],
        [0.00, 0.50, 0.90],
        [1.00, 0.85, 0.00],
        [0.90, 0.00, 0.05],
    ], dtype=np.float32)
    t = np.clip(t, 0.0, 1.0).astype(np.float32)
    rgb = np.zeros((len(t), 3), dtype=np.float32)
    for idx in range(len(stops) - 1):
        lo, hi = stops[idx], stops[idx + 1]
        mask = (t >= lo) & (t <= hi)
        if not mask.any():
            continue
        frac = (t[mask] - lo) / (hi - lo)
        rgb[mask] = colors[idx] * (1.0 - frac[:, None]) + colors[idx + 1] * frac[:, None]
    return rgb


def _write_ply(
    path: Path,
    positions: np.ndarray,
    intensities: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    *,
    heatmap: bool = False,
    sampling_mode: str = "surface",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(positions)
    if heatmap:
        rgb = np.clip(_heatmap_colors(intensities) * 255, 0, 255).astype(np.uint8)
        opacity = np.clip(intensities, 0.0, 1.0).astype(np.float32)
    else:
        gray = np.clip(intensities * 255, 0, 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)
        if sampling_mode == "volume":
            opacity = _volume_opacity_from_intensity(intensities)
        else:
            opacity = np.clip(0.15 + intensities * 0.50, 0.0, 1.0).astype(np.float32)

    arr = np.zeros(n, dtype=_PLY_DTYPE)
    arr["x"] = positions[:, 0]
    arr["y"] = positions[:, 1]
    arr["z"] = positions[:, 2]
    arr["red"] = rgb[:, 0]
    arr["green"] = rgb[:, 1]
    arr["blue"] = rgb[:, 2]
    arr["opacity"] = opacity
    arr["scale_x"] = scales[:, 0]
    arr["scale_y"] = scales[:, 1]
    arr["scale_z"] = scales[:, 2]
    arr["rot_w"] = quats[:, 0]
    arr["rot_x"] = quats[:, 1]
    arr["rot_y"] = quats[:, 2]
    arr["rot_z"] = quats[:, 3]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + _PLY_PROPERTIES
        + "end_header\n"
    ).encode("ascii")
    with path.open("wb") as fh:
        fh.write(header)
        fh.write(arr.tobytes())


def _scatter_scores(
    vox_idx: np.ndarray,
    scores: np.ndarray,
    ref_shape: tuple[int, int, int],
    smooth_sigma: float,
) -> np.ndarray:
    vol = np.zeros(ref_shape, dtype=np.float32)
    np.maximum.at(vol, (vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]), scores.astype(np.float32))
    if smooth_sigma > 0:
        vol = gaussian_filter(vol, sigma=smooth_sigma)
    vmax = float(vol.max())
    if vmax > 0:
        vol = (vol / vmax).astype(np.float32)
    return np.clip(vol, 0.0, 1.0)


def _voxel_spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    spacing = np.linalg.norm(np.asarray(affine, dtype=np.float64)[:3, :3], axis=0)
    spacing = np.where(np.isfinite(spacing) & (spacing > 1e-6), spacing, 1.0)
    return spacing.astype(np.float64)


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
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


def _normalize_heatmap(vol: np.ndarray) -> np.ndarray:
    vmax = float(vol.max())
    if vmax > 0:
        vol = (vol / vmax).astype(np.float32)
    return np.clip(vol, 0.0, 1.0).astype(np.float32)


def _gaussian_scores_to_grid(
    vox_idx: np.ndarray,
    scores: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    ref_shape: tuple[int, int, int],
    ref_affine: np.ndarray,
    *,
    smooth_sigma: float,
    gaussian_sigma_extent: float,
    exact_max_points: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if gaussian_sigma_extent <= 0:
        raise ValueError("gaussian_sigma_extent must be > 0")
    if exact_max_points < 0:
        raise ValueError("exact_max_points must be >= 0")

    scores = np.clip(scores.astype(np.float32), 0.0, None)
    vol = np.zeros(ref_shape, dtype=np.float32)
    positive = scores > 0
    if not np.any(positive):
        return vol, {
            "heatmap_backend": "empty",
            "heatmap_gaussian_sigma_extent": float(gaussian_sigma_extent),
            "heatmap_exact_max_points": int(exact_max_points),
        }

    active_count = int(positive.sum())
    if active_count > exact_max_points:
        # Large dense product runs would be too slow with a Python loop over
        # every anisotropic splat.  Use the same max-composition semantics as
        # the exact path, then apply an isotropic Gaussian whose sigma is tied
        # to the stored splat scale in voxel units.
        np.maximum.at(
            vol,
            (vox_idx[positive, 0], vox_idx[positive, 1], vox_idx[positive, 2]),
            scores[positive],
        )
        spacing = _voxel_spacing_from_affine(ref_affine)
        mean_scale_mm = float(np.mean(np.abs(scales[positive]))) if active_count else 1.0
        mean_spacing_mm = float(np.mean(spacing))
        sigma_vox = max(float(smooth_sigma), mean_scale_mm / max(mean_spacing_mm, 1e-6))
        if sigma_vox > 0:
            vol = gaussian_filter(vol, sigma=sigma_vox)
        return _normalize_heatmap(vol), {
            "heatmap_backend": "isotropic_convolution",
            "heatmap_gaussian_sigma_extent": float(gaussian_sigma_extent),
            "heatmap_exact_max_points": int(exact_max_points),
            "heatmap_active_gaussians": active_count,
            "heatmap_effective_sigma_vox": float(sigma_vox),
        }

    spacing = _voxel_spacing_from_affine(ref_affine)
    linear = ref_affine[:3, :3].astype(np.float64)
    origin = ref_affine[:3, 3].astype(np.float64)
    positions = _voxel_to_world(vox_idx.astype(np.float32), ref_affine).astype(np.float64)
    extent_sq = float(gaussian_sigma_extent * gaussian_sigma_extent)

    for center_vox, center_world, score, scale_vec, quat in zip(
        vox_idx[positive],
        positions[positive],
        scores[positive],
        scales[positive],
        quats[positive],
    ):
        scale_vec = np.clip(np.abs(scale_vec).astype(np.float64), 1e-3, None)
        world_radius = float(gaussian_sigma_extent * scale_vec.max())
        radius_vox = np.ceil(world_radius / spacing).astype(np.int64) + 1
        lo = np.maximum(center_vox.astype(np.int64) - radius_vox, 0)
        hi = np.minimum(center_vox.astype(np.int64) + radius_vox + 1, ref_shape)
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
        local = delta @ _quat_wxyz_to_matrix(quat)
        mahal = np.sum((local / scale_vec) ** 2, axis=1)
        keep = mahal <= extent_sq
        if not np.any(keep):
            continue
        kept_ijk = ijk[keep].astype(np.int64)
        contribution = np.exp(-0.5 * mahal[keep]).astype(np.float32) * float(score)
        np.maximum.at(
            vol,
            (kept_ijk[:, 0], kept_ijk[:, 1], kept_ijk[:, 2]),
            contribution,
        )

    if smooth_sigma > 0:
        vol = gaussian_filter(vol, sigma=float(smooth_sigma))
    return _normalize_heatmap(vol), {
        "heatmap_backend": "exact_anisotropic_gaussian",
        "heatmap_gaussian_sigma_extent": float(gaussian_sigma_extent),
        "heatmap_exact_max_points": int(exact_max_points),
        "heatmap_active_gaussians": active_count,
        "heatmap_effective_sigma_vox": float(smooth_sigma),
    }


def _roi_metrics(scores: np.ndarray, roi_at_support: np.ndarray) -> dict[str, object]:
    eps = 1e-8
    in_roi = roi_at_support > 0
    n_in = int(in_roi.sum())
    n_out = int((~in_roi).sum())
    valid = n_in > 0 and n_out > 0
    roi_mean = float(scores[in_roi].mean()) if n_in else 0.0
    non_roi_mean = float(scores[~in_roi].mean()) if n_out else 0.0
    total = float(scores.sum())
    return {
        "roi_gaussian_count": n_in,
        "non_roi_gaussian_count": n_out,
        "roi_metrics_valid": valid,
        "roi_residual_mean": roi_mean,
        "non_roi_residual_mean": non_roi_mean,
        "roi_to_non_roi_ratio": roi_mean / (non_roi_mean + eps) if valid else 1.0,
        "roi_residual_concentration": float(scores[in_roi].sum() / (total + eps)) if n_in else 0.0,
    }


def build_paired_gaussian_residual(
    *,
    patient_volume: Path,
    normal_volume: Path,
    output_dir: Path,
    roi: Optional[Path] = None,
    max_points: int = 50_000,
    percentile_threshold: float = 5.0,
    surface_percentile_threshold: float = 70.0,
    sampling_mode: str = "surface",
    support_basis: str = "union",
    rng_seed: int = 42,
    heatmap_percentile: float = 99.0,
    smooth_sigma: float = 1.0,
    heatmap_method: str = "gaussian",
    heatmap_gaussian_sigma_extent: float = 2.5,
    heatmap_gaussian_exact_max_points: int = 50_000,
    subject: str = "",
    group: str = "",
) -> dict[str, object]:
    """Write paired patient/normal/residual Gaussian splats and metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_data, patient_affine, patient_header = _load_canonical(Path(patient_volume))
    normal_data, normal_affine, _ = _load_canonical(Path(normal_volume))
    ref_shape = patient_data.shape[:3]
    normal_data, normal_resampled = _maybe_resample_to_reference(
        normal_data, normal_affine, ref_shape, patient_affine
    )

    patient_norm = _normalize_for_pairing(patient_data)
    normal_norm = _normalize_for_pairing(normal_data)
    support_signal, support_basis_label = _support_signal_for_basis(
        patient_norm,
        normal_norm,
        support_basis,
    )

    vox_idx, support_meta = _sample_support(
        support_signal,
        support_basis_label=support_basis_label,
        max_points=max_points,
        percentile_threshold=percentile_threshold,
        surface_percentile_threshold=surface_percentile_threshold,
        sampling_mode=sampling_mode,
        rng_seed=rng_seed,
    )
    patient_i = patient_norm[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]].astype(np.float32)
    normal_i = normal_norm[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]].astype(np.float32)
    residual_raw = np.abs(patient_i - normal_i).astype(np.float32)
    percentile_value = float(np.percentile(residual_raw, heatmap_percentile))
    max_value = float(residual_raw.max())
    normalization_value = percentile_value if percentile_value > 1e-8 else max_value
    if normalization_value > 1e-8:
        residual_norm = np.clip(residual_raw / normalization_value, 0.0, 1.0).astype(np.float32)
    else:
        residual_norm = np.zeros_like(residual_raw, dtype=np.float32)

    positions = _voxel_to_world(vox_idx.astype(np.float32), patient_affine)
    scales, quats, orientation_meta = oriented_gaussian_parameters(
        support_signal,
        vox_idx.astype(np.float32),
        patient_affine,
        sampling_mode=sampling_mode,
        support_voxel_count=int(support_meta["foreground_voxels_total"]),
        sampled_point_count=int(support_meta["n_gaussians"]),
    )

    patient_ply = output_dir / "paired_patient_splat.ply"
    normal_ply = output_dir / "paired_normal_splat.ply"
    residual_ply = output_dir / "paired_gaussian_residual_splat.ply"
    _write_ply(patient_ply, positions, patient_i, scales, quats,
               heatmap=False, sampling_mode=sampling_mode)
    _write_ply(normal_ply, positions, normal_i, scales, quats,
               heatmap=False, sampling_mode=sampling_mode)
    _write_ply(residual_ply, positions, residual_norm, scales, quats,
               heatmap=True, sampling_mode=sampling_mode)

    if heatmap_method == "scatter":
        heatmap = _scatter_scores(vox_idx, residual_norm, ref_shape, smooth_sigma=smooth_sigma)
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
    heatmap_path = output_dir / "paired_gaussian_residual_heatmap.nii.gz"
    heatmap_img = nib.Nifti1Image(heatmap, patient_affine, patient_header)
    heatmap_img.header.set_data_dtype(np.float32)
    nib.save(heatmap_img, str(heatmap_path))

    scores_path = output_dir / "paired_gaussian_residual_scores.npy"
    np.save(str(scores_path), residual_raw)

    roi_at_support_state = np.zeros(len(vox_idx), dtype=np.uint8)
    roi_metrics: dict[str, object] = {}
    if roi is not None and Path(roi).is_file():
        roi_data, roi_affine, _ = _load_canonical(Path(roi))
        roi_data, roi_resampled = _maybe_resample_to_reference(
            (roi_data > 0).astype(np.float32),
            roi_affine,
            ref_shape,
            patient_affine,
            nearest=True,
        )
        roi_at_support = roi_data[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]] > 0
        roi_at_support_state = roi_at_support.astype(np.uint8)
        roi_metrics.update(_roi_metrics(residual_norm, roi_at_support))
        roi_metrics["roi_path"] = str(roi)
        roi_metrics["roi_resampled_to_patient_grid"] = roi_resampled

    state_path = output_dir / "paired_gaussian_state.npz"
    np.savez_compressed(
        state_path,
        state_schema=np.array(_PAIRED_GAUSSIAN_STATE_SCHEMA),
        method=np.array("paired_correspondence"),
        patient_volume=np.array(str(patient_volume)),
        normal_volume=np.array(str(normal_volume)),
        subject=np.array(subject),
        group=np.array(group),
        sampling_mode=np.array(sampling_mode),
        support_basis_mode=np.array(support_basis),
        support_basis=np.array(str(support_meta["support_basis"])),
        positions_world=positions.astype(np.float32),
        positions_voxel=vox_idx.astype(np.int32),
        scales=scales.astype(np.float32),
        rotations_quat=quats.astype(np.float32),
        patient_intensity=patient_i.astype(np.float32),
        normal_intensity=normal_i.astype(np.float32),
        residual_raw=residual_raw.astype(np.float32),
        residual_normalized=residual_norm.astype(np.float32),
        roi_at_support=roi_at_support_state,
        affine=patient_affine.astype(np.float64),
        source_shape=np.array(ref_shape, dtype=np.int64),
        normal_resampled_to_patient_grid=np.array(bool(normal_resampled)),
        heatmap_percentile=np.array(float(heatmap_percentile), dtype=np.float32),
        residual_normalization_value=np.array(
            float(normalization_value),
            dtype=np.float32,
        ),
        rng_seed=np.array(int(rng_seed), dtype=np.int64),
    )

    metrics: dict[str, object] = {
        "method": "paired_correspondence",
        "gaussian_state_schema": _PAIRED_GAUSSIAN_STATE_SCHEMA,
        "same_gaussian_support": True,
        "patient_volume": str(patient_volume),
        "normal_volume": str(normal_volume),
        "subject": subject,
        "group": group,
        "normal_resampled_to_patient_grid": normal_resampled,
        "reference_shape": list(ref_shape),
        "support_basis_mode": support_basis,
        "heatmap_percentile": heatmap_percentile,
        **heatmap_meta,
        "residual_normalization_percentile_value": percentile_value,
        "residual_normalization_value": normalization_value,
        "residual_nonzero_count": int((residual_raw > 1e-8).sum()),
        "residual_nonzero_fraction": float((residual_raw > 1e-8).mean()),
        "smooth_sigma": smooth_sigma,
        "residual_min": float(residual_raw.min()),
        "residual_max": float(residual_raw.max()),
        "residual_mean": float(residual_raw.mean()),
        "residual_p95": float(np.percentile(residual_raw, 95)),
        "residual_p99": float(np.percentile(residual_raw, 99)),
        "residual_norm_mean": float(residual_norm.mean()),
        "residual_norm_p99": float(np.percentile(residual_norm, 99)),
        "world_bounds_min": positions.min(axis=0).tolist(),
        "world_bounds_max": positions.max(axis=0).tolist(),
        "artifacts": {
            "paired_patient_splat": patient_ply.name,
            "paired_normal_splat": normal_ply.name,
            "paired_gaussian_residual_splat": residual_ply.name,
            "paired_gaussian_residual_heatmap": heatmap_path.name,
            "paired_gaussian_residual_scores": scores_path.name,
            "paired_gaussian_state": state_path.name,
        },
        **support_meta,
        **orientation_meta,
        **roi_metrics,
    }

    metrics_path = output_dir / "paired_gaussian_residual_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build paired-support Gaussian splat residuals.")
    p.add_argument("--patient-volume", type=Path, required=True)
    p.add_argument("--normal-volume", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--roi", type=Path, default=None)
    p.add_argument("--max-points", type=int, default=50_000)
    p.add_argument("--percentile-threshold", type=float, default=5.0)
    p.add_argument("--surface-percentile-threshold", type=float, default=70.0)
    p.add_argument("--sampling-mode", choices=["surface", "volume", "uniform"], default="surface")
    p.add_argument(
        "--support-basis",
        choices=_SUPPORT_BASIS_CHOICES,
        default="union",
        help=(
            "Signal used to sample the shared Gaussian support before patient/"
            "normal intensities are compared. union preserves the historical "
            "max(patient, synthetic_normal) behavior."
        ),
    )
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--heatmap-percentile", type=float, default=99.0)
    p.add_argument("--smooth-sigma", type=float, default=1.0)
    p.add_argument("--heatmap-method", choices=["scatter", "gaussian"], default="gaussian")
    p.add_argument("--heatmap-gaussian-sigma-extent", type=float, default=2.5)
    p.add_argument("--heatmap-gaussian-exact-max-points", type=int, default=50_000)
    p.add_argument("--subject", default="")
    p.add_argument("--group", default="")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    metrics = build_paired_gaussian_residual(
        patient_volume=args.patient_volume,
        normal_volume=args.normal_volume,
        output_dir=args.output_dir,
        roi=args.roi,
        max_points=args.max_points,
        percentile_threshold=args.percentile_threshold,
        surface_percentile_threshold=args.surface_percentile_threshold,
        sampling_mode=args.sampling_mode,
        support_basis=args.support_basis,
        rng_seed=args.rng_seed,
        heatmap_percentile=args.heatmap_percentile,
        smooth_sigma=args.smooth_sigma,
        heatmap_method=args.heatmap_method,
        heatmap_gaussian_sigma_extent=args.heatmap_gaussian_sigma_extent,
        heatmap_gaussian_exact_max_points=args.heatmap_gaussian_exact_max_points,
        subject=args.subject,
        group=args.group,
    )
    print(
        f"Paired Gaussian residual complete: "
        f"N={metrics['n_gaussians']} p99={metrics['residual_p99']:.4f}"
    )


if __name__ == "__main__":
    main()
