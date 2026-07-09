"""Direct volumetric Gaussian splat exporter to browser-loadable PLY.

Samples brain voxels from a NIfTI volume and writes a binary PLY with
position, grayscale color, opacity, anisotropic scale, and quaternion fields.

Three sampling modes are supported:
  "surface"  — surface/edge-biased weighted sampling; softer opacity to
               allow accumulation blending for cortical boundary views.
  "volume"   — full-foreground weighted sampling for the presentation PLY;
               intended to render the whole brain volume, not only the shell.
  "uniform"  — legacy uniform foreground sampling.

This is direct anatomy-oriented volumetric sampling — NOT optimized 3DGS/NeRF
training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_closing, binary_erosion, binary_fill_holes, label


_PLY_DTYPE = np.dtype([
    ("x", "<f4"),
    ("y", "<f4"),
    ("z", "<f4"),
    ("red", "u1"),
    ("green", "u1"),
    ("blue", "u1"),
    ("opacity", "<f4"),
    ("scale_x", "<f4"),
    ("scale_y", "<f4"),
    ("scale_z", "<f4"),
    ("rot_w", "<f4"),
    ("rot_x", "<f4"),
    ("rot_y", "<f4"),
    ("rot_z", "<f4"),
])

_PLY_PROPERTIES = (
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "property uchar red\n"
    "property uchar green\n"
    "property uchar blue\n"
    "property float opacity\n"
    "property float scale_x\n"
    "property float scale_y\n"
    "property float scale_z\n"
    "property float rot_w\n"
    "property float rot_x\n"
    "property float rot_y\n"
    "property float rot_z\n"
)

_SURFACE_PERCENTILE_FLOOR = 70.0
_VOLUME_PERCENTILE_FLOOR = 20.0
_VOLUME_INTENSITY_FLOOR = 0.08
_GAUSSIAN_STATE_SCHEMA = "mri_direct_splat_v1"
_COORDINATE_FRAME_SCHEMA = "mri_direct_splat_coordinate_frame_v1"
_LESION_SPLAT_ORANGE_RGB = np.array([255, 149, 0], dtype=np.uint8)


def _load_canonical(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data, img.affine.astype(np.float64)


def _robust_normalize(data: np.ndarray) -> np.ndarray:
    fg = data[data > 0]
    if fg.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lo = float(np.percentile(fg, 1))
    hi = float(np.percentile(fg, 99))
    if hi == lo:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _largest_connected_component(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    labels, n_labels = label(mask)
    if n_labels == 0:
        return mask, 0, 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest_label = int(sizes.argmax())
    largest_size = int(sizes[largest_label])
    return labels == largest_label, n_labels, largest_size


def _foreground_mask_for_sampling(
    norm: np.ndarray,
    *,
    percentile_threshold: float,
    sampling_mode: str,
    surface_percentile_threshold: float = _SURFACE_PERCENTILE_FLOOR,
) -> tuple[np.ndarray, float, float, dict[str, object]]:
    """Return an auditable foreground mask for the requested splat sampler."""
    fg_values = norm[norm > 0]
    if fg_values.size == 0:
        raise ValueError("No foreground voxels found in normalized volume.")

    if sampling_mode == "surface":
        effective = max(percentile_threshold, surface_percentile_threshold)
        mask_model = "percentile_surface_foreground"
    elif sampling_mode == "volume":
        effective = max(percentile_threshold, _VOLUME_PERCENTILE_FLOOR)
        mask_model = "largest_connected_tissue_component"
    else:
        effective = percentile_threshold
        mask_model = "percentile_foreground"

    threshold = float(np.percentile(fg_values, effective))
    threshold_floor = None
    if sampling_mode == "volume":
        threshold_floor = _VOLUME_INTENSITY_FLOOR
        threshold = max(threshold, threshold_floor)
    raw_mask = norm > threshold
    if not raw_mask.any():
        raw_mask = norm >= threshold
    if not raw_mask.any():
        raise ValueError("Foreground threshold removed all sampling voxels.")

    meta: dict[str, object] = {
        "foreground_threshold_value": threshold,
        "foreground_mask_model": mask_model,
        "raw_foreground_voxels_total": int(raw_mask.sum()),
    }
    if threshold_floor is not None:
        meta["foreground_threshold_intensity_floor"] = threshold_floor

    if sampling_mode != "volume":
        return raw_mask, effective, threshold, meta

    largest, n_components, largest_size = _largest_connected_component(raw_mask)
    cleaned = largest
    try:
        closed = binary_closing(cleaned, iterations=1)
        if closed.any():
            cleaned = closed
        filled = binary_fill_holes(cleaned)
        if filled.any():
            cleaned = filled
    except Exception:
        cleaned = largest

    if not cleaned.any():
        cleaned = raw_mask

    meta.update(
        {
            "volume_percentile_threshold_floor": _VOLUME_PERCENTILE_FLOOR,
            "volume_intensity_threshold_floor": _VOLUME_INTENSITY_FLOOR,
            "foreground_components_total": int(n_components),
            "largest_component_voxels_total": int(largest_size),
            "cleaned_foreground_voxels_total": int(cleaned.sum()),
        }
    )
    return cleaned.astype(bool), effective, threshold, meta


def _voxel_to_world(vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    ones = np.ones((len(vox), 1), dtype=np.float64)
    hom = np.hstack([vox.astype(np.float64), ones])
    return ((affine @ hom.T).T[:, :3]).astype(np.float32)


def _voxel_spacing(affine: np.ndarray) -> list[float]:
    return [float(v) for v in np.sqrt((affine[:3, :3] ** 2).sum(axis=0))]


def _coordinate_frame_contract(
    *,
    vox: np.ndarray,
    positions_world: np.ndarray,
    affine: np.ndarray,
    source_shape: tuple[int, int, int],
) -> dict[str, object]:
    """Describe and self-check the exported splat coordinate frame."""
    vox64 = vox.astype(np.float64, copy=False)
    expected_world = _voxel_to_world(vox64, affine).astype(np.float64)
    observed_world = positions_world.astype(np.float64, copy=False)
    world_error = np.abs(observed_world - expected_world)
    in_bounds = (
        (vox64[:, 0] >= 0) & (vox64[:, 0] < source_shape[0]) &
        (vox64[:, 1] >= 0) & (vox64[:, 1] < source_shape[1]) &
        (vox64[:, 2] >= 0) & (vox64[:, 2] < source_shape[2])
    )
    return {
        "schema": _COORDINATE_FRAME_SCHEMA,
        "coordinate_frame": "canonical_ras_world_mm",
        "voxel_index_frame": "canonical_ijk_zero_based",
        "source_shape": [int(v) for v in source_shape],
        "source_orientation": [str(v) for v in nib.aff2axcodes(affine)],
        "source_voxel_spacing_mm": _voxel_spacing(affine),
        "voxel_to_world_max_abs_error_mm": (
            float(world_error.max()) if world_error.size else 0.0
        ),
        "voxel_positions_in_bounds_fraction": (
            float(in_bounds.mean()) if in_bounds.size else 0.0
        ),
        "voxel_bounds_min": vox64.min(axis=0).tolist() if len(vox64) else None,
        "voxel_bounds_max": vox64.max(axis=0).tolist() if len(vox64) else None,
        "world_bounds_min": (
            observed_world.min(axis=0).tolist() if len(observed_world) else None
        ),
        "world_bounds_max": (
            observed_world.max(axis=0).tolist() if len(observed_world) else None
        ),
    }


def _build_surface_weights(
    norm: np.ndarray,
    fg_mask: np.ndarray,
    surface_weight: float = 8.0,
    interior_weight: float = 1.0,
) -> np.ndarray:
    """Compute per-voxel sampling weights biased toward the foreground surface.

    Erodes the foreground mask once to find the surface shell (voxels that are
    foreground but not in the fully-eroded interior).  Surface voxels get a
    higher base weight multiplied by local gradient magnitude.
    """
    try:
        interior = binary_erosion(fg_mask, iterations=2)
    except Exception:
        interior = fg_mask.copy()
    surface_shell = fg_mask & ~interior

    # Gradient magnitude as a continuous measure of edge strength
    grad_sq = np.zeros(norm.shape, dtype=np.float32)
    for axis in range(norm.ndim):
        g = np.gradient(norm, axis=axis)
        grad_sq += g * g
    grad_mag = np.sqrt(grad_sq)
    # Normalise gradient within foreground to [0, 1]
    fg_grad = grad_mag[fg_mask]
    gmax = float(fg_grad.max()) if fg_grad.size > 0 else 1.0
    if gmax == 0.0:
        gmax = 1.0
    grad_norm = grad_mag / gmax

    # Combine: surface shell → high weight; interior → lower weight
    base = np.where(surface_shell, surface_weight, interior_weight).astype(np.float32)
    weights = base * (1.0 + grad_norm)  # gradient bonus on top

    fg_weights = weights[fg_mask]
    total = float(fg_weights.sum())
    if total == 0.0:
        fg_weights = np.ones(int(fg_mask.sum()), dtype=np.float32)
        total = float(fg_weights.sum())
    return fg_weights / total, surface_shell


def _build_volume_weights(norm: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """Compute per-voxel sampling weights that preserve full brain volume.

    Unlike surface mode, this intentionally keeps a high nonzero base weight for
    every foreground voxel. Gradient and intensity only nudge the distribution so
    gray/white boundaries are represented without hollowing the anatomy.
    """
    gradients = np.gradient(norm.astype(np.float32), edge_order=1)
    grad_sq = np.zeros(norm.shape, dtype=np.float32)
    for g in gradients:
        grad_sq += g * g
    grad_mag = np.sqrt(grad_sq)

    fg_grad = grad_mag[fg_mask]
    gmax = float(fg_grad.max()) if fg_grad.size > 0 else 1.0
    if gmax <= 0.0:
        gmax = 1.0
    grad_norm = grad_mag[fg_mask] / gmax
    intensities = norm[fg_mask]

    weights = (0.65 + 0.25 * intensities + 0.10 * grad_norm).astype(np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        weights = np.ones(int(fg_mask.sum()), dtype=np.float64)
        total = float(weights.sum())
    return weights / total


def _volume_opacity_from_intensity(intensities: np.ndarray) -> np.ndarray:
    """Opacity curve for dense MRI volume splats.

    Low-signal MRI background often survives permissive tissue masks. Keep it
    almost transparent while allowing brighter gray/white matter to accumulate
    into a continuous brain volume.
    """
    t = np.clip(intensities.astype(np.float32), 0.0, 1.0)
    opacity = 0.01 + 0.42 * np.power(t, 1.6)
    return np.clip(opacity, 0.005, 0.55).astype(np.float32)


def _stratified_weighted_sample_indices(
    fg_idx: np.ndarray,
    weights: np.ndarray,
    *,
    n: int,
    rng: np.random.Generator,
    max_cells_per_axis: int = 16,
) -> tuple[np.ndarray, dict[str, object]]:
    """Sample foreground indices with an anatomical coverage first pass.

    The direct dense splat is meant to read as a whole MRI volume. Pure weighted
    random sampling can spend too much of a small point budget in one bright or
    high-gradient region, leaving holes elsewhere. This sampler first chooses
    one representative from as many occupied foreground grid cells as the budget
    allows, then spends any remaining budget according to the original weights.
    """
    fg_idx = np.asarray(fg_idx, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    n_fg = int(len(fg_idx))
    n = min(max(0, int(n)), n_fg)
    if n_fg == 0 or n == 0:
        return np.zeros(0, dtype=np.int64), {
            "volume_sampling_strategy": "foreground_grid_stratified_weighted",
            "stratified_cells_per_axis": [0, 0, 0],
            "stratified_occupied_cells_total": 0,
            "stratified_cells_sampled": 0,
        }

    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(n_fg, dtype=np.float64) / float(n_fg)
    else:
        weights = weights / float(weights.sum())

    mins = fg_idx.min(axis=0)
    maxs = fg_idx.max(axis=0)
    extent = np.maximum(maxs - mins + 1, 1)
    cube_root_budget = max(1, int(round(n ** (1.0 / 3.0))))
    base_cells = max(1, min(int(max_cells_per_axis), cube_root_budget))
    longest = float(extent.max())
    cells_per_axis = np.maximum(
        1,
        np.round(base_cells * (extent.astype(np.float64) / longest)).astype(np.int64),
    )
    cells_per_axis = np.minimum(cells_per_axis, int(max_cells_per_axis))

    rel = (fg_idx - mins).astype(np.float64) / extent.astype(np.float64)
    cell_coords = np.floor(rel * cells_per_axis.astype(np.float64)).astype(np.int64)
    cell_coords = np.minimum(cell_coords, cells_per_axis - 1)
    cell_ids = (
        cell_coords[:, 0] * cells_per_axis[1] * cells_per_axis[2]
        + cell_coords[:, 1] * cells_per_axis[2]
        + cell_coords[:, 2]
    )

    order = np.argsort(cell_ids)
    sorted_cells = cell_ids[order]
    unique_cells, starts = np.unique(sorted_cells, return_index=True)
    stops = np.r_[starts[1:], len(order)]
    cell_weight_sums = np.add.reduceat(weights[order], starts)
    occupied_total = int(len(unique_cells))

    if occupied_total > n:
        chosen_cells = np.argpartition(cell_weight_sums, -n)[-n:]
        chosen_cells = chosen_cells[np.argsort(-cell_weight_sums[chosen_cells])]
    else:
        chosen_cells = np.arange(occupied_total)

    selected: list[int] = []
    selected_mask = np.zeros(n_fg, dtype=bool)
    for cell_pos in chosen_cells:
        member_idx = order[starts[cell_pos]:stops[cell_pos]]
        member_weights = weights[member_idx]
        total = float(member_weights.sum())
        if total > 0.0:
            local = rng.choice(len(member_idx), p=member_weights / total)
        else:
            local = rng.integers(0, len(member_idx))
        picked = int(member_idx[int(local)])
        selected.append(picked)
        selected_mask[picked] = True

    remaining = n - len(selected)
    if remaining > 0:
        available = np.flatnonzero(~selected_mask)
        available_weights = weights[available]
        total = float(available_weights.sum())
        if total > 0.0:
            extra = rng.choice(
                available,
                size=remaining,
                replace=False,
                p=available_weights / total,
            )
        else:
            extra = rng.choice(available, size=remaining, replace=False)
        selected.extend(int(v) for v in np.atleast_1d(extra))
        selected_mask[np.asarray(extra, dtype=np.int64)] = True

    selected_arr = np.asarray(selected[:n], dtype=np.int64)
    sampled_cells = int(np.unique(cell_ids[selected_arr]).size)
    return selected_arr, {
        "volume_sampling_strategy": "foreground_grid_stratified_weighted",
        "stratified_cells_per_axis": cells_per_axis.astype(int).tolist(),
        "stratified_occupied_cells_total": occupied_total,
        "stratified_cells_sampled": sampled_cells,
    }


def _safe_normalize_vectors(vecs: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    out = np.empty_like(vecs, dtype=np.float32)
    valid = norms[:, 0] > 1e-8
    out[valid] = vecs[valid] / norms[valid]
    out[~valid] = fallback.astype(np.float32)
    return out.astype(np.float32)


def _quaternions_from_rotation_matrices(rot: np.ndarray) -> np.ndarray:
    """Convert local-to-world rotation matrices to normalized wxyz quaternions."""
    rot = rot.astype(np.float32, copy=False)
    n = rot.shape[0]
    q = np.empty((n, 4), dtype=np.float32)

    m00 = rot[:, 0, 0]
    m01 = rot[:, 0, 1]
    m02 = rot[:, 0, 2]
    m10 = rot[:, 1, 0]
    m11 = rot[:, 1, 1]
    m12 = rot[:, 1, 2]
    m20 = rot[:, 2, 0]
    m21 = rot[:, 2, 1]
    m22 = rot[:, 2, 2]
    trace = m00 + m11 + m22

    mask = trace > 0.0
    s = np.sqrt(np.maximum(trace[mask] + 1.0, 1e-8)) * 2.0
    q[mask, 0] = 0.25 * s
    q[mask, 1] = (m21[mask] - m12[mask]) / s
    q[mask, 2] = (m02[mask] - m20[mask]) / s
    q[mask, 3] = (m10[mask] - m01[mask]) / s

    rem = ~mask
    mask_x = rem & (m00 > m11) & (m00 > m22)
    s = np.sqrt(np.maximum(1.0 + m00[mask_x] - m11[mask_x] - m22[mask_x], 1e-8)) * 2.0
    q[mask_x, 0] = (m21[mask_x] - m12[mask_x]) / s
    q[mask_x, 1] = 0.25 * s
    q[mask_x, 2] = (m01[mask_x] + m10[mask_x]) / s
    q[mask_x, 3] = (m02[mask_x] + m20[mask_x]) / s

    mask_y = rem & ~mask_x & (m11 > m22)
    s = np.sqrt(np.maximum(1.0 + m11[mask_y] - m00[mask_y] - m22[mask_y], 1e-8)) * 2.0
    q[mask_y, 0] = (m02[mask_y] - m20[mask_y]) / s
    q[mask_y, 1] = (m01[mask_y] + m10[mask_y]) / s
    q[mask_y, 2] = 0.25 * s
    q[mask_y, 3] = (m12[mask_y] + m21[mask_y]) / s

    mask_z = rem & ~mask_x & ~mask_y
    s = np.sqrt(np.maximum(1.0 + m22[mask_z] - m00[mask_z] - m11[mask_z], 1e-8)) * 2.0
    q[mask_z, 0] = (m10[mask_z] - m01[mask_z]) / s
    q[mask_z, 1] = (m02[mask_z] + m20[mask_z]) / s
    q[mask_z, 2] = (m12[mask_z] + m21[mask_z]) / s
    q[mask_z, 3] = 0.25 * s

    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    q_norm[q_norm < 1e-8] = 1.0
    q = q / q_norm
    q[q[:, 0] < 0] *= -1.0
    return q.astype(np.float32)


def _gradient_tangent_frames(
    norm: np.ndarray,
    vox: np.ndarray,
    affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local-to-world tangent frames and sampled gradient magnitudes.

    The local z-axis follows the world-space image gradient, while local x/y
    span the tissue tangent plane.  Low-gradient voxels fall back to identity.
    """
    gradients = np.gradient(norm.astype(np.float32), edge_order=1)
    ijk = np.round(vox).astype(np.int64)
    for axis, size in enumerate(norm.shape[:3]):
        np.clip(ijk[:, axis], 0, size - 1, out=ijk[:, axis])
    grad_vox = np.stack(
        [g[ijk[:, 0], ijk[:, 1], ijk[:, 2]] for g in gradients],
        axis=1,
    ).astype(np.float32)

    inv_linear = np.linalg.inv(affine[:3, :3]).astype(np.float32)
    grad_world = grad_vox @ inv_linear
    grad_mag = np.linalg.norm(grad_world, axis=1).astype(np.float32)

    normal = _safe_normalize_vectors(grad_world, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    low_grad = grad_mag < 1e-6
    if low_grad.any():
        normal[low_grad] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    ref = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(normal), 1))
    near_z = np.abs(normal[:, 2]) > 0.9
    ref[near_z] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    tangent_x = _safe_normalize_vectors(np.cross(ref, normal), np.array([1.0, 0.0, 0.0], dtype=np.float32))
    tangent_y = _safe_normalize_vectors(np.cross(normal, tangent_x), np.array([0.0, 1.0, 0.0], dtype=np.float32))

    frames = np.stack([tangent_x, tangent_y, normal], axis=2).astype(np.float32)
    if low_grad.any():
        frames[low_grad] = np.eye(3, dtype=np.float32)
    return frames, grad_mag


def oriented_gaussian_parameters(
    norm: np.ndarray,
    vox: np.ndarray,
    affine: np.ndarray,
    *,
    sampling_mode: str = "surface",
    support_voxel_count: Optional[int] = None,
    sampled_point_count: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build anatomically oriented Gaussian scales and quaternions for MRI splats."""
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).astype(np.float32)
    base_spacing = float(np.mean(spacing))
    density_scale_factor: Optional[float] = None
    density_voxels_per_splat: Optional[float] = None
    if sampling_mode == "surface":
        scale_vec = np.array(
            [
                max(base_spacing * 2.20, 0.75),
                max(base_spacing * 2.20, 0.75),
                max(float(spacing.min()) * 0.65, 0.35),
            ],
            dtype=np.float32,
        )
        scale_model = "gradient_tangent_disk"
    elif sampling_mode == "volume":
        density_scale_factor = 1.35
        if support_voxel_count is not None and sampled_point_count:
            density_voxels_per_splat = float(support_voxel_count) / max(
                float(sampled_point_count),
                1.0,
            )
            cell_width_vox = max(density_voxels_per_splat, 1.0) ** (1.0 / 3.0)
            density_scale_factor = float(np.clip(0.18 * cell_width_vox, 1.35, 2.25))
        scale_vec = np.array(
            [
                max(float(spacing[0]) * density_scale_factor, 0.45),
                max(float(spacing[1]) * density_scale_factor, 0.45),
                max(float(spacing[2]) * density_scale_factor, 0.45),
            ],
            dtype=np.float32,
        )
        scale_model = "coverage_balanced_density_adaptive_volume_kernel"
    else:
        scale_vec = np.array(
            [
                max(float(spacing[0]), 0.35),
                max(float(spacing[1]), 0.35),
                max(float(spacing[2]), 0.35),
            ],
            dtype=np.float32,
        )
        scale_model = "gradient_oriented_voxel"

    frames, grad_mag = _gradient_tangent_frames(norm, vox, affine)
    quats = _quaternions_from_rotation_matrices(frames)
    scales = np.tile(scale_vec, (len(vox), 1)).astype(np.float32)
    meta = {
        "gaussian_orientation": "local_gradient_tangent_frame",
        "gaussian_scale_model": scale_model,
        "gaussian_scale": scale_vec.tolist(),
        "gradient_magnitude_mean": float(grad_mag.mean()) if len(grad_mag) else 0.0,
        "gradient_magnitude_p95": float(np.percentile(grad_mag, 95)) if len(grad_mag) else 0.0,
    }
    if density_scale_factor is not None:
        meta["density_scale_factor"] = density_scale_factor
    if density_voxels_per_splat is not None:
        meta["density_voxels_per_splat"] = density_voxels_per_splat
    return scales, quats, meta


def _cap_browser_scale_anisotropy(
    scales: np.ndarray,
    *,
    max_ratio_to_median: float = 1.6,
) -> np.ndarray:
    """Limit visual splat streaking from thick-slice MRI acquisitions.

    Positions stay in the original millimeter coordinate frame. Only the
    drawn Gaussian ellipsoid size is capped, so a 5 mm slice direction does
    not become a long browser-visible spike.
    """
    if len(scales) == 0:
        return scales.astype(np.float32)
    safe = np.maximum(scales.astype(np.float32), 1e-6)
    median = np.median(safe, axis=1, keepdims=True).astype(np.float32)
    capped = np.minimum(safe, median * float(max_ratio_to_median))
    return capped.astype(np.float32)


def export_splat_ply(
    volume_path: Path,
    output_ply: Path,
    max_points: int = 50_000,
    percentile_threshold: float = 5.0,
    surface_percentile_threshold: float = _SURFACE_PERCENTILE_FLOOR,
    rng_seed: int = 42,
    subject: str = "",
    group: str = "",
    sampling_mode: str = "surface",
) -> dict:
    """Export a NIfTI volume as a binary PLY point cloud for Three.js rendering.

    Parameters
    ----------
    volume_path:
        Input NIfTI (.nii.gz).
    output_ply:
        Output PLY path.  A JSON metadata sidecar is written alongside it.
    max_points:
        Maximum number of points.  Deterministic with rng_seed.
    percentile_threshold:
        Foreground threshold: voxels below this percentile of non-zero values
        are treated as background and excluded.  Legacy default is 5.0.
    surface_percentile_threshold:
        Minimum effective threshold enforced in surface mode (default 70.0).
        Low MRI background/noise fills the acquisition FOV; surface mode applies
        max(percentile_threshold, surface_percentile_threshold) so that at most
        the top 30 % of signal is treated as foreground by default.  Ignored in
        uniform mode, where percentile_threshold is used unchanged.
    rng_seed:
        RNG seed for deterministic sampling.
    subject, group:
        Recorded in metadata only.
    sampling_mode:
        "surface" (default) — surface/edge-biased weighted sampling.
        "uniform" — legacy uniform foreground sampling.

    Returns
    -------
    metadata dict (also written as <stem>_meta.json beside the PLY).
    """
    if sampling_mode not in ("surface", "volume", "uniform"):
        raise ValueError(
            "sampling_mode must be 'surface', 'volume', or 'uniform', "
            f"got {sampling_mode!r}"
        )

    volume_path = Path(volume_path)
    output_ply = Path(output_ply)
    output_ply.parent.mkdir(parents=True, exist_ok=True)

    data, affine = _load_canonical(volume_path)
    norm = _robust_normalize(data)

    try:
        fg_mask, effective_percentile, _threshold, foreground_meta = _foreground_mask_for_sampling(
            norm,
            percentile_threshold=percentile_threshold,
            sampling_mode=sampling_mode,
            surface_percentile_threshold=surface_percentile_threshold,
        )
    except ValueError as exc:
        raise ValueError(f"No foreground voxels in {volume_path}") from exc

    fg_idx = np.argwhere(fg_mask)
    n_fg = len(fg_idx)

    rng = np.random.default_rng(rng_seed)
    n = min(max_points, n_fg)

    surface_voxels_total: Optional[int] = None
    edge_voxels_total: Optional[int] = None

    if sampling_mode == "surface":
        weights, surface_shell = _build_surface_weights(norm, fg_mask)
        surface_voxels_total = int(surface_shell.sum())
        edge_voxels_total = surface_voxels_total
        if n < n_fg:
            sel = rng.choice(n_fg, size=n, replace=False, p=weights)
        else:
            sel = np.arange(n_fg)
        sampling_meta: dict[str, object] = {}
    elif sampling_mode == "volume":
        weights = _build_volume_weights(norm, fg_mask)
        sel, sampling_meta = _stratified_weighted_sample_indices(
            fg_idx,
            weights,
            n=n,
            rng=rng,
        )
    else:
        # uniform — preserve legacy behaviour exactly
        sel = rng.choice(n_fg, size=n, replace=False)
        sampling_meta = {}

    intensities_all = norm[fg_mask]
    vox = fg_idx[sel].astype(np.float32)
    intensities = intensities_all[sel]

    positions = _voxel_to_world(vox, affine)
    coordinate_contract = _coordinate_frame_contract(
        vox=vox,
        positions_world=positions,
        affine=affine,
        source_shape=tuple(int(v) for v in data.shape[:3]),
    )
    gray = np.clip(intensities * 255, 0, 255).astype(np.uint8)
    scales, quats, orientation_meta = oriented_gaussian_parameters(
        norm,
        vox,
        affine,
        sampling_mode=sampling_mode,
        support_voxel_count=n_fg,
        sampled_point_count=n,
    )

    if sampling_mode == "surface":
        opacity_vals = (0.15 + intensities * 0.50).astype(np.float32)
        opacity_vals = np.clip(opacity_vals, 0.0, 1.0)
    elif sampling_mode == "volume":
        opacity_vals = _volume_opacity_from_intensity(intensities)
    else:
        opacity_vals = intensities

    arr = np.zeros(n, dtype=_PLY_DTYPE)
    arr["x"] = positions[:, 0]
    arr["y"] = positions[:, 1]
    arr["z"] = positions[:, 2]
    arr["red"] = gray
    arr["green"] = gray
    arr["blue"] = gray
    arr["opacity"] = opacity_vals
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
    with open(output_ply, "wb") as fh:
        fh.write(header)
        fh.write(arr.tobytes())

    state_path = output_ply.parent / (output_ply.stem + "_state.npz")
    np.savez_compressed(
        state_path,
        state_schema=np.array(_GAUSSIAN_STATE_SCHEMA),
        coordinate_frame_schema=np.array(_COORDINATE_FRAME_SCHEMA),
        source_volume=np.array(str(volume_path)),
        subject=np.array(subject),
        group=np.array(group),
        sampling_mode=np.array(sampling_mode),
        positions_world=positions.astype(np.float32),
        positions_voxel=vox.astype(np.float32),
        scales=scales.astype(np.float32),
        rotations_quat=quats.astype(np.float32),
        opacity=opacity_vals.astype(np.float32),
        intensity=intensities.astype(np.float32),
        color_rgb=np.stack([gray, gray, gray], axis=-1).astype(np.uint8),
        affine=affine.astype(np.float64),
        source_shape=np.array(data.shape[:3], dtype=np.int64),
        source_orientation=np.array(coordinate_contract["source_orientation"]),
        source_voxel_spacing_mm=np.array(
            coordinate_contract["source_voxel_spacing_mm"],
            dtype=np.float64,
        ),
        foreground_voxels_total=np.array(int(n_fg), dtype=np.int64),
        percentile_threshold=np.array(float(percentile_threshold), dtype=np.float32),
        effective_percentile_threshold=np.array(float(effective_percentile), dtype=np.float32),
        rng_seed=np.array(int(rng_seed), dtype=np.int64),
    )

    meta_path = output_ply.parent / (output_ply.stem + "_meta.json")
    meta: dict = {
        "subject": subject,
        "group": group,
        "source_volume": str(volume_path),
        "n_points": n,
        "max_points": max_points,
        "rng_seed": rng_seed,
        "sampling_mode": sampling_mode,
        "percentile_threshold": percentile_threshold,
        "effective_percentile_threshold": effective_percentile,
        "world_bounds_min": positions.min(axis=0).tolist(),
        "world_bounds_max": positions.max(axis=0).tolist(),
        "intensity_range": [float(intensities.min()), float(intensities.max())],
        "foreground_voxels_total": int(n_fg),
        "ply_format": "binary_little_endian",
        "ply_properties": list(_PLY_DTYPE.names),
        "gaussian_state_npz": state_path.name,
        "gaussian_state_schema": _GAUSSIAN_STATE_SCHEMA,
        "coordinate_frame": coordinate_contract,
        "note": (
            "dense surface-biased direct MRI splat; not optimized neural 3DGS."
            if sampling_mode == "surface"
            else (
                "full-volume anatomy-oriented direct MRI splat; not optimized neural 3DGS."
                if sampling_mode == "volume"
                else "Direct volumetric Gaussian splat — uniform foreground sampling. "
                     "NOT optimized 3DGS/NeRF training."
            )
        ),
        **foreground_meta,
        **sampling_meta,
        **orientation_meta,
    }
    if sampling_mode == "volume":
        meta["volume_voxels_total"] = int(n_fg)
        meta["sampling_weight_model"] = "full_foreground_base_plus_intensity_gradient"
    if surface_voxels_total is not None:
        meta["surface_voxels_total"] = surface_voxels_total
    if edge_voxels_total is not None:
        meta["edge_voxels_total"] = edge_voxels_total

    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def export_lesion_highlighted_splat_ply(
    volume_path: Path,
    roi_path: Path,
    output_ply: Path,
    max_anatomy_points: int = 220_000,
    max_lesion_points: int = 60_000,
    percentile_threshold: float = 5.0,
    rng_seed: int = 42,
    subject: str = "",
    group: str = "",
    sampling_mode: str = "volume",
) -> dict:
    """Export one browser PLY with MRI anatomy plus highlighted FCD ROI splats.

    This is the product-facing MRI 3DGS artifact: the anatomy is sampled from
    the input NIfTI, and the provided lesion mask is rendered as high-opacity
    orange Gaussians in the same canonical RAS millimeter coordinate frame.
    """
    volume_path = Path(volume_path)
    roi_path = Path(roi_path)
    output_ply = Path(output_ply)

    meta = export_splat_ply(
        volume_path=volume_path,
        output_ply=output_ply,
        max_points=max_anatomy_points,
        percentile_threshold=percentile_threshold,
        rng_seed=rng_seed,
        subject=subject,
        group=group,
        sampling_mode=sampling_mode,
    )

    state_path = output_ply.parent / (output_ply.stem + "_state.npz")
    with np.load(state_path) as state:
        anatomy_positions = state["positions_world"].astype(np.float32)
        anatomy_voxels = state["positions_voxel"].astype(np.float32)
        anatomy_scales_raw = state["scales"].astype(np.float32)
        anatomy_quats = state["rotations_quat"].astype(np.float32)
        anatomy_opacity = state["opacity"].astype(np.float32)
        anatomy_intensity = state["intensity"].astype(np.float32)
        anatomy_color = state["color_rgb"].astype(np.uint8)
        affine = state["affine"].astype(np.float64)
    anatomy_scales = _cap_browser_scale_anisotropy(anatomy_scales_raw)

    volume_data, volume_affine = _load_canonical(volume_path)
    roi_data, roi_affine = _load_canonical(roi_path)
    if roi_data.shape[:3] != volume_data.shape[:3]:
        raise ValueError(
            "ROI shape must match volume shape for lesion-highlighted splat export; "
            f"got ROI {roi_data.shape[:3]} and volume {volume_data.shape[:3]}"
        )
    if not np.allclose(roi_affine, volume_affine, atol=1e-4):
        raise ValueError("ROI affine must match volume affine for lesion-highlighted splat export")

    roi_idx = np.argwhere(roi_data > 0)
    if len(roi_idx) == 0:
        lesion_voxels = np.zeros((0, 3), dtype=np.float32)
    else:
        rng = np.random.default_rng(rng_seed + 991)
        n_lesion = min(int(max_lesion_points), len(roi_idx))
        if n_lesion < len(roi_idx):
            sel = rng.choice(len(roi_idx), size=n_lesion, replace=False)
            lesion_voxels = roi_idx[sel].astype(np.float32)
        else:
            lesion_voxels = roi_idx.astype(np.float32)

    norm = _robust_normalize(volume_data)
    anatomy_opacity = np.clip(anatomy_opacity * 0.62, 0.004, 0.28).astype(np.float32)

    if len(lesion_voxels):
        lesion_positions = _voxel_to_world(lesion_voxels, volume_affine)
        lesion_scales, lesion_quats, lesion_orientation_meta = oriented_gaussian_parameters(
            norm,
            lesion_voxels,
            volume_affine,
            sampling_mode="volume",
            support_voxel_count=int((roi_data > 0).sum()),
            sampled_point_count=len(lesion_voxels),
        )
        lesion_scales = (lesion_scales * np.array([3.8, 3.8, 2.6], dtype=np.float32)).astype(np.float32)
        lesion_scales = _cap_browser_scale_anisotropy(lesion_scales)
        ijk = np.round(lesion_voxels).astype(np.int64)
        for axis, size in enumerate(norm.shape[:3]):
            np.clip(ijk[:, axis], 0, size - 1, out=ijk[:, axis])
        lesion_intensity = norm[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(np.float32)
        lesion_opacity = np.full(len(lesion_voxels), 1.0, dtype=np.float32)
        lesion_color = np.tile(_LESION_SPLAT_ORANGE_RGB, (len(lesion_voxels), 1))
    else:
        lesion_positions = np.zeros((0, 3), dtype=np.float32)
        lesion_scales = np.zeros((0, 3), dtype=np.float32)
        lesion_quats = np.zeros((0, 4), dtype=np.float32)
        lesion_intensity = np.zeros(0, dtype=np.float32)
        lesion_opacity = np.zeros(0, dtype=np.float32)
        lesion_color = np.zeros((0, 3), dtype=np.uint8)
        lesion_orientation_meta = {}

    positions = np.vstack([anatomy_positions, lesion_positions]).astype(np.float32)
    voxels = np.vstack([anatomy_voxels, lesion_voxels]).astype(np.float32)
    scales = np.vstack([anatomy_scales, lesion_scales]).astype(np.float32)
    quats = np.vstack([anatomy_quats, lesion_quats]).astype(np.float32)
    opacity = np.concatenate([anatomy_opacity, lesion_opacity]).astype(np.float32)
    intensity = np.concatenate([anatomy_intensity, lesion_intensity]).astype(np.float32)
    color = np.vstack([anatomy_color, lesion_color]).astype(np.uint8)

    arr = np.zeros(len(positions), dtype=_PLY_DTYPE)
    arr["x"] = positions[:, 0]
    arr["y"] = positions[:, 1]
    arr["z"] = positions[:, 2]
    arr["red"] = color[:, 0]
    arr["green"] = color[:, 1]
    arr["blue"] = color[:, 2]
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
        f"element vertex {len(arr)}\n"
        + _PLY_PROPERTIES
        + "end_header\n"
    ).encode("ascii")
    with open(output_ply, "wb") as fh:
        fh.write(header)
        fh.write(arr.tobytes())

    np.savez_compressed(
        state_path,
        state_schema=np.array(_GAUSSIAN_STATE_SCHEMA),
        coordinate_frame_schema=np.array(_COORDINATE_FRAME_SCHEMA),
        source_volume=np.array(str(volume_path)),
        source_roi=np.array(str(roi_path)),
        subject=np.array(subject),
        group=np.array(group),
        sampling_mode=np.array(sampling_mode),
        positions_world=positions,
        positions_voxel=voxels,
        scales=scales,
        rotations_quat=quats,
        opacity=opacity,
        intensity=intensity,
        color_rgb=color,
        is_lesion=np.concatenate([
            np.zeros(len(anatomy_positions), dtype=bool),
            np.ones(len(lesion_positions), dtype=bool),
        ]),
        affine=affine.astype(np.float64),
        source_shape=np.array(volume_data.shape[:3], dtype=np.int64),
        source_orientation=np.array(nib.aff2axcodes(volume_affine)),
        source_voxel_spacing_mm=np.array(_voxel_spacing(volume_affine), dtype=np.float64),
        foreground_voxels_total=np.array(int(meta["foreground_voxels_total"]), dtype=np.int64),
        lesion_voxels_total=np.array(int((roi_data > 0).sum()), dtype=np.int64),
        rng_seed=np.array(int(rng_seed), dtype=np.int64),
    )

    lesion_meta = {
        **meta,
        "source_roi": str(roi_path),
        "n_points": int(len(arr)),
        "n_anatomy_points": int(len(anatomy_positions)),
        "n_lesion_points": int(len(lesion_positions)),
        "lesion_voxels_total": int((roi_data > 0).sum()),
        "lesion_highlight": True,
        "lesion_color_model": "uniform_orange_roi_gaussians",
        "lesion_color_rgb": _LESION_SPLAT_ORANGE_RGB.astype(int).tolist(),
        "anatomy_opacity_scale": 0.62,
        "lesion_opacity": 1.0,
        "lesion_scale_multiplier": [3.8, 3.8, 2.6],
        "browser_scale_anisotropy_cap_ratio": 1.6,
        "browser_scale_policy": (
            "positions preserve canonical RAS millimeter MRI space; displayed "
            "Gaussian ellipsoid radii are capped to avoid thick-slice spike artifacts"
        ),
        "gaussian_state_npz": state_path.name,
    }
    lesion_meta.update({
        f"lesion_{k}": v for k, v in lesion_orientation_meta.items()
        if k in {"gaussian_orientation", "gaussian_scale_model", "gaussian_scale"}
    })

    meta_path = output_ply.parent / (output_ply.stem + "_meta.json")
    meta_path.write_text(json.dumps(lesion_meta, indent=2), encoding="utf-8")
    return lesion_meta
