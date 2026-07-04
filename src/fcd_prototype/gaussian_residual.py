"""Gaussian-space residual comparison between patient and normal Gaussian fields.

Two heatmap methods:
  - field (default): Evaluates both Gaussian fields on a shared NIfTI grid,
    subtracts field values voxel-wise. Directly implements paper's core objective.
  - scatter (legacy): Uses nearest-neighbour KD-tree to estimate normal parameters
    at patient Gaussian locations. Retained for backward compatibility.

Field method (preferred):
  Directly evaluates the continuous 3D Gaussian fields on the reference voxel grid,
  then subtracts to localize differences. See docs/gaussian-residual-method.md.

Scatter method (legacy):
  Builds a KD-tree on the *normal* field's world-space Gaussian means. For each
  *patient* Gaussian, queries k nearest normals, computes inverse-distance
  weighted estimate of normal intensity/scale at that location, and defines the
  residual as the absolute difference from the patient value.

Outputs (in --output-dir)
--------------------------
gaussian_residual_scores.npy         — raw per-patient-Gaussian residual scores
gaussian_residual_splat.ply          — heatmap PLY at patient Gaussian means
gaussian_residual_heatmap.nii.gz     — residual heatmap on reference NIfTI grid
gaussian_residual_metrics.json       — summary statistics + heatmap_method + field_* keys
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from fcd_prototype.gaussian_field import (
    GAUSSIAN_SUPPORT_FINGERPRINT_SCHEMA,
    gaussian_support_fingerprint,
)


# ---------------------------------------------------------------------------
# Heatmap colormap — 4-stop, no matplotlib dependency
# ---------------------------------------------------------------------------

# dark navy → blue-cyan → yellow → red
_CMAP_T = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float32)
_CMAP_RGB = np.array([
    [0.05, 0.05, 0.30],   # t=0.00  dark navy
    [0.00, 0.50, 0.90],   # t=0.33  blue-cyan
    [1.00, 0.85, 0.00],   # t=0.67  yellow
    [0.90, 0.00, 0.05],   # t=1.00  red
], dtype=np.float32)


def _apply_heatmap(t: np.ndarray) -> np.ndarray:
    """Map t ∈ [0,1] → (N, 3) float32 RGB via piecewise-linear colormap."""
    t = np.clip(t, 0.0, 1.0).astype(np.float32)
    rgb = np.zeros((len(t), 3), dtype=np.float32)
    for i in range(len(_CMAP_T) - 1):
        lo, hi = _CMAP_T[i], _CMAP_T[i + 1]
        seg = (t >= lo) & (t <= hi)
        if not seg.any():
            continue
        frac = (t[seg] - lo) / (hi - lo)
        rgb[seg] = _CMAP_RGB[i] * (1.0 - frac[:, None]) + _CMAP_RGB[i + 1] * frac[:, None]
    return rgb


# ---------------------------------------------------------------------------
# State loading
# ---------------------------------------------------------------------------

def _load_state(path: Path) -> dict:
    """Load a gaussian_field.pt checkpoint; return numpy arrays."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load Gaussian field checkpoints (.pt). "
            "Install with: pip install torch"
        ) from exc
    raw = torch.load(str(path), weights_only=False)

    def _to_np(v, dtype=np.float32):
        if hasattr(v, "detach"):
            return v.detach().cpu().numpy().astype(dtype)
        return np.asarray(v, dtype=dtype)

    return {
        "means_norm":    _to_np(raw["means_norm"]),
        "log_scales":    _to_np(raw["log_scales"]),
        "quat_wxyz":     _to_np(raw.get("quat_wxyz", _identity_quaternions(len(raw["means_norm"])))),
        "opacity_logit": _to_np(raw["opacity_logit"]),
        "colors_raw":    _to_np(raw["colors_raw"]),
        "norm_center":   np.asarray(raw["norm_center"], dtype=np.float32),
        "norm_scale":    float(raw["norm_scale"]),
        "affine":        np.asarray(raw["affine"], dtype=np.float64),
        "support_fingerprint_schema": str(
            raw.get("support_fingerprint_schema", GAUSSIAN_SUPPORT_FINGERPRINT_SCHEMA)
        ),
        "support_fingerprint_sha256": raw.get("support_fingerprint_sha256"),
    }


def _denormalize(
    means_norm: np.ndarray,
    norm_center: np.ndarray,
    norm_scale: float,
) -> np.ndarray:
    """Convert normalised Gaussian means back to world (mm) coordinates."""
    return (means_norm * norm_scale + norm_center).astype(np.float32)


def _scalar_intensity(opacity_logit: np.ndarray, colors_raw: np.ndarray) -> np.ndarray:
    """Scalar intensity per Gaussian: sigmoid(opacity_logit) × mean(tanh_color + 1)/2."""
    # Use float64 internally to avoid overflow in exp
    opacity = 1.0 / (1.0 + np.exp(-opacity_logit.astype(np.float64)))
    colors  = (np.tanh(colors_raw.astype(np.float64)) + 1.0) / 2.0  # (N, 3) in [0,1]
    return (opacity * colors.mean(axis=-1)).astype(np.float32)


def _mean_scale_world(log_scales: np.ndarray, norm_scale: float) -> np.ndarray:
    """Mean world-space scale (mm) per Gaussian."""
    return (np.exp(log_scales) * norm_scale).mean(axis=-1).astype(np.float32)


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.shape != b.shape:
        return None
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _support_diagnostics(patient: dict, normal: dict, atol: float = 1e-5) -> dict:
    """Return support comparison diagnostics for patient/normal fields."""
    p_quat = _normalize_quaternions(patient["quat_wxyz"])
    n_quat = _normalize_quaternions(normal["quat_wxyz"])
    p_stored_fingerprint = patient.get("support_fingerprint_sha256")
    n_stored_fingerprint = normal.get("support_fingerprint_sha256")
    p_fingerprint = gaussian_support_fingerprint(
        patient["means_norm"],
        patient["log_scales"],
        p_quat,
        patient["norm_center"],
        patient["norm_scale"],
    )
    n_fingerprint = gaussian_support_fingerprint(
        normal["means_norm"],
        normal["log_scales"],
        n_quat,
        normal["norm_center"],
        normal["norm_scale"],
    )
    diffs = {
        "means_norm_max_abs_diff": _max_abs_diff(patient["means_norm"], normal["means_norm"]),
        "log_scales_max_abs_diff": _max_abs_diff(patient["log_scales"], normal["log_scales"]),
        "quat_wxyz_max_abs_diff": _max_abs_diff(p_quat, n_quat),
        "norm_center_max_abs_diff": _max_abs_diff(patient["norm_center"], normal["norm_center"]),
        "norm_scale_abs_diff": abs(float(patient["norm_scale"]) - float(normal["norm_scale"])),
    }
    same_shapes = (
        patient["means_norm"].shape == normal["means_norm"].shape
        and patient["log_scales"].shape == normal["log_scales"].shape
        and p_quat.shape == n_quat.shape
    )
    same_support = bool(
        same_shapes
        and all(
            value is not None and value <= atol
            for value in diffs.values()
        )
    )
    return {
        "same_gaussian_support": same_support,
        "support_fingerprint_schema": GAUSSIAN_SUPPORT_FINGERPRINT_SCHEMA,
        "patient_support_fingerprint_sha256": str(p_fingerprint),
        "normal_support_fingerprint_sha256": str(n_fingerprint),
        "support_fingerprint_match": bool(p_fingerprint == n_fingerprint),
        "patient_stored_support_fingerprint_sha256": (
            str(p_stored_fingerprint) if p_stored_fingerprint else None
        ),
        "normal_stored_support_fingerprint_sha256": (
            str(n_stored_fingerprint) if n_stored_fingerprint else None
        ),
        "patient_support_fingerprint_matches_geometry": bool(
            p_stored_fingerprint is not None and str(p_stored_fingerprint) == p_fingerprint
        ),
        "normal_support_fingerprint_matches_geometry": bool(
            n_stored_fingerprint is not None and str(n_stored_fingerprint) == n_fingerprint
        ),
        "patient_support_fingerprint_stale": bool(
            p_stored_fingerprint is not None and str(p_stored_fingerprint) != p_fingerprint
        ),
        "normal_support_fingerprint_stale": bool(
            n_stored_fingerprint is not None and str(n_stored_fingerprint) != n_fingerprint
        ),
        "support_comparison_tolerance": atol,
        **diffs,
    }


def _identity_quaternions(n: int) -> np.ndarray:
    """Return n identity quaternions in wxyz order."""
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    return quats


def _normalize_quaternions(quat_wxyz: np.ndarray) -> np.ndarray:
    """Normalize quaternions in wxyz order, replacing degenerate values with identity."""
    quats = np.asarray(quat_wxyz, dtype=np.float32).copy()
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError(f"Expected quaternion array shape (N, 4), got {quats.shape}")
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    bad = norms[:, 0] < 1e-8
    quats = quats / np.clip(norms, 1e-8, None)
    if bad.any():
        quats[bad] = _identity_quaternions(int(bad.sum()))
    return quats.astype(np.float32)


def _quaternion_rotation_matrices(quat_wxyz: np.ndarray) -> np.ndarray:
    """Return rotation matrices R where world_vector = R @ local_vector."""
    q = _normalize_quaternions(quat_wxyz).astype(np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    mats = np.empty((len(q), 3, 3), dtype=np.float32)
    mats[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    mats[:, 0, 1] = 2.0 * (x * y - z * w)
    mats[:, 0, 2] = 2.0 * (x * z + y * w)
    mats[:, 1, 0] = 2.0 * (x * y + z * w)
    mats[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    mats[:, 1, 2] = 2.0 * (y * z - x * w)
    mats[:, 2, 0] = 2.0 * (x * z - y * w)
    mats[:, 2, 1] = 2.0 * (y * z + x * w)
    mats[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return mats


# ---------------------------------------------------------------------------
# Core residual computation
# ---------------------------------------------------------------------------

def compute_residual(
    patient_world:     np.ndarray,
    patient_intensity: np.ndarray,
    patient_scale:     np.ndarray,
    normal_world:      np.ndarray,
    normal_intensity:  np.ndarray,
    normal_scale:      np.ndarray,
    k_neighbors:       int = 4,
) -> np.ndarray:
    """Return raw per-patient-Gaussian residual scores (not normalised to [0,1]).

    For each patient Gaussian p_i:
      1. Find k nearest normals by world-space Euclidean distance.
      2. Compute inverse-distance weights w_j = 1 / (d_j + eps), normalised to sum 1.
         A patient Gaussian that lands exactly on a normal Gaussian (d=0) receives
         weight ≈ 1 for that neighbour, so it reads off the local normal value.
      3. Estimate normal_intensity and normal_scale at p_i as weighted sums.
      4. residual_i = |patient_intensity_i − est_normal_intensity_i|
                    + 0.2 × |patient_scale_i − est_normal_scale_i|
                          / (patient_scale_i + est_normal_scale_i + eps)

    The 0.2 coefficient keeps scale anomalies visible but subordinate to
    intensity differences.  All operations are deterministic given fixed inputs.
    """
    eps = 1e-8
    k = min(k_neighbors, len(normal_world))
    tree = cKDTree(normal_world)
    dists, idxs = tree.query(patient_world, k=k)

    # cKDTree returns 1-D arrays when k==1
    if k == 1:
        dists = dists[:, None]
        idxs  = idxs[:, None]

    weights = 1.0 / (dists + eps)              # (Np, k)
    weights /= weights.sum(axis=1, keepdims=True)

    normal_int_est   = (weights * normal_intensity[idxs]).sum(axis=1)
    normal_scale_est = (weights * normal_scale[idxs]).sum(axis=1)

    intensity_diff = np.abs(patient_intensity - normal_int_est)
    scale_mismatch = (
        np.abs(patient_scale - normal_scale_est)
        / (patient_scale + normal_scale_est + eps)
    )
    return (intensity_diff + 0.2 * scale_mismatch).astype(np.float32)


# ---------------------------------------------------------------------------
# PLY export — same binary schema as gaussian_field.py
# ---------------------------------------------------------------------------

_PLY_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ("opacity", "<f4"),
    ("scale_x", "<f4"), ("scale_y", "<f4"), ("scale_z", "<f4"),
    ("rot_w", "<f4"), ("rot_x", "<f4"), ("rot_y", "<f4"), ("rot_z", "<f4"),
])

_PLY_PROPERTIES = (
    "property float x\nproperty float y\nproperty float z\n"
    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    "property float opacity\n"
    "property float scale_x\nproperty float scale_y\nproperty float scale_z\n"
    "property float rot_w\nproperty float rot_x\nproperty float rot_y\nproperty float rot_z\n"
)


def _save_residual_ply(
    world:         np.ndarray,
    log_scales:    np.ndarray,
    quat_wxyz:     np.ndarray,
    norm_scale:    float,
    residual_norm: np.ndarray,
    path:          Path,
    max_points:    int = 200_000,
) -> None:
    """Write heatmap PLY; subsample to max_points by keeping highest-residual points."""
    N = len(world)
    if N > max_points:
        top_idx    = np.argpartition(residual_norm, -max_points)[-max_points:]
        world        = world[top_idx]
        log_scales   = log_scales[top_idx]
        quat_wxyz    = quat_wxyz[top_idx]
        residual_norm = residual_norm[top_idx]
        N = max_points

    rgb_f = _apply_heatmap(residual_norm)
    rgb   = np.clip(rgb_f * 255, 0, 255).astype(np.uint8)
    scales_world = np.exp(log_scales) * norm_scale  # (N, 3) world-space
    quats = _normalize_quaternions(quat_wxyz)

    arr = np.zeros(N, dtype=_PLY_DTYPE)
    arr["x"] = world[:, 0];  arr["y"] = world[:, 1];  arr["z"] = world[:, 2]
    arr["red"]   = rgb[:, 0];  arr["green"] = rgb[:, 1];  arr["blue"] = rgb[:, 2]
    arr["opacity"]  = residual_norm.astype(np.float32)
    arr["scale_x"]  = scales_world[:, 0]
    arr["scale_y"]  = scales_world[:, 1]
    arr["scale_z"]  = scales_world[:, 2]
    arr["rot_w"] = quats[:, 0]
    arr["rot_x"] = quats[:, 1]
    arr["rot_y"] = quats[:, 2]
    arr["rot_z"] = quats[:, 3]

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        + _PLY_PROPERTIES
        + "end_header\n"
    ).encode("ascii")
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(arr.tobytes())


# ---------------------------------------------------------------------------
# NIfTI heatmap scatter
# ---------------------------------------------------------------------------

def _scatter_to_nifti(
    world_pts:    np.ndarray,
    scores:       np.ndarray,
    ref_affine:   np.ndarray,
    ref_shape:    tuple,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """Scatter residual scores onto the reference NIfTI grid (max per voxel + smooth)."""
    shape3 = ref_shape[:3]
    inv_aff = np.linalg.inv(ref_affine)
    n = len(world_pts)
    hom   = np.hstack([world_pts.astype(np.float64), np.ones((n, 1))])
    vox_f = (inv_aff @ hom.T).T[:, :3]
    vox_i = np.round(vox_f).astype(int)

    in_bounds = np.all((vox_i >= 0) & (vox_i < np.array(shape3)), axis=1)
    vi = vox_i[in_bounds]
    sc = scores[in_bounds].astype(np.float32)

    vol = np.zeros(shape3, dtype=np.float32)
    np.maximum.at(vol, (vi[:, 0], vi[:, 1], vi[:, 2]), sc)
    vol = gaussian_filter(vol, sigma=smooth_sigma)
    return np.clip(vol, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Field-mode Gaussian evaluation on shared voxel grid
# ---------------------------------------------------------------------------

def _evaluate_gaussian_field_on_grid(
    means_world:   np.ndarray,
    log_scales:    np.ndarray,
    quat_wxyz:     np.ndarray,
    intensity:     np.ndarray,
    norm_scale:    float,
    ref_affine:    np.ndarray,
    ref_shape:     tuple,
    chunk_size:    int = 128,
) -> tuple[np.ndarray, int]:
    """Evaluate field Gaussians on reference NIfTI voxel grid; return field volume and chunk count.

    Treats each Gaussian as a rotated anisotropic 3D covariance with:
      - means_world: (N, 3) world-space means (mm)
      - log_scales: (N, 3) log of normalized scale per axis
      - quat_wxyz: (N, 4) local-to-world rotation quaternion in wxyz order
      - intensity: (N,) scalar intensity per Gaussian
      - norm_scale: denormalisation scale used to convert scales back to world units

    Evaluates Gaussians in chunks to avoid huge allocations. For each Gaussian,
    builds a conservative world-space bounding box and accumulates Gaussian
    contributions using vectorized NumPy operations.

    Returns (field_volume, n_chunks_processed).
    """
    if chunk_size <= 0:
        raise ValueError(f"field_chunk_size must be > 0, got {chunk_size}")

    shape3 = ref_shape[:3]
    inv_aff = np.linalg.inv(ref_affine)
    n_gauss = len(means_world)

    field_vol = np.zeros(shape3, dtype=np.float32)

    # Convert Gaussian means to voxel indices (continuous) for bounding boxes.
    hom = np.hstack([means_world.astype(np.float64), np.ones((n_gauss, 1))])
    vox_f = (inv_aff @ hom.T).T[:, :3].astype(np.float32)  # (N, 3)

    # Convert log_scales to world-space covariance axes and rotations.
    scales_world = (np.exp(log_scales.astype(np.float64)) * norm_scale).astype(np.float32)
    rot_mats = _quaternion_rotation_matrices(quat_wxyz)

    chunk_count = 0
    for g_start in range(0, n_gauss, chunk_size):
        g_end = min(g_start + chunk_size, n_gauss)
        chunk_count += 1

        means_vox_chunk = vox_f[g_start:g_end]  # (chunk, 3)
        means_world_chunk = means_world[g_start:g_end].astype(np.float32)
        scales_chunk = scales_world[g_start:g_end]  # (chunk, 3)
        rotations_chunk = rot_mats[g_start:g_end]  # (chunk, 3, 3)
        intensity_chunk = intensity[g_start:g_end]  # (chunk,)

        for mean_vox, mean_world, scale, rot, inten in zip(
            means_vox_chunk, means_world_chunk, scales_chunk, rotations_chunk, intensity_chunk
        ):
            # Skip if scale is too small
            if np.any(scale < 1e-4):
                continue

            # Conservative 3-sigma axis-aligned world extent for the rotated
            # covariance.  Diagonal(cov_world) = sum_j (R_ij * scale_j)^2.
            extent_world = 3.0 * np.sqrt(((rot * scale[None, :]) ** 2).sum(axis=1))
            corners = np.array(
                [
                    [sx, sy, sz]
                    for sx in (-1.0, 1.0)
                    for sy in (-1.0, 1.0)
                    for sz in (-1.0, 1.0)
                ],
                dtype=np.float32,
            ) * extent_world[None, :] + mean_world[None, :]
            corners_h = np.hstack([corners.astype(np.float64), np.ones((8, 1))])
            corners_vox = (inv_aff @ corners_h.T).T[:, :3]

            # Add the center as a fallback for tiny or degenerate extents.
            vox_bounds = np.vstack([corners_vox, mean_vox[None, :]])
            lo = np.maximum(0, np.floor(vox_bounds.min(axis=0)).astype(int))
            hi = np.minimum(shape3, np.ceil(vox_bounds.max(axis=0)).astype(int) + 1)

            if np.any(lo >= hi):
                continue

            # Vectorized evaluation in bounding box
            # Create integer grid of voxel coordinates
            ix_range = np.arange(lo[0], hi[0], dtype=int)
            iy_range = np.arange(lo[1], hi[1], dtype=int)
            iz_range = np.arange(lo[2], hi[2], dtype=int)

            # Mesh to all combinations
            ixx, iyy, izz = np.meshgrid(ix_range, iy_range, iz_range, indexing="ij")

            # Stack to (N, 3) array of voxel positions (as float for distance calculation)
            vox_grid = np.stack([ixx.ravel().astype(np.float32),
                                iyy.ravel().astype(np.float32),
                                izz.ravel().astype(np.float32)], axis=1)  # (M, 3)

            # Convert grid samples to world coordinates, then into Gaussian-local axes.
            grid_h = np.hstack([vox_grid.astype(np.float64), np.ones((len(vox_grid), 1))])
            world_grid = (ref_affine @ grid_h.T).T[:, :3].astype(np.float32)
            diff_world = world_grid - mean_world[None, :]
            local_diff = diff_world @ rot
            dist_sq = ((local_diff / scale[None, :]) ** 2).sum(axis=1)

            # Evaluate Gaussian
            gauss_vals = inten * np.exp(-0.5 * dist_sq)  # (M,)

            # Scatter back to volume using integer indices
            ix_idx = ixx.ravel().astype(int)
            iy_idx = iyy.ravel().astype(int)
            iz_idx = izz.ravel().astype(int)
            field_vol[ix_idx, iy_idx, iz_idx] += gauss_vals

    return field_vol, chunk_count


def _field_method_heatmap(
    patient_world:    np.ndarray,
    patient_log_scales: np.ndarray,
    patient_quat_wxyz: np.ndarray,
    patient_intensity: np.ndarray,
    patient_norm_scale: float,
    normal_world:     np.ndarray,
    normal_log_scales: np.ndarray,
    normal_quat_wxyz: np.ndarray,
    normal_intensity: np.ndarray,
    normal_norm_scale: float,
    ref_affine:       np.ndarray,
    ref_shape:        tuple,
    chunk_size:       int = 128,
) -> tuple[np.ndarray, dict]:
    """Evaluate both fields on shared grid, subtract, and return heatmap + stats.

    Returns (heatmap_norm, field_stats) where:
      - heatmap_norm: (shape3,) float32 in [0, 1] after normalisation
      - field_stats: dict with field_patient_sum, field_normal_sum, field_residual_mean, etc.
    """
    patient_field, patient_chunks = _evaluate_gaussian_field_on_grid(
        patient_world, patient_log_scales, patient_quat_wxyz, patient_intensity, patient_norm_scale,
        ref_affine, ref_shape, chunk_size=chunk_size,
    )
    normal_field, normal_chunks = _evaluate_gaussian_field_on_grid(
        normal_world, normal_log_scales, normal_quat_wxyz, normal_intensity, normal_norm_scale,
        ref_affine, ref_shape, chunk_size=chunk_size,
    )

    # Absolute difference
    residual_field = np.abs(patient_field - normal_field)

    # Normalise to [0, 1]
    max_val = residual_field.max()
    if max_val > 0:
        heatmap_norm = (residual_field / max_val).astype(np.float32)
    else:
        heatmap_norm = np.zeros_like(residual_field, dtype=np.float32)

    # Ensure all metrics are finite
    field_stats = {
        "field_patient_sum": float(np.nan_to_num(patient_field.sum(), nan=0.0, posinf=0.0, neginf=0.0)),
        "field_normal_sum": float(np.nan_to_num(normal_field.sum(), nan=0.0, posinf=0.0, neginf=0.0)),
        "field_residual_mean": float(np.nan_to_num(residual_field.mean(), nan=0.0, posinf=0.0, neginf=0.0)),
        "field_residual_min": float(np.nan_to_num(residual_field.min(), nan=0.0, posinf=0.0, neginf=0.0)),
        "field_residual_max": float(np.nan_to_num(residual_field.max(), nan=0.0, posinf=0.0, neginf=0.0)),
        "field_residual_p95": float(np.nan_to_num(np.percentile(residual_field, 95), nan=0.0, posinf=0.0, neginf=0.0)),
        "field_residual_p99": float(np.nan_to_num(np.percentile(residual_field, 99), nan=0.0, posinf=0.0, neginf=0.0)),
        "field_chunk_size": chunk_size,
        "field_covariance_mode": "rotated_anisotropic",
        "patient_chunks_processed": patient_chunks,
        "normal_chunks_processed": normal_chunks,
    }

    return heatmap_norm, field_stats


# ---------------------------------------------------------------------------
# ROI metrics helper
# ---------------------------------------------------------------------------

def _compute_roi_metrics(residual_norm: np.ndarray, in_roi: np.ndarray) -> dict:
    """Build ROI metrics dict; always returns all 7 keys, never NaN/Inf.

    roi_metrics_valid is False whenever the in/out split is missing one side,
    making the ratio/concentration numbers unreliable for reporting.
    """
    eps = 1e-8
    n_in  = int(in_roi.sum())
    n_out = int((~in_roi).sum())
    valid = (n_in > 0) and (n_out > 0)

    roi_mean     = float(residual_norm[in_roi].mean())  if n_in  > 0 else 0.0
    non_roi_mean = float(residual_norm[~in_roi].mean()) if n_out > 0 else 0.0
    total_sum    = float(residual_norm.sum())

    return {
        "roi_gaussian_count":         n_in,
        "non_roi_gaussian_count":     n_out,
        "roi_metrics_valid":          valid,
        "roi_residual_mean":          roi_mean,
        "non_roi_residual_mean":      non_roi_mean,
        "roi_to_non_roi_ratio":       roi_mean / (non_roi_mean + eps) if valid else 1.0,
        "roi_residual_concentration": float(residual_norm[in_roi].sum() / (total_sum + eps)) if n_in > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# ROI membership
# ---------------------------------------------------------------------------

def _roi_membership(
    world_pts: np.ndarray,
    roi_data:  np.ndarray,
    roi_affine: np.ndarray,
) -> np.ndarray:
    """Boolean mask: True if the patient Gaussian's nearest voxel is inside the ROI."""
    inv_aff = np.linalg.inv(roi_affine)
    n = len(world_pts)
    hom   = np.hstack([world_pts.astype(np.float64), np.ones((n, 1))])
    vox_f = (inv_aff @ hom.T).T[:, :3]
    vox_i = np.round(vox_f).astype(int)
    shape3 = roi_data.shape[:3]
    mask = np.zeros(n, dtype=bool)
    in_bounds = np.all((vox_i >= 0) & (vox_i < np.array(shape3)), axis=1)
    vi = vox_i[in_bounds]
    mask[in_bounds] = roi_data[vi[:, 0], vi[:, 1], vi[:, 2]] > 0
    return mask


# ---------------------------------------------------------------------------
# Main comparison entry point
# ---------------------------------------------------------------------------

def compare_gaussian_fields(
    patient_field:      Path,
    normal_field:       Path,
    reference_volume:   Path,
    output_dir:         Path,
    roi:                Optional[Path] = None,
    k_neighbors:        int   = 4,
    max_points:         int   = 200_000,
    heatmap_percentile: float = 99.0,
    heatmap_method:     str   = "field",
    field_chunk_size:   int   = 128,
) -> dict:
    """Compare patient vs. normal Gaussian fields; write all residual outputs.

    Args:
        heatmap_method: "field" (default) evaluates fields on shared grid and subtracts;
                        "scatter" uses nearest-neighbor parameter residuals (legacy).
        field_chunk_size: When heatmap_method="field", chunk size for grid evaluation
                          to manage memory usage on CPU.

    Returns metrics dict mirroring gaussian_residual_metrics.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if heatmap_method not in ("field", "scatter"):
        raise ValueError(f"heatmap_method must be 'field' or 'scatter', got {heatmap_method!r}")

    p = _load_state(Path(patient_field))
    n = _load_state(Path(normal_field))

    p_world = _denormalize(p["means_norm"], p["norm_center"], p["norm_scale"])
    n_world = _denormalize(n["means_norm"], n["norm_center"], n["norm_scale"])

    p_int   = _scalar_intensity(p["opacity_logit"], p["colors_raw"])
    n_int   = _scalar_intensity(n["opacity_logit"], n["colors_raw"])

    p_scale = _mean_scale_world(p["log_scales"], p["norm_scale"])
    n_scale = _mean_scale_world(n["log_scales"], n["norm_scale"])

    support_meta = _support_diagnostics(p, n)
    if support_meta["same_gaussian_support"]:
        residual_raw = np.abs(p_int - n_int).astype(np.float32)
        residual_score_method = "indexwise_same_support_intensity"
        gaussian_subtraction_claim_allowed = True
    else:
        residual_raw = compute_residual(
            p_world, p_int, p_scale,
            n_world, n_int, n_scale,
            k_neighbors=k_neighbors,
        )
        residual_score_method = "nearest_neighbor_common_space"
        gaussian_subtraction_claim_allowed = False

    p_hi = float(np.percentile(residual_raw, heatmap_percentile))
    residual_norm = np.clip(residual_raw / (p_hi + 1e-8), 0.0, 1.0)

    np.save(str(output_dir / "gaussian_residual_scores.npy"), residual_raw)

    _save_residual_ply(
        p_world,
        p["log_scales"],
        p["quat_wxyz"],
        p["norm_scale"],
        residual_norm,
        output_dir / "gaussian_residual_splat.ply",
        max_points=max_points,
    )

    ref_img = nib.as_closest_canonical(nib.load(str(reference_volume)))
    ref_data = ref_img.get_fdata(dtype=np.float32)
    ref_affine = ref_img.affine.astype(np.float64)

    field_stats: dict = {}
    if heatmap_method == "field":
        heatmap_vol, field_stats = _field_method_heatmap(
            p_world, p["log_scales"], p["quat_wxyz"], p_int, p["norm_scale"],
            n_world, n["log_scales"], n["quat_wxyz"], n_int, n["norm_scale"],
            ref_affine, ref_data.shape,
            chunk_size=field_chunk_size,
        )
    else:  # scatter
        heatmap_vol = _scatter_to_nifti(p_world, residual_norm, ref_affine, ref_data.shape)

    nib.save(nib.Nifti1Image(heatmap_vol, ref_affine),
             str(output_dir / "gaussian_residual_heatmap.nii.gz"))

    roi_metrics: dict = {}
    if roi is not None and Path(roi).is_file():
        roi_img  = nib.as_closest_canonical(nib.load(str(roi)))
        roi_data = roi_img.get_fdata(dtype=np.float32)
        roi_aff  = roi_img.affine.astype(np.float64)
        in_roi   = _roi_membership(p_world, roi_data, roi_aff)
        roi_metrics = _compute_roi_metrics(residual_norm, in_roi)

    lesion_metrics: dict = {}
    lesion_mask_path = Path(patient_field).parent / "lesion_gaussian_mask.npy"
    if lesion_mask_path.is_file():
        lesion_mask = np.load(str(lesion_mask_path))
        lesion_metrics["lesion_gaussian_count"] = int(lesion_mask.sum())

    metrics = {
        "k_neighbors":        k_neighbors,
        "n_patient":          int(len(p_world)),
        "n_normal":           int(len(n_world)),
        "heatmap_percentile": heatmap_percentile,
        "heatmap_method":     heatmap_method,
        "residual_score_method": residual_score_method,
        "gaussian_subtraction_claim_allowed": gaussian_subtraction_claim_allowed,
        "residual_min":       float(residual_raw.min()),
        "residual_max":       float(residual_raw.max()),
        "residual_mean":      float(residual_raw.mean()),
        "residual_p95":       float(np.percentile(residual_raw, 95)),
        "residual_p99":       float(np.percentile(residual_raw, 99)),
        **support_meta,
        **field_stats,
        **roi_metrics,
        **lesion_metrics,
    }
    (output_dir / "gaussian_residual_metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Gaussian-space residual comparison between patient and normal Gaussian fields."
        )
    )
    p.add_argument("--patient-field",     type=Path, required=True,
                   help="Patient gaussian_field.pt")
    p.add_argument("--normal-field",      type=Path, required=True,
                   help="Normal/counterfactual gaussian_field.pt")
    p.add_argument("--reference-volume",  type=Path, required=True,
                   help="Patient canonical NIfTI (for affine/shape of heatmap output)")
    p.add_argument("--roi",               type=Path, default=None,
                   help="Optional ROI mask NIfTI for per-region metrics")
    p.add_argument("--output-dir",        type=Path, required=True)
    p.add_argument("--k-neighbors",       type=int,   default=4)
    p.add_argument("--max-points",        type=int,   default=200_000,
                   help="Max points in residual PLY; highest-residual kept (default 200000)")
    p.add_argument("--heatmap-percentile", type=float, default=99.0,
                   help="Percentile for residual normalisation (default 99.0)")
    p.add_argument("--heatmap-method",    type=str,   default="field",
                   choices=["field", "scatter"],
                   help="Heatmap generation method: 'field' (default) evaluates fields on grid "
                        "and subtracts; 'scatter' uses nearest-neighbor parameter residuals")
    p.add_argument("--field-chunk-size",  type=int,   default=128,
                   help="When using heatmap-method=field, chunk size for grid evaluation (default 128)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    metrics = compare_gaussian_fields(
        patient_field=args.patient_field,
        normal_field=args.normal_field,
        reference_volume=args.reference_volume,
        output_dir=args.output_dir,
        roi=args.roi,
        k_neighbors=args.k_neighbors,
        max_points=args.max_points,
        heatmap_percentile=args.heatmap_percentile,
        heatmap_method=args.heatmap_method,
        field_chunk_size=args.field_chunk_size,
    )
    print(
        f"Residual complete: {metrics['n_patient']} patient / "
        f"{metrics['n_normal']} normal Gaussians  "
        f"method={metrics['heatmap_method']}  "
        f"p99={metrics['residual_p99']:.4f}"
    )
    if metrics.get("roi_metrics_valid"):
        print(
            f"  ROI/non-ROI ratio: {metrics['roi_to_non_roi_ratio']:.2f}  "
            f"concentration: {metrics['roi_residual_concentration']:.3f}"
        )


if __name__ == "__main__":
    main()
