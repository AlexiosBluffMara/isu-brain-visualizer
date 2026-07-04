"""MRI → optimized Gaussian field training.

Initializes trainable 3D Gaussian parameters from foreground MRI voxels and
optimizes them against orthographic projections (MIPs) of the volume.  An
optional sampled voxel-space loss anchors the field to the 3D MRI volume itself.

Backends
--------
gsplat   — uses gsplat.rendering.rasterization with camera_model='ortho'.
            Requires gsplat>=1.5 and a CUDA device.
fallback  — pure PyTorch differentiable splatting (alpha-composited per-ray sum).
            CPU/GPU, no gsplat dependency; suitable for unit tests and small volumes.

Outputs (in --output-dir)
--------------------------
gaussian_field.pt               — torch save of params + metadata dict
optimized_gaussian_splat.ply    — binary PLY (x/y/z, rgb, opacity, scale_x/y/z, rot_w/x/y/z)
training_metrics.json           — loss history and run metadata
reconstruction_preview.png      — target vs reconstruction panels
lesion_gaussian_mask.npy        — (optional, only when --roi supplied)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import warnings
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np


GAUSSIAN_SUPPORT_FINGERPRINT_SCHEMA = "gaussian_support_v1"


# ---------------------------------------------------------------------------
# Dependency probing
# ---------------------------------------------------------------------------

def _check_torch(device: str) -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for gaussian_field training. "
            "Install with: pip install torch"
        ) from exc
    if device == "cuda":
        import torch as _torch
        if not _torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is False. "
                "Check your CUDA installation or use --device cpu."
            )


def _gsplat_available() -> bool:
    try:
        from gsplat.rendering import rasterization  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# NIfTI loading helpers
# ---------------------------------------------------------------------------

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


def _as_normalized_volume(data: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
        return np.clip(finite, 0.0, 1.0).astype(np.float32)
    return _robust_normalize(finite)


def _colors_raw_from_intensity(intensity: np.ndarray) -> np.ndarray:
    """Encode [0, 1] grayscale so (tanh(raw)+1)/2 reconstructs it."""
    clipped = np.clip(intensity.astype(np.float32), 1e-4, 1.0 - 1e-4)
    raw = np.arctanh(clipped * 2.0 - 1.0).astype(np.float32)
    return np.stack([raw, raw, raw], axis=-1)


def _normalize_support_quaternions(quat_wxyz: np.ndarray) -> np.ndarray:
    quats = np.asarray(quat_wxyz, dtype=np.float32).copy()
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError(f"Expected quaternion array shape (N, 4), got {quats.shape}")
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    bad = norms[:, 0] < 1e-8
    quats = quats / np.clip(norms, 1e-8, None)
    if bad.any():
        quats[bad] = 0.0
        quats[bad, 0] = 1.0
    return quats.astype(np.float32)


def gaussian_support_fingerprint(
    means_norm: np.ndarray,
    log_scales: np.ndarray,
    quat_wxyz: np.ndarray,
    norm_center: np.ndarray,
    norm_scale: float,
) -> str:
    """Return a stable SHA-256 digest for Gaussian geometry/support.

    The digest intentionally excludes intensity-bearing fields (`colors_raw`,
    `opacity_logit`). Patient and synthetic-normal fields can differ in signal
    while still proving that Gaussian i occupies the same position, scale, and
    orientation in both fields.
    """
    arrays = (
        ("means_norm", np.asarray(means_norm, dtype=np.float32)),
        ("log_scales", np.asarray(log_scales, dtype=np.float32)),
        ("quat_wxyz", _normalize_support_quaternions(quat_wxyz)),
        ("norm_center", np.asarray(norm_center, dtype=np.float32)),
        ("norm_scale", np.asarray([norm_scale], dtype=np.float32)),
    )
    h = hashlib.sha256()
    h.update(GAUSSIAN_SUPPORT_FINGERPRINT_SCHEMA.encode("ascii"))
    for name, arr in arrays:
        contiguous = np.ascontiguousarray(arr)
        h.update(name.encode("ascii"))
        h.update(str(contiguous.shape).encode("ascii"))
        h.update(contiguous.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Target projection building (orthographic MIP / mean-slab)
# ---------------------------------------------------------------------------

def _build_targets(
    norm: np.ndarray,
    image_size: int,
    views: list[str],
) -> dict[str, np.ndarray]:
    """Build float32 target images for each view by max-intensity projection."""
    from scipy.ndimage import zoom  # already in deps

    targets: dict[str, np.ndarray] = {}
    for view in views:
        if view == "axial":
            proj = norm.max(axis=2)          # (X, Y)
        elif view == "coronal":
            proj = norm.max(axis=1)          # (X, Z)
        elif view == "sagittal":
            proj = norm.max(axis=0)          # (Y, Z)
        else:
            raise ValueError(f"Unknown view {view!r}; use axial/coronal/sagittal")
        # Resize to image_size x image_size
        factor = (image_size / proj.shape[0], image_size / proj.shape[1])
        resized = zoom(proj, factor, order=1)
        resized = np.clip(resized, 0.0, 1.0).astype(np.float32)
        targets[view] = resized
    return targets


# ---------------------------------------------------------------------------
# Gaussian parameter initialization
# ---------------------------------------------------------------------------

def _sample_foreground_indices(
    norm: np.ndarray,
    roi_data: Optional[np.ndarray],
    num_gaussians: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample voxel indices biased toward intensity gradients and ROI."""
    from scipy.ndimage import generic_gradient_magnitude, gaussian_filter  # noqa: F401

    threshold = float(np.percentile(norm[norm > 0], 5)) if (norm > 0).any() else 0.0
    fg_mask = norm > threshold
    fg_idx = np.argwhere(fg_mask)
    if len(fg_idx) == 0:
        raise ValueError("No foreground voxels found in volume.")

    # Gradient magnitude as sampling weight
    smoothed = gaussian_filter(norm.astype(np.float64), sigma=1.0)
    gx = np.gradient(smoothed, axis=0)
    gy = np.gradient(smoothed, axis=1)
    gz = np.gradient(smoothed, axis=2)
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2).astype(np.float32)
    grad_fg = grad_mag[fg_mask]
    gmax = float(grad_fg.max()) if grad_fg.max() > 0 else 1.0
    weights = (1.0 + grad_fg / gmax).astype(np.float64)  # [1, 2]

    # ROI boost: up to 3× for ROI voxels, capped so ROI <= 50% of budget
    roi_mask_fg = np.zeros(len(fg_idx), dtype=bool)
    if roi_data is not None:
        roi_fg = roi_data[fg_mask] > 0
        roi_mask_fg = roi_fg
        weights[roi_fg] *= 3.0

    weights /= weights.sum()

    n = min(num_gaussians, len(fg_idx))
    sel = rng.choice(len(fg_idx), size=n, replace=False, p=weights)

    # If ROI would exceed 50%, re-balance (avoids all gaussians collapsing to ROI)
    roi_sel = roi_mask_fg[sel]
    roi_n = int(roi_sel.sum())
    max_roi = n // 2
    if roi_n > max_roi and max_roi > 0:
        roi_positions = np.where(roi_sel)[0]
        drop = rng.choice(roi_positions, size=roi_n - max_roi, replace=False)
        mask = np.ones(n, dtype=bool)
        mask[drop] = False
        # Fill dropped slots from non-ROI pool
        non_roi_remaining = np.where(~roi_mask_fg)[0]
        already_selected = set(sel[mask].tolist())
        candidates = [i for i in non_roi_remaining if i not in already_selected]
        fill_n = min(roi_n - max_roi, len(candidates))
        if fill_n > 0:
            fill = rng.choice(candidates, size=fill_n, replace=False)
            sel = np.concatenate([sel[mask], fill])
        else:
            sel = sel[mask]

    return fg_idx[sel]


def _init_gaussians(
    norm: np.ndarray,
    affine: np.ndarray,
    roi_data: Optional[np.ndarray],
    num_gaussians: int,
    seed: int,
    jitter_voxels: bool = True,
) -> dict[str, "np.ndarray"]:
    """
    Return numpy arrays for all trainable Gaussian parameters.

    means_world  (N, 3) — world-space coordinates (mm), normalized to [-1, 1]^3
    log_scales   (N, 3) — log of isotropic voxel scales
    quat_wxyz    (N, 4) — identity quaternions [1, 0, 0, 0]
    opacity_logit (N,)  — logit(initial_opacity)
    colors_raw   (N, 3) — linear grayscale RGB in [-inf, inf] (pre-sigmoid)
    is_roi       (N,)   — bool, True if voxel is in ROI
    norm_center  (3,)   — world center used for normalization
    norm_scale   float  — half-range used for normalization

    When jitter_voxels is True, each Gaussian mean is offset by a uniform
    sample in [-0.5, 0.5] voxels per axis before applying the affine.
    voxel_indices and is_roi are always based on the original integer voxels.
    """
    rng = np.random.default_rng(seed)

    vox_idx = _sample_foreground_indices(norm, roi_data, num_gaussians, rng)
    n = len(vox_idx)

    # World coordinates — optionally apply sub-voxel jitter so splats fill
    # tissue continuously rather than sitting exactly at voxel centers.
    vox_coords = vox_idx.astype(np.float64)
    if jitter_voxels:
        vox_coords = vox_coords + rng.uniform(-0.5, 0.5, size=(n, 3))

    ones = np.ones((n, 1), dtype=np.float64)
    hom = np.hstack([vox_coords, ones])
    world = (affine @ hom.T).T[:, :3].astype(np.float32)

    # Normalize to [-1, 1]^3
    center = world.mean(axis=0)
    half_range = float(np.abs(world - center).max())
    if half_range == 0.0:
        half_range = 1.0
    means_norm = (world - center) / half_range

    # Voxel spacing → initial scale
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).astype(np.float32)
    scale_init = spacing / half_range  # normalized
    log_scales = np.log(np.clip(scale_init, 1e-6, None))
    log_scales = np.tile(log_scales, (n, 1))

    # Identity quaternions
    quat = np.zeros((n, 4), dtype=np.float32)
    quat[:, 0] = 1.0

    # Opacity: initialized around 0.5 (logit(0.5) = 0)
    intensities = norm[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]]
    opacity_init = np.clip(intensities, 0.05, 0.95).astype(np.float32)
    opacity_logit = np.log(opacity_init / (1.0 - opacity_init))

    # Color: grayscale in [-1, 1] pre-tanh space
    colors_raw = (intensities * 2.0 - 1.0).astype(np.float32)
    colors_raw = np.stack([colors_raw, colors_raw, colors_raw], axis=-1)

    # ROI membership
    is_roi = np.zeros(n, dtype=bool)
    if roi_data is not None:
        is_roi = roi_data[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]] > 0

    return {
        "means_norm": means_norm,
        "log_scales": log_scales,
        "quat_wxyz": quat,
        "opacity_logit": opacity_logit,
        "colors_raw": colors_raw,
        "is_roi": is_roi,
        "norm_center": center,
        "norm_scale": np.float32(half_range),
        "voxel_indices": vox_idx.astype(np.int32),
        "jitter_voxels": jitter_voxels,
    }


# ---------------------------------------------------------------------------
# Volume-space sampled renderer/loss helpers
# ---------------------------------------------------------------------------

def _build_volume_samples(
    norm: np.ndarray,
    affine: np.ndarray,
    norm_center: np.ndarray,
    norm_scale: float,
    sample_count: int,
    seed: int,
    foreground_fraction: float = 0.75,
) -> dict[str, np.ndarray | int | float]:
    """Build deterministic voxel samples for a 3D reconstruction loss.

    Samples foreground tissue plus low-signal/background voxels so the field is
    penalized both for missing anatomy and for filling the acquisition padding.
    """
    if sample_count <= 0:
        raise ValueError("volume_sample_count must be positive when volume loss is enabled.")

    rng = np.random.default_rng(seed)
    positive = norm[norm > 0]
    foreground_threshold = float(np.percentile(positive, 5)) if positive.size else 0.0
    fg_mask = norm > foreground_threshold
    fg_idx = np.argwhere(fg_mask)
    if len(fg_idx) == 0:
        raise ValueError("No foreground voxels found for volume reconstruction loss.")

    bg_idx = np.argwhere(~fg_mask)
    if len(bg_idx) == 0:
        fg_n = sample_count
        bg_n = 0
    else:
        fg_n = int(round(sample_count * foreground_fraction))
        fg_n = max(1, min(sample_count, fg_n))
        bg_n = sample_count - fg_n

    def _draw(idx: np.ndarray, n: int) -> np.ndarray:
        if n <= 0 or len(idx) == 0:
            return np.empty((0, 3), dtype=np.int64)
        choice = rng.choice(len(idx), size=n, replace=n > len(idx))
        return idx[choice]

    fg_sel = _draw(fg_idx, fg_n)
    bg_sel = _draw(bg_idx, bg_n)
    vox_idx = np.vstack([fg_sel, bg_sel]).astype(np.int64, copy=False)
    labels = np.concatenate([
        np.ones(len(fg_sel), dtype=np.uint8),
        np.zeros(len(bg_sel), dtype=np.uint8),
    ])

    perm = rng.permutation(len(vox_idx))
    vox_idx = vox_idx[perm]
    labels = labels[perm]

    hom = np.hstack([
        vox_idx.astype(np.float64),
        np.ones((len(vox_idx), 1), dtype=np.float64),
    ])
    world = (affine @ hom.T).T[:, :3]
    sample_points_norm = ((world - norm_center) / float(norm_scale)).astype(np.float32)
    targets = norm[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]].astype(np.float32)

    return {
        "sample_points_norm": sample_points_norm,
        "targets": targets,
        "voxel_indices": vox_idx.astype(np.int32),
        "is_foreground": labels,
        "foreground_count": int(labels.sum()),
        "background_count": int((labels == 0).sum()),
        "foreground_fraction": float(labels.mean()) if len(labels) else 0.0,
        "foreground_threshold": foreground_threshold,
    }


def _render_volume_samples(
    means: "torch.Tensor",
    log_scales: "torch.Tensor",
    opacity_logit: "torch.Tensor",
    colors_raw: "torch.Tensor",
    sample_points_norm: "torch.Tensor",
    gaussian_chunk_size: int = 512,
) -> "torch.Tensor":
    """Evaluate the Gaussian field at normalized 3D sample points.

    This is a differentiable density accumulation, not a screen-space renderer.
    It gives the optimizer a volumetric target in the MRI coordinate system.
    """
    import torch

    M = sample_points_norm.shape[0]
    density_sum = torch.zeros(M, device=means.device, dtype=torch.float32)
    weighted_color_sum = torch.zeros(M, device=means.device, dtype=torch.float32)

    scales = torch.exp(log_scales).clamp(min=1e-4)
    opacities = torch.sigmoid(opacity_logit)
    colors = ((torch.tanh(colors_raw) + 1.0) / 2.0).mean(dim=-1)

    chunk = max(1, int(gaussian_chunk_size))
    for start in range(0, means.shape[0], chunk):
        end = min(start + chunk, means.shape[0])
        delta = sample_points_norm[:, None, :] - means[None, start:end, :]
        scaled = delta / scales[None, start:end, :]
        density = torch.exp(-0.5 * torch.sum(scaled * scaled, dim=-1))
        density = density * opacities[start:end].unsqueeze(0)
        density_sum = density_sum + density.sum(dim=1)
        weighted_color_sum = weighted_color_sum + density @ colors[start:end]

    mean_color = weighted_color_sum / density_sum.clamp_min(1e-8)
    coverage = -torch.expm1(-density_sum.clamp(max=60.0))
    return (mean_color * coverage).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Project a volume onto an existing Gaussian support
# ---------------------------------------------------------------------------

def _to_numpy(v, dtype=np.float32) -> np.ndarray:
    if hasattr(v, "detach"):
        return v.detach().cpu().numpy().astype(dtype)
    return np.asarray(v, dtype=dtype)


def _sample_volume_at_world(
    norm: np.ndarray,
    affine: np.ndarray,
    world: np.ndarray,
) -> tuple[np.ndarray, int]:
    from scipy.ndimage import map_coordinates

    hom = np.hstack([
        world.astype(np.float64),
        np.ones((len(world), 1), dtype=np.float64),
    ])
    inv_affine = np.linalg.inv(affine)
    vox = (inv_affine @ hom.T).T[:, :3]
    shape = np.array(norm.shape[:3], dtype=np.float64)
    in_bounds = np.all((vox >= 0.0) & (vox <= (shape - 1.0)), axis=1)
    sampled = map_coordinates(
        norm,
        [vox[:, 0], vox[:, 1], vox[:, 2]],
        order=1,
        mode="nearest",
    ).astype(np.float32)
    return np.clip(sampled, 0.0, 1.0), int(in_bounds.sum())


def project_volume_to_gaussian_support(
    support_field: Path,
    volume_path: Path,
    output_dir: Path,
    *,
    preview_image_size: int = 64,
    copy_opacity: bool = True,
) -> dict:
    """Write a gaussian_field.pt/PLY by sampling a volume on an existing support.

    Geometry is copied from `support_field`. Only the intensity-bearing color
    attributes are replaced by samples from `volume_path`, making patient vs.
    normal Gaussian subtraction index-wise meaningful.
    """
    import torch

    support_field = Path(support_field)
    volume_path = Path(volume_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = torch.load(str(support_field), weights_only=False)
    means_norm = _to_numpy(raw["means_norm"])
    log_scales = _to_numpy(raw["log_scales"])
    quat_wxyz = _to_numpy(raw.get("quat_wxyz", np.zeros((len(means_norm), 4), dtype=np.float32)))
    if quat_wxyz.shape == (len(means_norm), 4) and np.allclose(quat_wxyz, 0.0):
        quat_wxyz[:, 0] = 1.0
    support_opacity = _to_numpy(raw["opacity_logit"])
    norm_center = np.asarray(raw["norm_center"], dtype=np.float32)
    norm_scale = float(raw["norm_scale"])
    support_affine = np.asarray(raw["affine"], dtype=np.float64)
    support_fingerprint = gaussian_support_fingerprint(
        means_norm,
        log_scales,
        quat_wxyz,
        norm_center,
        norm_scale,
    )

    data, volume_affine = _load_canonical(volume_path)
    norm = _as_normalized_volume(data)
    world = means_norm * norm_scale + norm_center
    sampled_intensity, in_bounds_count = _sample_volume_at_world(norm, volume_affine, world)
    colors_raw = _colors_raw_from_intensity(sampled_intensity)

    if copy_opacity:
        opacity_logit = support_opacity.astype(np.float32)
        opacity_source = "support_field"
    else:
        opacity_init = np.clip(sampled_intensity, 0.05, 0.95).astype(np.float32)
        opacity_logit = np.log(opacity_init / (1.0 - opacity_init)).astype(np.float32)
        opacity_source = "sampled_volume_intensity"

    pt_path = output_dir / "gaussian_field.pt"
    torch.save({
        "means_norm": torch.tensor(means_norm, dtype=torch.float32),
        "log_scales": torch.tensor(log_scales, dtype=torch.float32),
        "quat_wxyz": torch.tensor(quat_wxyz, dtype=torch.float32),
        "opacity_logit": torch.tensor(opacity_logit, dtype=torch.float32),
        "colors_raw": torch.tensor(colors_raw, dtype=torch.float32),
        "norm_center": norm_center,
        "norm_scale": norm_scale,
        "affine": support_affine,
        "source_volume_affine": volume_affine,
        "num_gaussians": int(len(means_norm)),
        "support_fingerprint_schema": GAUSSIAN_SUPPORT_FINGERPRINT_SCHEMA,
        "support_fingerprint_sha256": support_fingerprint,
        "backend": "support_projection",
        "device": "cpu",
        "seed": raw.get("seed"),
        "jitter_voxels": raw.get("jitter_voxels"),
        "volume_path": str(volume_path),
        "roi_path": None,
        "support_field": str(support_field),
        "same_gaussian_support_as": str(support_field),
        "same_gaussian_support_as_fingerprint": support_fingerprint,
        "support_projection": True,
        "opacity_source": opacity_source,
    }, str(pt_path))

    ply_path = output_dir / "optimized_gaussian_splat.ply"
    _save_ply(
        means_norm,
        log_scales,
        quat_wxyz,
        opacity_logit,
        colors_raw,
        norm_center,
        norm_scale,
        support_affine,
        ply_path,
    )

    view_list = ["axial", "coronal", "sagittal"]
    targets = _build_targets(norm, preview_image_size, view_list)
    renders = {view: np.repeat(targets[view][..., None], 3, axis=-1) for view in view_list}
    _save_preview(targets, renders, view_list, output_dir / "reconstruction_preview.png")

    metrics = {
        "volume_path": str(volume_path),
        "support_field": str(support_field),
        "num_gaussians": int(len(means_norm)),
        "steps": 0,
        "image_size": preview_image_size,
        "views": view_list,
        "learning_rate": 0.0,
        "device": "cpu",
        "backend": "support_projection",
        "training_objective": "projected_volume_on_existing_gaussian_support",
        "support_fingerprint_schema": GAUSSIAN_SUPPORT_FINGERPRINT_SCHEMA,
        "support_fingerprint_sha256": support_fingerprint,
        "same_gaussian_support_as": str(support_field),
        "same_gaussian_support_as_fingerprint": support_fingerprint,
        "support_projection": True,
        "copy_opacity": copy_opacity,
        "opacity_source": opacity_source,
        "sampled_intensity_min": float(sampled_intensity.min()) if len(sampled_intensity) else None,
        "sampled_intensity_max": float(sampled_intensity.max()) if len(sampled_intensity) else None,
        "sampled_intensity_mean": float(sampled_intensity.mean()) if len(sampled_intensity) else None,
        "support_points_in_volume_bounds": in_bounds_count,
        "support_points_in_volume_bounds_fraction": (
            float(in_bounds_count / len(means_norm)) if len(means_norm) else 0.0
        ),
        "final_loss": None,
        "final_mip_loss": None,
        "final_volume_loss": None,
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


# ---------------------------------------------------------------------------
# Fallback renderer (pure PyTorch, no gsplat)
# ---------------------------------------------------------------------------

def _fallback_render(
    means: "torch.Tensor",    # (N, 3) normalized
    log_scales: "torch.Tensor",  # (N, 3)
    opacity_logit: "torch.Tensor",  # (N,)
    colors_raw: "torch.Tensor",     # (N, 3)
    image_size: int,
    view: str,
) -> "torch.Tensor":
    """
    Differentiable orthographic alpha-compositing renderer.
    Projects each Gaussian onto the image plane and accumulates contributions.
    Returns (H, W, 3) float32 in [0, 1].
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    N = means.shape[0]
    H = W = image_size

    scales = torch.exp(log_scales).clamp(min=1e-4)   # (N, 3)
    opacities = torch.sigmoid(opacity_logit)           # (N,)
    colors = (torch.tanh(colors_raw) + 1.0) / 2.0     # (N, 3) in [0,1]

    # Choose projection axes based on view
    if view == "axial":
        px, py, pz = 0, 1, 2   # project along z; screen is x,y
    elif view == "coronal":
        px, py, pz = 0, 2, 1
    else:  # sagittal
        px, py, pz = 1, 2, 0

    cx = means[:, px]   # (N,)
    cy = means[:, py]   # (N,)
    sx = scales[:, px]
    sy = scales[:, py]

    # Map [-1,1] → [0, H-1]
    px_f = (cx + 1.0) / 2.0 * (W - 1)
    py_f = (cy + 1.0) / 2.0 * (H - 1)

    # Sort back-to-front along depth axis so alpha blending is consistent
    depth = means[:, pz].detach()
    order = torch.argsort(depth, descending=True)

    px_f = px_f[order]
    py_f = py_f[order]
    sx = sx[order]
    sy = sy[order]
    opacities = opacities[order]
    colors = colors[order]

    # Rasterize with a small Gaussian footprint per splat
    # Use a grid of pixel centers
    grid_y = torch.linspace(0, H - 1, H, device=means.device)
    grid_x = torch.linspace(0, W - 1, W, device=means.device)
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")  # (H, W)

    # Accumulate in a transmittance loop — vectorized over Gaussians in chunks
    accum_color = torch.zeros(H, W, 3, device=means.device, dtype=torch.float32)
    transmittance = torch.ones(H, W, device=means.device, dtype=torch.float32)

    chunk = 512
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        # (chunk, H, W)
        dx = gx.unsqueeze(0) - px_f[start:end, None, None]
        dy = gy.unsqueeze(0) - py_f[start:end, None, None]
        # Convert to pixel units first, then clamp to 0.5 px minimum.
        # Clamping normalized scales before multiplying makes tiny Gaussians
        # artificially huge (0.5 * W/2 = huge splat for a small volume).
        sx_px = (sx[start:end, None, None] * (W / 2.0)).clamp(min=0.5)
        sy_px = (sy[start:end, None, None] * (H / 2.0)).clamp(min=0.5)
        gauss = torch.exp(-0.5 * ((dx / sx_px) ** 2 + (dy / sy_px) ** 2))  # (chunk,H,W)
        alpha = opacities[start:end, None, None] * gauss                    # (chunk,H,W)

        for i in range(end - start):
            a = alpha[i]                                      # (H, W)
            weighted_alpha = a * transmittance                # (H, W)
            c = colors[start + i]                            # (3,)
            accum_color += weighted_alpha.unsqueeze(-1) * c.view(1, 1, 3)
            transmittance = transmittance * (1.0 - a)

    return accum_color.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# gsplat renderer wrapper
# ---------------------------------------------------------------------------

def _gsplat_viewmat(view: str, device: "torch.device") -> "torch.Tensor":
    """Return a (1, 4, 4) world-to-camera viewmat for an axis-aligned ortho view.

    Translation is always along camera-z (column 3, index 2) so that all views
    place the scene in front of the near plane regardless of which world axes
    are swapped into camera space.

    Candidate matrices verified to render non-black with gsplat ortho:
      axial:    R=I,                      t=[0,0,2]
      coronal:  R=[[1,0,0],[0,0,1],[0,1,0]], t=[0,0,2]  (world y→cam z)
      sagittal: R=[[0,1,0],[0,0,1],[1,0,0]], t=[0,0,2]  (world x→cam z)
    """
    import torch

    if view == "axial":
        R = torch.eye(3, device=device, dtype=torch.float32)
    elif view == "coronal":
        R = torch.tensor([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ], device=device, dtype=torch.float32)
    else:  # sagittal
        R = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], device=device, dtype=torch.float32)

    t = torch.tensor([0.0, 0.0, 2.0], device=device, dtype=torch.float32)
    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t
    return viewmat.unsqueeze(0)  # (1, 4, 4)


def _gsplat_render(
    means: "torch.Tensor",
    log_scales: "torch.Tensor",
    quat_wxyz: "torch.Tensor",
    opacity_logit: "torch.Tensor",
    colors_raw: "torch.Tensor",
    image_size: int,
    view: str,
) -> "torch.Tensor":
    """Render one view using gsplat.rendering.rasterization with camera_model='ortho'."""
    import torch
    from gsplat.rendering import rasterization

    N = means.shape[0]
    H = W = image_size

    scales = torch.exp(log_scales).clamp(min=1e-6)
    opacities = torch.sigmoid(opacity_logit)
    colors = (torch.tanh(colors_raw) + 1.0) / 2.0

    viewmat = _gsplat_viewmat(view, means.device)

    # Orthographic K: fx=fy=W/2, cx=cy=W/2
    fx = fy = float(W) / 2.0
    cx = cy = float(W) / 2.0
    Ks = torch.tensor(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        device=means.device, dtype=torch.float32,
    ).unsqueeze(0)  # (1, 3, 3)

    quats_norm = torch.nn.functional.normalize(quat_wxyz, dim=-1, eps=1e-8)

    renders, alphas, _ = rasterization(
        means=means,
        quats=quats_norm,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=Ks,
        width=W,
        height=H,
        camera_model="ortho",
        near_plane=0.01,
        far_plane=10.0,
        sh_degree=None,
    )
    return renders[0].clamp(0.0, 1.0)  # (H, W, 3)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_gaussian_field(
    volume_path: Path,
    output_dir: Path,
    roi_path: Optional[Path] = None,
    num_gaussians: int = 20_000,
    steps: int = 300,
    image_size: int = 128,
    views: str = "all",
    device: str = "auto",
    seed: int = 42,
    learning_rate: float = 1e-3,
    preview_every: int = 0,
    jitter_voxels: bool = True,
    volume_loss_weight: float = 0.0,
    volume_sample_count: int = 4096,
) -> dict:
    """
    Full MRI → optimized Gaussian field training.

    Returns a metadata dict mirroring training_metrics.json.
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _check_torch(device)
    dev = torch.device(device)

    # Resolve views
    all_views = ["axial", "coronal", "sagittal"]
    if views in ("all", "axial,coronal,sagittal"):
        view_list = all_views
    else:
        view_list = [v.strip() for v in views.split(",")]

    _valid_views = set(all_views)
    unknown = [v for v in view_list if v not in _valid_views]
    if unknown:
        raise ValueError(
            f"Unknown view(s) {unknown!r}; valid choices are {all_views}"
        )

    # Resolve backend
    use_gsplat = (device == "cuda") and _gsplat_available()
    backend = "gsplat" if use_gsplat else "fallback"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load volume
    data, affine = _load_canonical(Path(volume_path))
    norm = _robust_normalize(data)

    roi_data: Optional[np.ndarray] = None
    if roi_path is not None and Path(roi_path).is_file():
        roi_data, _ = _load_canonical(Path(roi_path))

    if volume_loss_weight < 0:
        raise ValueError("volume_loss_weight must be non-negative.")

    # Build target projections
    targets = _build_targets(norm, image_size, view_list)

    # Initialize parameters
    init = _init_gaussians(norm, affine, roi_data, num_gaussians, seed, jitter_voxels=jitter_voxels)
    N = len(init["means_norm"])

    volume_samples: Optional[dict[str, np.ndarray | int | float]] = None
    if volume_loss_weight > 0.0:
        volume_samples = _build_volume_samples(
            norm=norm,
            affine=affine,
            norm_center=init["norm_center"],
            norm_scale=float(init["norm_scale"]),
            sample_count=volume_sample_count,
            seed=seed + 17,
        )

    means_p = torch.nn.Parameter(
        torch.tensor(init["means_norm"], dtype=torch.float32, device=dev)
    )
    log_scales_p = torch.nn.Parameter(
        torch.tensor(init["log_scales"], dtype=torch.float32, device=dev)
    )
    quat_p = torch.nn.Parameter(
        torch.tensor(init["quat_wxyz"], dtype=torch.float32, device=dev)
    )
    opacity_logit_p = torch.nn.Parameter(
        torch.tensor(init["opacity_logit"], dtype=torch.float32, device=dev)
    )
    colors_raw_p = torch.nn.Parameter(
        torch.tensor(init["colors_raw"], dtype=torch.float32, device=dev)
    )

    params = [means_p, log_scales_p, quat_p, opacity_logit_p, colors_raw_p]
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Target tensors
    target_tensors = {
        v: torch.tensor(targets[v], dtype=torch.float32, device=dev).unsqueeze(-1).expand(-1, -1, 3)
        for v in view_list
    }
    volume_sample_points_t = None
    volume_sample_targets_t = None
    if volume_samples is not None:
        volume_sample_points_t = torch.tensor(
            volume_samples["sample_points_norm"], dtype=torch.float32, device=dev
        )
        volume_sample_targets_t = torch.tensor(
            volume_samples["targets"], dtype=torch.float32, device=dev
        )

    loss_history: list[float] = []
    mip_loss_history: list[float] = []
    volume_loss_history: list[float] = []
    preview_frames: list[dict] = []

    for step in range(steps):
        optimizer.zero_grad()
        mip_loss = torch.zeros(1, device=dev)

        for view in view_list:
            if use_gsplat:
                rendered = _gsplat_render(
                    means_p, log_scales_p, quat_p, opacity_logit_p, colors_raw_p,
                    image_size, view,
                )
            else:
                rendered = _fallback_render(
                    means_p, log_scales_p, opacity_logit_p, colors_raw_p,
                    image_size, view,
                )
            tgt = target_tensors[view]
            mip_loss = mip_loss + F.mse_loss(rendered, tgt)

        mip_loss = mip_loss / len(view_list)
        total_loss = mip_loss
        volume_loss = None
        if volume_sample_points_t is not None and volume_sample_targets_t is not None:
            volume_pred = _render_volume_samples(
                means_p,
                log_scales_p,
                opacity_logit_p,
                colors_raw_p,
                volume_sample_points_t,
            )
            volume_loss = F.mse_loss(volume_pred, volume_sample_targets_t)
            total_loss = total_loss + float(volume_loss_weight) * volume_loss

        total_loss.backward()
        optimizer.step()

        loss_val = float(total_loss.item())
        loss_history.append(loss_val)
        mip_loss_history.append(float(mip_loss.item()))
        if volume_loss is not None:
            volume_loss_history.append(float(volume_loss.item()))

        if preview_every and step % preview_every == 0:
            preview_frames.append({"step": step, "loss": loss_val})

    # Final render for preview
    final_renders: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for view in view_list:
            if use_gsplat:
                r = _gsplat_render(
                    means_p, log_scales_p, quat_p, opacity_logit_p, colors_raw_p,
                    image_size, view,
                )
            else:
                r = _fallback_render(
                    means_p, log_scales_p, opacity_logit_p, colors_raw_p,
                    image_size, view,
                )
            final_renders[view] = r.cpu().numpy()

    # Save gaussian_field.pt
    pt_path = output_dir / "gaussian_field.pt"
    means_np = means_p.detach().cpu().numpy()
    log_scales_np = log_scales_p.detach().cpu().numpy()
    quat_np = quat_p.detach().cpu().numpy()
    opacity_np = opacity_logit_p.detach().cpu().numpy()
    colors_np = colors_raw_p.detach().cpu().numpy()
    support_fingerprint = gaussian_support_fingerprint(
        means_np,
        log_scales_np,
        quat_np,
        init["norm_center"],
        float(init["norm_scale"]),
    )
    torch.save({
        "means_norm": means_p.detach().cpu(),
        "log_scales": log_scales_p.detach().cpu(),
        "quat_wxyz": quat_p.detach().cpu(),
        "opacity_logit": opacity_logit_p.detach().cpu(),
        "colors_raw": colors_raw_p.detach().cpu(),
        "norm_center": init["norm_center"],
        "norm_scale": float(init["norm_scale"]),
        "support_fingerprint_schema": GAUSSIAN_SUPPORT_FINGERPRINT_SCHEMA,
        "support_fingerprint_sha256": support_fingerprint,
        "affine": affine,
        "num_gaussians": N,
        "backend": backend,
        "device": str(device),
        "seed": seed,
        "jitter_voxels": jitter_voxels,
        "training_objective": (
            "mip_projection_mse_plus_volume_sample_mse"
            if volume_samples is not None else "mip_projection_mse"
        ),
        "volume_loss_weight": float(volume_loss_weight),
        "volume_sample_count": int(len(volume_samples["targets"])) if volume_samples else 0,
        "volume_sample_count_requested": int(volume_sample_count),
        "volume_sample_foreground_count": (
            int(volume_samples["foreground_count"]) if volume_samples else 0
        ),
        "volume_sample_background_count": (
            int(volume_samples["background_count"]) if volume_samples else 0
        ),
        "volume_path": str(volume_path),
        "roi_path": str(roi_path) if roi_path else None,
    }, str(pt_path))

    # Save PLY
    ply_path = output_dir / "optimized_gaussian_splat.ply"
    _save_ply(
        means_np,
        log_scales_np,
        quat_np,
        opacity_np,
        colors_np,
        init["norm_center"],
        float(init["norm_scale"]),
        affine,
        ply_path,
    )

    # ROI mask
    lesion_metrics: dict = {}
    if roi_data is not None:
        is_roi = init["is_roi"]
        roi_n = int(is_roi.sum())
        roi_frac = float(roi_n / N) if N > 0 else 0.0
        mask_path = output_dir / "lesion_gaussian_mask.npy"
        np.save(str(mask_path), is_roi)
        lesion_metrics = {
            "lesion_gaussian_count": roi_n,
            "lesion_gaussian_fraction": roi_frac,
        }

    # Preview PNG
    preview_path = output_dir / "reconstruction_preview.png"
    _save_preview(targets, final_renders, view_list, preview_path)

    # Metrics
    metrics = {
        "volume_path": str(volume_path),
        "roi_path": str(roi_path) if roi_path else None,
        "num_gaussians": N,
        "steps": steps,
        "image_size": image_size,
        "views": view_list,
        "learning_rate": learning_rate,
        "device": str(device),
        "backend": backend,
        "support_fingerprint_schema": GAUSSIAN_SUPPORT_FINGERPRINT_SCHEMA,
        "support_fingerprint_sha256": support_fingerprint,
        "seed": seed,
        "jitter_voxels": jitter_voxels,
        "training_objective": (
            "mip_projection_mse_plus_volume_sample_mse"
            if volume_samples is not None else "mip_projection_mse"
        ),
        "volume_loss_weight": float(volume_loss_weight),
        "volume_sample_count": int(len(volume_samples["targets"])) if volume_samples else 0,
        "volume_sample_count_requested": int(volume_sample_count),
        "volume_sample_foreground_count": (
            int(volume_samples["foreground_count"]) if volume_samples else 0
        ),
        "volume_sample_background_count": (
            int(volume_samples["background_count"]) if volume_samples else 0
        ),
        "volume_sample_foreground_fraction": (
            float(volume_samples["foreground_fraction"]) if volume_samples else 0.0
        ),
        "volume_sample_foreground_threshold": (
            float(volume_samples["foreground_threshold"]) if volume_samples else None
        ),
        "loss_history": loss_history,
        "mip_loss_history": mip_loss_history,
        "volume_loss_history": volume_loss_history,
        "final_loss": loss_history[-1] if loss_history else None,
        "final_mip_loss": mip_loss_history[-1] if mip_loss_history else None,
        "final_volume_loss": volume_loss_history[-1] if volume_loss_history else None,
        **lesion_metrics,
    }
    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return metrics


# ---------------------------------------------------------------------------
# PLY export
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


def _save_ply(
    means_norm: np.ndarray,
    log_scales: np.ndarray,
    quat_wxyz: np.ndarray,
    opacity_logit: np.ndarray,
    colors_raw: np.ndarray,
    norm_center: np.ndarray,
    norm_scale: float,
    affine: np.ndarray,
    path: Path,
) -> None:
    N = len(means_norm)
    # Denormalize means back to world (mm) space
    world = means_norm * norm_scale + norm_center

    scales = np.exp(log_scales).clip(1e-6, None)  # (N, 3)
    # Denormalize scales to world units
    world_scales = scales * norm_scale

    opacities = 1.0 / (1.0 + np.exp(-opacity_logit))
    colors = (np.tanh(colors_raw) + 1.0) / 2.0
    rgb = np.clip(colors * 255, 0, 255).astype(np.uint8)
    quats = quat_wxyz.astype(np.float32, copy=True)
    q_norm = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.clip(q_norm, 1e-8, None)

    arr = np.zeros(N, dtype=_PLY_DTYPE)
    arr["x"] = world[:, 0]
    arr["y"] = world[:, 1]
    arr["z"] = world[:, 2]
    arr["red"] = rgb[:, 0]
    arr["green"] = rgb[:, 1]
    arr["blue"] = rgb[:, 2]
    arr["opacity"] = opacities.astype(np.float32)
    arr["scale_x"] = world_scales[:, 0]
    arr["scale_y"] = world_scales[:, 1]
    arr["scale_z"] = world_scales[:, 2]
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
# Preview PNG
# ---------------------------------------------------------------------------

def _save_preview(
    targets: dict[str, np.ndarray],
    renders: dict[str, np.ndarray],
    view_list: list[str],
    path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_views = len(view_list)
    fig, axes = plt.subplots(2, n_views, figsize=(4 * n_views, 8))
    if n_views == 1:
        axes = axes.reshape(2, 1)

    for col, view in enumerate(view_list):
        tgt = targets[view]
        ren = renders[view]
        if ren.shape[-1] == 3:
            ren_gray = ren.mean(axis=-1)
        else:
            ren_gray = ren
        axes[0, col].imshow(tgt.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[0, col].set_title(f"Target {view}")
        axes[0, col].axis("off")
        axes[1, col].imshow(ren_gray.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[1, col].set_title(f"Rendered {view}")
        axes[1, col].axis("off")

    fig.tight_layout()
    fig.savefig(str(path), dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "MRI → optimized Gaussian field training. "
            "Initializes 3D Gaussians from foreground voxels and optimizes "
            "against orthographic MRI projections."
        )
    )
    p.add_argument("--volume", type=Path, required=True,
                   help="Input NIfTI volume (.nii.gz)")
    p.add_argument("--roi", type=Path, default=None,
                   help="Optional ROI mask NIfTI")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output directory")
    p.add_argument("--num-gaussians", type=int, default=20_000)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--views", type=str, default="all",
                   help="Comma-separated views or 'all' (default: all)")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--preview-every", type=int, default=0,
                   help="Log preview frame every N steps (0=disabled)")
    p.add_argument("--no-jitter", action="store_true", default=False,
                   help="Disable sub-voxel jitter; place Gaussians exactly at voxel centers")
    p.add_argument("--volume-loss-weight", type=float, default=0.0,
                   help="Optional weight for sampled 3D voxel-space reconstruction loss.")
    p.add_argument("--volume-sample-count", type=int, default=4096,
                   help="Number of deterministic voxel samples for volume loss.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    _check_torch(args.device if args.device != "auto" else "cpu")

    metrics = train_gaussian_field(
        volume_path=args.volume,
        output_dir=args.output_dir,
        roi_path=args.roi,
        num_gaussians=args.num_gaussians,
        steps=args.steps,
        image_size=args.image_size,
        views=args.views,
        device=args.device,
        seed=args.seed,
        learning_rate=args.learning_rate,
        preview_every=args.preview_every,
        jitter_voxels=not args.no_jitter,
        volume_loss_weight=args.volume_loss_weight,
        volume_sample_count=args.volume_sample_count,
    )
    print(f"Training complete. Backend={metrics['backend']}  "
          f"N={metrics['num_gaussians']}  "
          f"final_loss={metrics['final_loss']:.6f}")


if __name__ == "__main__":
    main()
