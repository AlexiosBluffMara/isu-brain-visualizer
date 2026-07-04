"""Convert a NIfTI volume into a Gaussian seed state for Volumetric 3DGS.

This is NOT the full optimizer — it produces the native seed arrays
(positions, scales, rotations, opacity, intensity) needed by the 3DGS phase.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib


def _load_canonical(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (data float32, affine 4x4) in RAS canonical orientation."""
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data, img.affine.astype(np.float64)


def _voxel_to_world(voxel_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Map (N, 3) voxel coordinates to (N, 3) world (mm) coordinates."""
    ones = np.ones((len(voxel_coords), 1), dtype=np.float64)
    hom = np.hstack([voxel_coords.astype(np.float64), ones])
    world = (affine @ hom.T).T[:, :3]
    return world.astype(np.float32)


def _voxel_scales(affine: np.ndarray, n: int) -> np.ndarray:
    """Isotropic scale = voxel spacing (mm), replicated for all points."""
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).astype(np.float32)
    return np.tile(spacing, (n, 1))


def build_seed(
    volume_path: Path,
    roi_path: Optional[Path] = None,
    max_points: int = 100_000,
    percentile_threshold: float = 5.0,
    rng_seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Sample a sparse set of Gaussian seeds from a NIfTI volume.

    Returns a dict with keys:
        positions_world  (N, 3) float32  — mm coordinates
        positions_voxel  (N, 3) float32  — voxel coordinates
        scales           (N, 3) float32  — voxel spacing (isotropic)
        rotations_quat   (N, 4) float32  — identity quaternions [w, x, y, z]
        opacity          (N,)   float32  — normalised intensity in [0, 1]
        intensity        (N,)   float32  — raw voxel value
        affine           (4, 4) float64  — NIfTI affine
        source_shape     (3,)   int64    — original volume shape
        is_roi           (N,)   bool     — True if voxel is inside the ROI mask
    """
    rng = np.random.default_rng(rng_seed)

    data, affine = _load_canonical(volume_path)
    source_shape = np.array(data.shape[:3], dtype=np.int64)

    roi_data: Optional[np.ndarray] = None
    if roi_path is not None and roi_path.is_file():
        roi_data, _ = _load_canonical(roi_path)

    # Threshold: ignore very low intensity (background / air)
    threshold = np.percentile(data[data > 0], percentile_threshold) if (data > 0).any() else 0.0
    foreground_mask = data > threshold

    # All foreground voxel indices
    fg_idx = np.argwhere(foreground_mask)

    if len(fg_idx) == 0:
        raise ValueError(f"No foreground voxels found above {percentile_threshold}th percentile in {volume_path}")

    # Prioritize ROI voxels
    is_roi_full = np.zeros(len(fg_idx), dtype=bool)
    if roi_data is not None:
        roi_fg = roi_data[foreground_mask] > 0
        is_roi_full = roi_fg

    roi_count = int(is_roi_full.sum())
    non_roi_count = len(fg_idx) - roi_count

    if len(fg_idx) <= max_points:
        selected = np.arange(len(fg_idx))
    else:
        # Reserve up to half the budget for ROI voxels (if any)
        roi_budget = min(roi_count, max_points // 2) if roi_count > 0 else 0
        non_roi_budget = max_points - roi_budget

        roi_indices = np.where(is_roi_full)[0]
        non_roi_indices = np.where(~is_roi_full)[0]

        sel_roi = rng.choice(roi_indices, size=min(roi_budget, len(roi_indices)), replace=False)
        sel_non_roi = rng.choice(non_roi_indices, size=min(non_roi_budget, len(non_roi_indices)), replace=False)
        selected = np.concatenate([sel_roi, sel_non_roi])

    vox = fg_idx[selected].astype(np.float32)
    intensities = data[foreground_mask][selected]
    is_roi = is_roi_full[selected]

    # Normalise intensity to [0, 1]
    i_min, i_max = intensities.min(), intensities.max()
    if i_max > i_min:
        opacity = ((intensities - i_min) / (i_max - i_min)).astype(np.float32)
    else:
        opacity = np.ones(len(selected), dtype=np.float32)

    positions_world = _voxel_to_world(vox, affine)
    scales = _voxel_scales(affine, len(selected))
    # Identity quaternion: w=1, x=0, y=0, z=0
    rotations_quat = np.zeros((len(selected), 4), dtype=np.float32)
    rotations_quat[:, 0] = 1.0

    return {
        "positions_world": positions_world,
        "positions_voxel": vox,
        "scales": scales,
        "rotations_quat": rotations_quat,
        "opacity": opacity,
        "intensity": intensities,
        "affine": affine,
        "source_shape": source_shape,
        "is_roi": is_roi,
    }


def save_seed(seed: dict[str, np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **seed)
    n = len(seed["positions_world"])
    roi_count = int(seed["is_roi"].sum())
    print(f"Saved {n} Gaussian seeds ({roi_count} ROI) → {output_path}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert a NIfTI volume into a Gaussian seed state (.npz)."
    )
    p.add_argument(
        "volume",
        type=Path,
        help="Path to input NIfTI volume (.nii.gz).",
    )
    p.add_argument(
        "--roi",
        type=Path,
        default=None,
        help="Optional ROI mask NIfTI. ROI voxels are prioritised and labelled.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .npz file path (e.g. outputs/seeds/sub-00058_T1w_seed.npz).",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=100_000,
        help="Maximum number of Gaussian seeds to sample (default: 100000).",
    )
    p.add_argument(
        "--percentile-threshold",
        type=float,
        default=5.0,
        help="Background percentile to exclude (default: 5.0).",
    )
    p.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    seed = build_seed(
        volume_path=args.volume,
        roi_path=args.roi,
        max_points=args.max_points,
        percentile_threshold=args.percentile_threshold,
        rng_seed=args.rng_seed,
    )
    save_seed(seed, args.output)


if __name__ == "__main__":
    main()
