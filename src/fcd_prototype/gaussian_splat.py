"""CPU Volumetric Gaussian Splatting preview renderer.

CPU PREVIEW ONLY — not the final GPU optimizer.  Scatters seed points into a
dense grid and applies Gaussian blur to approximate splatting.  Output is
labelled as a preview in the manifest; the actual optimized splat requires the
GPU 3DGS phase (planned, not yet implemented).

Two public functions:
  render_splat()        — whole-seed scatter+blur (for GPU seed validation / tests)
  render_lesion_splat() — ROI-points-only scatter+blur (product-facing overlay)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


def render_lesion_splat(
    seed_path: Path,
    reference_path: Path,
    output_path: Path,
    heatmap_path: Optional[Path] = None,
    sigma_vox: float = 4.0,
) -> Path:
    """Render a lesion-focused splat using only ROI-flagged seed points.

    Filters the seed to is_roi==True entries and applies a Gaussian blur
    (default sigma=4.0 vox) to produce a smooth halo around the known lesion
    rather than scatter across the whole brain.

    Fallback: if the seed contains no ROI-marked points, the heatmap NIfTI is
    used directly (when provided), otherwise a zero volume is saved.

    CPU PREVIEW ONLY — not the final GPU optimizer.
    """
    seed = dict(np.load(str(seed_path), allow_pickle=False))
    positions_voxel: np.ndarray = seed["positions_voxel"].astype(np.float32)
    opacity: np.ndarray = seed["opacity"].astype(np.float32)
    intensity: np.ndarray = seed["intensity"].astype(np.float32)

    # Keep only ROI-flagged points; fall through to heatmap if none available
    is_roi = seed.get("is_roi")
    if is_roi is not None and np.asarray(is_roi, dtype=bool).any():
        roi_mask = np.asarray(is_roi, dtype=bool)
        positions_voxel = positions_voxel[roi_mask]
        opacity = opacity[roi_mask]
        intensity = intensity[roi_mask]
    else:
        positions_voxel = positions_voxel[:0]
        opacity = opacity[:0]
        intensity = intensity[:0]

    ref_img = nib.load(str(reference_path))
    ref_img = nib.as_closest_canonical(ref_img)
    ref_data = ref_img.get_fdata(dtype=np.float32)
    if ref_data.ndim == 4:
        ref_data = ref_data[..., 0]
    ref_shape: tuple[int, int, int] = ref_data.shape[:3]  # type: ignore[assignment]
    affine = ref_img.affine.copy()
    header = ref_img.header.copy()

    n_roi = len(positions_voxel)

    if n_roi == 0:
        # No ROI seed points — use heatmap directly when available
        if heatmap_path is not None:
            hm_img = nib.load(str(heatmap_path))
            hm_img = nib.as_closest_canonical(hm_img)
            grid = hm_img.get_fdata(dtype=np.float32)
            if grid.ndim == 4:
                grid = grid[..., 0]
        else:
            grid = np.zeros(ref_shape, dtype=np.float32)
        print(
            f"Lesion splat (heatmap fallback) → {output_path}  "
            f"shape={ref_shape}  roi_points=0"
        )
    else:
        weighted = (intensity * opacity).astype(np.float32)
        grid = np.zeros(ref_shape, dtype=np.float32)
        ijk = np.round(positions_voxel).astype(np.int32)

        valid = (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < ref_shape[0])
            & (ijk[:, 1] >= 0) & (ijk[:, 1] < ref_shape[1])
            & (ijk[:, 2] >= 0) & (ijk[:, 2] < ref_shape[2])
        )
        ijk_v = ijk[valid]
        w_v = weighted[valid]
        if len(ijk_v) > 0:
            np.add.at(grid, (ijk_v[:, 0], ijk_v[:, 1], ijk_v[:, 2]), w_v)

        if sigma_vox > 0:
            grid = gaussian_filter(grid, sigma=sigma_vox).astype(np.float32)

        print(
            f"Lesion splat preview → {output_path}  "
            f"shape={ref_shape}  roi_points={int(valid.sum())}  sigma={sigma_vox}"
        )

    g_max = float(grid.max())
    if g_max > 0:
        grid = (grid / g_max).astype(np.float32)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(grid, affine, header)
    out_img.header.set_data_dtype(np.float32)
    out_img.header["cal_min"] = 0.0
    out_img.header["cal_max"] = 1.0
    nib.save(out_img, str(output_path))
    return output_path


def render_splat(
    seed_path: Path,
    reference_path: Path,
    output_path: Path,
    max_points: Optional[int] = None,
    sigma_vox: float = 1.5,
) -> Path:
    """Render an approximate Gaussian splat volume from a seed .npz file.

    Parameters
    ----------
    seed_path:       .npz seed produced by gaussian_seed.build_seed
    reference_path:  NIfTI whose canonical shape/affine/header the output inherits
    output_path:     Destination .nii.gz path
    max_points:      Cap on seed points used (default: all)
    sigma_vox:       Gaussian blur sigma in voxels to approximate splatting

    Returns the output path.
    """
    seed = dict(np.load(str(seed_path), allow_pickle=False))
    positions_voxel: np.ndarray = seed["positions_voxel"].astype(np.float32)
    opacity: np.ndarray = seed["opacity"].astype(np.float32)
    intensity: np.ndarray = seed["intensity"].astype(np.float32)

    ref_img = nib.load(str(reference_path))
    ref_img = nib.as_closest_canonical(ref_img)
    ref_data = ref_img.get_fdata(dtype=np.float32)
    if ref_data.ndim == 4:
        ref_data = ref_data[..., 0]
    ref_shape: tuple[int, int, int] = ref_data.shape[:3]  # type: ignore[assignment]
    affine = ref_img.affine.copy()
    header = ref_img.header.copy()

    if max_points is not None:
        positions_voxel = positions_voxel[:max_points]
        opacity = opacity[:max_points]
        intensity = intensity[:max_points]

    weighted = (intensity * opacity).astype(np.float32)

    grid = np.zeros(ref_shape, dtype=np.float32)
    ijk = np.round(positions_voxel).astype(np.int32)

    valid = (
        (ijk[:, 0] >= 0) & (ijk[:, 0] < ref_shape[0])
        & (ijk[:, 1] >= 0) & (ijk[:, 1] < ref_shape[1])
        & (ijk[:, 2] >= 0) & (ijk[:, 2] < ref_shape[2])
    )
    ijk_v = ijk[valid]
    w_v = weighted[valid]

    if len(ijk_v) > 0:
        np.add.at(grid, (ijk_v[:, 0], ijk_v[:, 1], ijk_v[:, 2]), w_v)

    if sigma_vox > 0:
        grid = gaussian_filter(grid, sigma=sigma_vox).astype(np.float32)

    g_max = float(grid.max())
    if g_max > 0:
        grid = (grid / g_max).astype(np.float32)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(grid, affine, header)
    out_img.header.set_data_dtype(np.float32)
    out_img.header["cal_min"] = 0.0
    out_img.header["cal_max"] = 1.0
    nib.save(out_img, str(output_path))
    print(
        f"CPU splat preview → {output_path}  "
        f"shape={ref_shape}  points={int(valid.sum())}  sigma={sigma_vox}"
    )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "CPU Gaussian splat preview renderer.  "
            "NOT the final GPU optimizer — approximate only."
        )
    )
    p.add_argument("seed", type=Path, help="Input .npz seed file.")
    p.add_argument("reference", type=Path, help="Reference NIfTI to inherit shape/affine.")
    p.add_argument("--output", type=Path, required=True, help="Output .nii.gz path.")
    p.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Cap on seed points used (default: all).",
    )
    p.add_argument(
        "--sigma-vox",
        type=float,
        default=1.5,
        help="Gaussian blur sigma in voxels (default: 1.5).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    render_splat(
        seed_path=args.seed,
        reference_path=args.reference,
        output_path=args.output,
        max_points=args.max_points,
        sigma_vox=args.sigma_vox,
    )


if __name__ == "__main__":
    main()
