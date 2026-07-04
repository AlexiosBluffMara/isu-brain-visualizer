"""Contact-sheet preview for a real ds004199 subject (T1w + FLAIR + ROI overlay).

Slice selection uses world-space coordinates so that T1w and FLAIR panels show
the same anatomical location even if they have different voxel grids.

Priority order for choosing the focus world coordinate:
  1. ROI centroid (if ROI mask exists and is non-empty)
  2. Foreground intensity centroid of the first available volume
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from fcd_prototype.nifti_preview import _robust_normalize


def _load_canonical(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (data float64, canonical affine) for a NIfTI file."""
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float64)
    if data.ndim == 4:
        data = data[..., 0]
    return data, img.affine.copy()


def _world_to_voxel(world_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Map a 3-element world coordinate (mm) to nearest voxel indices."""
    inv = np.linalg.inv(affine)
    hom = np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0])
    vox = (inv @ hom)[:3]
    return vox


def _clamp_voxel(vox: np.ndarray, shape: tuple[int, ...]) -> tuple[int, int, int]:
    ci = int(np.clip(round(vox[0]), 0, shape[0] - 1))
    cj = int(np.clip(round(vox[1]), 0, shape[1] - 1))
    ck = int(np.clip(round(vox[2]), 0, shape[2] - 1))
    return ci, cj, ck


def _roi_world_centroid(roi_data: np.ndarray, roi_affine: np.ndarray) -> np.ndarray:
    """Return world-space centroid of non-zero ROI voxels."""
    ijk = np.argwhere(roi_data > 0).astype(np.float64)
    centroid_vox = ijk.mean(axis=0)
    hom = np.array([centroid_vox[0], centroid_vox[1], centroid_vox[2], 1.0])
    return (roi_affine @ hom)[:3]


def _foreground_world_centroid(data: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Return world-space centroid of voxels above the 50th percentile (robust foreground)."""
    threshold = float(np.percentile(data[data > 0], 50)) if (data > 0).any() else 0.0
    ijk = np.argwhere(data > threshold).astype(np.float64)
    if len(ijk) == 0:
        ijk = np.array([[s / 2 for s in data.shape]])
    centroid_vox = ijk.mean(axis=0)
    hom = np.array([centroid_vox[0], centroid_vox[1], centroid_vox[2], 1.0])
    return (affine @ hom)[:3]


def _voxel_aspect(affine: np.ndarray, ax1: int, ax2: int) -> float:
    """Physical aspect ratio (height/width) for a 2D slice plane."""
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    h = float(spacing[ax2])
    w = float(spacing[ax1])
    return h / w if w > 0 else 1.0


def _roi_contour_overlay(ax, roi_slice: np.ndarray, alpha: float = 0.35) -> None:
    mask = roi_slice > 0
    if not mask.any():
        return
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., 0] = 1.0
    rgba[..., 3] = mask.astype(np.float32) * alpha
    ax.imshow(rgba, origin="lower", aspect="auto")


def _save_plane(
    fig: plt.Figure,
    axes: list[plt.Axes],
    volumes: list[tuple[str, np.ndarray, np.ndarray]],  # (label, data, affine)
    roi_vol: Optional[tuple[np.ndarray, np.ndarray]],   # (data, affine) or None
    plane: str,
    world_xyz: np.ndarray,
) -> None:
    for ax, (label, vol, vol_affine) in zip(axes, volumes):
        cx, cy, cz = _clamp_voxel(_world_to_voxel(world_xyz, vol_affine), vol.shape)

        if plane == "axial":
            sl = vol[:, :, cz].T
            aspect = _voxel_aspect(vol_affine, 0, 1)
        elif plane == "coronal":
            sl = vol[:, cy, :].T
            aspect = _voxel_aspect(vol_affine, 0, 2)
        else:
            sl = vol[cx, :, :].T
            aspect = _voxel_aspect(vol_affine, 1, 2)

        norm = _robust_normalize(sl)
        ax.imshow(norm, cmap="gray", origin="lower", aspect=aspect, vmin=0, vmax=1)

        if roi_vol is not None:
            roi_data, roi_affine = roi_vol
            rcx, rcy, rcz = _clamp_voxel(
                _world_to_voxel(world_xyz, roi_affine), roi_data.shape
            )
            if plane == "axial":
                roi_sl = roi_data[:, :, rcz].T
            elif plane == "coronal":
                roi_sl = roi_data[:, rcy, :].T
            else:
                roi_sl = roi_data[rcx, :, :].T
            _roi_contour_overlay(ax, roi_sl)

        ax.set_title(label, fontsize=9, pad=3)
        ax.axis("off")


def save_real_preview(
    t1w_path: Optional[Path],
    flair_path: Optional[Path],
    roi_path: Optional[Path],
    output_dir: Path,
    subject: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    volumes: list[tuple[str, np.ndarray, np.ndarray]] = []
    if t1w_path is not None and t1w_path.is_file():
        d, a = _load_canonical(t1w_path)
        volumes.append(("T1w", d, a))
    if flair_path is not None and flair_path.is_file():
        d, a = _load_canonical(flair_path)
        volumes.append(("FLAIR", d, a))

    if not volumes:
        raise FileNotFoundError(f"No T1w or FLAIR found for {subject}")

    roi_tuple: Optional[tuple[np.ndarray, np.ndarray]] = None
    if roi_path is not None and roi_path.is_file():
        roi_d, roi_a = _load_canonical(roi_path)
        roi_tuple = (roi_d, roi_a)

    # Choose focus world coordinate
    if roi_tuple is not None and (roi_tuple[0] > 0).any():
        world_xyz = _roi_world_centroid(roi_tuple[0], roi_tuple[1])
    else:
        first_data, first_affine = volumes[0][1], volumes[0][2]
        world_xyz = _foreground_world_centroid(first_data, first_affine)

    n_vols = len(volumes)
    saved: dict[str, Path] = {}

    for plane in ("axial", "coronal", "sagittal"):
        fig, axes = plt.subplots(
            1, n_vols, figsize=(4 * n_vols, 4.5), dpi=120, constrained_layout=True
        )
        if n_vols == 1:
            axes = [axes]
        else:
            axes = list(axes)

        fig.suptitle(f"{subject} — {plane}", fontsize=11)
        _save_plane(fig, axes, volumes, roi_tuple, plane, world_xyz)

        out_path = output_dir / f"{subject}_{plane}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        saved[plane] = out_path
        print(f"Saved: {out_path}")

    return saved


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate contact-sheet previews for a real ds004199 subject."
    )
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--subject", required=True, help="Subject ID, e.g. sub-00058.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/real_ds004199"),
    )
    p.add_argument("--t1w", type=Path, default=None)
    p.add_argument("--flair", type=Path, default=None)
    p.add_argument("--roi", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    from fcd_prototype.bids_discovery import discover_subject

    rec = discover_subject(args.dataset_dir, args.subject)

    t1w = args.t1w or rec.t1w
    flair = args.flair or rec.flair
    roi = args.roi or rec.flair_roi

    out_dir = args.output_dir / args.subject
    save_real_preview(
        t1w_path=t1w,
        flair_path=flair,
        roi_path=roi,
        output_dir=out_dir,
        subject=args.subject,
    )


if __name__ == "__main__":
    main()
