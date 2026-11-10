"""Product demo builder.

Exports viewer-ready NIfTI artifacts for a ds004199 subject into
outputs/product_demo/<subject>/ and writes manifest.json.

All outputs are CPU-safe and do not require GPU inference.
The Gaussian splat in the output is labelled as a CPU preview,
not the final optimized Volumetric 3DGS result.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from fcd_prototype.bids_discovery import discover_subject, load_participants
from fcd_prototype.gaussian_seed import build_seed, save_seed
from fcd_prototype.gaussian_splat import render_lesion_splat, render_splat
from fcd_prototype.nifti_preview import _robust_normalize


def _load_canonical_with_affine(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data, img.affine.copy(), img.header.copy()


def _save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header,
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data, affine, header)
    img.header.set_data_dtype(np.float32)
    nib.save(img, str(path))
    return path


def _normalize_volume(data: np.ndarray) -> np.ndarray:
    return _robust_normalize(data, low=1.0, high=99.0)


def _resample_nearest_to_ref(
    data: np.ndarray,
    src_affine: np.ndarray,
    ref_shape: tuple[int, ...],
    ref_affine: np.ndarray,
) -> np.ndarray:
    """Resample data from src_affine into ref_affine/ref_shape (nearest-neighbor)."""
    # Build voxel coordinates in ref space and map to src voxel space
    i, j, k = np.meshgrid(
        np.arange(ref_shape[0]),
        np.arange(ref_shape[1]),
        np.arange(ref_shape[2]),
        indexing="ij",
    )
    ref_vox = np.stack([i, j, k, np.ones_like(i)], axis=-1).reshape(-1, 4)

    # ref voxel → world → src voxel
    world = (ref_affine @ ref_vox.T).T[:, :3]
    src_inv = np.linalg.inv(src_affine)
    ones = np.ones((len(world), 1))
    world_h = np.hstack([world, ones])
    src_vox = (src_inv @ world_h.T).T[:, :3]

    src_ijk = np.round(src_vox).astype(np.int32).reshape(ref_shape + (3,))
    np.clip(src_ijk[..., 0], 0, data.shape[0] - 1, out=src_ijk[..., 0])
    np.clip(src_ijk[..., 1], 0, data.shape[1] - 1, out=src_ijk[..., 1])
    np.clip(src_ijk[..., 2], 0, data.shape[2] - 1, out=src_ijk[..., 2])

    return data[src_ijk[..., 0], src_ijk[..., 1], src_ijk[..., 2]]


def _make_heatmap(roi_data: np.ndarray, smooth_sigma: float = 4.0) -> np.ndarray:
    roi_f = (roi_data > 0).astype(np.float32)
    heatmap = gaussian_filter(roi_f, sigma=smooth_sigma).astype(np.float32)
    h_max = heatmap.max()
    if h_max > 0:
        heatmap = heatmap / h_max
    return heatmap


def build_product_demo(
    dataset_dir: Path,
    subject: str,
    output_dir: Path,
    max_seed_points: int = 50_000,
    splat_max_points: Optional[int] = None,
    splat_sigma: float = 4.0,
    force: bool = False,
) -> dict:
    """Build viewer-ready artifacts for one subject.

    Returns the manifest dict.
    """
    out_subj = output_dir / subject
    out_subj.mkdir(parents=True, exist_ok=True)

    manifest_path = out_subj / "manifest.json"
    if manifest_path.exists() and not force:
        print(f"Manifest already exists at {manifest_path} — skipping (use --force to rebuild).")
        with manifest_path.open() as fh:
            return json.load(fh)

    participants_tsv = dataset_dir / "participants.tsv"
    if participants_tsv.is_file():
        known = load_participants(participants_tsv)
        base = known.get(subject)
        group = base.group if base else ""
        meta = base.metadata if base else {}
    else:
        group, meta = "", {}
    rec = discover_subject(dataset_dir, subject, group=group, metadata=meta)

    artifacts: dict[str, str] = {}
    notes: list[str] = []
    notes.append("CPU splat preview (lesion-focused) — ROI seed points only, NOT the final GPU 3DGS result.")
    notes.append("ROI heatmap is Gaussian-smoothed binary mask, not a diffusion/SDEdit baseline.")

    # --- Base volume (FLAIR preferred, T1w fallback) -----------------------
    base_path: Optional[Path] = rec.flair if rec.flair and rec.flair.is_file() else None
    t1w_path: Optional[Path] = rec.t1w if rec.t1w and rec.t1w.is_file() else None
    seed_source_path: Optional[Path] = None

    if base_path is None and t1w_path is None:
        raise FileNotFoundError(f"No T1w or FLAIR found for {subject} in {dataset_dir}")

    # base_shape/affine/header track whichever volume is used as the spatial reference
    base_shape: Optional[tuple] = None
    base_affine: Optional[np.ndarray] = None
    base_header: Optional[nib.Nifti1Header] = None

    # Export FLAIR
    if base_path is not None:
        flair_data, base_affine, base_header = _load_canonical_with_affine(base_path)
        base_shape = flair_data.shape
        flair_norm = _normalize_volume(flair_data)
        flair_out = out_subj / "patient_flair.nii.gz"
        _save_nifti(flair_norm, base_affine, base_header, flair_out)
        artifacts["patient_flair"] = flair_out.name
        seed_source_path = flair_out

    # Export T1w
    if t1w_path is not None:
        t1w_data, t1w_affine, t1w_header = _load_canonical_with_affine(t1w_path)
        t1w_norm = _normalize_volume(t1w_data)
        t1w_out = out_subj / "patient_t1w.nii.gz"
        _save_nifti(t1w_norm, t1w_affine, t1w_header, t1w_out)
        artifacts["patient_t1w"] = t1w_out.name
        if seed_source_path is None:
            seed_source_path = t1w_out
            base_shape = t1w_data.shape
            base_affine = t1w_affine
            base_header = t1w_header

    assert base_shape is not None and base_affine is not None and base_header is not None

    # --- ROI mask ----------------------------------------------------------
    roi_out_path: Optional[Path] = None

    if rec.flair_roi and rec.flair_roi.is_file():
        roi_raw, roi_affine_src, _ = _load_canonical_with_affine(rec.flair_roi)
        roi_bin = (roi_raw > 0).astype(np.float32)

        # Resample ROI to base space if shape or affine differs
        if roi_bin.shape != base_shape or not np.allclose(roi_affine_src, base_affine):
            roi_bin = _resample_nearest_to_ref(
                roi_bin, roi_affine_src, base_shape, base_affine
            )

        roi_out_path = out_subj / "lesion_roi.nii.gz"
        _save_nifti(roi_bin, base_affine, base_header, roi_out_path)
        artifacts["lesion_roi"] = roi_out_path.name

        # Heatmap
        heatmap = _make_heatmap(roi_bin)
        heatmap_out = out_subj / "lesion_heatmap.nii.gz"
        _save_nifti(heatmap, base_affine, base_header, heatmap_out)
        artifacts["lesion_heatmap"] = heatmap_out.name
    else:
        notes.append("No ROI mask found for this subject — lesion_roi and lesion_heatmap omitted.")

    # --- Gaussian seed (kept for future GPU 3DGS training, not exposed as default overlay)
    seed_out = out_subj / "gaussian_seed.npz"
    assert seed_source_path is not None
    seed = build_seed(
        volume_path=seed_source_path,
        roi_path=roi_out_path,
        max_points=max_seed_points,
        rng_seed=42,
    )
    save_seed(seed, seed_out)
    artifacts["gaussian_seed"] = seed_out.name

    # --- Lesion-focused CPU splat preview (product overlay) ----------------
    # Only generated when an ROI exists; uses ROI seed points + wide halo blur.
    # Kept under the "gaussian_splat" artifact key for viewer backward compat.
    if roi_out_path is not None:
        splat_out = out_subj / "gaussian_splat.nii.gz"
        render_lesion_splat(
            seed_path=seed_out,
            reference_path=seed_source_path,
            output_path=splat_out,
            heatmap_path=heatmap_out if "lesion_heatmap" in artifacts else None,
            sigma_vox=splat_sigma,
        )
        artifacts["gaussian_splat"] = splat_out.name
    else:
        notes.append("No lesion_splat generated — no ROI mask available for this subject.")

    # --- Manifest ----------------------------------------------------------
    manifest = {
        "subject": subject,
        "group": rec.group or "unknown",
        "artifacts": {k: f"{v}" for k, v in artifacts.items()},
        "generated": datetime.utcnow().isoformat() + "Z",
        "notes": notes,
    }
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Manifest written → {manifest_path}")

    return manifest


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build viewer-ready product demo artifacts for a ds004199 subject."
    )
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--subject", required=True, help="e.g. sub-00058")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/product_demo"),
    )
    p.add_argument("--max-seed-points", type=int, default=50_000)
    p.add_argument("--splat-max-points", type=int, default=None)
    p.add_argument("--splat-sigma", type=float, default=4.0)
    p.add_argument("--force", action="store_true", help="Rebuild even if manifest exists.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    build_product_demo(
        dataset_dir=args.dataset_dir,
        subject=args.subject,
        output_dir=args.output_dir,
        max_seed_points=args.max_seed_points,
        splat_max_points=args.splat_max_points,
        splat_sigma=args.splat_sigma,
        force=args.force,
    )


if __name__ == "__main__":
    main()
