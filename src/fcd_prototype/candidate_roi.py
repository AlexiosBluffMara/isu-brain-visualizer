"""Heuristic candidate ROI proposal for ROI-free uploaded MRI volumes.

This module provides a purely MRI-signal-driven proposal mechanism.  It is
NOT a validated FCD detector; it is a bridge that lets the downstream fuzzing
and NVIDIA/3DGS residual pipeline operate on subjects that have no manually
delineated lesion mask.

Entry point: fcd-propose-roi
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, label as nd_label

from fcd_prototype.nifti_preview import _robust_normalize


PROPOSAL_MODES = ("brightness-gradient", "asymmetry-hybrid")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_canonical(path: Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data, img.affine.copy(), img.header.copy()


def _save_nifti(data: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(np.float32), affine, header)
    img.header.set_data_dtype(np.float32)
    nib.save(img, str(path))


def _voxel_size_mm(affine: np.ndarray) -> np.ndarray:
    """Return per-axis voxel sizes in mm from the affine."""
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def _foreground_mask(data: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Simple intensity threshold foreground mask (brain vs. background)."""
    norm = _robust_normalize(data)
    rough = norm > threshold
    # Keep only the largest connected component
    labeled, n = nd_label(rough)
    if n == 0:
        return rough
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax()).astype(bool)


def _norm01(arr: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    fg_vals = arr[fg_mask]
    if fg_vals.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = float(fg_vals.min()), float(fg_vals.max())
    if hi == lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    out[~fg_mask] = 0.0
    return out


def _left_right_asymmetry_map(norm: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """Compare each canonical voxel with its mirrored left-right homolog."""
    mirrored_norm = norm[::-1, :, :]
    mirrored_fg = fg_mask[::-1, :, :]
    paired_fg = fg_mask & mirrored_fg
    asym = np.abs(norm - mirrored_norm).astype(np.float32)
    asym[~paired_fg] = 0.0
    return _norm01(asym, fg_mask)


def _proposal_map(
    data: np.ndarray,
    fg_mask: np.ndarray,
    voxel_mm: np.ndarray,
    smooth_sigma_mm: float = 3.0,
) -> np.ndarray:
    """Compute a scalar proposal map highlighting locally bright / edge voxels.

    ``brightness-gradient`` is the legacy heuristic. ``asymmetry-hybrid`` adds
    canonical left-right signal difference, which is useful for unilateral
    abnormalities where absolute FLAIR brightness is subtle.
    """
    proposal, _meta = _proposal_map_with_mode(
        data,
        fg_mask,
        voxel_mm,
        smooth_sigma_mm=smooth_sigma_mm,
        proposal_mode="brightness-gradient",
    )
    return proposal


def _proposal_map_with_mode(
    data: np.ndarray,
    fg_mask: np.ndarray,
    voxel_mm: np.ndarray,
    *,
    smooth_sigma_mm: float = 3.0,
    proposal_mode: str = "brightness-gradient",
) -> tuple[np.ndarray, dict[str, object]]:
    if proposal_mode not in PROPOSAL_MODES:
        raise ValueError(
            "proposal_mode must be one of "
            f"{', '.join(PROPOSAL_MODES)}, got {proposal_mode!r}"
        )

    norm = _robust_normalize(data)
    norm_fg = norm * fg_mask

    # Gradient magnitude in physical space (account for voxel size)
    gz = np.gradient(norm_fg, voxel_mm[2], axis=2)
    gy = np.gradient(norm_fg, voxel_mm[1], axis=1)
    gx = np.gradient(norm_fg, voxel_mm[0], axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    bright = _norm01(norm_fg, fg_mask)
    grad = _norm01(grad_mag, fg_mask)
    asym = _left_right_asymmetry_map(norm, fg_mask)

    if proposal_mode == "brightness-gradient":
        weights = {
            "brightness": 0.60,
            "gradient": 0.40,
            "left_right_asymmetry": 0.0,
        }
    else:
        weights = {
            "brightness": 0.35,
            "gradient": 0.30,
            "left_right_asymmetry": 0.35,
        }
    combined = (
        weights["brightness"] * bright
        + weights["gradient"] * grad
        + weights["left_right_asymmetry"] * asym
    )

    # Smooth with Gaussian kernel sized in voxels from mm
    sigma_vox = smooth_sigma_mm / voxel_mm
    smoothed = gaussian_filter(combined, sigma=sigma_vox)
    proposal = (smoothed * fg_mask).astype(np.float32)
    fg_values = proposal[fg_mask]
    asym_values = asym[fg_mask]
    meta = {
        "proposal_mode": proposal_mode,
        "feature_weights": weights,
        "feature_summary": {
            "foreground_voxels": int(fg_mask.sum()),
            "proposal_max": float(fg_values.max()) if fg_values.size else 0.0,
            "proposal_mean": float(fg_values.mean()) if fg_values.size else 0.0,
            "left_right_asymmetry_max": (
                float(asym_values.max()) if asym_values.size else 0.0
            ),
            "left_right_asymmetry_mean": (
                float(asym_values.mean()) if asym_values.size else 0.0
            ),
        },
    }
    return proposal, meta


def _mm_to_vox(center_mm: np.ndarray, affine: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(affine)
    h = np.append(center_mm, 1.0)
    return (inv @ h)[:3]


def _vox_to_mm(center_vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    h = np.append(center_vox, 1.0)
    return (affine @ h)[:3]


def _sphere_mask(shape: tuple, center_vox: np.ndarray, radius_vox: np.ndarray) -> np.ndarray:
    """Boolean mask of an ellipsoidal sphere in voxel space."""
    i, j, k = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dist2 = (
        ((i - center_vox[0]) / radius_vox[0]) ** 2
        + ((j - center_vox[1]) / radius_vox[1]) ** 2
        + ((k - center_vox[2]) / radius_vox[2]) ** 2
    )
    return dist2 <= 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_candidate_rois(
    volume_path: Path | str,
    output_dir: Path | str,
    max_candidates: int = 5,
    radius_mm: float = 8.0,
    min_separation_mm: float = 16.0,
    percentile: float = 99.5,
    proposal_mode: str = "brightness-gradient",
    union_filename: str = "candidate_roi_union.nii.gz",
) -> dict:
    """Generate up to *max_candidates* spherical candidate ROI masks.

    Candidates are chosen as the top peaks of an MRI-signal heuristic map
    subject to a minimum mutual separation in world/mm space.

    Parameters
    ----------
    volume_path:
        Path to any normalised NIfTI brain volume (FLAIR or T1w).
    output_dir:
        Directory where candidate masks and the manifest are written.
    max_candidates:
        Maximum number of candidate spheres to select.
    radius_mm:
        Sphere radius in mm for each candidate mask.
    min_separation_mm:
        Minimum centre-to-centre distance in mm between any two candidates.
    percentile:
        Percentile of the proposal map used as the initial search threshold.
    proposal_mode:
        Proposal feature model. ``brightness-gradient`` preserves the legacy
        local signal heuristic. ``asymmetry-hybrid`` adds canonical left-right
        asymmetry to favor unilateral abnormalities.
    union_filename:
        Filename for the union candidate ROI NIfTI.

    Returns
    -------
    dict
        Manifest content (also written to *output_dir*/candidate_roi_manifest.json).
    """
    volume_path = Path(volume_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if proposal_mode not in PROPOSAL_MODES:
        raise ValueError(
            "proposal_mode must be one of "
            f"{', '.join(PROPOSAL_MODES)}, got {proposal_mode!r}"
        )

    data, affine, header = _load_canonical(volume_path)
    voxel_mm = _voxel_size_mm(affine)
    shape = data.shape[:3]

    fg_mask = _foreground_mask(data)
    proposal_meta: dict[str, object] = {
        "proposal_mode": proposal_mode,
        "feature_weights": {},
        "feature_summary": {"foreground_voxels": int(fg_mask.sum())},
    }
    if fg_mask.any():
        proposal, proposal_meta = _proposal_map_with_mode(
            data,
            fg_mask,
            voxel_mm,
            proposal_mode=proposal_mode,
        )
    else:
        proposal = np.zeros(shape, dtype=np.float32)

    # Radius in voxels per axis
    radius_vox = radius_mm / voxel_mm

    # Iteratively pick the highest proposal value, suppress neighbourhood,
    # repeat until we have max_candidates or run out of signal above threshold.
    suppression_radius_vox = min_separation_mm / voxel_mm
    suppressed = proposal.copy()
    threshold = np.percentile(proposal[fg_mask], percentile) if fg_mask.any() else 0.0

    centers_vox: list[np.ndarray] = []
    centers_world: list[np.ndarray] = []
    individual_filenames: list[str] = []
    candidate_masks: list[np.ndarray] = []

    union_mask = np.zeros(shape, dtype=np.float32)

    proposal_max = float(suppressed.max()) if suppressed.size else 0.0
    if not fg_mask.any() or proposal_max <= 0.0 or proposal_max < threshold:
        # Empty foreground or no signal above threshold: write empty union and manifest with 0 selected
        _save_nifti(union_mask, affine, header, output_dir / union_filename)
    else:
        for idx in range(max_candidates):
            current_max = float(suppressed.max()) if suppressed.size else 0.0
            if current_max <= 0.0 or current_max < threshold:
                break
            peak_flat = int(np.argmax(suppressed))
            peak_vox = np.array(np.unravel_index(peak_flat, shape), dtype=np.float64)

            # Suppress a sphere around the selected centre
            sup_mask = _sphere_mask(shape, peak_vox, suppression_radius_vox)
            suppressed[sup_mask] = 0.0

            # Build the spherical candidate mask at the candidate radius
            cand_mask = _sphere_mask(shape, peak_vox, radius_vox).astype(np.float32)
            cand_mask *= fg_mask  # constrain to brain tissue

            centers_vox.append(peak_vox)
            peak_world = _vox_to_mm(peak_vox, affine)
            centers_world.append(peak_world)
            candidate_masks.append(cand_mask)
            union_mask = np.clip(union_mask + cand_mask, 0, 1)

            fname = f"candidate_roi_{idx:02d}.nii.gz"
            _save_nifti(cand_mask, affine, header, output_dir / fname)
            individual_filenames.append(fname)

        # Write union mask
        union_path = output_dir / union_filename
        _save_nifti(union_mask, affine, header, union_path)

    manifest = {
        "volume_path": str(volume_path),
        "max_candidates": max_candidates,
        "radius_mm": radius_mm,
        "min_separation_mm": min_separation_mm,
        "percentile": percentile,
        "proposal_mode": proposal_mode,
        "proposal_feature_weights": proposal_meta.get("feature_weights", {}),
        "proposal_feature_summary": proposal_meta.get("feature_summary", {}),
        "selected_count": len(centers_vox),
        "centers_voxel": [c.tolist() for c in centers_vox],
        "centers_world": [c.tolist() for c in centers_world],
        "individual_masks": individual_filenames,
        "union_mask": union_filename,
        "artifacts": {
            "candidate_roi_union": union_filename,
            "candidate_roi_manifest": "candidate_roi_manifest.json",
            **{f"candidate_roi_{i:02d}": fname for i, fname in enumerate(individual_filenames)},
        },
        "limitations": [
            "Candidate ROI is a heuristic proposal, NOT a validated FCD detector.",
            "Candidate centres are driven by local brightness, gradient signal, "
            "and optional left-right asymmetry; "
            "they may coincide with normal anatomy (e.g. cortical sulcal rims, "
            "white-matter hyperintensities) rather than true lesions.",
            "No machine learning or atlas registration is used; proposals are "
            "generated from the single volume only.",
            "All downstream metrics (residual, Gaussian-space scores) derived "
            "from these candidates require clinical validation before use in "
            "any diagnostic context.",
            "Paper validation against ds004199 ROI labels is required before "
            "claiming sensitivity/specificity for any candidate selection.",
        ],
    }

    manifest_path = output_dir / "candidate_roi_manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate heuristic candidate ROI masks from a single MRI volume."
    )
    p.add_argument("volume", type=Path, help="Input NIfTI volume (FLAIR or T1w).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-candidates", type=int, default=5)
    p.add_argument("--radius-mm", type=float, default=8.0)
    p.add_argument("--min-separation-mm", type=float, default=16.0)
    p.add_argument("--percentile", type=float, default=99.5)
    p.add_argument(
        "--proposal-mode",
        choices=PROPOSAL_MODES,
        default="brightness-gradient",
        help=(
            "Feature model for candidate ranking. brightness-gradient is the "
            "legacy local signal heuristic; asymmetry-hybrid adds canonical "
            "left-right FLAIR asymmetry."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = generate_candidate_rois(
        volume_path=args.volume,
        output_dir=args.output_dir,
        max_candidates=args.max_candidates,
        radius_mm=args.radius_mm,
        min_separation_mm=args.min_separation_mm,
        percentile=args.percentile,
        proposal_mode=args.proposal_mode,
    )
    print(f"Selected {result['selected_count']} candidate(s) -> {args.output_dir}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
