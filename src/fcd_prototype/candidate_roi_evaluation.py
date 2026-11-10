"""Candidate ROI recall evaluation against manual FCD ROI labels.

Evaluates how well the heuristic candidate ROI generator covers the manually
delineated lesion labels. Computes per-subject recall (manual coverage) and
precision metrics, plus cohort-level summaries and sweep analysis.

Entry point: fcd-evaluate-candidate-roi
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from fcd_prototype.bids_discovery import discover_dataset
from fcd_prototype.candidate_roi import PROPOSAL_MODES, generate_candidate_rois


# ---------------------------------------------------------------------------
# Dice and coverage metrics
# ---------------------------------------------------------------------------


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    """Dice similarity coefficient between two binary masks.

    Returns float in [0, 1].
    """
    a_bin = (a > 0.5).astype(bool)
    b_bin = (b > 0.5).astype(bool)
    intersection = np.logical_and(a_bin, b_bin).sum()
    union = a_bin.sum() + b_bin.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / union


def mask_coverage(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Fraction of reference voxels covered by candidate.

    Returns float in [0, 1].
    """
    ref_bin = (reference > 0.5).astype(bool)
    cand_bin = (candidate > 0.5).astype(bool)
    if ref_bin.sum() == 0:
        return 1.0
    intersection = np.logical_and(cand_bin, ref_bin).sum()
    return float(intersection) / float(ref_bin.sum())


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------


def _resample_to_grid(
    source_data: np.ndarray,
    source_affine: np.ndarray,
    target_affine: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Resample source to target grid using nearest-neighbor interpolation."""
    from nibabel.processing import resample_from_to
    source_img = nib.Nifti1Image(source_data, source_affine)
    target_img = nib.Nifti1Image(np.zeros(target_shape, dtype=np.float32), target_affine)
    resampled_img = resample_from_to(source_img, target_img, order=0)
    return resampled_img.get_fdata(dtype=np.float32)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_candidate_recall(
    dataset_dir: Path | str,
    output_dir: Path | str,
    subjects: Optional[list[str]] = None,
    max_candidates: int = 5,
    radius_mm: float = 8.0,
    min_separation_mm: float = 16.0,
    percentile: float = 99.5,
    proposal_mode: str = "brightness-gradient",
    force: bool = False,
) -> dict:
    """Evaluate candidate ROI recall on FCD subjects with manual FLAIR ROI labels.

    Generates candidate ROIs, computes intersection metrics, and writes per-subject
    and cohort-level summaries.

    Parameters
    ----------
    dataset_dir:
        Root of BIDS dataset (must contain participants.tsv, sub-* dirs).
    output_dir:
        Directory where candidate_roi/<config_id>/<subject> is written.
    subjects:
        Explicit list of subject IDs to evaluate; None means all FCD with ROI.
    max_candidates:
        Max candidate spheres per subject.
    radius_mm:
        Candidate sphere radius.
    min_separation_mm:
        Min sphere-to-sphere separation.
    percentile:
        Proposal map percentile threshold.
    proposal_mode:
        Candidate proposal feature model passed to ``generate_candidate_rois``.
    force:
        If True, regenerate even if cached.

    Returns
    -------
    dict
        Summary with keys:
        - config_id: "mc<max_candidates>_r<radius>_p<percentile>"
        - n_subjects: count of evaluated subjects
        - subject_metrics: list of dicts with per-subject results
        - mean_dice, median_dice
        - mean_manual_coverage, median_manual_coverage
        - fractions_coverage_ge_{0.1, 0.25, 0.5, 0.8}
        - recommendation: string summary
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config_id = f"mc{max_candidates}_r{radius_mm}_ms{min_separation_mm}_p{percentile}"
    config_id = (
        base_config_id
        if proposal_mode == "brightness-gradient"
        else f"{proposal_mode.replace('-', '')}_{base_config_id}"
    )
    config_output_dir = output_dir / config_id
    config_output_dir.mkdir(parents=True, exist_ok=True)

    # Discover subjects
    all_subjects = list(discover_dataset(dataset_dir))
    if subjects:
        subject_ids_set = set(subjects)
        all_subjects = [s for s in all_subjects if s.subject_id in subject_ids_set]

    # Filter to FCD with manual FLAIR ROI
    fcd_with_roi = [s for s in all_subjects if s.group == "fcd" and s.has_roi and s.flair]

    subject_metrics: list[dict] = []

    for subject in fcd_with_roi:
        subject_id = subject.subject_id
        subject_output_dir = config_output_dir / subject_id

        # Generate or load cached candidates
        candidate_dir = subject_output_dir / "candidate_roi"
        candidate_union_path = candidate_dir / "candidate_roi_union.nii.gz"

        if not force and candidate_union_path.exists():
            candidate_union = nib.load(str(candidate_union_path))
            candidate_data = candidate_union.get_fdata(dtype=np.float32)
            if candidate_data.ndim == 4:
                candidate_data = candidate_data[..., 0]
            candidate_affine = candidate_union.affine.copy()
        else:
            result = generate_candidate_rois(
                volume_path=subject.flair,
                output_dir=candidate_dir,
                max_candidates=max_candidates,
                radius_mm=radius_mm,
                min_separation_mm=min_separation_mm,
                percentile=percentile,
                proposal_mode=proposal_mode,
            )
            candidate_union = nib.load(str(candidate_union_path))
            candidate_data = candidate_union.get_fdata(dtype=np.float32)
            if candidate_data.ndim == 4:
                candidate_data = candidate_data[..., 0]
            candidate_affine = candidate_union.affine.copy()

        # Load manual ROI
        manual_roi = nib.load(str(subject.flair_roi))
        manual_data = manual_roi.get_fdata(dtype=np.float32)
        if manual_data.ndim == 4:
            manual_data = manual_data[..., 0]
        manual_affine = manual_roi.affine.copy()
        manual_shape = manual_data.shape[:3]

        # Canonicalize
        candidate_canonical = nib.as_closest_canonical(nib.Nifti1Image(candidate_data, candidate_affine))
        candidate_data = candidate_canonical.get_fdata(dtype=np.float32)
        if candidate_data.ndim == 4:
            candidate_data = candidate_data[..., 0]
        candidate_affine = candidate_canonical.affine.copy()

        manual_canonical = nib.as_closest_canonical(nib.Nifti1Image(manual_data, manual_affine))
        manual_data = manual_canonical.get_fdata(dtype=np.float32)
        if manual_data.ndim == 4:
            manual_data = manual_data[..., 0]
        manual_affine = manual_canonical.affine.copy()

        # Resample candidate to manual grid if needed
        if candidate_data.shape != manual_data.shape or not np.allclose(candidate_affine, manual_affine):
            candidate_data = _resample_to_grid(
                candidate_data,
                candidate_affine,
                manual_affine,
                manual_data.shape,
            )

        # Compute metrics
        manual_voxels = int((manual_data > 0.5).sum())
        candidate_voxels = int((candidate_data > 0.5).sum())
        intersection_voxels = int(np.logical_and(manual_data > 0.5, candidate_data > 0.5).sum())

        dice = dice_score(candidate_data, manual_data)
        manual_coverage = mask_coverage(candidate_data, manual_data)
        candidate_precision = (
            float(intersection_voxels) / float(candidate_voxels)
            if candidate_voxels > 0 else 1.0
        )

        # Try to load candidate manifest for selected_count
        manifest_path = candidate_dir / "candidate_roi_manifest.json"
        selected_count = 0
        if manifest_path.exists():
            with manifest_path.open() as fh:
                manifest = json.load(fh)
                selected_count = manifest.get("selected_count", 0)

        metric = {
            "subject_id": subject_id,
            "manual_voxels": manual_voxels,
            "candidate_voxels": candidate_voxels,
            "intersection_voxels": intersection_voxels,
            "dice": float(dice),
            "manual_coverage": float(manual_coverage),
            "candidate_precision": float(candidate_precision),
            "selected_count": selected_count,
            "proposal_mode": proposal_mode,
        }
        subject_metrics.append(metric)

    if not subject_metrics:
        summary = {
            "config_id": config_id,
            "n_subjects": 0,
            "subject_metrics": [],
            "mean_dice": None,
            "median_dice": None,
            "mean_manual_coverage": None,
            "median_manual_coverage": None,
            "fractions_coverage_ge_0_1": None,
            "fractions_coverage_ge_0_25": None,
            "fractions_coverage_ge_0_5": None,
            "fractions_coverage_ge_0_8": None,
            "recommendation": "No subjects with FCD + FLAIR ROI found.",
        }
    else:
        dices = [m["dice"] for m in subject_metrics]
        coverages = [m["manual_coverage"] for m in subject_metrics]

        mean_dice = float(np.mean(dices))
        median_dice = float(np.median(dices)) if dices else 0.0
        mean_coverage = float(np.mean(coverages))
        median_coverage = float(np.median(coverages)) if coverages else 0.0

        frac_ge_0_1 = float(sum(1 for c in coverages if c >= 0.1) / len(coverages))
        frac_ge_0_25 = float(sum(1 for c in coverages if c >= 0.25) / len(coverages))
        frac_ge_0_5 = float(sum(1 for c in coverages if c >= 0.5) / len(coverages))
        frac_ge_0_8 = float(sum(1 for c in coverages if c >= 0.8) / len(coverages))

        n = len(subject_metrics)
        recommendation = f"{n} subjects: median recall={median_coverage:.2f}, dice={median_dice:.2f}"

        summary = {
            "config_id": config_id,
            "n_subjects": n,
            "subject_metrics": subject_metrics,
            "mean_dice": mean_dice,
            "median_dice": median_dice,
            "mean_manual_coverage": mean_coverage,
            "median_manual_coverage": median_coverage,
            "fractions_coverage_ge_0_1": frac_ge_0_1,
            "fractions_coverage_ge_0_25": frac_ge_0_25,
            "fractions_coverage_ge_0_5": frac_ge_0_5,
            "fractions_coverage_ge_0_8": frac_ge_0_8,
            "recommendation": recommendation,
        }

    # Write summary.json
    summary_path = config_output_dir / "summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)

    # Write subjects.csv
    csv_path = config_output_dir / "subjects.csv"
    if subject_metrics:
        keys = list(subject_metrics[0].keys())
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(subject_metrics)

    return summary


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def sweep_candidate_recall(
    dataset_dir: Path | str,
    output_dir: Path | str,
    max_candidates_values: tuple[int, ...] = (5,),
    radius_mm_values: tuple[float, ...] = (8.0,),
    percentile_values: tuple[float, ...] = (99.5,),
    min_separation_mm: float = 16.0,
    proposal_modes: tuple[str, ...] = ("brightness-gradient",),
    subjects: Optional[list[str]] = None,
    force: bool = False,
) -> dict:
    """Sweep over parameter combinations and compare.

    Returns dict with keys:
    - configs: list of per-config summaries
    - best_by_mean_coverage: config_id of best mean recall
    - best_by_fraction_ge_0_25: config_id of best fraction with coverage >= 0.25
    """
    configs = []
    best_mean_coverage = None
    best_mean_coverage_val = -1.0
    best_frac_025 = None
    best_frac_025_val = -1.0

    for proposal_mode in proposal_modes:
        for mc in max_candidates_values:
            for r in radius_mm_values:
                for p in percentile_values:
                    summary = evaluate_candidate_recall(
                        dataset_dir=dataset_dir,
                        output_dir=output_dir,
                        subjects=subjects,
                        max_candidates=mc,
                        radius_mm=r,
                        min_separation_mm=min_separation_mm,
                        percentile=p,
                        proposal_mode=proposal_mode,
                        force=force,
                    )
                    configs.append(summary)

                    mean_cov = summary.get("mean_manual_coverage")
                    if mean_cov is not None and mean_cov > best_mean_coverage_val:
                        best_mean_coverage_val = mean_cov
                        best_mean_coverage = summary["config_id"]

                    frac_025 = summary.get("fractions_coverage_ge_0_25")
                    if frac_025 is not None and frac_025 > best_frac_025_val:
                        best_frac_025_val = frac_025
                        best_frac_025 = summary["config_id"]

    return {
        "configs": configs,
        "best_by_mean_coverage": best_mean_coverage,
        "best_by_fraction_ge_0_25": best_frac_025,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate candidate ROI recall against manual FCD labels."
    )
    p.add_argument("dataset_dir", type=Path, help="BIDS dataset root.")
    p.add_argument("output_dir", type=Path, help="Output directory for results.")
    p.add_argument("--subjects", type=str, help="Comma-separated subject IDs to evaluate.")
    p.add_argument("--max-candidates", type=str, default="5", help="Comma-separated values.")
    p.add_argument("--radius-mm", type=str, default="8.0", help="Comma-separated values.")
    p.add_argument("--percentile", type=str, default="99.5", help="Comma-separated values.")
    p.add_argument("--min-separation-mm", type=float, default=16.0)
    p.add_argument(
        "--proposal-mode",
        type=str,
        default="brightness-gradient",
        help=(
            "Comma-separated proposal feature modes. Choices: "
            + ", ".join(PROPOSAL_MODES)
        ),
    )
    p.add_argument("--force", action="store_true", help="Regenerate candidates.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # Parse multi-value arguments
    subjects = None
    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",")]

    max_candidates_values = tuple(
        int(x.strip()) for x in args.max_candidates.split(",")
    )
    radius_mm_values = tuple(
        float(x.strip()) for x in args.radius_mm.split(",")
    )
    percentile_values = tuple(
        float(x.strip()) for x in args.percentile.split(",")
    )
    proposal_modes = tuple(
        x.strip() for x in args.proposal_mode.split(",") if x.strip()
    )
    if not proposal_modes:
        raise ValueError("--proposal-mode must include at least one mode")
    invalid_modes = [mode for mode in proposal_modes if mode not in PROPOSAL_MODES]
    if invalid_modes:
        raise ValueError(
            "proposal-mode must be drawn from "
            f"{', '.join(PROPOSAL_MODES)}, got {invalid_modes!r}"
        )

    if (
        len(max_candidates_values) > 1
        or len(radius_mm_values) > 1
        or len(percentile_values) > 1
        or len(proposal_modes) > 1
    ):
        result = sweep_candidate_recall(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            max_candidates_values=max_candidates_values,
            radius_mm_values=radius_mm_values,
            percentile_values=percentile_values,
            min_separation_mm=args.min_separation_mm,
            proposal_modes=proposal_modes,
            subjects=subjects,
            force=args.force,
        )
        print(json.dumps(result, indent=2))
    else:
        result = evaluate_candidate_recall(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            subjects=subjects,
            max_candidates=max_candidates_values[0],
            radius_mm=radius_mm_values[0],
            min_separation_mm=args.min_separation_mm,
            percentile=percentile_values[0],
            proposal_mode=proposal_modes[0],
            force=args.force,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
