#!/usr/bin/env python3
"""Regenerate lightweight browser splats from MRI data plus expert lesion masks.

The output is intentionally demo-oriented:
- anatomy comes from the patient MRI intensity volume
- lesion highlight comes only from the provided expert lesion mask
- point budget is low enough for responsive browser rotation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fcd_prototype.splat_ply_export import export_lesion_highlighted_splat_ply


def _artifact(manifest: dict, key: str) -> str | None:
    artifacts = manifest.get("artifacts") or {}
    value = artifacts.get(key) or manifest.get(key)
    return str(value) if value else None


def _resolve_existing(subject_dir: Path, manifest: dict, keys: list[str], fallbacks: list[str]) -> Path:
    for key in keys:
        name = _artifact(manifest, key)
        if name and (subject_dir / name).exists():
            return subject_dir / name
    for name in fallbacks:
        if (subject_dir / name).exists():
            return subject_dir / name
    raise FileNotFoundError(f"No existing artifact found for keys={keys} fallbacks={fallbacks} in {subject_dir}")


def regenerate_subject(
    subject_dir: Path,
    *,
    anatomy_points: int,
    lesion_points: int,
    percentile_threshold: float,
    rng_seed: int,
    output_name: str,
) -> dict:
    manifest_path = subject_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    volume_path = _resolve_existing(
        subject_dir,
        manifest,
        ["patient_flair", "flair", "flair_nifti", "patient_mri"],
        ["patient_flair.nii.gz"],
    )
    roi_path = _resolve_existing(
        subject_dir,
        manifest,
        ["manual_lesion_roi", "lesion_roi", "roi"],
        ["manual_lesion_roi.nii.gz", "lesion_roi.nii.gz"],
    )
    out_ply = subject_dir / output_name

    meta = export_lesion_highlighted_splat_ply(
        volume_path=volume_path,
        roi_path=roi_path,
        output_ply=out_ply,
        max_anatomy_points=anatomy_points,
        max_lesion_points=lesion_points,
        percentile_threshold=percentile_threshold,
        rng_seed=rng_seed,
        subject=manifest.get("subject", subject_dir.name),
        group=manifest.get("group", "fcd"),
        sampling_mode="volume",
    )
    meta.update(
        {
            "browser_profile": "responsive_expert_roi_only",
            "browser_profile_reason": (
                "Lower point budget for smooth browser interaction. The lesion "
                "highlight is sampled only from the provided expert lesion mask."
            ),
            "uses_algorithmic_candidate_regions": False,
            "uses_residual_or_subtraction_signal": False,
            "source_truth_for_lesion": "provided_expert_lesion_mask",
        }
    )

    meta_path = subject_dir / f"{Path(output_name).stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    manifest.setdefault("artifacts", {})
    manifest["artifacts"]["patient_splat_lesion_highlighted_ply"] = output_name
    manifest["artifacts"]["patient_splat_lesion_highlighted_meta"] = meta_path.name
    manifest["patient_splat_lesion_highlighted_meta"] = meta
    manifest["browser_splat_profile"] = {
        "profile": "responsive_expert_roi_only",
        "anatomy_points": int(meta.get("n_anatomy_points", 0)),
        "lesion_points": int(meta.get("n_lesion_points", 0)),
        "total_points": int(meta.get("n_points", 0)),
        "source_truth_for_lesion": "provided_expert_lesion_mask",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/end_to_end_fcd85_hd"))
    parser.add_argument("--subject", action="append", default=None)
    parser.add_argument("--anatomy-points", type=int, default=45_000)
    parser.add_argument("--lesion-points", type=int, default=25_000)
    parser.add_argument("--percentile-threshold", type=float, default=12.0)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument("--output-name", default="patient_splat_lesion_highlighted_browser.ply")
    args = parser.parse_args()

    root = args.root
    subjects = args.subject
    if not subjects:
        discovered: list[str] = []
        index_path = root / "index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            entries = index.get("subjects", index if isinstance(index, list) else [])
            discovered.extend(
                str(entry.get("subject", entry) if isinstance(entry, dict) else entry)
                for entry in entries
            )
        discovered.extend(p.name for p in sorted(root.glob("sub-*")) if (p / "manifest.json").exists())
        subjects = sorted(dict.fromkeys(discovered))

    total = 0
    for subject in subjects:
        subject_dir = root / str(subject)
        meta = regenerate_subject(
            subject_dir,
            anatomy_points=args.anatomy_points,
            lesion_points=args.lesion_points,
            percentile_threshold=args.percentile_threshold,
            rng_seed=args.rng_seed,
            output_name=args.output_name,
        )
        total += int(meta.get("n_points", 0))
        print(
            f"{subject}: {meta.get('n_points')} points "
            f"({meta.get('n_anatomy_points')} anatomy, {meta.get('n_lesion_points')} lesion)"
        )
    print(f"done: {len(subjects)} subjects, {total} total browser points")


if __name__ == "__main__":
    main()
