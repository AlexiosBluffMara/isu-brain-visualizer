"""Verify that the unified viewer can load every advertised FCD subject."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubjectViewerCheck:
    subject: str
    manifest_exists: bool
    lesion_splat_exists: bool
    niivue_base_exists: bool
    roi_exists: bool
    splat_points: int | None

    @property
    def ok(self) -> bool:
        return (
            self.manifest_exists
            and self.lesion_splat_exists
            and self.niivue_base_exists
            and self.roi_exists
        )


def read_subjects(subjects_file: Path) -> list[str]:
    subjects = []
    for line in subjects_file.read_text(encoding="utf-8").splitlines():
        subject = line.strip()
        if subject.startswith("sub-") and len(subject) == 9:
            subjects.append(subject)
    return subjects


def _artifact(manifest: dict, key: str) -> str | None:
    artifacts = manifest.get("artifacts") or {}
    value = artifacts.get(key) or manifest.get(key)
    return str(value) if value else None


def check_subject(root: Path, subject: str) -> SubjectViewerCheck:
    subject_dir = root / subject
    manifest_path = subject_dir / "manifest.json"
    if not manifest_path.exists():
        return SubjectViewerCheck(subject, False, False, False, False, None)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splat_name = _artifact(manifest, "patient_splat_lesion_highlighted_ply")
    flair_name = _artifact(manifest, "patient_flair")
    t1w_name = _artifact(manifest, "patient_t1w")
    roi_name = _artifact(manifest, "lesion_roi") or _artifact(manifest, "manual_lesion_roi")
    splat_meta = manifest.get("patient_splat_lesion_highlighted_meta") or {}
    browser_profile = manifest.get("browser_splat_profile") or {}
    splat_points = splat_meta.get("n_points") or browser_profile.get("total_points")

    return SubjectViewerCheck(
        subject=subject,
        manifest_exists=True,
        lesion_splat_exists=bool(splat_name and (subject_dir / splat_name).exists()),
        niivue_base_exists=bool(
            (flair_name and (subject_dir / flair_name).exists())
            or (t1w_name and (subject_dir / t1w_name).exists())
        ),
        roi_exists=bool(roi_name and (subject_dir / roi_name).exists()),
        splat_points=splat_points,
    )


def verify_viewer_dataset(root: Path, subjects_file: Path) -> list[SubjectViewerCheck]:
    return [check_subject(root, subject) for subject in read_subjects(subjects_file)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/end_to_end_fcd85_hd"))
    parser.add_argument("--subjects-file", type=Path, default=Path("outputs/fcd85_subjects.txt"))
    args = parser.parse_args(argv)

    checks = verify_viewer_dataset(args.root, args.subjects_file)
    bad = [check for check in checks if not check.ok]
    total_points = sum(check.splat_points or 0 for check in checks)

    print(f"subjects={len(checks)}")
    print(f"ok={len(checks) - len(bad)}")
    print(f"bad={len(bad)}")
    print(f"total_splat_points={total_points}")
    for check in bad:
        print(
            "BAD "
            f"{check.subject} manifest={check.manifest_exists} "
            f"lesion_splat={check.lesion_splat_exists} "
            f"niivue_base={check.niivue_base_exists} roi={check.roi_exists}"
        )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
