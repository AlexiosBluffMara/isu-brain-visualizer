from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "verify_viewer_dataset",
    Path("scripts/verify_viewer_dataset.py"),
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

check_subject = _MODULE.check_subject
read_subjects = _MODULE.read_subjects
verify_viewer_dataset = _MODULE.verify_viewer_dataset


def _write_subject(root: Path, subject: str, *, include_splat: bool = True) -> None:
    subject_dir = root / subject
    subject_dir.mkdir(parents=True)
    manifest = {
        "artifacts": {
            "patient_splat_lesion_highlighted_ply": "patient_splat_lesion_highlighted.ply",
            "patient_flair": "patient_flair.nii.gz",
            "lesion_roi": "lesion_roi.nii.gz",
        },
        "patient_splat_lesion_highlighted_meta": {"n_points": 123},
    }
    (subject_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (subject_dir / "patient_flair.nii.gz").write_text("base", encoding="utf-8")
    (subject_dir / "lesion_roi.nii.gz").write_text("roi", encoding="utf-8")
    if include_splat:
        (subject_dir / "patient_splat_lesion_highlighted.ply").write_text("ply", encoding="utf-8")


def test_read_subjects_ignores_non_subject_lines(tmp_path):
    subjects_file = tmp_path / "subjects.txt"
    subjects_file.write_text("sub-00058\nnotes\nsub-00043\n", encoding="utf-8")
    assert read_subjects(subjects_file) == ["sub-00058", "sub-00043"]


def test_check_subject_passes_when_viewer_artifacts_exist(tmp_path):
    _write_subject(tmp_path, "sub-00058")
    check = check_subject(tmp_path, "sub-00058")
    assert check.ok
    assert check.splat_points == 123


def test_check_subject_fails_when_lesion_splat_is_missing(tmp_path):
    _write_subject(tmp_path, "sub-00058", include_splat=False)
    check = check_subject(tmp_path, "sub-00058")
    assert not check.ok
    assert check.manifest_exists
    assert not check.lesion_splat_exists


def test_verify_viewer_dataset_returns_one_check_per_subject(tmp_path):
    subjects_file = tmp_path / "subjects.txt"
    subjects_file.write_text("sub-00058\nsub-00043\n", encoding="utf-8")
    _write_subject(tmp_path, "sub-00058")
    checks = verify_viewer_dataset(tmp_path, subjects_file)
    assert [check.subject for check in checks] == ["sub-00058", "sub-00043"]
    assert checks[0].ok
    assert not checks[1].ok
