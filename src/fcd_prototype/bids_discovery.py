"""BIDS/ds004199 subject discovery utilities (no pandas dependency)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class SubjectRecord:
    subject_id: str
    group: str  # 'fcd' or 'hc'
    metadata: dict = field(default_factory=dict)
    t1w: Path | None = None
    t2w: Path | None = None
    flair: Path | None = None
    flair_roi: Path | None = None

    @property
    def has_roi(self) -> bool:
        return self.flair_roi is not None


def load_participants(participants_tsv: Path) -> dict[str, SubjectRecord]:
    """Parse participants.tsv and return {subject_id: SubjectRecord}."""
    records: dict[str, SubjectRecord] = {}
    with participants_tsv.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            sub_id = row.get("participant_id", "").strip()
            if not sub_id:
                continue
            group = row.get("group", "").strip().lower()
            meta = {k: v for k, v in row.items() if k not in ("participant_id", "group")}
            records[sub_id] = SubjectRecord(
                subject_id=sub_id, group=group, metadata=meta
            )
    return records


def _find_anat_file(anat_dir: Path, suffix: str) -> Path | None:
    """Return first .nii.gz file under anat_dir whose name contains suffix."""
    if not anat_dir.is_dir():
        return None
    for p in sorted(anat_dir.glob("*.nii.gz")):
        if suffix in p.name:
            return p
    return None


def discover_subject(
    dataset_dir: Path,
    subject_id: str,
    group: str = "",
    metadata: dict | None = None,
) -> SubjectRecord:
    """Build a SubjectRecord by inspecting the local filesystem."""
    anat_dir = dataset_dir / subject_id / "anat"

    # Plain FLAIR: match *_FLAIR.nii.gz but not *_FLAIR_roi.nii.gz
    flair: Path | None = None
    if anat_dir.is_dir():
        for p in sorted(anat_dir.glob("*_FLAIR.nii.gz")):
            if "_roi" not in p.name:
                flair = p
                break

    return SubjectRecord(
        subject_id=subject_id,
        group=group,
        metadata=metadata or {},
        t1w=_find_anat_file(anat_dir, "_T1w"),
        t2w=_find_anat_file(anat_dir, "_T2w"),
        flair=flair,
        flair_roi=_find_anat_file(anat_dir, "_FLAIR_roi"),
    )


_SIDECAR_DISPLAY_KEYS = (
    "Manufacturer",
    "ManufacturersModelName",
    "MagneticFieldStrength",
    "MRAcquisitionType",
    "SeriesDescription",
    "ProtocolName",
    "SliceThickness",
    "EchoTime",
    "RepetitionTime",
    "InversionTime",
    "FlipAngle",
    "PixelBandwidth",
    "ConversionSoftware",
)


def load_sidecar(nifti_path: Path) -> dict:
    """Return selected acquisition fields from the BIDS JSON sidecar next to nifti_path.

    Returns an empty dict if no sidecar exists or it cannot be parsed.
    Only fields in _SIDECAR_DISPLAY_KEYS are included.
    """
    # .with_suffix("").with_suffix(".json") strips both .gz and .nii extensions
    candidate = nifti_path.with_suffix("").with_suffix(".json")
    if not candidate.is_file():
        return {}
    try:
        raw = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {k: raw[k] for k in _SIDECAR_DISPLAY_KEYS if k in raw}


def discover_dataset(dataset_dir: Path) -> Iterator[SubjectRecord]:
    """
    Yield SubjectRecord for every subject found in dataset_dir.

    Uses participants.tsv for group/metadata if present; falls back to
    filesystem discovery only.
    """
    dataset_dir = Path(dataset_dir)
    participants_file = dataset_dir / "participants.tsv"

    known: dict[str, SubjectRecord] = {}
    if participants_file.is_file():
        known = load_participants(participants_file)

    # Walk sub-* directories
    for sub_dir in sorted(dataset_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        sub_id = sub_dir.name
        base = known.get(sub_id)
        group = base.group if base else ""
        meta = base.metadata if base else {}
        yield discover_subject(dataset_dir, sub_id, group=group, metadata=meta)
