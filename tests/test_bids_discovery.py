"""Tests for bids_discovery — uses temp folder fixtures, no real data."""

import csv
from pathlib import Path

import pytest

from fcd_prototype.bids_discovery import (
    SubjectRecord,
    discover_dataset,
    discover_subject,
    load_participants,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> Path:
    """Create a minimal BIDS directory with two subjects."""
    # participants.tsv
    tsv = tmp_path / "participants.tsv"
    with tsv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["participant_id", "group", "age"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"participant_id": "sub-00001", "group": "fcd", "age": "34"})
        writer.writerow({"participant_id": "sub-00002", "group": "hc", "age": "29"})

    # sub-00001: T1w + FLAIR + ROI
    anat1 = tmp_path / "sub-00001" / "anat"
    anat1.mkdir(parents=True)
    (anat1 / "sub-00001_acq-sag111_T1w.nii.gz").touch()
    (anat1 / "sub-00001_acq-tse_T2w.nii.gz").touch()
    (anat1 / "sub-00001_acq-tse3dvfl_FLAIR.nii.gz").touch()
    (anat1 / "sub-00001_acq-tse3dvfl_FLAIR_roi.nii.gz").touch()

    # sub-00002: T1w + FLAIR only (HC, no ROI)
    anat2 = tmp_path / "sub-00002" / "anat"
    anat2.mkdir(parents=True)
    (anat2 / "sub-00002_acq-sag111_T1w.nii.gz").touch()
    (anat2 / "sub-00002_acq-tse3dvfl_FLAIR.nii.gz").touch()

    return tmp_path


@pytest.fixture()
def missing_anat_dataset(tmp_path: Path) -> Path:
    """Subject directory exists but anat/ is absent."""
    tsv = tmp_path / "participants.tsv"
    with tsv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["participant_id", "group"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"participant_id": "sub-00099", "group": "hc"})
    (tmp_path / "sub-00099").mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# load_participants
# ---------------------------------------------------------------------------

def test_load_participants_count(tiny_dataset):
    recs = load_participants(tiny_dataset / "participants.tsv")
    assert len(recs) == 2


def test_load_participants_groups(tiny_dataset):
    recs = load_participants(tiny_dataset / "participants.tsv")
    assert recs["sub-00001"].group == "fcd"
    assert recs["sub-00002"].group == "hc"


def test_load_participants_metadata(tiny_dataset):
    recs = load_participants(tiny_dataset / "participants.tsv")
    assert recs["sub-00001"].metadata.get("age") == "34"


# ---------------------------------------------------------------------------
# discover_subject
# ---------------------------------------------------------------------------

def test_discover_subject_t1w(tiny_dataset):
    rec = discover_subject(tiny_dataset, "sub-00001", group="fcd")
    assert rec.t1w is not None
    assert rec.t1w.name.endswith("_T1w.nii.gz")


def test_discover_subject_t2w(tiny_dataset):
    rec = discover_subject(tiny_dataset, "sub-00001", group="fcd")
    assert rec.t2w is not None
    assert rec.t2w.name.endswith("_T2w.nii.gz")


def test_discover_subject_flair(tiny_dataset):
    rec = discover_subject(tiny_dataset, "sub-00001", group="fcd")
    assert rec.flair is not None
    assert "_roi" not in rec.flair.name


def test_discover_subject_roi(tiny_dataset):
    rec = discover_subject(tiny_dataset, "sub-00001", group="fcd")
    assert rec.flair_roi is not None
    assert "_roi" in rec.flair_roi.name


def test_discover_subject_hc_no_roi(tiny_dataset):
    rec = discover_subject(tiny_dataset, "sub-00002", group="hc")
    assert rec.flair_roi is None
    assert rec.flair is not None


def test_discover_subject_has_roi_property(tiny_dataset):
    fcd_rec = discover_subject(tiny_dataset, "sub-00001")
    hc_rec = discover_subject(tiny_dataset, "sub-00002")
    assert fcd_rec.has_roi is True
    assert hc_rec.has_roi is False


def test_discover_subject_missing_anat(missing_anat_dataset):
    rec = discover_subject(missing_anat_dataset, "sub-00099")
    assert rec.t1w is None
    assert rec.t2w is None
    assert rec.flair is None
    assert rec.flair_roi is None


# ---------------------------------------------------------------------------
# discover_dataset
# ---------------------------------------------------------------------------

def test_discover_dataset_yields_both(tiny_dataset):
    recs = list(discover_dataset(tiny_dataset))
    ids = {r.subject_id for r in recs}
    assert "sub-00001" in ids
    assert "sub-00002" in ids


def test_discover_dataset_group_from_tsv(tiny_dataset):
    recs = {r.subject_id: r for r in discover_dataset(tiny_dataset)}
    assert recs["sub-00001"].group == "fcd"
    assert recs["sub-00002"].group == "hc"


def test_discover_dataset_no_tsv(tmp_path):
    """Falls back gracefully when participants.tsv is absent."""
    anat = tmp_path / "sub-00010" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-00010_T1w.nii.gz").touch()
    recs = list(discover_dataset(tmp_path))
    assert any(r.subject_id == "sub-00010" for r in recs)


def test_discover_dataset_empty_dir(tmp_path):
    recs = list(discover_dataset(tmp_path))
    assert recs == []


# ---------------------------------------------------------------------------
# BOM handling
# ---------------------------------------------------------------------------

def test_load_participants_bom(tmp_path: Path):
    """participants.tsv with UTF-8 BOM is parsed correctly (real ds004199 case)."""
    tsv = tmp_path / "participants.tsv"
    # Write with BOM — simulates files saved by Windows tools / OpenNeuro exports
    tsv.write_bytes(
        "﻿participant_id\tgroup\tage\n"
        "sub-00058\tfcd\t32\n"
        "sub-00059\thc\t28\n".encode("utf-8")
    )
    recs = load_participants(tsv)
    assert "sub-00058" in recs, "BOM caused participant_id header corruption — sub-00058 missing"
    assert recs["sub-00058"].group == "fcd"
    assert recs["sub-00059"].group == "hc"
    assert len(recs) == 2
