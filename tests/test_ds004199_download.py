"""Tests for ds004199_download — no network calls."""

import subprocess
import sys
from pathlib import Path

import pytest

from fcd_prototype.ds004199_download import (
    _modality_suffix,
    _require_aws,
    download,
    download_dataset_files,
    download_subject,
    parse_participants_tsv,
    _s3_cp,
    _s3_ls,
)


# ---------------------------------------------------------------------------
# _modality_suffix
# ---------------------------------------------------------------------------

def test_modality_suffix_t1w():
    assert _modality_suffix("T1w") == "_T1w"


def test_modality_suffix_flair():
    assert _modality_suffix("FLAIR") == "_FLAIR"


def test_modality_suffix_unknown():
    assert _modality_suffix("DWI") == "_DWI"


# ---------------------------------------------------------------------------
# _require_aws — monkeypatched
# ---------------------------------------------------------------------------

def test_require_aws_missing(monkeypatch):
    monkeypatch.setattr("fcd_prototype.ds004199_download.shutil.which", lambda _: None)
    with pytest.raises(SystemExit) as exc_info:
        _require_aws()
    assert "aws" in str(exc_info.value).lower()


def test_require_aws_present(monkeypatch):
    monkeypatch.setattr("fcd_prototype.ds004199_download.shutil.which", lambda _: "/usr/bin/aws")
    _require_aws()  # should not raise


# ---------------------------------------------------------------------------
# _s3_cp dry-run (no subprocess)
# ---------------------------------------------------------------------------

def test_s3_cp_dry_run(tmp_path, capsys):
    dst = tmp_path / "some" / "file.tsv"
    _s3_cp("s3://openneuro.org/ds004199/participants.tsv", dst, dry_run=True)
    captured = capsys.readouterr()
    assert "dry-run" in captured.out
    assert "participants.tsv" in captured.out
    assert not dst.exists()


# ---------------------------------------------------------------------------
# _s3_ls — mocked subprocess
# ---------------------------------------------------------------------------

def test_s3_ls_parses_output(monkeypatch):
    fake_output = (
        "2023-01-01 00:00:00    123456 sub-00058_acq-sag111_T1w.nii.gz\n"
        "2023-01-01 00:00:00      1234 sub-00058_acq-sag111_T1w.json\n"
    )
    class FakeResult:
        stdout = fake_output
        returncode = 0
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeResult())
    keys = _s3_ls("s3://openneuro.org/ds004199/sub-00058/anat/")
    assert "sub-00058_acq-sag111_T1w.nii.gz" in keys
    assert "sub-00058_acq-sag111_T1w.json" in keys


# ---------------------------------------------------------------------------
# download_subject — mocked _s3_ls and _s3_cp
# ---------------------------------------------------------------------------

def test_download_subject_filters_modality(tmp_path, monkeypatch):
    remote_files = [
        "sub-00058_acq-sag111_T1w.nii.gz",
        "sub-00058_acq-sag111_T1w.json",
        "sub-00058_acq-tse3dvfl_FLAIR.nii.gz",
        "sub-00058_acq-tse3dvfl_FLAIR_roi.nii.gz",
    ]
    monkeypatch.setattr("fcd_prototype.ds004199_download._s3_ls", lambda _: remote_files)

    copied = []
    def fake_cp(src, dst, dry_run):
        copied.append(src)
    monkeypatch.setattr("fcd_prototype.ds004199_download._s3_cp", fake_cp)

    download_subject("sub-00058", tmp_path, modalities=["T1w"], include_roi=False, dry_run=False)

    assert any("T1w.nii.gz" in s for s in copied)
    assert any("T1w.json" in s for s in copied)
    assert not any("FLAIR" in s for s in copied)
    assert not any("roi" in s for s in copied)


def test_download_subject_includes_roi(tmp_path, monkeypatch):
    remote_files = [
        "sub-00058_acq-tse3dvfl_FLAIR.nii.gz",
        "sub-00058_acq-tse3dvfl_FLAIR_roi.nii.gz",
    ]
    monkeypatch.setattr("fcd_prototype.ds004199_download._s3_ls", lambda _: remote_files)

    copied = []
    monkeypatch.setattr("fcd_prototype.ds004199_download._s3_cp", lambda src, dst, dry_run: copied.append(src))

    download_subject("sub-00058", tmp_path, modalities=["FLAIR"], include_roi=True, dry_run=False)

    assert any("roi" in s for s in copied)
    assert any("FLAIR.nii.gz" in s for s in copied)


def test_download_subject_excludes_roi_by_default(tmp_path, monkeypatch):
    remote_files = ["sub-00058_acq-tse3dvfl_FLAIR_roi.nii.gz"]
    monkeypatch.setattr("fcd_prototype.ds004199_download._s3_ls", lambda _: remote_files)

    copied = []
    monkeypatch.setattr("fcd_prototype.ds004199_download._s3_cp", lambda src, dst, dry_run: copied.append(src))

    download_subject("sub-00058", tmp_path, modalities=["FLAIR"], include_roi=False, dry_run=False)

    assert not copied


def test_download_subject_handles_ls_failure(tmp_path, monkeypatch):
    def raise_cp_error(_):
        raise subprocess.CalledProcessError(1, "aws")
    monkeypatch.setattr("fcd_prototype.ds004199_download._s3_ls", raise_cp_error)

    # Should warn and return, not crash
    download_subject("sub-00999", tmp_path, modalities=["T1w"], include_roi=False, dry_run=False)


# ---------------------------------------------------------------------------
# download_dataset_files — dry-run, no network
# ---------------------------------------------------------------------------

def test_download_dataset_files_dry_run(tmp_path, capsys):
    import fcd_prototype.ds004199_download as mod
    orig_cp = mod._s3_cp

    calls = []
    def fake_cp(src, dst, dry_run):
        calls.append((src, dry_run))
    mod._s3_cp = fake_cp

    try:
        download_dataset_files(tmp_path, dry_run=True)
    finally:
        mod._s3_cp = orig_cp

    sources = [c[0] for c in calls]
    assert any("participants.tsv" in s for s in sources)
    assert any("dataset_description.json" in s for s in sources)
    assert all(c[1] for c in calls)  # all dry_run=True


# ---------------------------------------------------------------------------
# download — dry-run with all_participants=True and no local participants.tsv
# ---------------------------------------------------------------------------

def test_download_dry_run_all_participants_no_tsv(tmp_path, monkeypatch, capsys):
    """--dry-run + --all-participants must not perform real downloads when
    participants.tsv is absent: dataset-file commands are printed (dry-run),
    a clear advisory message is emitted, and the function returns immediately
    without attempting to list or fetch any subjects."""

    # Stub _require_aws so the test is fully offline.
    monkeypatch.setattr("fcd_prototype.ds004199_download._require_aws", lambda: None)

    # Capture every _s3_cp call; fail loudly if dry_run is ever False.
    cp_calls: list[tuple[str, bool]] = []

    def fake_cp(src: str, dst: Path, dry_run: bool) -> None:
        assert dry_run, (
            f"_s3_cp called with dry_run=False during a dry-run — "
            f"real download attempted for {src}"
        )
        cp_calls.append((src, dry_run))

    monkeypatch.setattr("fcd_prototype.ds004199_download._s3_cp", fake_cp)

    # _s3_ls must never be called (no subject expansion should happen).
    def forbidden_ls(prefix: str) -> list[str]:
        raise AssertionError(f"_s3_ls called unexpectedly with prefix={prefix!r}")

    monkeypatch.setattr("fcd_prototype.ds004199_download._s3_ls", forbidden_ls)

    # participants.tsv intentionally absent from tmp_path.
    download(
        destination=tmp_path,
        subjects=None,
        modalities=["T1w", "FLAIR"],
        include_roi=False,
        dry_run=True,
        all_participants=True,
    )

    captured = capsys.readouterr()

    # Dataset-file dry-run commands must have been issued via _s3_cp.
    cp_sources = [s for s, _ in cp_calls]
    assert any("participants.tsv" in s for s in cp_sources)
    assert any("dataset_description.json" in s for s in cp_sources)

    # Every cp call must have been a dry-run.
    assert cp_calls, "Expected at least one _s3_cp call for dataset files"
    assert all(dr for _, dr in cp_calls)

    # Advisory message about needing a local participants.tsv must be present.
    assert "participants.tsv" in captured.out
    assert any(
        keyword in captured.out.lower()
        for keyword in ("subject", "expansion", "re-run", "without")
    )


# ---------------------------------------------------------------------------
# parse_participants_tsv — BOM-prefixed TSV (utf-8-sig)
# ---------------------------------------------------------------------------

_BOM_TSV_CONTENT = (
    "participant_id\tgroup\tage\n"
    "sub-00001\tfcd\t30\n"
    "sub-00002\thc\t28\n"
    "sub-00003\tfcd\t35\n"
    "sub-00004\thc\t25\n"
)


@pytest.fixture()
def bom_tsv(tmp_path: Path) -> Path:
    """Write a BOM-prefixed participants.tsv and return its path."""
    tsv = tmp_path / "participants.tsv"
    tsv.write_text(_BOM_TSV_CONTENT, encoding="utf-8-sig")
    return tsv


def test_parse_participants_tsv_all(bom_tsv: Path) -> None:
    """group='all' returns every participant regardless of group column."""
    subjects = parse_participants_tsv(bom_tsv, group="all")
    assert subjects == ["sub-00001", "sub-00002", "sub-00003", "sub-00004"]


def test_parse_participants_tsv_fcd(bom_tsv: Path) -> None:
    """group='fcd' returns only FCD participants."""
    subjects = parse_participants_tsv(bom_tsv, group="fcd")
    assert subjects == ["sub-00001", "sub-00003"]


def test_parse_participants_tsv_hc(bom_tsv: Path) -> None:
    """group='hc' returns only healthy-control participants."""
    subjects = parse_participants_tsv(bom_tsv, group="hc")
    assert subjects == ["sub-00002", "sub-00004"]
