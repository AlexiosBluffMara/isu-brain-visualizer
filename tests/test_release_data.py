"""Exercise data packaging, download integrity, and incremental installation."""

import hashlib
import importlib
import json
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def release_tools(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    return importlib.import_module("package_demo"), importlib.import_module("run_viewer")


def write_case(root, subject):
    folder = root / subject
    folder.mkdir(parents=True)
    artifacts = {"patient_flair": "flair.nii.gz", "patient_t1w": "t1.nii.gz", "lesion_roi": "roi.nii.gz", "patient_splat_lesion_highlighted_ply": "brain.ply"}
    for name in artifacts.values():
        (folder / name).write_bytes(b"public-test-fixture")
    manifest = {"subject": subject, "group": "fcd", "artifacts": artifacts, "source_files": {"flair": "/private/local/path"}, "patient_splat_lesion_highlighted_meta": {"n_points": 10, "source_volume": "/private/local/path", "n_lesion_points": 2}}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_package_install_upgrade_and_repeat(tmp_path, release_tools):
    pack, runner = release_tools
    root = tmp_path / "source"
    repo = tmp_path / "fresh-clone"
    write_case(root, "sub-00058")
    write_case(root, "sub-00100")
    first = tmp_path / "demo.zip"
    digest = pack.package_demo(root, first, ["sub-00058"])
    runner.install_bundle(first, repo, digest)
    assert runner.validate_data(repo) == ["sub-00058"]
    published_manifest = (repo / runner.DATA_ROOT / "sub-00058/manifest.json").read_text()
    assert "/private/local/path" not in published_manifest
    full = tmp_path / "full.zip"
    digest = pack.package_demo(root, full, ["sub-00058", "sub-00100"])
    runner.install_bundle(full, repo, digest)
    runner.install_bundle(full, repo, digest)
    assert runner.validate_data(repo) == ["sub-00058", "sub-00100"]


def test_bad_checksum_does_not_install(tmp_path, release_tools):
    _, runner = release_tools
    archive = tmp_path / "fake.zip"
    archive.write_bytes(b"wrong download")
    with pytest.raises(ValueError, match="checksum"):
        runner.install_bundle(archive, tmp_path / "repo", "0" * 64)
    assert not (tmp_path / "repo/outputs").exists()


def test_archive_cannot_escape_outputs(tmp_path, release_tools):
    _, runner = release_tools
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("outputs/../../escaped.txt", "bad")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Unsafe"):
        runner.install_bundle(archive, tmp_path / "repo", digest)
    assert not (tmp_path / "escaped.txt").exists()


def test_existing_different_case_is_preserved(tmp_path, release_tools):
    pack, runner = release_tools
    root, repo = tmp_path / "source", tmp_path / "repo"
    write_case(root, "sub-00058")
    archive = tmp_path / "case.zip"
    digest = pack.package_demo(root, archive, ["sub-00058"])
    runner.install_bundle(archive, repo, digest)
    changed = repo / runner.DATA_ROOT / "sub-00058/flair.nii.gz"
    changed.write_bytes(b"user's own data")
    with pytest.raises(ValueError, match="existing data differs"):
        runner.install_bundle(archive, repo, digest)
    assert changed.read_bytes() == b"user's own data"


def test_empty_dataset_is_not_reported_as_ready(tmp_path, release_tools):
    _, runner = release_tools
    root = tmp_path / runner.DATA_ROOT
    root.mkdir(parents=True)
    (root / "index.json").write_text('{"subjects": []}')
    with pytest.raises(ValueError, match="no cases"):
        runner.validate_data(tmp_path)
