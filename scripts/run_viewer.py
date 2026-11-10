#!/usr/bin/env python3
"""Download the public demo if requested, check its data, and serve locally.

Uses Python's standard library only. No pip install or GPU is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.error
import urllib.request
import webbrowser
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("outputs/end_to_end_fcd85_hd")


def validate_data(repo_root: Path) -> list[str]:
    root = repo_root / DATA_ROOT
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    subjects = [entry["subject"] if isinstance(entry, dict) else entry for entry in index["subjects"]]
    if not subjects:
        raise ValueError("The data index contains no cases")
    for subject in subjects:
        if not isinstance(subject, str) or len(subject) != 9 or not subject.startswith("sub-") or not subject[4:].isdigit():
            raise ValueError("Invalid subject ID in the data index")
        folder = root / subject
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        artifacts = manifest.get("artifacts", {})
        required = ("patient_flair", "lesion_roi", "patient_splat_lesion_highlighted_ply")
        for key in required:
            name = artifacts.get(key)
            if not name or not (folder / name).resolve().is_relative_to(folder.resolve()) or not (folder / name).is_file():
                raise ValueError(f"{subject}: missing or invalid {key}")
    return subjects


def install_bundle(archive_path: Path, repo_root: Path, expected_sha256: str) -> None:
    with archive_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_sha256:
        raise ValueError("Demo checksum does not match this release. Download the v0.2.0 demo again.")
    with tempfile.TemporaryDirectory(prefix="isu-demo-") as temp:
        stage = Path(temp)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                name = PurePosixPath(member.filename)
                if name.is_absolute() or ".." in name.parts or "\\" in member.filename or not member.filename.startswith("outputs/"):
                    raise ValueError("Unsafe filename in the demo archive")
            archive.extractall(stage)
        incoming = validate_data(stage)
        target = repo_root / DATA_ROOT
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = validate_data(repo_root) if (target / "index.json").exists() else []
        # Compare overlapping cases first. Never overwrite someone's different data.
        for subject in incoming:
            current = target / subject
            if current.exists():
                for source in (stage / DATA_ROOT / subject).iterdir():
                    destination = current / source.name
                    with source.open("rb") as stream:
                        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                    if not destination.is_file():
                        raise ValueError(f"{subject}: existing data differs; move it aside before installing")
                    with destination.open("rb") as stream:
                        destination_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                    if source_hash != destination_hash:
                        raise ValueError(f"{subject}: existing data differs; move it aside before installing")
        for subject in incoming:
            if not (target / subject).exists():
                shutil.copytree(stage / DATA_ROOT / subject, target / subject)
        combined = sorted(set(existing + incoming))
        index = {"dataset": "OpenNeuro ds004199", "subjects": [{"subject": subject, "group": "fcd"} for subject in combined]}
        temporary_index = target / "index.json.tmp"
        temporary_index.write_text(json.dumps(index, indent=2), encoding="utf-8")
        temporary_index.replace(target / "index.json")
        (repo_root / "outputs/fcd85_subjects.txt").write_text("\n".join(combined) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--download-demo", action="store_true", help="Fetch the two-case public release demo")
    source.add_argument("--download-all", action="store_true", help="Fetch all 85 cases (about 2.5 GiB unpacked)")
    source.add_argument("--demo-archive", type=Path, help="Install an already downloaded v0.2.0 demo ZIP")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true", help="Check the data and exit without starting a server")
    args = parser.parse_args(argv)
    try:
        catalog = json.loads((REPO_ROOT / "data/release-assets.json").read_text(encoding="utf-8"))
        if args.demo_archive:
            asset = next((item for item in catalog["assets"] if item["name"] == args.demo_archive.name), None)
            if asset is None:
                raise ValueError("Use an archive filename listed in data/release-assets.json")
            install_bundle(args.demo_archive, REPO_ROOT, asset["sha256"])
        elif args.download_demo or args.download_all:
            kind = "full" if args.download_all else "demo"
            for asset in (item for item in catalog["assets"] if item["kind"] == kind):
                existing = validate_data(REPO_ROOT) if (REPO_ROOT / DATA_ROOT / "index.json").exists() else []
                if set(asset["subjects"]).issubset(existing):
                    print(f"Already installed: {asset['name']}", flush=True)
                    continue
                print(f"Downloading {asset['name']} ({asset['bytes'] / 2**20:.1f} MiB)...", flush=True)
                with tempfile.TemporaryDirectory(prefix="isu-download-") as temp:
                    downloaded = Path(temp) / asset["name"]
                    with urllib.request.urlopen(catalog["release_base_url"] + asset["name"], timeout=120) as response, downloaded.open("wb") as target:
                        shutil.copyfileobj(response, target)
                    install_bundle(downloaded, REPO_ROOT, asset["sha256"])
        subjects = validate_data(REPO_ROOT)
        print(f"Data ready: {len(subjects)} case(s).", flush=True)
        if args.check:
            return 0
        url = f"http://127.0.0.1:{args.port}/viewer/app.html?subject={subjects[0]}&mode=mri2d&lesion=shown"
        handler = partial(SimpleHTTPRequestHandler, directory=str(REPO_ROOT))
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            print(f"Open: {url}\nKeep this terminal open. Press Ctrl+C to stop.", flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("\nViewer stopped.")
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"Could not start the viewer: {error}")
        print("From a fresh clone, run: python scripts/run_viewer.py --download-demo")
        print("If port 8081 is busy, add --port 8082. See docs/running-the-project.md.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
