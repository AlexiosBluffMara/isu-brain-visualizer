#!/usr/bin/env python3
"""Build the two-case demo, full viewer data packs, and their download catalog."""

import argparse
import csv
import json
from pathlib import Path

from package_demo import package_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("dist/release-data"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    subjects = sorted(path.parent.name for path in args.root.glob("sub-*/manifest.json"))
    if len(subjects) != 85:
        raise ValueError(f"Expected 85 public FCD cases, found {len(subjects)}")
    assets = []
    batches = [("demo", "isu-brain-demo-v0.2.0.zip", ["sub-00058", "sub-00100"])]
    batches += [("full", f"isu-brain-cases-{i // 10 + 1:02d}-v0.2.0.zip", subjects[i:i + 10]) for i in range(0, len(subjects), 10)]
    for kind, name, selected in batches:
        path = args.output_dir / name
        digest = package_demo(args.root, path, selected)
        assets.append({"kind": kind, "name": name, "subjects": selected, "bytes": path.stat().st_size, "sha256": digest})
        print(f"Ready: {name}, {len(selected)} cases, {path.stat().st_size / 2**20:.1f} MiB", flush=True)
    catalog = {
        "version": "0.2.0", "dataset": "OpenNeuro ds004199 v1.0.6", "data_license": "CC0",
        "release_base_url": "https://github.com/AlexiosBluffMara/isu-brain-visualizer/releases/download/v0.2.0/",
        "full_case_count": len(subjects), "assets": assets,
    }
    (repo / "data/release-assets.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "release-assets.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "SHA256SUMS.txt").write_text("".join(f"{asset['sha256']}  {asset['name']}\n" for asset in assets), encoding="utf-8")
    with (repo / "data/cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject", "hemisphere", "lobe", "histopathology", "pack", "source"])
        for subject in subjects:
            manifest = json.loads((args.root / subject / "manifest.json").read_text(encoding="utf-8"))
            participant = manifest.get("viewer_meta", {}).get("participant", {})
            pack = next(asset["name"] for asset in assets if asset["kind"] == "full" and subject in asset["subjects"])
            writer.writerow([subject, participant.get("hemisphere", ""), participant.get("lobe", ""), participant.get("histopathology", ""), pack, "https://openneuro.org/datasets/ds004199/versions/1.0.6"])


if __name__ == "__main__":
    main()
