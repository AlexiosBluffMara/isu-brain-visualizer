#!/usr/bin/env python3
"""Package only the public MRI assets needed by the educational viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

DATASET_NOTICE = """ISU Brain Visualizer demo data

Source: OpenNeuro ds004199, version 1.0.6 (CC0).
https://openneuro.org/datasets/ds004199/versions/1.0.6
https://doi.org/10.18112/openneuro.ds004199.v1.0.6
Schuch F et al., Scientific Data 10, 475 (2023).
https://doi.org/10.1038/s41597-023-02386-7

These are processed public research images and the provided expert lesion mask.
The browser splat is derived from MRI anatomy and that expert mask.
It does not predict or diagnose a lesion. For education and research only.
Local source paths and unrelated experimental artifacts are omitted.
"""


def package_demo(root: Path, destination: Path, subjects: list[str]) -> str:
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("Provide one or more unique subjects")
    if any(not re.fullmatch(r"sub-\d{5}", subject) for subject in subjects):
        raise ValueError("Subject IDs must have the form sub-00058")
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_root = "outputs/end_to_end_fcd85_hd"
    entries = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for subject in subjects:
            subject_dir = root / subject
            original = json.loads((subject_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest = {key: original[key] for key in (
                "subject", "group", "modalities", "roi_available", "roi_source",
                "viewer_meta", "browser_splat_profile",
            ) if key in original}
            artifacts = {}
            for key in ("patient_flair", "patient_t1w", "lesion_roi", "patient_splat_lesion_highlighted_ply"):
                value = original.get("artifacts", {}).get(key)
                if not value and key == "patient_t1w":
                    continue
                if not value or not (subject_dir / value).is_file():
                    raise FileNotFoundError(f"{subject}: missing {key}")
                source = (subject_dir / value).resolve()
                if not source.is_relative_to(subject_dir.resolve()):
                    raise ValueError(f"{subject}: artifact is outside the subject folder")
                artifacts[key] = source.name
                archive.write(source, f"{archive_root}/{subject}/{source.name}")
            manifest["artifacts"] = artifacts
            # The renderer needs point ranges and world bounds, but not local paths
            # or training-state filenames from the research pipeline.
            meta = dict(original.get("patient_splat_lesion_highlighted_meta", {}))
            for key in ("source_volume", "source_roi", "gaussian_state_npz"):
                meta.pop(key, None)
            manifest["patient_splat_lesion_highlighted_meta"] = meta
            manifest["dataset"] = "OpenNeuro ds004199 v1.0.6 (CC0)"
            manifest["demo_scope"] = "Expert-labeled educational viewer; no experimental pipeline outputs."
            archive.writestr(f"{archive_root}/{subject}/manifest.json", json.dumps(manifest, indent=2))
            entries.append({"subject": subject, "group": manifest.get("group", "fcd")})
        archive.writestr(f"{archive_root}/index.json", json.dumps({"dataset": "OpenNeuro ds004199", "subjects": entries}, indent=2))
        archive.writestr("outputs/fcd85_subjects.txt", "\n".join(subjects) + "\n")
        archive.writestr(f"{archive_root}/DATASET-NOTICE.txt", DATASET_NOTICE)
    with destination.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/end_to_end_fcd85_hd"))
    parser.add_argument("--output", type=Path, default=Path("dist/isu-brain-demo-v0.2.0.zip"))
    parser.add_argument("--subjects", nargs="+", default=["sub-00058"])
    args = parser.parse_args()
    digest = package_demo(args.root, args.output, args.subjects)
    print(f"Created {args.output} ({args.output.stat().st_size:,} bytes)")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
