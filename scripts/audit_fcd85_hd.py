#!/usr/bin/env python3
import json
from pathlib import Path


def main() -> int:
    out = Path("outputs/end_to_end_fcd85_hd")
    verified = []
    bad = []
    partial = []

    for subject_dir in sorted(p for p in out.glob("sub-*") if p.is_dir()):
        manifest_path = subject_dir / "manifest.json"
        if not manifest_path.exists():
            partial.append(subject_dir.name)
            continue

        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:
            bad.append(f"{subject_dir.name}:manifest_read_error:{exc}")
            continue

        stage_status = manifest.get("stage_status", {})
        artifacts = manifest.get("artifacts", {})
        ok = (
            stage_status.get("splat_ply") == "ok"
            and stage_status.get("gaussian_field") == "ok"
            and str(stage_status.get("optimized_gaussian_field_audit", "")).startswith("ok")
            and artifacts.get("patient_splat_lesion_highlighted_ply")
            and artifacts.get("optimized_gaussian_splat_ply")
        )
        (verified if ok else bad).append(subject_dir.name)

    print("verified_count", len(verified))
    print("verified_tail", verified[-20:])
    print("bad_count", len(bad))
    print("bad_tail", bad[-10:])
    print("partial_count", len(partial))
    print("partial_tail", partial[-10:])
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
