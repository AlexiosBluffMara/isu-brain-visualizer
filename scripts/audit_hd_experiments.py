#!/usr/bin/env python3
import json
from pathlib import Path


def main() -> int:
    root = Path("outputs/experiments_hd")
    if not root.exists():
        print("no_experiments_dir")
        return 0

    bad = []
    for exp_dir in sorted(p for p in root.glob("sub-00058_*") if p.is_dir()):
        manifest_path = exp_dir / "sub-00058" / "manifest.json"
        if not manifest_path.exists():
            print(exp_dir.name, "no_manifest")
            continue

        manifest = json.loads(manifest_path.read_text())
        status = manifest.get("stage_status", {})
        artifacts = manifest.get("artifacts", {})
        ok = (
            status.get("splat_ply") == "ok"
            and status.get("gaussian_field") == "ok"
            and str(status.get("optimized_gaussian_field_audit", "")).startswith("ok")
            and bool(artifacts.get("patient_splat_lesion_highlighted_ply"))
            and bool(artifacts.get("optimized_gaussian_splat_ply"))
        )
        state = "ok" if ok else "bad"
        if not ok:
            bad.append(exp_dir.name)
        print(
            exp_dir.name,
            state,
            "gaussian_field=" + str(status.get("gaussian_field")),
            "audit=" + str(status.get("optimized_gaussian_field_audit")),
        )

    print("bad_experiment_count", len(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
