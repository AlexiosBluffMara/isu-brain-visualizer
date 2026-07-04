"""OpenNeuro ds004199 subset downloader via AWS S3 (no-sign-request)."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

S3_BASE = "s3://openneuro.org/ds004199"
DATASET_FILES = ["participants.tsv", "dataset_description.json"]


def _require_aws() -> None:
    if shutil.which("aws") is None:
        sys.exit(
            "ERROR: 'aws' CLI not found on PATH.\n"
            "Install it from https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html\n"
            "No AWS account is required — downloads use --no-sign-request."
        )


def _s3_cp(src: str, dst: Path, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "cp", "--no-sign-request", src, str(dst)]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
    else:
        subprocess.run(cmd, check=True)


def _s3_ls(prefix: str) -> list[str]:
    """Return list of S3 keys under prefix."""
    result = subprocess.run(
        ["aws", "s3", "ls", "--no-sign-request", prefix],
        capture_output=True,
        text=True,
        check=True,
    )
    keys = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts:
            keys.append(parts[-1])
    return keys


def _modality_suffix(modality: str) -> str:
    """Map modality name to NIfTI filename suffix fragment."""
    mapping = {
        "T1w": "_T1w",
        "FLAIR": "_FLAIR",
    }
    return mapping.get(modality, f"_{modality}")


def parse_participants_tsv(tsv_path: Path, group: str = "all") -> list[str]:
    """Parse participants.tsv and return subject IDs filtered by group.

    Args:
        tsv_path: Path to the participants.tsv file.
        group:    One of ``"fcd"``, ``"hc"``, or ``"all"`` (default).
                  When ``"all"`` every row is included regardless of the
                  ``group`` column value.

    Returns:
        Ordered list of ``participant_id`` strings that match *group*.
    """
    subjects: list[str] = []
    with tsv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            pid = row.get("participant_id", "").strip()
            if not pid:
                continue
            if group == "all":
                subjects.append(pid)
            else:
                row_group = row.get("group", "").strip().lower()
                if row_group == group.lower():
                    subjects.append(pid)
    return subjects


def download_dataset_files(destination: Path, dry_run: bool) -> None:
    for fname in DATASET_FILES:
        _s3_cp(f"{S3_BASE}/{fname}", destination / fname, dry_run)


def download_subject(
    subject: str,
    destination: Path,
    modalities: list[str],
    include_roi: bool,
    dry_run: bool,
) -> None:
    anat_prefix = f"{S3_BASE}/{subject}/anat/"
    try:
        remote_files = _s3_ls(anat_prefix)
    except subprocess.CalledProcessError as exc:
        print(f"WARNING: Could not list {anat_prefix}: {exc}", file=sys.stderr)
        return

    suffixes = [_modality_suffix(m) for m in modalities]
    local_anat = destination / subject / "anat"

    for fname in remote_files:
        # fname is just the filename, not a full key
        fpath = fname.strip()
        if not (fpath.endswith(".nii.gz") or fpath.endswith(".json")):
            continue

        is_roi = fpath.endswith("_roi.nii.gz")
        if is_roi and not include_roi:
            continue

        # For non-ROI files, check if it matches a requested modality
        if not is_roi:
            if not any(suf in fpath for suf in suffixes):
                continue

        src = f"{anat_prefix}{fpath}"
        dst = local_anat / fpath
        _s3_cp(src, dst, dry_run)


def download(
    destination: Path,
    subjects: list[str] | None,
    modalities: list[str],
    include_roi: bool,
    dry_run: bool,
    *,
    all_participants: bool = False,
    group: str = "all",
    limit: int | None = None,
) -> None:
    """Download a ds004199 subset.

    Either *subjects* (explicit list) **or** *all_participants=True* must be
    provided — not both.

    When *all_participants* is ``True``:

    * If ``participants.tsv`` is absent from *destination*:

      - **Normal mode** — the dataset metadata files are fetched from S3
        (``dry_run=False``) so that ``participants.tsv`` is present locally
        and can be parsed for subject expansion.
      - **Dry-run mode** — the metadata ``aws s3 cp`` commands are printed
        (not executed) and the function returns immediately, because subject
        expansion requires a local ``participants.tsv`` that does not yet
        exist.

    * The TSV is then parsed and filtered by *group* (``"fcd"``, ``"hc"``,
      or ``"all"``).
    * If *limit* is set the list is truncated to at most *N* subjects.

    When *subjects* is provided it is used verbatim — *group* and *limit*
    are ignored (explicit subjects are preserved).
    """
    _require_aws()
    destination.mkdir(parents=True, exist_ok=True)

    if all_participants:
        tsv_path = destination / "participants.tsv"
        if not tsv_path.exists():
            if dry_run:
                download_dataset_files(destination, dry_run=True)
                print(
                    "NOTE: Subject expansion requires a local participants.tsv — "
                    "re-run without --dry-run to fetch it, then retry."
                )
                return
            print(
                "participants.tsv not found locally — "
                "fetching dataset metadata first…"
            )
            download_dataset_files(destination, dry_run=False)
        resolved = parse_participants_tsv(tsv_path, group=group)
        if limit is not None:
            resolved = resolved[:limit]
        effective_subjects: list[str] = resolved
    else:
        if subjects is None:
            raise ValueError(
                "Either subjects or all_participants=True must be provided."
            )
        effective_subjects = subjects

    download_dataset_files(destination, dry_run)
    for subject in effective_subjects:
        print(f"Downloading subject {subject}…")
        download_subject(subject, destination, modalities, include_roi, dry_run)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download a ds004199 subset from OpenNeuro S3 (no credentials needed)."
    )
    p.add_argument(
        "--destination",
        required=True,
        type=Path,
        help="Local directory to save files into (e.g. data/ds004199).",
    )

    # Exactly one of --subjects / --all-participants is required.
    subject_src = p.add_mutually_exclusive_group(required=True)
    subject_src.add_argument(
        "--subjects",
        nargs="+",
        metavar="SUB",
        help="Explicit subject IDs to download (e.g. sub-00058 sub-00001).",
    )
    subject_src.add_argument(
        "--all-participants",
        action="store_true",
        help=(
            "Download all participants listed in participants.tsv. "
            "If the file is not present locally it is fetched automatically. "
            "Use --group and --limit to narrow the selection."
        ),
    )

    p.add_argument(
        "--group",
        choices=["fcd", "hc", "all"],
        default="all",
        help=(
            "Filter participants by diagnostic group when using "
            "--all-participants: fcd, hc, or all (default: all)."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum number of subjects to download when using "
            "--all-participants."
        ),
    )
    p.add_argument(
        "--modalities",
        nargs="+",
        default=["T1w", "FLAIR"],
        metavar="MOD",
        help="Modalities to download: T1w, FLAIR (default: T1w FLAIR).",
    )
    p.add_argument(
        "--include-roi",
        action="store_true",
        help="Also download *_roi.nii.gz masks (FCD subjects only).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print aws commands without executing them.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    download(
        destination=args.destination,
        subjects=args.subjects,
        modalities=args.modalities,
        include_roi=args.include_roi,
        dry_run=args.dry_run,
        all_participants=args.all_participants,
        group=args.group,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
