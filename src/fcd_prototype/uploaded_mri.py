"""Single-scan MRI intake for non-BIDS uploaded MRI volumes.

This wraps the same end-to-end machinery used by ds004199 so an arbitrary
uploaded FLAIR/T2w/T1w NIfTI or DICOM series can produce viewer-ready splats,
fuzzing, synthetic normal, and residual artifacts without pretending it is part
of a BIDS dataset.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import nibabel as nib

from fcd_prototype.bids_discovery import SubjectRecord, discover_dataset
from fcd_prototype.candidate_roi import PROPOSAL_MODES
from fcd_prototype.end_to_end import (
    RoiStrategy,
    SyntheticMode,
    _data_quality_stage_status,
    _is_control_group,
    _pick_control_prior_volume,
    build_product_readiness,
    process_subject,
)


class DicomConversionError(RuntimeError):
    """Raised when an uploaded DICOM series cannot be converted safely."""


def _safe_subject_id(raw: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip())
    safe = safe.strip("._-")
    return safe or "uploaded_mri"


def _is_nifti_path(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _validate_nifti(path: Optional[Path], label: str) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} NIfTI does not exist: {path}")
    if not _is_nifti_path(path):
        raise ValueError(f"{label} must be a .nii or .nii.gz file: {path}")
    try:
        nib.load(str(path))
    except Exception as exc:
        raise ValueError(f"{label} is not a readable NIfTI file: {path} ({exc})") from exc
    return path


def _readiness_from_issues(
    blockers: list[str],
    review_items: list[str],
    *,
    pass_interpretation: str,
    review_interpretation: str,
    fail_interpretation: str,
) -> dict[str, object]:
    blockers = sorted(dict.fromkeys(blockers))
    review_items = sorted(dict.fromkeys(review_items))
    status = "fail" if blockers else ("review" if review_items else "pass")
    return {
        "status": status,
        "blockers": blockers,
        "review_items": review_items,
        "blocking_issue_count": len(blockers),
        "review_issue_count": len(review_items),
        "can_attempt_processing": status != "fail",
        "can_support_paper_claim": status == "pass",
        "interpretation": (
            pass_interpretation
            if status == "pass"
            else (review_interpretation if status == "review" else fail_interpretation)
        ),
    }


def _tail_text(text: str, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _resolve_dcm2niix(dcm2niix_bin: str | Path) -> str:
    candidate = Path(dcm2niix_bin)
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(str(dcm2niix_bin))
    if found is None:
        raise DicomConversionError(
            "dcm2niix converter was not found. Install dcm2niix and ensure it is "
            "on PATH, or pass --dcm2niix-bin /path/to/dcm2niix."
        )
    return found


def _validate_dicom_dir(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} DICOM directory does not exist: {path}")
    if not any(child.is_file() for child in path.rglob("*")):
        raise DicomConversionError(f"{label} DICOM directory contains no files: {path}")
    return path


def _dicom_dir_inventory(path: Path, max_samples: int = 20) -> dict[str, object]:
    """Return lightweight file inventory for conversion provenance.

    We intentionally do not parse PHI-bearing DICOM headers here. The purpose is
    to record enough filesystem evidence to debug conversion and detect mixed
    studies without storing patient metadata in the manifest.
    """
    path = Path(path)
    files = sorted(child for child in path.rglob("*") if child.is_file())
    suffix_counts: dict[str, int] = {}
    total_bytes = 0
    samples: list[str] = []
    for idx, child in enumerate(files):
        suffix = child.suffix.lower() or "<none>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        try:
            total_bytes += int(child.stat().st_size)
        except OSError:
            pass
        if idx < max_samples:
            try:
                samples.append(child.relative_to(path).as_posix())
            except ValueError:
                samples.append(child.name)

    directories = [child for child in path.rglob("*") if child.is_dir()]
    return {
        "input_dir": str(path),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "nested_directory_count": len(directories),
        "sample_relative_paths": samples,
    }


def _nifti_candidates(output_dir: Path) -> list[Path]:
    return sorted(
        [path for path in output_dir.iterdir() if path.is_file() and _is_nifti_path(path)],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


_SIDECAR_SAFE_FIELDS = (
    "Modality",
    "MRAcquisitionType",
    "ScanningSequence",
    "SequenceVariant",
    "ScanOptions",
    "ImageType",
    "SliceThickness",
    "SpacingBetweenSlices",
    "EchoTime",
    "RepetitionTime",
    "InversionTime",
    "FlipAngle",
    "MagneticFieldStrength",
    "ConversionSoftware",
    "ConversionSoftwareVersion",
    "ImageOrientationPatientDICOM",
)

_SIDECAR_TEXT_FIELDS = (
    "SeriesDescription",
    "ProtocolName",
    "SequenceName",
    "ImageType",
    "ScanningSequence",
    "SequenceVariant",
)

_SERIES_TERM_PATTERNS = {
    "flair": ("flair",),
    "t1": ("t1", "mprage", "spgr", "bravo", "tfl"),
    "t2": ("t2",),
    "swi": ("swi", "swan"),
    "localizer_or_scout": ("localizer", "scout", "survey", "3-plane", "3_plane", "locator"),
    "diffusion_or_adc": ("dwi", "diffusion", "adc", "tracew", "trace-weighted"),
    "bold_or_fmri": ("bold", "fmri", "resting", "rest-state", "rest_state"),
    "perfusion": ("perfusion", "asl", "cbf", "dsc"),
    "angiography": ("angio", "mra", "tof"),
}

_STRUCTURAL_BLOCKER_TERMS = (
    "localizer_or_scout",
    "diffusion_or_adc",
    "bold_or_fmri",
    "perfusion",
    "angiography",
)

_ROI_FILENAME_TERMS = ("roi", "mask", "seg", "lesion")


def _detect_terms_from_text(text: str) -> list[str]:
    text = text.lower()
    found: list[str] = []
    for term, needles in _SERIES_TERM_PATTERNS.items():
        if any(needle in text for needle in needles):
            found.append(term)
    return found


def _infer_structural_label_from_terms(terms: list[str]) -> Optional[str]:
    """Infer one supported structural MRI role from filename/series terms.

    FLAIR sequence names often include T2 wording, so FLAIR is allowed to win
    over T2 when both are present. T1 mixed with FLAIR/T2 remains ambiguous.
    """
    term_set = set(terms)
    if "flair" in term_set and "t1" not in term_set:
        return "FLAIR"
    if "t2" in term_set and "t1" not in term_set and "flair" not in term_set:
        return "T2w"
    if "t1" in term_set and "t2" not in term_set and "flair" not in term_set:
        return "T1w"
    return None


def _infer_structural_label_from_path(path: Path) -> tuple[Optional[str], list[str]]:
    terms = _detect_terms_from_text(str(path.name))
    return _infer_structural_label_from_terms(terms), terms


def _infer_modality_from_sidecar(
    sidecar: dict[str, object],
) -> tuple[Optional[str], list[str]]:
    """Infer a structural modality label from NIfTI JSON sidecar text fields.

    Inspects ``SeriesDescription``, ``ProtocolName``, and related BIDS sidecar
    fields (via :func:`_detect_series_terms`) for FLAIR, T2w, or T1w indicator
    words.  Returns ``(label, terms)`` where *label* is ``None`` when the
    sidecar text is ambiguous, non-structural, or contains no recognised terms.
    """
    terms = _detect_series_terms(sidecar)
    return _infer_structural_label_from_terms(terms), terms


def _resolve_auto_structural_upload(
    *,
    scan: Optional[Path],
    scan_dicom_dir: Optional[Path],
    flair: Optional[Path],
    t2w: Optional[Path],
    t1w: Optional[Path],
    flair_dicom_dir: Optional[Path],
    t2w_dicom_dir: Optional[Path],
    t1w_dicom_dir: Optional[Path],
    strict: bool = False,
) -> tuple[dict[str, Optional[Path]], dict[str, object]]:
    """Route a single generic structural upload to FLAIR/T2w/T1w if safe.

    For NIfTI ``--scan`` paths a co-located JSON sidecar (e.g. ``scan.json``
    next to ``scan.nii.gz``) is inspected for ``SeriesDescription`` and
    ``ProtocolName`` when the filename alone cannot infer the modality.  This
    allows a generic ``scan.nii.gz`` to route correctly when a BIDS-compatible
    sidecar is present.

    Conflict detection: when *both* the filename and the sidecar independently
    infer a modality and they disagree (e.g. filename says ``T1w`` but the
    sidecar says ``FLAIR``), resolution fails with the blocker
    ``auto_scan_modality_conflict``.  Use an explicit modality flag
    (``--flair``, ``--t2w``, or ``--t1w``) to override.

    Full inference provenance is recorded in the returned *meta* dict under
    ``filename_inferred_label``, ``sidecar_inferred_label``,
    ``inference_source``, ``sidecar_path``, ``sidecar_exists``, and
    ``sidecar_terms``.
    """
    updated = {
        "flair": flair,
        "t2w": t2w,
        "t1w": t1w,
        "flair_dicom_dir": flair_dicom_dir,
        "t2w_dicom_dir": t2w_dicom_dir,
        "t1w_dicom_dir": t1w_dicom_dir,
    }
    meta: dict[str, object] = {
        "provided": bool(scan is not None or scan_dicom_dir is not None),
        "kind": None,
        "path": None,
        # --- filename provenance ---
        "filename_terms": [],
        "filename_inferred_label": None,
        # --- sidecar provenance (NIfTI --scan only) ---
        "sidecar_path": None,
        "sidecar_exists": False,
        "sidecar_json_parse_ok": False,
        "sidecar_json_error": None,
        "sidecar_terms": [],
        "sidecar_inferred_label": None,
        # --- resolved ---
        "inference_source": None,
        "inferred_label": None,
        "status": "not_applicable",
        "blockers": [],
    }

    def _fail(code: str, message: str) -> tuple[dict[str, Optional[Path]], dict[str, object]]:
        meta["status"] = "fail"
        meta["blockers"] = [code]
        meta["message"] = message
        if strict:
            raise ValueError(message)
        return updated, meta

    if scan is None and scan_dicom_dir is None:
        return updated, meta
    if scan is not None and scan_dicom_dir is not None:
        return _fail(
            "auto_scan_multiple_sources",
            "Provide either --scan or --scan-dicom-dir, not both.",
        )

    explicit_sources = [
        name for name, value in (
            ("--flair", flair),
            ("--t2w", t2w),
            ("--t1w", t1w),
            ("--flair-dicom-dir", flair_dicom_dir),
            ("--t2w-dicom-dir", t2w_dicom_dir),
            ("--t1w-dicom-dir", t1w_dicom_dir),
        )
        if value is not None
    ]
    if explicit_sources:
        return _fail(
            "auto_scan_conflicts_with_explicit_modality",
            "--scan/--scan-dicom-dir cannot be combined with explicit structural "
            f"modality inputs: {', '.join(explicit_sources)}.",
        )

    source_path = Path(scan if scan is not None else scan_dicom_dir)
    filename_inferred, filename_terms = _infer_structural_label_from_path(source_path)
    meta.update({
        "kind": "nifti" if scan is not None else "dicom_dir",
        "path": str(source_path),
        "filename_terms": filename_terms,
        "filename_inferred_label": filename_inferred,
    })

    # ------------------------------------------------------------------
    # Sidecar-based inference (NIfTI --scan only).
    # A co-located .json file (e.g. scan.json beside scan.nii.gz) is read
    # and its SeriesDescription / ProtocolName are checked for structural
    # modality terms.  DICOM directories have no pre-conversion sidecar.
    # ------------------------------------------------------------------
    sidecar_inferred: Optional[str] = None
    sidecar_terms: list[str] = []
    if scan is not None:
        sidecar_path_obj = _nifti_sidecar_path(source_path)
        sidecar_path_str = str(sidecar_path_obj)
        sidecar_exists = sidecar_path_obj.is_file()
        meta["sidecar_path"] = sidecar_path_str
        meta["sidecar_exists"] = sidecar_exists
        if sidecar_exists:
            try:
                raw = json.loads(sidecar_path_obj.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    meta["sidecar_json_parse_ok"] = True
                    sidecar_inferred, sidecar_terms = _infer_modality_from_sidecar(raw)
                else:
                    meta["sidecar_json_error"] = "sidecar_json_not_object"
            except Exception:
                # Unparseable sidecar: treat as absent for routing, but keep
                # the provenance so preflight can explain why it was ignored.
                meta["sidecar_json_error"] = "sidecar_json_parse_failed"
        meta["sidecar_terms"] = sidecar_terms
        meta["sidecar_inferred_label"] = sidecar_inferred

    # ------------------------------------------------------------------
    # Resolve final label with conflict detection.
    # ------------------------------------------------------------------
    if filename_inferred is not None and sidecar_inferred is not None:
        if filename_inferred != sidecar_inferred:
            # Filename and sidecar disagree — fail loudly rather than guess.
            meta["inference_source"] = "conflict"
            meta["inferred_label"] = None
            return _fail(
                "auto_scan_modality_conflict",
                f"Filename suggests {filename_inferred!r} but JSON sidecar "
                f"suggests {sidecar_inferred!r} for {source_path.name!r}. "
                "Use --flair, --t2w, or --t1w explicitly.",
            )
        final_label: Optional[str] = filename_inferred
        inference_source: Optional[str] = "filename_and_sidecar"
    elif filename_inferred is not None:
        final_label = filename_inferred
        inference_source = "filename"
    elif sidecar_inferred is not None:
        final_label = sidecar_inferred
        inference_source = "sidecar"
    else:
        final_label = None
        inference_source = None

    meta["inference_source"] = inference_source
    meta["inferred_label"] = final_label

    if final_label is None:
        source_desc = (
            f"{source_path.name!r} or its JSON sidecar"
            if scan is not None
            else f"{source_path.name!r}"
        )
        return _fail(
            "auto_scan_modality_not_inferred",
            "Could not infer whether the uploaded scan is FLAIR, T2w, or T1w "
            f"from {source_desc}. Use --flair, --t2w, --t1w, "
            "--flair-dicom-dir, --t2w-dicom-dir, or --t1w-dicom-dir explicitly.",
        )

    if scan is not None:
        key = {"FLAIR": "flair", "T2w": "t2w", "T1w": "t1w"}[final_label]
    else:
        key = {"FLAIR": "flair_dicom_dir", "T2w": "t2w_dicom_dir", "T1w": "t1w_dicom_dir"}[final_label]
    updated[key] = source_path
    meta["status"] = "pass"
    return updated, meta


def _nifti_label_preflight(path: Optional[Path], label: str) -> dict[str, object]:
    """Check one uploaded NIfTI path before the expensive pipeline runs."""
    blockers: list[str] = []
    review_items: list[str] = []
    info: dict[str, object] = {
        "label": label,
        "path": str(path) if path is not None else None,
        "kind": "nifti",
        "exists": False,
        "is_nifti": False,
        "readable": False,
        "filename_terms": [],
        "blockers": blockers,
        "review_items": review_items,
    }
    if path is None:
        blockers.append("missing_input")
    else:
        path = Path(path)
        info["exists"] = path.is_file()
        info["suffix"] = ".nii.gz" if path.name.lower().endswith(".nii.gz") else path.suffix.lower()
        info["is_nifti"] = _is_nifti_path(path)
        terms = _detect_terms_from_text(path.name)
        info["filename_terms"] = terms
        if not path.is_file():
            blockers.append("missing_file")
        elif not _is_nifti_path(path):
            blockers.append("unsupported_file_type")
        else:
            try:
                img = nib.load(str(path))
                info["readable"] = True
                info["shape"] = [int(v) for v in img.shape]
                info["dtype"] = str(img.header.get_data_dtype())
            except Exception as exc:
                blockers.append("unreadable_nifti")
                info["error"] = str(exc)

        if label.upper() in {"FLAIR", "T2W", "T1W"}:
            expected_terms = _expected_modality_terms(label)
            for term in _STRUCTURAL_BLOCKER_TERMS:
                if term in terms:
                    blockers.append(f"filename_suggests_{term}")
            if expected_terms and not any(term in terms for term in expected_terms):
                review_items.append(f"filename_expected_{expected_terms[0]}_term_not_detected")
            if label.upper() == "FLAIR" and "t1" in terms:
                review_items.append("filename_suggests_t1_for_flair_input")
            if label.upper() == "T2W" and "flair" in terms:
                review_items.append("filename_suggests_flair_for_t2w_input")
            if label.upper() == "T2W" and "t1" in terms:
                review_items.append("filename_suggests_t1_for_t2w_input")
            if label.upper() == "T1W" and "flair" in terms:
                review_items.append("filename_suggests_flair_for_t1w_input")
            if label.upper() == "T1W" and "t2" in terms:
                review_items.append("filename_suggests_t2_for_t1w_input")
        elif label.upper() == "ROI" and path is not None and path.is_file() and _is_nifti_path(path):
            name = path.name.lower()
            if not any(term in name for term in _ROI_FILENAME_TERMS):
                review_items.append("filename_does_not_suggest_roi_or_mask")

    readiness = _readiness_from_issues(
        blockers,
        review_items,
        pass_interpretation="nifti_input_looks_compatible",
        review_interpretation="nifti_input_requires_label_or_mask_review",
        fail_interpretation="nifti_input_not_processable",
    )
    info.update(readiness)
    return info


def _dicom_label_preflight(
    path: Optional[Path],
    label: str,
    *,
    dcm2niix_bin: str | Path,
) -> dict[str, object]:
    """Check one uploaded DICOM directory before conversion."""
    blockers: list[str] = []
    review_items: list[str] = []
    info: dict[str, object] = {
        "label": label,
        "path": str(path) if path is not None else None,
        "kind": "dicom_dir",
        "exists": False,
        "is_dir": False,
        "requires_conversion": True,
        "converter": str(dcm2niix_bin),
        "converter_available": False,
        "blockers": blockers,
        "review_items": review_items,
    }
    converter_candidate = Path(dcm2niix_bin)
    converter_path = str(converter_candidate) if converter_candidate.is_file() else shutil.which(str(dcm2niix_bin))
    if converter_path is None:
        blockers.append("dcm2niix_converter_not_found")
    else:
        info["converter_available"] = True
        info["resolved_converter"] = converter_path

    if path is None:
        blockers.append("missing_input")
    else:
        path = Path(path)
        info["exists"] = path.exists()
        info["is_dir"] = path.is_dir()
        if not path.exists():
            blockers.append("missing_directory")
        elif not path.is_dir():
            blockers.append("not_a_directory")
        else:
            inventory = _dicom_dir_inventory(path)
            info["input_inventory"] = inventory
            if int(inventory.get("file_count", 0)) == 0:
                blockers.append("dicom_directory_contains_no_files")

    readiness = _readiness_from_issues(
        blockers,
        review_items,
        pass_interpretation="dicom_directory_ready_for_dcm2niix_conversion",
        review_interpretation="dicom_directory_requires_manual_review",
        fail_interpretation="dicom_directory_not_processable",
    )
    info.update(readiness)
    info["can_support_paper_claim"] = False
    info["paper_claim_requires"] = "successful dcm2niix conversion plus structural-MR JSON sidecar safety"
    return info


def build_upload_input_compatibility(
    *,
    scan: Optional[Path] = None,
    scan_dicom_dir: Optional[Path] = None,
    flair: Optional[Path],
    t2w: Optional[Path] = None,
    t1w: Optional[Path] = None,
    roi: Optional[Path] = None,
    flair_dicom_dir: Optional[Path] = None,
    t2w_dicom_dir: Optional[Path] = None,
    t1w_dicom_dir: Optional[Path] = None,
    dcm2niix_bin: str | Path = "dcm2niix",
) -> dict[str, object]:
    """Return a machine-readable preflight report for arbitrary MRI uploads."""
    inputs: dict[str, dict[str, object]] = {}
    blockers: list[str] = []
    review_items: list[str] = []
    auto_resolution_input = _resolve_auto_structural_upload(
        scan=scan,
        scan_dicom_dir=scan_dicom_dir,
        flair=flair,
        t2w=t2w,
        t1w=t1w,
        flair_dicom_dir=flair_dicom_dir,
        t2w_dicom_dir=t2w_dicom_dir,
        t1w_dicom_dir=t1w_dicom_dir,
        strict=False,
    )
    resolved, auto_resolution = auto_resolution_input
    flair = resolved["flair"]
    t2w = resolved["t2w"]
    t1w = resolved["t1w"]
    flair_dicom_dir = resolved["flair_dicom_dir"]
    t2w_dicom_dir = resolved["t2w_dicom_dir"]
    t1w_dicom_dir = resolved["t1w_dicom_dir"]
    for item in auto_resolution.get("blockers", []):
        blockers.append(f"auto_scan:{item}")

    modality_sources = {
        "FLAIR": [source for source, value in (("nifti", flair), ("dicom_dir", flair_dicom_dir)) if value is not None],
        "T2w": [source for source, value in (("nifti", t2w), ("dicom_dir", t2w_dicom_dir)) if value is not None],
        "T1w": [source for source, value in (("nifti", t1w), ("dicom_dir", t1w_dicom_dir)) if value is not None],
    }
    for label, sources in modality_sources.items():
        if len(sources) > 1:
            blockers.append(f"{label}:multiple_sources_for_modality")

    if flair is not None:
        inputs["FLAIR"] = _nifti_label_preflight(flair, "FLAIR")
    elif flair_dicom_dir is not None:
        inputs["FLAIR"] = _dicom_label_preflight(flair_dicom_dir, "FLAIR", dcm2niix_bin=dcm2niix_bin)
    if t2w is not None:
        inputs["T2w"] = _nifti_label_preflight(t2w, "T2w")
    elif t2w_dicom_dir is not None:
        inputs["T2w"] = _dicom_label_preflight(t2w_dicom_dir, "T2w", dcm2niix_bin=dcm2niix_bin)
    if t1w is not None:
        inputs["T1w"] = _nifti_label_preflight(t1w, "T1w")
    elif t1w_dicom_dir is not None:
        inputs["T1w"] = _dicom_label_preflight(t1w_dicom_dir, "T1w", dcm2niix_bin=dcm2niix_bin)
    if roi is not None:
        inputs["ROI"] = _nifti_label_preflight(roi, "ROI")

    primary_labels = [label for label in ("FLAIR", "T2w", "T1w") if label in inputs]
    if not primary_labels:
        blockers.append("no_primary_mri_input")

    for label, info in inputs.items():
        for item in info.get("blockers", []):
            blockers.append(f"{label}:{item}")
        for item in info.get("review_items", []):
            review_items.append(f"{label}:{item}")

    readiness = _readiness_from_issues(
        blockers,
        review_items,
        pass_interpretation="upload_inputs_ready_for_processing",
        review_interpretation="upload_inputs_processable_but_require_review",
        fail_interpretation="upload_inputs_not_processable",
    )
    return {
        "status": readiness["status"],
        "auto_resolution": auto_resolution,
        "inputs": inputs,
        "primary_modalities": primary_labels,
        "blockers": readiness["blockers"],
        "review_items": readiness["review_items"],
        "blocking_issue_count": readiness["blocking_issue_count"],
        "review_issue_count": readiness["review_issue_count"],
        "can_attempt_processing": readiness["can_attempt_processing"],
        "can_support_paper_claim": (
            readiness["status"] == "pass"
            and all(not info.get("requires_conversion") for info in inputs.values())
        ),
        "interpretation": readiness["interpretation"],
        "paper_risk_summary": [
            "A readable NIfTI can still be mislabeled as the wrong structural modality.",
            "DICOM directories must be converted with dcm2niix and then pass structural-MR sidecar checks.",
            "Unsupported screenshots, archives, localizers, diffusion/ADC, BOLD/fMRI, perfusion, and angiography inputs cannot support the MRI-to-3DGS paper claim.",
        ],
    }


def _nifti_sidecar_path(path: Path) -> Path:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return path.with_name(name[:-7] + ".json")
    return path.with_suffix(".json")


def _json_safe_value(value):
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sidecar_text_blob(sidecar: dict[str, object]) -> str:
    parts: list[str] = []
    for key in _SIDECAR_TEXT_FIELDS:
        value = sidecar.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).lower()


def _detect_series_terms(sidecar: dict[str, object]) -> list[str]:
    text = _sidecar_text_blob(sidecar)
    found: list[str] = []
    for term, needles in _SERIES_TERM_PATTERNS.items():
        if any(needle in text for needle in needles):
            found.append(term)
    return found


def _expected_modality_terms(label: str) -> tuple[str, ...]:
    lower = label.strip().lower()
    if "flair" in lower:
        return ("flair",)
    if lower in {"t2", "t2w"} or "t2" in lower:
        return ("t2",)
    if lower in {"t1", "t1w"} or "t1" in lower:
        return ("t1",)
    return ()


def _conversion_readiness_from_warnings(warnings: list[str]) -> dict[str, object]:
    blockers = [
        warning for warning in warnings
        if warning == "sidecar_modality_not_mr"
        or warning.startswith("sidecar_series_suggests_")
    ]
    review_items = [warning for warning in warnings if warning not in blockers]
    status = "fail" if blockers else ("review" if review_items else "pass")
    return {
        "status": status,
        "blockers": sorted(dict.fromkeys(blockers)),
        "review_items": sorted(dict.fromkeys(review_items)),
        "blocking_issue_count": len(set(blockers)),
        "review_issue_count": len(set(review_items)),
        "can_use_converted_nifti_for_3dgs": status != "fail",
        "can_use_converted_nifti_for_paper_claim": status == "pass",
        "interpretation": (
            "converted_structural_mri"
            if status == "pass"
            else (
                "converted_mri_requires_series_review"
                if status == "review"
                else "converted_series_not_safe_for_structural_fcd_pipeline"
            )
        ),
        "paper_risk_summary": [
            "A valid NIfTI file can still be the wrong DICOM series for FCD analysis.",
            "Localizers, diffusion/ADC, BOLD/fMRI, perfusion, angiography, or non-MR conversions cannot support this structural MRI-to-3DGS pipeline.",
            "Missing or inconclusive dcm2niix sidecars require manual sequence review before paper-facing claims.",
        ],
    }


def _summarize_dcm2niix_sidecar(nifti_path: Path, label: str) -> dict[str, object]:
    sidecar_path = _nifti_sidecar_path(nifti_path)
    warnings: list[str] = []
    summary: dict[str, object] = {
        "path": str(sidecar_path),
        "exists": sidecar_path.is_file(),
        "json_parse_ok": False,
        "safe_fields": {},
        "series_descriptor_terms": [],
        "warnings": warnings,
    }
    if not sidecar_path.is_file():
        warnings.append("sidecar_missing")
        summary["conversion_readiness"] = _conversion_readiness_from_warnings(warnings)
        return summary

    try:
        raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append("sidecar_json_parse_failed")
        summary["error"] = str(exc)
        summary["conversion_readiness"] = _conversion_readiness_from_warnings(warnings)
        return summary

    if not isinstance(raw, dict):
        warnings.append("sidecar_json_not_object")
        summary["conversion_readiness"] = _conversion_readiness_from_warnings(warnings)
        return summary

    safe_fields = {
        key: _json_safe_value(raw[key])
        for key in _SIDECAR_SAFE_FIELDS
        if key in raw
    }
    terms = _detect_series_terms(raw)
    modality = str(raw.get("Modality", "")).strip().upper()
    image_type = raw.get("ImageType", [])
    image_type_values = [str(v).lower() for v in image_type] if isinstance(image_type, list) else [str(image_type).lower()]

    if modality and modality != "MR":
        warnings.append("sidecar_modality_not_mr")
    if not modality:
        warnings.append("sidecar_missing_modality")
    for term in _STRUCTURAL_BLOCKER_TERMS:
        if term in terms:
            warnings.append(f"sidecar_series_suggests_{term}")
    if any(value in {"derived", "secondary"} for value in image_type_values):
        warnings.append("sidecar_image_type_derived_or_secondary")
    if "ImageOrientationPatientDICOM" not in raw:
        warnings.append("sidecar_missing_image_orientation_patient")

    expected_terms = _expected_modality_terms(label)
    if expected_terms and not any(term in terms for term in expected_terms):
        warnings.append(f"sidecar_expected_{expected_terms[0]}_term_not_detected")

    summary.update(
        {
            "json_parse_ok": True,
            "safe_fields": safe_fields,
            "series_descriptor_terms": terms,
            "sidecar_keys_present": sorted(str(key) for key in raw.keys()),
        }
    )
    summary["conversion_readiness"] = _conversion_readiness_from_warnings(warnings)
    return summary


def _convert_dicom_series(
    *,
    dicom_dir: Path,
    output_dir: Path,
    label: str,
    dcm2niix_bin: str | Path = "dcm2niix",
    timeout_seconds: int = 300,
) -> tuple[Path, dict[str, object]]:
    """Convert one DICOM series directory to NIfTI using dcm2niix."""
    dicom_dir = _validate_dicom_dir(dicom_dir, label)
    input_inventory = _dicom_dir_inventory(dicom_dir)
    converter = _resolve_dcm2niix(dcm2niix_bin)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    before_niftis = {
        str(path.resolve()): path.stat().st_mtime
        for path in _nifti_candidates(output_dir)
    }

    filename = _safe_subject_id(label.lower())
    command = [
        converter,
        "-z",
        "y",
        "-f",
        filename,
        "-o",
        str(output_dir),
        str(dicom_dir),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise DicomConversionError(
            f"dcm2niix converter could not be executed: {converter}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise DicomConversionError(
            f"dcm2niix timed out after {timeout_seconds}s while converting {label}: {dicom_dir}"
        ) from exc

    if result.returncode != 0:
        raise DicomConversionError(
            f"dcm2niix failed converting {label} with exit code {result.returncode}. "
            f"stderr: {_tail_text(result.stderr or '').strip()}"
        )

    all_candidates = _nifti_candidates(output_dir)
    candidates = [
        path for path in all_candidates
        if str(path.resolve()) not in before_niftis
        or path.stat().st_mtime > before_niftis[str(path.resolve())]
    ]
    if not candidates:
        raise DicomConversionError(
            f"dcm2niix completed but produced no new .nii or .nii.gz file for {label} in {output_dir}."
        )
    if len(candidates) > 1:
        candidate_names = ", ".join(path.name for path in candidates)
        raise DicomConversionError(
            f"dcm2niix produced multiple NIfTI candidates for {label}: "
            f"{candidate_names}. Split the DICOM export into one series per "
            "directory or run dcm2niix manually and pass the selected .nii/.nii.gz "
            "with --flair, --t2w, or --t1w."
        )

    nifti_path = _validate_nifti(candidates[0], f"{label} converted NIfTI")
    assert nifti_path is not None
    sidecar = _summarize_dcm2niix_sidecar(nifti_path, label)
    conversion_readiness = sidecar["conversion_readiness"]
    if conversion_readiness["status"] == "fail":
        blockers = ", ".join(conversion_readiness["blockers"])
        raise DicomConversionError(
            f"dcm2niix converted {label}, but sidecar review found a series "
            f"unsafe for structural FCD processing: {blockers}"
        )
    meta = {
        "input_dir": str(dicom_dir),
        "input_inventory": input_inventory,
        "output_dir": str(output_dir),
        "output_nifti": str(nifti_path),
        "selected_output_nifti": str(nifti_path),
        "converter": converter,
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": _tail_text(result.stdout or ""),
        "stderr_tail": _tail_text(result.stderr or ""),
        "candidate_niftis": [str(path) for path in candidates],
        "all_output_niftis": [str(path) for path in all_candidates],
        "sidecar": sidecar,
        "conversion_warnings": list(sidecar.get("warnings", [])),
        "conversion_readiness": conversion_readiness,
    }
    return nifti_path, meta


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _path_key(path: Path) -> str:
    return str(Path(path).resolve())


def _discover_control_volumes(
    *,
    control_dataset_dir: Optional[Path],
    explicit_control_volumes: Optional[list[Path]],
    preferred_modality: str,
    excluded_subject_ids: Optional[set[str]] = None,
    excluded_volume_paths: Optional[list[Path]] = None,
) -> tuple[list[Path], dict[str, object]]:
    controls: list[Path] = []
    excluded_subject_keys = {
        item.strip().lower()
        for item in (excluded_subject_ids or set())
        if item and item.strip()
    }
    excluded_volume_keys = {
        _path_key(path)
        for path in (excluded_volume_paths or [])
        if path is not None and Path(path).is_file()
    }
    excluded_control_subjects: list[str] = []
    excluded_control_volumes: list[dict[str, str]] = []

    def _append_control(path: Path, *, source: str, subject_id: Optional[str] = None) -> None:
        validated = _validate_nifti(path, "control volume")
        if validated is None:
            return
        key = _path_key(validated)
        if key in excluded_volume_keys:
            excluded_control_volumes.append({
                "path": str(validated),
                "source": source,
                "subject_id": subject_id or "",
                "reason": "same_as_uploaded_volume",
            })
            return
        controls.append(validated)

    if explicit_control_volumes:
        for path in explicit_control_volumes:
            _append_control(path, source="explicit")

    if control_dataset_dir is not None:
        control_dataset_dir = Path(control_dataset_dir)
        if not control_dataset_dir.is_dir():
            raise FileNotFoundError(f"control dataset directory does not exist: {control_dataset_dir}")
        records = list(discover_dataset(control_dataset_dir))
        for record in records:
            if not _is_control_group(record.group):
                continue
            subject_key = record.subject_id.strip().lower()
            path = _pick_control_prior_volume(record, preferred_modality=preferred_modality)
            if subject_key in excluded_subject_keys:
                excluded_control_subjects.append(record.subject_id)
                if path is not None and path.is_file():
                    excluded_control_volumes.append({
                        "path": str(path),
                        "source": "control_dataset",
                        "subject_id": record.subject_id,
                        "reason": "same_subject_as_upload",
                    })
                continue
            if path is not None and path.is_file():
                _append_control(path, source="control_dataset", subject_id=record.subject_id)

    deduped = _dedupe_paths(controls)
    meta: dict[str, object] = {
        "leakage_guard_applied": bool(excluded_subject_keys or excluded_volume_keys),
        "excluded_subject_ids": sorted(excluded_subject_keys),
        "excluded_control_subjects": sorted(set(excluded_control_subjects)),
        "excluded_control_count": len(excluded_control_volumes),
        "excluded_control_volumes": excluded_control_volumes,
        "candidate_control_count_before_dedupe": len(controls),
        "resolved_control_count_after_dedupe": len(deduped),
    }
    return deduped, meta


def _summarize_dicom_conversion_safety(dicom_conversion: dict[str, object]) -> dict[str, object]:
    if not dicom_conversion:
        return {
            "status": "not_applicable",
            "converted_modalities": [],
            "blocking_issue_count": 0,
            "review_issue_count": 0,
            "blockers": [],
            "review_items": [],
        }

    blockers: list[str] = []
    review_items: list[str] = []
    converted_modalities: list[str] = []
    for modality, meta in sorted(dicom_conversion.items()):
        converted_modalities.append(str(modality))
        if not isinstance(meta, dict):
            review_items.append(f"{modality}:missing_conversion_metadata")
            continue
        readiness = meta.get("conversion_readiness", {})
        if not isinstance(readiness, dict):
            review_items.append(f"{modality}:missing_conversion_readiness")
            continue
        blockers.extend(f"{modality}:{item}" for item in readiness.get("blockers", []))
        review_items.extend(f"{modality}:{item}" for item in readiness.get("review_items", []))

    blockers = sorted(dict.fromkeys(blockers))
    review_items = sorted(dict.fromkeys(review_items))
    status = "fail" if blockers else ("review" if review_items else "pass")
    return {
        "status": status,
        "converted_modalities": converted_modalities,
        "blocking_issue_count": len(blockers),
        "review_issue_count": len(review_items),
        "blockers": blockers,
        "review_items": review_items,
        "can_use_converted_inputs_for_3dgs": status != "fail",
        "can_use_converted_inputs_for_paper_claim": status == "pass",
        "interpretation": (
            "all_converted_inputs_structural_mri"
            if status == "pass"
            else (
                "converted_inputs_require_series_review"
                if status == "review"
                else "converted_inputs_not_safe_for_structural_fcd_pipeline"
            )
        ),
    }


def _merge_upload_input_compatibility(manifest: dict, compatibility: dict[str, object]) -> None:
    """Attach upload preflight evidence and keep readiness fields consistent."""
    data_quality = manifest.setdefault("data_quality", {})
    data_quality["upload_input_compatibility"] = compatibility

    warnings = list(data_quality.get("warnings", []))
    for item in compatibility.get("review_items", []):
        warning = f"upload_input_compatibility:{item}"
        if warning not in warnings:
            warnings.append(warning)
    data_quality["warnings"] = warnings
    data_quality["warning_count"] = len(warnings)
    if warnings and data_quality.get("status") != "fail":
        data_quality["status"] = "warning"

    readiness = dict(data_quality.get("upload_readiness", {}))
    readiness_blockers = list(readiness.get("blockers", []))
    readiness_reviews = list(readiness.get("review_items", []))
    for item in compatibility.get("blockers", []):
        readiness_blockers.append(f"upload_input_compatibility:{item}")
    for item in compatibility.get("review_items", []):
        readiness_reviews.append(f"upload_input_compatibility:{item}")
    readiness_blockers = sorted(dict.fromkeys(readiness_blockers))
    readiness_reviews = sorted(dict.fromkeys(readiness_reviews))
    readiness["blockers"] = readiness_blockers
    readiness["review_items"] = readiness_reviews
    readiness["blocking_issue_count"] = len(readiness_blockers)
    readiness["review_issue_count"] = len(readiness_reviews)
    if readiness_blockers:
        readiness["status"] = "fail"
        readiness["can_run_3dgs"] = False
        readiness["can_subtract_patient_normal"] = False
    elif readiness_reviews and readiness.get("status") == "pass":
        readiness["status"] = "review"
    data_quality["upload_readiness"] = readiness

    manifest.setdefault("stage_status", {})["data_quality"] = _data_quality_stage_status(data_quality)
    manifest["product_readiness"] = build_product_readiness(manifest)


def build_uploaded_mri(
    *,
    scan: Optional[Path] = None,
    scan_dicom_dir: Optional[Path] = None,
    flair: Optional[Path],
    t2w: Optional[Path] = None,
    t1w: Optional[Path] = None,
    flair_dicom_dir: Optional[Path] = None,
    t2w_dicom_dir: Optional[Path] = None,
    t1w_dicom_dir: Optional[Path] = None,
    roi: Optional[Path] = None,
    output_root: Path,
    subject_id: str = "uploaded_mri",
    group: str = "uploaded",
    synthetic_mode: SyntheticMode = "control-prior",
    roi_strategy: RoiStrategy = "provided-or-auto",
    candidate_roi_proposal_mode: str = "brightness-gradient",
    control_dataset_dir: Optional[Path] = None,
    control_volumes: Optional[list[Path]] = None,
    max_splat_points: int = 50_000,
    presentation_splat_points: int = 300_000,
    splat_sampling_mode: str = "surface",
    presentation_splat_sampling_mode: str = "volume",
    paired_gaussian_support_basis: str = "union",
    compute_control_gaussian_prior_residual: bool = False,
    control_gaussian_prior_max_points: Optional[int] = None,
    control_gaussian_prior_min_controls: int = 2,
    control_gaussian_prior_heatmap_percentile: float = 99.0,
    control_gaussian_prior_smooth_sigma: float = 1.0,
    control_gaussian_prior_heatmap_method: str = "gaussian",
    control_gaussian_prior_heatmap_exact_max_points: int = 50_000,
    control_gaussian_prior_std_floor_fraction: float = 0.05,
    compute_splat_reconstruction_audit: bool = True,
    splat_reconstruction_audit_max_points: int = 100_000,
    splat_reconstruction_audit_smooth_sigma: float = 1.0,
    splat_reconstruction_audit_kernel_mode: str = "gaussian",
    splat_reconstruction_audit_gaussian_sigma_extent: float = 2.5,
    rng_seed: int = 42,
    force: bool = False,
    paper_3dgs_mode: bool = False,
    train_gaussian_field: bool = False,
    gaussian_num_gaussians: int = 20_000,
    gaussian_steps: int = 300,
    gaussian_image_size: int = 128,
    gaussian_views: str = "all",
    gaussian_device: str = "auto",
    gaussian_learning_rate: float = 1e-3,
    gaussian_volume_loss_weight: Optional[float] = None,
    gaussian_volume_sample_count: Optional[int] = None,
    train_synthetic_gaussian_field: bool = False,
    compute_gaussian_residual: bool = False,
    supervised_roi_checkpoint: Optional[Path] = None,
    supervised_roi_operating_point: Optional[Path] = None,
    supervised_roi_threshold: float = 0.5,
    supervised_roi_min_component_voxels: int = 0,
    supervised_roi_min_voxels: int = 1,
    supervised_roi_max_foreground_fraction: Optional[float] = 0.01,
    supervised_roi_device: str = "auto",
    supervised_roi_selection_mode: str = "threshold",
    supervised_roi_foreground_fraction: Optional[float] = None,
    converted_dicom_dir: Optional[Path] = None,
    dcm2niix_bin: str | Path = "dcm2niix",
) -> Path:
    """Process one uploaded scan and return the written index path."""
    resolved, auto_resolution = _resolve_auto_structural_upload(
        scan=scan,
        scan_dicom_dir=scan_dicom_dir,
        flair=flair,
        t2w=t2w,
        t1w=t1w,
        flair_dicom_dir=flair_dicom_dir,
        t2w_dicom_dir=t2w_dicom_dir,
        t1w_dicom_dir=t1w_dicom_dir,
        strict=True,
    )
    flair = resolved["flair"]
    t2w = resolved["t2w"]
    t1w = resolved["t1w"]
    flair_dicom_dir = resolved["flair_dicom_dir"]
    t2w_dicom_dir = resolved["t2w_dicom_dir"]
    t1w_dicom_dir = resolved["t1w_dicom_dir"]

    if flair is not None and flair_dicom_dir is not None:
        raise ValueError("Provide either --flair or --flair-dicom-dir for FLAIR, not both.")
    if t2w is not None and t2w_dicom_dir is not None:
        raise ValueError("Provide either --t2w or --t2w-dicom-dir for T2w, not both.")
    if t1w is not None and t1w_dicom_dir is not None:
        raise ValueError("Provide either --t1w or --t1w-dicom-dir for T1w, not both.")

    # --paper-3dgs-mode preset: flip on the 3DGS training flags and fill in
    # volume-loss/sample-count defaults only when the caller did not supply them.
    _PAPER_3DGS_VOLUME_LOSS_DEFAULT = 0.25
    _PAPER_3DGS_SAMPLE_COUNT_DEFAULT = 8192
    _DEFAULT_VOLUME_LOSS_WEIGHT = 0.0
    _DEFAULT_VOLUME_SAMPLE_COUNT = 4096

    if paper_3dgs_mode:
        train_gaussian_field = True
        train_synthetic_gaussian_field = True
        compute_gaussian_residual = True
        compute_control_gaussian_prior_residual = True
        if gaussian_volume_loss_weight is None:
            gaussian_volume_loss_weight = _PAPER_3DGS_VOLUME_LOSS_DEFAULT
        if gaussian_volume_sample_count is None:
            gaussian_volume_sample_count = _PAPER_3DGS_SAMPLE_COUNT_DEFAULT

    # Fill in non-preset sentinel defaults
    if gaussian_volume_loss_weight is None:
        gaussian_volume_loss_weight = _DEFAULT_VOLUME_LOSS_WEIGHT
    if gaussian_volume_sample_count is None:
        gaussian_volume_sample_count = _DEFAULT_VOLUME_SAMPLE_COUNT

    pipeline_preset: Optional[str] = "paper_3dgs" if paper_3dgs_mode else None

    upload_input_compatibility = build_upload_input_compatibility(
        scan=None,
        scan_dicom_dir=None,
        flair=flair,
        t2w=t2w,
        t1w=t1w,
        roi=roi,
        flair_dicom_dir=flair_dicom_dir,
        t2w_dicom_dir=t2w_dicom_dir,
        t1w_dicom_dir=t1w_dicom_dir,
        dcm2niix_bin=dcm2niix_bin,
    )
    upload_input_compatibility["auto_resolution"] = auto_resolution

    output_root = Path(output_root)
    safe_subject = _safe_subject_id(subject_id)
    conversion_root = (
        Path(converted_dicom_dir) if converted_dicom_dir is not None
        else output_root / "_converted_dicom"
    ) / safe_subject
    dicom_conversion: dict[str, object] = {}

    if flair_dicom_dir is not None:
        flair_path, meta = _convert_dicom_series(
            dicom_dir=flair_dicom_dir,
            output_dir=conversion_root / "flair",
            label="FLAIR",
            dcm2niix_bin=dcm2niix_bin,
        )
        dicom_conversion["FLAIR"] = meta
    else:
        flair_path = _validate_nifti(flair, "FLAIR")

    if t2w_dicom_dir is not None:
        t2w_path, meta = _convert_dicom_series(
            dicom_dir=t2w_dicom_dir,
            output_dir=conversion_root / "t2w",
            label="T2w",
            dcm2niix_bin=dcm2niix_bin,
        )
        dicom_conversion["T2w"] = meta
    else:
        t2w_path = _validate_nifti(t2w, "T2w")

    if t1w_dicom_dir is not None:
        t1w_path, meta = _convert_dicom_series(
            dicom_dir=t1w_dicom_dir,
            output_dir=conversion_root / "t1w",
            label="T1w",
            dcm2niix_bin=dcm2niix_bin,
        )
        dicom_conversion["T1w"] = meta
    else:
        t1w_path = _validate_nifti(t1w, "T1w")

    roi_path = _validate_nifti(roi, "ROI")
    if flair_path is None and t2w_path is None and t1w_path is None:
        raise ValueError(
            "At least one of --flair, --t2w, --t1w, --flair-dicom-dir, "
            "--t2w-dicom-dir, or --t1w-dicom-dir is required."
        )

    dicom_conversion_safety = _summarize_dicom_conversion_safety(dicom_conversion)
    preferred_modality = "FLAIR" if flair_path is not None else ("T2w" if t2w_path is not None else "T1w")
    controls: Optional[list[Path]]
    control_pool_meta: dict[str, object] = {}
    if synthetic_mode == "control-prior" or compute_control_gaussian_prior_residual:
        controls, control_pool_meta = _discover_control_volumes(
            control_dataset_dir=control_dataset_dir,
            explicit_control_volumes=control_volumes,
            preferred_modality=preferred_modality,
            excluded_subject_ids={subject_id, safe_subject},
            excluded_volume_paths=[
                path for path in (flair_path, t2w_path, t1w_path) if path is not None
            ],
        )
    else:
        controls = None

    rec = SubjectRecord(
        subject_id=safe_subject,
        group=group,
        metadata={
            "input_mode": "uploaded_mri",
            "original_subject_id": subject_id,
            "control_dataset_dir": str(control_dataset_dir) if control_dataset_dir else None,
            "dicom_conversion": dicom_conversion,
            "auto_structural_resolution": auto_resolution,
        },
        flair=flair_path,
        t2w=t2w_path,
        t1w=t1w_path,
        flair_roi=roi_path,
    )

    manifest = process_subject(
        rec=rec,
        output_root=output_root,
        synthetic_mode=synthetic_mode,
        roi_strategy=roi_strategy,
        candidate_roi_proposal_mode=candidate_roi_proposal_mode,
        max_splat_points=max_splat_points,
        rng_seed=rng_seed,
        force=force,
        splat_sampling_mode=splat_sampling_mode,
        presentation_splat_points=presentation_splat_points,
        presentation_splat_sampling_mode=presentation_splat_sampling_mode,
        paired_gaussian_support_basis=paired_gaussian_support_basis,
        compute_control_gaussian_prior_residual=compute_control_gaussian_prior_residual,
        control_gaussian_prior_max_points=control_gaussian_prior_max_points,
        control_gaussian_prior_min_controls=control_gaussian_prior_min_controls,
        control_gaussian_prior_heatmap_percentile=(
            control_gaussian_prior_heatmap_percentile
        ),
        control_gaussian_prior_smooth_sigma=control_gaussian_prior_smooth_sigma,
        control_gaussian_prior_heatmap_method=(
            control_gaussian_prior_heatmap_method
        ),
        control_gaussian_prior_heatmap_exact_max_points=(
            control_gaussian_prior_heatmap_exact_max_points
        ),
        control_gaussian_prior_std_floor_fraction=(
            control_gaussian_prior_std_floor_fraction
        ),
        compute_splat_reconstruction_audit=compute_splat_reconstruction_audit,
        splat_reconstruction_audit_max_points=splat_reconstruction_audit_max_points,
        splat_reconstruction_audit_smooth_sigma=splat_reconstruction_audit_smooth_sigma,
        splat_reconstruction_audit_kernel_mode=splat_reconstruction_audit_kernel_mode,
        splat_reconstruction_audit_gaussian_sigma_extent=(
            splat_reconstruction_audit_gaussian_sigma_extent
        ),
        train_gaussian_field=train_gaussian_field,
        gaussian_num_gaussians=gaussian_num_gaussians,
        gaussian_steps=gaussian_steps,
        gaussian_image_size=gaussian_image_size,
        gaussian_views=gaussian_views,
        gaussian_device=gaussian_device,
        gaussian_learning_rate=gaussian_learning_rate,
        gaussian_volume_loss_weight=gaussian_volume_loss_weight,
        gaussian_volume_sample_count=gaussian_volume_sample_count,
        train_synthetic_gaussian_field=train_synthetic_gaussian_field,
        compute_gaussian_residual=compute_gaussian_residual,
        supervised_roi_checkpoint=supervised_roi_checkpoint,
        supervised_roi_operating_point=supervised_roi_operating_point,
        supervised_roi_threshold=supervised_roi_threshold,
        supervised_roi_min_component_voxels=supervised_roi_min_component_voxels,
        supervised_roi_min_voxels=supervised_roi_min_voxels,
        supervised_roi_max_foreground_fraction=supervised_roi_max_foreground_fraction,
        supervised_roi_device=supervised_roi_device,
        supervised_roi_selection_mode=supervised_roi_selection_mode,
        supervised_roi_foreground_fraction=supervised_roi_foreground_fraction,
        control_prior_volumes=controls,
        pipeline_preset=pipeline_preset,
    )

    manifest["input_mode"] = "uploaded_mri"
    if dicom_conversion:
        manifest.setdefault("data_quality", {})["dicom_conversion_safety"] = dicom_conversion_safety
    _merge_upload_input_compatibility(manifest, upload_input_compatibility)
    supervised_resolved = manifest.get("supervised_roi_resolved_params", {})
    if not isinstance(supervised_resolved, dict):
        supervised_resolved = {}
    manifest["upload_meta"] = {
        "original_subject_id": subject_id,
        "safe_subject_id": safe_subject,
        "flair": str(flair_path) if flair_path else None,
        "t2w": str(t2w_path) if t2w_path else None,
        "t1w": str(t1w_path) if t1w_path else None,
        "roi": str(roi_path) if roi_path else None,
        "preferred_structural_modality": preferred_modality,
        "control_dataset_dir": str(control_dataset_dir) if control_dataset_dir else None,
        "explicit_control_volume_count": len(control_volumes or []),
        "resolved_control_volume_count": len(controls or []),
        "control_pool_meta": control_pool_meta,
        "synthetic_mode": synthetic_mode,
        "roi_strategy": roi_strategy,
        "candidate_roi_proposal_mode": candidate_roi_proposal_mode,
        "splat_reconstruction_audit_kernel_mode": splat_reconstruction_audit_kernel_mode,
        "splat_reconstruction_audit_gaussian_sigma_extent": (
            splat_reconstruction_audit_gaussian_sigma_extent
        ),
        "supervised_roi_checkpoint": str(supervised_roi_checkpoint) if supervised_roi_checkpoint else None,
        "supervised_roi_operating_point": (
            str(supervised_roi_operating_point) if supervised_roi_operating_point else None
        ),
        "supervised_roi_operating_point_meta": manifest.get(
            "supervised_roi_operating_point_meta",
            {},
        ),
        "supervised_roi_threshold": supervised_resolved.get(
            "threshold",
            supervised_roi_threshold,
        ),
        "supervised_roi_min_component_voxels": supervised_resolved.get(
            "min_component_voxels",
            supervised_roi_min_component_voxels,
        ),
        "supervised_roi_min_voxels": supervised_resolved.get(
            "min_voxels",
            supervised_roi_min_voxels,
        ),
        "supervised_roi_max_foreground_fraction": supervised_resolved.get(
            "max_foreground_fraction",
            supervised_roi_max_foreground_fraction,
        ),
        "supervised_roi_device": supervised_resolved.get("device", supervised_roi_device),
        "supervised_roi_selection_mode": supervised_resolved.get(
            "selection_mode",
            supervised_roi_selection_mode,
        ),
        "supervised_roi_foreground_fraction": supervised_resolved.get(
            "foreground_fraction",
            supervised_roi_foreground_fraction,
        ),
        "pipeline_preset": pipeline_preset,
        "paper_3dgs_mode": paper_3dgs_mode,
        "paired_gaussian_support_basis": paired_gaussian_support_basis,
        "compute_control_gaussian_prior_residual": compute_control_gaussian_prior_residual,
        "control_gaussian_prior_max_points": control_gaussian_prior_max_points,
        "control_gaussian_prior_min_controls": control_gaussian_prior_min_controls,
        "control_gaussian_prior_heatmap_percentile": (
            control_gaussian_prior_heatmap_percentile
        ),
        "control_gaussian_prior_smooth_sigma": control_gaussian_prior_smooth_sigma,
        "control_gaussian_prior_heatmap_method": (
            control_gaussian_prior_heatmap_method
        ),
        "control_gaussian_prior_heatmap_exact_max_points": (
            control_gaussian_prior_heatmap_exact_max_points
        ),
        "control_gaussian_prior_std_floor_fraction": (
            control_gaussian_prior_std_floor_fraction
        ),
        "gaussian_volume_loss_weight": gaussian_volume_loss_weight,
        "gaussian_volume_sample_count": gaussian_volume_sample_count,
        "dicom_conversion": dicom_conversion,
        "dicom_conversion_safety": dicom_conversion_safety,
        "auto_structural_resolution": auto_resolution,
        "upload_input_compatibility": upload_input_compatibility,
        "data_quality_gate": manifest.get("data_quality", {}).get("upload_readiness", {}),
    }

    subject_dir = output_root / safe_subject
    manifest_path = subject_dir / "manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)

    index = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "dataset": "Uploaded MRI",
        "input_mode": "uploaded_mri",
        "subjects": [
            {
                "subject": safe_subject,
                "group": group,
                "roi_available": roi_path is not None,
                "modalities": manifest.get("modalities", []),
                "stage_status": manifest.get("stage_status", {}),
                "synthetic_normal_source": manifest.get("synthetic_normal_source", "none"),
                "manifest": f"{safe_subject}/manifest.json",
            }
        ],
    }
    index_path = output_root / "index.json"
    with index_path.open("w") as fh:
        json.dump(index, fh, indent=2)
    print(f"Uploaded MRI index -> {index_path}")
    return index_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Process one uploaded/non-BIDS MRI scan through the FCD pipeline."
    )
    p.add_argument("--scan", type=Path, default=None,
                   help="Single structural NIfTI with inferable FLAIR/T2w/T1w filename.")
    p.add_argument("--scan-dicom-dir", type=Path, default=None,
                   help="Single structural DICOM series directory with inferable FLAIR/T2w/T1w folder name.")
    p.add_argument("--flair", type=Path, default=None, help="Standalone FLAIR NIfTI.")
    p.add_argument("--t2w", type=Path, default=None, help="Standalone T2w NIfTI.")
    p.add_argument("--t1w", type=Path, default=None, help="Standalone T1w NIfTI.")
    p.add_argument("--flair-dicom-dir", type=Path, default=None,
                   help="Directory containing one FLAIR DICOM series; converted with dcm2niix.")
    p.add_argument("--t2w-dicom-dir", type=Path, default=None,
                   help="Directory containing one T2w DICOM series; converted with dcm2niix.")
    p.add_argument("--t1w-dicom-dir", type=Path, default=None,
                   help="Directory containing one T1w DICOM series; converted with dcm2niix.")
    p.add_argument("--roi", type=Path, default=None, help="Optional lesion ROI NIfTI.")
    p.add_argument("--subject-id", default="uploaded_mri")
    p.add_argument("--group", default="uploaded")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/uploaded_mri"))
    p.add_argument("--converted-dicom-dir", type=Path, default=None,
                   help="Optional directory for dcm2niix NIfTI outputs; defaults under output-dir.")
    p.add_argument("--dcm2niix-bin", default="dcm2niix",
                   help="dcm2niix executable name or path for DICOM conversion.")
    p.add_argument("--preflight-only", action="store_true", default=False,
                   help="Print upload input compatibility JSON without running the pipeline.")
    p.add_argument("--preflight-output-json", type=Path, default=None,
                   help="Optional JSON path for --preflight-only output.")
    p.add_argument("--synthetic-mode",
                   choices=["placeholder", "nvidia", "control-prior", "contralateral"],
                   default="control-prior")
    p.add_argument("--roi-strategy",
                   choices=["provided", "auto", "provided-or-auto", "supervised", "provided-or-supervised", "none"],
                   default="provided-or-auto")
    p.add_argument("--candidate-roi-proposal-mode",
                   choices=PROPOSAL_MODES,
                   default="brightness-gradient",
                   help="Heuristic feature model for roi-strategy auto/provided-or-auto.")
    p.add_argument("--control-dataset-dir", type=Path, default=None,
                   help="Optional BIDS dataset whose healthy controls seed control-prior mode.")
    p.add_argument("--control-volume", type=Path, action="append", default=[],
                   help="Explicit healthy-control NIfTI for control-prior mode; repeatable.")
    p.add_argument("--max-splat-points", type=int, default=50_000)
    p.add_argument("--presentation-splat-points", type=int, default=300_000)
    p.add_argument("--splat-sampling-mode", choices=["surface", "volume", "uniform"], default="surface")
    p.add_argument("--presentation-splat-sampling-mode", choices=["surface", "volume", "uniform"],
                   default="volume",
                   help="Sampling mode for patient_splat_dense.ply; volume is the product-facing default.")
    p.add_argument("--paired-gaussian-support-basis",
                   choices=["union", "patient", "normal", "intersection"],
                   default="union",
                   help=(
                       "Signal used to sample the shared paired-Gaussian support. "
                       "union preserves max(patient, synthetic_normal); patient is "
                       "useful for patient-geometry ablations."
                   ))
    p.add_argument("--compute-control-gaussian-prior-residual",
                   action="store_true", default=False,
                   help=(
                       "Build uploaded-patient/support healthy-control prior mean "
                       "and z-residual Gaussian splats."
                   ))
    p.add_argument("--control-gaussian-prior-max-points", type=int, default=None,
                   help=(
                       "Maximum uploaded patient-support control-prior Gaussians; "
                       "defaults to the larger direct/presentation splat cap."
                   ))
    p.add_argument("--control-gaussian-prior-min-controls", type=int, default=2,
                   help="Minimum healthy controls required for control-prior residuals.")
    p.add_argument("--control-gaussian-prior-heatmap-percentile",
                   type=float, default=99.0,
                   help="Percentile for uploaded control-prior z-residual normalization.")
    p.add_argument("--control-gaussian-prior-smooth-sigma", type=float, default=1.0,
                   help="Voxel sigma for uploaded control-prior residual heatmap smoothing.")
    p.add_argument("--control-gaussian-prior-heatmap-method",
                   choices=["scatter", "gaussian"], default="gaussian",
                   help="How uploaded control-prior z-residual scores are projected to NIfTI.")
    p.add_argument("--control-gaussian-prior-heatmap-exact-max-points",
                   type=int, default=50_000,
                   help=(
                       "Maximum active uploaded control-prior residual Gaussians "
                       "for exact anisotropic heatmap accumulation."
                   ))
    p.add_argument("--control-gaussian-prior-std-floor-fraction",
                   type=float, default=0.05,
                   help=(
                       "Fraction of global control intensity std used as the "
                       "per-Gaussian z-score denominator floor."
                   ))
    p.add_argument("--skip-splat-reconstruction-audit", action="store_true", default=False,
                   help="Disable MRI-to-splat reconstruction quality audit.")
    p.add_argument("--splat-reconstruction-audit-max-points", type=int, default=100_000,
                   help="Maximum PLY vertices used by the reconstruction audit.")
    p.add_argument("--splat-reconstruction-audit-smooth-sigma", type=float, default=1.0,
                   help="Voxel sigma used when scattering PLY vertices back to NIfTI space.")
    p.add_argument("--splat-reconstruction-audit-kernel-mode",
                   choices=["nearest", "gaussian"], default="gaussian",
                   help="Audit PLYs by nearest-center scatter or stored Gaussian scale/rotation.")
    p.add_argument("--splat-reconstruction-audit-gaussian-sigma-extent", type=float, default=2.5,
                   help="Gaussian support radius, in sigmas, for Gaussian splat audit mode.")
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p.add_argument("--train-gaussian-field", action="store_true", default=False)
    p.add_argument("--gaussian-num-gaussians", type=int, default=20_000)
    p.add_argument("--gaussian-steps", type=int, default=300)
    p.add_argument("--gaussian-image-size", type=int, default=128)
    p.add_argument("--gaussian-views", type=str, default="all")
    p.add_argument("--gaussian-device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--gaussian-learning-rate", type=float, default=1e-3)
    p.add_argument("--gaussian-volume-loss-weight", type=float, default=None,
                   help="Optional sampled 3D voxel reconstruction loss weight. "
                        "Defaults to 0.0 normally, or 0.25 under --paper-3dgs-mode.")
    p.add_argument("--gaussian-volume-sample-count", type=int, default=None,
                   help="Number of deterministic voxel samples used by the volume loss. "
                        "Defaults to 4096 normally, or 8192 under --paper-3dgs-mode.")
    p.add_argument("--train-synthetic-gaussian-field", action="store_true", default=False)
    p.add_argument("--compute-gaussian-residual", action="store_true", default=False)
    p.add_argument("--paper-3dgs-mode", action="store_true", default=False,
                   help=(
                       "Enable the paper-facing uploaded-MRI 3DGS preset: uploaded "
                       "MRI -> optimized patient Gaussian field -> synthetic normal "
                       "projected onto patient Gaussian support -> optimized "
                       "Gaussian-field residual. Sets --train-gaussian-field, "
                       "--train-synthetic-gaussian-field, and "
                       "--compute-gaussian-residual. Volume loss defaults to 0.25 "
                       "and sample count to 8192 unless explicitly overridden."
                   ))
    p.add_argument("--supervised-roi-checkpoint", type=Path, default=None,
                   help="Path to supervised lesion proposer checkpoint for supervised ROI strategies.")
    p.add_argument("--supervised-roi-operating-point", type=Path, default=None,
                   help=(
                       "JSON operating point from fcd-lesion-proposer sweep, or a "
                       "direct JSON object with threshold/selection settings."
                   ))
    p.add_argument("--supervised-roi-threshold", type=float, default=0.5,
                   help="Probability threshold for supervised uploaded-MRI ROI prediction.")
    p.add_argument("--supervised-roi-min-component-voxels", type=int, default=0,
                   help="Minimum connected-component size retained in supervised uploaded-MRI ROI.")
    p.add_argument("--supervised-roi-min-voxels", type=int, default=1,
                   help="Reject supervised uploaded-MRI ROI predictions smaller than this voxel count.")
    p.add_argument("--supervised-roi-max-foreground-fraction", type=float, default=0.01,
                   help=(
                       "Reject supervised uploaded-MRI ROI predictions larger than this "
                       "brain fraction; set negative to disable."
                   ))
    p.add_argument("--supervised-roi-device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="Device for supervised uploaded-MRI ROI prediction.")
    p.add_argument("--supervised-roi-selection-mode",
                   choices=["threshold", "top-fraction", "threshold-and-top-fraction"],
                   default="threshold",
                   help="How supervised probability maps are converted to uploaded-MRI ROI masks.")
    p.add_argument("--supervised-roi-foreground-fraction", type=float, default=None,
                   help=(
                       "Requested uploaded-MRI ROI foreground fraction for top-fraction "
                       "supervised selection modes."
                   ))
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.preflight_only:
        report = build_upload_input_compatibility(
            scan=args.scan,
            scan_dicom_dir=args.scan_dicom_dir,
            flair=args.flair,
            t2w=args.t2w,
            t1w=args.t1w,
            roi=args.roi,
            flair_dicom_dir=args.flair_dicom_dir,
            t2w_dicom_dir=args.t2w_dicom_dir,
            t1w_dicom_dir=args.t1w_dicom_dir,
            dcm2niix_bin=args.dcm2niix_bin,
        )
        text = json.dumps(report, indent=2)
        if args.preflight_output_json is not None:
            args.preflight_output_json.parent.mkdir(parents=True, exist_ok=True)
            args.preflight_output_json.write_text(text, encoding="utf-8")
        print(text)
        return

    build_uploaded_mri(
        scan=args.scan,
        scan_dicom_dir=args.scan_dicom_dir,
        flair=args.flair,
        t2w=args.t2w,
        t1w=args.t1w,
        flair_dicom_dir=args.flair_dicom_dir,
        t2w_dicom_dir=args.t2w_dicom_dir,
        t1w_dicom_dir=args.t1w_dicom_dir,
        roi=args.roi,
        output_root=args.output_dir,
        subject_id=args.subject_id,
        group=args.group,
        synthetic_mode=args.synthetic_mode,
        roi_strategy=args.roi_strategy,
        candidate_roi_proposal_mode=args.candidate_roi_proposal_mode,
        control_dataset_dir=args.control_dataset_dir,
        control_volumes=args.control_volume,
        max_splat_points=args.max_splat_points,
        presentation_splat_points=args.presentation_splat_points,
        splat_sampling_mode=args.splat_sampling_mode,
        presentation_splat_sampling_mode=args.presentation_splat_sampling_mode,
        paired_gaussian_support_basis=args.paired_gaussian_support_basis,
        compute_control_gaussian_prior_residual=(
            args.compute_control_gaussian_prior_residual
        ),
        control_gaussian_prior_max_points=args.control_gaussian_prior_max_points,
        control_gaussian_prior_min_controls=args.control_gaussian_prior_min_controls,
        control_gaussian_prior_heatmap_percentile=(
            args.control_gaussian_prior_heatmap_percentile
        ),
        control_gaussian_prior_smooth_sigma=args.control_gaussian_prior_smooth_sigma,
        control_gaussian_prior_heatmap_method=(
            args.control_gaussian_prior_heatmap_method
        ),
        control_gaussian_prior_heatmap_exact_max_points=(
            args.control_gaussian_prior_heatmap_exact_max_points
        ),
        control_gaussian_prior_std_floor_fraction=(
            args.control_gaussian_prior_std_floor_fraction
        ),
        compute_splat_reconstruction_audit=not args.skip_splat_reconstruction_audit,
        splat_reconstruction_audit_max_points=args.splat_reconstruction_audit_max_points,
        splat_reconstruction_audit_smooth_sigma=args.splat_reconstruction_audit_smooth_sigma,
        splat_reconstruction_audit_kernel_mode=args.splat_reconstruction_audit_kernel_mode,
        splat_reconstruction_audit_gaussian_sigma_extent=(
            args.splat_reconstruction_audit_gaussian_sigma_extent
        ),
        rng_seed=args.rng_seed,
        force=args.force,
        paper_3dgs_mode=args.paper_3dgs_mode,
        train_gaussian_field=args.train_gaussian_field,
        gaussian_num_gaussians=args.gaussian_num_gaussians,
        gaussian_steps=args.gaussian_steps,
        gaussian_image_size=args.gaussian_image_size,
        gaussian_views=args.gaussian_views,
        gaussian_device=args.gaussian_device,
        gaussian_learning_rate=args.gaussian_learning_rate,
        gaussian_volume_loss_weight=args.gaussian_volume_loss_weight,
        gaussian_volume_sample_count=args.gaussian_volume_sample_count,
        train_synthetic_gaussian_field=args.train_synthetic_gaussian_field,
        compute_gaussian_residual=args.compute_gaussian_residual,
        supervised_roi_checkpoint=args.supervised_roi_checkpoint,
        supervised_roi_operating_point=args.supervised_roi_operating_point,
        supervised_roi_threshold=args.supervised_roi_threshold,
        supervised_roi_min_component_voxels=args.supervised_roi_min_component_voxels,
        supervised_roi_min_voxels=args.supervised_roi_min_voxels,
        supervised_roi_max_foreground_fraction=(
            None if args.supervised_roi_max_foreground_fraction < 0
            else args.supervised_roi_max_foreground_fraction
        ),
        supervised_roi_device=args.supervised_roi_device,
        supervised_roi_selection_mode=args.supervised_roi_selection_mode,
        supervised_roi_foreground_fraction=args.supervised_roi_foreground_fraction,
        converted_dicom_dir=args.converted_dicom_dir,
        dcm2niix_bin=args.dcm2niix_bin,
    )


if __name__ == "__main__":
    main()
