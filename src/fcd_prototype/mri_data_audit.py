"""MRI/NIfTI geometry and data-quality audit helpers.

The Gaussian and residual pipeline is only meaningful when all volumes share a
known physical frame.  These helpers produce serializable manifest fields that
make orientation, spacing, finite-value, and ROI-alignment assumptions explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from fcd_prototype.bids_discovery import SubjectRecord, discover_dataset


AFFINE_ATOL = 1e-3
SPACING_ATOL = 1e-3
ORTHOGONAL_ATOL = 1e-3
THICK_SLICE_MM = 3.0
ANISOTROPY_RATIO = 2.0
TINY_ROI_VOXELS = 5
HUGE_ROI_FRACTION = 0.10


_PRIMARY_BLOCKER_WARNINGS = {
    "nonfinite_voxels_present",
    "low_dynamic_range",
    "affine_spatial_matrix_rank_lt_3",
}


def _to_list(values) -> list:
    return [float(v) if isinstance(v, (np.floating, float)) else int(v) for v in values]


def _shape_list(shape: tuple[int, ...]) -> list[int]:
    return [int(v) for v in shape]


def _voxel_spacing(affine: np.ndarray) -> list[float]:
    return [float(v) for v in np.sqrt((affine[:3, :3] ** 2).sum(axis=0))]


def _orientation(affine: np.ndarray) -> list[str]:
    return [str(v) for v in nib.aff2axcodes(affine)]


def _affine_orthogonality_error(affine: np.ndarray) -> float:
    basis = np.asarray(affine, dtype=np.float64)[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    if np.any(spacing <= 1e-9):
        return float("inf")
    unit = basis / spacing
    gram = unit.T @ unit
    off_diag = gram - np.eye(3)
    return float(np.max(np.abs(off_diag)))


def _finite_stats(data: np.ndarray) -> dict:
    finite = np.isfinite(data)
    finite_count = int(finite.sum())
    total = int(data.size)
    if finite_count:
        finite_data = data[finite]
        min_val = float(finite_data.min())
        max_val = float(finite_data.max())
        mean_val = float(finite_data.mean())
    else:
        min_val = max_val = mean_val = None
    return {
        "finite_count": finite_count,
        "finite_fraction": float(finite_count / total) if total else 0.0,
        "nan_count": int(np.isnan(data).sum()),
        "inf_count": int(np.isinf(data).sum()),
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "low_dynamic_range": bool(finite_count > 0 and min_val == max_val),
    }


def inspect_nifti(path: Path) -> dict:
    """Return serializable geometry/data-quality metadata for one NIfTI file."""
    path = Path(path)
    img = nib.load(str(path))
    canonical = nib.as_closest_canonical(img)
    data = np.asanyarray(img.dataobj)
    canonical_data = canonical.get_fdata(dtype=np.float32)
    header = img.header
    spacing = _voxel_spacing(img.affine)
    canonical_spacing = _voxel_spacing(canonical.affine)
    qform_code = int(header["qform_code"])
    sform_code = int(header["sform_code"])
    spatial_units, temporal_units = header.get_xyzt_units()
    qform = header.get_qform()
    sform = header.get_sform()
    qform_sform_max_abs_diff = (
        float(np.abs(qform - sform).max())
        if qform_code > 0 and sform_code > 0 else None
    )
    orthogonality_error = _affine_orthogonality_error(img.affine)

    warnings: list[str] = []
    if data.ndim == 4:
        warnings.append("4d_volume_first_frame_used_downstream")
    elif data.ndim != 3:
        warnings.append(f"unexpected_ndim_{data.ndim}")
    if max(canonical_spacing) > THICK_SLICE_MM:
        warnings.append("thick_slice_spacing_gt_3mm")
    if min(canonical_spacing) > 0 and max(canonical_spacing) / min(canonical_spacing) > ANISOTROPY_RATIO:
        warnings.append("anisotropic_spacing_ratio_gt_2")
    if _orientation(img.affine) != _orientation(canonical.affine):
        warnings.append("orientation_reordered_to_canonical")
    if spatial_units not in ("mm", "unknown"):
        warnings.append(f"spatial_units_not_mm:{spatial_units}")
    if qform_code == 0:
        warnings.append("qform_code_zero")
    if sform_code == 0:
        warnings.append("sform_code_zero")
    if qform_code == 0 and sform_code == 0:
        warnings.append("no_qform_or_sform_code")
    if qform_sform_max_abs_diff is not None and qform_sform_max_abs_diff > AFFINE_ATOL:
        warnings.append("qform_sform_affine_mismatch")
    if orthogonality_error > ORTHOGONAL_ATOL:
        warnings.append("affine_axes_not_orthogonal_or_sheared")

    stats = _finite_stats(data.astype(np.float64, copy=False))
    if stats["nan_count"] or stats["inf_count"]:
        warnings.append("nonfinite_voxels_present")
    if stats["low_dynamic_range"]:
        warnings.append("low_dynamic_range")

    affine_rank = int(np.linalg.matrix_rank(img.affine[:3, :3]))
    if affine_rank < 3:
        warnings.append("affine_spatial_matrix_rank_lt_3")

    return {
        "path": str(path),
        "exists": path.is_file(),
        "dtype": str(header.get_data_dtype()),
        "shape": _shape_list(tuple(img.shape)),
        "canonical_shape": _shape_list(tuple(canonical_data.shape)),
        "orientation": _orientation(img.affine),
        "canonical_orientation": _orientation(canonical.affine),
        "voxel_spacing_mm": spacing,
        "canonical_voxel_spacing_mm": canonical_spacing,
        "qform_code": qform_code,
        "sform_code": sform_code,
        "xyzt_units": {
            "spatial": spatial_units,
            "temporal": temporal_units,
        },
        "qform_sform_max_abs_diff": qform_sform_max_abs_diff,
        "affine_orthogonality_error": orthogonality_error,
        "affine_det": float(np.linalg.det(img.affine[:3, :3])),
        "affine_rank": affine_rank,
        "is_4d": bool(data.ndim == 4),
        "finite": stats,
        "warnings": warnings,
    }


def _load_canonical_geometry(path: Path) -> tuple[tuple[int, ...], np.ndarray, list[float], list[str]]:
    img = nib.as_closest_canonical(nib.load(str(path)))
    data_shape = tuple(int(v) for v in img.shape[:3])
    affine = img.affine.astype(np.float64)
    return data_shape, affine, _voxel_spacing(affine), _orientation(affine)


def compare_canonical_geometry(
    reference_path: Path,
    moving_path: Path,
    reference_label: str,
    moving_label: str,
) -> dict:
    """Compare canonicalized geometry for two NIfTI files."""
    ref_shape, ref_affine, ref_spacing, ref_orientation = _load_canonical_geometry(reference_path)
    mov_shape, mov_affine, mov_spacing, mov_orientation = _load_canonical_geometry(moving_path)
    affine_abs = np.abs(ref_affine - mov_affine)
    spacing_abs = np.abs(np.asarray(ref_spacing) - np.asarray(mov_spacing))
    shape_match = ref_shape == mov_shape
    affine_max_abs_diff = float(affine_abs.max())
    spacing_max_abs_diff = float(spacing_abs.max())
    orientation_match = ref_orientation == mov_orientation
    needs_resample = (
        not shape_match
        or affine_max_abs_diff > AFFINE_ATOL
        or spacing_max_abs_diff > SPACING_ATOL
        or not orientation_match
    )
    return {
        "reference": reference_label,
        "moving": moving_label,
        "reference_shape": _shape_list(ref_shape),
        "moving_shape": _shape_list(mov_shape),
        "shape_match": shape_match,
        "reference_spacing_mm": ref_spacing,
        "moving_spacing_mm": mov_spacing,
        "spacing_max_abs_diff": spacing_max_abs_diff,
        "orientation_match": orientation_match,
        "affine_max_abs_diff": affine_max_abs_diff,
        "needs_resample_to_reference": bool(needs_resample),
    }


def roi_stats(roi_path: Path) -> dict:
    """Return nonzero/fraction warnings for a canonical ROI mask."""
    img = nib.as_closest_canonical(nib.load(str(roi_path)))
    data = img.get_fdata(dtype=np.float32)
    mask = data > 0
    nonzero = int(mask.sum())
    total = int(mask.size)
    fraction = float(nonzero / total) if total else 0.0
    warnings: list[str] = []
    if nonzero == 0:
        warnings.append("roi_empty")
    elif nonzero < TINY_ROI_VOXELS:
        warnings.append("roi_tiny_lt_5_voxels")
    if fraction > HUGE_ROI_FRACTION:
        warnings.append("roi_fraction_gt_10_percent")
    return {
        "nonzero_voxels": nonzero,
        "voxel_fraction": fraction,
        "unique_values": _to_list(np.unique(data)[:16]),
        "warnings": warnings,
    }


def build_subject_data_quality(
    *,
    flair_path: Optional[Path],
    t2w_path: Optional[Path] = None,
    t1w_path: Optional[Path] = None,
    roi_path: Optional[Path] = None,
) -> dict:
    """Build a subject-level audit for raw input volumes."""
    volumes: dict[str, dict] = {}
    comparisons: dict[str, dict] = {}
    warnings: list[str] = []

    for label, path in (
        ("flair", flair_path),
        ("t2w", t2w_path),
        ("t1w", t1w_path),
        ("flair_roi", roi_path),
    ):
        if path is None:
            continue
        if not Path(path).is_file():
            volumes[label] = {"path": str(path), "exists": False}
            warnings.append(f"{label}:missing_file")
            continue
        try:
            volumes[label] = inspect_nifti(Path(path))
            warnings.extend(f"{label}:{w}" for w in volumes[label].get("warnings", []))
        except Exception as exc:
            volumes[label] = {"path": str(path), "exists": Path(path).is_file(), "error": str(exc)}
            warnings.append(f"{label}:inspection_failed")

    if flair_path is not None and Path(flair_path).is_file():
        if t2w_path is not None and Path(t2w_path).is_file():
            try:
                comparisons["t2w_to_flair"] = compare_canonical_geometry(
                    Path(flair_path), Path(t2w_path), "flair", "t2w"
                )
                if comparisons["t2w_to_flair"]["needs_resample_to_reference"]:
                    warnings.append("t2w_to_flair:geometry_mismatch_requires_resampling")
            except Exception as exc:
                comparisons["t2w_to_flair"] = {"error": str(exc)}
                warnings.append("t2w_to_flair:comparison_failed")
        if t1w_path is not None and Path(t1w_path).is_file():
            try:
                comparisons["t1w_to_flair"] = compare_canonical_geometry(
                    Path(flair_path), Path(t1w_path), "flair", "t1w"
                )
                if comparisons["t1w_to_flair"]["needs_resample_to_reference"]:
                    warnings.append("t1w_to_flair:geometry_mismatch_requires_resampling")
            except Exception as exc:
                comparisons["t1w_to_flair"] = {"error": str(exc)}
                warnings.append("t1w_to_flair:comparison_failed")
        if roi_path is not None and Path(roi_path).is_file():
            try:
                comparisons["flair_roi_to_flair"] = compare_canonical_geometry(
                    Path(flair_path), Path(roi_path), "flair", "flair_roi"
                )
                if comparisons["flair_roi_to_flair"]["needs_resample_to_reference"]:
                    warnings.append("flair_roi_to_flair:geometry_mismatch_requires_resampling")
                comparisons["flair_roi_stats"] = roi_stats(Path(roi_path))
                warnings.extend(f"flair_roi:{w}" for w in comparisons["flair_roi_stats"]["warnings"])
            except Exception as exc:
                comparisons["flair_roi_to_flair"] = {"error": str(exc)}
                warnings.append("flair_roi_to_flair:comparison_failed")
    elif t2w_path is not None and Path(t2w_path).is_file():
        if t1w_path is not None and Path(t1w_path).is_file():
            try:
                comparisons["t1w_to_t2w"] = compare_canonical_geometry(
                    Path(t2w_path), Path(t1w_path), "t2w", "t1w"
                )
                if comparisons["t1w_to_t2w"]["needs_resample_to_reference"]:
                    warnings.append("t1w_to_t2w:geometry_mismatch_requires_resampling")
            except Exception as exc:
                comparisons["t1w_to_t2w"] = {"error": str(exc)}
                warnings.append("t1w_to_t2w:comparison_failed")
        if roi_path is not None and Path(roi_path).is_file():
            try:
                comparisons["flair_roi_to_t2w"] = compare_canonical_geometry(
                    Path(t2w_path), Path(roi_path), "t2w", "flair_roi"
                )
                if comparisons["flair_roi_to_t2w"]["needs_resample_to_reference"]:
                    warnings.append("flair_roi_to_t2w:geometry_mismatch_requires_resampling")
                comparisons["flair_roi_stats"] = roi_stats(Path(roi_path))
                warnings.extend(f"flair_roi:{w}" for w in comparisons["flair_roi_stats"]["warnings"])
            except Exception as exc:
                comparisons["flair_roi_to_t2w"] = {"error": str(exc)}
                warnings.append("flair_roi_to_t2w:comparison_failed")
    elif t1w_path is not None and Path(t1w_path).is_file():
        if roi_path is not None and Path(roi_path).is_file():
            try:
                comparisons["flair_roi_to_t1w"] = compare_canonical_geometry(
                    Path(t1w_path), Path(roi_path), "t1w", "flair_roi"
                )
                if comparisons["flair_roi_to_t1w"]["needs_resample_to_reference"]:
                    warnings.append("flair_roi_to_t1w:geometry_mismatch_requires_resampling")
                comparisons["flair_roi_stats"] = roi_stats(Path(roi_path))
                warnings.extend(f"flair_roi:{w}" for w in comparisons["flair_roi_stats"]["warnings"])
            except Exception as exc:
                comparisons["flair_roi_to_t1w"] = {"error": str(exc)}
                warnings.append("flair_roi_to_t1w:comparison_failed")

    audit = {
        "volumes": volumes,
        "comparisons": comparisons,
        "warnings": warnings,
        "warning_count": len(warnings),
        "status": "warning" if warnings else "ok",
    }
    audit["upload_readiness"] = build_upload_readiness(audit)
    return audit


def build_upload_readiness(audit: dict) -> dict[str, object]:
    """Classify whether a subject/upload is safe enough for 3DGS subtraction.

    This is intentionally stricter than the raw warning list. Some warnings are
    survivable with affine-aware resampling and should trigger review; others
    mean the MRI cannot support a credible 3D Gaussian or residual claim.
    """
    volumes = audit.get("volumes", {})
    comparisons = audit.get("comparisons", {})
    primary_labels = [
        label
        for label in ("flair", "t2w", "t1w")
        if label in volumes and volumes.get(label)
    ]
    blockers: list[str] = []
    review_items: list[str] = []

    if not primary_labels:
        blockers.append("no_primary_mri_volume")

    for label in primary_labels:
        volume = volumes.get(label, {})
        if not volume.get("exists", True):
            blockers.append(f"{label}:missing_file")
            continue
        if "error" in volume:
            blockers.append(f"{label}:inspection_failed")
            continue
        for warning in volume.get("warnings", []):
            warning = str(warning)
            if warning.startswith("unexpected_ndim_") or warning in _PRIMARY_BLOCKER_WARNINGS:
                blockers.append(f"{label}:{warning}")
            else:
                review_items.append(f"{label}:{warning}")

    roi_volume = volumes.get("flair_roi")
    if roi_volume:
        if not roi_volume.get("exists", True):
            review_items.append("flair_roi:missing_file")
        elif "error" in roi_volume:
            review_items.append("flair_roi:inspection_failed")
        else:
            for warning in roi_volume.get("warnings", []):
                review_items.append(f"flair_roi:{warning}")

    roi_stats_info = comparisons.get("flair_roi_stats", {})
    roi_warnings = [str(w) for w in roi_stats_info.get("warnings", [])]
    if "roi_empty" in roi_warnings:
        blockers.append("flair_roi:roi_empty")
    for warning in roi_warnings:
        if warning != "roi_empty":
            review_items.append(f"flair_roi:{warning}")

    for comparison_key in (
        "t2w_to_flair",
        "t1w_to_flair",
        "t1w_to_t2w",
        "flair_roi_to_flair",
        "flair_roi_to_t2w",
        "flair_roi_to_t1w",
    ):
        comparison = comparisons.get(comparison_key, {})
        if comparison.get("error"):
            review_items.append(f"{comparison_key}:comparison_failed")
        elif comparison.get("needs_resample_to_reference"):
            review_items.append(f"{comparison_key}:geometry_mismatch_requires_resampling")

    blockers = sorted(dict.fromkeys(blockers))
    review_items = sorted(dict.fromkeys(review_items))
    if blockers:
        status = "fail"
    elif review_items:
        status = "review"
    else:
        status = "pass"

    return {
        "status": status,
        "blocking_issue_count": len(blockers),
        "review_issue_count": len(review_items),
        "blockers": blockers,
        "review_items": review_items,
        "primary_modalities": primary_labels,
        "can_run_3dgs": status != "fail",
        "can_subtract_patient_normal": status != "fail",
        "can_use_provided_roi_for_fuzzing": "flair_roi:roi_empty" not in blockers,
        "interpretation": (
            "processable"
            if status == "pass"
            else (
                "processable_with_geometry_review"
                if status == "review"
                else "not_safe_for_paper_grade_3dgs_subtraction"
            )
        ),
        "paper_risk_summary": [
            "Nonfinite, constant, rank-deficient, missing, or unsupported-dimensional primary MRI blocks paper-grade 3DGS.",
            "Geometry mismatch, thick slices, anisotropy, qform/sform issues, and ROI resampling require explicit review.",
            "An empty provided ROI cannot support lesion fuzzing or ROI-overlap validation.",
        ],
    }


def _inc(counter: dict[str, int], key: str) -> None:
    counter[key] = int(counter.get(key, 0)) + 1


def _normalize_group(group: str | None) -> str:
    raw = (group or "").strip().lower()
    if raw in {"fcd", "patient", "lesion"}:
        return "fcd"
    if raw in {"hc", "control", "healthy", "healthy-control"}:
        return "hc"
    return raw or "unknown"


def _comparison_needs_resample(audit: dict, key: str) -> bool:
    comparison = audit.get("comparisons", {}).get(key, {})
    return bool(comparison.get("needs_resample_to_reference"))


def _subject_audit_row(rec: SubjectRecord, audit: dict) -> dict[str, object]:
    volumes = audit.get("volumes", {})
    comparisons = audit.get("comparisons", {})
    roi_stats_info = comparisons.get("flair_roi_stats", {})
    flair = volumes.get("flair", {})
    t2w = volumes.get("t2w", {})
    t1w = volumes.get("t1w", {})
    roi = volumes.get("flair_roi", {})

    def _spacing(vol: dict) -> str:
        values = vol.get("canonical_voxel_spacing_mm")
        if not values:
            return ""
        return ",".join(f"{float(v):.6g}" for v in values)

    def _orientation_str(vol: dict) -> str:
        values = vol.get("canonical_orientation")
        return "".join(values) if values else ""

    return {
        "subject": rec.subject_id,
        "group": _normalize_group(rec.group),
        "has_flair": bool(rec.flair and Path(rec.flair).is_file()),
        "has_t2w": bool(rec.t2w and Path(rec.t2w).is_file()),
        "has_t1w": bool(rec.t1w and Path(rec.t1w).is_file()),
        "has_roi": bool(rec.flair_roi and Path(rec.flair_roi).is_file()),
        "status": audit.get("status", "unknown"),
        "warning_count": int(audit.get("warning_count", 0)),
        "warnings": ";".join(audit.get("warnings", [])),
        "flair_shape": "x".join(str(v) for v in flair.get("canonical_shape", [])[:3]),
        "flair_spacing_mm": _spacing(flair),
        "flair_orientation": _orientation_str(flair),
        "t2w_shape": "x".join(str(v) for v in t2w.get("canonical_shape", [])[:3]),
        "t2w_spacing_mm": _spacing(t2w),
        "t2w_orientation": _orientation_str(t2w),
        "t1w_shape": "x".join(str(v) for v in t1w.get("canonical_shape", [])[:3]),
        "t1w_spacing_mm": _spacing(t1w),
        "t1w_orientation": _orientation_str(t1w),
        "roi_shape": "x".join(str(v) for v in roi.get("canonical_shape", [])[:3]),
        "roi_spacing_mm": _spacing(roi),
        "roi_orientation": _orientation_str(roi),
        "t2w_to_flair_needs_resample": _comparison_needs_resample(audit, "t2w_to_flair"),
        "t1w_to_flair_needs_resample": _comparison_needs_resample(audit, "t1w_to_flair"),
        "t1w_to_t2w_needs_resample": _comparison_needs_resample(audit, "t1w_to_t2w"),
        "roi_to_flair_needs_resample": _comparison_needs_resample(audit, "flair_roi_to_flair"),
        "roi_to_t2w_needs_resample": _comparison_needs_resample(audit, "flair_roi_to_t2w"),
        "roi_to_t1w_needs_resample": _comparison_needs_resample(audit, "flair_roi_to_t1w"),
        "roi_nonzero_voxels": roi_stats_info.get("nonzero_voxels"),
        "roi_voxel_fraction": roi_stats_info.get("voxel_fraction"),
    }


def audit_dataset(
    dataset_dir: Path | str,
    *,
    subjects: Optional[list[str]] = None,
    output_json: Optional[Path | str] = None,
    output_csv: Optional[Path | str] = None,
) -> dict[str, object]:
    """Audit all discoverable BIDS/ds004199 subjects and summarize conversion risks."""
    dataset_dir = Path(dataset_dir)
    wanted = {s.strip() for s in subjects or [] if s.strip()}
    rows: list[dict[str, object]] = []
    subject_audits: dict[str, dict] = {}
    warning_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    orientation_counts: dict[str, dict[str, int]] = {
        "flair": {},
        "t2w": {},
        "t1w": {},
        "flair_roi": {},
    }
    modality_counts = {"flair": 0, "t2w": 0, "t1w": 0, "flair_roi": 0}
    geometry_mismatch_counts = {
        "t2w_to_flair": 0,
        "t1w_to_flair": 0,
        "t1w_to_t2w": 0,
        "flair_roi_to_flair": 0,
        "flair_roi_to_t2w": 0,
        "flair_roi_to_t1w": 0,
    }

    discovered = list(discover_dataset(dataset_dir))
    for rec in discovered:
        if wanted and rec.subject_id not in wanted:
            continue
        group = _normalize_group(rec.group)
        _inc(group_counts, group)
        audit = build_subject_data_quality(
            flair_path=rec.flair,
            t2w_path=rec.t2w,
            t1w_path=rec.t1w,
            roi_path=rec.flair_roi,
        )
        subject_audits[rec.subject_id] = audit
        row = _subject_audit_row(rec, audit)
        rows.append(row)

        for label, path in (
            ("flair", rec.flair),
            ("t2w", rec.t2w),
            ("t1w", rec.t1w),
            ("flair_roi", rec.flair_roi),
        ):
            if path is not None and Path(path).is_file():
                modality_counts[label] += 1
            orientation = audit.get("volumes", {}).get(label, {}).get("canonical_orientation")
            if orientation:
                _inc(orientation_counts[label], "".join(orientation))

        for warning in audit.get("warnings", []):
            _inc(warning_counts, str(warning))

        for key in geometry_mismatch_counts:
            if _comparison_needs_resample(audit, key):
                geometry_mismatch_counts[key] += 1

    missing_requested = sorted(wanted - {r["subject"] for r in rows})
    if missing_requested:
        for subject in missing_requested:
            _inc(warning_counts, "requested_subject_not_found")
            rows.append({
                "subject": subject,
                "group": "unknown",
                "has_flair": False,
                "has_t2w": False,
                "has_t1w": False,
                "has_roi": False,
                "status": "missing",
                "warning_count": 1,
                "warnings": "requested_subject_not_found",
                "flair_shape": "",
                "flair_spacing_mm": "",
                "flair_orientation": "",
                "t2w_shape": "",
                "t2w_spacing_mm": "",
                "t2w_orientation": "",
                "t1w_shape": "",
                "t1w_spacing_mm": "",
                "t1w_orientation": "",
                "roi_shape": "",
                "roi_spacing_mm": "",
                "roi_orientation": "",
                "t2w_to_flair_needs_resample": False,
                "t1w_to_flair_needs_resample": False,
                "t1w_to_t2w_needs_resample": False,
                "roi_to_flair_needs_resample": False,
                "roi_to_t2w_needs_resample": False,
                "roi_to_t1w_needs_resample": False,
                "roi_nonzero_voxels": None,
                "roi_voxel_fraction": None,
            })

    top_warnings = [
        {"warning": warning, "count": count}
        for warning, count in sorted(
            warning_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    n_total = len(rows)
    n_warning = sum(1 for row in rows if row.get("status") != "ok")
    summary = {
        "dataset_dir": str(dataset_dir),
        "n_total": n_total,
        "n_warning": n_warning,
        "n_ok": n_total - n_warning,
        "group_counts": group_counts,
        "modality_counts": modality_counts,
        "geometry_mismatch_counts": geometry_mismatch_counts,
        "orientation_counts": orientation_counts,
        "warning_counts": warning_counts,
        "top_warnings": top_warnings[:25],
        "requested_subjects_missing": missing_requested,
        "conversion_risks": [
            "All downstream 3DGS, residual, and candidate masks assume canonical RAS world-space geometry.",
            "FLAIR/T2w/T1w or ROI geometry mismatches require affine-aware resampling before subtraction.",
            "Thick-slice and anisotropic scans reduce lesion localization resolution.",
            "Missing qform/sform, non-mm spatial units, or qform/sform affine disagreement must be inspected before upload processing.",
        ],
    }
    result = {
        "summary": summary,
        "subjects": rows,
        "subject_audits": subject_audits,
    }

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2))

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "subject", "group", "has_flair", "has_t2w", "has_t1w", "has_roi",
            "status", "warning_count", "warnings",
            "flair_shape", "flair_spacing_mm", "flair_orientation",
            "t2w_shape", "t2w_spacing_mm", "t2w_orientation",
            "t1w_shape", "t1w_spacing_mm", "t1w_orientation",
            "roi_shape", "roi_spacing_mm", "roi_orientation",
            "t2w_to_flair_needs_resample", "t1w_to_flair_needs_resample",
            "t1w_to_t2w_needs_resample",
            "roi_to_flair_needs_resample", "roi_to_t2w_needs_resample",
            "roi_to_t1w_needs_resample",
            "roi_nonzero_voxels", "roi_voxel_fraction",
        ]
        with output_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return result


def build_viewer_metadata(
    *,
    subject_record,
    base_nifti_path: Optional[Path],
    base_modality: str,
    roi_nifti_path: Optional[Path] = None,
    participant_metadata: Optional[dict] = None,
    sidecar_fields: Optional[dict] = None,
) -> dict:
    """Assemble a compact, display-ready metadata summary for the browser viewer.

    Returns a dict with sections: subject, scan, geometry, roi (when present).
    All values are plain Python scalars (str, int, float, bool, None) so the
    result is safe to embed directly in manifest.json.
    """
    from fcd_prototype.bids_discovery import load_sidecar  # avoid circular at module level

    subject_id = getattr(subject_record, "subject_id", str(subject_record))
    group = getattr(subject_record, "group", "") or ""
    meta = participant_metadata or getattr(subject_record, "metadata", {}) or {}

    participant_section: dict = {"subject_id": subject_id, "group": group}
    for field in ("sex", "age_scan", "age_epilepsyonset", "mri_diagnosis",
                  "hemisphere", "lobe", "split", "histopathology", "outcome"):
        val = meta.get(field)
        if val not in (None, "", "n/a", "N/A"):
            participant_section[field] = val

    acq: dict = {}
    if sidecar_fields:
        acq.update(sidecar_fields)
    elif base_nifti_path is not None and Path(base_nifti_path).is_file():
        acq.update(load_sidecar(Path(base_nifti_path)))
    acq["modality"] = base_modality

    geometry: dict = {}
    if base_nifti_path is not None and Path(base_nifti_path).is_file():
        try:
            img = nib.load(str(base_nifti_path))
            canonical = nib.as_closest_canonical(img)
            spacing = _voxel_spacing(canonical.affine)
            shape = [int(v) for v in canonical.shape[:3]]
            geometry = {
                "shape": shape,
                "voxel_spacing_mm": [round(v, 4) for v in spacing],
                "orientation": _orientation(canonical.affine),
                "dtype": str(img.header.get_data_dtype()),
            }
            data = canonical.get_fdata(dtype=np.float32)
            finite = data[np.isfinite(data)]
            if finite.size:
                p1, p99 = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
                geometry["intensity_p1"] = round(p1, 4)
                geometry["intensity_p99"] = round(p99, 4)
                geometry["intensity_mean"] = round(float(finite.mean()), 4)
        except Exception:
            pass

    roi_section: Optional[dict] = None
    if roi_nifti_path is not None and Path(roi_nifti_path).is_file():
        try:
            roi_img = nib.as_closest_canonical(nib.load(str(roi_nifti_path)))
            roi_data = roi_img.get_fdata(dtype=np.float32)
            roi_mask = roi_data > 0
            voxel_count = int(roi_mask.sum())
            spacing = _voxel_spacing(roi_img.affine)
            voxel_vol_mm3 = float(np.prod(spacing))
            volume_mm3 = round(voxel_count * voxel_vol_mm3, 2)
            ijk = np.argwhere(roi_mask)
            center = [round(float(v), 1) for v in ijk.mean(axis=0)] if voxel_count else None
            bounds_min = [int(v) for v in ijk.min(axis=0)] if voxel_count else None
            bounds_max = [int(v) for v in ijk.max(axis=0)] if voxel_count else None
            roi_section = {
                "voxel_count": voxel_count,
                "volume_mm3": volume_mm3,
                "center_ijk": center,
                "bounds_min_ijk": bounds_min,
                "bounds_max_ijk": bounds_max,
            }
        except Exception:
            roi_section = {"error": "could not compute ROI stats"}

    result: dict = {
        "participant": participant_section,
        "scan": acq,
        "geometry": geometry,
    }
    if roi_section is not None:
        result["roi"] = roi_section
    return result


def record_roi_processing(
    audit: dict,
    *,
    source: str,
    source_shape: tuple[int, ...],
    source_affine: np.ndarray,
    target_shape: tuple[int, ...] | None,
    target_affine: np.ndarray | None,
    resampled: bool,
) -> None:
    """Attach actual ROI processing outcome to an existing subject audit."""
    info = {
        "source": source,
        "source_shape": _shape_list(tuple(source_shape[:3])),
        "target_shape": _shape_list(tuple(target_shape[:3])) if target_shape is not None else None,
        "resampled_to_base": bool(resampled),
        "source_spacing_mm": _voxel_spacing(source_affine),
        "target_spacing_mm": _voxel_spacing(target_affine) if target_affine is not None else None,
        "affine_max_abs_diff_to_target": (
            float(np.abs(source_affine - target_affine).max())
            if target_affine is not None else None
        ),
    }
    audit["roi_processing"] = info
    if resampled:
        audit.setdefault("warnings", []).append(f"{source}:roi_resampled_to_base_grid")
        audit["warning_count"] = int(audit.get("warning_count", 0)) + 1
        audit["status"] = "warning"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit MRI/NIfTI geometry for FCD pipeline inputs.")
    p.add_argument("--dataset-dir", type=Path, default=None,
                   help="Audit every discoverable BIDS/ds004199 subject under this dataset root.")
    p.add_argument("--subjects", nargs="*", default=None,
                   help="Optional subject IDs to audit when --dataset-dir is used.")
    p.add_argument("--output-json", type=Path, default=None,
                   help="Optional path for audit JSON output.")
    p.add_argument("--output-csv", type=Path, default=None,
                   help="Optional path for per-subject CSV output when --dataset-dir is used.")
    p.add_argument("--flair", type=Path, default=None)
    p.add_argument("--t2w", type=Path, default=None)
    p.add_argument("--t1w", type=Path, default=None)
    p.add_argument("--roi", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.dataset_dir is not None:
        result = audit_dataset(
            args.dataset_dir,
            subjects=args.subjects,
            output_json=args.output_json,
            output_csv=args.output_csv,
        )
        print(json.dumps(result["summary"], indent=2))
        return

    audit = build_subject_data_quality(
        flair_path=args.flair,
        t2w_path=args.t2w,
        t1w_path=args.t1w,
        roi_path=args.roi,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
