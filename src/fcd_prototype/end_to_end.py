"""Dataset-wide end-to-end pipeline.

For each discovered subject:
  1. Build canonical patient volumes (FLAIR preferred, T2w then T1w fallback).
  2. Export a direct volumetric Gaussian splat PLY for Three.js.
  3. Fuzz the lesion ROI (FCD subjects only).
  4. Generate a synthetic normal brain (placeholder, control-prior, or nvidia mode).
  5. Compute residual = |fuzzed_patient - synthetic_normal|.
  6. Write a per-subject manifest.json under outputs/end_to_end/<subject>/.
  7. Write an index.json listing all processed subjects for the viewer.

Entry point: fcd-build-end-to-end
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import nibabel as nib
import numpy as np

from fcd_prototype.bids_discovery import SubjectRecord, discover_dataset, discover_subject, load_participants, load_sidecar
from fcd_prototype.candidate_roi import PROPOSAL_MODES, generate_candidate_rois
from fcd_prototype.lesion_fuzzer import fuzz_lesion
from fcd_prototype.mri_data_audit import (
    build_subject_data_quality,
    build_upload_readiness,
    build_viewer_metadata,
    record_roi_processing,
)
from fcd_prototype.nifti_preview import _robust_normalize
from fcd_prototype.residual_heatmap import compute_residual
from fcd_prototype.splat_ply_export import export_lesion_highlighted_splat_ply, export_splat_ply
from fcd_prototype.synthetic_normal import NvidiaDependencyError, generate_synthetic_normal


_GAUSSIAN_FIELD_COMPACT_KEYS = (
    "backend", "device", "training_objective", "final_loss",
    "final_mip_loss", "final_volume_loss", "volume_loss_weight",
    "volume_sample_count", "support_projection", "same_gaussian_support_as",
    "same_gaussian_support_as_fingerprint", "support_fingerprint_schema",
    "support_fingerprint_sha256", "support_points_in_volume_bounds_fraction",
    "num_gaussians", "steps",
)

_GAUSSIAN_FIELD_AUDIT_COMPACT_KEYS = (
    "schema", "sample_count", "seed", "device", "affine_mismatch",
    "sample_counts", "metrics", "checkpoint_metadata", "readiness_gate",
)

_CONTROL_GAUSSIAN_PRIOR_COMPACT_KEYS = (
    "method", "gaussian_state_schema", "same_gaussian_support",
    "healthy_reference", "registration_method", "intensity_matching",
    "n_controls", "min_controls", "n_gaussians", "max_points",
    "foreground_voxels_total", "sampling_mode", "density_scale_factor",
    "density_voxels_per_splat", "gaussian_orientation", "gaussian_scale_model",
    "control_resampled_to_patient_grid_count", "reference_mask_voxels",
    "std_floor_fraction", "std_floor_absolute", "std_floor_value",
    "std_clamped_count", "std_clamped_fraction", "heatmap_percentile",
    "heatmap_method", "heatmap_backend", "heatmap_active_gaussians",
    "heatmap_gaussian_sigma_extent", "heatmap_exact_max_points",
    "heatmap_effective_sigma_vox", "residual_z_normalization_value",
    "residual_z_min", "residual_z_max", "residual_z_mean",
    "residual_z_p95", "residual_z_p99", "residual_norm_mean",
    "residual_norm_p99", "roi_metrics_valid", "roi_gaussians",
    "non_roi_gaussians", "roi_to_non_roi_ratio",
    "roi_residual_concentration", "roi_path",
    "roi_resampled_to_patient_grid",
)


SyntheticMode = Literal["placeholder", "nvidia", "control-prior", "contralateral"]
RoiStrategy = Literal["provided", "auto", "provided-or-auto", "supervised", "provided-or-supervised", "none"]

ROI_LOCALIZATION_MIN_FCD_COVERAGE_GE_50_RATE = 0.50
ROI_LOCALIZATION_MAX_HC_FOREGROUND_FRACTION = 0.01
ROI_LOCALIZATION_MIN_FCD_SUBJECTS = 1


def _optional_float(value: object) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_control_group(group: str | None) -> bool:
    normalized = (group or "").strip().lower().replace("_", "-").replace(" ", "-")
    return normalized in {"hc", "control", "controls", "healthy", "healthy-control"}


def _preferred_modality_for_record(rec: SubjectRecord) -> str:
    if rec.flair is not None:
        return "FLAIR"
    if rec.t2w is not None:
        return "T2w"
    return "T1w"


def _pick_control_prior_volume(
    control: SubjectRecord,
    preferred_modality: str = "FLAIR",
) -> Optional[Path]:
    preferred = preferred_modality.strip().lower()
    if preferred == "t2w" or preferred == "t2":
        return control.t2w or control.flair or control.t1w
    if preferred == "t1w" or preferred == "t1":
        return control.t1w or control.flair or control.t2w
    return control.flair or control.t2w or control.t1w


def _extract_subject_id(path: str) -> str:
    for part in Path(path).parts:
        if part.startswith("sub-"):
            return part
    return Path(path).stem


def _data_quality_stage_status(data_quality: dict) -> str:
    readiness = data_quality.get("upload_readiness", {})
    if readiness.get("status") == "fail":
        return f"failed: {readiness.get('blocking_issue_count', 0)} blocker(s)"
    warning_count = int(data_quality.get("warning_count", 0))
    return "ok" if warning_count == 0 else f"warning: {warning_count} issue(s)"


def _control_prior_volumes_for(
    rec: SubjectRecord,
    all_records: list[SubjectRecord],
    preferred_modality: str,
    max_controls: Optional[int] = None,
) -> list[Path]:
    controls: list[tuple[str, Path]] = []
    for control in all_records:
        if control.subject_id == rec.subject_id:
            continue
        if not _is_control_group(control.group):
            continue
        path = _pick_control_prior_volume(control, preferred_modality=preferred_modality)
        if path is not None and path.is_file():
            controls.append((control.subject_id, path))
    controls.sort(key=lambda item: item[0])
    if max_controls is not None and max_controls > 0 and len(controls) > max_controls:
        digest = hashlib.sha256(rec.subject_id.encode("utf-8")).hexdigest()
        offset = int(digest[:8], 16) % len(controls)
        controls = (controls[offset:] + controls[:offset])[:max_controls]
    return [path for _, path in controls]


def _load_canonical(path: Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data, img.affine.copy(), img.header.copy()


def _save_nifti(data: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data, affine, header)
    img.header.set_data_dtype(np.float32)
    nib.save(img, str(path))
    return path


def _artifact_rel(path: Path, out_dir: Path) -> str:
    """Return a browser-friendly artifact path relative to a subject output dir."""
    return Path(path).relative_to(out_dir).as_posix()


def _resample_nearest(
    data: np.ndarray,
    src_affine: np.ndarray,
    ref_shape: tuple,
    ref_affine: np.ndarray,
) -> np.ndarray:
    i, j, k = np.meshgrid(
        np.arange(ref_shape[0]),
        np.arange(ref_shape[1]),
        np.arange(ref_shape[2]),
        indexing="ij",
    )
    ref_vox = np.stack([i, j, k, np.ones_like(i)], axis=-1).reshape(-1, 4)
    world = (ref_affine @ ref_vox.T).T[:, :3]
    src_inv = np.linalg.inv(src_affine)
    world_h = np.hstack([world, np.ones((len(world), 1))])
    src_vox = (src_inv @ world_h.T).T[:, :3]
    src_ijk = np.round(src_vox).astype(np.int32).reshape(ref_shape + (3,))
    np.clip(src_ijk[..., 0], 0, data.shape[0] - 1, out=src_ijk[..., 0])
    np.clip(src_ijk[..., 1], 0, data.shape[1] - 1, out=src_ijk[..., 1])
    np.clip(src_ijk[..., 2], 0, data.shape[2] - 1, out=src_ijk[..., 2])
    return data[src_ijk[..., 0], src_ijk[..., 1], src_ijk[..., 2]]


def _gate(status: str, interpretation: str, **observed: object) -> dict:
    return {
        "status": status,
        "interpretation": interpretation,
        **observed,
    }


def _normalize_roi_source_label(value: object) -> str:
    normalized = str(value or "unknown").strip().lower()
    normalized = normalized.replace("_", "-").replace(" ", "-")
    if normalized in {
        "provided", "manual", "manual-roi", "expert", "radiologist",
        "label", "ground-truth", "groundtruth", "annotation", "annotated",
    }:
        return "provided"
    if normalized in {"candidate", "auto", "automatic", "auto-candidate", "heuristic"}:
        return "candidate"
    if normalized in {"supervised", "model", "predicted", "segmentation-model"}:
        return "supervised"
    if normalized in {"none", "missing", "no-roi"}:
        return "none"
    return "unknown"


def _manual_overlap_independence_status(
    *,
    selected_roi_source: object,
    manual_roi_source: object,
    manual_roi_available: object,
) -> tuple[bool, str]:
    """Classify whether manual overlap is independent localization evidence."""
    if manual_roi_available is not True:
        return False, "manual_roi_unavailable"

    manual_source = _normalize_roi_source_label(manual_roi_source)
    if manual_source != "provided":
        return False, "manual_roi_source_not_oracle"

    selected_source = _normalize_roi_source_label(selected_roi_source)
    if selected_source == "provided":
        return False, "manual_roi_reused_as_fuzzing_roi"
    if selected_source == "unknown":
        return False, "selected_roi_source_unknown"
    return True, "independent_manual_roi_not_used_for_fuzzing"


def _is_detection_subject(manifest: dict) -> bool:
    return str(manifest.get("group", "unknown")).lower() not in (
        "hc", "control", "healthy", "healthy-control",
    )


def _roi_localization_evidence_gate(manifest: dict) -> dict:
    """Gate whether a generated ROI is evidence-backed enough for detection.

    ROI provenance only says where the mask came from. This gate says whether
    that source is defensible for an uploaded-MRI FCD detection claim.
    """
    roi_source = manifest.get("roi_source", "unknown")
    detection_claim = _is_detection_subject(manifest)
    supervised_meta = manifest.get("supervised_roi_meta", {})
    if not isinstance(supervised_meta, dict):
        supervised_meta = {}
    op_meta = manifest.get("supervised_roi_operating_point_meta", {})
    if not isinstance(op_meta, dict):
        op_meta = {}
    embedded_op = supervised_meta.get("operating_point")
    if not isinstance(embedded_op, dict):
        embedded_op = {}
    operating_point = embedded_op or op_meta
    operating_point_source = operating_point.get("source_kind")
    op_constraints = operating_point.get("constraints")
    if not isinstance(op_constraints, dict):
        op_constraints = {}
    selected_operating_point = operating_point.get("selected_operating_point")
    meets_operating_constraints = operating_point.get("meets_operating_constraints")
    fcd_coverage_ge_50_rate = _optional_float(
        operating_point.get("fcd_coverage_ge_0_50_rate")
    )
    hc_mean_foreground_fraction = _optional_float(
        operating_point.get("hc_mean_predicted_foreground_fraction")
    )
    n_fcd = _optional_int(operating_point.get("n_fcd"))
    required_min_coverage = _optional_float(
        op_constraints.get("min_coverage_ge_0_50_rate")
    )
    if required_min_coverage is None:
        required_min_coverage = _optional_float(
            operating_point.get("min_coverage_ge_0_50_rate")
        )
    if required_min_coverage is None:
        required_min_coverage = ROI_LOCALIZATION_MIN_FCD_COVERAGE_GE_50_RATE
    else:
        required_min_coverage = max(
            ROI_LOCALIZATION_MIN_FCD_COVERAGE_GE_50_RATE,
            required_min_coverage,
        )
    required_max_hc_fraction = _optional_float(
        op_constraints.get("max_hc_mean_predicted_foreground_fraction")
    )
    if required_max_hc_fraction is None:
        required_max_hc_fraction = _optional_float(
            operating_point.get("max_hc_mean_predicted_foreground_fraction")
        )
    if required_max_hc_fraction is None:
        required_max_hc_fraction = ROI_LOCALIZATION_MAX_HC_FOREGROUND_FRACTION
    else:
        required_max_hc_fraction = min(
            ROI_LOCALIZATION_MAX_HC_FOREGROUND_FRACTION,
            required_max_hc_fraction,
        )
    supervised_accepted = bool(supervised_meta.get("accepted"))
    structurally_calibrated_supervised = (
        roi_source == "supervised"
        and supervised_accepted
        and operating_point_source == "threshold_sweep"
        and "error" not in operating_point
    )
    has_population_metrics = (
        fcd_coverage_ge_50_rate is not None
        and hc_mean_foreground_fraction is not None
        and n_fcd is not None
    )
    calibrated_supervised = (
        structurally_calibrated_supervised
        and selected_operating_point is True
        and meets_operating_constraints is True
        and has_population_metrics
        and n_fcd >= ROI_LOCALIZATION_MIN_FCD_SUBJECTS
        and fcd_coverage_ge_50_rate >= required_min_coverage
        and hc_mean_foreground_fraction <= required_max_hc_fraction
    )

    if not detection_claim:
        status = "pass"
        reason = "not_required_for_control_or_non_detection_subject"
    elif calibrated_supervised:
        status = "pass"
        reason = "calibrated_supervised_roi_threshold_sweep_population_validated"
    elif structurally_calibrated_supervised and not has_population_metrics:
        status = "fail"
        reason = "supervised_roi_threshold_sweep_missing_population_metrics"
    elif structurally_calibrated_supervised:
        status = "fail"
        reason = "supervised_roi_threshold_sweep_failed_population_constraints"
    elif roi_source == "supervised" and supervised_accepted:
        status = "fail"
        reason = "supervised_roi_accepted_without_calibrated_threshold_sweep"
    elif roi_source == "supervised":
        status = "fail"
        reason = "supervised_roi_missing_or_rejected"
    elif roi_source == "candidate":
        status = "fail"
        reason = "heuristic_candidate_roi_unvalidated"
    elif roi_source == "provided":
        status = "fail"
        reason = "provided_manual_roi_is_oracle_for_detection_claim"
    elif roi_source == "none":
        status = "fail"
        reason = "no_pre_subtraction_roi_for_fuzzing"
    else:
        status = "fail"
        reason = "unknown_roi_source"

    return _gate(
        status,
        f"roi_source={roi_source}, reason={reason}",
        roi_source=roi_source,
        detection_claim=detection_claim,
        evidence_reason=reason,
        supervised_roi_accepted=supervised_accepted,
        operating_point_source_kind=operating_point_source,
        has_supervised_operating_point=bool(operating_point),
        selected_operating_point=selected_operating_point,
        meets_operating_constraints=meets_operating_constraints,
        fcd_coverage_ge_0_50_rate=fcd_coverage_ge_50_rate,
        required_min_fcd_coverage_ge_0_50_rate=required_min_coverage,
        hc_mean_predicted_foreground_fraction=hc_mean_foreground_fraction,
        required_max_hc_mean_predicted_foreground_fraction=required_max_hc_fraction,
        n_fcd=n_fcd,
        minimum_fcd_subjects=ROI_LOCALIZATION_MIN_FCD_SUBJECTS,
        accepted_detection_roi_sources=["supervised"],
        calibrated_supervised_required="held_out_threshold_sweep_with_population_metrics",
        heuristic_candidate_roi_allowed_for_detection=False,
        provided_manual_roi_allowed_for_detection=False,
    )


def build_product_readiness(manifest: dict) -> dict:
    """Assess product/paper readiness from manifest artifacts, metadata, and stage status.

    Returns a dict with overall_status, blockers, warnings, and gates. Each gate
    is a dict with status, interpretation, and observed values for debugging.
    """
    blockers: list[str] = []
    warnings: list[str] = []
    gates: dict[str, dict] = {}

    data_quality = manifest.get("data_quality", {})
    upload_readiness = data_quality.get("upload_readiness", {})
    stage_status = manifest.get("stage_status", {})
    artifacts = manifest.get("artifacts", {})

    dq_status = upload_readiness.get("status", "unknown")
    can_run_3dgs = upload_readiness.get("can_run_3dgs", False)
    can_subtract = upload_readiness.get("can_subtract_patient_normal", False)

    data_quality_ok = (
        dq_status in ("pass", "review")
        and bool(can_run_3dgs)
        and bool(can_subtract)
        and int(upload_readiness.get("blocking_issue_count", 0) or 0) == 0
    )
    gates["data_quality"] = {
        "status": "pass" if data_quality_ok else "fail",
        "interpretation": (
            f"upload_readiness={dq_status}, can_run_3dgs={can_run_3dgs}, "
            f"can_subtract={can_subtract}"
        ),
        "upload_readiness_status": dq_status,
        "can_run_3dgs": can_run_3dgs,
        "can_subtract_patient_normal": can_subtract,
        "blocking_issue_count": upload_readiness.get("blocking_issue_count", 0),
        "review_issue_count": upload_readiness.get("review_issue_count", 0),
    }
    if not data_quality_ok:
        blockers.append(f"data_quality failed: {upload_readiness.get('blockers', [])}")
    elif dq_status == "review":
        warnings.append(
            "data_quality is review: geometry/resampling issues need methods documentation"
        )

    patient_splat = "patient_splat_dense_ply" in artifacts or "patient_splat_ply" in artifacts
    splat_audit_meta = manifest.get("splat_reconstruction_audit_meta", {})
    audit_present = bool(splat_audit_meta)
    mri_to_3dgs_allowed = splat_audit_meta.get("mri_to_3dgs_claim_allowed", False)

    gates["direct_browser_splat"] = {
        "status": "pass" if (patient_splat and (not audit_present or mri_to_3dgs_allowed)) else "fail",
        "interpretation": (
            f"patient_splat={patient_splat}, audit_present={audit_present}, "
            f"mri_to_3dgs_allowed={mri_to_3dgs_allowed}"
        ),
        "has_patient_splat": patient_splat,
        "audit_present": audit_present,
        "mri_to_3dgs_claim_allowed": mri_to_3dgs_allowed,
    }
    if not patient_splat:
        blockers.append("No patient splat (patient_splat_ply or patient_splat_dense_ply)")
    elif audit_present and not mri_to_3dgs_allowed:
        blockers.append("splat_reconstruction_audit present but mri_to_3dgs_claim_allowed=false")

    opt_splat = "optimized_gaussian_splat_ply" in artifacts
    opt_checkpoint = "gaussian_field_checkpoint" in artifacts
    gf_meta = manifest.get("gaussian_field_meta", {})
    gf_backend = gf_meta.get("backend") if gf_meta else None
    gf_stage = stage_status.get("gaussian_field", "skipped")
    gf_audit_meta = manifest.get("optimized_gaussian_field_audit_meta", {})
    gf_audit_gate = gf_audit_meta.get("readiness_gate", {}) if gf_audit_meta else {}
    gf_audit_allowed = bool(
        gf_audit_gate.get("optimized_mri_to_3dgs_claim_allowed")
    )
    gf_audit_status = gf_audit_gate.get("status", "missing")
    gf_ok = (
        opt_splat
        and opt_checkpoint
        and gf_stage == "ok"
        and bool(gf_backend)
        and gf_audit_allowed
    )

    gates["optimized_gaussian_field"] = {
        "status": "pass" if gf_ok else "fail",
        "interpretation": (
            f"opt_splat={opt_splat}, opt_checkpoint={opt_checkpoint}, "
            f"backend={gf_backend}, stage_status={gf_stage}, "
            f"audit_status={gf_audit_status}, audit_claim_allowed={gf_audit_allowed}"
        ),
        "has_optimized_gaussian_splat_ply": opt_splat,
        "has_gaussian_field_checkpoint": opt_checkpoint,
        "gaussian_field_backend": gf_backend,
        "gaussian_field_stage_status": gf_stage,
        "optimized_gaussian_field_audit_status": gf_audit_status,
        "optimized_mri_to_3dgs_claim_allowed": gf_audit_allowed,
    }
    if not opt_splat or not opt_checkpoint:
        blockers.append("Optimized Gaussian field artifacts absent (required for paper 3DGS path)")
    elif gf_stage != "ok" or not gf_backend:
        blockers.append(
            f"gaussian_field stage/backend not claim-ready: stage={gf_stage}, backend={gf_backend}"
        )
    elif not gf_audit_allowed:
        blockers.append("optimized_gaussian_field_fidelity_not_proven")
    elif gf_backend != "gsplat":
        warnings.append(
            f"gaussian_field backend={gf_backend}; CUDA/gsplat is preferred for paper-scale runs"
        )

    synth_normal = "synthetic_normal" in artifacts
    synth_source = manifest.get("synthetic_normal_source", "none")
    synth_audit_meta = manifest.get("synthetic_normal_audit_meta", {})
    synth_audit_status = synth_audit_meta.get("subtraction_readiness_status", "unknown")
    synth_audit_allowed = synth_audit_meta.get("gaussian_subtraction_claim_allowed", False)
    synth_placeholder = str(synth_source).startswith("placeholder")

    gates["healthy_reference"] = {
        "status": (
            "pass"
            if (
                synth_normal
                and synth_audit_status == "pass"
                and synth_audit_allowed
                and not synth_placeholder
            )
            else "fail"
        ),
        "interpretation": (
            f"synth_normal={synth_normal}, subtraction_readiness={synth_audit_status}, "
            f"claim_allowed={synth_audit_allowed}, source={synth_source}"
        ),
        "has_synthetic_normal": synth_normal,
        "synthetic_normal_source": synth_source,
        "subtraction_readiness_status": synth_audit_status,
        "gaussian_subtraction_claim_allowed": synth_audit_allowed,
    }
    if not synth_normal:
        blockers.append("synthetic_normal artifact absent")
    elif synth_audit_status != "pass":
        blockers.append(f"synthetic_normal_audit subtraction_readiness_status={synth_audit_status}, not pass")
    elif not synth_audit_allowed:
        blockers.append("synthetic_normal_audit gaussian_subtraction_claim_allowed=false")
    elif synth_placeholder:
        blockers.append(
            "synthetic_normal_source=placeholder blocks product/paper healthy-reference claims"
        )

    res_splat = "gaussian_residual_splat" in artifacts
    res_heatmap = "gaussian_residual_heatmap" in artifacts
    res_metrics = "gaussian_residual_metrics" in artifacts
    res_scores = "gaussian_residual_scores" in artifacts
    res_meta = manifest.get("gaussian_residual_meta", {})
    res_same_support = res_meta.get("same_gaussian_support", False)
    res_claim_allowed = res_meta.get("gaussian_subtraction_claim_allowed", False)
    res_score_method = res_meta.get("residual_score_method", "unknown")
    res_heatmap_method = res_meta.get("heatmap_method", "unknown")
    res_fingerprint_match = res_meta.get("support_fingerprint_match", res_same_support)
    patient_support_fingerprint = res_meta.get("patient_support_fingerprint_sha256")
    normal_support_fingerprint = res_meta.get("normal_support_fingerprint_sha256")

    gates["optimized_gaussian_subtraction"] = {
        "status": (
            "pass" if (
                res_splat and res_heatmap and res_metrics and res_scores
                and res_same_support and res_claim_allowed
                and res_score_method == "indexwise_same_support_intensity"
                and res_heatmap_method == "field"
                and res_fingerprint_match
            ) else "fail"
        ),
        "interpretation": (
            f"artifacts={res_splat and res_heatmap and res_metrics and res_scores}, "
            f"same_support={res_same_support}, claim_allowed={res_claim_allowed}, "
            f"score_method={res_score_method}, heatmap_method={res_heatmap_method}, "
            f"support_fingerprint_match={res_fingerprint_match}"
        ),
        "has_gaussian_residual_splat": res_splat,
        "has_gaussian_residual_heatmap": res_heatmap,
        "has_gaussian_residual_metrics": res_metrics,
        "has_gaussian_residual_scores": res_scores,
        "same_gaussian_support": res_same_support,
        "gaussian_subtraction_claim_allowed": res_claim_allowed,
        "residual_score_method": res_score_method,
        "heatmap_method": res_heatmap_method,
        "support_fingerprint_match": res_fingerprint_match,
        "patient_support_fingerprint_sha256": patient_support_fingerprint,
        "normal_support_fingerprint_sha256": normal_support_fingerprint,
    }
    if not (res_splat and res_heatmap and res_metrics and res_scores):
        blockers.append("Optimized Gaussian residual artifacts incomplete")
    elif not res_same_support:
        blockers.append("gaussian_residual same_gaussian_support=false")
    elif not res_claim_allowed:
        blockers.append("gaussian_residual gaussian_subtraction_claim_allowed=false")
    elif res_score_method != "indexwise_same_support_intensity":
        blockers.append(f"gaussian_residual residual_score_method={res_score_method}, expected indexwise_same_support_intensity")
    elif res_heatmap_method != "field":
        blockers.append(f"gaussian_residual heatmap_method={res_heatmap_method}, expected field")
    elif not res_fingerprint_match:
        blockers.append("gaussian_residual support_fingerprint_match=false")

    patient_flair = "patient_flair" in artifacts
    patient_t2w = "patient_t2w" in artifacts
    patient_t1w = "patient_t1w" in artifacts
    res_heatmap_for_overlay = res_heatmap or "residual_heatmap" in artifacts

    gates["niivue_residual_overlay"] = {
        "status": "pass" if ((patient_flair or patient_t2w or patient_t1w) and res_heatmap_for_overlay) else "fail",
        "interpretation": (
            f"has_flair={patient_flair}, has_t2w={patient_t2w}, has_t1w={patient_t1w}, "
            f"has_residual_heatmap={res_heatmap_for_overlay}"
        ),
        "has_patient_flair": patient_flair,
        "has_patient_t2w": patient_t2w,
        "has_patient_t1w": patient_t1w,
        "has_residual_heatmap": res_heatmap_for_overlay,
    }
    if not ((patient_flair or patient_t2w or patient_t1w) and res_heatmap_for_overlay):
        warnings.append("niivue_residual_overlay cannot display: missing modality or heatmap")

    res_candidate_mask = "residual_candidate_mask" in artifacts
    is_control = manifest.get("group", "unknown").lower() in ("hc", "control", "healthy", "healthy-control")

    gates["lesion_candidate_overlay"] = {
        "status": "pass" if res_candidate_mask else ("warning" if is_control else "fail"),
        "interpretation": (
            f"has_residual_candidate_mask={res_candidate_mask}, is_control={is_control}"
        ),
        "has_residual_candidate_mask": res_candidate_mask,
        "is_control_or_no_roi": is_control,
    }
    if not res_candidate_mask and not is_control:
        blockers.append("residual_candidate_mask absent (required for FCD cases)")
    elif not res_candidate_mask and is_control:
        warnings.append("residual_candidate_mask absent (expected for control/no-ROI cases)")

    roi_source = manifest.get("roi_source", "unknown")
    detection_claim = manifest.get("group", "unknown").lower() not in (
        "hc", "control", "healthy", "healthy-control",
    )

    gates["roi_provenance"] = {
        "status": (
            "fail" if (roi_source == "provided" and detection_claim) else
            "pass" if roi_source in ("candidate", "supervised") else
            "warning"
        ),
        "interpretation": (
            f"roi_source={roi_source}, detection_claim={detection_claim}"
        ),
        "roi_source": roi_source,
        "detection_claim": detection_claim,
    }
    if roi_source == "provided" and detection_claim:
        blockers.append("roi_source=provided blocks uploaded-MRI detection claims (use auto/supervised instead)")
    elif roi_source == "none":
        warnings.append("roi_source=none: whole-brain or no-roi analysis")
    elif roi_source not in ("candidate", "supervised", "provided"):
        warnings.append(f"roi_source={roi_source} (unknown)")

    roi_evidence_gate = _roi_localization_evidence_gate(manifest)
    gates["roi_localization_evidence"] = roi_evidence_gate
    if roi_evidence_gate["status"] == "fail" and _is_detection_subject(manifest):
        blockers.append(
            "roi_localization_evidence failed: "
            f"{roi_evidence_gate['evidence_reason']}"
        )

    overall = "ready" if not blockers else "not_ready"

    return {
        "overall_status": overall,
        "blockers": blockers,
        "warnings": warnings,
        "gates": gates,
    }


def build_paper_method_feasibility(manifest: dict) -> dict:
    """Assess the original paper pitch separately from engineering readiness.

    Product readiness can pass for useful fallback workflows such as a
    control-prior or contralateral reference. The original scientific pitch is
    stricter: uploaded MRI -> MRI-native 3DGS -> lesion fuzzing -> NVIDIA
    patient-conditioned healthy counterfactual -> same-support Gaussian
    subtraction -> NiiVue/Three.js lesion review.
    """
    blockers: list[str] = []
    warnings: list[str] = []
    gates: dict[str, dict] = {}

    artifacts = manifest.get("artifacts", {})
    stage_status = manifest.get("stage_status", {})
    data_quality = manifest.get("data_quality", {})
    upload_readiness = data_quality.get("upload_readiness", {})
    splat_audit = manifest.get("splat_reconstruction_audit_meta", {})
    gaussian_field = manifest.get("gaussian_field_meta", {})
    gaussian_field_audit = manifest.get("optimized_gaussian_field_audit_meta", {})
    gaussian_residual = manifest.get("gaussian_residual_meta", {})
    synthetic_audit = manifest.get("synthetic_normal_audit_meta", {})
    nvidia_preflight = manifest.get("nvidia_preflight_meta", {})
    product_readiness = manifest.get("product_readiness", {})

    dq_status = upload_readiness.get("status", "unknown")
    data_quality_ok = (
        dq_status in ("pass", "review")
        and bool(upload_readiness.get("can_run_3dgs"))
        and bool(upload_readiness.get("can_subtract_patient_normal"))
        and int(upload_readiness.get("blocking_issue_count", 0) or 0) == 0
    )
    gates["input_data_quality"] = _gate(
        "pass" if data_quality_ok else "fail",
        (
            f"upload_readiness={dq_status}, "
            f"can_run_3dgs={upload_readiness.get('can_run_3dgs')}, "
            f"can_subtract={upload_readiness.get('can_subtract_patient_normal')}"
        ),
        upload_readiness_status=dq_status,
        can_run_3dgs=bool(upload_readiness.get("can_run_3dgs")),
        can_subtract_patient_normal=bool(
            upload_readiness.get("can_subtract_patient_normal")
        ),
        blocking_issue_count=upload_readiness.get("blocking_issue_count", 0),
    )
    if not data_quality_ok:
        blockers.append("input_data_quality")
    elif dq_status == "review":
        warnings.append("input_data_quality_review")

    direct_splat = "patient_splat_dense_ply" in artifacts or "patient_splat_ply" in artifacts
    direct_audit_allowed = bool(splat_audit.get("mri_to_3dgs_claim_allowed"))
    direct_status = "pass" if direct_splat and direct_audit_allowed else "fail"
    gates["direct_threejs_splat"] = _gate(
        direct_status,
        (
            f"has_browser_ply={direct_splat}, "
            f"mri_to_3dgs_claim_allowed={direct_audit_allowed}"
        ),
        has_browser_ply=direct_splat,
        mri_to_3dgs_claim_allowed=direct_audit_allowed,
        mri_to_3dgs_readiness_status=splat_audit.get(
            "mri_to_3dgs_readiness_status", "unknown"
        ),
        foreground_coverage=splat_audit.get("foreground_coverage"),
        mip_mean_mae=splat_audit.get("mip_mean_mae"),
    )
    if direct_status != "pass":
        blockers.append("direct_threejs_splat")

    optimized_has_artifacts = (
        "optimized_gaussian_splat_ply" in artifacts
        and "gaussian_field_checkpoint" in artifacts
    )
    objective = gaussian_field.get("training_objective")
    volume_loss_weight = float(gaussian_field.get("volume_loss_weight") or 0.0)
    volume_sample_count = int(gaussian_field.get("volume_sample_count") or 0)
    optimized_volume_native = (
        objective == "mip_projection_mse_plus_volume_sample_mse"
        and volume_loss_weight > 0.0
        and volume_sample_count > 0
    )
    optimized_stage_ok = stage_status.get("gaussian_field") == "ok"
    optimized_audit_gate = (
        gaussian_field_audit.get("readiness_gate", {})
        if gaussian_field_audit else {}
    )
    optimized_audit_allowed = bool(
        optimized_audit_gate.get("optimized_mri_to_3dgs_claim_allowed")
    )
    optimized_status = (
        "pass"
        if (
            optimized_has_artifacts
            and optimized_stage_ok
            and optimized_volume_native
            and optimized_audit_allowed
        )
        else "fail"
    )
    gates["optimized_mri_native_3dgs"] = _gate(
        optimized_status,
        (
            f"artifacts={optimized_has_artifacts}, stage={stage_status.get('gaussian_field')}, "
            f"objective={objective}, volume_loss_weight={volume_loss_weight}, "
            f"volume_sample_count={volume_sample_count}, "
            f"audit_status={optimized_audit_gate.get('status', 'missing')}, "
            f"audit_claim_allowed={optimized_audit_allowed}"
        ),
        has_optimized_gaussian_splat_ply="optimized_gaussian_splat_ply" in artifacts,
        has_gaussian_field_checkpoint="gaussian_field_checkpoint" in artifacts,
        gaussian_field_stage_status=stage_status.get("gaussian_field", "unknown"),
        backend=gaussian_field.get("backend"),
        training_objective=objective,
        volume_loss_weight=volume_loss_weight,
        volume_sample_count=volume_sample_count,
        optimized_gaussian_field_audit_status=optimized_audit_gate.get(
            "status", "missing"
        ),
        optimized_mri_to_3dgs_claim_allowed=optimized_audit_allowed,
    )
    if optimized_status != "pass":
        blockers.append("optimized_mri_native_3dgs")
    elif gaussian_field.get("backend") != "gsplat":
        warnings.append("optimized_mri_native_3dgs_backend_not_gsplat")

    roi_source = manifest.get("roi_source", "unknown")
    detection_claim = str(manifest.get("group", "unknown")).lower() not in (
        "hc", "control", "healthy", "healthy-control",
    )
    roi_ok_for_uploaded_detection = roi_source in ("candidate", "supervised")
    gates["uploaded_detection_roi_provenance"] = _gate(
        "pass" if (roi_ok_for_uploaded_detection or not detection_claim) else "fail",
        f"roi_source={roi_source}, detection_claim={detection_claim}",
        roi_source=roi_source,
        detection_claim=detection_claim,
        acceptable_uploaded_detection_sources=["candidate", "supervised"],
    )
    if detection_claim and not roi_ok_for_uploaded_detection:
        blockers.append("uploaded_detection_roi_provenance")

    roi_evidence_gate = _roi_localization_evidence_gate(manifest)
    gates["roi_localization_evidence"] = roi_evidence_gate
    if roi_evidence_gate["status"] == "fail" and detection_claim:
        blockers.append("roi_localization_evidence")

    lesion_fuzzed = "fuzzed_patient_flair" in artifacts or "fuzzed_patient_base" in artifacts
    fuzz_stage = str(stage_status.get("lesion_fuzzing", "unknown"))
    gates["lesion_fuzzing"] = _gate(
        "pass" if lesion_fuzzed and fuzz_stage.startswith("ok") else "fail",
        f"artifact={lesion_fuzzed}, stage={fuzz_stage}",
        has_fuzzed_patient=lesion_fuzzed,
        lesion_fuzzing_stage_status=fuzz_stage,
    )
    if not (lesion_fuzzed and fuzz_stage.startswith("ok")):
        blockers.append("lesion_fuzzing")

    nvidia_claim_readiness = nvidia_preflight.get("paper_claim_readiness", {}) or {}
    nvidia_patient_conditioned = bool(
        nvidia_preflight.get("ready_for_patient_conditioned_counterfactual")
        or nvidia_preflight.get("patient_conditioned_inpainting_supported")
    )
    model_card_supports_original_pitch = nvidia_claim_readiness.get(
        "model_card_supports_original_pitch"
    )
    if model_card_supports_original_pitch is None:
        model_card_supports_original_pitch = nvidia_patient_conditioned
    model_card_supports_patient_conditioned = nvidia_claim_readiness.get(
        "model_card_supports_patient_conditioned_counterfactual"
    )
    if model_card_supports_patient_conditioned is None:
        model_card_supports_patient_conditioned = nvidia_patient_conditioned
    released_workflow_accepts_patient_roi = nvidia_claim_readiness.get(
        "released_workflow_accepts_patient_mri_or_roi_mask"
    )
    if released_workflow_accepts_patient_roi is None:
        released_workflow_accepts_patient_roi = nvidia_patient_conditioned
    healthy_patient_conditioned = bool(
        synthetic_audit.get("patient_conditioned_counterfactual_claim_allowed")
    )
    nvidia_ready = bool(nvidia_preflight.get("ready_for_inference"))
    nvidia_gate_pass = bool(
        nvidia_ready
        and nvidia_patient_conditioned
        and bool(model_card_supports_original_pitch)
        and bool(model_card_supports_patient_conditioned)
        and bool(released_workflow_accepts_patient_roi)
        and healthy_patient_conditioned
    )
    gates["nvidia_patient_conditioned_counterfactual"] = _gate(
        "pass" if nvidia_gate_pass else "fail",
        (
            f"ready_for_inference={nvidia_ready}, "
            f"ready_for_patient_conditioned_counterfactual={nvidia_patient_conditioned}, "
            f"model_card_supports_patient_conditioned={bool(model_card_supports_patient_conditioned)}, "
            f"released_workflow_accepts_patient_roi={bool(released_workflow_accepts_patient_roi)}, "
            f"healthy_reference_patient_conditioned={healthy_patient_conditioned}"
        ),
        synthetic_normal_source=manifest.get("synthetic_normal_source", "none"),
        ready_for_inference=nvidia_ready,
        ready_for_patient_conditioned_counterfactual=nvidia_patient_conditioned,
        model_card_task=nvidia_claim_readiness.get("model_card_task"),
        model_card_supports_original_pitch=bool(model_card_supports_original_pitch),
        model_card_supports_patient_conditioned_counterfactual=bool(
            model_card_supports_patient_conditioned
        ),
        model_card_supports_fcd_mask_guided_generation=bool(
            nvidia_claim_readiness.get("model_card_supports_fcd_mask_guided_generation")
        ),
        released_workflow_accepts_patient_mri_or_roi_mask=bool(
            released_workflow_accepts_patient_roi
        ),
        paper_claim_requires_new_conditioning_method=bool(
            nvidia_claim_readiness.get("paper_claim_requires_new_conditioning_method")
        ),
        patient_conditioned_blocker_reason=nvidia_claim_readiness.get(
            "patient_conditioned_blocker_reason"
        ),
        patient_conditioned_counterfactual_claim_allowed=healthy_patient_conditioned,
        patient_conditioned_inpainting=synthetic_audit.get(
            "patient_conditioned_inpainting"
        ),
        nvidia_claim_boundary=nvidia_claim_readiness.get("claim_boundary"),
        synthetic_claim_boundary=synthetic_audit.get("claim_boundary"),
    )
    if not nvidia_gate_pass:
        blockers.append("nvidia_patient_conditioned_counterfactual")

    same_support_subtraction = bool(
        gaussian_residual.get("same_gaussian_support")
        and gaussian_residual.get("support_fingerprint_match", True)
        and gaussian_residual.get("gaussian_subtraction_claim_allowed")
        and gaussian_residual.get("residual_score_method")
        == "indexwise_same_support_intensity"
    )
    subtraction_artifacts = all(
        key in artifacts
        for key in (
            "gaussian_residual_splat",
            "gaussian_residual_heatmap",
            "gaussian_residual_metrics",
            "gaussian_residual_scores",
        )
    )
    gates["same_support_gaussian_subtraction"] = _gate(
        "pass" if same_support_subtraction and subtraction_artifacts else "fail",
        (
            f"artifacts={subtraction_artifacts}, same_support={gaussian_residual.get('same_gaussian_support')}, "
            f"fingerprint_match={gaussian_residual.get('support_fingerprint_match')}, "
            f"score_method={gaussian_residual.get('residual_score_method')}"
        ),
        has_gaussian_residual_artifacts=subtraction_artifacts,
        same_gaussian_support=gaussian_residual.get("same_gaussian_support", False),
        support_fingerprint_match=gaussian_residual.get(
            "support_fingerprint_match", False
        ),
        gaussian_subtraction_claim_allowed=gaussian_residual.get(
            "gaussian_subtraction_claim_allowed", False
        ),
        residual_score_method=gaussian_residual.get("residual_score_method"),
        heatmap_method=gaussian_residual.get("heatmap_method"),
    )
    if not (same_support_subtraction and subtraction_artifacts):
        blockers.append("same_support_gaussian_subtraction")

    niivue_ready = (
        ("patient_flair" in artifacts or "patient_t2w" in artifacts or "patient_t1w" in artifacts)
        and ("gaussian_residual_heatmap" in artifacts or "residual_heatmap" in artifacts)
    )
    gates["niivue_residual_review"] = _gate(
        "pass" if niivue_ready else "fail",
        f"has_base_mri_and_residual_heatmap={niivue_ready}",
        has_patient_base_mri=(
            "patient_flair" in artifacts
            or "patient_t2w" in artifacts
            or "patient_t1w" in artifacts
        ),
        has_residual_heatmap=(
            "gaussian_residual_heatmap" in artifacts or "residual_heatmap" in artifacts
        ),
    )
    if not niivue_ready:
        blockers.append("niivue_residual_review")

    original_pitch_feasible = len(blockers) == 0
    fallback_possible = (
        product_readiness.get("overall_status") == "ready"
        or any(
            key in artifacts
            for key in (
                "paired_gaussian_residual_splat",
                "gaussian_residual_splat",
                "residual_heatmap",
            )
        )
    )

    claim_boundaries = {
        "original_pitch_claim": (
            "patient_mri_to_mri_native_3dgs_to_nvidia_patient_conditioned_healthy_counterfactual_to_same_support_gaussian_subtraction"
        ),
        "original_pitch_feasible": original_pitch_feasible,
        "nvidia_patient_conditioned_counterfactual_required": True,
        "nvidia_patient_conditioned_counterfactual_available": nvidia_gate_pass,
        "nvidia_model_card_supports_original_pitch": bool(
            model_card_supports_original_pitch
        ),
        "nvidia_model_card_supports_patient_conditioned_counterfactual": bool(
            model_card_supports_patient_conditioned
        ),
        "nvidia_released_workflow_accepts_patient_mri_or_roi_mask": bool(
            released_workflow_accepts_patient_roi
        ),
        "nvidia_paper_claim_requires_new_conditioning_method": bool(
            nvidia_claim_readiness.get("paper_claim_requires_new_conditioning_method")
        ),
        "fallback_population_or_contralateral_prior_possible": bool(fallback_possible),
        "fallback_is_not_original_pitch": True,
    }

    return {
        "overall_status": (
            "feasible_for_original_pitch"
            if original_pitch_feasible
            else "not_feasible_for_original_pitch"
        ),
        "fallback_status": (
            "fallback_research_workflow_possible"
            if fallback_possible
            else "fallback_research_workflow_unproven"
        ),
        "blocking_gates": blockers,
        "warnings": warnings,
        "gates": gates,
        "claim_boundaries": claim_boundaries,
        "recommended_next_step": (
            "Run cohort-scale validation of the original method."
            if original_pitch_feasible
            else (
                "Do not claim NVIDIA patient-specific lesion inpainting; implement "
                "or acquire a patient-conditioned healthy-reference model, or frame "
                "the paper around same-support MRI 3DGS residuals with a validated "
                "control/contralateral prior."
            )
        ),
    }


def process_subject(
    rec: SubjectRecord,
    output_root: Path,
    synthetic_mode: SyntheticMode = "placeholder",
    roi_strategy: RoiStrategy = "provided",
    candidate_roi_proposal_mode: str = "brightness-gradient",
    max_splat_points: int = 50_000,
    rng_seed: int = 42,
    force: bool = False,
    splat_sampling_mode: str = "surface",
    presentation_splat_points: int = 300_000,
    presentation_splat_sampling_mode: str = "volume",
    compute_splat_reconstruction_audit: bool = True,
    splat_reconstruction_audit_max_points: int = 100_000,
    splat_reconstruction_audit_smooth_sigma: float = 1.0,
    splat_reconstruction_audit_kernel_mode: str = "gaussian",
    splat_reconstruction_audit_gaussian_sigma_extent: float = 2.5,
    train_gaussian_field: bool = False,
    gaussian_num_gaussians: int = 20_000,
    gaussian_steps: int = 300,
    gaussian_image_size: int = 128,
    gaussian_views: str = "all",
    gaussian_device: str = "auto",
    gaussian_learning_rate: float = 1e-3,
    gaussian_volume_loss_weight: float = 0.0,
    gaussian_volume_sample_count: int = 4096,
    compute_optimized_gaussian_field_audit: bool = True,
    optimized_gaussian_field_audit_sample_count: int = 8192,
    optimized_gaussian_field_audit_device: str = "cpu",
    train_synthetic_gaussian_field: bool = False,
    compute_paired_gaussian_residual: bool = True,
    paired_gaussian_max_points: Optional[int] = None,
    paired_gaussian_heatmap_percentile: float = 99.0,
    paired_gaussian_smooth_sigma: float = 1.0,
    paired_gaussian_heatmap_method: str = "gaussian",
    paired_gaussian_heatmap_sigma_extent: float = 2.5,
    paired_gaussian_heatmap_exact_max_points: int = 50_000,
    paired_gaussian_support_basis: str = "union",
    compute_control_gaussian_prior_residual: bool = False,
    control_gaussian_prior_max_points: Optional[int] = None,
    control_gaussian_prior_min_controls: int = 2,
    control_gaussian_prior_heatmap_percentile: float = 99.0,
    control_gaussian_prior_smooth_sigma: float = 1.0,
    control_gaussian_prior_heatmap_method: str = "gaussian",
    control_gaussian_prior_heatmap_exact_max_points: int = 50_000,
    control_gaussian_prior_std_floor_fraction: float = 0.05,
    compute_residual_candidates: bool = True,
    residual_candidate_max_candidates: int = 5,
    residual_candidate_percentile: float = 90.0,
    residual_candidate_min_threshold: float = 0.05,
    residual_candidate_min_component_voxels: int = 8,
    residual_candidate_dilate_mm: float = 5.0,
    compute_gaussian_residual: bool = False,
    gaussian_residual_k_neighbors: int = 4,
    gaussian_residual_max_points: int = 200_000,
    gaussian_residual_heatmap_percentile: float = 99.0,
    gaussian_residual_heatmap_method: str = "field",
    gaussian_residual_field_chunk_size: int = 128,
    supervised_roi_checkpoint: Optional[Path] = None,
    supervised_roi_operating_point: Optional[Path] = None,
    supervised_roi_threshold: float = 0.5,
    supervised_roi_min_component_voxels: int = 0,
    supervised_roi_min_voxels: int = 1,
    supervised_roi_max_foreground_fraction: Optional[float] = 0.01,
    supervised_roi_device: str = "auto",
    supervised_roi_selection_mode: str = "threshold",
    supervised_roi_foreground_fraction: Optional[float] = None,
    control_prior_volumes: Optional[list[Path]] = None,
    pipeline_preset: Optional[str] = None,
) -> dict:
    """Run end-to-end pipeline for one subject and return its manifest."""
    out_dir = output_root / rec.subject_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    if manifest_path.exists() and not force:
        print(f"  {rec.subject_id}: manifest exists, skipping (--force to rebuild)")
        with manifest_path.open() as fh:
            return json.load(fh)

    artifacts: dict[str, str] = {}
    stage_status: dict[str, str] = {}
    notes: list[str] = []
    limitations: list[str] = []

    has_roi = rec.flair_roi is not None and Path(rec.flair_roi).is_file()
    modalities: list[str] = []
    data_quality = build_subject_data_quality(
        flair_path=rec.flair,
        t2w_path=rec.t2w,
        t1w_path=rec.t1w,
        roi_path=rec.flair_roi,
    )
    stage_status["data_quality"] = _data_quality_stage_status(data_quality)
    notes.extend(f"Data quality warning: {w}" for w in data_quality.get("warnings", []))

    # -------------------------------------------------------------------------
    # Stage 1: canonical patient volumes
    # -------------------------------------------------------------------------
    base_path: Optional[Path] = None
    base_shape: Optional[tuple] = None
    base_affine: Optional[np.ndarray] = None
    base_header: Optional[nib.Nifti1Header] = None

    try:
        if rec.flair and rec.flair.is_file():
            flair_data, base_affine, base_header = _load_canonical(rec.flair)
            base_shape = flair_data.shape[:3]
            flair_norm = _robust_normalize(flair_data)
            flair_out = out_dir / "patient_flair.nii.gz"
            _save_nifti(flair_norm, base_affine, base_header, flair_out)
            artifacts["patient_flair"] = flair_out.name
            base_path = flair_out
            modalities.append("FLAIR")

        if rec.t2w and rec.t2w.is_file():
            t2w_data, t2w_affine, t2w_header = _load_canonical(rec.t2w)
            t2w_norm = _robust_normalize(t2w_data)
            t2w_out = out_dir / "patient_t2w.nii.gz"
            _save_nifti(t2w_norm, t2w_affine, t2w_header, t2w_out)
            artifacts["patient_t2w"] = t2w_out.name
            modalities.append("T2w")
            if base_path is None:
                base_path = t2w_out
                base_shape = t2w_data.shape[:3]
                base_affine = t2w_affine
                base_header = t2w_header

        if rec.t1w and rec.t1w.is_file():
            t1w_data, t1w_affine, t1w_header = _load_canonical(rec.t1w)
            t1w_norm = _robust_normalize(t1w_data)
            t1w_out = out_dir / "patient_t1w.nii.gz"
            _save_nifti(t1w_norm, t1w_affine, t1w_header, t1w_out)
            artifacts["patient_t1w"] = t1w_out.name
            modalities.append("T1w")
            if base_path is None:
                base_path = t1w_out
                base_shape = t1w_data.shape[:3]
                base_affine = t1w_affine
                base_header = t1w_header

        if base_path is None:
            stage_status["patient_volumes"] = "failed: no FLAIR, T2w, or T1w found"
            notes.append("No FLAIR, T2w, or T1w found - cannot process this subject.")
        else:
            stage_status["patient_volumes"] = "ok"
    except Exception as exc:
        stage_status["patient_volumes"] = f"failed: {exc}"
        notes.append(f"Patient volume stage failed: {exc}")

    # Keep the provided lesion mask available strictly for validation, even
    # when the active ROI strategy uses generated candidates for fuzzing.
    manual_roi_validation_path: Optional[Path] = None
    manual_roi_validation_source = "none"
    if has_roi and base_path is not None:
        try:
            manual_raw, manual_affine_src, _ = _load_canonical(Path(rec.flair_roi))
            manual_bin = (manual_raw > 0).astype(np.float32)
            manual_resampled = (
                manual_bin.shape[:3] != base_shape
                or not np.allclose(manual_affine_src, base_affine)
            )
            if manual_resampled:
                manual_bin = _resample_nearest(
                    manual_bin, manual_affine_src, base_shape, base_affine
                )
            manual_roi_validation_path = out_dir / "manual_lesion_roi.nii.gz"
            _save_nifti(
                manual_bin,
                base_affine,
                base_header,
                manual_roi_validation_path,
            )
            artifacts["manual_lesion_roi"] = manual_roi_validation_path.name
            manual_roi_validation_source = "provided"
            stage_status["manual_validation_roi"] = "ok"
        except Exception as exc:
            manual_roi_validation_path = None
            manual_roi_validation_source = "failed"
            stage_status["manual_validation_roi"] = f"failed: {exc}"
            notes.append(f"Manual validation ROI stage failed: {exc}")
    elif has_roi:
        stage_status["manual_validation_roi"] = "skipped: no base volume"
    else:
        stage_status["manual_validation_roi"] = "skipped: no provided ROI"

    # -------------------------------------------------------------------------
    # Stage 2a: Candidate ROI generation (optional, auto mode)
    # -------------------------------------------------------------------------
    candidate_roi_used = False
    should_generate_candidates = (
        (roi_strategy == "auto" and base_path is not None) or
        (roi_strategy == "provided-or-auto" and base_path is not None and not has_roi)
    )
    if should_generate_candidates:
        try:
            candidate_dir = out_dir / "candidate_roi"
            candidate_manifest = generate_candidate_rois(
                volume_path=base_path,
                output_dir=candidate_dir,
                proposal_mode=candidate_roi_proposal_mode,
            )
            union_path = candidate_dir / "candidate_roi_union.nii.gz"
            if union_path.is_file():
                artifacts["candidate_roi_union"] = f"candidate_roi/{union_path.name}"
                for idx in range(candidate_manifest.get("selected_count", 0)):
                    fname = f"candidate_roi_{idx:02d}.nii.gz"
                    artifacts[f"candidate_roi_{idx:02d}"] = f"candidate_roi/{fname}"
                artifacts["candidate_roi_manifest"] = "candidate_roi/candidate_roi_manifest.json"
                stage_status["candidate_roi"] = f"ok ({candidate_manifest['selected_count']} candidates)"
                notes.append(
                    f"Generated {candidate_manifest['selected_count']} candidate ROI(s) "
                    f"from {candidate_roi_proposal_mode} heuristic signal proposal "
                    "(NOT validated detection)."
                )
                candidate_roi_used = True
        except Exception as exc:
            stage_status["candidate_roi"] = f"failed: {exc}"
            notes.append(f"Candidate ROI generation failed: {exc}")
    else:
        if roi_strategy == "auto":
            stage_status["candidate_roi"] = "skipped: no base volume"
        elif roi_strategy == "provided-or-auto":
            if base_path is None:
                stage_status["candidate_roi"] = "skipped: no base volume"
            else:
                stage_status["candidate_roi"] = "skipped: provided ROI available"
        else:
            stage_status["candidate_roi"] = f"skipped: roi_strategy={roi_strategy}"

    # -------------------------------------------------------------------------
    # Stage 2b: Supervised ROI prediction (optional)
    # -------------------------------------------------------------------------
    supervised_roi_used = False
    supervised_roi_meta: dict[str, object] = {}
    supervised_roi_operating_point_meta: dict[str, object] = {}
    supervised_roi_operating_point_failed = False
    if (
        supervised_roi_operating_point is not None
        and roi_strategy in ("supervised", "provided-or-supervised")
    ):
        try:
            from fcd_prototype.supervised_lesion_proposal import (
                load_supervised_roi_operating_point,
            )
            supervised_roi_operating_point_meta = load_supervised_roi_operating_point(
                supervised_roi_operating_point
            )
            if "threshold" in supervised_roi_operating_point_meta:
                supervised_roi_threshold = float(
                    supervised_roi_operating_point_meta["threshold"]
                )
            if "selection_mode" in supervised_roi_operating_point_meta:
                supervised_roi_selection_mode = str(
                    supervised_roi_operating_point_meta["selection_mode"]
                )
            if "foreground_fraction" in supervised_roi_operating_point_meta:
                supervised_roi_foreground_fraction = float(
                    supervised_roi_operating_point_meta["foreground_fraction"]
                )
            if "min_component_voxels" in supervised_roi_operating_point_meta:
                supervised_roi_min_component_voxels = int(
                    supervised_roi_operating_point_meta["min_component_voxels"]
                )
            if "max_foreground_fraction" in supervised_roi_operating_point_meta:
                supervised_roi_max_foreground_fraction = float(
                    supervised_roi_operating_point_meta["max_foreground_fraction"]
                )
            stage_status["supervised_roi_operating_point"] = "ok"
        except Exception as exc:
            supervised_roi_operating_point_failed = True
            supervised_roi_operating_point_meta = {
                "source_path": str(supervised_roi_operating_point),
                "error": str(exc),
            }
            stage_status["supervised_roi_operating_point"] = f"failed: {exc}"
            notes.append(f"Supervised ROI operating point failed: {exc}")

    supervised_roi_resolved_params = {
        "checkpoint": str(supervised_roi_checkpoint) if supervised_roi_checkpoint else None,
        "operating_point": (
            str(supervised_roi_operating_point)
            if supervised_roi_operating_point is not None else None
        ),
        "threshold": supervised_roi_threshold,
        "min_component_voxels": supervised_roi_min_component_voxels,
        "min_voxels": supervised_roi_min_voxels,
        "max_foreground_fraction": supervised_roi_max_foreground_fraction,
        "selection_mode": supervised_roi_selection_mode,
        "foreground_fraction": supervised_roi_foreground_fraction,
        "device": supervised_roi_device,
    }
    should_run_supervised = (
        (roi_strategy == "supervised" and base_path is not None) or
        (roi_strategy == "provided-or-supervised" and base_path is not None and not has_roi)
    )

    if should_run_supervised:
        if supervised_roi_operating_point_failed:
            stage_status["supervised_roi"] = "skipped: invalid operating point"
        elif supervised_roi_checkpoint is None or not Path(supervised_roi_checkpoint).is_file():
            stage_status["supervised_roi"] = "skipped: checkpoint not provided or missing"
            notes.append("Supervised ROI prediction skipped - checkpoint not available.")
        else:
            try:
                from fcd_prototype.supervised_lesion_proposal import predict_lesion_roi
                supervised_dir = out_dir / "supervised_roi"
                supervised_dir.mkdir(parents=True, exist_ok=True)
                pred_path = supervised_dir / "predicted_lesion_roi.nii.gz"
                pred_result = predict_lesion_roi(
                    checkpoint_path=supervised_roi_checkpoint,
                    volume_path=base_path,
                    output_path=pred_path,
                    threshold=supervised_roi_threshold,
                    min_component_voxels=supervised_roi_min_component_voxels,
                    device=supervised_roi_device,
                    selection_mode=supervised_roi_selection_mode,
                    foreground_fraction=supervised_roi_foreground_fraction,
                )
                pred_raw, _pred_affine, _pred_header = _load_canonical(pred_path)
                pred_mask = pred_raw > 0
                pred_voxels = int(pred_mask.sum())
                pred_fraction = (
                    float(pred_voxels / pred_mask.size) if pred_mask.size else 0.0
                )
                pred_result["predicted_voxels"] = pred_voxels
                pred_result["predicted_foreground_fraction"] = pred_fraction
                acceptance_gate = {
                    "accepted": True,
                    "rejection_reason": None,
                    "min_predicted_voxels": supervised_roi_min_voxels,
                    "max_predicted_foreground_fraction": supervised_roi_max_foreground_fraction,
                    "predicted_voxels": pred_voxels,
                    "predicted_foreground_fraction": pred_fraction,
                    "threshold": supervised_roi_threshold,
                    "min_component_voxels": supervised_roi_min_component_voxels,
                    "selection_mode": supervised_roi_selection_mode,
                    "foreground_fraction": supervised_roi_foreground_fraction,
                    "operating_point": supervised_roi_operating_point_meta or None,
                }
                if pred_voxels < supervised_roi_min_voxels:
                    acceptance_gate["accepted"] = False
                    acceptance_gate["rejection_reason"] = "below_min_predicted_voxels"
                elif (
                    supervised_roi_max_foreground_fraction is not None
                    and pred_fraction > supervised_roi_max_foreground_fraction
                ):
                    acceptance_gate["accepted"] = False
                    acceptance_gate["rejection_reason"] = "above_max_foreground_fraction"

                pred_result["acceptance_gate"] = acceptance_gate
                supervised_roi_meta = acceptance_gate.copy()
                artifacts["supervised_roi"] = f"supervised_roi/{pred_path.name}"
                pred_meta_path = supervised_dir / "supervised_roi_prediction.json"
                with pred_meta_path.open("w") as fh:
                    json.dump(pred_result, fh, indent=2)
                artifacts["supervised_roi_meta"] = f"supervised_roi/{pred_meta_path.name}"
                if acceptance_gate["accepted"]:
                    stage_status["supervised_roi"] = "ok"
                    supervised_roi_used = True
                    notes.append(
                        f"Supervised ROI accepted: {pred_voxels} voxels "
                        f"({pred_fraction:.4f} foreground, prob mean "
                        f"{pred_result.get('probability_mean', 0):.3f})"
                    )
                else:
                    stage_status["supervised_roi"] = (
                        f"rejected: {acceptance_gate['rejection_reason']}"
                    )
                    notes.append(
                        f"Supervised ROI rejected: {pred_voxels} voxels "
                        f"({pred_fraction:.4f} foreground), reason="
                        f"{acceptance_gate['rejection_reason']}."
                    )
            except Exception as exc:
                stage_status["supervised_roi"] = f"failed: {exc}"
                notes.append(f"Supervised ROI prediction failed: {exc}")
    else:
        if roi_strategy in ("supervised", "provided-or-supervised"):
            if base_path is None:
                stage_status["supervised_roi"] = "skipped: no base volume"
            elif roi_strategy == "provided-or-supervised" and has_roi:
                stage_status["supervised_roi"] = "skipped: provided ROI available"
            else:
                stage_status["supervised_roi"] = f"skipped: roi_strategy={roi_strategy}"
        else:
            stage_status["supervised_roi"] = f"skipped: roi_strategy={roi_strategy}"

    # -------------------------------------------------------------------------
    # Stage 2c: ROI mask
    # -------------------------------------------------------------------------
    roi_out_path: Optional[Path] = None
    roi_source = "none"

    # Determine which ROI to use based on strategy
    # 'auto': use candidate if available, never use provided
    # 'provided-or-auto': prefer provided, fallback to candidate if provided unavailable
    # 'provided': use provided only
    # 'supervised': use supervised prediction if available
    # 'provided-or-supervised': prefer provided, fallback to supervised if provided unavailable
    # 'none': never use ROI
    use_provided = has_roi and (roi_strategy in ("provided", "provided-or-auto", "provided-or-supervised"))
    use_candidate = candidate_roi_used and (
        (roi_strategy == "auto") or
        (roi_strategy == "provided-or-auto" and not has_roi)
    )
    use_supervised = supervised_roi_used and (
        (roi_strategy == "supervised") or
        (roi_strategy == "provided-or-supervised" and not has_roi)
    )

    if roi_strategy == "none":
        stage_status["roi_mask"] = "skipped: roi_strategy=none"
        notes.append("ROI mask skipped by roi_strategy=none.")
    elif use_provided:
        try:
            roi_raw, roi_affine_src, _ = _load_canonical(Path(rec.flair_roi))
            roi_bin = (roi_raw > 0).astype(np.float32)
            roi_resampled = roi_bin.shape[:3] != base_shape or not np.allclose(roi_affine_src, base_affine)
            if roi_resampled:
                roi_bin = _resample_nearest(roi_bin, roi_affine_src, base_shape, base_affine)
            record_roi_processing(
                data_quality,
                source="provided",
                source_shape=roi_raw.shape,
                source_affine=roi_affine_src,
                target_shape=base_shape,
                target_affine=base_affine,
                resampled=roi_resampled,
            )
            roi_out_path = out_dir / "lesion_roi.nii.gz"
            _save_nifti(roi_bin, base_affine, base_header, roi_out_path)
            artifacts["lesion_roi"] = roi_out_path.name
            roi_source = "provided"
            stage_status["roi_mask"] = "ok"
        except Exception as exc:
            stage_status["roi_mask"] = f"failed: {exc}"
            notes.append(f"ROI mask stage failed: {exc}")
    elif use_candidate:
        try:
            union_path = out_dir / "candidate_roi" / "candidate_roi_union.nii.gz"
            if union_path.is_file():
                roi_raw, roi_affine_src, _ = _load_canonical(union_path)
                roi_bin = (roi_raw > 0).astype(np.float32)
                roi_resampled = roi_bin.shape[:3] != base_shape or not np.allclose(roi_affine_src, base_affine)
                if roi_resampled:
                    roi_bin = _resample_nearest(roi_bin, roi_affine_src, base_shape, base_affine)
                record_roi_processing(
                    data_quality,
                    source="candidate",
                    source_shape=roi_raw.shape,
                    source_affine=roi_affine_src,
                    target_shape=base_shape,
                    target_affine=base_affine,
                    resampled=roi_resampled,
                )
                roi_out_path = out_dir / "lesion_roi.nii.gz"
                _save_nifti(roi_bin, base_affine, base_header, roi_out_path)
                artifacts["lesion_roi"] = roi_out_path.name
                roi_source = "candidate"
                stage_status["roi_mask"] = "ok (from candidate)"
                notes.append("ROI mask derived from candidate ROI union.")
            else:
                stage_status["roi_mask"] = "skipped: candidate ROI union not available"
        except Exception as exc:
            stage_status["roi_mask"] = f"failed: {exc}"
            notes.append(f"ROI mask from candidate failed: {exc}")
    elif use_supervised:
        try:
            sup_path = out_dir / "supervised_roi" / "predicted_lesion_roi.nii.gz"
            if sup_path.is_file():
                roi_raw, roi_affine_src, _ = _load_canonical(sup_path)
                roi_bin = (roi_raw > 0).astype(np.float32)
                roi_resampled = roi_bin.shape[:3] != base_shape or not np.allclose(roi_affine_src, base_affine)
                if roi_resampled:
                    roi_bin = _resample_nearest(roi_bin, roi_affine_src, base_shape, base_affine)
                record_roi_processing(
                    data_quality,
                    source="supervised",
                    source_shape=roi_raw.shape,
                    source_affine=roi_affine_src,
                    target_shape=base_shape,
                    target_affine=base_affine,
                    resampled=roi_resampled,
                )
                roi_out_path = out_dir / "lesion_roi.nii.gz"
                _save_nifti(roi_bin, base_affine, base_header, roi_out_path)
                artifacts["lesion_roi"] = roi_out_path.name
                roi_source = "supervised"
                stage_status["roi_mask"] = "ok (from supervised)"
                notes.append("ROI mask derived from supervised ROI prediction.")
            else:
                stage_status["roi_mask"] = "skipped: supervised ROI not available"
        except Exception as exc:
            stage_status["roi_mask"] = f"failed: {exc}"
            notes.append(f"ROI mask from supervised failed: {exc}")
    else:
        stage_status["roi_mask"] = "skipped: no ROI mask available"
        notes.append("No ROI mask - control subject or ROI strategy=none.")

    # -------------------------------------------------------------------------
    # Stage 3: volumetric Gaussian splat PLY
    # -------------------------------------------------------------------------
    splat_reconstruction_audit_meta: dict = {}
    splat_for_reconstruction_audit: Optional[Path] = None
    if base_path is not None:
        try:
            splat_ply = out_dir / "patient_splat.ply"
            export_splat_ply(
                volume_path=base_path,
                output_ply=splat_ply,
                max_points=max_splat_points,
                rng_seed=rng_seed,
                subject=rec.subject_id,
                group=rec.group,
                sampling_mode=splat_sampling_mode,
            )
            artifacts["patient_splat_ply"] = splat_ply.name
            splat_for_reconstruction_audit = splat_ply
            meta_file = out_dir / "patient_splat_meta.json"
            if meta_file.exists():
                artifacts["patient_splat_meta"] = meta_file.name
            state_file = out_dir / "patient_splat_state.npz"
            if state_file.exists():
                artifacts["patient_splat_state"] = state_file.name

            # Dense presentation PLY (higher point count, full-volume by default)
            if presentation_splat_points > max_splat_points:
                dense_ply = out_dir / "patient_splat_dense.ply"
                export_splat_ply(
                    volume_path=base_path,
                    output_ply=dense_ply,
                    max_points=presentation_splat_points,
                    rng_seed=rng_seed,
                    subject=rec.subject_id,
                    group=rec.group,
                    sampling_mode=presentation_splat_sampling_mode,
                )
                artifacts["patient_splat_dense_ply"] = dense_ply.name
                splat_for_reconstruction_audit = dense_ply
                dense_meta_file = out_dir / "patient_splat_dense_meta.json"
                if dense_meta_file.exists():
                    artifacts["patient_splat_dense_meta"] = dense_meta_file.name
                dense_state_file = out_dir / "patient_splat_dense_state.npz"
                if dense_state_file.exists():
                    artifacts["patient_splat_dense_state"] = dense_state_file.name

            if roi_out_path is not None:
                lesion_highlight_ply = out_dir / "patient_splat_lesion_highlighted.ply"
                export_lesion_highlighted_splat_ply(
                    volume_path=base_path,
                    roi_path=roi_out_path,
                    output_ply=lesion_highlight_ply,
                    max_anatomy_points=presentation_splat_points,
                    max_lesion_points=60_000,
                    rng_seed=rng_seed,
                    subject=rec.subject_id,
                    group=rec.group,
                    sampling_mode=presentation_splat_sampling_mode,
                )
                artifacts["patient_splat_lesion_highlighted_ply"] = (
                    lesion_highlight_ply.name
                )
                artifacts["patient_splat_lesion_highlighted_meta"] = (
                    "patient_splat_lesion_highlighted_meta.json"
                )
                artifacts["patient_splat_lesion_highlighted_state"] = (
                    "patient_splat_lesion_highlighted_state.npz"
                )
                splat_for_reconstruction_audit = lesion_highlight_ply

            stage_status["splat_ply"] = "ok"
        except Exception as exc:
            stage_status["splat_ply"] = f"failed: {exc}"
            notes.append(f"Splat PLY stage failed: {exc}")

    # -------------------------------------------------------------------------
    # Stage 3b: MRI -> splat reconstruction audit
    # -------------------------------------------------------------------------
    if compute_splat_reconstruction_audit:
        if base_path is None:
            stage_status["splat_reconstruction_audit"] = "skipped: no base volume"
        elif splat_for_reconstruction_audit is None or not splat_for_reconstruction_audit.is_file():
            stage_status["splat_reconstruction_audit"] = "skipped: no splat PLY"
        else:
            try:
                from fcd_prototype.splat_reconstruction_audit import (
                    audit_splat_reconstruction as _audit_splat_reconstruction,
                )
                audit_path = out_dir / "splat_reconstruction_audit.json"
                recon_path = out_dir / "splat_reconstruction.nii.gz"
                audit = _audit_splat_reconstruction(
                    reference_volume=base_path,
                    splat_ply=splat_for_reconstruction_audit,
                    output_json=audit_path,
                    output_nifti=recon_path,
                    max_points=splat_reconstruction_audit_max_points,
                    smooth_sigma_vox=splat_reconstruction_audit_smooth_sigma,
                    kernel_mode=splat_reconstruction_audit_kernel_mode,  # type: ignore[arg-type]
                    gaussian_sigma_extent=splat_reconstruction_audit_gaussian_sigma_extent,
                    rng_seed=rng_seed,
                )
                artifacts["splat_reconstruction_audit"] = audit_path.name
                artifacts["splat_reconstruction"] = recon_path.name
                all_metrics = audit.get("all_foreground_metrics", {})
                represented_metrics = audit.get("represented_foreground_metrics", {})
                splat_reconstruction_audit_meta = {
                    "method": audit.get("method"),
                    "audited_artifact": splat_for_reconstruction_audit.name,
                    "total_points": audit.get("total_points"),
                    "audited_points": audit.get("audited_points"),
                    "subsampled": audit.get("subsampled"),
                    "kernel_mode": audit.get("kernel_mode"),
                    "gaussian_sigma_extent": audit.get("gaussian_sigma_extent"),
                    "foreground_coverage": audit.get("foreground_coverage"),
                    "all_foreground_mae": all_metrics.get("mae")
                    if isinstance(all_metrics, dict) else None,
                    "all_foreground_rmse": all_metrics.get("rmse")
                    if isinstance(all_metrics, dict) else None,
                    "all_foreground_psnr": all_metrics.get("psnr")
                    if isinstance(all_metrics, dict) else None,
                    "represented_foreground_mae": represented_metrics.get("mae")
                    if isinstance(represented_metrics, dict) else None,
                    "represented_foreground_psnr": represented_metrics.get("psnr")
                    if isinstance(represented_metrics, dict) else None,
                    "mip_mean_mae": audit.get("mip_metrics", {}).get("mean_mae")
                    if isinstance(audit.get("mip_metrics"), dict) else None,
                    "mri_to_3dgs_readiness_status": audit.get(
                        "mri_to_3dgs_readiness_status"
                    ),
                    "mri_to_3dgs_readiness_blocker_count": audit.get(
                        "mri_to_3dgs_readiness_blocker_count",
                        0,
                    ),
                    "mri_to_3dgs_readiness_warning_count": audit.get(
                        "mri_to_3dgs_readiness_warning_count",
                        0,
                    ),
                    "mri_to_3dgs_claim_allowed": audit.get(
                        "mri_to_3dgs_claim_allowed"
                    ),
                    "mri_to_3dgs_claim_class": audit.get(
                        "mri_to_3dgs_claim_class"
                    ),
                    "representation_readiness_gate": audit.get(
                        "representation_readiness_gate",
                        {},
                    ),
                }
                stage_status["splat_reconstruction_audit"] = "ok"
            except Exception as exc:
                stage_status["splat_reconstruction_audit"] = f"failed: {exc}"
                notes.append(f"Splat reconstruction audit failed (non-fatal): {exc}")
                limitations.append(
                    "Splat reconstruction audit unavailable - MRI-to-Gaussian "
                    "representation fidelity was not quantified for this run."
                )
    else:
        stage_status["splat_reconstruction_audit"] = "skipped: disabled"

    # -------------------------------------------------------------------------
    # Stage 4: lesion fuzzing
    # -------------------------------------------------------------------------
    fuzzed_path: Optional[Path] = None
    if base_path is not None:
        try:
            fuzz_out = out_dir / "fuzzed_patient_flair.nii.gz"
            fuzz_info = fuzz_lesion(
                patient_volume=base_path,
                roi_mask=roi_out_path,
                output_path=fuzz_out,
                rng_seed=rng_seed,
            )
            artifacts["fuzzed_patient_flair"] = fuzz_out.name
            artifacts["fuzzed_patient_base"] = fuzz_out.name
            fuzzed_path = fuzz_out
            stage_status["lesion_fuzzing"] = "ok" if fuzz_info["fuzzed"] else "ok (no roi, copy)"
            notes.extend(fuzz_info["notes"])
        except Exception as exc:
            stage_status["lesion_fuzzing"] = f"failed: {exc}"
            notes.append(f"Lesion fuzzing stage failed: {exc}")
            fuzzed_path = base_path

    # -------------------------------------------------------------------------
    # Stage 5: synthetic normal
    # -------------------------------------------------------------------------
    # Determine modality hint from whichever volume was used as base_path
    modality_hint: Optional[str] = None
    if base_path is not None:
        if rec.flair and rec.flair.is_file() and base_path == (out_dir / "patient_flair.nii.gz"):
            modality_hint = "FLAIR"
        elif rec.t2w and rec.t2w.is_file() and base_path == (out_dir / "patient_t2w.nii.gz"):
            modality_hint = "T2w"
        elif rec.t1w and rec.t1w.is_file() and base_path == (out_dir / "patient_t1w.nii.gz"):
            modality_hint = "T1w"
        elif "FLAIR" in modalities:
            modality_hint = "FLAIR"
        elif "T2w" in modalities:
            modality_hint = "T2w"
        elif "T1w" in modalities:
            modality_hint = "T1w"

    synth_path: Optional[Path] = None
    synth_info: dict = {}
    nvidia_preflight_meta: dict = {}
    synthetic_normal_audit_meta: dict = {}
    if base_path is not None:
        try:
            synth_out = out_dir / "synthetic_normal.nii.gz"
            if synthetic_mode == "nvidia":
                from fcd_prototype.nvidia_preflight import (
                    preflight_nvidia_mr_brain as _preflight_nvidia,
                )
                preflight_path = out_dir / "nvidia_preflight.json"
                preflight = _preflight_nvidia(
                    patient_volume=base_path,
                    roi_mask=roi_out_path,
                    modality_hint=modality_hint,
                )
                preflight_path.write_text(json.dumps(preflight, indent=2))
                artifacts["nvidia_preflight"] = preflight_path.name
                nvidia_preflight_meta = {
                    "ready_for_inference": preflight.get("ready_for_inference"),
                    "ready_for_patient_conditioned_counterfactual": preflight.get(
                        "ready_for_patient_conditioned_counterfactual"
                    ),
                    "root_available": preflight.get("root", {}).get("available")
                    if isinstance(preflight.get("root"), dict) else None,
                    "modality_code": preflight.get("modality_code"),
                    "generation_dim": preflight.get("generation_dim"),
                    "generation_spacing": preflight.get("generation_spacing"),
                    "generation_fov_mm": preflight.get("generation_fov_mm"),
                    "patient_conditioned_inpainting_supported": preflight.get(
                        "patient_conditioned_inpainting_supported"
                    ),
                    "paper_claim_readiness": preflight.get("paper_claim_readiness"),
                    "model_card_contract": preflight.get("model_card_contract"),
                    "local_adapter_contract": preflight.get("local_adapter_contract"),
                    "blocker_count": len(preflight.get("blockers", []))
                    if isinstance(preflight.get("blockers"), list) else None,
                    "warning_count": len(preflight.get("warnings", []))
                    if isinstance(preflight.get("warnings"), list) else None,
                }
                if not preflight.get("ready_for_inference"):
                    stage_status["nvidia_preflight"] = (
                        f"failed: {nvidia_preflight_meta['blocker_count']} blocker(s)"
                    )
                    notes.extend(str(x) for x in preflight.get("blockers", [])[:3])
                else:
                    warning_count = nvidia_preflight_meta["warning_count"] or 0
                    stage_status["nvidia_preflight"] = (
                        "ok" if warning_count == 0 else f"warning: {warning_count} issue(s)"
                    )

            synth_info = generate_synthetic_normal(
                patient_volume=base_path,
                roi_mask=roi_out_path,
                output_path=synth_out,
                mode=synthetic_mode,
                modality_hint=modality_hint,
                subject_id=rec.subject_id,
                rng_seed=rng_seed,
                control_prior_volumes=control_prior_volumes,
            )
            artifacts["synthetic_normal"] = synth_out.name
            # Merge any extra NVIDIA artifact names
            raw_prior = synth_info.get("nvidia_raw_prior_artifact")
            if raw_prior and (out_dir / raw_prior).is_file():
                artifacts["nvidia_raw_prior"] = raw_prior
            synth_path = synth_out
            stage_status["synthetic_normal"] = "ok"
            notes.extend(synth_info.get("notes", []))
            limitations.extend(synth_info.get("limitations", []))
        except NvidiaDependencyError as exc:
            stage_status["synthetic_normal"] = "failed: NVIDIA not available"
            notes.append(str(exc)[:200])
            limitations.append("NVIDIA synthetic normal unavailable - install and configure MAISI-v2.")
        except Exception as exc:
            stage_status["synthetic_normal"] = f"failed: {exc}"
            notes.append(f"Synthetic normal stage failed: {exc}")

    # -------------------------------------------------------------------------
    # Stage 6: synthetic-normal geometry and claim audit
    # -------------------------------------------------------------------------
    if base_path is not None and synth_path is not None:
        try:
            from fcd_prototype.synthetic_normal_audit import (
                audit_synthetic_normal as _audit_synth,
            )
            audit_path = out_dir / "synthetic_normal_audit.json"
            source_metadata = {
                "nvidia": {
                    k: v for k, v in synth_info.items()
                    if k.startswith("nvidia_")
                },
                "control_prior": {
                    k: v for k, v in synth_info.items()
                    if k.startswith("control_prior_")
                },
            }
            synth_audit = _audit_synth(
                patient_volume=base_path,
                synthetic_normal=synth_path,
                output_json=audit_path,
                roi_mask=roi_out_path,
                synthetic_normal_source=synth_info.get("synthetic_normal_source", "none"),
                source_metadata=source_metadata,
            )
            artifacts["synthetic_normal_audit"] = audit_path.name
            outside = synth_audit.get("outside_roi", {})
            inside = synth_audit.get("inside_roi", {})
            subtraction_gate = synth_audit.get("subtraction_readiness_gate", {})
            if not isinstance(subtraction_gate, dict):
                subtraction_gate = {}
            synthetic_normal_audit_meta = {
                "counterfactual_class": synth_audit.get("counterfactual_class"),
                "patient_conditioned_inpainting": synth_audit.get("patient_conditioned_inpainting"),
                "claim_boundary": synth_audit.get("claim_boundary"),
                "subtraction_interpretation": synth_audit.get("subtraction_interpretation"),
                "subtraction_readiness_status": subtraction_gate.get("status"),
                "subtraction_readiness_blocker_count": subtraction_gate.get("blocker_count"),
                "subtraction_readiness_warning_count": subtraction_gate.get("warning_count"),
                "gaussian_subtraction_claim_allowed": subtraction_gate.get(
                    "gaussian_subtraction_claim_allowed"
                ),
                "patient_conditioned_counterfactual_claim_allowed": subtraction_gate.get(
                    "patient_conditioned_counterfactual_claim_allowed"
                ),
                "synthetic_normal_claim_class": subtraction_gate.get("claim_class"),
                "nvidia_steps": subtraction_gate.get("nvidia_steps"),
                "nvidia_min_inference_steps": subtraction_gate.get(
                    "nvidia_min_inference_steps"
                ),
                "nvidia_generated_shape_matches_config": subtraction_gate.get(
                    "nvidia_generated_shape_matches_config"
                ),
                "same_grid_before_resample": synth_audit.get("same_grid_before_resample"),
                "synthetic_resampled_to_patient_grid": synth_audit.get(
                    "synthetic_resampled_to_patient_grid"
                ),
                "outside_roi_mae": outside.get("mae") if isinstance(outside, dict) else None,
                "outside_roi_similarity_1_minus_mae": outside.get(
                    "similarity_1_minus_mae"
                ) if isinstance(outside, dict) else None,
                "inside_roi_mae": inside.get("mae") if isinstance(inside, dict) else None,
                "inside_to_outside_mae_ratio": synth_audit.get(
                    "inside_to_outside_mae_ratio"
                ),
            }
            stage_status["synthetic_normal_audit"] = "ok"
        except Exception as exc:
            stage_status["synthetic_normal_audit"] = f"failed: {exc}"
            notes.append(f"Synthetic normal audit failed (non-fatal): {exc}")
            limitations.append(
                "Synthetic normal audit unavailable - residual interpretation has weaker "
                "geometry and claim-boundary evidence."
            )
    else:
        stage_status["synthetic_normal_audit"] = (
            "skipped: missing patient or synthetic normal volume"
        )

    # -------------------------------------------------------------------------
    # Stage 7: residual heatmap
    # -------------------------------------------------------------------------
    if fuzzed_path is not None and synth_path is not None:
        try:
            residual_out = out_dir / "residual_heatmap.nii.gz"
            residual_stats = compute_residual(
                patient_volume=fuzzed_path,
                synthetic_normal=synth_path,
                output_path=residual_out,
                roi_mask=roi_out_path,
            )
            artifacts["residual_heatmap"] = residual_out.name
            stage_status["residual"] = "ok"
            notes.extend(residual_stats.get("notes", []))
            if "roi_concentration" in residual_stats:
                artifacts["residual_roi_concentration"] = str(
                    round(residual_stats["roi_concentration"], 4)
                )
        except Exception as exc:
            stage_status["residual"] = f"failed: {exc}"
            notes.append(f"Residual stage failed: {exc}")
    else:
        stage_status["residual"] = "skipped: missing fuzzed or synthetic volume"

    # -------------------------------------------------------------------------
    # Stage 8: paired-support Gaussian splat residual (automatic)
    # -------------------------------------------------------------------------
    paired_gaussian_residual_meta: dict = {}
    paired_gr_dir = out_dir / "paired_gaussian_residual"

    if compute_paired_gaussian_residual:
        if base_path is None:
            stage_status["paired_gaussian_residual"] = "skipped: no patient base volume"
        elif synth_path is None:
            stage_status["paired_gaussian_residual"] = "skipped: no synthetic normal volume"
        else:
            try:
                from fcd_prototype.paired_gaussian_residual import (
                    build_paired_gaussian_residual as _build_paired_gr,
                )
                paired_support_max_points = (
                    paired_gaussian_max_points
                    if paired_gaussian_max_points is not None
                    else max(max_splat_points, presentation_splat_points)
                )
                paired_metrics = _build_paired_gr(
                    patient_volume=base_path,
                    normal_volume=synth_path,
                    output_dir=paired_gr_dir,
                    roi=roi_out_path,
                    max_points=paired_support_max_points,
                    sampling_mode=presentation_splat_sampling_mode,
                    support_basis=paired_gaussian_support_basis,
                    rng_seed=rng_seed,
                    heatmap_percentile=paired_gaussian_heatmap_percentile,
                    smooth_sigma=paired_gaussian_smooth_sigma,
                    heatmap_method=paired_gaussian_heatmap_method,
                    heatmap_gaussian_sigma_extent=paired_gaussian_heatmap_sigma_extent,
                    heatmap_gaussian_exact_max_points=(
                        paired_gaussian_heatmap_exact_max_points
                    ),
                    subject=rec.subject_id,
                    group=rec.group or "",
                )
                for key, rel in paired_metrics.get("artifacts", {}).items():
                    candidate = paired_gr_dir / str(rel)
                    if candidate.exists():
                        artifacts[key] = _artifact_rel(candidate, out_dir)
                metrics_path = paired_gr_dir / "paired_gaussian_residual_metrics.json"
                if metrics_path.exists():
                    artifacts["paired_gaussian_residual_metrics"] = _artifact_rel(
                        metrics_path, out_dir
                    )
                paired_gaussian_residual_meta = {
                    k: paired_metrics[k]
                    for k in (
                        "method", "same_gaussian_support", "n_gaussians",
                        "gaussian_state_schema",
                        "max_points", "foreground_voxels_total",
                        "sampling_mode", "support_basis", "residual_mean",
                        "support_basis_mode",
                        "residual_p95", "residual_p99", "residual_norm_p99",
                        "residual_normalization_value",
                        "residual_nonzero_fraction",
                        "heatmap_method", "heatmap_backend",
                        "heatmap_active_gaussians",
                        "heatmap_gaussian_sigma_extent",
                        "heatmap_exact_max_points",
                        "heatmap_effective_sigma_vox",
                        "normal_resampled_to_patient_grid",
                        "gaussian_orientation", "gaussian_scale_model",
                        "density_scale_factor", "density_voxels_per_splat",
                        "foreground_mask_model", "foreground_threshold_value",
                        "raw_foreground_voxels_total",
                        "cleaned_foreground_voxels_total",
                        "volume_percentile_threshold_floor",
                    )
                    if k in paired_metrics
                }
                if paired_metrics.get("roi_metrics_valid"):
                    paired_gaussian_residual_meta["roi_to_non_roi_ratio"] = paired_metrics.get(
                        "roi_to_non_roi_ratio"
                    )
                    paired_gaussian_residual_meta["roi_residual_concentration"] = paired_metrics.get(
                        "roi_residual_concentration"
                    )
                stage_status["paired_gaussian_residual"] = "ok"
                if paired_gaussian_max_points is None and paired_support_max_points > max_splat_points:
                    notes.append(
                        "Paired-support Gaussian residual used the presentation "
                        f"splat density by default ({paired_support_max_points} max points)."
                    )
            except Exception as exc:
                stage_status["paired_gaussian_residual"] = f"failed: {exc}"
                notes.append(f"Paired Gaussian residual stage failed (non-fatal): {exc}")
                limitations.append(
                    "Paired-support Gaussian residual unavailable - viewer can still use "
                    "direct splat and NIfTI residual artifacts."
                )
    else:
        stage_status["paired_gaussian_residual"] = (
            "skipped: paired Gaussian residual disabled"
        )

    # -------------------------------------------------------------------------
    # Stage 9: patient-support healthy-control Gaussian prior residual
    # -------------------------------------------------------------------------
    control_gaussian_prior_meta: dict = {}
    control_gp_dir = out_dir / "control_gaussian_prior"

    if compute_control_gaussian_prior_residual:
        control_paths = [
            Path(p) for p in (control_prior_volumes or [])
            if Path(p).is_file()
        ]
        if base_path is None:
            stage_status["control_gaussian_prior"] = (
                "skipped: no patient base volume"
            )
        elif len(control_paths) < int(control_gaussian_prior_min_controls):
            stage_status["control_gaussian_prior"] = (
                "skipped: fewer than "
                f"{int(control_gaussian_prior_min_controls)} healthy controls"
            )
        else:
            try:
                from fcd_prototype.control_gaussian_prior import (
                    build_control_gaussian_prior_residual as _build_control_gp,
                )

                control_prior_support_max_points = (
                    control_gaussian_prior_max_points
                    if control_gaussian_prior_max_points is not None
                    else max(max_splat_points, presentation_splat_points)
                )
                roi_for_control_prior = (
                    manual_roi_validation_path
                    if manual_roi_validation_path is not None
                    else roi_out_path
                )
                control_metrics = _build_control_gp(
                    patient_volume=base_path,
                    control_volumes=control_paths,
                    output_dir=control_gp_dir,
                    roi=roi_for_control_prior,
                    min_controls=control_gaussian_prior_min_controls,
                    max_points=control_prior_support_max_points,
                    sampling_mode=presentation_splat_sampling_mode,
                    rng_seed=rng_seed,
                    heatmap_percentile=control_gaussian_prior_heatmap_percentile,
                    smooth_sigma=control_gaussian_prior_smooth_sigma,
                    heatmap_method=control_gaussian_prior_heatmap_method,
                    heatmap_gaussian_exact_max_points=(
                        control_gaussian_prior_heatmap_exact_max_points
                    ),
                    std_floor_fraction=control_gaussian_prior_std_floor_fraction,
                    subject=rec.subject_id,
                    group=rec.group or "",
                )
                for key, rel in control_metrics.get("artifacts", {}).items():
                    candidate = control_gp_dir / str(rel)
                    if candidate.exists():
                        artifacts[key] = _artifact_rel(candidate, out_dir)
                metrics_path = control_gp_dir / "control_gaussian_prior_metrics.json"
                if metrics_path.exists():
                    artifacts["control_gaussian_prior_metrics"] = _artifact_rel(
                        metrics_path, out_dir
                    )
                control_gaussian_prior_meta = {
                    k: control_metrics[k]
                    for k in _CONTROL_GAUSSIAN_PRIOR_COMPACT_KEYS
                    if k in control_metrics
                }
                control_gaussian_prior_meta["max_points"] = int(
                    control_prior_support_max_points
                )
                control_gaussian_prior_meta["control_prior_subjects"] = [
                    _extract_subject_id(str(path)) for path in control_paths
                ]
                control_gaussian_prior_meta["roi_source_for_exclusion"] = (
                    manual_roi_validation_source
                    if manual_roi_validation_path is not None
                    else roi_source
                )
                manual_roi_reused_as_fuzzing_roi = (
                    manual_roi_validation_path is not None
                    and manual_roi_validation_source == "provided"
                    and roi_source == "provided"
                )
                control_gaussian_prior_meta["roi_metrics_independent_validation"] = (
                    manual_roi_validation_path is not None
                    and not manual_roi_reused_as_fuzzing_roi
                )
                control_gaussian_prior_meta["roi_metrics_independence_reason"] = (
                    "manual_roi_reused_as_fuzzing_roi"
                    if manual_roi_reused_as_fuzzing_roi
                    else (
                        "manual_roi_not_used_for_fuzzing"
                        if manual_roi_validation_path is not None
                        else "no_manual_roi_available"
                    )
                )
                if (
                    roi_for_control_prior is not None
                    and manual_roi_validation_path is None
                    and roi_source != "none"
                ):
                    limitations.append(
                        "Control-prior Gaussian ROI concentration used the pipeline ROI "
                        "for exclusion/measurement and is not independent validation."
                    )
                stage_status["control_gaussian_prior"] = "ok"
                notes.append(
                    "Built patient-support healthy-control Gaussian prior and z-residual "
                    "splat for direct same-support subtraction review."
                )
            except Exception as exc:
                stage_status["control_gaussian_prior"] = f"failed: {exc}"
                notes.append(
                    f"Control Gaussian prior residual stage failed (non-fatal): {exc}"
                )
                limitations.append(
                    "Patient-support healthy-control Gaussian prior residual unavailable."
                )
    else:
        stage_status["control_gaussian_prior"] = (
            "skipped: control Gaussian prior residual disabled"
        )

    # -------------------------------------------------------------------------
    # Stage 10: residual-derived lesion candidate extraction
    # -------------------------------------------------------------------------
    residual_candidate_meta: dict = {}
    residual_candidate_dir = out_dir / "residual_candidates"
    if compute_residual_candidates:
        residual_candidate_source: Optional[Path] = None
        if (control_gp_dir / "control_prior_z_residual_heatmap.nii.gz").is_file():
            residual_candidate_source = (
                control_gp_dir / "control_prior_z_residual_heatmap.nii.gz"
            )
        elif (paired_gr_dir / "paired_gaussian_residual_heatmap.nii.gz").is_file():
            residual_candidate_source = paired_gr_dir / "paired_gaussian_residual_heatmap.nii.gz"
        elif (out_dir / "residual_heatmap.nii.gz").is_file():
            residual_candidate_source = out_dir / "residual_heatmap.nii.gz"

        if residual_candidate_source is None:
            stage_status["residual_candidates"] = "skipped: no residual heatmap available"
        elif base_path is None:
            stage_status["residual_candidates"] = "skipped: no reference patient volume"
        else:
            try:
                from fcd_prototype.residual_candidates import (
                    extract_residual_candidates as _extract_residual_candidates,
                )
                candidate_result = _extract_residual_candidates(
                    residual_heatmap=residual_candidate_source,
                    output_dir=residual_candidate_dir,
                    reference_volume=base_path,
                    manual_roi=manual_roi_validation_path,
                    max_candidates=residual_candidate_max_candidates,
                    percentile=residual_candidate_percentile,
                    min_threshold=residual_candidate_min_threshold,
                    min_component_voxels=residual_candidate_min_component_voxels,
                    dilate_mm=residual_candidate_dilate_mm,
                )
                for key, rel in candidate_result.get("artifacts", {}).items():
                    candidate_path = residual_candidate_dir / str(rel)
                    if candidate_path.exists():
                        artifacts[key] = _artifact_rel(candidate_path, out_dir)
                manual_overlap = candidate_result.get("manual_overlap", {})
                manual_roi_available = (
                    manual_overlap.get("manual_roi_available")
                    if isinstance(manual_overlap, dict) else False
                )
                (
                    manual_overlap_independent,
                    manual_overlap_independence_reason,
                ) = _manual_overlap_independence_status(
                    selected_roi_source=roi_source,
                    manual_roi_source=manual_roi_validation_source,
                    manual_roi_available=manual_roi_available,
                )
                residual_candidate_meta = {
                    "method": candidate_result.get("method"),
                    "source_heatmap": _artifact_rel(residual_candidate_source, out_dir),
                    "selected_count": candidate_result.get("selected_count"),
                    "threshold_value": candidate_result.get("threshold_value"),
                    "percentile": candidate_result.get("percentile"),
                    "candidate_voxels": candidate_result.get("candidate_voxels"),
                    "candidate_volume_ml": candidate_result.get("candidate_volume_ml"),
                    "candidate_foreground_fraction": candidate_result.get(
                        "candidate_foreground_fraction"
                    ),
                    "dilate_mm": candidate_result.get("dilate_mm"),
                    "dilation_mode": candidate_result.get("dilation_mode"),
                    "selected_roi_source": roi_source,
                    "manual_roi_source": manual_roi_validation_source,
                    "manual_roi_artifact": (
                        _artifact_rel(manual_roi_validation_path, out_dir)
                        if manual_roi_validation_path is not None else None
                    ),
                    "manual_roi_available": manual_roi_available,
                    "manual_overlap_independent_from_fuzzing_roi": (
                        manual_overlap_independent
                    ),
                    "manual_overlap_independence_reason": (
                        manual_overlap_independence_reason
                    ),
                    "manual_coverage": manual_overlap.get("manual_coverage")
                    if isinstance(manual_overlap, dict) else None,
                    "candidate_precision": manual_overlap.get("candidate_precision")
                    if isinstance(manual_overlap, dict) else None,
                    "dice": manual_overlap.get("dice")
                    if isinstance(manual_overlap, dict) else None,
                    "jaccard": manual_overlap.get("jaccard")
                    if isinstance(manual_overlap, dict) else None,
                }
                stage_status["residual_candidates"] = (
                    f"ok ({candidate_result.get('selected_count', 0)} candidates)"
                )
                source_kind = (
                    "control-prior Gaussian z-residual"
                    if residual_candidate_source.parent == control_gp_dir
                    else (
                        "paired Gaussian residual"
                        if residual_candidate_source.parent == paired_gr_dir
                        else "legacy NIfTI residual"
                    )
                )
                notes.append(
                    "Residual-derived lesion candidates extracted from the "
                    f"{source_kind} heatmap."
                )
            except Exception as exc:
                stage_status["residual_candidates"] = f"failed: {exc}"
                notes.append(f"Residual candidate extraction failed (non-fatal): {exc}")
                limitations.append(
                    "Residual-derived lesion candidate masks unavailable - viewer can still "
                    "show residual heatmaps."
                )
    else:
        stage_status["residual_candidates"] = "skipped: residual candidate extraction disabled"

    # -------------------------------------------------------------------------
    # Stage 10: optimized Gaussian field (optional)
    # -------------------------------------------------------------------------
    gaussian_field_meta: dict = {}
    optimized_gaussian_field_audit_meta: dict = {}
    if train_gaussian_field and base_path is not None:
        try:
            from fcd_prototype.gaussian_field import train_gaussian_field as _train_gf
            gf_metrics = _train_gf(
                volume_path=base_path,
                output_dir=out_dir,
                roi_path=roi_out_path,
                num_gaussians=gaussian_num_gaussians,
                steps=gaussian_steps,
                image_size=gaussian_image_size,
                views=gaussian_views,
                device=gaussian_device,
                seed=rng_seed,
                learning_rate=gaussian_learning_rate,
                volume_loss_weight=gaussian_volume_loss_weight,
                volume_sample_count=gaussian_volume_sample_count,
            )
            # The function writes into out_dir; record canonical artifact names
            ply_gf = out_dir / "optimized_gaussian_splat.ply"
            pt_gf = out_dir / "gaussian_field.pt"
            metrics_gf = out_dir / "training_metrics.json"
            preview_gf = out_dir / "reconstruction_preview.png"
            mask_gf = out_dir / "lesion_gaussian_mask.npy"
            if ply_gf.exists():
                artifacts["optimized_gaussian_splat_ply"] = ply_gf.name
            if pt_gf.exists():
                artifacts["gaussian_field_checkpoint"] = pt_gf.name
            if metrics_gf.exists():
                artifacts["gaussian_field_metrics"] = metrics_gf.name
            if preview_gf.exists():
                artifacts["gaussian_field_preview"] = preview_gf.name
            if mask_gf.exists():
                artifacts["lesion_gaussian_mask"] = mask_gf.name
            gaussian_field_meta = {k: gf_metrics[k] for k in _GAUSSIAN_FIELD_COMPACT_KEYS if k in gf_metrics}
            stage_status["gaussian_field"] = "ok"
        except Exception as exc:
            stage_status["gaussian_field"] = f"failed: {exc}"
            notes.append(f"Gaussian field stage failed (non-fatal): {exc}")
            limitations.append(
                "Optimized Gaussian field not available for this subject - "
                "viewer will fall back to direct splat."
            )
    elif not train_gaussian_field:
        stage_status["gaussian_field"] = "skipped: --train-gaussian-field not set"
    else:
        stage_status["gaussian_field"] = "skipped: no base volume"

    # -------------------------------------------------------------------------
    # Stage 10b: held-out optimized Gaussian field fidelity audit
    # -------------------------------------------------------------------------
    if compute_optimized_gaussian_field_audit:
        gf_checkpoint = out_dir / "gaussian_field.pt"
        if not train_gaussian_field:
            stage_status["optimized_gaussian_field_audit"] = (
                "skipped: --train-gaussian-field not set"
            )
        elif base_path is None:
            stage_status["optimized_gaussian_field_audit"] = "skipped: no base volume"
        elif not gf_checkpoint.is_file():
            stage_status["optimized_gaussian_field_audit"] = (
                "skipped: gaussian_field.pt not found"
            )
        else:
            try:
                from fcd_prototype.gaussian_field_audit import (
                    audit_gaussian_field_fidelity as _audit_gf_fidelity,
                )
                audit_path = out_dir / "optimized_gaussian_field_audit.json"
                audit = _audit_gf_fidelity(
                    gaussian_field_path=gf_checkpoint,
                    reference_volume_path=base_path,
                    output_json=audit_path,
                    sample_count=optimized_gaussian_field_audit_sample_count,
                    seed=rng_seed + 1009,
                    device=optimized_gaussian_field_audit_device,
                )
                artifacts["optimized_gaussian_field_audit"] = audit_path.name
                optimized_gaussian_field_audit_meta = {
                    k: audit[k]
                    for k in _GAUSSIAN_FIELD_AUDIT_COMPACT_KEYS
                    if k in audit
                }
                gate = audit.get("readiness_gate", {})
                stage_status["optimized_gaussian_field_audit"] = (
                    f"ok: {gate.get('status', 'unknown')}"
                )
                if not gate.get("optimized_mri_to_3dgs_claim_allowed"):
                    limitations.append(
                        "Optimized Gaussian field failed held-out MRI fidelity audit; "
                        "paper MRI-native 3DGS claim is not allowed for this subject."
                    )
            except Exception as exc:
                stage_status["optimized_gaussian_field_audit"] = f"failed: {exc}"
                notes.append(
                    f"Optimized Gaussian field fidelity audit failed (non-fatal): {exc}"
                )
                limitations.append(
                    "Optimized Gaussian field fidelity is unproven for this subject."
                )
    else:
        stage_status["optimized_gaussian_field_audit"] = (
            "skipped: --skip-optimized-gaussian-field-audit"
        )

    # -------------------------------------------------------------------------
    # Stage 11: synthetic normal Gaussian field (optional)
    # -------------------------------------------------------------------------
    synth_gaussian_field_meta: dict = {}
    synth_normal_gf_dir = out_dir / "synthetic_normal_gaussian"

    if train_synthetic_gaussian_field and synth_path is not None:
        try:
            from fcd_prototype.gaussian_field import (
                project_volume_to_gaussian_support as _project_gf_support,
                train_gaussian_field as _train_gf,
            )
            patient_support = out_dir / "gaussian_field.pt"
            if patient_support.is_file():
                sgf_metrics = _project_gf_support(
                    support_field=patient_support,
                    volume_path=synth_path,
                    output_dir=synth_normal_gf_dir,
                )
                notes.append(
                    "Synthetic normal Gaussian field projected onto the patient "
                    "Gaussian support for index-wise optimized splat subtraction."
                )
            else:
                sgf_metrics = _train_gf(
                    volume_path=synth_path,
                    output_dir=synth_normal_gf_dir,
                    roi_path=None,          # normal reference; no ROI
                    num_gaussians=gaussian_num_gaussians,
                    steps=gaussian_steps,
                    image_size=gaussian_image_size,
                    views=gaussian_views,
                    device=gaussian_device,
                    seed=rng_seed,
                    learning_rate=gaussian_learning_rate,
                    volume_loss_weight=gaussian_volume_loss_weight,
                    volume_sample_count=gaussian_volume_sample_count,
                )
                limitations.append(
                    "Synthetic normal Gaussian field was trained independently "
                    "because the patient Gaussian support checkpoint was unavailable; "
                    "optimized residual cannot claim index-wise splat subtraction."
                )
            sgf_ply     = synth_normal_gf_dir / "optimized_gaussian_splat.ply"
            sgf_pt      = synth_normal_gf_dir / "gaussian_field.pt"
            sgf_metrics_f = synth_normal_gf_dir / "training_metrics.json"
            sgf_preview = synth_normal_gf_dir / "reconstruction_preview.png"
            if sgf_ply.exists():
                artifacts["synthetic_normal_gaussian_splat_ply"] = _artifact_rel(
                    sgf_ply, out_dir)
            if sgf_pt.exists():
                artifacts["synthetic_normal_gaussian_checkpoint"] = _artifact_rel(
                    sgf_pt, out_dir)
            if sgf_metrics_f.exists():
                artifacts["synthetic_normal_gaussian_metrics"] = _artifact_rel(
                    sgf_metrics_f, out_dir)
            if sgf_preview.exists():
                artifacts["synthetic_normal_gaussian_preview"] = _artifact_rel(
                    sgf_preview, out_dir)
            synth_gaussian_field_meta = {
                k: sgf_metrics[k] for k in _GAUSSIAN_FIELD_COMPACT_KEYS
                if k in sgf_metrics
            }
            stage_status["synthetic_gaussian_field"] = "ok"
        except Exception as exc:
            stage_status["synthetic_gaussian_field"] = f"failed: {exc}"
            notes.append(f"Synthetic normal Gaussian field stage failed (non-fatal): {exc}")
            limitations.append(
                "Synthetic normal Gaussian field not available - "
                "Gaussian residual stage will be skipped."
            )
    elif not train_synthetic_gaussian_field:
        stage_status["synthetic_gaussian_field"] = (
            "skipped: --train-synthetic-gaussian-field not set"
        )
    else:
        stage_status["synthetic_gaussian_field"] = "skipped: no synthetic normal volume"

    # -------------------------------------------------------------------------
    # Stage 12: optimized Gaussian-space residual comparison (optional)
    # -------------------------------------------------------------------------
    gaussian_residual_meta: dict = {}
    gr_dir = out_dir / "gaussian_residual"

    if compute_gaussian_residual:
        patient_pt = out_dir / "gaussian_field.pt"
        normal_pt  = synth_normal_gf_dir / "gaussian_field.pt"

        # Guard: both checkpoints must exist and base_path must be available
        if not patient_pt.is_file():
            stage_status["gaussian_residual"] = (
                "skipped: patient gaussian_field.pt not found "
                "(run --train-gaussian-field first)"
            )
        elif not normal_pt.is_file():
            stage_status["gaussian_residual"] = (
                "skipped: synthetic_normal_gaussian/gaussian_field.pt not found "
                "(run --train-synthetic-gaussian-field first)"
            )
        elif base_path is None:
            stage_status["gaussian_residual"] = "skipped: no base volume (reference)"
        else:
            try:
                from fcd_prototype.gaussian_residual import compare_gaussian_fields as _compare_gf
                gr_metrics = _compare_gf(
                    patient_field=patient_pt,
                    normal_field=normal_pt,
                    reference_volume=base_path,
                    output_dir=gr_dir,
                    roi=roi_out_path,
                    k_neighbors=gaussian_residual_k_neighbors,
                    max_points=gaussian_residual_max_points,
                    heatmap_percentile=gaussian_residual_heatmap_percentile,
                    heatmap_method=gaussian_residual_heatmap_method,
                    field_chunk_size=gaussian_residual_field_chunk_size,
                )
                gr_splat    = gr_dir / "gaussian_residual_splat.ply"
                gr_heatmap  = gr_dir / "gaussian_residual_heatmap.nii.gz"
                gr_metrics_f = gr_dir / "gaussian_residual_metrics.json"
                gr_scores   = gr_dir / "gaussian_residual_scores.npy"
                if gr_splat.exists():
                    artifacts["gaussian_residual_splat"] = _artifact_rel(
                        gr_splat, out_dir)
                if gr_heatmap.exists():
                    artifacts["gaussian_residual_heatmap"] = _artifact_rel(
                        gr_heatmap, out_dir)
                if gr_metrics_f.exists():
                    artifacts["gaussian_residual_metrics"] = _artifact_rel(
                        gr_metrics_f, out_dir)
                if gr_scores.exists():
                    artifacts["gaussian_residual_scores"] = _artifact_rel(
                        gr_scores, out_dir)
                gaussian_residual_meta = {
                    k: gr_metrics[k]
                    for k in (
                        "k_neighbors", "n_patient", "n_normal",
                        "heatmap_method", "residual_score_method",
                        "same_gaussian_support",
                        "support_fingerprint_schema",
                        "support_fingerprint_match",
                        "patient_support_fingerprint_sha256",
                        "normal_support_fingerprint_sha256",
                        "gaussian_subtraction_claim_allowed",
                        "means_norm_max_abs_diff",
                        "log_scales_max_abs_diff",
                        "quat_wxyz_max_abs_diff",
                        "residual_mean", "residual_p95", "residual_p99",
                        "field_residual_mean", "field_residual_p95",
                        "field_residual_p99", "field_chunk_size",
                    )
                    if k in gr_metrics
                }
                # Attach ROI ratio when available and valid
                if gr_metrics.get("roi_metrics_valid"):
                    gaussian_residual_meta["roi_to_non_roi_ratio"] = gr_metrics.get(
                        "roi_to_non_roi_ratio"
                    )
                stage_status["gaussian_residual"] = "ok"
            except Exception as exc:
                stage_status["gaussian_residual"] = f"failed: {exc}"
                notes.append(f"Gaussian residual stage failed (non-fatal): {exc}")
    else:
        stage_status["gaussian_residual"] = (
            "skipped: --compute-gaussian-residual not set"
        )

    # -------------------------------------------------------------------------
    # Manifest
    # -------------------------------------------------------------------------
    data_quality["warning_count"] = len(data_quality.get("warnings", []))
    data_quality["status"] = "warning" if data_quality["warning_count"] else "ok"
    data_quality["upload_readiness"] = build_upload_readiness(data_quality)
    stage_status["data_quality"] = _data_quality_stage_status(data_quality)
    for warning in data_quality.get("warnings", []):
        note = f"Data quality warning: {warning}"
        if note not in notes:
            notes.append(note)

    nvidia_meta: dict = {}
    for key in (
        "nvidia_modality_code", "nvidia_modality_hint", "nvidia_generation_dim",
        "nvidia_generation_spacing", "nvidia_generation_fov_mm", "nvidia_patient_fov_mm",
        "nvidia_generated_shape", "nvidia_generated_shape_matches_config",
        "nvidia_generated_effective_spacing", "nvidia_prior_alignment",
        "nvidia_seed", "nvidia_steps", "nvidia_cfg_scale",
        "nvidia_upstream_commit", "nvidia_command_summary", "nvidia_raw_prior_artifact",
        "nvidia_skip_reason",
    ):
        if key in synth_info:
            nvidia_meta[key] = synth_info[key]

    control_prior_meta: dict = {
        key: value for key, value in synth_info.items()
        if key.startswith("control_prior_")
    }

    paper_3dgs_mode_meta: dict = {}
    if pipeline_preset == "paper_3dgs":
        paper_3dgs_mode_meta = {
            "enabled": True,
            "train_gaussian_field": bool(train_gaussian_field),
            "train_synthetic_gaussian_field": bool(train_synthetic_gaussian_field),
            "compute_paired_gaussian_residual": bool(compute_paired_gaussian_residual),
            "compute_control_gaussian_prior_residual": bool(
                compute_control_gaussian_prior_residual
            ),
            "compute_gaussian_residual": bool(compute_gaussian_residual),
            "gaussian_volume_loss_weight": float(gaussian_volume_loss_weight),
            "gaussian_volume_sample_count": int(gaussian_volume_sample_count),
            "paired_gaussian_support_basis": paired_gaussian_support_basis,
            "synthetic_gaussian_support_strategy": (
                "project_synthetic_normal_to_patient_gaussian_support"
            ),
        }

    synthetic_normal_meta: dict = {}
    for key in (
        "mode", "synthetic_normal_source", "roi_voxel_count",
        "mirrored_voxel_fraction", "fallback_voxel_fraction",
        "mirrored_voxel_count", "fallback_voxel_count",
        "feather_sigma", "dilation_iterations",
        "midline_voxel_index", "midline_world_mm",
        "midline_estimation", "contralateral_axis",
        "fallback_method", "fallback_smooth_sigma",
        "contralateral_skip_reason",
    ):
        if key in synth_info:
            synthetic_normal_meta[key] = synth_info[key]

    _base_src = rec.flair or rec.t2w or rec.t1w
    _base_mod = _preferred_modality_for_record(rec)
    _sidecar = load_sidecar(_base_src) if _base_src and _base_src.is_file() else {}
    _roi_src = Path(rec.flair_roi) if rec.flair_roi else None
    viewer_meta = build_viewer_metadata(
        subject_record=rec,
        base_nifti_path=_base_src,
        base_modality=_base_mod,
        roi_nifti_path=_roi_src,
        sidecar_fields=_sidecar,
    )

    manifest = {
        "subject": rec.subject_id,
        "group": rec.group or "unknown",
        "modalities": modalities,
        "roi_available": has_roi,
        "roi_source": roi_source,
        "candidate_roi_proposal_mode": candidate_roi_proposal_mode,
        "pipeline_preset": pipeline_preset,
        "paper_3dgs_mode_meta": paper_3dgs_mode_meta,
        "source_files": {
            "flair": str(rec.flair) if rec.flair else None,
            "t2w": str(rec.t2w) if rec.t2w else None,
            "t1w": str(rec.t1w) if rec.t1w else None,
            "flair_roi": str(rec.flair_roi) if rec.flair_roi else None,
        },
        "artifacts": artifacts,
        "stage_status": stage_status,
        "synthetic_normal_source": synth_info.get("synthetic_normal_source", "none"),
        "synthetic_normal_is_diagnostic": synth_info.get("synthetic_normal_is_diagnostic", False),
        "nvidia_meta": nvidia_meta,
        "nvidia_preflight_meta": nvidia_preflight_meta,
        "control_prior_meta": control_prior_meta,
        "control_gaussian_prior_meta": control_gaussian_prior_meta,
        "synthetic_normal_meta": synthetic_normal_meta,
        "synthetic_normal_audit_meta": synthetic_normal_audit_meta,
        "splat_reconstruction_audit_meta": splat_reconstruction_audit_meta,
        "supervised_roi_meta": supervised_roi_meta,
        "supervised_roi_operating_point_meta": supervised_roi_operating_point_meta,
        "supervised_roi_resolved_params": supervised_roi_resolved_params,
        "residual_candidate_meta": residual_candidate_meta,
        "data_quality": data_quality,
        "paired_gaussian_residual_meta": paired_gaussian_residual_meta,
        "gaussian_field_meta": gaussian_field_meta,
        "optimized_gaussian_field_audit_meta": optimized_gaussian_field_audit_meta,
        "synth_gaussian_field_meta": synth_gaussian_field_meta,
        "gaussian_residual_meta": gaussian_residual_meta,
        "generated": datetime.now(timezone.utc).isoformat(),
        "viewer_meta": viewer_meta,
        "notes": notes,
        "limitations": limitations,
    }

    manifest["product_readiness"] = build_product_readiness(manifest)
    manifest["paper_method_feasibility"] = build_paper_method_feasibility(manifest)

    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  {rec.subject_id}: manifest -> {manifest_path}")
    return manifest


def build_end_to_end(
    dataset_dir: Path,
    output_root: Path,
    subjects: Optional[list[str]] = None,
    synthetic_mode: SyntheticMode = "placeholder",
    roi_strategy: RoiStrategy = "provided",
    candidate_roi_proposal_mode: str = "brightness-gradient",
    max_splat_points: int = 50_000,
    rng_seed: int = 42,
    force: bool = False,
    splat_sampling_mode: str = "surface",
    presentation_splat_points: int = 300_000,
    presentation_splat_sampling_mode: str = "volume",
    compute_splat_reconstruction_audit: bool = True,
    splat_reconstruction_audit_max_points: int = 100_000,
    splat_reconstruction_audit_smooth_sigma: float = 1.0,
    splat_reconstruction_audit_kernel_mode: str = "gaussian",
    splat_reconstruction_audit_gaussian_sigma_extent: float = 2.5,
    train_gaussian_field: bool = False,
    gaussian_num_gaussians: int = 20_000,
    gaussian_steps: int = 300,
    gaussian_image_size: int = 128,
    gaussian_views: str = "all",
    gaussian_device: str = "auto",
    gaussian_learning_rate: float = 1e-3,
    gaussian_volume_loss_weight: float = 0.0,
    gaussian_volume_sample_count: int = 4096,
    compute_optimized_gaussian_field_audit: bool = True,
    optimized_gaussian_field_audit_sample_count: int = 8192,
    optimized_gaussian_field_audit_device: str = "cpu",
    train_synthetic_gaussian_field: bool = False,
    compute_paired_gaussian_residual: bool = True,
    paired_gaussian_max_points: Optional[int] = None,
    paired_gaussian_heatmap_percentile: float = 99.0,
    paired_gaussian_smooth_sigma: float = 1.0,
    paired_gaussian_heatmap_method: str = "gaussian",
    paired_gaussian_heatmap_sigma_extent: float = 2.5,
    paired_gaussian_heatmap_exact_max_points: int = 50_000,
    paired_gaussian_support_basis: str = "union",
    compute_control_gaussian_prior_residual: bool = False,
    control_gaussian_prior_max_points: Optional[int] = None,
    control_gaussian_prior_min_controls: int = 2,
    control_gaussian_prior_heatmap_percentile: float = 99.0,
    control_gaussian_prior_smooth_sigma: float = 1.0,
    control_gaussian_prior_heatmap_method: str = "gaussian",
    control_gaussian_prior_heatmap_exact_max_points: int = 50_000,
    control_gaussian_prior_std_floor_fraction: float = 0.05,
    compute_residual_candidates: bool = True,
    residual_candidate_max_candidates: int = 5,
    residual_candidate_percentile: float = 90.0,
    residual_candidate_min_threshold: float = 0.05,
    residual_candidate_min_component_voxels: int = 8,
    residual_candidate_dilate_mm: float = 5.0,
    compute_gaussian_residual: bool = False,
    gaussian_residual_k_neighbors: int = 4,
    gaussian_residual_max_points: int = 200_000,
    gaussian_residual_heatmap_percentile: float = 99.0,
    gaussian_residual_heatmap_method: str = "field",
    gaussian_residual_field_chunk_size: int = 128,
    supervised_roi_checkpoint: Optional[Path] = None,
    supervised_roi_operating_point: Optional[Path] = None,
    supervised_roi_threshold: float = 0.5,
    supervised_roi_min_component_voxels: int = 0,
    supervised_roi_min_voxels: int = 1,
    supervised_roi_max_foreground_fraction: Optional[float] = 0.01,
    supervised_roi_device: str = "auto",
    supervised_roi_selection_mode: str = "threshold",
    supervised_roi_foreground_fraction: Optional[float] = None,
    control_prior_max_controls: Optional[int] = None,
    pipeline_preset: Optional[str] = None,
) -> Path:
    """Run end-to-end pipeline for all (or selected) subjects.

    Returns path to the written index.json.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_records = list(discover_dataset(dataset_dir))

    if subjects:
        by_id = {record.subject_id: record for record in all_records}
        participants_tsv = dataset_dir / "participants.tsv"
        known = load_participants(participants_tsv) if participants_tsv.is_file() else {}
        records = []
        for sub_id in subjects:
            if sub_id in by_id:
                records.append(by_id[sub_id])
            else:
                base = known.get(sub_id)
                group = base.group if base else ""
                meta = base.metadata if base else {}
                records.append(discover_subject(dataset_dir, sub_id, group=group, metadata=meta))
    else:
        records = all_records

    if not records:
        print("No subjects found.")
        return output_root / "index.json"

    print(f"Processing {len(records)} subject(s)...")
    index_entries = []
    for rec in records:
        print(f"Subject: {rec.subject_id} ({rec.group})")
        preferred_modality = _preferred_modality_for_record(rec)
        control_prior_volumes = (
            _control_prior_volumes_for(
                rec,
                all_records,
                preferred_modality=preferred_modality,
                max_controls=control_prior_max_controls,
            )
            if (
                synthetic_mode == "control-prior"
                or compute_control_gaussian_prior_residual
            )
            else None
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
            compute_optimized_gaussian_field_audit=compute_optimized_gaussian_field_audit,
            optimized_gaussian_field_audit_sample_count=(
                optimized_gaussian_field_audit_sample_count
            ),
            optimized_gaussian_field_audit_device=(
                optimized_gaussian_field_audit_device
            ),
            train_synthetic_gaussian_field=train_synthetic_gaussian_field,
            compute_paired_gaussian_residual=compute_paired_gaussian_residual,
            paired_gaussian_max_points=paired_gaussian_max_points,
            paired_gaussian_heatmap_percentile=paired_gaussian_heatmap_percentile,
            paired_gaussian_smooth_sigma=paired_gaussian_smooth_sigma,
            paired_gaussian_heatmap_method=paired_gaussian_heatmap_method,
            paired_gaussian_heatmap_sigma_extent=paired_gaussian_heatmap_sigma_extent,
            paired_gaussian_heatmap_exact_max_points=(
                paired_gaussian_heatmap_exact_max_points
            ),
            paired_gaussian_support_basis=paired_gaussian_support_basis,
            compute_control_gaussian_prior_residual=(
                compute_control_gaussian_prior_residual
            ),
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
            compute_residual_candidates=compute_residual_candidates,
            residual_candidate_max_candidates=residual_candidate_max_candidates,
            residual_candidate_percentile=residual_candidate_percentile,
            residual_candidate_min_threshold=residual_candidate_min_threshold,
            residual_candidate_min_component_voxels=residual_candidate_min_component_voxels,
            residual_candidate_dilate_mm=residual_candidate_dilate_mm,
            compute_gaussian_residual=compute_gaussian_residual,
            gaussian_residual_k_neighbors=gaussian_residual_k_neighbors,
            gaussian_residual_max_points=gaussian_residual_max_points,
            gaussian_residual_heatmap_percentile=gaussian_residual_heatmap_percentile,
            gaussian_residual_heatmap_method=gaussian_residual_heatmap_method,
            gaussian_residual_field_chunk_size=gaussian_residual_field_chunk_size,
            supervised_roi_checkpoint=supervised_roi_checkpoint,
            supervised_roi_operating_point=supervised_roi_operating_point,
            supervised_roi_threshold=supervised_roi_threshold,
            supervised_roi_min_component_voxels=supervised_roi_min_component_voxels,
            supervised_roi_min_voxels=supervised_roi_min_voxels,
            supervised_roi_max_foreground_fraction=supervised_roi_max_foreground_fraction,
            supervised_roi_device=supervised_roi_device,
            supervised_roi_selection_mode=supervised_roi_selection_mode,
            supervised_roi_foreground_fraction=supervised_roi_foreground_fraction,
            control_prior_volumes=control_prior_volumes,
            pipeline_preset=pipeline_preset,
        )
        index_entries.append({
            "subject": rec.subject_id,
            "group": rec.group or "unknown",
            "roi_available": rec.has_roi,
            "modalities": manifest.get("modalities", []),
            "stage_status": manifest.get("stage_status", {}),
            "synthetic_normal_source": manifest.get("synthetic_normal_source", "none"),
            "manifest": f"{rec.subject_id}/manifest.json",
        })

    index = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "dataset": "OpenNeuro ds004199",
        "subjects": index_entries,
    }
    index_path = output_root / "index.json"
    with index_path.open("w") as fh:
        json.dump(index, fh, indent=2)
    print(f"\nIndex written -> {index_path}")
    return index_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dataset-wide FCD end-to-end pipeline."
    )
    p.add_argument("--dataset-dir", type=Path, required=True,
                   help="Root of ds004199 BIDS directory.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/end_to_end"))
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Specific subject IDs to process (default: all discovered).")
    p.add_argument("--synthetic-mode",
                   choices=["placeholder", "nvidia", "control-prior", "contralateral"],
                   default="placeholder")
    p.add_argument("--control-prior-max-controls", type=int, default=None,
                   help="Engineering pilot cap for --synthetic-mode control-prior. "
                        "Positive values use a deterministic subset of healthy controls; "
                        "omit for final all-controls validation.")
    p.add_argument("--roi-strategy",
                   choices=["provided", "auto", "provided-or-auto", "supervised", "provided-or-supervised", "none"],
                   default="provided",
                   help="ROI selection strategy: provided (given ROI), auto (heuristic proposal), "
                        "provided-or-auto (prefer given, fallback to proposal), "
                        "supervised (supervised model prediction), "
                        "provided-or-supervised (prefer given, fallback to supervised), "
                        "none (no ROI).")
    p.add_argument("--candidate-roi-proposal-mode",
                   choices=PROPOSAL_MODES,
                   default="brightness-gradient",
                   help="Heuristic feature model for roi-strategy auto/provided-or-auto.")
    p.add_argument("--max-splat-points", type=int, default=50_000)
    p.add_argument("--splat-sampling-mode", choices=["surface", "volume", "uniform"],
                   default="surface")
    p.add_argument("--presentation-splat-points", type=int, default=300_000)
    p.add_argument("--presentation-splat-sampling-mode", choices=["surface", "volume", "uniform"],
                   default="volume",
                   help="Sampling mode for patient_splat_dense.ply; volume is the product-facing default.")
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
    # Gaussian field optional stage
    p.add_argument("--train-gaussian-field", action="store_true", default=False,
                   help="Run optimized Gaussian field training after direct splat stage.")
    p.add_argument("--gaussian-num-gaussians", type=int, default=20_000)
    p.add_argument("--gaussian-steps", type=int, default=300)
    p.add_argument("--gaussian-image-size", type=int, default=128)
    p.add_argument("--gaussian-views", type=str, default="all",
                   help="Comma-separated views or 'all' for Gaussian field training.")
    p.add_argument("--gaussian-device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--gaussian-learning-rate", type=float, default=1e-3)
    p.add_argument("--gaussian-volume-loss-weight", type=float, default=None,
                   help="Optional sampled 3D voxel reconstruction loss weight. "
                        "Defaults to 0.0 normally, or 0.25 under --paper-3dgs-mode.")
    p.add_argument("--gaussian-volume-sample-count", type=int, default=None,
                   help="Number of deterministic voxel samples used by the volume loss. "
                        "Defaults to 4096 normally, or 8192 under --paper-3dgs-mode.")
    p.add_argument("--skip-optimized-gaussian-field-audit", action="store_true", default=False,
                   help="Disable held-out MRI fidelity audit for optimized Gaussian fields.")
    p.add_argument("--optimized-gaussian-field-audit-sample-count", type=int, default=8192,
                   help="Held-out voxel samples for optimized Gaussian field fidelity audit.")
    p.add_argument("--optimized-gaussian-field-audit-device",
                   choices=["auto", "cuda", "cpu"], default="cpu",
                   help="Device for optimized Gaussian field fidelity audit.")
    # Synthetic normal Gaussian field optional stage
    p.add_argument("--train-synthetic-gaussian-field", action="store_true", default=False,
                   help="Run optimized Gaussian field training on synthetic normal volume.")
    # Paired-support Gaussian residual stage
    p.add_argument("--skip-paired-gaussian-residual", action="store_true", default=False,
                   help="Disable automatic paired-support patient-vs-normal splat subtraction.")
    p.add_argument("--paired-gaussian-max-points", type=int, default=None,
                   help=(
                       "Maximum paired-support Gaussians; defaults to the larger "
                       "of --max-splat-points and --presentation-splat-points."
                   ))
    p.add_argument("--paired-gaussian-heatmap-percentile", type=float, default=99.0,
                   help="Percentile for paired residual splat normalization.")
    p.add_argument("--paired-gaussian-smooth-sigma", type=float, default=1.0,
                   help="Voxel sigma for paired residual NIfTI heatmap smoothing.")
    p.add_argument("--paired-gaussian-heatmap-method",
                   choices=["scatter", "gaussian"], default="gaussian",
                   help="How paired residual scores are projected back to NIfTI.")
    p.add_argument("--paired-gaussian-heatmap-sigma-extent", type=float, default=2.5,
                   help="Gaussian support radius, in sigmas, for paired residual heatmaps.")
    p.add_argument("--paired-gaussian-heatmap-exact-max-points", type=int, default=50_000,
                   help=(
                       "Maximum active residual Gaussians for exact anisotropic "
                       "heatmap accumulation before using convolution fallback."
                   ))
    p.add_argument("--paired-gaussian-support-basis",
                   choices=["union", "patient", "normal", "intersection"],
                   default="union",
                   help=(
                       "Signal used to sample the shared paired-Gaussian support. "
                       "union preserves the historical max(patient, synthetic_normal) "
                       "support; patient is useful for patient-geometry ablations."
                   ))
    # Patient-support healthy-control Gaussian prior residual stage
    p.add_argument("--compute-control-gaussian-prior-residual",
                   action="store_true", default=False,
                   help=(
                       "Build patient-support Gaussian splats for the patient, the "
                       "healthy-control prior mean, and the z-residual subtraction."
                   ))
    p.add_argument("--control-gaussian-prior-max-points", type=int, default=None,
                   help=(
                       "Maximum patient-support control-prior Gaussians; defaults "
                       "to the larger of --max-splat-points and "
                       "--presentation-splat-points."
                   ))
    p.add_argument("--control-gaussian-prior-min-controls", type=int, default=2,
                   help="Minimum healthy controls required for control-prior residuals.")
    p.add_argument("--control-gaussian-prior-heatmap-percentile",
                   type=float, default=99.0,
                   help="Percentile for control-prior z-residual splat normalization.")
    p.add_argument("--control-gaussian-prior-smooth-sigma", type=float, default=1.0,
                   help="Voxel sigma for control-prior residual NIfTI heatmap smoothing.")
    p.add_argument("--control-gaussian-prior-heatmap-method",
                   choices=["scatter", "gaussian"], default="gaussian",
                   help="How control-prior z-residual scores are projected to NIfTI.")
    p.add_argument("--control-gaussian-prior-heatmap-exact-max-points",
                   type=int, default=50_000,
                   help=(
                       "Maximum active control-prior residual Gaussians for exact "
                       "anisotropic heatmap accumulation before fallback."
                   ))
    p.add_argument("--control-gaussian-prior-std-floor-fraction",
                   type=float, default=0.05,
                   help=(
                       "Fraction of global control intensity std used as the per-Gaussian "
                       "standard-deviation floor for z-residual stability."
                   ))
    # Residual-derived candidate extraction
    p.add_argument("--skip-residual-candidates", action="store_true", default=False,
                   help="Disable residual-derived lesion candidate masks.")
    p.add_argument("--residual-candidate-max-candidates", type=int, default=5,
                   help="Maximum connected residual candidates to export.")
    p.add_argument("--residual-candidate-percentile", type=float, default=90.0,
                   help="Residual percentile used for candidate thresholding.")
    p.add_argument("--residual-candidate-min-threshold", type=float, default=0.05,
                   help="Minimum normalized residual value for candidate thresholding.")
    p.add_argument("--residual-candidate-min-component-voxels", type=int, default=8,
                   help="Minimum component size for residual-derived candidates.")
    p.add_argument("--residual-candidate-dilate-mm", type=float, default=5.0,
                   help="Physical radius used to grow residual candidate masks for review.")
    # Gaussian-space residual comparison optional stage
    p.add_argument("--compute-gaussian-residual", action="store_true", default=False,
                   help="Compute Gaussian-space residual comparison between patient and normal fields.")
    p.add_argument("--gaussian-residual-k-neighbors", type=int, default=4,
                   help="Number of neighbors for KNN in Gaussian residual computation.")
    p.add_argument("--gaussian-residual-max-points", type=int, default=200_000,
                   help="Maximum number of points for Gaussian residual computation.")
    p.add_argument("--gaussian-residual-heatmap-percentile", type=float, default=99.0,
                   help="Percentile for Gaussian residual heatmap normalization.")
    p.add_argument("--gaussian-residual-heatmap-method", choices=["field", "scatter"],
                   default="field",
                   help="Gaussian residual heatmap method: field evaluates both Gaussian fields "
                        "on the reference grid before subtraction; scatter uses legacy KNN scatter.")
    p.add_argument("--gaussian-residual-field-chunk-size", type=int, default=128,
                   help="Gaussian chunk size for field-mode residual heatmap evaluation.")
    # Supervised ROI arguments
    p.add_argument("--supervised-roi-checkpoint", type=Path, default=None,
                   help="Path to supervised lesion proposer checkpoint (.pt file).")
    p.add_argument("--supervised-roi-operating-point", type=Path, default=None,
                   help=(
                       "JSON operating point from fcd-lesion-proposer sweep, or a "
                       "direct JSON object with threshold/selection settings."
                   ))
    p.add_argument("--supervised-roi-threshold", type=float, default=0.5,
                   help="Probability threshold for supervised ROI binarization.")
    p.add_argument("--supervised-roi-min-component-voxels", type=int, default=0,
                   help="Minimum voxel count to retain connected components in supervised ROI.")
    p.add_argument("--supervised-roi-min-voxels", type=int, default=1,
                   help="Reject supervised ROI predictions smaller than this voxel count.")
    p.add_argument("--supervised-roi-max-foreground-fraction", type=float, default=0.01,
                   help=(
                       "Reject supervised ROI predictions whose foreground fraction exceeds "
                       "this value; set negative to disable."
                   ))
    p.add_argument("--supervised-roi-selection-mode",
                   choices=["threshold", "top-fraction", "threshold-and-top-fraction"],
                   default="threshold",
                   help="How supervised probability maps are converted to an ROI mask.")
    p.add_argument("--supervised-roi-foreground-fraction", type=float, default=None,
                   help=(
                       "Requested ROI foreground fraction for top-fraction supervised "
                       "selection modes."
                   ))
    p.add_argument("--supervised-roi-device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="Device for supervised ROI prediction (auto=GPU if available, else CPU).")
    # Paper-facing 3DGS preset
    p.add_argument("--paper-3dgs-mode", action="store_true", default=False,
                   help=(
                       "Enable the paper-facing 3DGS preset: patient MRI -> optimized "
                       "Gaussian field -> synthetic normal projected onto patient support -> "
                       "optimized Gaussian-field residual. Sets --train-gaussian-field, "
                       "--train-synthetic-gaussian-field, --compute-gaussian-residual, "
                       "and keeps paired splat subtraction on. Volume loss defaults "
                       "to 0.25 and sample count to 8192 unless explicitly overridden."
                   ))
    return p


def _nvidia_stages_ok(manifests: list[dict]) -> bool:
    """Return False if any subject has a non-ok synthetic_normal or residual stage."""
    for m in manifests:
        ss = m.get("stage_status", {})
        if ss.get("synthetic_normal") != "ok":
            return False
        if ss.get("residual") != "ok":
            return False
    return True


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # --paper-3dgs-mode preset: flip on the 3DGS training flags and fill in
    # volume-loss defaults only when the user did not explicitly pass them.
    _PAPER_3DGS_VOLUME_LOSS_DEFAULT = 0.25
    _PAPER_3DGS_SAMPLE_COUNT_DEFAULT = 8192
    _DEFAULT_VOLUME_LOSS_WEIGHT = 0.0
    _DEFAULT_VOLUME_SAMPLE_COUNT = 4096

    if args.paper_3dgs_mode:
        args.train_gaussian_field = True
        args.train_synthetic_gaussian_field = True
        args.compute_gaussian_residual = True
        args.skip_paired_gaussian_residual = False
        args.compute_control_gaussian_prior_residual = True
        if args.gaussian_volume_loss_weight is None:
            args.gaussian_volume_loss_weight = _PAPER_3DGS_VOLUME_LOSS_DEFAULT
        if args.gaussian_volume_sample_count is None:
            args.gaussian_volume_sample_count = _PAPER_3DGS_SAMPLE_COUNT_DEFAULT

    # Fill in non-preset sentinel defaults
    if args.gaussian_volume_loss_weight is None:
        args.gaussian_volume_loss_weight = _DEFAULT_VOLUME_LOSS_WEIGHT
    if args.gaussian_volume_sample_count is None:
        args.gaussian_volume_sample_count = _DEFAULT_VOLUME_SAMPLE_COUNT

    pipeline_preset = "paper_3dgs" if args.paper_3dgs_mode else None

    # build_end_to_end returns the index path; we need manifests for the exit check.
    # Re-read manifests only when needed (--synthetic-mode nvidia).
    output_root = args.output_dir
    index_path = build_end_to_end(
        dataset_dir=args.dataset_dir,
        output_root=output_root,
        subjects=args.subjects,
        synthetic_mode=args.synthetic_mode,
        roi_strategy=args.roi_strategy,
        candidate_roi_proposal_mode=args.candidate_roi_proposal_mode,
        max_splat_points=args.max_splat_points,
        rng_seed=args.rng_seed,
        force=args.force,
        splat_sampling_mode=args.splat_sampling_mode,
        presentation_splat_points=args.presentation_splat_points,
        presentation_splat_sampling_mode=args.presentation_splat_sampling_mode,
        compute_splat_reconstruction_audit=not args.skip_splat_reconstruction_audit,
        splat_reconstruction_audit_max_points=args.splat_reconstruction_audit_max_points,
        splat_reconstruction_audit_smooth_sigma=args.splat_reconstruction_audit_smooth_sigma,
        splat_reconstruction_audit_kernel_mode=args.splat_reconstruction_audit_kernel_mode,
        splat_reconstruction_audit_gaussian_sigma_extent=(
            args.splat_reconstruction_audit_gaussian_sigma_extent
        ),
        train_gaussian_field=args.train_gaussian_field,
        gaussian_num_gaussians=args.gaussian_num_gaussians,
        gaussian_steps=args.gaussian_steps,
        gaussian_image_size=args.gaussian_image_size,
        gaussian_views=args.gaussian_views,
        gaussian_device=args.gaussian_device,
        gaussian_learning_rate=args.gaussian_learning_rate,
        gaussian_volume_loss_weight=args.gaussian_volume_loss_weight,
        gaussian_volume_sample_count=args.gaussian_volume_sample_count,
        compute_optimized_gaussian_field_audit=(
            not args.skip_optimized_gaussian_field_audit
        ),
        optimized_gaussian_field_audit_sample_count=(
            args.optimized_gaussian_field_audit_sample_count
        ),
        optimized_gaussian_field_audit_device=(
            args.optimized_gaussian_field_audit_device
        ),
        train_synthetic_gaussian_field=args.train_synthetic_gaussian_field,
        compute_paired_gaussian_residual=not args.skip_paired_gaussian_residual,
        paired_gaussian_max_points=args.paired_gaussian_max_points,
        paired_gaussian_heatmap_percentile=args.paired_gaussian_heatmap_percentile,
        paired_gaussian_smooth_sigma=args.paired_gaussian_smooth_sigma,
        paired_gaussian_heatmap_method=args.paired_gaussian_heatmap_method,
        paired_gaussian_heatmap_sigma_extent=args.paired_gaussian_heatmap_sigma_extent,
        paired_gaussian_heatmap_exact_max_points=(
            args.paired_gaussian_heatmap_exact_max_points
        ),
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
        compute_residual_candidates=not args.skip_residual_candidates,
        residual_candidate_max_candidates=args.residual_candidate_max_candidates,
        residual_candidate_percentile=args.residual_candidate_percentile,
        residual_candidate_min_threshold=args.residual_candidate_min_threshold,
        residual_candidate_min_component_voxels=args.residual_candidate_min_component_voxels,
        residual_candidate_dilate_mm=args.residual_candidate_dilate_mm,
        compute_gaussian_residual=args.compute_gaussian_residual,
        gaussian_residual_k_neighbors=args.gaussian_residual_k_neighbors,
        gaussian_residual_max_points=args.gaussian_residual_max_points,
        gaussian_residual_heatmap_percentile=args.gaussian_residual_heatmap_percentile,
        gaussian_residual_heatmap_method=args.gaussian_residual_heatmap_method,
        gaussian_residual_field_chunk_size=args.gaussian_residual_field_chunk_size,
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
        control_prior_max_controls=args.control_prior_max_controls,
        pipeline_preset=pipeline_preset,
    )

    if args.synthetic_mode == "nvidia":
        # Load the index to find which subjects were processed, then check manifests.
        manifests: list[dict] = []
        if index_path.is_file():
            with index_path.open() as fh:
                index = json.load(fh)
            for entry in index.get("subjects", []):
                manifest_rel = entry.get("manifest", "")
                manifest_path = output_root / manifest_rel
                if manifest_path.is_file():
                    with manifest_path.open() as fh:
                        manifests.append(json.load(fh))
        if not _nvidia_stages_ok(manifests):
            print(
                "ERROR: One or more subjects failed synthetic_normal or residual stage "
                "in --synthetic-mode nvidia.",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
