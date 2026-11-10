"""Cohort-level evaluation scaffolding for the FCD pipeline.

Reads index.json and per-subject manifest.json produced by the end-to-end
pipeline, labels subjects as FCD vs HC, pulls scalar residual scores and
post-subtraction residual-candidate localization metrics, then computes
group-level statistics plus AUROC via a rank-based Mann-Whitney U
implementation (no sklearn required).

Entry point: fcd-evaluate-cohort
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional


DEFAULT_PAPER_GATE_THRESHOLDS = {
    "expected_fcd": 85,
    "expected_hc": 85,
    "auroc_min": 0.75,
    "control_prior_similarity_min": 0.85,
    "splat_foreground_coverage_min": 0.50,
    "splat_mip_mean_mae_max": 0.05,
    "optimized_field_foreground_mae_max": 0.25,
    "optimized_field_all_mae_max": 0.20,
    "paired_gaussian_min_points": 50_000,
    "control_gaussian_min_points": 50_000,
    "control_gaussian_min_controls": 2,
    "optimized_gaussian_min_points": 50_000,
    "localization_mean_manual_coverage_min": 0.50,
    "hc_candidate_foreground_fraction_max": 0.01,
}

PAIRED_GAUSSIAN_STATE_SCHEMA = "paired_gaussian_residual_v1"
CONTROL_GAUSSIAN_PRIOR_STATE_SCHEMA = "patient_control_gaussian_prior_v1"
OPTIMIZED_GAUSSIAN_SCORE_METHOD = "indexwise_same_support_intensity"
OPTIMIZED_GAUSSIAN_HEATMAP_METHOD = "field"
PAIRED_GAUSSIAN_SUPPORT_BASIS_LABELS = {
    "max(patient, synthetic_normal)",
    "patient",
    "synthetic_normal",
    "min(patient, synthetic_normal)",
}

PAIRED_METRIC_KEYS = [
    "paired_residual_mean",
    "paired_residual_p95",
    "paired_residual_p99",
    "paired_residual_nonzero_fraction",
    "paired_residual_normalization_value",
    "paired_roi_to_non_roi_ratio",
    "paired_roi_residual_concentration",
]

LEGACY_METRIC_KEYS = [
    "residual_mean",
    "residual_p95",
    "residual_p99",
    "roi_to_non_roi_ratio",
]

RESIDUAL_CANDIDATE_METRIC_KEYS = [
    "residual_candidate_selected_count",
    "residual_candidate_voxels",
    "residual_candidate_volume_ml",
    "residual_candidate_foreground_fraction",
    "residual_candidate_manual_coverage",
    "residual_candidate_precision",
    "residual_candidate_dice",
    "residual_candidate_jaccard",
]

RESIDUAL_CANDIDATE_MANUAL_METRIC_KEYS = [
    "residual_candidate_manual_coverage",
    "residual_candidate_precision",
    "residual_candidate_dice",
    "residual_candidate_jaccard",
]

SPLAT_RECONSTRUCTION_METRIC_KEYS = [
    "splat_foreground_coverage",
    "splat_all_foreground_mae",
    "splat_all_foreground_rmse",
    "splat_all_foreground_psnr",
    "splat_represented_foreground_mae",
    "splat_represented_foreground_psnr",
    "splat_mip_mean_mae",
]

SPLAT_RECONSTRUCTION_EVIDENCE_KEYS = [
    "splat_reconstruction_audit_present",
    "splat_reconstruction_audited_artifact",
    "splat_reconstruction_kernel_mode",
    "splat_reconstruction_readiness_status",
    "splat_reconstruction_blocker_count",
    "splat_reconstruction_warning_count",
    "splat_mri_to_3dgs_claim_allowed",
    "splat_mri_to_3dgs_claim_class",
]

OPTIMIZED_GAUSSIAN_FIELD_AUDIT_EVIDENCE_KEYS = [
    "optimized_gaussian_field_audit_present",
    "optimized_gaussian_field_audit_status",
    "optimized_mri_to_3dgs_claim_allowed",
    "optimized_gaussian_field_audit_blocker_count",
    "optimized_gaussian_field_audit_warning_count",
    "optimized_gaussian_field_audit_sample_count",
    "optimized_gaussian_field_foreground_mae",
    "optimized_gaussian_field_foreground_rmse",
    "optimized_gaussian_field_all_mae",
    "optimized_gaussian_field_all_rmse",
    "optimized_gaussian_field_background_mae",
    "optimized_gaussian_field_training_objective",
    "optimized_gaussian_field_volume_loss_weight",
    "optimized_gaussian_field_volume_sample_count",
    "optimized_gaussian_field_affine_mismatch",
    "optimized_gaussian_field_affine_missing",
]

CONTROL_PRIOR_METRIC_KEYS = [
    "control_prior_reference_similarity_1_minus_mae",
    "control_prior_reference_mae",
    "control_prior_reference_rmse",
    "control_prior_reference_correlation",
]

CONTROL_GAUSSIAN_PRIOR_METRIC_KEYS = [
    "control_gaussian_residual_z_mean",
    "control_gaussian_residual_z_p95",
    "control_gaussian_residual_z_p99",
    "control_gaussian_residual_norm_mean",
    "control_gaussian_residual_norm_p99",
    "control_gaussian_roi_to_non_roi_ratio",
    "control_gaussian_roi_residual_concentration",
]

EVALUATED_METRIC_KEYS = (
    PAIRED_METRIC_KEYS
    + LEGACY_METRIC_KEYS
    + RESIDUAL_CANDIDATE_METRIC_KEYS
    + SPLAT_RECONSTRUCTION_METRIC_KEYS
    + CONTROL_PRIOR_METRIC_KEYS
    + CONTROL_GAUSSIAN_PRIOR_METRIC_KEYS
)

PAIRED_GAUSSIAN_EVIDENCE_KEYS = [
    "paired_gaussian_state_present",
    "paired_gaussian_state_schema",
    "paired_gaussian_same_support",
    "paired_gaussian_method",
    "paired_gaussian_sampling_mode",
    "paired_gaussian_support_basis_mode",
    "paired_gaussian_support_basis",
    "paired_gaussian_n_gaussians",
    "paired_patient_splat_present",
    "paired_normal_splat_present",
    "paired_residual_splat_present",
    "paired_residual_heatmap_present",
]

CONTROL_GAUSSIAN_PRIOR_EVIDENCE_KEYS = [
    "control_gaussian_prior_state_present",
    "control_gaussian_prior_state_schema",
    "control_gaussian_prior_same_support",
    "control_gaussian_prior_method",
    "control_gaussian_prior_sampling_mode",
    "control_gaussian_prior_healthy_reference",
    "control_gaussian_prior_registration_method",
    "control_gaussian_prior_intensity_matching",
    "control_gaussian_prior_n_controls",
    "control_gaussian_prior_min_controls",
    "control_gaussian_prior_n_gaussians",
    "control_gaussian_prior_patient_splat_present",
    "control_gaussian_prior_mean_splat_present",
    "control_gaussian_prior_residual_splat_present",
    "control_gaussian_prior_heatmap_present",
    "control_gaussian_prior_scores_present",
    "control_gaussian_prior_metrics_present",
    "control_gaussian_prior_roi_metrics_independent_validation",
]

OPTIMIZED_GAUSSIAN_EVIDENCE_KEYS = [
    "optimized_gaussian_residual_present",
    "optimized_gaussian_residual_splat_present",
    "optimized_gaussian_residual_heatmap_present",
    "optimized_gaussian_residual_metrics_present",
    "optimized_gaussian_residual_scores_present",
    "optimized_gaussian_same_support",
    "optimized_gaussian_subtraction_claim_allowed",
    "optimized_gaussian_score_method",
    "optimized_gaussian_heatmap_method",
    "optimized_gaussian_n_patient",
    "optimized_gaussian_n_normal",
    "optimized_gaussian_means_norm_max_abs_diff",
    "optimized_gaussian_log_scales_max_abs_diff",
    "optimized_gaussian_quat_wxyz_max_abs_diff",
]

DATA_QUALITY_EVIDENCE_KEYS = [
    "data_quality_present",
    "data_quality_stage_status",
    "data_quality_upload_readiness_status",
    "data_quality_can_run_3dgs",
    "data_quality_can_subtract_patient_normal",
    "data_quality_blocking_issue_count",
    "data_quality_review_issue_count",
    "data_quality_warning_count",
    "data_quality_primary_modalities",
    "data_quality_t1w_to_flair_needs_resample",
    "data_quality_roi_to_flair_needs_resample",
    "data_quality_roi_resampled_to_base",
    "data_quality_has_nonfinite_blocker",
    "data_quality_dicom_conversion_status",
    "data_quality_dicom_conversion_can_run_3dgs",
    "data_quality_dicom_conversion_paper_claim_allowed",
    "data_quality_dicom_conversion_blocking_issue_count",
    "data_quality_dicom_conversion_review_issue_count",
]

SYNTHETIC_NORMAL_AUDIT_EVIDENCE_KEYS = [
    "synthetic_normal_audit_present",
    "synthetic_normal_counterfactual_class",
    "synthetic_normal_claim_boundary",
    "synthetic_normal_claim_class",
    "synthetic_normal_subtraction_readiness_status",
    "synthetic_normal_subtraction_blocker_count",
    "synthetic_normal_subtraction_warning_count",
    "synthetic_normal_gaussian_subtraction_claim_allowed",
    "synthetic_normal_patient_conditioned_inpainting",
    "synthetic_normal_patient_conditioned_counterfactual_claim_allowed",
    "synthetic_normal_same_grid_before_resample",
    "synthetic_normal_resampled_to_patient_grid",
    "synthetic_normal_outside_roi_similarity_1_minus_mae",
    "synthetic_normal_inside_to_outside_mae_ratio",
]

ROI_FREE_SOURCES = {"candidate", "auto", "supervised"}
ORACLE_ROI_SOURCES = {"provided", "manual"}


# ---------------------------------------------------------------------------
# Mann-Whitney / AUROC (no sklearn)
# ---------------------------------------------------------------------------

def _mannwhitney_auroc(pos_scores: list[float], neg_scores: list[float]) -> float:
    """AUROC via rank-based Mann-Whitney U.

    Returns float in [0, 1], or 0.5 if either list is empty.
    """
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Count concordant pairs
    u = sum(
        1.0 if p > n else (0.5 if p == n else 0.0)
        for p in pos_scores
        for n in neg_scores
    )
    return u / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

def _coerce_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_score_and_source(manifest: dict) -> tuple[Optional[float], str]:
    """Pull the best available scalar residual score and source from a manifest.

    Priority favors the patient-support healthy-control Gaussian prior when it
    is present, because that is the current paper-facing fallback for the
    original patient-vs-healthy subtraction idea. Paired-support splat
    subtraction and legacy optimized-field/voxel residual scores are retained
    as fallbacks.
    """
    control_gp = manifest.get("control_gaussian_prior_meta", {})
    if isinstance(control_gp, dict):
        for key in (
            "residual_z_mean",
            "residual_norm_mean",
            "residual_z_p99",
            "residual_norm_p99",
        ):
            score = _coerce_float(control_gp.get(key))
            if score is not None:
                return score, f"control_gaussian_prior_meta.{key}"

    paired = manifest.get("paired_gaussian_residual_meta", {})
    for key in (
        "residual_mean",
        "residual_nonzero_fraction",
        "residual_normalization_value",
        "residual_p99",
    ):
        score = _coerce_float(paired.get(key))
        if score is not None:
            return score, f"paired_gaussian_residual_meta.{key}"

    gr = manifest.get("gaussian_residual_meta", {})
    for key in ("residual_p99", "residual_mean"):
        score = _coerce_float(gr.get(key))
        if score is not None:
            return score, f"gaussian_residual_meta.{key}"

    conc = manifest.get("artifacts", {}).get("residual_roi_concentration")
    score = _coerce_float(conc)
    if score is not None:
        return score, "artifacts.residual_roi_concentration"

    return None, "missing"


def _extract_score(manifest: dict) -> Optional[float]:
    """Pull the best available scalar residual score from a manifest.

    Priority:
      1. control_gaussian_prior_meta residual z/norm scores
      2. paired_gaussian_residual_meta residual scores
      3. gaussian_residual_meta residual scores
      4. artifacts.residual_roi_concentration  (stored as a numeric string)
    """
    return _extract_score_and_source(manifest)[0]


def _extract_all_metrics(
    manifest: dict, subject_output_dir: Path
) -> dict[str, Optional[float]]:
    """Extract all available paired and legacy residual metrics.

    Reads from paired_gaussian_residual_meta first, then legacy
    gaussian_residual_meta. Also checks sidecar metrics JSON artifacts when
    present. Resolves artifact paths relative to subject_output_dir.

    Returns dict with metric names as keys, float or None as values.
    """
    metrics: dict[str, Optional[float]] = {key: None for key in EVALUATED_METRIC_KEYS}

    paired = manifest.get("paired_gaussian_residual_meta", {})
    paired_mapping = {
        "residual_mean": "paired_residual_mean",
        "residual_p95": "paired_residual_p95",
        "residual_p99": "paired_residual_p99",
        "residual_nonzero_fraction": "paired_residual_nonzero_fraction",
        "residual_normalization_value": "paired_residual_normalization_value",
        "roi_to_non_roi_ratio": "paired_roi_to_non_roi_ratio",
        "roi_residual_concentration": "paired_roi_residual_concentration",
    }
    for source_key, metric_key in paired_mapping.items():
        value = _coerce_float(paired.get(source_key))
        if value is not None:
            metrics[metric_key] = value

    gr = manifest.get("gaussian_residual_meta", {})
    for key in ["residual_mean", "residual_p99"]:
        value = _coerce_float(gr.get(key))
        if value is not None:
            metrics[key] = value

    artifacts = manifest.get("artifacts", {})

    paired_artifact_file = artifacts.get("paired_gaussian_residual_metrics")
    if paired_artifact_file:
        paired_artifact_path = subject_output_dir / paired_artifact_file
        if paired_artifact_path.is_file():
            try:
                with paired_artifact_path.open() as fh:
                    paired_artifact_data = json.load(fh)
                for source_key, metric_key in paired_mapping.items():
                    value = _coerce_float(paired_artifact_data.get(source_key))
                    if value is not None:
                        metrics[metric_key] = value
            except (json.JSONDecodeError, IOError):
                pass

    artifact_file = artifacts.get("gaussian_residual_metrics")
    if artifact_file:
        artifact_path = subject_output_dir / artifact_file
        if artifact_path.is_file():
            try:
                with artifact_path.open() as fh:
                    artifact_data = json.load(fh)
                for key in ["residual_p95", "roi_to_non_roi_ratio"]:
                    value = _coerce_float(artifact_data.get(key))
                    if value is not None:
                        metrics[key] = value
            except (json.JSONDecodeError, IOError):
                pass

    candidate_meta = manifest.get("residual_candidate_meta", {})
    candidate_mapping = {
        "selected_count": "residual_candidate_selected_count",
        "candidate_voxels": "residual_candidate_voxels",
        "candidate_volume_ml": "residual_candidate_volume_ml",
        "candidate_foreground_fraction": "residual_candidate_foreground_fraction",
        "manual_coverage": "residual_candidate_manual_coverage",
        "candidate_precision": "residual_candidate_precision",
        "dice": "residual_candidate_dice",
        "jaccard": "residual_candidate_jaccard",
    }
    if isinstance(candidate_meta, dict):
        for source_key, metric_key in candidate_mapping.items():
            value = _coerce_float(candidate_meta.get(source_key))
            if value is not None:
                metrics[metric_key] = value

    candidate_artifact_file = artifacts.get("residual_candidate_manifest")
    if candidate_artifact_file:
        candidate_artifact_path = subject_output_dir / candidate_artifact_file
        if candidate_artifact_path.is_file():
            try:
                with candidate_artifact_path.open() as fh:
                    candidate_artifact_data = json.load(fh)
                for source_key, metric_key in candidate_mapping.items():
                    value = _coerce_float(candidate_artifact_data.get(source_key))
                    if value is not None:
                        metrics[metric_key] = value
                manual_overlap = candidate_artifact_data.get("manual_overlap", {})
                if isinstance(manual_overlap, dict):
                    for source_key, metric_key in candidate_mapping.items():
                        value = _coerce_float(manual_overlap.get(source_key))
                        if value is not None:
                            metrics[metric_key] = value
            except (json.JSONDecodeError, IOError):
                pass

    splat_meta = manifest.get("splat_reconstruction_audit_meta", {})
    if isinstance(splat_meta, dict):
        splat_mapping = {
            "foreground_coverage": "splat_foreground_coverage",
            "all_foreground_mae": "splat_all_foreground_mae",
            "all_foreground_rmse": "splat_all_foreground_rmse",
            "all_foreground_psnr": "splat_all_foreground_psnr",
            "represented_foreground_mae": "splat_represented_foreground_mae",
            "represented_foreground_psnr": "splat_represented_foreground_psnr",
            "mip_mean_mae": "splat_mip_mean_mae",
        }
        for source_key, metric_key in splat_mapping.items():
            value = _coerce_float(splat_meta.get(source_key))
            if value is not None:
                metrics[metric_key] = value

    control_prior_meta = manifest.get("control_prior_meta", {})
    if isinstance(control_prior_meta, dict):
        for key in CONTROL_PRIOR_METRIC_KEYS:
            value = _coerce_float(control_prior_meta.get(key))
            if value is not None:
                metrics[key] = value

    control_gp_mapping = {
        "residual_z_mean": "control_gaussian_residual_z_mean",
        "residual_z_p95": "control_gaussian_residual_z_p95",
        "residual_z_p99": "control_gaussian_residual_z_p99",
        "residual_norm_mean": "control_gaussian_residual_norm_mean",
        "residual_norm_p99": "control_gaussian_residual_norm_p99",
        "roi_to_non_roi_ratio": "control_gaussian_roi_to_non_roi_ratio",
        "roi_residual_concentration": "control_gaussian_roi_residual_concentration",
    }
    control_gp_meta = manifest.get("control_gaussian_prior_meta", {})
    if isinstance(control_gp_meta, dict):
        for source_key, metric_key in control_gp_mapping.items():
            value = _coerce_float(control_gp_meta.get(source_key))
            if value is not None:
                metrics[metric_key] = value

    control_gp_artifact_file = artifacts.get("control_gaussian_prior_metrics")
    if control_gp_artifact_file:
        control_gp_artifact_path = subject_output_dir / control_gp_artifact_file
        if control_gp_artifact_path.is_file():
            try:
                with control_gp_artifact_path.open() as fh:
                    control_gp_artifact_data = json.load(fh)
                for source_key, metric_key in control_gp_mapping.items():
                    value = _coerce_float(control_gp_artifact_data.get(source_key))
                    if value is not None:
                        metrics[metric_key] = value
            except (json.JSONDecodeError, IOError):
                pass

    return metrics


def _empty_paired_gaussian_evidence() -> dict[str, object]:
    return {
        "paired_gaussian_state_present": False,
        "paired_gaussian_state_schema": None,
        "paired_gaussian_same_support": None,
        "paired_gaussian_method": None,
        "paired_gaussian_sampling_mode": None,
        "paired_gaussian_support_basis_mode": None,
        "paired_gaussian_support_basis": None,
        "paired_gaussian_n_gaussians": None,
        "paired_patient_splat_present": False,
        "paired_normal_splat_present": False,
        "paired_residual_splat_present": False,
        "paired_residual_heatmap_present": False,
    }


def _empty_optimized_gaussian_evidence() -> dict[str, object]:
    return {
        "optimized_gaussian_residual_present": False,
        "optimized_gaussian_residual_splat_present": False,
        "optimized_gaussian_residual_heatmap_present": False,
        "optimized_gaussian_residual_metrics_present": False,
        "optimized_gaussian_residual_scores_present": False,
        "optimized_gaussian_same_support": None,
        "optimized_gaussian_subtraction_claim_allowed": None,
        "optimized_gaussian_score_method": None,
        "optimized_gaussian_heatmap_method": None,
        "optimized_gaussian_n_patient": None,
        "optimized_gaussian_n_normal": None,
        "optimized_gaussian_means_norm_max_abs_diff": None,
        "optimized_gaussian_log_scales_max_abs_diff": None,
        "optimized_gaussian_quat_wxyz_max_abs_diff": None,
    }


def _empty_control_gaussian_prior_evidence() -> dict[str, object]:
    return {
        "control_gaussian_prior_state_present": False,
        "control_gaussian_prior_state_schema": None,
        "control_gaussian_prior_same_support": None,
        "control_gaussian_prior_method": None,
        "control_gaussian_prior_sampling_mode": None,
        "control_gaussian_prior_healthy_reference": None,
        "control_gaussian_prior_registration_method": None,
        "control_gaussian_prior_intensity_matching": None,
        "control_gaussian_prior_n_controls": None,
        "control_gaussian_prior_min_controls": None,
        "control_gaussian_prior_n_gaussians": None,
        "control_gaussian_prior_patient_splat_present": False,
        "control_gaussian_prior_mean_splat_present": False,
        "control_gaussian_prior_residual_splat_present": False,
        "control_gaussian_prior_heatmap_present": False,
        "control_gaussian_prior_scores_present": False,
        "control_gaussian_prior_metrics_present": False,
        "control_gaussian_prior_roi_metrics_independent_validation": None,
    }


def _artifact_path(
    value: object,
    *,
    subject_output_dir: Path,
    base_dir: Optional[Path] = None,
) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return (base_dir or subject_output_dir) / path


def _artifact_present(
    value: object,
    *,
    subject_output_dir: Path,
    base_dir: Optional[Path] = None,
) -> tuple[bool, Optional[Path]]:
    path = _artifact_path(value, subject_output_dir=subject_output_dir, base_dir=base_dir)
    if path is None:
        return False, None
    return path.is_file(), path


def _read_json_file(path: Optional[Path]) -> dict:
    if path is None or not path.is_file():
        return {}
    try:
        with path.open() as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _first_present(mapping_sequence: list[dict], *keys: str) -> object:
    for mapping in mapping_sequence:
        if not isinstance(mapping, dict):
            continue
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
    return None


def _extract_npz_state_schema(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as state:
            if "state_schema" not in state:
                return None
            return str(state["state_schema"])
    except (OSError, ValueError, KeyError):
        return None


def _extract_npz_scalar_strings(
    path: Optional[Path],
    keys: tuple[str, ...],
) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    try:
        with np.load(path, allow_pickle=False) as state:
            values: dict[str, str] = {}
            for key in keys:
                if key not in state:
                    continue
                value = state[key]
                if np.ndim(value) == 0:
                    values[key] = str(value)
                elif np.size(value) > 0:
                    values[key] = str(np.asarray(value).reshape(-1)[0])
            return values
    except (OSError, ValueError, KeyError):
        return {}


def _extract_paired_gaussian_evidence(
    manifest: dict,
    subject_output_dir: Path,
) -> dict[str, object]:
    """Extract evidence that paired Gaussian subtraction really ran.

    This checks durable artifacts instead of trusting residual scalar values.
    The paired 3DGS paper claim requires patient, normal, and residual splats
    on the same Gaussian support plus the quantitative Gaussian-state sidecar.
    """
    evidence = _empty_paired_gaussian_evidence()
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    paired_meta = manifest.get("paired_gaussian_residual_meta", {})
    if not isinstance(paired_meta, dict):
        paired_meta = {}

    metrics_present, metrics_path = _artifact_present(
        artifacts.get("paired_gaussian_residual_metrics"),
        subject_output_dir=subject_output_dir,
    )
    paired_sidecar = _read_json_file(metrics_path) if metrics_present else {}
    paired_sidecar_artifacts = paired_sidecar.get("artifacts", {})
    if not isinstance(paired_sidecar_artifacts, dict):
        paired_sidecar_artifacts = {}
    paired_meta_artifacts = paired_meta.get("artifacts", {})
    if not isinstance(paired_meta_artifacts, dict):
        paired_meta_artifacts = {}

    data_sources = [paired_meta, paired_sidecar]
    evidence["paired_gaussian_method"] = _first_present(data_sources, "method")
    evidence["paired_gaussian_same_support"] = _first_present(
        data_sources,
        "same_gaussian_support",
    )
    evidence["paired_gaussian_sampling_mode"] = _first_present(
        data_sources,
        "sampling_mode",
    )
    evidence["paired_gaussian_support_basis"] = _first_present(
        data_sources,
        "support_basis",
    )
    evidence["paired_gaussian_support_basis_mode"] = _first_present(
        data_sources,
        "support_basis_mode",
    )
    evidence["paired_gaussian_n_gaussians"] = _coerce_float(
        _first_present(data_sources, "n_gaussians")
    )

    schema = _first_present(data_sources, "gaussian_state_schema", "state_schema")

    metrics_base = metrics_path.parent if metrics_path is not None else subject_output_dir
    state_candidates = [
        (artifacts.get("paired_gaussian_state"), subject_output_dir),
        (artifacts.get("paired_gaussian_residual_state"), subject_output_dir),
        (paired_meta_artifacts.get("paired_gaussian_state"), subject_output_dir),
        (paired_sidecar_artifacts.get("paired_gaussian_state"), metrics_base),
    ]
    state_path: Optional[Path] = None
    for value, base_dir in state_candidates:
        present, path = _artifact_present(
            value,
            subject_output_dir=subject_output_dir,
            base_dir=base_dir,
        )
        if present:
            state_path = path
            evidence["paired_gaussian_state_present"] = True
            break
    state_scalar_meta = _extract_npz_scalar_strings(
        state_path,
        ("state_schema", "sampling_mode", "support_basis", "support_basis_mode"),
    )
    state_schema = state_scalar_meta.get("state_schema") or _extract_npz_state_schema(
        state_path
    )
    evidence["paired_gaussian_state_schema"] = state_schema or (
        str(schema) if schema is not None else None
    )
    if state_scalar_meta.get("sampling_mode"):
        evidence["paired_gaussian_sampling_mode"] = state_scalar_meta["sampling_mode"]
    if state_scalar_meta.get("support_basis"):
        evidence["paired_gaussian_support_basis"] = state_scalar_meta["support_basis"]
    if state_scalar_meta.get("support_basis_mode"):
        evidence["paired_gaussian_support_basis_mode"] = state_scalar_meta[
            "support_basis_mode"
        ]

    artifact_specs = [
        ("paired_patient_splat_present", "paired_patient_splat"),
        ("paired_normal_splat_present", "paired_normal_splat"),
        ("paired_residual_splat_present", "paired_gaussian_residual_splat"),
        ("paired_residual_heatmap_present", "paired_gaussian_residual_heatmap"),
    ]
    for output_key, artifact_key in artifact_specs:
        candidates = [
            (artifacts.get(artifact_key), subject_output_dir),
            (paired_meta_artifacts.get(artifact_key), subject_output_dir),
            (paired_sidecar_artifacts.get(artifact_key), metrics_base),
        ]
        evidence[output_key] = any(
            _artifact_present(
                value,
                subject_output_dir=subject_output_dir,
                base_dir=base_dir,
            )[0]
            for value, base_dir in candidates
        )

    return evidence


def _extract_control_gaussian_prior_evidence(
    manifest: dict,
    subject_output_dir: Path,
) -> dict[str, object]:
    """Extract evidence for patient-support healthy-control Gaussian residuals."""
    evidence = _empty_control_gaussian_prior_evidence()
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    control_meta = manifest.get("control_gaussian_prior_meta", {})
    if not isinstance(control_meta, dict):
        control_meta = {}

    metrics_present, metrics_path = _artifact_present(
        artifacts.get("control_gaussian_prior_metrics"),
        subject_output_dir=subject_output_dir,
    )
    evidence["control_gaussian_prior_metrics_present"] = metrics_present
    sidecar = _read_json_file(metrics_path) if metrics_present else {}
    sidecar_artifacts = sidecar.get("artifacts", {})
    if not isinstance(sidecar_artifacts, dict):
        sidecar_artifacts = {}
    meta_artifacts = control_meta.get("artifacts", {})
    if not isinstance(meta_artifacts, dict):
        meta_artifacts = {}

    data_sources = [control_meta, sidecar]
    evidence["control_gaussian_prior_method"] = _first_present(
        data_sources,
        "method",
    )
    evidence["control_gaussian_prior_same_support"] = _first_present(
        data_sources,
        "same_gaussian_support",
    )
    evidence["control_gaussian_prior_sampling_mode"] = _first_present(
        data_sources,
        "sampling_mode",
    )
    evidence["control_gaussian_prior_healthy_reference"] = _first_present(
        data_sources,
        "healthy_reference",
    )
    evidence["control_gaussian_prior_registration_method"] = _first_present(
        data_sources,
        "registration_method",
    )
    evidence["control_gaussian_prior_intensity_matching"] = _first_present(
        data_sources,
        "intensity_matching",
    )
    evidence["control_gaussian_prior_n_controls"] = _coerce_float(
        _first_present(data_sources, "n_controls")
    )
    evidence["control_gaussian_prior_min_controls"] = _coerce_float(
        _first_present(data_sources, "min_controls")
    )
    evidence["control_gaussian_prior_n_gaussians"] = _coerce_float(
        _first_present(data_sources, "n_gaussians")
    )
    evidence["control_gaussian_prior_roi_metrics_independent_validation"] = (
        _first_present(data_sources, "roi_metrics_independent_validation")
    )

    schema = _first_present(data_sources, "gaussian_state_schema", "state_schema")

    metrics_base = metrics_path.parent if metrics_path is not None else subject_output_dir
    state_candidates = [
        (artifacts.get("control_gaussian_prior_state"), subject_output_dir),
        (meta_artifacts.get("control_gaussian_prior_state"), subject_output_dir),
        (sidecar_artifacts.get("control_gaussian_prior_state"), metrics_base),
    ]
    state_path: Optional[Path] = None
    for value, base_dir in state_candidates:
        present, path = _artifact_present(
            value,
            subject_output_dir=subject_output_dir,
            base_dir=base_dir,
        )
        if present:
            state_path = path
            evidence["control_gaussian_prior_state_present"] = True
            break

    state_scalar_meta = _extract_npz_scalar_strings(
        state_path,
        ("state_schema", "method", "sampling_mode"),
    )
    state_schema = state_scalar_meta.get("state_schema") or _extract_npz_state_schema(
        state_path
    )
    evidence["control_gaussian_prior_state_schema"] = state_schema or (
        str(schema) if schema is not None else None
    )
    if state_scalar_meta.get("method"):
        evidence["control_gaussian_prior_method"] = state_scalar_meta["method"]
    if state_scalar_meta.get("sampling_mode"):
        evidence["control_gaussian_prior_sampling_mode"] = (
            state_scalar_meta["sampling_mode"]
        )

    artifact_specs = [
        ("control_gaussian_prior_patient_splat_present", "control_prior_patient_splat"),
        ("control_gaussian_prior_mean_splat_present", "control_prior_mean_splat"),
        ("control_gaussian_prior_residual_splat_present", "control_prior_z_residual_splat"),
        ("control_gaussian_prior_heatmap_present", "control_prior_z_residual_heatmap"),
        ("control_gaussian_prior_scores_present", "control_prior_z_scores"),
    ]
    for output_key, artifact_key in artifact_specs:
        candidates = [
            (artifacts.get(artifact_key), subject_output_dir),
            (meta_artifacts.get(artifact_key), subject_output_dir),
            (sidecar_artifacts.get(artifact_key), metrics_base),
        ]
        evidence[output_key] = any(
            _artifact_present(
                value,
                subject_output_dir=subject_output_dir,
                base_dir=base_dir,
            )[0]
            for value, base_dir in candidates
        )

    return evidence


def _extract_optimized_gaussian_evidence(
    manifest: dict,
    subject_output_dir: Path,
) -> dict[str, object]:
    """Extract evidence that optimized same-support Gaussian residual path ran.

    Similar to paired Gaussian extraction but for the optimized field variant.
    Checks for residual splat, heatmap, and metrics artifacts.
    """
    evidence = _empty_optimized_gaussian_evidence()
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    opt_meta = manifest.get("gaussian_residual_meta", {})
    if not isinstance(opt_meta, dict):
        opt_meta = {}

    metrics_present, metrics_path = _artifact_present(
        artifacts.get("gaussian_residual_metrics"),
        subject_output_dir=subject_output_dir,
    )
    metrics_sidecar = _read_json_file(metrics_path) if metrics_present else {}
    metrics_artifacts = metrics_sidecar.get("artifacts", {})
    if not isinstance(metrics_artifacts, dict):
        metrics_artifacts = {}
    meta_artifacts = opt_meta.get("artifacts", {})
    if not isinstance(meta_artifacts, dict):
        meta_artifacts = {}
    data_sources = [opt_meta, metrics_sidecar]

    evidence["optimized_gaussian_residual_present"] = bool(
        opt_meta or metrics_present or metrics_sidecar
    )
    evidence["optimized_gaussian_same_support"] = _coerce_bool(
        _first_present(data_sources, "same_gaussian_support", "same_support")
    )
    evidence["optimized_gaussian_subtraction_claim_allowed"] = _coerce_bool(
        _first_present(
            data_sources,
            "gaussian_subtraction_claim_allowed",
            "subtraction_claim_allowed",
        )
    )
    evidence["optimized_gaussian_score_method"] = _first_present(
        data_sources,
        "residual_score_method",
        "score_method",
    )
    evidence["optimized_gaussian_heatmap_method"] = _first_present(
        data_sources,
        "heatmap_method",
    )
    evidence["optimized_gaussian_n_patient"] = _coerce_float(
        _first_present(data_sources, "n_patient")
    )
    evidence["optimized_gaussian_n_normal"] = _coerce_float(
        _first_present(data_sources, "n_normal")
    )
    evidence["optimized_gaussian_means_norm_max_abs_diff"] = _coerce_float(
        _first_present(data_sources, "means_norm_max_abs_diff")
    )
    evidence["optimized_gaussian_log_scales_max_abs_diff"] = _coerce_float(
        _first_present(data_sources, "log_scales_max_abs_diff")
    )
    evidence["optimized_gaussian_quat_wxyz_max_abs_diff"] = _coerce_float(
        _first_present(data_sources, "quat_wxyz_max_abs_diff")
    )

    evidence["optimized_gaussian_residual_metrics_present"] = metrics_present

    artifact_specs = [
        ("optimized_gaussian_residual_splat_present", "gaussian_residual_splat"),
        ("optimized_gaussian_residual_heatmap_present", "gaussian_residual_heatmap"),
        ("optimized_gaussian_residual_scores_present", "gaussian_residual_scores"),
    ]
    metrics_base = metrics_path.parent if metrics_path is not None else subject_output_dir
    for output_key, artifact_key in artifact_specs:
        candidates = [
            (artifacts.get(artifact_key), subject_output_dir),
            (meta_artifacts.get(artifact_key), subject_output_dir),
            (metrics_artifacts.get(artifact_key), metrics_base),
        ]
        evidence[output_key] = any(
            _artifact_present(
                value,
                subject_output_dir=subject_output_dir,
                base_dir=base_dir,
            )[0]
            for value, base_dir in candidates
        )

    return evidence


def _coerce_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "pass", "ok"}:
            return True
        if normalized in {"false", "no", "0", "fail", "failed"}:
            return False
    return None


def _coerce_int(value: object, default: int = 0) -> int:
    number = _coerce_float(value)
    return int(number) if number is not None else default


def _empty_data_quality_evidence() -> dict[str, object]:
    return {
        "data_quality_present": False,
        "data_quality_stage_status": None,
        "data_quality_upload_readiness_status": None,
        "data_quality_can_run_3dgs": None,
        "data_quality_can_subtract_patient_normal": None,
        "data_quality_blocking_issue_count": 0,
        "data_quality_review_issue_count": 0,
        "data_quality_warning_count": 0,
        "data_quality_primary_modalities": "",
        "data_quality_t1w_to_flair_needs_resample": False,
        "data_quality_roi_to_flair_needs_resample": False,
        "data_quality_roi_resampled_to_base": False,
        "data_quality_has_nonfinite_blocker": False,
        "data_quality_dicom_conversion_status": None,
        "data_quality_dicom_conversion_can_run_3dgs": None,
        "data_quality_dicom_conversion_paper_claim_allowed": None,
        "data_quality_dicom_conversion_blocking_issue_count": 0,
        "data_quality_dicom_conversion_review_issue_count": 0,
    }


def _extract_data_quality_evidence(manifest: dict) -> dict[str, object]:
    """Extract MRI conversion/readiness evidence from a subject manifest."""
    evidence = _empty_data_quality_evidence()
    stage_status = manifest.get("stage_status", {})
    if isinstance(stage_status, dict):
        evidence["data_quality_stage_status"] = stage_status.get("data_quality")

    data_quality = manifest.get("data_quality", {})
    if not isinstance(data_quality, dict) or not data_quality:
        return evidence

    evidence["data_quality_present"] = True
    evidence["data_quality_warning_count"] = _coerce_int(
        data_quality.get("warning_count"),
        default=0,
    )

    readiness = data_quality.get("upload_readiness", {})
    if not isinstance(readiness, dict):
        readiness = {}
    evidence["data_quality_upload_readiness_status"] = (
        readiness.get("status") or data_quality.get("status")
    )
    evidence["data_quality_can_run_3dgs"] = _coerce_bool(
        readiness.get("can_run_3dgs")
    )
    evidence["data_quality_can_subtract_patient_normal"] = _coerce_bool(
        readiness.get("can_subtract_patient_normal")
    )
    evidence["data_quality_blocking_issue_count"] = _coerce_int(
        readiness.get("blocking_issue_count"),
        default=0,
    )
    evidence["data_quality_review_issue_count"] = _coerce_int(
        readiness.get("review_issue_count"),
        default=0,
    )
    primary_modalities = readiness.get("primary_modalities") or []
    if isinstance(primary_modalities, (list, tuple)):
        evidence["data_quality_primary_modalities"] = ",".join(
            str(v) for v in primary_modalities
        )
    else:
        evidence["data_quality_primary_modalities"] = str(primary_modalities)

    comparisons = data_quality.get("comparisons", {})
    if not isinstance(comparisons, dict):
        comparisons = {}
    for source_key, output_key in (
        ("t1w_to_flair", "data_quality_t1w_to_flair_needs_resample"),
        ("flair_roi_to_flair", "data_quality_roi_to_flair_needs_resample"),
    ):
        comparison = comparisons.get(source_key, {})
        if isinstance(comparison, dict):
            evidence[output_key] = bool(
                comparison.get("needs_resample_to_reference")
            )

    roi_processing = data_quality.get("roi_processing", {})
    if isinstance(roi_processing, dict):
        evidence["data_quality_roi_resampled_to_base"] = bool(
            roi_processing.get("resampled_to_base")
        )

    blockers = readiness.get("blockers", [])
    if isinstance(blockers, (list, tuple)):
        blocker_text = " ".join(str(v).lower() for v in blockers)
        evidence["data_quality_has_nonfinite_blocker"] = "nonfinite" in blocker_text

    dicom_safety = data_quality.get("dicom_conversion_safety")
    if not isinstance(dicom_safety, dict):
        upload_meta = manifest.get("upload_meta", {})
        if isinstance(upload_meta, dict):
            dicom_safety = upload_meta.get("dicom_conversion_safety")
    if isinstance(dicom_safety, dict) and dicom_safety:
        status = dicom_safety.get("status")
        if status and status != "not_applicable":
            evidence["data_quality_dicom_conversion_status"] = status
            evidence["data_quality_dicom_conversion_can_run_3dgs"] = _coerce_bool(
                dicom_safety.get("can_use_converted_inputs_for_3dgs")
            )
            evidence["data_quality_dicom_conversion_paper_claim_allowed"] = (
                _coerce_bool(dicom_safety.get("can_use_converted_inputs_for_paper_claim"))
            )
            evidence["data_quality_dicom_conversion_blocking_issue_count"] = _coerce_int(
                dicom_safety.get("blocking_issue_count"),
                default=0,
            )
            evidence["data_quality_dicom_conversion_review_issue_count"] = _coerce_int(
                dicom_safety.get("review_issue_count"),
                default=0,
            )

    return evidence


def _empty_splat_reconstruction_evidence() -> dict[str, object]:
    return {
        "splat_reconstruction_audit_present": False,
        "splat_reconstruction_audited_artifact": None,
        "splat_reconstruction_kernel_mode": None,
        "splat_reconstruction_readiness_status": None,
        "splat_reconstruction_blocker_count": 0,
        "splat_reconstruction_warning_count": 0,
        "splat_mri_to_3dgs_claim_allowed": None,
        "splat_mri_to_3dgs_claim_class": None,
    }


def _extract_splat_reconstruction_evidence(manifest: dict) -> dict[str, object]:
    """Extract artifact-level MRI-to-3DGS readiness evidence."""
    evidence = _empty_splat_reconstruction_evidence()
    meta = manifest.get("splat_reconstruction_audit_meta", {})
    if not isinstance(meta, dict) or not meta:
        return evidence

    evidence["splat_reconstruction_audit_present"] = True
    evidence["splat_reconstruction_audited_artifact"] = meta.get("audited_artifact")
    evidence["splat_reconstruction_kernel_mode"] = meta.get("kernel_mode")
    gate = meta.get("representation_readiness_gate", {})
    if not isinstance(gate, dict):
        gate = {}
    evidence["splat_reconstruction_readiness_status"] = (
        meta.get("mri_to_3dgs_readiness_status")
        or gate.get("status")
    )
    evidence["splat_reconstruction_blocker_count"] = _coerce_int(
        meta.get("mri_to_3dgs_readiness_blocker_count")
        if meta.get("mri_to_3dgs_readiness_blocker_count") is not None
        else gate.get("blocker_count"),
        default=0,
    )
    evidence["splat_reconstruction_warning_count"] = _coerce_int(
        meta.get("mri_to_3dgs_readiness_warning_count")
        if meta.get("mri_to_3dgs_readiness_warning_count") is not None
        else gate.get("warning_count"),
        default=0,
    )
    evidence["splat_mri_to_3dgs_claim_allowed"] = _coerce_bool(
        meta.get("mri_to_3dgs_claim_allowed")
        if meta.get("mri_to_3dgs_claim_allowed") is not None
        else gate.get("mri_to_3dgs_claim_allowed")
    )
    evidence["splat_mri_to_3dgs_claim_class"] = (
        meta.get("mri_to_3dgs_claim_class")
        or gate.get("claim_class")
    )
    return evidence


def _empty_optimized_gaussian_field_audit_evidence() -> dict[str, object]:
    return {
        "optimized_gaussian_field_audit_present": False,
        "optimized_gaussian_field_audit_status": None,
        "optimized_mri_to_3dgs_claim_allowed": None,
        "optimized_gaussian_field_audit_blocker_count": 0,
        "optimized_gaussian_field_audit_warning_count": 0,
        "optimized_gaussian_field_audit_sample_count": None,
        "optimized_gaussian_field_foreground_mae": None,
        "optimized_gaussian_field_foreground_rmse": None,
        "optimized_gaussian_field_all_mae": None,
        "optimized_gaussian_field_all_rmse": None,
        "optimized_gaussian_field_background_mae": None,
        "optimized_gaussian_field_training_objective": None,
        "optimized_gaussian_field_volume_loss_weight": None,
        "optimized_gaussian_field_volume_sample_count": None,
        "optimized_gaussian_field_affine_mismatch": None,
        "optimized_gaussian_field_affine_missing": None,
    }


def _extract_optimized_gaussian_field_audit_evidence(
    manifest: dict,
    subject_output_dir: Path,
) -> dict[str, object]:
    """Extract held-out optimized Gaussian-field MRI fidelity evidence."""
    evidence = _empty_optimized_gaussian_field_audit_evidence()
    meta = manifest.get("optimized_gaussian_field_audit_meta", {})
    if not isinstance(meta, dict) or not meta:
        artifacts = manifest.get("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}
        present, path = _artifact_present(
            artifacts.get("optimized_gaussian_field_audit"),
            subject_output_dir=subject_output_dir,
        )
        meta = _read_json_file(path) if present else {}
    if not isinstance(meta, dict) or not meta:
        return evidence

    evidence["optimized_gaussian_field_audit_present"] = True
    evidence["optimized_gaussian_field_audit_sample_count"] = _coerce_float(
        meta.get("sample_count")
    )
    evidence["optimized_gaussian_field_affine_mismatch"] = _coerce_bool(
        meta.get("affine_mismatch")
    )
    evidence["optimized_gaussian_field_affine_missing"] = _coerce_bool(
        meta.get("affine_missing")
    )

    gate = meta.get("readiness_gate", {})
    if not isinstance(gate, dict):
        gate = {}
    blockers = gate.get("blockers", [])
    warnings = gate.get("warnings", [])
    evidence["optimized_gaussian_field_audit_status"] = gate.get("status")
    evidence["optimized_mri_to_3dgs_claim_allowed"] = _coerce_bool(
        gate.get("optimized_mri_to_3dgs_claim_allowed")
    )
    evidence["optimized_gaussian_field_audit_blocker_count"] = (
        len(blockers) if isinstance(blockers, (list, tuple))
        else _coerce_int(gate.get("blocker_count"), default=0)
    )
    evidence["optimized_gaussian_field_audit_warning_count"] = (
        len(warnings) if isinstance(warnings, (list, tuple))
        else _coerce_int(gate.get("warning_count"), default=0)
    )

    metrics = meta.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    for source_key, output_prefix in (
        ("foreground", "optimized_gaussian_field_foreground"),
        ("all", "optimized_gaussian_field_all"),
        ("background", "optimized_gaussian_field_background"),
    ):
        source = metrics.get(source_key, {})
        if not isinstance(source, dict):
            continue
        if f"{output_prefix}_mae" in evidence:
            evidence[f"{output_prefix}_mae"] = _coerce_float(source.get("mae"))
        if f"{output_prefix}_rmse" in evidence:
            evidence[f"{output_prefix}_rmse"] = _coerce_float(source.get("rmse"))

    checkpoint = meta.get("checkpoint_metadata", {})
    if not isinstance(checkpoint, dict):
        checkpoint = {}
    evidence["optimized_gaussian_field_training_objective"] = checkpoint.get(
        "training_objective"
    )
    evidence["optimized_gaussian_field_volume_loss_weight"] = _coerce_float(
        checkpoint.get("volume_loss_weight")
    )
    evidence["optimized_gaussian_field_volume_sample_count"] = _coerce_float(
        checkpoint.get("volume_sample_count")
    )
    return evidence


def _empty_synthetic_normal_audit_evidence() -> dict[str, object]:
    return {
        "synthetic_normal_audit_present": False,
        "synthetic_normal_counterfactual_class": None,
        "synthetic_normal_claim_boundary": None,
        "synthetic_normal_claim_class": None,
        "synthetic_normal_subtraction_readiness_status": None,
        "synthetic_normal_subtraction_blocker_count": 0,
        "synthetic_normal_subtraction_warning_count": 0,
        "synthetic_normal_gaussian_subtraction_claim_allowed": None,
        "synthetic_normal_patient_conditioned_inpainting": None,
        "synthetic_normal_patient_conditioned_counterfactual_claim_allowed": None,
        "synthetic_normal_same_grid_before_resample": None,
        "synthetic_normal_resampled_to_patient_grid": None,
        "synthetic_normal_outside_roi_similarity_1_minus_mae": None,
        "synthetic_normal_inside_to_outside_mae_ratio": None,
    }


def _extract_synthetic_normal_audit_evidence(manifest: dict) -> dict[str, object]:
    """Extract healthy-reference audit and claim-boundary evidence."""
    evidence = _empty_synthetic_normal_audit_evidence()
    meta = manifest.get("synthetic_normal_audit_meta", {})
    gaussian_residual_meta = manifest.get("gaussian_residual_meta", {})
    if not isinstance(gaussian_residual_meta, dict):
        gaussian_residual_meta = {}
    if not isinstance(meta, dict) or not meta:
        evidence["synthetic_normal_gaussian_subtraction_claim_allowed"] = _coerce_bool(
            gaussian_residual_meta.get("gaussian_subtraction_claim_allowed")
        )
        return evidence

    evidence["synthetic_normal_audit_present"] = True
    evidence["synthetic_normal_counterfactual_class"] = meta.get("counterfactual_class")
    evidence["synthetic_normal_claim_boundary"] = meta.get("claim_boundary")
    evidence["synthetic_normal_claim_class"] = meta.get("synthetic_normal_claim_class")
    evidence["synthetic_normal_subtraction_readiness_status"] = meta.get(
        "subtraction_readiness_status"
    )
    evidence["synthetic_normal_subtraction_blocker_count"] = _coerce_int(
        meta.get("subtraction_readiness_blocker_count"),
        default=0,
    )
    evidence["synthetic_normal_subtraction_warning_count"] = _coerce_int(
        meta.get("subtraction_readiness_warning_count"),
        default=0,
    )
    evidence["synthetic_normal_gaussian_subtraction_claim_allowed"] = _coerce_bool(
        meta.get("gaussian_subtraction_claim_allowed")
    )
    if evidence["synthetic_normal_gaussian_subtraction_claim_allowed"] is None:
        evidence["synthetic_normal_gaussian_subtraction_claim_allowed"] = _coerce_bool(
            gaussian_residual_meta.get("gaussian_subtraction_claim_allowed")
        )
    evidence["synthetic_normal_patient_conditioned_inpainting"] = _coerce_bool(
        meta.get("patient_conditioned_inpainting")
    )
    evidence["synthetic_normal_patient_conditioned_counterfactual_claim_allowed"] = (
        _coerce_bool(meta.get("patient_conditioned_counterfactual_claim_allowed"))
    )
    evidence["synthetic_normal_same_grid_before_resample"] = _coerce_bool(
        meta.get("same_grid_before_resample")
    )
    evidence["synthetic_normal_resampled_to_patient_grid"] = _coerce_bool(
        meta.get("synthetic_resampled_to_patient_grid")
    )
    evidence["synthetic_normal_outside_roi_similarity_1_minus_mae"] = _coerce_float(
        meta.get("outside_roi_similarity_1_minus_mae")
    )
    evidence["synthetic_normal_inside_to_outside_mae_ratio"] = _coerce_float(
        meta.get("inside_to_outside_mae_ratio")
    )
    return evidence


def _extract_control_prior_gate(manifest: dict) -> Optional[str]:
    control_prior_meta = manifest.get("control_prior_meta", {})
    if not isinstance(control_prior_meta, dict):
        return None
    gate = control_prior_meta.get("control_prior_quality_gate")
    if gate is None:
        return None
    return str(gate)


def _normalize_roi_source(value: object) -> str:
    normalized = str(value or "unknown").strip().lower().replace("_", "-")
    if normalized in {"provided", "manual", "manual-roi", "ground-truth"}:
        return "provided"
    if normalized in {"candidate", "auto", "automatic", "auto-candidate", "heuristic"}:
        return "candidate"
    if normalized in {"supervised", "model", "predicted", "segmentation-model"}:
        return "supervised"
    if normalized in {"none", "missing", "no-roi"}:
        return "none"
    return "unknown"


def _roi_source_summary(rows: list[dict]) -> dict[str, object]:
    counts_by_group: dict[str, dict[str, int]] = {}
    known_rows = [r for r in rows if r["group"] in {"FCD", "HC"}]
    for row in known_rows:
        group = row["group"]
        source = row.get("roi_source_normalized", "unknown")
        counts_by_group.setdefault(group, {})
        counts_by_group[group][source] = counts_by_group[group].get(source, 0) + 1

    roi_free_count = sum(
        1 for row in known_rows
        if row.get("roi_source_normalized") in ROI_FREE_SOURCES
    )
    oracle_fcd_count = sum(
        1 for row in known_rows
        if row["group"] == "FCD"
        and row.get("roi_source_normalized") in ORACLE_ROI_SOURCES
    )
    unknown_count = sum(
        1 for row in known_rows
        if row.get("roi_source_normalized") == "unknown"
    )
    none_count = sum(
        1 for row in known_rows
        if row.get("roi_source_normalized") == "none"
    )
    return {
        "counts_by_group": counts_by_group,
        "n_known_subjects": len(known_rows),
        "n_roi_free_subjects": roi_free_count,
        "n_oracle_fcd_subjects": oracle_fcd_count,
        "n_unknown_roi_source": unknown_count,
        "n_no_roi_source": none_count,
        "all_known_subjects_roi_free": (
            len(known_rows) > 0 and roi_free_count == len(known_rows)
        ),
    }


def _manual_candidate_validation_allowed(manifest: dict, group: str) -> bool:
    """Return whether residual-candidate manual metrics are valid evidence.

    Manual-overlap values only prove lesion localization when they compare an
    FCD subject's candidate mask against an independent provided/manual ROI.
    Generated candidate/supervised masks must not become their own validation
    target, provided/manual fuzzing ROIs must not be reused as validation, and
    healthy controls have no FCD lesion ROI by definition.
    """
    if group != "FCD":
        return False
    candidate_meta = manifest.get("residual_candidate_meta", {})
    if not isinstance(candidate_meta, dict):
        return True
    if candidate_meta.get("manual_roi_available") is False:
        return False
    source = candidate_meta.get("manual_roi_source")
    source_is_oracle = (
        True if source is None
        else _normalize_roi_source(source) in ORACLE_ROI_SOURCES
    )
    if not source_is_oracle:
        return False

    explicit_independence = candidate_meta.get(
        "manual_overlap_independent_from_fuzzing_roi"
    )
    if explicit_independence is False:
        return False

    selected_source = candidate_meta.get("selected_roi_source")
    if selected_source is not None:
        selected_normalized = _normalize_roi_source(selected_source)
        if selected_normalized in ORACLE_ROI_SOURCES:
            return False
        if selected_normalized == "unknown":
            return explicit_independence is True

    if explicit_independence is True:
        return True

    return True


def _drop_untrusted_manual_candidate_metrics(row: dict) -> None:
    for key in RESIDUAL_CANDIDATE_MANUAL_METRIC_KEYS:
        row[key] = None


def _gate_status(
    *,
    evidence_present: bool,
    passed: bool,
) -> str:
    if not evidence_present:
        return "insufficient_evidence"
    return "pass" if passed else "fail"


def _build_paper_readiness_gates(
    *,
    n_fcd: int,
    n_hc: int,
    n_scored: int,
    auroc: float,
    score_metric: str,
    roi_source_summary: dict[str, object],
    data_quality_summary: dict[str, object],
    candidate_validation: dict,
    splat_reconstruction_quality: dict,
    optimized_gaussian_field_quality: dict,
    paired_gaussian_subtraction_quality: dict,
    control_gaussian_prior_quality: dict,
    optimized_gaussian_subtraction_quality: dict,
    control_prior_quality: dict,
    synthetic_normal_quality: dict,
    thresholds: dict[str, float],
) -> dict[str, object]:
    """Build explicit paper-readiness gates from cohort metrics.

    The gates are intentionally conservative: missing evidence is not a pass.
    Thresholds are recorded alongside the observed values so reports remain
    interpretable when criteria change between pilot and paper runs.
    """
    expected_fcd = int(thresholds["expected_fcd"])
    expected_hc = int(thresholds["expected_hc"])
    expected_total = expected_fcd + expected_hc
    n_total_known = n_fcd + n_hc

    cohort_size_pass = n_fcd >= expected_fcd and n_hc >= expected_hc
    scoring_complete = n_scored >= n_total_known and n_total_known > 0

    data_quality_n = int(data_quality_summary.get("n_with_data_quality") or 0)
    data_quality_fail_n = int(data_quality_summary.get("n_upload_readiness_fail") or 0)
    data_quality_can_run_n = int(data_quality_summary.get("n_can_run_3dgs") or 0)
    data_quality_can_subtract_n = int(
        data_quality_summary.get("n_can_subtract_patient_normal") or 0
    )
    data_quality_blockers = int(data_quality_summary.get("total_blocking_issue_count") or 0)
    dicom_converted_n = int(data_quality_summary.get("n_with_dicom_conversion_safety") or 0)
    dicom_conversion_fail_n = int(data_quality_summary.get("n_dicom_conversion_fail") or 0)
    dicom_conversion_blockers = int(
        data_quality_summary.get("total_dicom_conversion_blocking_issue_count") or 0
    )
    dicom_conversion_claim_n = int(
        data_quality_summary.get("n_dicom_conversion_paper_claim_allowed") or 0
    )
    data_quality_evidence = n_total_known > 0 and data_quality_n >= n_total_known
    data_quality_pass = (
        data_quality_evidence
        and data_quality_fail_n == 0
        and data_quality_blockers == 0
        and data_quality_can_run_n >= n_total_known
        and data_quality_can_subtract_n >= n_total_known
        and dicom_conversion_fail_n == 0
        and dicom_conversion_blockers == 0
        and dicom_conversion_claim_n >= dicom_converted_n
    )

    splat_n = int(splat_reconstruction_quality.get("n_with_splat_audit") or 0)
    splat_readiness_n = int(
        splat_reconstruction_quality.get("n_with_splat_readiness_audit") or 0
    )
    splat_readiness_fail_n = int(
        splat_reconstruction_quality.get("n_readiness_fail") or 0
    )
    splat_readiness_review_n = int(
        splat_reconstruction_quality.get("n_readiness_review") or 0
    )
    splat_readiness_blockers = int(
        splat_reconstruction_quality.get("total_readiness_blocker_count") or 0
    )
    splat_claim_allowed_n = int(
        splat_reconstruction_quality.get("n_mri_to_3dgs_claim_allowed") or 0
    )
    splat_coverage = _coerce_float(
        splat_reconstruction_quality.get("mean_foreground_coverage")
    )
    splat_mip_mae = _coerce_float(splat_reconstruction_quality.get("mean_mip_mae"))
    direct_splat_evidence = (
        splat_n >= n_total_known
        and splat_readiness_n >= n_total_known
        and n_total_known > 0
        and splat_coverage is not None
        and splat_mip_mae is not None
    )
    opt_field_audit_n = int(
        optimized_gaussian_field_quality.get("n_with_optimized_field_audit") or 0
    )
    opt_field_pass_n = int(
        optimized_gaussian_field_quality.get("n_readiness_pass") or 0
    )
    opt_field_review_n = int(
        optimized_gaussian_field_quality.get("n_readiness_review") or 0
    )
    opt_field_fail_n = int(
        optimized_gaussian_field_quality.get("n_readiness_fail") or 0
    )
    opt_field_blockers = int(
        optimized_gaussian_field_quality.get("total_readiness_blocker_count") or 0
    )
    opt_field_claim_allowed_n = int(
        optimized_gaussian_field_quality.get("n_optimized_mri_to_3dgs_claim_allowed")
        or 0
    )
    opt_field_volume_anchored_n = int(
        optimized_gaussian_field_quality.get("n_volume_anchored_training") or 0
    )
    opt_field_affine_mismatch_n = int(
        optimized_gaussian_field_quality.get("n_affine_mismatch") or 0
    )
    opt_field_affine_missing_n = int(
        optimized_gaussian_field_quality.get("n_affine_missing") or 0
    )
    opt_field_fg_mae = _coerce_float(
        optimized_gaussian_field_quality.get("mean_foreground_mae")
    )
    opt_field_all_mae = _coerce_float(
        optimized_gaussian_field_quality.get("mean_all_mae")
    )
    optimized_field_evidence = (
        opt_field_audit_n >= n_total_known
        and n_total_known > 0
        and opt_field_fg_mae is not None
        and opt_field_all_mae is not None
    )
    splat_evidence = direct_splat_evidence or optimized_field_evidence
    splat_pass = (
        direct_splat_evidence
        and optimized_field_evidence
        and splat_readiness_fail_n == 0
        and splat_readiness_blockers == 0
        and splat_claim_allowed_n >= n_total_known
        and splat_coverage >= thresholds["splat_foreground_coverage_min"]
        and splat_mip_mae <= thresholds["splat_mip_mean_mae_max"]
        and opt_field_pass_n >= n_total_known
        and opt_field_review_n == 0
        and opt_field_fail_n == 0
        and opt_field_blockers == 0
        and opt_field_claim_allowed_n >= n_total_known
        and opt_field_volume_anchored_n >= n_total_known
        and opt_field_affine_mismatch_n == 0
        and opt_field_affine_missing_n == 0
        and opt_field_fg_mae <= thresholds["optimized_field_foreground_mae_max"]
        and opt_field_all_mae <= thresholds["optimized_field_all_mae_max"]
    )

    paired_state_n = int(
        paired_gaussian_subtraction_quality.get("n_with_paired_gaussian_state") or 0
    )
    paired_same_support_n = int(
        paired_gaussian_subtraction_quality.get("n_same_gaussian_support") or 0
    )
    paired_schema_n = int(
        paired_gaussian_subtraction_quality.get("n_valid_state_schema") or 0
    )
    paired_volume_support_n = int(
        paired_gaussian_subtraction_quality.get("n_volume_sampling_mode") or 0
    )
    paired_expected_basis_n = int(
        paired_gaussian_subtraction_quality.get("n_expected_support_basis") or 0
    )
    paired_unique_support_basis_n = int(
        paired_gaussian_subtraction_quality.get("n_unique_supported_support_basis")
        or 0
    )
    paired_splat_n = int(
        paired_gaussian_subtraction_quality.get(
            "n_with_patient_normal_residual_splats"
        )
        or 0
    )
    paired_min_n_gaussians = _coerce_float(
        paired_gaussian_subtraction_quality.get("min_n_gaussians")
    )
    paired_evidence = n_total_known > 0 and paired_state_n > 0
    paired_pass = (
        paired_evidence
        and paired_state_n >= n_total_known
        and paired_same_support_n >= n_total_known
        and paired_schema_n >= n_total_known
        and paired_volume_support_n >= n_total_known
        and paired_expected_basis_n >= n_total_known
        and paired_unique_support_basis_n == 1
        and paired_splat_n >= n_total_known
        and paired_min_n_gaussians is not None
        and paired_min_n_gaussians >= thresholds["paired_gaussian_min_points"]
    )

    control_gp_state_n = int(
        control_gaussian_prior_quality.get("n_with_control_gaussian_prior_state")
        or 0
    )
    control_gp_same_support_n = int(
        control_gaussian_prior_quality.get("n_same_gaussian_support") or 0
    )
    control_gp_schema_n = int(
        control_gaussian_prior_quality.get("n_valid_state_schema") or 0
    )
    control_gp_volume_support_n = int(
        control_gaussian_prior_quality.get("n_volume_sampling_mode") or 0
    )
    control_gp_splat_n = int(
        control_gaussian_prior_quality.get(
            "n_with_patient_prior_residual_splats"
        )
        or 0
    )
    control_gp_heatmap_n = int(
        control_gaussian_prior_quality.get("n_with_residual_heatmap") or 0
    )
    control_gp_scores_n = int(
        control_gaussian_prior_quality.get("n_with_residual_scores") or 0
    )
    control_gp_metrics_n = int(
        control_gaussian_prior_quality.get("n_with_residual_metrics") or 0
    )
    control_gp_min_n_gaussians = _coerce_float(
        control_gaussian_prior_quality.get("min_n_gaussians")
    )
    control_gp_min_n_controls = _coerce_float(
        control_gaussian_prior_quality.get("min_n_controls")
    )
    control_gp_enough_controls_n = int(
        control_gaussian_prior_quality.get("n_with_minimum_controls") or 0
    )
    control_gp_evidence = n_total_known > 0 and control_gp_state_n > 0
    control_gp_pass = (
        control_gp_evidence
        and control_gp_state_n >= n_total_known
        and control_gp_same_support_n >= n_total_known
        and control_gp_schema_n >= n_total_known
        and control_gp_volume_support_n >= n_total_known
        and control_gp_splat_n >= n_total_known
        and control_gp_heatmap_n >= n_total_known
        and control_gp_scores_n >= n_total_known
        and control_gp_metrics_n >= n_total_known
        and control_gp_enough_controls_n >= n_total_known
        and control_gp_min_n_controls is not None
        and control_gp_min_n_controls >= thresholds["control_gaussian_min_controls"]
        and control_gp_min_n_gaussians is not None
        and control_gp_min_n_gaussians >= thresholds["control_gaussian_min_points"]
    )

    optimized_residual_n = int(
        optimized_gaussian_subtraction_quality.get(
            "n_with_optimized_gaussian_residual"
        )
        or 0
    )
    optimized_same_support_n = int(
        optimized_gaussian_subtraction_quality.get("n_same_gaussian_support") or 0
    )
    optimized_claim_allowed_n = int(
        optimized_gaussian_subtraction_quality.get("n_gaussian_subtraction_claim_allowed")
        or 0
    )
    optimized_indexwise_n = int(
        optimized_gaussian_subtraction_quality.get("n_indexwise_score_method") or 0
    )
    optimized_field_heatmap_n = int(
        optimized_gaussian_subtraction_quality.get("n_field_heatmap_method") or 0
    )
    optimized_splat_n = int(
        optimized_gaussian_subtraction_quality.get("n_with_residual_splat") or 0
    )
    optimized_heatmap_n = int(
        optimized_gaussian_subtraction_quality.get("n_with_residual_heatmap") or 0
    )
    optimized_metrics_n = int(
        optimized_gaussian_subtraction_quality.get("n_with_residual_metrics") or 0
    )
    optimized_min_n_patient = _coerce_float(
        optimized_gaussian_subtraction_quality.get("min_n_patient_gaussians")
    )
    optimized_min_n_normal = _coerce_float(
        optimized_gaussian_subtraction_quality.get("min_n_normal_gaussians")
    )
    optimized_evidence = n_total_known > 0 and optimized_residual_n > 0
    optimized_pass = (
        optimized_evidence
        and optimized_residual_n >= n_total_known
        and optimized_same_support_n >= n_total_known
        and optimized_claim_allowed_n >= n_total_known
        and optimized_indexwise_n >= n_total_known
        and optimized_field_heatmap_n >= n_total_known
        and optimized_splat_n >= n_total_known
        and optimized_heatmap_n >= n_total_known
        and optimized_metrics_n >= n_total_known
        and optimized_min_n_patient is not None
        and optimized_min_n_patient >= thresholds["optimized_gaussian_min_points"]
        and optimized_min_n_normal is not None
        and optimized_min_n_normal >= thresholds["optimized_gaussian_min_points"]
    )

    localization_n = int(candidate_validation.get("n_with_manual_roi_overlap") or 0)
    localization_mean = _coerce_float(candidate_validation.get("mean_manual_coverage"))
    localization_evidence = localization_n >= n_fcd and n_fcd > 0 and localization_mean is not None
    localization_pass = (
        localization_evidence
        and localization_mean >= thresholds["localization_mean_manual_coverage_min"]
    )

    hc_burden = _coerce_float(candidate_validation.get("mean_candidate_foreground_fraction_hc"))
    hc_evidence = n_hc > 0 and hc_burden is not None
    hc_pass = (
        hc_evidence
        and hc_burden <= thresholds["hc_candidate_foreground_fraction_max"]
    )

    detection_evidence = n_fcd > 0 and n_hc > 0 and n_scored >= n_total_known
    detection_pass = detection_evidence and auroc >= thresholds["auroc_min"]

    roi_source_evidence = n_total_known > 0
    roi_free_pass = (
        roi_source_evidence
        and bool(roi_source_summary.get("all_known_subjects_roi_free"))
    )

    control_prior_subjects = int(control_prior_quality.get("n_with_control_prior_quality") or 0)
    control_prior_pass_count = int(control_prior_quality.get("n_quality_gate_pass") or 0)
    control_prior_review_count = int(control_prior_quality.get("n_quality_gate_review") or 0)
    control_prior_insufficient_count = int(
        control_prior_quality.get("n_quality_gate_insufficient_reference") or 0
    )
    control_prior_similarity = _coerce_float(
        control_prior_quality.get("mean_reference_similarity_1_minus_mae")
    )
    synthetic_audit_n = int(synthetic_normal_quality.get("n_with_synthetic_normal_audit") or 0)
    synthetic_readiness_pass_n = int(
        synthetic_normal_quality.get("n_subtraction_readiness_pass") or 0
    )
    synthetic_readiness_review_n = int(
        synthetic_normal_quality.get("n_subtraction_readiness_review") or 0
    )
    synthetic_readiness_fail_n = int(
        synthetic_normal_quality.get("n_subtraction_readiness_fail") or 0
    )
    synthetic_blockers = int(
        synthetic_normal_quality.get("total_subtraction_blocker_count") or 0
    )
    synthetic_gaussian_allowed_n = int(
        synthetic_normal_quality.get("n_gaussian_subtraction_claim_allowed") or 0
    )
    synthetic_patient_inpainting_n = int(
        synthetic_normal_quality.get("n_patient_conditioned_inpainting") or 0
    )
    synthetic_patient_counterfactual_allowed_n = int(
        synthetic_normal_quality.get(
            "n_patient_conditioned_counterfactual_claim_allowed"
        )
        or 0
    )
    synthetic_audit_evidence = (
        n_total_known > 0 and synthetic_audit_n >= n_total_known
    )
    synthetic_audit_pass = (
        synthetic_audit_evidence
        and synthetic_readiness_fail_n == 0
        and synthetic_blockers == 0
        and synthetic_gaussian_allowed_n >= n_total_known
    )
    control_prior_evidence = control_prior_subjects > 0
    control_prior_required_pass = (
        not control_prior_evidence
        or (
            control_prior_similarity is not None
            and control_prior_pass_count == control_prior_subjects
            and control_prior_review_count == 0
            and control_prior_insufficient_count == 0
            and control_prior_similarity >= thresholds["control_prior_similarity_min"]
        )
    )
    synthesis_evidence = synthetic_audit_evidence and (
        not control_prior_evidence or control_prior_similarity is not None
    )
    synthesis_pass = (
        synthesis_evidence
        and synthetic_audit_pass
        and control_prior_required_pass
    )

    gates = {
        "cohort_size": {
            "status": _gate_status(evidence_present=n_total_known > 0, passed=cohort_size_pass),
            "observed": {"n_fcd": n_fcd, "n_hc": n_hc, "n_total_known": n_total_known},
            "threshold": {
                "expected_fcd": expected_fcd,
                "expected_hc": expected_hc,
                "expected_total": expected_total,
            },
            "interpretation": "Full ds004199-scale evidence requires all expected FCD and HC subjects.",
        },
        "scoring_completeness": {
            "status": _gate_status(evidence_present=n_total_known > 0, passed=scoring_complete),
            "observed": {"n_scored": n_scored, "n_total_known": n_total_known},
            "threshold": {"all_known_subjects_scored": True},
            "interpretation": "Detection AUROC should use every known FCD/HC subject, not a surviving subset.",
        },
        "data_quality_conversion": {
            "status": _gate_status(
                evidence_present=data_quality_evidence,
                passed=bool(data_quality_pass),
            ),
            "observed": data_quality_summary,
            "threshold": {
                "all_known_subjects_have_data_quality_audit": True,
                "upload_readiness_fail_allowed": 0,
                "blocking_issue_count_allowed": 0,
                "all_known_subjects_can_run_3dgs": True,
                "all_known_subjects_can_subtract_patient_normal": True,
                "review_status_allowed_when_processible": True,
                "dicom_conversion_fail_allowed": 0,
                "dicom_conversion_blocking_issue_count_allowed": 0,
                "all_dicom_converted_subjects_allow_paper_claim": True,
            },
            "interpretation": (
                "MRI-to-3DGS subtraction requires audited canonical NIfTI "
                "geometry, finite primary images, and explicit handling of "
                "resampling/reorientation risks. Uploaded DICOM conversions "
                "also require a sidecar safety gate that rules out wrong-series "
                "inputs before paper-facing claims."
            ),
        },
        "detection_auroc": {
            "status": _gate_status(evidence_present=detection_evidence, passed=detection_pass),
            "observed": {"auroc_fcd_vs_hc": auroc, "score_metric": score_metric},
            "threshold": {"auroc_min": thresholds["auroc_min"]},
            "interpretation": "Subject-level residual score must separate FCD from healthy controls.",
        },
        "roi_source_provenance": {
            "status": _gate_status(evidence_present=roi_source_evidence, passed=roi_free_pass),
            "observed": roi_source_summary,
            "threshold": {
                "all_known_subjects_roi_free": True,
                "manual_or_provided_fcd_subjects_allowed": 0,
            },
            "interpretation": (
                "Uploaded-MRI detection claims require lesion candidates generated "
                "without manual/provided FCD ROI masks."
            ),
        },
        "healthy_reference_synthesis": {
            "status": _gate_status(
                evidence_present=synthesis_evidence,
                passed=bool(synthesis_pass),
            ),
            "observed": {
                "n_known_subjects": n_total_known,
                "n_with_synthetic_normal_audit": synthetic_audit_n,
                "n_subtraction_readiness_pass": synthetic_readiness_pass_n,
                "n_subtraction_readiness_review": synthetic_readiness_review_n,
                "n_subtraction_readiness_fail": synthetic_readiness_fail_n,
                "total_subtraction_blocker_count": synthetic_blockers,
                "n_gaussian_subtraction_claim_allowed": synthetic_gaussian_allowed_n,
                "n_with_control_prior_quality": control_prior_subjects,
                "n_quality_gate_pass": control_prior_pass_count,
                "n_quality_gate_review": control_prior_review_count,
                "n_quality_gate_insufficient_reference": control_prior_insufficient_count,
                "mean_reference_similarity_1_minus_mae": control_prior_similarity,
                "counterfactual_class_counts": synthetic_normal_quality.get(
                    "counterfactual_class_counts",
                    {},
                ),
                "claim_class_counts": synthetic_normal_quality.get(
                    "claim_class_counts",
                    {},
                ),
            },
            "threshold": {
                "all_known_subjects_have_synthetic_normal_audit": True,
                "subtraction_readiness_fail_allowed": 0,
                "subtraction_blocker_count_allowed": 0,
                "all_known_subjects_allow_gaussian_subtraction_claim": True,
                "control_prior_similarity_min": thresholds["control_prior_similarity_min"],
                "all_control_prior_subjects_quality_gate_pass": True,
            },
            "interpretation": (
                "Healthy references must have an audit proving non-degenerate "
                "subtraction readiness; control-prior runs must also pass "
                "reference-similarity quality checks before residuals are "
                "interpreted as lesion signal."
            ),
        },
        "gaussian_splat_fit": {
            "status": _gate_status(evidence_present=splat_evidence, passed=bool(splat_pass)),
            "observed": {
                "n_with_splat_audit": splat_n,
                "n_with_splat_readiness_audit": splat_readiness_n,
                "n_readiness_review": splat_readiness_review_n,
                "n_readiness_fail": splat_readiness_fail_n,
                "total_readiness_blocker_count": splat_readiness_blockers,
                "n_mri_to_3dgs_claim_allowed": splat_claim_allowed_n,
                "mean_foreground_coverage": splat_coverage,
                "mean_mip_mae": splat_mip_mae,
                "claim_class_counts": splat_reconstruction_quality.get(
                    "claim_class_counts",
                    {},
                ),
                "n_with_optimized_field_audit": opt_field_audit_n,
                "n_optimized_field_readiness_pass": opt_field_pass_n,
                "n_optimized_field_readiness_review": opt_field_review_n,
                "n_optimized_field_readiness_fail": opt_field_fail_n,
                "optimized_field_total_blocker_count": opt_field_blockers,
                "n_optimized_mri_to_3dgs_claim_allowed": opt_field_claim_allowed_n,
                "n_optimized_field_volume_anchored_training": (
                    opt_field_volume_anchored_n
                ),
                "n_optimized_field_affine_mismatch": opt_field_affine_mismatch_n,
                "n_optimized_field_affine_missing": opt_field_affine_missing_n,
                "optimized_field_mean_foreground_mae": opt_field_fg_mae,
                "optimized_field_mean_all_mae": opt_field_all_mae,
                "optimized_field_status_counts": (
                    optimized_gaussian_field_quality.get("status_counts", {})
                ),
            },
            "threshold": {
                "all_known_subjects_audited": True,
                "all_known_subjects_have_readiness_gate": True,
                "readiness_fail_allowed": 0,
                "readiness_blocker_count_allowed": 0,
                "all_known_subjects_allow_mri_to_3dgs_claim": True,
                "splat_foreground_coverage_min": thresholds["splat_foreground_coverage_min"],
                "splat_mip_mean_mae_max": thresholds["splat_mip_mean_mae_max"],
                "all_known_subjects_have_optimized_field_audit": True,
                "all_known_subjects_allow_optimized_mri_to_3dgs_claim": True,
                "optimized_field_readiness_fail_allowed": 0,
                "optimized_field_readiness_review_allowed": 0,
                "optimized_field_blocker_count_allowed": 0,
                "all_known_subjects_volume_anchored_training": True,
                "optimized_field_affine_mismatch_allowed": 0,
                "optimized_field_foreground_mae_max": (
                    thresholds["optimized_field_foreground_mae_max"]
                ),
                "optimized_field_all_mae_max": (
                    thresholds["optimized_field_all_mae_max"]
                ),
            },
            "interpretation": (
                "MRI-to-3DGS representation must reconstruct anatomy before "
                "residuals are trusted: the direct browser PLY needs a passing "
                "artifact audit, and the optimized Gaussian field needs held-out "
                "voxel fidelity evidence."
            ),
        },
        "paired_gaussian_subtraction": {
            "status": _gate_status(
                evidence_present=paired_evidence,
                passed=bool(paired_pass),
            ),
            "observed": {
                "n_known_subjects": n_total_known,
                "n_with_paired_gaussian_state": paired_state_n,
                "n_same_gaussian_support": paired_same_support_n,
                "n_valid_state_schema": paired_schema_n,
                "n_volume_sampling_mode": paired_volume_support_n,
                "n_expected_support_basis": paired_expected_basis_n,
                "n_unique_supported_support_basis": paired_unique_support_basis_n,
                "sampling_mode_counts": paired_gaussian_subtraction_quality.get(
                    "sampling_mode_counts",
                    {},
                ),
                "support_basis_counts": paired_gaussian_subtraction_quality.get(
                    "support_basis_counts",
                    {},
                ),
                "n_with_patient_normal_residual_splats": paired_splat_n,
                "min_n_gaussians": paired_min_n_gaussians,
                "mean_n_gaussians": paired_gaussian_subtraction_quality.get(
                    "mean_n_gaussians"
                ),
            },
            "threshold": {
                "all_known_subjects_have_paired_gaussian_state": True,
                "same_gaussian_support": True,
                "state_schema": PAIRED_GAUSSIAN_STATE_SCHEMA,
                "sampling_mode": "volume",
                "support_basis": sorted(PAIRED_GAUSSIAN_SUPPORT_BASIS_LABELS),
                "single_support_basis_per_run": True,
                "paired_gaussian_min_points": thresholds["paired_gaussian_min_points"],
            },
            "interpretation": (
                "The paper's core subtraction claim requires patient, synthetic-normal, "
                "and residual splats sharing one full-volume Gaussian support, "
                "backed by the paired Gaussian state sidecar."
            ),
        },
        "control_gaussian_prior_subtraction": {
            "status": _gate_status(
                evidence_present=control_gp_evidence,
                passed=bool(control_gp_pass),
            ),
            "observed": {
                "n_known_subjects": n_total_known,
                "n_with_control_gaussian_prior_state": control_gp_state_n,
                "n_same_gaussian_support": control_gp_same_support_n,
                "n_valid_state_schema": control_gp_schema_n,
                "n_volume_sampling_mode": control_gp_volume_support_n,
                "sampling_mode_counts": control_gaussian_prior_quality.get(
                    "sampling_mode_counts",
                    {},
                ),
                "n_with_patient_prior_residual_splats": control_gp_splat_n,
                "n_with_residual_heatmap": control_gp_heatmap_n,
                "n_with_residual_scores": control_gp_scores_n,
                "n_with_residual_metrics": control_gp_metrics_n,
                "n_with_minimum_controls": control_gp_enough_controls_n,
                "min_n_controls": control_gp_min_n_controls,
                "mean_n_controls": control_gaussian_prior_quality.get(
                    "mean_n_controls"
                ),
                "min_n_gaussians": control_gp_min_n_gaussians,
                "mean_n_gaussians": control_gaussian_prior_quality.get(
                    "mean_n_gaussians"
                ),
            },
            "threshold": {
                "all_known_subjects_have_control_gaussian_prior_state": True,
                "same_gaussian_support": True,
                "state_schema": CONTROL_GAUSSIAN_PRIOR_STATE_SCHEMA,
                "sampling_mode": "volume",
                "patient_prior_residual_splats_required": True,
                "residual_heatmap_required": True,
                "residual_scores_required": True,
                "residual_metrics_required": True,
                "control_gaussian_min_controls": (
                    thresholds["control_gaussian_min_controls"]
                ),
                "control_gaussian_min_points": (
                    thresholds["control_gaussian_min_points"]
                ),
            },
            "interpretation": (
                "The fallback paper method requires patient, healthy-control "
                "prior mean, and z-residual splats sharing one patient-derived "
                "full-volume Gaussian support, plus durable heatmap, score, "
                "metrics, and state sidecars."
            ),
        },
        "optimized_gaussian_subtraction": {
            "status": _gate_status(
                evidence_present=optimized_evidence,
                passed=bool(optimized_pass),
            ),
            "observed": {
                "n_known_subjects": n_total_known,
                "n_with_optimized_gaussian_residual": optimized_residual_n,
                "n_same_gaussian_support": optimized_same_support_n,
                "n_gaussian_subtraction_claim_allowed": optimized_claim_allowed_n,
                "n_indexwise_score_method": optimized_indexwise_n,
                "n_field_heatmap_method": optimized_field_heatmap_n,
                "score_method_counts": optimized_gaussian_subtraction_quality.get(
                    "score_method_counts",
                    {},
                ),
                "heatmap_method_counts": optimized_gaussian_subtraction_quality.get(
                    "heatmap_method_counts",
                    {},
                ),
                "n_with_residual_splat": optimized_splat_n,
                "n_with_residual_heatmap": optimized_heatmap_n,
                "n_with_residual_metrics": optimized_metrics_n,
                "n_with_residual_scores": optimized_gaussian_subtraction_quality.get(
                    "n_with_residual_scores",
                    0,
                ),
                "min_n_patient_gaussians": optimized_min_n_patient,
                "mean_n_patient_gaussians": optimized_gaussian_subtraction_quality.get(
                    "mean_n_patient_gaussians"
                ),
                "min_n_normal_gaussians": optimized_min_n_normal,
                "mean_n_normal_gaussians": optimized_gaussian_subtraction_quality.get(
                    "mean_n_normal_gaussians"
                ),
                "max_means_norm_abs_diff": optimized_gaussian_subtraction_quality.get(
                    "max_means_norm_abs_diff"
                ),
                "max_log_scales_abs_diff": optimized_gaussian_subtraction_quality.get(
                    "max_log_scales_abs_diff"
                ),
                "max_quat_wxyz_abs_diff": optimized_gaussian_subtraction_quality.get(
                    "max_quat_wxyz_abs_diff"
                ),
            },
            "threshold": {
                "all_known_subjects_have_optimized_gaussian_residual": True,
                "same_gaussian_support": True,
                "gaussian_subtraction_claim_allowed": True,
                "residual_score_method": OPTIMIZED_GAUSSIAN_SCORE_METHOD,
                "heatmap_method": OPTIMIZED_GAUSSIAN_HEATMAP_METHOD,
                "residual_splat_artifact_required": True,
                "residual_heatmap_artifact_required": True,
                "residual_metrics_artifact_required": True,
                "optimized_gaussian_min_points": (
                    thresholds["optimized_gaussian_min_points"]
                ),
            },
            "interpretation": (
                "The optimized 3DGS paper claim requires patient and "
                "synthetic-normal Gaussian fields on identical support, "
                "indexwise residual scores, field heatmap projection, and "
                "durable residual splat/heatmap/metrics artifacts."
            ),
        },
        "lesion_localization": {
            "status": _gate_status(
                evidence_present=localization_evidence,
                passed=bool(localization_pass),
            ),
            "observed": {
                "n_with_manual_roi_overlap": localization_n,
                "mean_manual_coverage": localization_mean,
                "coverage_ge_0_50_rate": candidate_validation.get("coverage_ge_0_50_rate"),
            },
            "threshold": {
                "all_fcd_subjects_have_manual_overlap_metrics": True,
                "localization_mean_manual_coverage_min": (
                    thresholds["localization_mean_manual_coverage_min"]
                ),
            },
            "interpretation": "Residual candidates must overlap known FCD masks, not just produce high scores.",
        },
        "healthy_control_false_positive_burden": {
            "status": _gate_status(evidence_present=hc_evidence, passed=bool(hc_pass)),
            "observed": {"mean_candidate_foreground_fraction_hc": hc_burden},
            "threshold": {
                "hc_candidate_foreground_fraction_max": (
                    thresholds["hc_candidate_foreground_fraction_max"]
                ),
            },
            "interpretation": "Healthy controls should not accumulate large residual-candidate masks.",
        },
    }
    blocking = [
        name for name, gate in gates.items()
        if gate["status"] != "pass"
    ]
    patient_specific_counterfactual_supported = (
        synthetic_audit_evidence
        and synthetic_patient_counterfactual_allowed_n >= n_total_known
    )
    if not synthetic_audit_evidence:
        patient_specific_counterfactual_status = "insufficient_evidence"
    elif patient_specific_counterfactual_supported:
        patient_specific_counterfactual_status = "supported"
    else:
        patient_specific_counterfactual_status = "not_supported"

    claim_boundaries = {
        "patient_specific_healthy_counterfactual": {
            "status": patient_specific_counterfactual_status,
            "observed": {
                "n_known_subjects": n_total_known,
                "n_with_synthetic_normal_audit": synthetic_audit_n,
                "n_patient_conditioned_counterfactual_claim_allowed": (
                    synthetic_patient_counterfactual_allowed_n
                ),
                "n_patient_conditioned_inpainting": synthetic_patient_inpainting_n,
                "counterfactual_class_counts": synthetic_normal_quality.get(
                    "counterfactual_class_counts",
                    {},
                ),
                "claim_class_counts": synthetic_normal_quality.get(
                    "claim_class_counts",
                    {},
                ),
            },
            "interpretation": (
                "This is a non-blocking wording boundary. The current paper "
                "readiness gates can support a healthy-reference Gaussian "
                "subtraction claim when they pass, but a patient-specific "
                "healthy counterfactual/inpainting claim requires every known "
                "subject to allow that claim. NVIDIA image-only or "
                "control-prior references must not be described as "
                "patient-conditioned lesion inpainting."
            ),
        },
    }
    return {
        "overall_status": "pass" if not blocking else "not_ready",
        "blocking_gates": blocking,
        "thresholds": thresholds,
        "gates": gates,
        "claim_boundaries": claim_boundaries,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_output_root(
    output_root: Path | str,
    output_json: Optional[Path | str] = None,
    output_csv: Optional[Path | str] = None,
    gate_thresholds: Optional[dict[str, float]] = None,
    score_metric: str = "auto",
) -> dict:
    """Evaluate all subjects under *output_root*.

    Parameters
    ----------
    output_root:
        Directory produced by ``fcd-build-end-to-end``; must contain
        ``index.json`` and per-subject ``<id>/manifest.json`` files.
    output_json:
        Optional path to write the result dict as JSON.
    output_csv:
        Optional path to write a per-subject CSV with columns:
        subject, group, score, score_source.

    Returns
    -------
    dict
        Evaluation result including counts, group means, AUROC, metrics dict,
        and per-subject rows with per-metric fields.
    """
    output_root = Path(output_root)
    if score_metric != "auto" and score_metric not in EVALUATED_METRIC_KEYS:
        valid = ", ".join(["auto", *EVALUATED_METRIC_KEYS])
        raise ValueError(f"Unknown score_metric {score_metric!r}. Valid choices: {valid}")
    thresholds = dict(DEFAULT_PAPER_GATE_THRESHOLDS)
    if gate_thresholds:
        thresholds.update(gate_thresholds)
    index_path = output_root / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"index.json not found at {index_path}")

    with index_path.open() as fh:
        index = json.load(fh)

    rows: list[dict] = []
    for entry in index.get("subjects", []):
        subject = entry.get("subject", "unknown")
        group_raw = (entry.get("group") or "unknown").strip().lower()
        # Normalise group label
        if group_raw in ("fcd", "patient", "lesion"):
            group = "FCD"
        elif group_raw in ("hc", "control", "healthy", "nc"):
            group = "HC"
        else:
            group = "unknown"

        # Load per-subject manifest for score extraction
        manifest_rel = entry.get("manifest", f"{subject}/manifest.json")
        manifest_path = output_root / manifest_rel
        score: Optional[float] = None
        score_source = "missing"
        control_prior_quality_gate: Optional[str] = None
        roi_source = entry.get("roi_source") or "unknown"
        roi_source_normalized = _normalize_roi_source(roi_source)
        subject_output_dir = manifest_path.parent
        all_metrics: dict[str, Optional[float]] = {
            key: None for key in EVALUATED_METRIC_KEYS
        }
        paired_gaussian_evidence = _empty_paired_gaussian_evidence()
        control_gaussian_prior_evidence = (
            _empty_control_gaussian_prior_evidence()
        )
        optimized_gaussian_evidence = _empty_optimized_gaussian_evidence()
        data_quality_evidence = _empty_data_quality_evidence()
        splat_reconstruction_evidence = _empty_splat_reconstruction_evidence()
        optimized_field_audit_evidence = (
            _empty_optimized_gaussian_field_audit_evidence()
        )
        synthetic_normal_evidence = _empty_synthetic_normal_audit_evidence()
        manifest_for_validation: dict = {}

        if manifest_path.is_file():
            with manifest_path.open() as fh:
                mf = json.load(fh)
            manifest_for_validation = mf
            # Allow group override from manifest if index entry is missing it
            if group == "unknown":
                g2 = (mf.get("group") or "unknown").strip().lower()
                if g2 in ("fcd", "patient", "lesion"):
                    group = "FCD"
                elif g2 in ("hc", "control", "healthy", "nc"):
                    group = "HC"
            score, score_source = _extract_score_and_source(mf)
            control_prior_quality_gate = _extract_control_prior_gate(mf)
            roi_source = mf.get("roi_source") or roi_source
            roi_source_normalized = _normalize_roi_source(roi_source)

            all_metrics = _extract_all_metrics(mf, subject_output_dir)
            paired_gaussian_evidence = _extract_paired_gaussian_evidence(
                mf,
                subject_output_dir,
            )
            control_gaussian_prior_evidence = (
                _extract_control_gaussian_prior_evidence(
                    mf,
                    subject_output_dir,
                )
            )
            optimized_gaussian_evidence = _extract_optimized_gaussian_evidence(
                mf,
                subject_output_dir,
            )
            data_quality_evidence = _extract_data_quality_evidence(mf)
            splat_reconstruction_evidence = _extract_splat_reconstruction_evidence(mf)
            optimized_field_audit_evidence = (
                _extract_optimized_gaussian_field_audit_evidence(
                    mf,
                    subject_output_dir,
                )
            )
            synthetic_normal_evidence = _extract_synthetic_normal_audit_evidence(mf)
            if score_metric != "auto":
                score = all_metrics.get(score_metric)
                score_source = f"metrics.{score_metric}" if score is not None else "missing"

        row = {
            "subject": subject,
            "group": group,
            "score": score,
            "score_source": score_source,
            "roi_source": roi_source,
            "roi_source_normalized": roi_source_normalized,
            "control_prior_quality_gate": control_prior_quality_gate,
        }
        row.update(all_metrics)
        row.update(paired_gaussian_evidence)
        row.update(control_gaussian_prior_evidence)
        row.update(optimized_gaussian_evidence)
        row.update(data_quality_evidence)
        row.update(splat_reconstruction_evidence)
        row.update(optimized_field_audit_evidence)
        row.update(synthetic_normal_evidence)

        if not _manual_candidate_validation_allowed(manifest_for_validation, group):
            _drop_untrusted_manual_candidate_metrics(row)
            if score_metric in RESIDUAL_CANDIDATE_MANUAL_METRIC_KEYS:
                row["score"] = None
                row["score_source"] = "missing"

        rows.append(row)

    effective_score_metric = score_metric
    if score_metric == "auto":
        known_rows = [r for r in rows if r["group"] in {"FCD", "HC"}]
        all_known_roi_free = (
            bool(known_rows)
            and all(r["roi_source_normalized"] in ROI_FREE_SOURCES for r in known_rows)
        )
        all_known_have_local_ratio = (
            bool(known_rows)
            and all(r.get("paired_roi_to_non_roi_ratio") is not None for r in known_rows)
        )
        if all_known_roi_free and all_known_have_local_ratio:
            for row in known_rows:
                row["score"] = row["paired_roi_to_non_roi_ratio"]
                row["score_source"] = "metrics.paired_roi_to_non_roi_ratio"
        known_scored_sources = {
            r["score_source"] for r in rows
            if r["group"] in {"FCD", "HC"} and r["score"] is not None
        }
        if known_scored_sources == {"metrics.paired_roi_to_non_roi_ratio"}:
            effective_score_metric = "paired_roi_to_non_roi_ratio"

    # Split by group for AUROC using the best available score source.
    fcd_scores = [r["score"] for r in rows if r["group"] == "FCD" and r["score"] is not None]
    hc_scores  = [r["score"] for r in rows if r["group"] == "HC"  and r["score"] is not None]
    missing_scores = sum(1 for r in rows if r["score"] is None)

    n_fcd = sum(1 for r in rows if r["group"] == "FCD")
    n_hc  = sum(1 for r in rows if r["group"] == "HC")
    n_unk = sum(1 for r in rows if r["group"] == "unknown")

    fcd_mean = float(np.mean(fcd_scores)) if fcd_scores else None
    hc_mean  = float(np.mean(hc_scores))  if hc_scores  else None

    auroc = _mannwhitney_auroc(fcd_scores, hc_scores)
    roi_summary = _roi_source_summary(rows)

    # Compute per-metric statistics
    metrics_result: dict[str, dict] = {}
    for metric_name in EVALUATED_METRIC_KEYS:
        metric_fcd = [r[metric_name] for r in rows if r["group"] == "FCD" and r[metric_name] is not None]
        metric_hc = [r[metric_name] for r in rows if r["group"] == "HC" and r[metric_name] is not None]
        metric_missing = sum(1 for r in rows if r[metric_name] is None)

        metric_fcd_mean = float(np.mean(metric_fcd)) if metric_fcd else None
        metric_hc_mean = float(np.mean(metric_hc)) if metric_hc else None
        metric_auroc = _mannwhitney_auroc(metric_fcd, metric_hc)

        metrics_result[metric_name] = {
            "n_scored": len(metric_fcd) + len(metric_hc),
            "n_missing": metric_missing,
            "fcd_mean": metric_fcd_mean,
            "hc_mean": metric_hc_mean,
            "auroc_fcd_vs_hc": metric_auroc,
        }

    candidate_manual_rows = [
        r for r in rows
        if r["group"] == "FCD"
        and r["residual_candidate_manual_coverage"] is not None
    ]
    candidate_coverages = [
        r["residual_candidate_manual_coverage"] for r in candidate_manual_rows
    ]
    candidate_precisions = [
        r["residual_candidate_precision"] for r in candidate_manual_rows
        if r["residual_candidate_precision"] is not None
    ]
    candidate_dice = [
        r["residual_candidate_dice"] for r in candidate_manual_rows
        if r["residual_candidate_dice"] is not None
    ]
    fcd_candidate_volumes = [
        r["residual_candidate_volume_ml"] for r in rows
        if r["group"] == "FCD" and r["residual_candidate_volume_ml"] is not None
    ]
    hc_candidate_volumes = [
        r["residual_candidate_volume_ml"] for r in rows
        if r["group"] == "HC" and r["residual_candidate_volume_ml"] is not None
    ]
    fcd_candidate_fractions = [
        r["residual_candidate_foreground_fraction"] for r in rows
        if r["group"] == "FCD" and r["residual_candidate_foreground_fraction"] is not None
    ]
    hc_candidate_fractions = [
        r["residual_candidate_foreground_fraction"] for r in rows
        if r["group"] == "HC" and r["residual_candidate_foreground_fraction"] is not None
    ]

    def _mean_or_none(values: list[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None

    def _median_or_none(values: list[float]) -> Optional[float]:
        return float(np.median(values)) if values else None

    data_quality_known_rows = [r for r in rows if r["group"] in {"FCD", "HC"}]
    data_quality_statuses = [
        str(r.get("data_quality_upload_readiness_status") or "missing")
        for r in data_quality_known_rows
    ]
    dicom_conversion_statuses = [
        str(r.get("data_quality_dicom_conversion_status"))
        for r in data_quality_known_rows
        if r.get("data_quality_dicom_conversion_status") is not None
    ]
    data_quality_summary = {
        "n_known_subjects": len(data_quality_known_rows),
        "n_with_data_quality": sum(
            1 for r in data_quality_known_rows if r["data_quality_present"]
        ),
        "n_upload_readiness_pass": sum(
            1 for value in data_quality_statuses if value == "pass"
        ),
        "n_upload_readiness_review": sum(
            1 for value in data_quality_statuses if value == "review"
        ),
        "n_upload_readiness_fail": sum(
            1 for value in data_quality_statuses if value == "fail"
        ),
        "n_can_run_3dgs": sum(
            1 for r in data_quality_known_rows
            if r["data_quality_can_run_3dgs"] is True
        ),
        "n_can_subtract_patient_normal": sum(
            1 for r in data_quality_known_rows
            if r["data_quality_can_subtract_patient_normal"] is True
        ),
        "n_with_t1w_to_flair_resampling_review": sum(
            1 for r in data_quality_known_rows
            if r["data_quality_t1w_to_flair_needs_resample"]
        ),
        "n_with_roi_to_flair_resampling_review": sum(
            1 for r in data_quality_known_rows
            if r["data_quality_roi_to_flair_needs_resample"]
        ),
        "n_with_roi_resampled_to_base": sum(
            1 for r in data_quality_known_rows
            if r["data_quality_roi_resampled_to_base"]
        ),
        "n_with_nonfinite_blocker": sum(
            1 for r in data_quality_known_rows
            if r["data_quality_has_nonfinite_blocker"]
        ),
        "n_with_dicom_conversion_safety": len(dicom_conversion_statuses),
        "n_dicom_conversion_pass": sum(
            1 for value in dicom_conversion_statuses if value == "pass"
        ),
        "n_dicom_conversion_review": sum(
            1 for value in dicom_conversion_statuses if value == "review"
        ),
        "n_dicom_conversion_fail": sum(
            1 for value in dicom_conversion_statuses if value == "fail"
        ),
        "n_dicom_conversion_can_run_3dgs": sum(
            1 for r in data_quality_known_rows
            if r["data_quality_dicom_conversion_can_run_3dgs"] is True
        ),
        "n_dicom_conversion_paper_claim_allowed": sum(
            1 for r in data_quality_known_rows
            if r["data_quality_dicom_conversion_paper_claim_allowed"] is True
        ),
        "total_blocking_issue_count": sum(
            int(r["data_quality_blocking_issue_count"])
            for r in data_quality_known_rows
        ),
        "total_review_issue_count": sum(
            int(r["data_quality_review_issue_count"])
            for r in data_quality_known_rows
        ),
        "total_dicom_conversion_blocking_issue_count": sum(
            int(r["data_quality_dicom_conversion_blocking_issue_count"])
            for r in data_quality_known_rows
        ),
        "total_dicom_conversion_review_issue_count": sum(
            int(r["data_quality_dicom_conversion_review_issue_count"])
            for r in data_quality_known_rows
        ),
        "total_warning_count": sum(
            int(r["data_quality_warning_count"])
            for r in data_quality_known_rows
        ),
    }

    candidate_validation = {
        "n_with_manual_roi_overlap": len(candidate_manual_rows),
        "mean_manual_coverage": _mean_or_none(candidate_coverages),
        "median_manual_coverage": _median_or_none(candidate_coverages),
        "mean_candidate_precision": _mean_or_none(candidate_precisions),
        "median_candidate_precision": _median_or_none(candidate_precisions),
        "mean_dice": _mean_or_none(candidate_dice),
        "median_dice": _median_or_none(candidate_dice),
        "n_coverage_ge_0_10": sum(1 for v in candidate_coverages if v >= 0.10),
        "n_coverage_ge_0_25": sum(1 for v in candidate_coverages if v >= 0.25),
        "n_coverage_ge_0_50": sum(1 for v in candidate_coverages if v >= 0.50),
        "coverage_ge_0_10_rate": (
            sum(1 for v in candidate_coverages if v >= 0.10) / len(candidate_coverages)
            if candidate_coverages else None
        ),
        "coverage_ge_0_25_rate": (
            sum(1 for v in candidate_coverages if v >= 0.25) / len(candidate_coverages)
            if candidate_coverages else None
        ),
        "coverage_ge_0_50_rate": (
            sum(1 for v in candidate_coverages if v >= 0.50) / len(candidate_coverages)
            if candidate_coverages else None
        ),
        "mean_candidate_volume_ml_fcd": _mean_or_none(fcd_candidate_volumes),
        "mean_candidate_volume_ml_hc": _mean_or_none(hc_candidate_volumes),
        "mean_candidate_foreground_fraction_fcd": _mean_or_none(fcd_candidate_fractions),
        "mean_candidate_foreground_fraction_hc": _mean_or_none(hc_candidate_fractions),
    }

    splat_rows = [r for r in rows if r["splat_foreground_coverage"] is not None]
    splat_known_rows = [r for r in rows if r["group"] in {"FCD", "HC"}]
    splat_statuses = [
        str(r.get("splat_reconstruction_readiness_status") or "missing")
        for r in splat_known_rows
    ]
    splat_reconstruction_quality = {
        "n_known_subjects": len(splat_known_rows),
        "n_with_splat_audit": len(splat_rows),
        "n_with_splat_readiness_audit": sum(
            1 for r in splat_known_rows
            if r["splat_reconstruction_audit_present"]
        ),
        "n_readiness_pass": sum(1 for value in splat_statuses if value == "pass"),
        "n_readiness_review": sum(1 for value in splat_statuses if value == "review"),
        "n_readiness_fail": sum(1 for value in splat_statuses if value == "fail"),
        "n_readiness_missing": sum(1 for value in splat_statuses if value == "missing"),
        "total_readiness_blocker_count": sum(
            int(r.get("splat_reconstruction_blocker_count") or 0)
            for r in splat_known_rows
        ),
        "total_readiness_warning_count": sum(
            int(r.get("splat_reconstruction_warning_count") or 0)
            for r in splat_known_rows
        ),
        "n_mri_to_3dgs_claim_allowed": sum(
            1 for r in splat_known_rows
            if r["splat_mri_to_3dgs_claim_allowed"] is True
        ),
        "mean_foreground_coverage": _mean_or_none([
            r["splat_foreground_coverage"] for r in splat_rows
        ]),
        "mean_all_foreground_mae": _mean_or_none([
            r["splat_all_foreground_mae"] for r in splat_rows
            if r["splat_all_foreground_mae"] is not None
        ]),
        "mean_all_foreground_psnr": _mean_or_none([
            r["splat_all_foreground_psnr"] for r in splat_rows
            if r["splat_all_foreground_psnr"] is not None
        ]),
        "mean_mip_mae": _mean_or_none([
            r["splat_mip_mean_mae"] for r in splat_rows
            if r["splat_mip_mean_mae"] is not None
        ]),
        "claim_class_counts": {
            value: [
                str(r.get("splat_mri_to_3dgs_claim_class") or "missing")
                for r in splat_known_rows
            ].count(value)
            for value in sorted({
                str(r.get("splat_mri_to_3dgs_claim_class") or "missing")
                for r in splat_known_rows
            })
        },
    }

    opt_field_known_rows = [r for r in rows if r["group"] in {"FCD", "HC"}]
    opt_field_statuses = [
        str(r.get("optimized_gaussian_field_audit_status") or "missing")
        for r in opt_field_known_rows
    ]
    opt_field_fg_mae = [
        r["optimized_gaussian_field_foreground_mae"]
        for r in opt_field_known_rows
        if r["optimized_gaussian_field_foreground_mae"] is not None
    ]
    opt_field_all_mae = [
        r["optimized_gaussian_field_all_mae"]
        for r in opt_field_known_rows
        if r["optimized_gaussian_field_all_mae"] is not None
    ]
    opt_field_fg_rmse = [
        r["optimized_gaussian_field_foreground_rmse"]
        for r in opt_field_known_rows
        if r["optimized_gaussian_field_foreground_rmse"] is not None
    ]
    opt_field_volume_anchored = [
        r for r in opt_field_known_rows
        if (
            r.get("optimized_gaussian_field_training_objective")
            == "mip_projection_mse_plus_volume_sample_mse"
            or (
                (r.get("optimized_gaussian_field_volume_loss_weight") or 0) > 0
                and (r.get("optimized_gaussian_field_volume_sample_count") or 0) > 0
            )
        )
    ]
    optimized_gaussian_field_quality = {
        "n_known_subjects": len(opt_field_known_rows),
        "n_with_optimized_field_audit": sum(
            1 for r in opt_field_known_rows
            if r["optimized_gaussian_field_audit_present"]
        ),
        "n_readiness_pass": sum(1 for value in opt_field_statuses if value == "pass"),
        "n_readiness_review": sum(1 for value in opt_field_statuses if value == "review"),
        "n_readiness_fail": sum(1 for value in opt_field_statuses if value == "fail"),
        "n_readiness_missing": sum(1 for value in opt_field_statuses if value == "missing"),
        "total_readiness_blocker_count": sum(
            int(r.get("optimized_gaussian_field_audit_blocker_count") or 0)
            for r in opt_field_known_rows
        ),
        "total_readiness_warning_count": sum(
            int(r.get("optimized_gaussian_field_audit_warning_count") or 0)
            for r in opt_field_known_rows
        ),
        "n_optimized_mri_to_3dgs_claim_allowed": sum(
            1 for r in opt_field_known_rows
            if r["optimized_mri_to_3dgs_claim_allowed"] is True
        ),
        "n_volume_anchored_training": len(opt_field_volume_anchored),
        "n_affine_mismatch": sum(
            1 for r in opt_field_known_rows
            if r["optimized_gaussian_field_affine_mismatch"] is True
        ),
        "n_affine_missing": sum(
            1 for r in opt_field_known_rows
            if r["optimized_gaussian_field_affine_missing"] is True
        ),
        "mean_foreground_mae": _mean_or_none(opt_field_fg_mae),
        "max_foreground_mae": (
            float(np.max(opt_field_fg_mae)) if opt_field_fg_mae else None
        ),
        "mean_all_mae": _mean_or_none(opt_field_all_mae),
        "max_all_mae": (
            float(np.max(opt_field_all_mae)) if opt_field_all_mae else None
        ),
        "mean_foreground_rmse": _mean_or_none(opt_field_fg_rmse),
        "status_counts": {
            value: opt_field_statuses.count(value)
            for value in sorted(set(opt_field_statuses))
        },
    }

    paired_known_rows = [r for r in rows if r["group"] in {"FCD", "HC"}]
    paired_n_gaussians = [
        r["paired_gaussian_n_gaussians"] for r in paired_known_rows
        if r["paired_gaussian_n_gaussians"] is not None
    ]
    paired_sampling_modes = [
        str(r.get("paired_gaussian_sampling_mode") or "missing")
        for r in paired_known_rows
    ]
    paired_support_bases = [
        str(r.get("paired_gaussian_support_basis") or "missing")
        for r in paired_known_rows
    ]
    paired_supported_bases = [
        value for value in paired_support_bases
        if value in PAIRED_GAUSSIAN_SUPPORT_BASIS_LABELS
    ]
    paired_gaussian_subtraction_quality = {
        "n_known_subjects": len(paired_known_rows),
        "n_with_paired_gaussian_state": sum(
            1 for r in paired_known_rows if r["paired_gaussian_state_present"]
        ),
        "n_same_gaussian_support": sum(
            1 for r in paired_known_rows
            if r["paired_gaussian_same_support"] is True
        ),
        "n_valid_state_schema": sum(
            1 for r in paired_known_rows
            if r["paired_gaussian_state_schema"] == PAIRED_GAUSSIAN_STATE_SCHEMA
        ),
        "n_volume_sampling_mode": sum(
            1 for r in paired_known_rows
            if r["paired_gaussian_sampling_mode"] == "volume"
        ),
        "n_expected_support_basis": sum(
            1 for r in paired_known_rows
            if r["paired_gaussian_support_basis"] in PAIRED_GAUSSIAN_SUPPORT_BASIS_LABELS
        ),
        "n_unique_supported_support_basis": len(set(paired_supported_bases)),
        "sampling_mode_counts": {
            value: paired_sampling_modes.count(value)
            for value in sorted(set(paired_sampling_modes))
        },
        "support_basis_counts": {
            value: paired_support_bases.count(value)
            for value in sorted(set(paired_support_bases))
        },
        "n_with_patient_normal_residual_splats": sum(
            1 for r in paired_known_rows
            if r["paired_patient_splat_present"]
            and r["paired_normal_splat_present"]
            and r["paired_residual_splat_present"]
        ),
        "n_with_residual_heatmap": sum(
            1 for r in paired_known_rows if r["paired_residual_heatmap_present"]
        ),
        "min_n_gaussians": (
            float(np.min(paired_n_gaussians)) if paired_n_gaussians else None
        ),
        "mean_n_gaussians": _mean_or_none(paired_n_gaussians),
    }

    control_gp_known_rows = [r for r in rows if r["group"] in {"FCD", "HC"}]
    control_gp_n_gaussians = [
        r["control_gaussian_prior_n_gaussians"] for r in control_gp_known_rows
        if r["control_gaussian_prior_n_gaussians"] is not None
    ]
    control_gp_n_controls = [
        r["control_gaussian_prior_n_controls"] for r in control_gp_known_rows
        if r["control_gaussian_prior_n_controls"] is not None
    ]
    control_gp_sampling_modes = [
        str(r.get("control_gaussian_prior_sampling_mode") or "missing")
        for r in control_gp_known_rows
    ]
    control_gp_min_controls_threshold = float(
        thresholds["control_gaussian_min_controls"]
    )
    control_gaussian_prior_quality = {
        "n_known_subjects": len(control_gp_known_rows),
        "n_with_control_gaussian_prior_state": sum(
            1 for r in control_gp_known_rows
            if r["control_gaussian_prior_state_present"]
        ),
        "n_same_gaussian_support": sum(
            1 for r in control_gp_known_rows
            if r["control_gaussian_prior_same_support"] is True
        ),
        "n_valid_state_schema": sum(
            1 for r in control_gp_known_rows
            if (
                r["control_gaussian_prior_state_schema"]
                == CONTROL_GAUSSIAN_PRIOR_STATE_SCHEMA
            )
        ),
        "n_volume_sampling_mode": sum(
            1 for r in control_gp_known_rows
            if r["control_gaussian_prior_sampling_mode"] == "volume"
        ),
        "sampling_mode_counts": {
            value: control_gp_sampling_modes.count(value)
            for value in sorted(set(control_gp_sampling_modes))
        },
        "n_with_patient_prior_residual_splats": sum(
            1 for r in control_gp_known_rows
            if r["control_gaussian_prior_patient_splat_present"]
            and r["control_gaussian_prior_mean_splat_present"]
            and r["control_gaussian_prior_residual_splat_present"]
        ),
        "n_with_residual_heatmap": sum(
            1 for r in control_gp_known_rows
            if r["control_gaussian_prior_heatmap_present"]
        ),
        "n_with_residual_scores": sum(
            1 for r in control_gp_known_rows
            if r["control_gaussian_prior_scores_present"]
        ),
        "n_with_residual_metrics": sum(
            1 for r in control_gp_known_rows
            if r["control_gaussian_prior_metrics_present"]
        ),
        "n_with_minimum_controls": sum(
            1 for r in control_gp_known_rows
            if (
                r["control_gaussian_prior_n_controls"] is not None
                and r["control_gaussian_prior_n_controls"]
                >= control_gp_min_controls_threshold
            )
        ),
        "min_n_controls": (
            float(np.min(control_gp_n_controls)) if control_gp_n_controls else None
        ),
        "mean_n_controls": _mean_or_none(control_gp_n_controls),
        "min_n_gaussians": (
            float(np.min(control_gp_n_gaussians)) if control_gp_n_gaussians else None
        ),
        "mean_n_gaussians": _mean_or_none(control_gp_n_gaussians),
        "n_roi_metrics_independent_validation": sum(
            1 for r in control_gp_known_rows
            if r[
                "control_gaussian_prior_roi_metrics_independent_validation"
            ] is True
        ),
    }

    opt_known_rows = [r for r in rows if r["group"] in {"FCD", "HC"}]
    opt_n_patient = [
        r["optimized_gaussian_n_patient"] for r in opt_known_rows
        if r["optimized_gaussian_n_patient"] is not None
    ]
    opt_n_normal = [
        r["optimized_gaussian_n_normal"] for r in opt_known_rows
        if r["optimized_gaussian_n_normal"] is not None
    ]
    opt_means_diffs = [
        r["optimized_gaussian_means_norm_max_abs_diff"] for r in opt_known_rows
        if r["optimized_gaussian_means_norm_max_abs_diff"] is not None
    ]
    opt_log_scales_diffs = [
        r["optimized_gaussian_log_scales_max_abs_diff"] for r in opt_known_rows
        if r["optimized_gaussian_log_scales_max_abs_diff"] is not None
    ]
    opt_quat_diffs = [
        r["optimized_gaussian_quat_wxyz_max_abs_diff"] for r in opt_known_rows
        if r["optimized_gaussian_quat_wxyz_max_abs_diff"] is not None
    ]
    optimized_gaussian_subtraction_quality = {
        "n_known_subjects": len(opt_known_rows),
        "n_with_optimized_gaussian_residual": sum(
            1 for r in opt_known_rows if r["optimized_gaussian_residual_present"]
        ),
        "n_same_gaussian_support": sum(
            1 for r in opt_known_rows
            if r["optimized_gaussian_same_support"] is True
        ),
        "n_gaussian_subtraction_claim_allowed": sum(
            1 for r in opt_known_rows
            if r["optimized_gaussian_subtraction_claim_allowed"] is True
        ),
        "n_indexwise_score_method": sum(
            1 for r in opt_known_rows
            if r["optimized_gaussian_score_method"]
            == OPTIMIZED_GAUSSIAN_SCORE_METHOD
        ),
        "n_field_heatmap_method": sum(
            1 for r in opt_known_rows
            if r["optimized_gaussian_heatmap_method"]
            == OPTIMIZED_GAUSSIAN_HEATMAP_METHOD
        ),
        "score_method_counts": {
            value: [
                str(r.get("optimized_gaussian_score_method") or "missing")
                for r in opt_known_rows
            ].count(value)
            for value in sorted({
                str(r.get("optimized_gaussian_score_method") or "missing")
                for r in opt_known_rows
            })
        },
        "heatmap_method_counts": {
            value: [
                str(r.get("optimized_gaussian_heatmap_method") or "missing")
                for r in opt_known_rows
            ].count(value)
            for value in sorted({
                str(r.get("optimized_gaussian_heatmap_method") or "missing")
                for r in opt_known_rows
            })
        },
        "n_with_residual_splat": sum(
            1 for r in opt_known_rows if r["optimized_gaussian_residual_splat_present"]
        ),
        "n_with_residual_heatmap": sum(
            1 for r in opt_known_rows if r["optimized_gaussian_residual_heatmap_present"]
        ),
        "n_with_residual_metrics": sum(
            1 for r in opt_known_rows if r["optimized_gaussian_residual_metrics_present"]
        ),
        "n_with_residual_scores": sum(
            1 for r in opt_known_rows if r["optimized_gaussian_residual_scores_present"]
        ),
        "min_n_patient_gaussians": (
            float(np.min(opt_n_patient)) if opt_n_patient else None
        ),
        "mean_n_patient_gaussians": _mean_or_none(opt_n_patient),
        "min_n_normal_gaussians": (
            float(np.min(opt_n_normal)) if opt_n_normal else None
        ),
        "mean_n_normal_gaussians": _mean_or_none(opt_n_normal),
        "max_means_norm_abs_diff": (
            float(np.max(opt_means_diffs)) if opt_means_diffs else None
        ),
        "max_log_scales_abs_diff": (
            float(np.max(opt_log_scales_diffs)) if opt_log_scales_diffs else None
        ),
        "max_quat_wxyz_abs_diff": (
            float(np.max(opt_quat_diffs)) if opt_quat_diffs else None
        ),
    }

    control_prior_rows = [
        r for r in rows
        if r.get("control_prior_quality_gate") is not None
        or r["control_prior_reference_similarity_1_minus_mae"] is not None
    ]
    gate_values = [
        str(r.get("control_prior_quality_gate"))
        for r in control_prior_rows
        if r.get("control_prior_quality_gate") is not None
    ]
    control_prior_quality = {
        "n_with_control_prior_quality": len(control_prior_rows),
        "n_quality_gate_pass": sum(1 for value in gate_values if value == "pass"),
        "n_quality_gate_review": sum(1 for value in gate_values if value == "review"),
        "n_quality_gate_insufficient_reference": sum(
            1 for value in gate_values if value == "insufficient_reference"
        ),
        "mean_reference_similarity_1_minus_mae": _mean_or_none([
            r["control_prior_reference_similarity_1_minus_mae"]
            for r in control_prior_rows
            if r["control_prior_reference_similarity_1_minus_mae"] is not None
        ]),
        "mean_reference_mae": _mean_or_none([
            r["control_prior_reference_mae"] for r in control_prior_rows
            if r["control_prior_reference_mae"] is not None
        ]),
        "mean_reference_rmse": _mean_or_none([
            r["control_prior_reference_rmse"] for r in control_prior_rows
            if r["control_prior_reference_rmse"] is not None
        ]),
        "mean_reference_correlation": _mean_or_none([
            r["control_prior_reference_correlation"] for r in control_prior_rows
            if r["control_prior_reference_correlation"] is not None
        ]),
    }

    synthetic_known_rows = [r for r in rows if r["group"] in {"FCD", "HC"}]
    synthetic_statuses = [
        str(r.get("synthetic_normal_subtraction_readiness_status") or "missing")
        for r in synthetic_known_rows
    ]
    counterfactual_classes = [
        str(r.get("synthetic_normal_counterfactual_class") or "missing")
        for r in synthetic_known_rows
    ]
    claim_classes = [
        str(r.get("synthetic_normal_claim_class") or "missing")
        for r in synthetic_known_rows
    ]
    synthetic_normal_quality = {
        "n_known_subjects": len(synthetic_known_rows),
        "n_with_synthetic_normal_audit": sum(
            1 for r in synthetic_known_rows
            if r["synthetic_normal_audit_present"]
        ),
        "n_subtraction_readiness_pass": sum(
            1 for value in synthetic_statuses if value == "pass"
        ),
        "n_subtraction_readiness_review": sum(
            1 for value in synthetic_statuses if value == "review"
        ),
        "n_subtraction_readiness_fail": sum(
            1 for value in synthetic_statuses if value == "fail"
        ),
        "n_subtraction_readiness_missing": sum(
            1 for value in synthetic_statuses if value == "missing"
        ),
        "total_subtraction_blocker_count": sum(
            int(r.get("synthetic_normal_subtraction_blocker_count") or 0)
            for r in synthetic_known_rows
        ),
        "total_subtraction_warning_count": sum(
            int(r.get("synthetic_normal_subtraction_warning_count") or 0)
            for r in synthetic_known_rows
        ),
        "n_gaussian_subtraction_claim_allowed": sum(
            1 for r in synthetic_known_rows
            if r["synthetic_normal_gaussian_subtraction_claim_allowed"] is True
        ),
        "n_patient_conditioned_inpainting": sum(
            1 for r in synthetic_known_rows
            if r["synthetic_normal_patient_conditioned_inpainting"] is True
        ),
        "n_patient_conditioned_counterfactual_claim_allowed": sum(
            1 for r in synthetic_known_rows
            if r[
                "synthetic_normal_patient_conditioned_counterfactual_claim_allowed"
            ] is True
        ),
        "n_resampled_to_patient_grid_by_audit": sum(
            1 for r in synthetic_known_rows
            if r["synthetic_normal_resampled_to_patient_grid"] is True
        ),
        "mean_outside_roi_similarity_1_minus_mae": _mean_or_none([
            r["synthetic_normal_outside_roi_similarity_1_minus_mae"]
            for r in synthetic_known_rows
            if r["synthetic_normal_outside_roi_similarity_1_minus_mae"] is not None
        ]),
        "counterfactual_class_counts": {
            value: counterfactual_classes.count(value)
            for value in sorted(set(counterfactual_classes))
        },
        "claim_class_counts": {
            value: claim_classes.count(value)
            for value in sorted(set(claim_classes))
        },
    }

    n_scored = len(fcd_scores) + len(hc_scores)
    paper_readiness = _build_paper_readiness_gates(
        n_fcd=n_fcd,
        n_hc=n_hc,
        n_scored=n_scored,
        auroc=auroc,
        score_metric=effective_score_metric,
        roi_source_summary=roi_summary,
        data_quality_summary=data_quality_summary,
        candidate_validation=candidate_validation,
        splat_reconstruction_quality=splat_reconstruction_quality,
        optimized_gaussian_field_quality=optimized_gaussian_field_quality,
        paired_gaussian_subtraction_quality=paired_gaussian_subtraction_quality,
        control_gaussian_prior_quality=control_gaussian_prior_quality,
        optimized_gaussian_subtraction_quality=optimized_gaussian_subtraction_quality,
        control_prior_quality=control_prior_quality,
        synthetic_normal_quality=synthetic_normal_quality,
        thresholds=thresholds,
    )

    limitations: list[str] = []
    if n_fcd == 0:
        limitations.append("No FCD subjects found; AUROC is not meaningful (defaulted to 0.5).")
    if n_hc == 0:
        limitations.append("No HC subjects found; AUROC is not meaningful (defaulted to 0.5).")
    if missing_scores > 0:
        limitations.append(
            f"{missing_scores} subject(s) had no extractable score "
            "(paired_gaussian_residual / gaussian_residual not run and "
            "residual_roi_concentration absent); "
            "excluded from AUROC computation."
        )
    if len(fcd_scores) < 5 or len(hc_scores) < 5:
        limitations.append(
            "Fewer than 5 scored subjects in one or both classes; "
            "AUROC estimate is unreliable at this sample size."
        )
    if data_quality_summary["n_with_data_quality"] == 0:
        limitations.append(
            "No known FCD/HC subjects with data-quality audit evidence; "
            "MRI conversion, canonicalization, and geometry-resampling risks "
            "cannot be evaluated from this output root."
        )
    elif data_quality_summary["n_upload_readiness_fail"] > 0:
        limitations.append(
            f"{data_quality_summary['n_upload_readiness_fail']} known subject(s) "
            "failed upload-readiness checks; paper-grade 3DGS subtraction is "
            "not supported for the full evaluated cohort."
        )
    if data_quality_summary.get("n_dicom_conversion_fail", 0) > 0:
        limitations.append(
            f"{data_quality_summary['n_dicom_conversion_fail']} known DICOM-converted "
            "subject(s) failed conversion safety checks; wrong-series inputs "
            "cannot support structural MRI-to-3DGS claims."
        )
    elif data_quality_summary.get("n_dicom_conversion_review", 0) > 0:
        limitations.append(
            f"{data_quality_summary['n_dicom_conversion_review']} known DICOM-converted "
            "subject(s) require conversion sidecar review; paper-facing claims "
            "need pass-level conversion safety evidence."
        )
    if not candidate_manual_rows:
        limitations.append(
            "No FCD subjects with residual-candidate manual-overlap metrics; "
            "candidate localization cannot be validated from this output root."
        )
    if not hc_candidate_volumes:
        limitations.append(
            "No HC subjects with residual-candidate volume metrics; "
            "control false-positive candidate burden cannot be estimated."
        )
    if not splat_rows:
        limitations.append(
            "No subjects with splat reconstruction audit metrics; MRI-to-Gaussian "
            "representation fidelity cannot be evaluated from this output root."
        )
    elif splat_reconstruction_quality["n_readiness_fail"] > 0:
        limitations.append(
            f"{splat_reconstruction_quality['n_readiness_fail']} known subject(s) "
            "failed MRI-to-3DGS representation-readiness checks."
        )
    elif splat_reconstruction_quality["n_mri_to_3dgs_claim_allowed"] < (
        n_fcd + n_hc
    ):
        limitations.append(
            "Not every known subject has an artifact-level MRI-to-3DGS claim "
            "allowed by the splat reconstruction audit."
        )
    if optimized_gaussian_field_quality["n_with_optimized_field_audit"] == 0:
        limitations.append(
            "No subjects with optimized Gaussian-field fidelity audits; the "
            "trained field-level MRI-to-3DGS representation claim cannot be "
            "evaluated from this output root."
        )
    elif optimized_gaussian_field_quality["n_readiness_fail"] > 0:
        limitations.append(
            f"{optimized_gaussian_field_quality['n_readiness_fail']} known "
            "subject(s) failed optimized Gaussian-field held-out fidelity checks."
        )
    elif optimized_gaussian_field_quality[
        "n_optimized_mri_to_3dgs_claim_allowed"
    ] < (n_fcd + n_hc):
        limitations.append(
            "Not every known subject has an optimized MRI-to-3DGS claim allowed "
            "by the held-out Gaussian-field fidelity audit."
        )
    if paired_gaussian_subtraction_quality["n_with_paired_gaussian_state"] == 0:
        limitations.append(
            "No subjects with paired Gaussian state artifacts; the core "
            "patient-vs-synthetic-normal 3DGS subtraction claim cannot be "
            "evaluated from this output root."
        )
    elif paired_gaussian_subtraction_quality.get("n_volume_sampling_mode", 0) < (
        paired_gaussian_subtraction_quality.get("n_known_subjects", 0)
    ):
        limitations.append(
            "Not every known subject uses volume-mode paired Gaussian support; "
            "surface/uniform paired splats are useful for debugging but do not "
            "support the full-brain splat subtraction paper claim."
        )
    elif paired_gaussian_subtraction_quality.get("n_expected_support_basis", 0) < (
        paired_gaussian_subtraction_quality.get("n_known_subjects", 0)
    ):
        limitations.append(
            "Not every known subject declares a supported paired Gaussian "
            "support basis; valid bases are max(patient, synthetic_normal), "
            "patient, synthetic_normal, and min(patient, synthetic_normal)."
        )
    elif paired_gaussian_subtraction_quality.get("n_unique_supported_support_basis", 0) != 1:
        limitations.append(
            "The evaluated cohort mixes paired Gaussian support bases; run union, "
            "patient, normal, or intersection support as separate ablations."
        )
    if control_gaussian_prior_quality["n_with_control_gaussian_prior_state"] == 0:
        limitations.append(
            "No subjects with control-prior Gaussian state artifacts; the "
            "patient-support healthy-control prior 3DGS subtraction fallback "
            "cannot be evaluated from this output root."
        )
    elif control_gaussian_prior_quality.get("n_same_gaussian_support", 0) < (
        control_gaussian_prior_quality.get("n_known_subjects", 0)
    ):
        limitations.append(
            "Not every known subject has same-support control-prior Gaussian "
            "evidence; patient/prior/residual splat subtraction is not proven "
            "for the full cohort."
        )
    elif control_gaussian_prior_quality.get("n_volume_sampling_mode", 0) < (
        control_gaussian_prior_quality.get("n_known_subjects", 0)
    ):
        limitations.append(
            "Not every known subject uses volume-mode control-prior Gaussian "
            "support; surface/uniform control-prior splats do not support the "
            "full-brain fallback paper claim."
        )
    if (
        optimized_gaussian_subtraction_quality["n_with_optimized_gaussian_residual"]
        == 0
    ):
        limitations.append(
            "No subjects with optimized same-support Gaussian residual artifacts; "
            "the field-level patient-vs-synthetic-normal 3DGS subtraction claim "
            "cannot be evaluated from this output root."
        )
    elif optimized_gaussian_subtraction_quality.get("n_same_gaussian_support", 0) < (
        optimized_gaussian_subtraction_quality.get("n_known_subjects", 0)
    ):
        limitations.append(
            "Not every known subject has same-support optimized Gaussian residual "
            "evidence; indexwise 3DGS subtraction is not proven for the full cohort."
        )
    elif optimized_gaussian_subtraction_quality.get(
        "n_indexwise_score_method", 0
    ) < optimized_gaussian_subtraction_quality.get("n_known_subjects", 0):
        limitations.append(
            "Not every known subject uses indexwise same-support Gaussian residual "
            "scores; optimized residuals are not uniform across the evaluated cohort."
        )
    if synthetic_normal_quality["n_with_synthetic_normal_audit"] == 0:
        limitations.append(
            "No subjects with synthetic-normal audit evidence; healthy-reference "
            "claim boundaries and subtraction readiness cannot be evaluated from "
            "this output root."
        )
    elif synthetic_normal_quality["n_subtraction_readiness_fail"] > 0:
        limitations.append(
            f"{synthetic_normal_quality['n_subtraction_readiness_fail']} known "
            "subject(s) failed synthetic-normal subtraction-readiness checks."
        )
    counterfactual_counts = synthetic_normal_quality.get("counterfactual_class_counts", {})
    has_control_prior_reference = any(
        str(key).startswith("healthy_control_population")
        for key in counterfactual_counts
    )
    if has_control_prior_reference and not control_prior_rows:
        limitations.append(
            "Control-prior synthetic normals were present but no control-prior "
            "quality metrics were extracted."
        )
    for metric_name, metric_stats in metrics_result.items():
        metric_fcd_scored = sum(
            1 for r in rows if r["group"] == "FCD" and r[metric_name] is not None
        )
        metric_hc_scored = sum(
            1 for r in rows if r["group"] == "HC" and r[metric_name] is not None
        )
        if metric_stats["n_scored"] > 0 and (metric_fcd_scored == 0 or metric_hc_scored == 0):
            limitations.append(
                f"Metric {metric_name} lacks scored subjects in one class; "
                "AUROC is not meaningful (defaulted to 0.5)."
            )

    result = {
        "n_total": len(rows),
        "n_fcd": n_fcd,
        "n_hc": n_hc,
        "n_unknown_group": n_unk,
        "n_scored": n_scored,
        "n_missing_score": missing_scores,
        "score_metric": effective_score_metric,
        "requested_score_metric": score_metric,
        "fcd_score_mean": fcd_mean,
        "hc_score_mean": hc_mean,
        "auroc_fcd_vs_hc": auroc,
        "metrics": metrics_result,
        "roi_source_summary": roi_summary,
        "data_quality_summary": data_quality_summary,
        "candidate_validation": candidate_validation,
        "splat_reconstruction_quality": splat_reconstruction_quality,
        "optimized_gaussian_field_quality": optimized_gaussian_field_quality,
        "paired_gaussian_subtraction_quality": paired_gaussian_subtraction_quality,
        "control_gaussian_prior_quality": control_gaussian_prior_quality,
        "optimized_gaussian_subtraction_quality": (
            optimized_gaussian_subtraction_quality
        ),
        "control_prior_quality": control_prior_quality,
        "synthetic_normal_quality": synthetic_normal_quality,
        "paper_readiness": paper_readiness,
        "limitations": limitations,
        "subjects": rows,
    }

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w") as fh:
            json.dump(result, fh, indent=2)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "subject", "group", "score", "score_source",
                    "roi_source", "roi_source_normalized",
                    "control_prior_quality_gate",
                    *EVALUATED_METRIC_KEYS,
                    *PAIRED_GAUSSIAN_EVIDENCE_KEYS,
                    *CONTROL_GAUSSIAN_PRIOR_EVIDENCE_KEYS,
                    *OPTIMIZED_GAUSSIAN_EVIDENCE_KEYS,
                    *DATA_QUALITY_EVIDENCE_KEYS,
                    *SPLAT_RECONSTRUCTION_EVIDENCE_KEYS,
                    *OPTIMIZED_GAUSSIAN_FIELD_AUDIT_EVIDENCE_KEYS,
                    *SYNTHETIC_NORMAL_AUDIT_EVIDENCE_KEYS,
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate FCD pipeline cohort outputs: group means and AUROC."
    )
    p.add_argument("output_root", type=Path,
                   help="Directory produced by fcd-build-end-to-end (contains index.json).")
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-csv",  type=Path, default=None)
    p.add_argument("--score-metric",
                   choices=["auto", *EVALUATED_METRIC_KEYS],
                   default="auto",
                   help="Scalar used for FCD-vs-HC AUROC. 'auto' uses the paired "
                        "ROI-to-non-ROI residual ratio for ROI-free candidate/"
                        "supervised runs, otherwise it preserves the legacy paired "
                        "residual mean fallback chain.")
    p.add_argument("--expected-fcd", type=int,
                   default=int(DEFAULT_PAPER_GATE_THRESHOLDS["expected_fcd"]),
                   help="Expected number of FCD subjects for paper-readiness gating.")
    p.add_argument("--expected-hc", type=int,
                   default=int(DEFAULT_PAPER_GATE_THRESHOLDS["expected_hc"]),
                   help="Expected number of healthy-control subjects for paper-readiness gating.")
    p.add_argument("--auroc-min", type=float,
                   default=DEFAULT_PAPER_GATE_THRESHOLDS["auroc_min"],
                   help="Minimum FCD-vs-HC AUROC required by paper-readiness gating.")
    p.add_argument("--control-prior-similarity-min", type=float,
                   default=DEFAULT_PAPER_GATE_THRESHOLDS["control_prior_similarity_min"],
                   help=(
                       "Minimum mean control-prior reference similarity required by "
                       "the healthy-reference synthesis gate."
                   ))
    p.add_argument("--splat-foreground-coverage-min", type=float,
                   default=DEFAULT_PAPER_GATE_THRESHOLDS["splat_foreground_coverage_min"],
                   help="Minimum mean MRI foreground coverage required for 3DGS fit gate.")
    p.add_argument("--splat-mip-mean-mae-max", type=float,
                   default=DEFAULT_PAPER_GATE_THRESHOLDS["splat_mip_mean_mae_max"],
                   help="Maximum mean MIP MAE allowed for 3DGS fit gate.")
    p.add_argument("--optimized-field-foreground-mae-max", type=float,
                   default=DEFAULT_PAPER_GATE_THRESHOLDS["optimized_field_foreground_mae_max"],
                   help="Maximum mean held-out foreground MAE allowed for optimized 3DGS fit gate.")
    p.add_argument("--optimized-field-all-mae-max", type=float,
                   default=DEFAULT_PAPER_GATE_THRESHOLDS["optimized_field_all_mae_max"],
                   help="Maximum mean held-out all-sample MAE allowed for optimized 3DGS fit gate.")
    p.add_argument("--paired-gaussian-min-points", type=int,
                   default=int(DEFAULT_PAPER_GATE_THRESHOLDS["paired_gaussian_min_points"]),
                   help=(
                       "Minimum per-subject paired Gaussian count required for the "
                       "same-support 3DGS subtraction gate."
                   ))
    p.add_argument("--control-gaussian-min-points", type=int,
                   default=int(DEFAULT_PAPER_GATE_THRESHOLDS["control_gaussian_min_points"]),
                   help=(
                       "Minimum per-subject control-prior Gaussian count required "
                       "for the patient-support healthy-control subtraction gate."
                   ))
    p.add_argument("--control-gaussian-min-controls", type=int,
                   default=int(DEFAULT_PAPER_GATE_THRESHOLDS["control_gaussian_min_controls"]),
                   help=(
                       "Minimum healthy-control volumes required per subject for "
                       "the control-prior Gaussian subtraction gate."
                   ))
    p.add_argument("--optimized-gaussian-min-points", type=int,
                   default=int(DEFAULT_PAPER_GATE_THRESHOLDS["optimized_gaussian_min_points"]),
                   help=(
                       "Minimum per-subject optimized Gaussian count required for "
                       "the field-level same-support 3DGS subtraction gate."
                   ))
    p.add_argument("--localization-mean-manual-coverage-min", type=float,
                   default=DEFAULT_PAPER_GATE_THRESHOLDS["localization_mean_manual_coverage_min"],
                   help="Minimum mean residual-candidate coverage of manual FCD ROIs.")
    p.add_argument("--hc-candidate-foreground-fraction-max", type=float,
                   default=DEFAULT_PAPER_GATE_THRESHOLDS["hc_candidate_foreground_fraction_max"],
                   help="Maximum mean HC residual-candidate foreground fraction.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = evaluate_output_root(
        output_root=args.output_root,
        output_json=args.output_json,
        output_csv=args.output_csv,
        score_metric=args.score_metric,
        gate_thresholds={
            "expected_fcd": args.expected_fcd,
            "expected_hc": args.expected_hc,
            "auroc_min": args.auroc_min,
            "control_prior_similarity_min": args.control_prior_similarity_min,
            "splat_foreground_coverage_min": args.splat_foreground_coverage_min,
            "splat_mip_mean_mae_max": args.splat_mip_mean_mae_max,
            "optimized_field_foreground_mae_max": (
                args.optimized_field_foreground_mae_max
            ),
            "optimized_field_all_mae_max": args.optimized_field_all_mae_max,
            "paired_gaussian_min_points": args.paired_gaussian_min_points,
            "control_gaussian_min_points": args.control_gaussian_min_points,
            "control_gaussian_min_controls": args.control_gaussian_min_controls,
            "optimized_gaussian_min_points": args.optimized_gaussian_min_points,
            "localization_mean_manual_coverage_min": (
                args.localization_mean_manual_coverage_min
            ),
            "hc_candidate_foreground_fraction_max": (
                args.hc_candidate_foreground_fraction_max
            ),
        },
    )
    print(json.dumps(
        {k: v for k, v in result.items() if k != "subjects"},
        indent=2,
    ))


# ---------------------------------------------------------------------------
# Late import to keep module importable without numpy installed at parse time
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402


if __name__ == "__main__":
    main()
