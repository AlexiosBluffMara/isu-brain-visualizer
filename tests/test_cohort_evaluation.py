"""Tests for cohort-level evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fcd_prototype.cohort_evaluation import evaluate_output_root, _mannwhitney_auroc


def _data_quality(
    *,
    status: str = "pass",
    blockers: list[str] | None = None,
    review_items: list[str] | None = None,
    t1w_resample: bool = False,
    roi_resample: bool = False,
    roi_resampled_to_base: bool = False,
    dicom_conversion_safety: dict | None = None,
) -> dict:
    blockers = blockers or []
    review_items = review_items or []
    can_process = status != "fail" and not blockers
    audit = {
        "status": "warning" if review_items or t1w_resample or roi_resample else "ok",
        "warning_count": len(review_items) + int(t1w_resample) + int(roi_resample),
        "warnings": list(review_items),
        "upload_readiness": {
            "status": status,
            "blocking_issue_count": len(blockers),
            "review_issue_count": len(review_items) + int(t1w_resample) + int(roi_resample),
            "blockers": blockers,
            "review_items": review_items,
            "primary_modalities": ["flair"],
            "can_run_3dgs": can_process,
            "can_subtract_patient_normal": can_process,
        },
        "comparisons": {
            "t1w_to_flair": {"needs_resample_to_reference": t1w_resample},
            "flair_roi_to_flair": {"needs_resample_to_reference": roi_resample},
        },
        "roi_processing": {"resampled_to_base": roi_resampled_to_base},
    }
    if dicom_conversion_safety is not None:
        audit["dicom_conversion_safety"] = dicom_conversion_safety
    return audit


def _optimized_field_audit_meta(
    *,
    status: str = "pass",
    claim_allowed: bool = True,
    foreground_mae: float = 0.05,
    all_mae: float = 0.04,
    blockers: list[str] | None = None,
) -> dict:
    blockers = blockers or []
    return {
        "schema": "gaussian_field_fidelity_audit_v1",
        "sample_count": 8192,
        "seed": 1051,
        "device": "cpu",
        "affine_mismatch": False,
        "affine_missing": False,
        "sample_counts": {
            "total": 8192,
            "foreground": 6144,
            "background": 2048,
        },
        "metrics": {
            "all": {
                "count": 8192,
                "mae": all_mae,
                "rmse": all_mae * 1.4,
                "psnr": 28.0,
                "correlation": 0.95,
            },
            "foreground": {
                "count": 6144,
                "mae": foreground_mae,
                "rmse": foreground_mae * 1.4,
                "psnr": 28.0,
                "correlation": 0.95,
            },
            "background": {
                "count": 2048,
                "mae": all_mae,
                "rmse": all_mae * 1.4,
                "psnr": 28.0,
                "correlation": 0.95,
            },
        },
        "checkpoint_metadata": {
            "training_objective": "mip_projection_mse_plus_volume_sample_mse",
            "volume_loss_weight": 0.25,
            "volume_sample_count": 8192,
            "volume_path": "patient_flair.nii.gz",
        },
        "readiness_gate": {
            "status": status,
            "optimized_mri_to_3dgs_claim_allowed": claim_allowed,
            "blockers": blockers,
            "warnings": [],
        },
    }


def _make_fake_output_root(tmp_path: Path, subjects: list[dict]) -> Path:
    """Create a fake output_root with index.json and per-subject manifests.

    subjects = [
        {"id": "sub-00001", "group": "FCD", "score": 0.8},
        {"id": "sub-00002", "group": "HC", "score": 0.3},
        ...
    ]
    """
    root = tmp_path / "output"
    root.mkdir(parents=True, exist_ok=True)

    index_subjects = []
    for subj in subjects:
        sub_id = subj["id"]
        group = subj["group"]
        score = subj.get("score")
        paired_score = subj.get("paired_score")
        paired_metrics = subj.get("paired_metrics", {})
        paired_gaussian_artifacts = subj.get("paired_gaussian_artifacts", False)
        paired_n_gaussians = subj.get("paired_n_gaussians", 60_000)
        paired_state_schema = subj.get(
            "paired_gaussian_state_schema",
            "paired_gaussian_residual_v1",
        )
        paired_same_support = subj.get("paired_gaussian_same_support", True)
        paired_sampling_mode = subj.get("paired_gaussian_sampling_mode", "volume")
        paired_support_basis = subj.get(
            "paired_gaussian_support_basis",
            "max(patient, synthetic_normal)",
        )
        paired_support_basis_mode = subj.get("paired_gaussian_support_basis_mode")
        control_gaussian_artifacts = subj.get("control_gaussian_artifacts", False)
        control_gaussian_n_gaussians = subj.get("control_gaussian_n_gaussians", 60_000)
        control_gaussian_n_controls = subj.get("control_gaussian_n_controls", 2)
        control_gaussian_min_controls = subj.get("control_gaussian_min_controls", 2)
        control_gaussian_state_schema = subj.get(
            "control_gaussian_state_schema",
            "patient_control_gaussian_prior_v1",
        )
        control_gaussian_same_support = subj.get(
            "control_gaussian_same_support",
            True,
        )
        control_gaussian_sampling_mode = subj.get(
            "control_gaussian_sampling_mode",
            "volume",
        )
        control_gaussian_default_score = (
            paired_score if paired_score is not None else 0.2
        )
        optimized_gaussian_artifacts = subj.get("optimized_gaussian_artifacts", False)
        optimized_n_patient = subj.get(
            "optimized_n_patient",
            subj.get("optimized_n_gaussians", 60_000),
        )
        optimized_n_normal = subj.get(
            "optimized_n_normal",
            subj.get("optimized_n_gaussians", 60_000),
        )
        optimized_same_support = subj.get("optimized_gaussian_same_support", True)
        optimized_claim_allowed = subj.get(
            "optimized_gaussian_subtraction_claim_allowed",
            True,
        )
        optimized_score_method = subj.get(
            "optimized_gaussian_score_method",
            "indexwise_same_support_intensity",
        )
        optimized_heatmap_method = subj.get("optimized_gaussian_heatmap_method", "field")
        optimized_field_audit_meta = subj.get(
            "optimized_gaussian_field_audit_meta",
            _optimized_field_audit_meta() if optimized_gaussian_artifacts else None,
        )
        optimized_support_diffs = subj.get(
            "optimized_gaussian_support_diffs",
            {
                "means_norm_max_abs_diff": 0.0,
                "log_scales_max_abs_diff": 0.0,
                "quat_wxyz_max_abs_diff": 0.0,
            },
        )
        data_quality = subj.get("data_quality")
        metrics = subj.get("metrics", {})
        residual_candidate_meta = subj.get("residual_candidate_meta")
        splat_reconstruction_audit_meta = subj.get("splat_reconstruction_audit_meta")
        if splat_reconstruction_audit_meta:
            splat_reconstruction_audit_meta = dict(splat_reconstruction_audit_meta)
            splat_reconstruction_audit_meta.setdefault(
                "mri_to_3dgs_readiness_status",
                "pass",
            )
            splat_reconstruction_audit_meta.setdefault(
                "mri_to_3dgs_readiness_blocker_count",
                0,
            )
            splat_reconstruction_audit_meta.setdefault(
                "mri_to_3dgs_readiness_warning_count",
                0,
            )
            splat_reconstruction_audit_meta.setdefault(
                "mri_to_3dgs_claim_allowed",
                True,
            )
            splat_reconstruction_audit_meta.setdefault(
                "mri_to_3dgs_claim_class",
                "volumetric_mri_gaussian_representation",
            )
        control_prior_meta = subj.get("control_prior_meta")
        synthetic_normal_audit_meta = subj.get(
            "synthetic_normal_audit_meta",
            {
                "counterfactual_class": "healthy_control_population_prior_roi_hybrid",
                "claim_boundary": "not_patient_conditioned_inpainting",
                "synthetic_normal_claim_class": "healthy_control_population_reference",
                "subtraction_readiness_status": "pass",
                "subtraction_readiness_blocker_count": 0,
                "subtraction_readiness_warning_count": 0,
                "gaussian_subtraction_claim_allowed": True,
                "patient_conditioned_inpainting": False,
                "patient_conditioned_counterfactual_claim_allowed": False,
                "same_grid_before_resample": True,
                "synthetic_resampled_to_patient_grid": False,
                "outside_roi_similarity_1_minus_mae": 0.99,
                "inside_to_outside_mae_ratio": 2.0,
            },
        )
        roi_source = subj.get("roi_source")

        # Write per-subject manifest
        sub_dir = root / sub_id
        sub_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "subject": sub_id,
            "group": group,
            "stage_status": {"synthetic_normal": "ok", "residual": "ok"},
        }

        if score is not None:
            manifest["gaussian_residual_meta"] = {
                "residual_p99": score,
                "residual_mean": score * 0.8,
            }
        if optimized_gaussian_artifacts:
            optimized_score = subj.get(
                "optimized_score",
                score if score is not None else (paired_score or 0.01),
            )
            manifest.setdefault("gaussian_residual_meta", {}).update({
                "n_patient": optimized_n_patient,
                "n_normal": optimized_n_normal,
                "heatmap_method": optimized_heatmap_method,
                "residual_score_method": optimized_score_method,
                "same_gaussian_support": optimized_same_support,
                "gaussian_subtraction_claim_allowed": optimized_claim_allowed,
                "residual_p99": optimized_score,
                "residual_mean": optimized_score * 0.8,
                **optimized_support_diffs,
            })
        # else no score, missing

        if paired_score is not None:
            manifest["paired_gaussian_residual_meta"] = {
                "residual_mean": paired_score,
                "residual_p95": paired_score * 1.4,
                "residual_p99": paired_score * 1.6,
                "residual_nonzero_fraction": paired_score * 0.5,
                "residual_normalization_value": paired_score * 2.0,
            }

        if paired_metrics:
            manifest.setdefault("paired_gaussian_residual_meta", {}).update(paired_metrics)
        if paired_gaussian_artifacts:
            manifest.setdefault("paired_gaussian_residual_meta", {}).update({
                "method": "paired_correspondence",
                "same_gaussian_support": paired_same_support,
                "gaussian_state_schema": paired_state_schema,
                "sampling_mode": paired_sampling_mode,
                "support_basis_mode": paired_support_basis_mode,
                "support_basis": paired_support_basis,
                "n_gaussians": paired_n_gaussians,
            })
        if control_gaussian_artifacts:
            manifest["control_gaussian_prior_meta"] = {
                "method": "patient_support_control_prior_zscore",
                "gaussian_state_schema": control_gaussian_state_schema,
                "same_gaussian_support": control_gaussian_same_support,
                "healthy_reference": "affine_resampled_control_gaussian_prior",
                "registration_method": "affine_header_resample_to_patient_grid",
                "intensity_matching": "zscore_match_over_patient_foreground_excluding_roi",
                "n_controls": control_gaussian_n_controls,
                "min_controls": control_gaussian_min_controls,
                "n_gaussians": control_gaussian_n_gaussians,
                "sampling_mode": control_gaussian_sampling_mode,
                "residual_z_mean": subj.get(
                    "control_gaussian_residual_z_mean",
                    control_gaussian_default_score,
                ),
                "residual_z_p95": subj.get("control_gaussian_residual_z_p95", 1.0),
                "residual_z_p99": subj.get("control_gaussian_residual_z_p99", 2.0),
                "residual_norm_mean": subj.get("control_gaussian_residual_norm_mean", 0.1),
                "residual_norm_p99": subj.get("control_gaussian_residual_norm_p99", 0.9),
                "roi_metrics_independent_validation": subj.get(
                    "control_gaussian_roi_metrics_independent_validation",
                    False,
                ),
            }
        if residual_candidate_meta:
            manifest["residual_candidate_meta"] = residual_candidate_meta
        if splat_reconstruction_audit_meta:
            manifest["splat_reconstruction_audit_meta"] = splat_reconstruction_audit_meta
        if isinstance(optimized_field_audit_meta, dict):
            manifest["optimized_gaussian_field_audit_meta"] = optimized_field_audit_meta
        if control_prior_meta:
            manifest["control_prior_meta"] = control_prior_meta
        if synthetic_normal_audit_meta:
            manifest["synthetic_normal_audit_meta"] = synthetic_normal_audit_meta
        if roi_source is not None:
            manifest["roi_source"] = roi_source
        if data_quality is True:
            manifest["data_quality"] = _data_quality()
            manifest["stage_status"]["data_quality"] = "ok"
        elif isinstance(data_quality, dict):
            manifest["data_quality"] = data_quality
            dq_status = data_quality.get("upload_readiness", {}).get("status", "unknown")
            manifest["stage_status"]["data_quality"] = (
                "ok" if dq_status == "pass" else f"{dq_status}: data quality"
            )

        # Add artifact metrics file if provided
        artifacts = {}
        if metrics:
            artifact_file = sub_dir / "gaussian_residual_metrics.json"
            with artifact_file.open("w") as fh:
                json.dump(metrics, fh)
            artifacts["gaussian_residual_metrics"] = "gaussian_residual_metrics.json"
        if paired_gaussian_artifacts:
            paired_dir = sub_dir / "paired_gaussian_residual"
            paired_dir.mkdir(parents=True, exist_ok=True)
            paired_artifact_names = {
                "paired_patient_splat": "paired_patient_splat.ply",
                "paired_normal_splat": "paired_normal_splat.ply",
                "paired_gaussian_residual_splat": "paired_gaussian_residual_splat.ply",
                "paired_gaussian_residual_heatmap": "paired_gaussian_residual_heatmap.nii.gz",
                "paired_gaussian_state": "paired_gaussian_state.npz",
            }
            for key, filename in paired_artifact_names.items():
                path = paired_dir / filename
                if key == "paired_gaussian_state":
                    np.savez_compressed(
                        path,
                        state_schema=np.array(paired_state_schema),
                        sampling_mode=np.array(paired_sampling_mode),
                        support_basis_mode=np.array(
                            paired_support_basis_mode or ""
                        ),
                        support_basis=np.array(paired_support_basis),
                    )
                else:
                    path.write_bytes(b"placeholder")
                artifacts[key] = f"paired_gaussian_residual/{filename}"

            paired_artifact_payload = {
                "method": "paired_correspondence",
                "same_gaussian_support": paired_same_support,
                "gaussian_state_schema": paired_state_schema,
                "sampling_mode": paired_sampling_mode,
                "support_basis_mode": paired_support_basis_mode,
                "support_basis": paired_support_basis,
                "n_gaussians": paired_n_gaussians,
                "residual_mean": paired_score if paired_score is not None else 0.0,
                "artifacts": paired_artifact_names,
            }
            paired_artifact_payload.update(subj.get("paired_artifact_metrics", {}))
            paired_artifact_payload.setdefault("artifacts", paired_artifact_names)
            paired_artifact_file = paired_dir / "paired_gaussian_residual_metrics.json"
            with paired_artifact_file.open("w") as fh:
                json.dump(paired_artifact_payload, fh)
            artifacts["paired_gaussian_residual_metrics"] = (
                "paired_gaussian_residual/paired_gaussian_residual_metrics.json"
            )
        if control_gaussian_artifacts:
            control_dir = sub_dir / "control_gaussian_prior"
            control_dir.mkdir(parents=True, exist_ok=True)
            control_artifact_names = {
                "control_prior_patient_splat": "control_prior_patient_splat.ply",
                "control_prior_mean_splat": "control_prior_mean_splat.ply",
                "control_prior_z_residual_splat": "control_prior_z_residual_splat.ply",
                "control_prior_z_residual_heatmap": "control_prior_z_residual_heatmap.nii.gz",
                "control_prior_z_scores": "control_prior_z_scores.npy",
                "control_gaussian_prior_state": "control_gaussian_prior_state.npz",
            }
            for key, filename in control_artifact_names.items():
                path = control_dir / filename
                if key == "control_gaussian_prior_state":
                    np.savez_compressed(
                        path,
                        state_schema=np.array(control_gaussian_state_schema),
                        method=np.array("patient_support_control_prior_zscore"),
                        sampling_mode=np.array(control_gaussian_sampling_mode),
                    )
                else:
                    path.write_bytes(b"placeholder")
                artifacts[key] = f"control_gaussian_prior/{filename}"

            control_artifact_payload = {
                "method": "patient_support_control_prior_zscore",
                "gaussian_state_schema": control_gaussian_state_schema,
                "same_gaussian_support": control_gaussian_same_support,
                "healthy_reference": "affine_resampled_control_gaussian_prior",
                "registration_method": "affine_header_resample_to_patient_grid",
                "intensity_matching": "zscore_match_over_patient_foreground_excluding_roi",
                "n_controls": control_gaussian_n_controls,
                "min_controls": control_gaussian_min_controls,
                "n_gaussians": control_gaussian_n_gaussians,
                "sampling_mode": control_gaussian_sampling_mode,
                "residual_z_mean": subj.get(
                    "control_gaussian_residual_z_mean",
                    control_gaussian_default_score,
                ),
                "residual_z_p95": subj.get("control_gaussian_residual_z_p95", 1.0),
                "residual_z_p99": subj.get("control_gaussian_residual_z_p99", 2.0),
                "residual_norm_mean": subj.get("control_gaussian_residual_norm_mean", 0.1),
                "residual_norm_p99": subj.get("control_gaussian_residual_norm_p99", 0.9),
                "artifacts": control_artifact_names,
            }
            control_artifact_payload.update(
                subj.get("control_gaussian_artifact_metrics", {})
            )
            control_artifact_file = control_dir / "control_gaussian_prior_metrics.json"
            with control_artifact_file.open("w") as fh:
                json.dump(control_artifact_payload, fh)
            artifacts["control_gaussian_prior_metrics"] = (
                "control_gaussian_prior/control_gaussian_prior_metrics.json"
            )
        if subj.get("paired_artifact_metrics") and not paired_gaussian_artifacts:
            paired_artifact_file = sub_dir / "paired_gaussian_residual_metrics.json"
            with paired_artifact_file.open("w") as fh:
                json.dump(subj["paired_artifact_metrics"], fh)
            artifacts["paired_gaussian_residual_metrics"] = "paired_gaussian_residual_metrics.json"
        if optimized_gaussian_artifacts:
            optimized_dir = sub_dir / "gaussian_residual"
            optimized_dir.mkdir(parents=True, exist_ok=True)
            optimized_artifact_names = {
                "gaussian_residual_splat": "gaussian_residual_splat.ply",
                "gaussian_residual_heatmap": "gaussian_residual_heatmap.nii.gz",
                "gaussian_residual_metrics": "gaussian_residual_metrics.json",
                "gaussian_residual_scores": "gaussian_residual_scores.npy",
            }
            for key, filename in optimized_artifact_names.items():
                if key == "gaussian_residual_metrics":
                    continue
                path = optimized_dir / filename
                path.write_bytes(b"placeholder")
                artifacts[key] = f"gaussian_residual/{filename}"

            optimized_score = subj.get(
                "optimized_score",
                score if score is not None else (paired_score or 0.01),
            )
            optimized_artifact_payload = {
                "n_patient": optimized_n_patient,
                "n_normal": optimized_n_normal,
                "heatmap_method": optimized_heatmap_method,
                "residual_score_method": optimized_score_method,
                "same_gaussian_support": optimized_same_support,
                "gaussian_subtraction_claim_allowed": optimized_claim_allowed,
                "residual_p99": optimized_score,
                "residual_mean": optimized_score * 0.8,
                "artifacts": optimized_artifact_names,
                **optimized_support_diffs,
            }
            optimized_artifact_payload.update(subj.get("optimized_artifact_metrics", {}))
            optimized_artifact_file = optimized_dir / "gaussian_residual_metrics.json"
            with optimized_artifact_file.open("w") as fh:
                json.dump(optimized_artifact_payload, fh)
            artifacts["gaussian_residual_metrics"] = (
                "gaussian_residual/gaussian_residual_metrics.json"
            )
        if subj.get("residual_candidate_artifact"):
            candidate_dir = sub_dir / "residual_candidates"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate_artifact_file = candidate_dir / "residual_candidate_manifest.json"
            with candidate_artifact_file.open("w") as fh:
                json.dump(subj["residual_candidate_artifact"], fh)
            artifacts["residual_candidate_manifest"] = (
                "residual_candidates/residual_candidate_manifest.json"
            )
        if artifacts:
            manifest["artifacts"] = artifacts

        manifest_path = sub_dir / "manifest.json"
        with manifest_path.open("w") as fh:
            json.dump(manifest, fh)

        index_entry = {
            "subject": sub_id,
            "group": group,
            "roi_available": True,
            "modalities": ["FLAIR"],
            "stage_status": manifest["stage_status"],
            "synthetic_normal_source": "placeholder",
            "manifest": f"{sub_id}/manifest.json",
        }
        if roi_source is not None:
            index_entry["roi_source"] = roi_source
        index_subjects.append(index_entry)

    # Write index
    index = {
        "generated": "2024-01-01T00:00:00Z",
        "dataset": "test",
        "subjects": index_subjects,
    }
    index_path = root / "index.json"
    with index_path.open("w") as fh:
        json.dump(index, fh)

    return root


def test_mannwhitney_auroc_perfect_separation():
    """AUROC should be 1.0 when pos > all neg."""
    auroc = _mannwhitney_auroc([1.0, 0.9, 0.8], [0.3, 0.2, 0.1])
    assert auroc == 1.0


def test_mannwhitney_auroc_perfect_overlap():
    """AUROC should be 0.5 when pos and neg are identical."""
    auroc = _mannwhitney_auroc([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert auroc == 0.5


def test_mannwhitney_auroc_empty_lists():
    """AUROC should be 0.5 when either list is empty."""
    assert _mannwhitney_auroc([], [0.1, 0.2]) == 0.5
    assert _mannwhitney_auroc([0.1, 0.2], []) == 0.5
    assert _mannwhitney_auroc([], []) == 0.5


def test_evaluate_output_root_reads_index(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {"id": "sub-00001", "group": "FCD", "score": 0.8},
        {"id": "sub-00002", "group": "HC", "score": 0.2},
    ])

    result = evaluate_output_root(root)
    assert result["n_total"] == 2
    assert result["n_fcd"] == 1
    assert result["n_hc"] == 1


def test_evaluate_output_root_scores_extracted(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {"id": "sub-00001", "group": "FCD", "score": 0.8},
        {"id": "sub-00002", "group": "HC", "score": 0.2},
    ])

    result = evaluate_output_root(root)
    assert result["n_scored"] == 2
    assert result["n_missing_score"] == 0


def test_evaluate_output_root_prefers_paired_score(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {"id": "sub-00001", "group": "FCD", "score": 0.1, "paired_score": 0.8},
        {"id": "sub-00002", "group": "HC", "score": 0.9, "paired_score": 0.2},
    ])

    result = evaluate_output_root(root)
    assert result["auroc_fcd_vs_hc"] == 1.0
    fcd_row = next(r for r in result["subjects"] if r["subject"] == "sub-00001")
    assert fcd_row["score"] == pytest.approx(0.8)
    assert fcd_row["score_source"] == "paired_gaussian_residual_meta.residual_mean"


def test_evaluate_output_root_can_score_named_local_ratio_metric(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.01,
            "paired_metrics": {"roi_to_non_roi_ratio": 8.0},
        },
        {
            "id": "sub-fcd-02",
            "group": "FCD",
            "paired_score": 0.02,
            "paired_metrics": {"roi_to_non_roi_ratio": 7.0},
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.05,
            "paired_metrics": {"roi_to_non_roi_ratio": 2.0},
        },
        {
            "id": "sub-hc-02",
            "group": "HC",
            "paired_score": 0.04,
            "paired_metrics": {"roi_to_non_roi_ratio": 3.0},
        },
    ])

    default_result = evaluate_output_root(root)
    ratio_result = evaluate_output_root(
        root,
        score_metric="paired_roi_to_non_roi_ratio",
    )

    assert default_result["auroc_fcd_vs_hc"] == 0.0
    assert ratio_result["auroc_fcd_vs_hc"] == 1.0
    assert ratio_result["score_metric"] == "paired_roi_to_non_roi_ratio"
    fcd_row = next(r for r in ratio_result["subjects"] if r["subject"] == "sub-fcd-01")
    assert fcd_row["score"] == pytest.approx(8.0)
    assert fcd_row["score_source"] == "metrics.paired_roi_to_non_roi_ratio"
    observed = ratio_result["paper_readiness"]["gates"]["detection_auroc"]["observed"]
    assert observed["score_metric"] == "paired_roi_to_non_roi_ratio"


def test_evaluate_output_root_auto_scores_roi_free_runs_with_local_ratio(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.01,
            "roi_source": "candidate",
            "paired_metrics": {"roi_to_non_roi_ratio": 8.0},
        },
        {
            "id": "sub-fcd-02",
            "group": "FCD",
            "paired_score": 0.02,
            "roi_source": "candidate",
            "paired_metrics": {"roi_to_non_roi_ratio": 7.0},
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.05,
            "roi_source": "candidate",
            "paired_metrics": {"roi_to_non_roi_ratio": 2.0},
        },
        {
            "id": "sub-hc-02",
            "group": "HC",
            "paired_score": 0.04,
            "roi_source": "candidate",
            "paired_metrics": {"roi_to_non_roi_ratio": 3.0},
        },
    ])

    result = evaluate_output_root(root)

    assert result["requested_score_metric"] == "auto"
    assert result["score_metric"] == "paired_roi_to_non_roi_ratio"
    assert result["auroc_fcd_vs_hc"] == 1.0
    fcd_row = next(r for r in result["subjects"] if r["subject"] == "sub-fcd-01")
    assert fcd_row["score"] == pytest.approx(8.0)
    assert fcd_row["score_source"] == "metrics.paired_roi_to_non_roi_ratio"
    observed = result["paper_readiness"]["gates"]["detection_auroc"]["observed"]
    assert observed["score_metric"] == "paired_roi_to_non_roi_ratio"


def test_evaluate_output_root_scores_paired_only_manifests(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {"id": "sub-fcd-01", "group": "FCD", "paired_score": 0.03},
        {"id": "sub-fcd-02", "group": "FCD", "paired_score": 0.04},
        {"id": "sub-hc-01", "group": "HC", "paired_score": 0.0},
        {"id": "sub-hc-02", "group": "HC", "paired_score": 0.001},
    ])

    result = evaluate_output_root(root)
    assert result["n_scored"] == 4
    assert result["n_missing_score"] == 0
    assert result["fcd_score_mean"] > result["hc_score_mean"]
    assert result["metrics"]["paired_residual_mean"]["auroc_fcd_vs_hc"] == 1.0


def test_evaluate_output_root_missing_scores(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {"id": "sub-00001", "group": "FCD", "score": 0.8},
        {"id": "sub-00002", "group": "HC", "score": None},  # no score
    ])

    result = evaluate_output_root(root)
    assert result["n_scored"] == 1
    assert result["n_missing_score"] == 1


def test_evaluate_output_root_group_normalization(tmp_path):
    """Group labels should be normalized (fcd -> FCD, hc -> HC)."""
    root = _make_fake_output_root(tmp_path, [
        {"id": "sub-00001", "group": "fcd", "score": 0.8},
        {"id": "sub-00002", "group": "hc", "score": 0.2},
    ])

    result = evaluate_output_root(root)
    assert result["n_fcd"] == 1
    assert result["n_hc"] == 1


def test_evaluate_output_root_computes_auroc(tmp_path):
    """AUROC should separate FCD (high scores) from HC (low scores)."""
    root = _make_fake_output_root(tmp_path, [
        {"id": "sub-fcd-01", "group": "FCD", "score": 0.9},
        {"id": "sub-fcd-02", "group": "FCD", "score": 0.8},
        {"id": "sub-fcd-03", "group": "FCD", "score": 0.85},
        {"id": "sub-hc-01", "group": "HC", "score": 0.2},
        {"id": "sub-hc-02", "group": "HC", "score": 0.15},
        {"id": "sub-hc-03", "group": "HC", "score": 0.25},
    ])

    result = evaluate_output_root(root)
    assert result["auroc_fcd_vs_hc"] >= 0.9, "Should have high AUROC with clear separation"


def test_evaluate_output_root_mean_scores(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {"id": "sub-fcd-01", "group": "FCD", "score": 0.8},
        {"id": "sub-fcd-02", "group": "FCD", "score": 0.6},
        {"id": "sub-hc-01", "group": "HC", "score": 0.3},
        {"id": "sub-hc-02", "group": "HC", "score": 0.2},
    ])

    result = evaluate_output_root(root)
    assert result["fcd_score_mean"] == pytest.approx(0.7, abs=0.01)
    assert result["hc_score_mean"] == pytest.approx(0.25, abs=0.01)


def test_evaluate_output_root_json_output(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {"id": "sub-00001", "group": "FCD", "score": 0.8},
    ])
    output_json = tmp_path / "result.json"

    evaluate_output_root(root, output_json=output_json)
    assert output_json.exists()

    with output_json.open() as fh:
        result = json.load(fh)
    assert result["n_total"] == 1


def test_evaluate_output_root_csv_output(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {"id": "sub-00001", "group": "FCD", "score": 0.8, "paired_score": 0.9},
        {"id": "sub-00002", "group": "HC", "score": 0.2, "paired_score": 0.1},
    ])
    output_csv = tmp_path / "result.csv"

    evaluate_output_root(root, output_csv=output_csv)
    assert output_csv.exists()

    lines = output_csv.read_text().strip().split("\n")
    assert len(lines) == 3  # header + 2 subjects
    assert "paired_residual_mean" in lines[0]
    assert "paired_gaussian_state_present" in lines[0]
    assert "data_quality_upload_readiness_status" in lines[0]
    assert "splat_mri_to_3dgs_claim_allowed" in lines[0]
    assert "control_prior_quality_gate" in lines[0]


def test_evaluate_output_root_no_index_raises(tmp_path):
    empty_root = tmp_path / "empty"
    empty_root.mkdir()

    with pytest.raises(FileNotFoundError):
        evaluate_output_root(empty_root)


def test_evaluate_output_root_limitations_on_few_samples(tmp_path):
    """Should add limitation note when fewer than 5 subjects per class."""
    root = _make_fake_output_root(tmp_path, [
        {"id": "sub-fcd-01", "group": "FCD", "score": 0.8},
        {"id": "sub-hc-01", "group": "HC", "score": 0.2},
    ])

    result = evaluate_output_root(root)
    assert any("sample size" in lim.lower() for lim in result["limitations"])


def test_evaluate_output_root_limitations_on_missing_group(tmp_path):
    """Should add limitation when one group is missing."""
    root = _make_fake_output_root(tmp_path, [
        {"id": "sub-fcd-01", "group": "Fcd", "score": 0.8},
        {"id": "sub-fcd-02", "group": "FCD", "score": 0.7},
    ])

    result = evaluate_output_root(root)
    assert result["n_hc"] == 0
    assert any("HC" in lim for lim in result["limitations"])


def test_evaluate_output_root_extracts_all_metrics(tmp_path):
    """Should extract per-subject metric fields from manifest and artifact file."""
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "score": 0.9,
            "metrics": {
                "residual_p95": 0.85,
                "roi_to_non_roi_ratio": 1.2,
            },
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "score": 0.2,
            "metrics": {
                "residual_p95": 0.18,
                "roi_to_non_roi_ratio": 0.8,
            },
        },
    ])

    result = evaluate_output_root(root)
    # Check per-subject metric fields
    fcd_row = next(r for r in result["subjects"] if r["subject"] == "sub-fcd-01")
    assert fcd_row["residual_mean"] == pytest.approx(0.9 * 0.8)
    assert fcd_row["residual_p99"] == pytest.approx(0.9)
    assert fcd_row["residual_p95"] == pytest.approx(0.85)
    assert fcd_row["roi_to_non_roi_ratio"] == pytest.approx(1.2)


def test_evaluate_output_root_extracts_paired_metrics(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.05,
            "paired_metrics": {
                "roi_to_non_roi_ratio": 8.0,
                "roi_residual_concentration": 0.7,
            },
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.0,
            "paired_artifact_metrics": {
                "residual_mean": 0.0,
                "residual_p95": 0.0,
                "residual_p99": 0.0,
                "residual_nonzero_fraction": 0.0,
                "residual_normalization_value": 0.0,
                "roi_to_non_roi_ratio": 1.0,
                "roi_residual_concentration": 0.0,
            },
        },
    ])

    result = evaluate_output_root(root)
    fcd_row = next(r for r in result["subjects"] if r["subject"] == "sub-fcd-01")
    assert fcd_row["paired_residual_mean"] == pytest.approx(0.05)
    assert fcd_row["paired_residual_p95"] == pytest.approx(0.07)
    assert fcd_row["paired_residual_p99"] == pytest.approx(0.08)
    assert fcd_row["paired_residual_nonzero_fraction"] == pytest.approx(0.025)
    assert fcd_row["paired_residual_normalization_value"] == pytest.approx(0.1)
    assert fcd_row["paired_roi_to_non_roi_ratio"] == pytest.approx(8.0)
    assert fcd_row["paired_roi_residual_concentration"] == pytest.approx(0.7)

    assert "paired_residual_nonzero_fraction" in result["metrics"]
    assert result["metrics"]["paired_residual_mean"]["n_scored"] == 2


def test_evaluate_output_root_extracts_paired_gaussian_evidence(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.05,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 123,
        },
        {"id": "sub-hc-01", "group": "HC", "paired_score": 0.01},
    ])

    result = evaluate_output_root(root)
    fcd_row = next(r for r in result["subjects"] if r["subject"] == "sub-fcd-01")
    hc_row = next(r for r in result["subjects"] if r["subject"] == "sub-hc-01")
    quality = result["paired_gaussian_subtraction_quality"]

    assert fcd_row["paired_gaussian_state_present"] is True
    assert fcd_row["paired_gaussian_state_schema"] == "paired_gaussian_residual_v1"
    assert fcd_row["paired_gaussian_same_support"] is True
    assert fcd_row["paired_gaussian_method"] == "paired_correspondence"
    assert fcd_row["paired_gaussian_sampling_mode"] == "volume"
    assert fcd_row["paired_gaussian_support_basis"] == "max(patient, synthetic_normal)"
    assert fcd_row["paired_gaussian_n_gaussians"] == pytest.approx(123)
    assert fcd_row["paired_patient_splat_present"] is True
    assert fcd_row["paired_normal_splat_present"] is True
    assert fcd_row["paired_residual_splat_present"] is True
    assert fcd_row["paired_residual_heatmap_present"] is True
    assert hc_row["paired_gaussian_state_present"] is False
    assert quality["n_with_paired_gaussian_state"] == 1
    assert quality["n_same_gaussian_support"] == 1
    assert quality["n_valid_state_schema"] == 1
    assert quality["n_volume_sampling_mode"] == 1
    assert quality["n_expected_support_basis"] == 1
    assert quality["n_unique_supported_support_basis"] == 1
    assert quality["sampling_mode_counts"] == {"missing": 1, "volume": 1}
    assert quality["support_basis_counts"] == {
        "max(patient, synthetic_normal)": 1,
        "missing": 1,
    }
    assert quality["n_with_patient_normal_residual_splats"] == 1
    assert quality["min_n_gaussians"] == pytest.approx(123)


def test_evaluate_output_root_extracts_control_gaussian_prior_evidence(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.01,
            "control_gaussian_artifacts": True,
            "control_gaussian_n_gaussians": 321,
            "control_gaussian_n_controls": 3,
            "control_gaussian_residual_z_mean": 4.2,
            "control_gaussian_residual_z_p99": 9.9,
        },
        {"id": "sub-hc-01", "group": "HC", "paired_score": 0.01},
    ])

    result = evaluate_output_root(root)
    fcd_row = next(r for r in result["subjects"] if r["subject"] == "sub-fcd-01")
    hc_row = next(r for r in result["subjects"] if r["subject"] == "sub-hc-01")
    quality = result["control_gaussian_prior_quality"]

    assert fcd_row["score"] == pytest.approx(4.2)
    assert fcd_row["score_source"] == "control_gaussian_prior_meta.residual_z_mean"
    assert fcd_row["control_gaussian_residual_z_mean"] == pytest.approx(4.2)
    assert fcd_row["control_gaussian_residual_z_p99"] == pytest.approx(9.9)
    assert fcd_row["control_gaussian_prior_state_present"] is True
    assert (
        fcd_row["control_gaussian_prior_state_schema"]
        == "patient_control_gaussian_prior_v1"
    )
    assert fcd_row["control_gaussian_prior_same_support"] is True
    assert fcd_row["control_gaussian_prior_method"] == (
        "patient_support_control_prior_zscore"
    )
    assert fcd_row["control_gaussian_prior_sampling_mode"] == "volume"
    assert fcd_row["control_gaussian_prior_n_controls"] == pytest.approx(3)
    assert fcd_row["control_gaussian_prior_n_gaussians"] == pytest.approx(321)
    assert fcd_row["control_gaussian_prior_patient_splat_present"] is True
    assert fcd_row["control_gaussian_prior_mean_splat_present"] is True
    assert fcd_row["control_gaussian_prior_residual_splat_present"] is True
    assert fcd_row["control_gaussian_prior_heatmap_present"] is True
    assert fcd_row["control_gaussian_prior_scores_present"] is True
    assert fcd_row["control_gaussian_prior_metrics_present"] is True
    assert hc_row["control_gaussian_prior_state_present"] is False
    assert quality["n_with_control_gaussian_prior_state"] == 1
    assert quality["n_same_gaussian_support"] == 1
    assert quality["n_valid_state_schema"] == 1
    assert quality["n_volume_sampling_mode"] == 1
    assert quality["sampling_mode_counts"] == {"missing": 1, "volume": 1}
    assert quality["n_with_patient_prior_residual_splats"] == 1
    assert quality["n_with_residual_heatmap"] == 1
    assert quality["n_with_residual_scores"] == 1
    assert quality["n_with_residual_metrics"] == 1
    assert quality["min_n_controls"] == pytest.approx(3)
    assert quality["min_n_gaussians"] == pytest.approx(321)


def test_evaluate_output_root_accepts_patient_paired_support_basis(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.05,
            "paired_gaussian_artifacts": True,
            "paired_gaussian_support_basis": "patient",
            "paired_gaussian_support_basis_mode": "patient",
            "paired_n_gaussians": 123,
        },
    ])

    result = evaluate_output_root(root)
    row = result["subjects"][0]
    quality = result["paired_gaussian_subtraction_quality"]
    gate = result["paper_readiness"]["gates"]["paired_gaussian_subtraction"]

    assert row["paired_gaussian_support_basis"] == "patient"
    assert row["paired_gaussian_support_basis_mode"] == "patient"
    assert quality["n_expected_support_basis"] == 1
    assert quality["n_unique_supported_support_basis"] == 1
    assert quality["support_basis_counts"] == {"patient": 1}
    assert "patient" in gate["threshold"]["support_basis"]
    assert gate["threshold"]["single_support_basis_per_run"] is True


def test_evaluate_output_root_flags_mixed_paired_support_basis(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.05,
            "paired_gaussian_artifacts": True,
            "paired_gaussian_support_basis": "max(patient, synthetic_normal)",
            "paired_n_gaussians": 60_000,
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.01,
            "paired_gaussian_artifacts": True,
            "paired_gaussian_support_basis": "patient",
            "paired_n_gaussians": 60_000,
        },
    ])

    result = evaluate_output_root(root)
    quality = result["paired_gaussian_subtraction_quality"]

    assert quality["n_expected_support_basis"] == 2
    assert quality["n_unique_supported_support_basis"] == 2
    assert any("mixes paired Gaussian support bases" in text for text in result["limitations"])


def test_evaluate_output_root_extracts_optimized_gaussian_evidence(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.05,
            "optimized_gaussian_artifacts": True,
            "optimized_n_patient": 321,
            "optimized_n_normal": 320,
        },
        {"id": "sub-hc-01", "group": "HC", "paired_score": 0.01},
    ])

    result = evaluate_output_root(root)
    fcd_row = next(r for r in result["subjects"] if r["subject"] == "sub-fcd-01")
    hc_row = next(r for r in result["subjects"] if r["subject"] == "sub-hc-01")
    quality = result["optimized_gaussian_subtraction_quality"]

    assert fcd_row["optimized_gaussian_residual_present"] is True
    assert fcd_row["optimized_gaussian_residual_splat_present"] is True
    assert fcd_row["optimized_gaussian_residual_heatmap_present"] is True
    assert fcd_row["optimized_gaussian_residual_metrics_present"] is True
    assert fcd_row["optimized_gaussian_residual_scores_present"] is True
    assert fcd_row["optimized_gaussian_same_support"] is True
    assert fcd_row["optimized_gaussian_subtraction_claim_allowed"] is True
    assert (
        fcd_row["optimized_gaussian_score_method"]
        == "indexwise_same_support_intensity"
    )
    assert fcd_row["optimized_gaussian_heatmap_method"] == "field"
    assert fcd_row["optimized_gaussian_n_patient"] == pytest.approx(321)
    assert fcd_row["optimized_gaussian_n_normal"] == pytest.approx(320)
    assert fcd_row["optimized_gaussian_means_norm_max_abs_diff"] == pytest.approx(0.0)
    assert hc_row["optimized_gaussian_residual_present"] is False
    assert quality["n_with_optimized_gaussian_residual"] == 1
    assert quality["n_same_gaussian_support"] == 1
    assert quality["n_gaussian_subtraction_claim_allowed"] == 1
    assert quality["n_indexwise_score_method"] == 1
    assert quality["n_field_heatmap_method"] == 1
    assert quality["n_with_residual_splat"] == 1
    assert quality["n_with_residual_heatmap"] == 1
    assert quality["n_with_residual_metrics"] == 1
    assert quality["n_with_residual_scores"] == 1
    assert quality["min_n_patient_gaussians"] == pytest.approx(321)
    assert quality["min_n_normal_gaussians"] == pytest.approx(320)
    assert quality["score_method_counts"] == {
        "indexwise_same_support_intensity": 1,
        "missing": 1,
    }
    assert quality["heatmap_method_counts"] == {"field": 1, "missing": 1}


def test_evaluate_output_root_extracts_data_quality_evidence(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.05,
            "data_quality": _data_quality(
                status="review",
                review_items=["flair:thick_slice_spacing_gt_3mm"],
                t1w_resample=True,
                roi_resample=True,
                roi_resampled_to_base=True,
            ),
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.01,
            "data_quality": _data_quality(
                status="fail",
                blockers=["flair:nonfinite_voxels_present"],
            ),
        },
    ])

    result = evaluate_output_root(root)
    fcd_row = next(r for r in result["subjects"] if r["subject"] == "sub-fcd-01")
    hc_row = next(r for r in result["subjects"] if r["subject"] == "sub-hc-01")
    summary = result["data_quality_summary"]

    assert fcd_row["data_quality_present"] is True
    assert fcd_row["data_quality_upload_readiness_status"] == "review"
    assert fcd_row["data_quality_can_run_3dgs"] is True
    assert fcd_row["data_quality_can_subtract_patient_normal"] is True
    assert fcd_row["data_quality_t1w_to_flair_needs_resample"] is True
    assert fcd_row["data_quality_roi_to_flair_needs_resample"] is True
    assert fcd_row["data_quality_roi_resampled_to_base"] is True
    assert hc_row["data_quality_upload_readiness_status"] == "fail"
    assert hc_row["data_quality_has_nonfinite_blocker"] is True
    assert summary["n_with_data_quality"] == 2
    assert summary["n_upload_readiness_review"] == 1
    assert summary["n_upload_readiness_fail"] == 1
    assert summary["n_can_run_3dgs"] == 1
    assert summary["n_with_t1w_to_flair_resampling_review"] == 1
    assert summary["n_with_roi_to_flair_resampling_review"] == 1
    assert summary["n_with_roi_resampled_to_base"] == 1
    assert summary["n_with_nonfinite_blocker"] == 1


def test_evaluate_output_root_extracts_dicom_conversion_safety(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.05,
            "data_quality": _data_quality(
                dicom_conversion_safety={
                    "status": "review",
                    "blocking_issue_count": 0,
                    "review_issue_count": 1,
                    "blockers": [],
                    "review_items": ["FLAIR:sidecar_missing"],
                    "can_use_converted_inputs_for_3dgs": True,
                    "can_use_converted_inputs_for_paper_claim": False,
                }
            ),
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.01,
            "data_quality": _data_quality(
                dicom_conversion_safety={
                    "status": "pass",
                    "blocking_issue_count": 0,
                    "review_issue_count": 0,
                    "blockers": [],
                    "review_items": [],
                    "can_use_converted_inputs_for_3dgs": True,
                    "can_use_converted_inputs_for_paper_claim": True,
                }
            ),
        },
    ])

    result = evaluate_output_root(root)
    fcd_row = next(r for r in result["subjects"] if r["subject"] == "sub-fcd-01")
    summary = result["data_quality_summary"]

    assert fcd_row["data_quality_dicom_conversion_status"] == "review"
    assert fcd_row["data_quality_dicom_conversion_can_run_3dgs"] is True
    assert fcd_row["data_quality_dicom_conversion_paper_claim_allowed"] is False
    assert fcd_row["data_quality_dicom_conversion_review_issue_count"] == 1
    assert summary["n_with_dicom_conversion_safety"] == 2
    assert summary["n_dicom_conversion_pass"] == 1
    assert summary["n_dicom_conversion_review"] == 1
    assert summary["n_dicom_conversion_paper_claim_allowed"] == 1
    assert summary["total_dicom_conversion_review_issue_count"] == 1


def test_evaluate_output_root_extracts_residual_candidate_metrics(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.05,
            "residual_candidate_meta": {
                "selected_count": 2,
                "candidate_voxels": 1200,
                "candidate_volume_ml": 3.6,
                "candidate_foreground_fraction": 0.02,
                "manual_coverage": 0.4,
                "candidate_precision": 0.6,
                "dice": 0.48,
                "jaccard": 0.32,
            },
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.01,
            "residual_candidate_meta": {
                "selected_count": 1,
                "candidate_voxels": 300,
                "candidate_volume_ml": 0.9,
                "candidate_foreground_fraction": 0.005,
            },
        },
    ])

    result = evaluate_output_root(root)
    fcd_row = next(r for r in result["subjects"] if r["subject"] == "sub-fcd-01")
    assert fcd_row["residual_candidate_selected_count"] == pytest.approx(2)
    assert fcd_row["residual_candidate_volume_ml"] == pytest.approx(3.6)
    assert fcd_row["residual_candidate_manual_coverage"] == pytest.approx(0.4)
    assert fcd_row["residual_candidate_precision"] == pytest.approx(0.6)
    assert fcd_row["residual_candidate_dice"] == pytest.approx(0.48)
    assert fcd_row["residual_candidate_jaccard"] == pytest.approx(0.32)
    assert result["metrics"]["residual_candidate_volume_ml"]["n_scored"] == 2
    assert result["candidate_validation"]["n_with_manual_roi_overlap"] == 1
    assert result["candidate_validation"]["mean_manual_coverage"] == pytest.approx(0.4)
    assert result["candidate_validation"]["coverage_ge_0_25_rate"] == pytest.approx(1.0)
    assert result["candidate_validation"]["mean_candidate_volume_ml_hc"] == pytest.approx(0.9)


def test_evaluate_output_root_requires_independent_manual_roi_for_overlap_metrics(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-validated",
            "group": "FCD",
            "paired_score": 0.05,
            "residual_candidate_meta": {
                "selected_count": 1,
                "candidate_foreground_fraction": 0.02,
                "manual_roi_available": True,
                "selected_roi_source": "candidate",
                "manual_roi_source": "provided",
                "manual_overlap_independent_from_fuzzing_roi": True,
                "manual_coverage": 0.6,
                "candidate_precision": 0.4,
                "dice": 0.48,
                "jaccard": 0.32,
            },
        },
        {
            "id": "sub-fcd-self-validated",
            "group": "FCD",
            "paired_score": 0.04,
            "residual_candidate_meta": {
                "selected_count": 1,
                "candidate_foreground_fraction": 0.03,
                "manual_roi_available": True,
                "manual_roi_source": "candidate",
                "manual_coverage": 0.99,
                "candidate_precision": 0.99,
                "dice": 0.99,
                "jaccard": 0.99,
            },
        },
        {
            "id": "sub-fcd-circular-provided-roi",
            "group": "FCD",
            "paired_score": 0.045,
            "residual_candidate_meta": {
                "selected_count": 1,
                "candidate_foreground_fraction": 0.025,
                "manual_roi_available": True,
                "selected_roi_source": "provided",
                "manual_roi_source": "provided",
                "manual_overlap_independent_from_fuzzing_roi": False,
                "manual_coverage": 0.95,
                "candidate_precision": 0.95,
                "dice": 0.95,
                "jaccard": 0.95,
            },
        },
        {
            "id": "sub-fcd-explicitly-not-independent",
            "group": "FCD",
            "paired_score": 0.043,
            "residual_candidate_meta": {
                "selected_count": 1,
                "candidate_foreground_fraction": 0.022,
                "manual_roi_available": True,
                "selected_roi_source": "candidate",
                "manual_roi_source": "provided",
                "manual_overlap_independent_from_fuzzing_roi": False,
                "manual_coverage": 0.88,
                "candidate_precision": 0.77,
                "dice": 0.81,
                "jaccard": 0.68,
            },
        },
        {
            "id": "sub-hc-with-stale-overlap",
            "group": "HC",
            "paired_score": 0.01,
            "residual_candidate_meta": {
                "selected_count": 1,
                "candidate_foreground_fraction": 0.004,
                "manual_roi_available": True,
                "manual_roi_source": "provided",
                "manual_coverage": 0.9,
                "candidate_precision": 0.8,
                "dice": 0.7,
                "jaccard": 0.6,
            },
        },
    ])

    result = evaluate_output_root(root)

    validated = next(r for r in result["subjects"] if r["subject"] == "sub-fcd-validated")
    self_validated = next(
        r for r in result["subjects"] if r["subject"] == "sub-fcd-self-validated"
    )
    circular = next(
        r for r in result["subjects"] if r["subject"] == "sub-fcd-circular-provided-roi"
    )
    explicitly_blocked = next(
        r for r in result["subjects"] if r["subject"] == "sub-fcd-explicitly-not-independent"
    )
    stale_hc = next(r for r in result["subjects"] if r["subject"] == "sub-hc-with-stale-overlap")

    assert validated["residual_candidate_manual_coverage"] == pytest.approx(0.6)
    assert self_validated["residual_candidate_manual_coverage"] is None
    assert circular["residual_candidate_manual_coverage"] is None
    assert circular["residual_candidate_precision"] is None
    assert explicitly_blocked["residual_candidate_manual_coverage"] is None
    assert stale_hc["residual_candidate_manual_coverage"] is None
    assert stale_hc["residual_candidate_precision"] is None
    assert result["metrics"]["residual_candidate_manual_coverage"]["n_scored"] == 1
    assert result["candidate_validation"]["n_with_manual_roi_overlap"] == 1
    assert result["candidate_validation"]["mean_manual_coverage"] == pytest.approx(0.6)


def test_evaluate_output_root_reads_residual_candidate_sidecar(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.05,
            "residual_candidate_artifact": {
                "selected_count": 1,
                "candidate_voxels": 800,
                "candidate_volume_ml": 2.4,
                "candidate_foreground_fraction": 0.015,
                "manual_overlap": {
                    "manual_coverage": 0.3,
                    "candidate_precision": 0.7,
                    "dice": 0.42,
                    "jaccard": 0.27,
                },
            },
        },
        {"id": "sub-hc-01", "group": "HC", "paired_score": 0.01},
    ])

    result = evaluate_output_root(root)
    fcd_row = next(r for r in result["subjects"] if r["subject"] == "sub-fcd-01")
    assert fcd_row["residual_candidate_selected_count"] == pytest.approx(1)
    assert fcd_row["residual_candidate_volume_ml"] == pytest.approx(2.4)
    assert fcd_row["residual_candidate_manual_coverage"] == pytest.approx(0.3)
    assert fcd_row["residual_candidate_precision"] == pytest.approx(0.7)
    assert result["candidate_validation"]["mean_dice"] == pytest.approx(0.42)


def test_evaluate_output_root_extracts_splat_reconstruction_metrics(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.05,
            "splat_reconstruction_audit_meta": {
                "foreground_coverage": 0.72,
                "all_foreground_mae": 0.18,
                "all_foreground_rmse": 0.22,
                "all_foreground_psnr": 13.15,
                "represented_foreground_mae": 0.04,
                "represented_foreground_psnr": 28.0,
                "mip_mean_mae": 0.12,
            },
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.01,
            "splat_reconstruction_audit_meta": {
                "foreground_coverage": 0.68,
                "all_foreground_mae": 0.2,
                "all_foreground_psnr": 12.5,
                "mip_mean_mae": 0.14,
            },
        },
    ])

    result = evaluate_output_root(root)
    fcd_row = next(r for r in result["subjects"] if r["subject"] == "sub-fcd-01")
    assert fcd_row["splat_foreground_coverage"] == pytest.approx(0.72)
    assert fcd_row["splat_all_foreground_mae"] == pytest.approx(0.18)
    assert fcd_row["splat_all_foreground_psnr"] == pytest.approx(13.15)
    assert fcd_row["splat_mip_mean_mae"] == pytest.approx(0.12)
    assert fcd_row["splat_reconstruction_readiness_status"] == "pass"
    assert fcd_row["splat_mri_to_3dgs_claim_allowed"] is True
    assert (
        fcd_row["splat_mri_to_3dgs_claim_class"]
        == "volumetric_mri_gaussian_representation"
    )
    quality = result["splat_reconstruction_quality"]
    assert quality["n_with_splat_audit"] == 2
    assert quality["n_with_splat_readiness_audit"] == 2
    assert quality["n_mri_to_3dgs_claim_allowed"] == 2
    assert quality["mean_foreground_coverage"] == pytest.approx(0.7)
    assert quality["mean_all_foreground_mae"] == pytest.approx(0.19)


def test_evaluate_output_root_extracts_optimized_field_audit(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.05,
            "optimized_gaussian_artifacts": True,
            "optimized_gaussian_field_audit_meta": _optimized_field_audit_meta(
                foreground_mae=0.06,
                all_mae=0.04,
            ),
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.01,
            "optimized_gaussian_artifacts": True,
            "optimized_gaussian_field_audit_meta": _optimized_field_audit_meta(
                foreground_mae=0.08,
                all_mae=0.05,
            ),
        },
    ])

    result = evaluate_output_root(root)
    fcd_row = next(r for r in result["subjects"] if r["subject"] == "sub-fcd-01")
    assert fcd_row["optimized_gaussian_field_audit_present"] is True
    assert fcd_row["optimized_gaussian_field_audit_status"] == "pass"
    assert fcd_row["optimized_mri_to_3dgs_claim_allowed"] is True
    assert fcd_row["optimized_gaussian_field_foreground_mae"] == pytest.approx(0.06)
    assert fcd_row["optimized_gaussian_field_all_mae"] == pytest.approx(0.04)
    quality = result["optimized_gaussian_field_quality"]
    assert quality["n_with_optimized_field_audit"] == 2
    assert quality["n_optimized_mri_to_3dgs_claim_allowed"] == 2
    assert quality["mean_foreground_mae"] == pytest.approx(0.07)
    assert quality["mean_all_mae"] == pytest.approx(0.045)


def test_evaluate_output_root_paper_readiness_defaults_not_ready_for_tiny_run(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {"id": "sub-fcd-01", "group": "FCD", "paired_score": 0.05},
        {"id": "sub-hc-01", "group": "HC", "paired_score": 0.01},
    ])

    result = evaluate_output_root(root)
    readiness = result["paper_readiness"]

    assert readiness["overall_status"] == "not_ready"
    assert readiness["gates"]["cohort_size"]["status"] == "fail"
    assert "cohort_size" in readiness["blocking_gates"]
    assert readiness["thresholds"]["expected_fcd"] == 85
    assert readiness["thresholds"]["expected_hc"] == 85


def test_evaluate_output_root_paper_readiness_requires_paired_gaussian_state(tmp_path):
    shared_splat = {
        "foreground_coverage": 0.72,
        "all_foreground_mae": 0.10,
        "all_foreground_psnr": 20.0,
        "mip_mean_mae": 0.03,
    }
    shared_control_prior = {
        "control_prior_quality_gate": "pass",
        "control_prior_reference_similarity_1_minus_mae": 0.90,
        "control_prior_reference_mae": 0.10,
        "control_prior_reference_rmse": 0.15,
        "control_prior_reference_correlation": 0.80,
    }
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.09,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "manual_coverage": 0.60,
                "candidate_foreground_fraction": 0.02,
                "candidate_volume_ml": 2.0,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": shared_control_prior,
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.01,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "candidate_foreground_fraction": 0.004,
                "candidate_volume_ml": 0.4,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": shared_control_prior,
        },
    ])

    result = evaluate_output_root(
        root,
        gate_thresholds={
            "expected_fcd": 1,
            "expected_hc": 1,
            "paired_gaussian_min_points": 10,
        },
    )
    readiness = result["paper_readiness"]
    gate = readiness["gates"]["paired_gaussian_subtraction"]

    assert readiness["overall_status"] == "not_ready"
    assert "paired_gaussian_subtraction" in readiness["blocking_gates"]
    assert gate["status"] == "insufficient_evidence"
    assert gate["observed"]["n_with_paired_gaussian_state"] == 0


def test_evaluate_output_root_paper_readiness_requires_optimized_gaussian_residual(tmp_path):
    shared_splat = {
        "foreground_coverage": 0.72,
        "all_foreground_mae": 0.10,
        "all_foreground_psnr": 20.0,
        "mip_mean_mae": 0.03,
    }
    shared_control_prior = {
        "control_prior_quality_gate": "pass",
        "control_prior_reference_similarity_1_minus_mae": 0.90,
        "control_prior_reference_mae": 0.10,
        "control_prior_reference_rmse": 0.15,
        "control_prior_reference_correlation": 0.80,
    }
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.09,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "manual_coverage": 0.60,
                "candidate_foreground_fraction": 0.02,
                "candidate_volume_ml": 2.0,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": shared_control_prior,
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.01,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "candidate_foreground_fraction": 0.004,
                "candidate_volume_ml": 0.4,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": shared_control_prior,
        },
    ])

    result = evaluate_output_root(
        root,
        gate_thresholds={
            "expected_fcd": 1,
            "expected_hc": 1,
            "paired_gaussian_min_points": 10,
            "optimized_gaussian_min_points": 10,
        },
    )
    readiness = result["paper_readiness"]
    gate = readiness["gates"]["optimized_gaussian_subtraction"]

    assert readiness["overall_status"] == "not_ready"
    assert "optimized_gaussian_subtraction" in readiness["blocking_gates"]
    assert gate["status"] == "insufficient_evidence"
    assert gate["observed"]["n_with_optimized_gaussian_residual"] == 0
    assert any(
        "No subjects with optimized same-support Gaussian residual artifacts" in item
        for item in result["limitations"]
    )


def test_evaluate_output_root_paper_readiness_requires_control_gaussian_prior(
    tmp_path,
):
    shared_splat = {
        "foreground_coverage": 0.72,
        "all_foreground_mae": 0.10,
        "all_foreground_psnr": 20.0,
        "mip_mean_mae": 0.03,
    }
    shared_control_prior = {
        "control_prior_quality_gate": "pass",
        "control_prior_reference_similarity_1_minus_mae": 0.90,
        "control_prior_reference_mae": 0.10,
        "control_prior_reference_rmse": 0.15,
        "control_prior_reference_correlation": 0.80,
    }
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.09,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "optimized_gaussian_artifacts": True,
            "optimized_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "manual_coverage": 0.60,
                "candidate_foreground_fraction": 0.02,
                "candidate_volume_ml": 2.0,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": shared_control_prior,
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.01,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "optimized_gaussian_artifacts": True,
            "optimized_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "candidate_foreground_fraction": 0.004,
                "candidate_volume_ml": 0.4,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": shared_control_prior,
        },
    ])

    result = evaluate_output_root(
        root,
        gate_thresholds={
            "expected_fcd": 1,
            "expected_hc": 1,
            "paired_gaussian_min_points": 10,
            "optimized_gaussian_min_points": 10,
            "control_gaussian_min_points": 10,
            "control_gaussian_min_controls": 2,
        },
    )
    readiness = result["paper_readiness"]
    gate = readiness["gates"]["control_gaussian_prior_subtraction"]

    assert readiness["overall_status"] == "not_ready"
    assert "control_gaussian_prior_subtraction" in readiness["blocking_gates"]
    assert gate["status"] == "insufficient_evidence"
    assert gate["observed"]["n_with_control_gaussian_prior_state"] == 0
    assert result["control_gaussian_prior_quality"]["n_known_subjects"] == 2
    assert any(
        "No subjects with control-prior Gaussian state artifacts" in item
        for item in result["limitations"]
    )


def test_evaluate_output_root_paper_readiness_requires_volume_paired_support(tmp_path):
    shared_splat = {
        "foreground_coverage": 0.72,
        "all_foreground_mae": 0.10,
        "all_foreground_psnr": 20.0,
        "mip_mean_mae": 0.03,
    }
    shared_control_prior = {
        "control_prior_quality_gate": "pass",
        "control_prior_reference_similarity_1_minus_mae": 0.90,
        "control_prior_reference_mae": 0.10,
        "control_prior_reference_rmse": 0.15,
        "control_prior_reference_correlation": 0.80,
    }
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.09,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "paired_gaussian_sampling_mode": "surface",
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "manual_coverage": 0.60,
                "candidate_foreground_fraction": 0.02,
                "candidate_volume_ml": 2.0,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": shared_control_prior,
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.01,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "candidate_foreground_fraction": 0.004,
                "candidate_volume_ml": 0.4,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": shared_control_prior,
        },
    ])

    result = evaluate_output_root(
        root,
        gate_thresholds={
            "expected_fcd": 1,
            "expected_hc": 1,
            "paired_gaussian_min_points": 10,
        },
    )
    readiness = result["paper_readiness"]
    gate = readiness["gates"]["paired_gaussian_subtraction"]

    assert readiness["overall_status"] == "not_ready"
    assert "paired_gaussian_subtraction" in readiness["blocking_gates"]
    assert gate["status"] == "fail"
    assert gate["observed"]["n_volume_sampling_mode"] == 1
    assert gate["observed"]["sampling_mode_counts"] == {"surface": 1, "volume": 1}
    assert any("volume-mode paired Gaussian support" in item for item in result["limitations"])


def test_evaluate_output_root_paper_readiness_requires_data_quality_audit(tmp_path):
    shared_splat = {
        "foreground_coverage": 0.72,
        "all_foreground_mae": 0.10,
        "all_foreground_psnr": 20.0,
        "mip_mean_mae": 0.03,
    }
    shared_control_prior = {
        "control_prior_quality_gate": "pass",
        "control_prior_reference_similarity_1_minus_mae": 0.90,
        "control_prior_reference_mae": 0.10,
        "control_prior_reference_rmse": 0.15,
        "control_prior_reference_correlation": 0.80,
    }
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.09,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "manual_coverage": 0.60,
                "candidate_foreground_fraction": 0.02,
                "candidate_volume_ml": 2.0,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": shared_control_prior,
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.01,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "candidate_foreground_fraction": 0.004,
                "candidate_volume_ml": 0.4,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": shared_control_prior,
        },
    ])

    result = evaluate_output_root(
        root,
        gate_thresholds={
            "expected_fcd": 1,
            "expected_hc": 1,
            "paired_gaussian_min_points": 10,
        },
    )
    readiness = result["paper_readiness"]
    gate = readiness["gates"]["data_quality_conversion"]

    assert readiness["overall_status"] == "not_ready"
    assert "data_quality_conversion" in readiness["blocking_gates"]
    assert gate["status"] == "insufficient_evidence"
    assert gate["observed"]["n_with_data_quality"] == 0


def test_evaluate_output_root_paper_readiness_blocks_failed_splat_claim(tmp_path):
    failed_splat = {
        "foreground_coverage": 0.12,
        "all_foreground_mae": 0.30,
        "all_foreground_psnr": 8.0,
        "mip_mean_mae": 0.18,
        "mri_to_3dgs_readiness_status": "fail",
        "mri_to_3dgs_readiness_blocker_count": 2,
        "mri_to_3dgs_readiness_warning_count": 0,
        "mri_to_3dgs_claim_allowed": False,
        "mri_to_3dgs_claim_class": "sparse_or_low_fidelity_visualization_only",
    }
    passing_splat = {
        "foreground_coverage": 0.72,
        "all_foreground_mae": 0.10,
        "all_foreground_psnr": 20.0,
        "mip_mean_mae": 0.03,
    }
    shared_control_prior = {
        "control_prior_quality_gate": "pass",
        "control_prior_reference_similarity_1_minus_mae": 0.90,
        "control_prior_reference_mae": 0.10,
        "control_prior_reference_rmse": 0.15,
        "control_prior_reference_correlation": 0.80,
    }
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.09,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "manual_coverage": 0.60,
                "candidate_foreground_fraction": 0.02,
                "candidate_volume_ml": 2.0,
            },
            "splat_reconstruction_audit_meta": failed_splat,
            "control_prior_meta": shared_control_prior,
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.01,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "candidate_foreground_fraction": 0.004,
                "candidate_volume_ml": 0.4,
            },
            "splat_reconstruction_audit_meta": passing_splat,
            "control_prior_meta": shared_control_prior,
        },
    ])

    result = evaluate_output_root(
        root,
        gate_thresholds={
            "expected_fcd": 1,
            "expected_hc": 1,
            "paired_gaussian_min_points": 10,
        },
    )
    readiness = result["paper_readiness"]
    gate = readiness["gates"]["gaussian_splat_fit"]

    assert readiness["overall_status"] == "not_ready"
    assert "gaussian_splat_fit" in readiness["blocking_gates"]
    assert gate["status"] == "fail"
    assert gate["observed"]["n_readiness_fail"] == 1
    assert gate["observed"]["total_readiness_blocker_count"] == 2
    assert gate["observed"]["n_mri_to_3dgs_claim_allowed"] == 1
    assert result["splat_reconstruction_quality"]["n_readiness_fail"] == 1


def test_evaluate_output_root_paper_readiness_requires_optimized_field_audit(tmp_path):
    shared_splat = {
        "foreground_coverage": 0.72,
        "all_foreground_mae": 0.10,
        "all_foreground_psnr": 20.0,
        "mip_mean_mae": 0.03,
    }
    shared_control_prior = {
        "control_prior_quality_gate": "pass",
        "control_prior_reference_similarity_1_minus_mae": 0.90,
        "control_prior_reference_mae": 0.10,
        "control_prior_reference_rmse": 0.15,
        "control_prior_reference_correlation": 0.80,
    }
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.09,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "control_gaussian_artifacts": True,
            "control_gaussian_n_gaussians": 25,
            "control_gaussian_n_controls": 2,
            "optimized_gaussian_artifacts": True,
            "optimized_n_gaussians": 25,
            "optimized_gaussian_field_audit_meta": None,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "manual_coverage": 0.60,
                "candidate_foreground_fraction": 0.02,
                "candidate_volume_ml": 2.0,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": shared_control_prior,
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.01,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "control_gaussian_artifacts": True,
            "control_gaussian_n_gaussians": 25,
            "control_gaussian_n_controls": 2,
            "optimized_gaussian_artifacts": True,
            "optimized_n_gaussians": 25,
            "optimized_gaussian_field_audit_meta": _optimized_field_audit_meta(),
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "candidate_foreground_fraction": 0.004,
                "candidate_volume_ml": 0.4,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": shared_control_prior,
        },
    ])

    result = evaluate_output_root(
        root,
        gate_thresholds={
            "expected_fcd": 1,
            "expected_hc": 1,
            "paired_gaussian_min_points": 10,
            "control_gaussian_min_points": 10,
            "control_gaussian_min_controls": 2,
            "optimized_gaussian_min_points": 10,
        },
    )
    readiness = result["paper_readiness"]
    gate = readiness["gates"]["gaussian_splat_fit"]

    assert readiness["overall_status"] == "not_ready"
    assert "gaussian_splat_fit" in readiness["blocking_gates"]
    assert gate["status"] == "fail"
    assert gate["observed"]["n_with_optimized_field_audit"] == 1
    assert gate["observed"]["n_optimized_mri_to_3dgs_claim_allowed"] == 1
    assert result["optimized_gaussian_field_quality"]["n_with_optimized_field_audit"] == 1


def test_evaluate_output_root_paper_readiness_can_pass_custom_pilot_thresholds(tmp_path):
    shared_splat = {
        "foreground_coverage": 0.72,
        "all_foreground_mae": 0.10,
        "all_foreground_psnr": 20.0,
        "mip_mean_mae": 0.03,
    }
    shared_control_prior = {
        "control_prior_quality_gate": "pass",
        "control_prior_reference_similarity_1_minus_mae": 0.90,
        "control_prior_reference_mae": 0.10,
        "control_prior_reference_rmse": 0.15,
        "control_prior_reference_correlation": 0.80,
    }
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.09,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "control_gaussian_artifacts": True,
            "control_gaussian_n_gaussians": 25,
            "control_gaussian_n_controls": 2,
            "optimized_gaussian_artifacts": True,
            "optimized_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "manual_coverage": 0.60,
                "candidate_foreground_fraction": 0.02,
                "candidate_volume_ml": 2.0,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": shared_control_prior,
        },
        {
            "id": "sub-fcd-02",
            "group": "FCD",
            "paired_score": 0.08,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "control_gaussian_artifacts": True,
            "control_gaussian_n_gaussians": 25,
            "control_gaussian_n_controls": 2,
            "optimized_gaussian_artifacts": True,
            "optimized_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "manual_coverage": 0.55,
                "candidate_foreground_fraction": 0.018,
                "candidate_volume_ml": 1.8,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": shared_control_prior,
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.01,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "control_gaussian_artifacts": True,
            "control_gaussian_n_gaussians": 25,
            "control_gaussian_n_controls": 2,
            "optimized_gaussian_artifacts": True,
            "optimized_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "candidate_foreground_fraction": 0.004,
                "candidate_volume_ml": 0.4,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": shared_control_prior,
        },
        {
            "id": "sub-hc-02",
            "group": "HC",
            "paired_score": 0.02,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "control_gaussian_artifacts": True,
            "control_gaussian_n_gaussians": 25,
            "control_gaussian_n_controls": 2,
            "optimized_gaussian_artifacts": True,
            "optimized_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "candidate_foreground_fraction": 0.006,
                "candidate_volume_ml": 0.5,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": shared_control_prior,
        },
    ])

    result = evaluate_output_root(
        root,
        gate_thresholds={
            "expected_fcd": 2,
            "expected_hc": 2,
            "paired_gaussian_min_points": 10,
            "optimized_gaussian_min_points": 10,
            "control_gaussian_min_points": 10,
            "control_gaussian_min_controls": 2,
        },
    )
    readiness = result["paper_readiness"]

    assert readiness["overall_status"] == "pass"
    assert readiness["blocking_gates"] == []
    for gate in readiness["gates"].values():
        assert gate["status"] == "pass"
    assert readiness["gates"]["detection_auroc"]["observed"]["auroc_fcd_vs_hc"] == 1.0
    assert readiness["gates"]["healthy_reference_synthesis"]["status"] == "pass"
    assert readiness["gates"]["control_gaussian_prior_subtraction"]["status"] == "pass"
    assert (
        result["control_gaussian_prior_quality"]["n_with_control_gaussian_prior_state"]
        == 4
    )
    assert (
        result["control_prior_quality"]["mean_reference_similarity_1_minus_mae"]
        == pytest.approx(0.90)
    )
    assert (
        readiness["gates"]["healthy_control_false_positive_burden"]["observed"]
        ["mean_candidate_foreground_fraction_hc"]
        == pytest.approx(0.005)
    )
    assert readiness["gates"]["roi_source_provenance"]["status"] == "pass"
    patient_specific = readiness["claim_boundaries"][
        "patient_specific_healthy_counterfactual"
    ]
    assert patient_specific["status"] == "not_supported"
    assert (
        patient_specific["observed"][
            "n_patient_conditioned_counterfactual_claim_allowed"
        ]
        == 0
    )
    assert readiness["blocking_gates"] == []


def test_evaluate_output_root_patient_specific_claim_boundary_supported(tmp_path):
    shared_splat = {
        "foreground_coverage": 0.72,
        "all_foreground_mae": 0.10,
        "all_foreground_psnr": 20.0,
        "mip_mean_mae": 0.03,
    }
    patient_specific_audit = {
        "counterfactual_class": (
            "patient_specific_contralateral_geometric_roi_completion"
        ),
        "claim_boundary": "patient_specific_geometric_counterfactual",
        "synthetic_normal_claim_class": "patient_specific_geometric_counterfactual",
        "subtraction_readiness_status": "pass",
        "subtraction_readiness_blocker_count": 0,
        "subtraction_readiness_warning_count": 0,
        "gaussian_subtraction_claim_allowed": True,
        "patient_conditioned_inpainting": True,
        "patient_conditioned_counterfactual_claim_allowed": True,
        "same_grid_before_resample": True,
        "synthetic_resampled_to_patient_grid": False,
        "outside_roi_similarity_1_minus_mae": 0.99,
        "inside_to_outside_mae_ratio": 2.0,
    }
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.09,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "control_gaussian_artifacts": True,
            "control_gaussian_n_gaussians": 25,
            "control_gaussian_n_controls": 2,
            "optimized_gaussian_artifacts": True,
            "optimized_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "manual_coverage": 0.60,
                "candidate_foreground_fraction": 0.02,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "synthetic_normal_audit_meta": patient_specific_audit,
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.01,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "control_gaussian_artifacts": True,
            "control_gaussian_n_gaussians": 25,
            "control_gaussian_n_controls": 2,
            "optimized_gaussian_artifacts": True,
            "optimized_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "candidate_foreground_fraction": 0.004,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "synthetic_normal_audit_meta": patient_specific_audit,
        },
    ])

    result = evaluate_output_root(
        root,
        gate_thresholds={
            "expected_fcd": 1,
            "expected_hc": 1,
            "paired_gaussian_min_points": 10,
            "optimized_gaussian_min_points": 10,
            "control_gaussian_min_points": 10,
            "control_gaussian_min_controls": 2,
        },
    )
    readiness = result["paper_readiness"]
    patient_specific = readiness["claim_boundaries"][
        "patient_specific_healthy_counterfactual"
    ]

    assert readiness["overall_status"] == "pass"
    assert patient_specific["status"] == "supported"
    assert (
        patient_specific["observed"][
            "n_patient_conditioned_counterfactual_claim_allowed"
        ]
        == 2
    )
    assert patient_specific["observed"]["n_patient_conditioned_inpainting"] == 2


def test_evaluate_output_root_patient_specific_claim_boundary_needs_audit(tmp_path):
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.05,
            "synthetic_normal_audit_meta": None,
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.01,
            "synthetic_normal_audit_meta": None,
        },
    ])

    result = evaluate_output_root(
        root,
        gate_thresholds={"expected_fcd": 1, "expected_hc": 1},
    )
    patient_specific = result["paper_readiness"]["claim_boundaries"][
        "patient_specific_healthy_counterfactual"
    ]

    assert patient_specific["status"] == "insufficient_evidence"
    assert patient_specific["observed"]["n_with_synthetic_normal_audit"] == 0


def test_evaluate_output_root_paper_readiness_blocks_dicom_conversion_review(tmp_path):
    shared_splat = {
        "foreground_coverage": 0.72,
        "all_foreground_mae": 0.10,
        "all_foreground_psnr": 20.0,
        "mip_mean_mae": 0.03,
    }
    shared_control_prior = {
        "control_prior_quality_gate": "pass",
        "control_prior_reference_similarity_1_minus_mae": 0.90,
        "control_prior_reference_mae": 0.10,
        "control_prior_reference_rmse": 0.15,
        "control_prior_reference_correlation": 0.80,
    }
    dicom_review = _data_quality(
        dicom_conversion_safety={
            "status": "review",
            "blocking_issue_count": 0,
            "review_issue_count": 1,
            "blockers": [],
            "review_items": ["FLAIR:sidecar_missing"],
            "can_use_converted_inputs_for_3dgs": True,
            "can_use_converted_inputs_for_paper_claim": False,
        }
    )
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.09,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "data_quality": dicom_review,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "manual_coverage": 0.60,
                "candidate_foreground_fraction": 0.02,
                "candidate_volume_ml": 2.0,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": shared_control_prior,
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.01,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "candidate_foreground_fraction": 0.004,
                "candidate_volume_ml": 0.4,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": shared_control_prior,
        },
    ])

    result = evaluate_output_root(
        root,
        gate_thresholds={
            "expected_fcd": 1,
            "expected_hc": 1,
            "paired_gaussian_min_points": 10,
        },
    )
    readiness = result["paper_readiness"]
    gate = readiness["gates"]["data_quality_conversion"]

    assert readiness["overall_status"] == "not_ready"
    assert "data_quality_conversion" in readiness["blocking_gates"]
    assert gate["status"] == "fail"
    assert gate["observed"]["n_with_dicom_conversion_safety"] == 1
    assert gate["observed"]["n_dicom_conversion_review"] == 1
    assert gate["observed"]["n_dicom_conversion_paper_claim_allowed"] == 0


def test_evaluate_output_root_paper_readiness_blocks_oracle_fcd_roi(tmp_path):
    shared_splat = {
        "foreground_coverage": 0.72,
        "all_foreground_mae": 0.10,
        "all_foreground_psnr": 20.0,
        "mip_mean_mae": 0.03,
    }
    shared_control_prior = {
        "control_prior_quality_gate": "pass",
        "control_prior_reference_similarity_1_minus_mae": 0.90,
        "control_prior_reference_mae": 0.10,
        "control_prior_reference_rmse": 0.15,
        "control_prior_reference_correlation": 0.80,
    }
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.09,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "provided",
            "residual_candidate_meta": {
                "manual_coverage": 0.60,
                "candidate_foreground_fraction": 0.02,
                "candidate_volume_ml": 2.0,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": shared_control_prior,
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.01,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "candidate_foreground_fraction": 0.004,
                "candidate_volume_ml": 0.4,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": shared_control_prior,
        },
    ])

    result = evaluate_output_root(
        root,
        gate_thresholds={
            "expected_fcd": 1,
            "expected_hc": 1,
            "paired_gaussian_min_points": 10,
        },
    )
    readiness = result["paper_readiness"]
    gate = readiness["gates"]["roi_source_provenance"]

    assert readiness["overall_status"] == "not_ready"
    assert "roi_source_provenance" in readiness["blocking_gates"]
    assert gate["status"] == "fail"
    assert gate["observed"]["n_oracle_fcd_subjects"] == 1
    assert gate["observed"]["counts_by_group"]["FCD"]["provided"] == 1
    assert gate["observed"]["counts_by_group"]["HC"]["candidate"] == 1


def test_evaluate_output_root_synthesis_gate_blocks_review_control_prior(tmp_path):
    shared_splat = {
        "foreground_coverage": 0.72,
        "all_foreground_mae": 0.10,
        "all_foreground_psnr": 20.0,
        "mip_mean_mae": 0.03,
    }
    passing_control_prior = {
        "control_prior_quality_gate": "pass",
        "control_prior_reference_similarity_1_minus_mae": 0.90,
        "control_prior_reference_mae": 0.10,
        "control_prior_reference_rmse": 0.15,
        "control_prior_reference_correlation": 0.80,
    }
    review_control_prior = {
        "control_prior_quality_gate": "review",
        "control_prior_reference_similarity_1_minus_mae": 0.70,
        "control_prior_reference_mae": 0.30,
        "control_prior_reference_rmse": 0.35,
        "control_prior_reference_correlation": 0.40,
    }
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.09,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "manual_coverage": 0.60,
                "candidate_foreground_fraction": 0.02,
                "candidate_volume_ml": 2.0,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": review_control_prior,
        },
        {
            "id": "sub-fcd-02",
            "group": "FCD",
            "paired_score": 0.08,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "manual_coverage": 0.55,
                "candidate_foreground_fraction": 0.018,
                "candidate_volume_ml": 1.8,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": passing_control_prior,
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.01,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "candidate_foreground_fraction": 0.004,
                "candidate_volume_ml": 0.4,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": passing_control_prior,
        },
        {
            "id": "sub-hc-02",
            "group": "HC",
            "paired_score": 0.02,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "candidate_foreground_fraction": 0.006,
                "candidate_volume_ml": 0.5,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": passing_control_prior,
        },
    ])

    result = evaluate_output_root(
        root,
        gate_thresholds={
            "expected_fcd": 2,
            "expected_hc": 2,
            "paired_gaussian_min_points": 10,
        },
    )
    readiness = result["paper_readiness"]
    synthesis_gate = readiness["gates"]["healthy_reference_synthesis"]

    assert readiness["overall_status"] == "not_ready"
    assert "healthy_reference_synthesis" in readiness["blocking_gates"]
    assert synthesis_gate["status"] == "fail"
    assert synthesis_gate["observed"]["n_quality_gate_review"] == 1
    assert (
        synthesis_gate["observed"]["mean_reference_similarity_1_minus_mae"]
        == pytest.approx(0.85)
    )


def test_evaluate_output_root_synthesis_gate_blocks_failed_synthetic_audit(tmp_path):
    shared_splat = {
        "foreground_coverage": 0.72,
        "all_foreground_mae": 0.10,
        "all_foreground_psnr": 20.0,
        "mip_mean_mae": 0.03,
    }
    passing_control_prior = {
        "control_prior_quality_gate": "pass",
        "control_prior_reference_similarity_1_minus_mae": 0.90,
        "control_prior_reference_mae": 0.10,
    }
    failed_audit = {
        "counterfactual_class": "patient_copy_degenerate_reference",
        "claim_boundary": "not_patient_conditioned_inpainting",
        "synthetic_normal_claim_class": "not_supported_for_patient_specific_counterfactual",
        "subtraction_readiness_status": "fail",
        "subtraction_readiness_blocker_count": 1,
        "subtraction_readiness_warning_count": 0,
        "gaussian_subtraction_claim_allowed": False,
        "patient_conditioned_inpainting": False,
        "patient_conditioned_counterfactual_claim_allowed": False,
        "same_grid_before_resample": True,
        "synthetic_resampled_to_patient_grid": False,
        "outside_roi_similarity_1_minus_mae": 1.0,
    }
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "paired_score": 0.09,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "manual_coverage": 0.60,
                "candidate_foreground_fraction": 0.02,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": passing_control_prior,
            "synthetic_normal_audit_meta": failed_audit,
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "paired_score": 0.01,
            "paired_gaussian_artifacts": True,
            "paired_n_gaussians": 25,
            "data_quality": True,
            "roi_source": "candidate",
            "residual_candidate_meta": {
                "candidate_foreground_fraction": 0.004,
            },
            "splat_reconstruction_audit_meta": shared_splat,
            "control_prior_meta": passing_control_prior,
        },
    ])

    result = evaluate_output_root(
        root,
        gate_thresholds={
            "expected_fcd": 1,
            "expected_hc": 1,
            "paired_gaussian_min_points": 10,
        },
    )
    synthesis_gate = result["paper_readiness"]["gates"]["healthy_reference_synthesis"]

    assert synthesis_gate["status"] == "fail"
    assert synthesis_gate["observed"]["n_subtraction_readiness_fail"] == 1
    assert synthesis_gate["observed"]["total_subtraction_blocker_count"] == 1
    assert synthesis_gate["observed"]["n_gaussian_subtraction_claim_allowed"] == 1
    assert result["synthetic_normal_quality"]["n_subtraction_readiness_fail"] == 1


def test_evaluate_output_root_computes_metrics_mapping(tmp_path):
    """Should compute separate AUROC and means for each metric."""
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "score": 0.9,
            "metrics": {"residual_p95": 0.85, "roi_to_non_roi_ratio": 1.5},
        },
        {
            "id": "sub-fcd-02",
            "group": "FCD",
            "score": 0.8,
            "metrics": {"residual_p95": 0.75, "roi_to_non_roi_ratio": 1.3},
        },
        {
            "id": "sub-fcd-03",
            "group": "FCD",
            "score": 0.85,
            "metrics": {"residual_p95": 0.80, "roi_to_non_roi_ratio": 1.4},
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "score": 0.2,
            "metrics": {"residual_p95": 0.18, "roi_to_non_roi_ratio": 0.7},
        },
        {
            "id": "sub-hc-02",
            "group": "HC",
            "score": 0.15,
            "metrics": {"residual_p95": 0.14, "roi_to_non_roi_ratio": 0.6},
        },
        {
            "id": "sub-hc-03",
            "group": "HC",
            "score": 0.25,
            "metrics": {"residual_p95": 0.22, "roi_to_non_roi_ratio": 0.8},
        },
    ])

    result = evaluate_output_root(root)

    # Check metrics dict exists and has all metric names
    assert "metrics" in result
    for metric_name in ["residual_mean", "residual_p95", "residual_p99", "roi_to_non_roi_ratio"]:
        assert metric_name in result["metrics"]

    # Check p99 metrics (available for all subjects)
    p99_metrics = result["metrics"]["residual_p99"]
    assert p99_metrics["n_scored"] == 6
    assert p99_metrics["n_missing"] == 0
    assert p99_metrics["fcd_mean"] == pytest.approx((0.9 + 0.8 + 0.85) / 3)
    assert p99_metrics["hc_mean"] == pytest.approx((0.2 + 0.15 + 0.25) / 3)
    assert p99_metrics["auroc_fcd_vs_hc"] > 0.9, "Should separate FCD from HC"

    # Check p95 metrics (available for all subjects)
    p95_metrics = result["metrics"]["residual_p95"]
    assert p95_metrics["n_scored"] == 6
    assert p95_metrics["fcd_mean"] == pytest.approx((0.85 + 0.75 + 0.80) / 3)
    assert p95_metrics["hc_mean"] == pytest.approx((0.18 + 0.14 + 0.22) / 3)
    assert p95_metrics["auroc_fcd_vs_hc"] > 0.9

    # Check roi_ratio metrics
    roi_metrics = result["metrics"]["roi_to_non_roi_ratio"]
    assert roi_metrics["n_scored"] == 6
    assert roi_metrics["fcd_mean"] == pytest.approx((1.5 + 1.3 + 1.4) / 3)
    assert roi_metrics["hc_mean"] == pytest.approx((0.7 + 0.6 + 0.8) / 3)
    assert roi_metrics["auroc_fcd_vs_hc"] > 0.9


def test_evaluate_output_root_metrics_with_missing_values(tmp_path):
    """Metrics with missing values should exclude those subjects from that metric's AUROC."""
    root = _make_fake_output_root(tmp_path, [
        {
            "id": "sub-fcd-01",
            "group": "FCD",
            "score": 0.9,
            "metrics": {"residual_p95": 0.85},  # No roi_ratio
        },
        {
            "id": "sub-fcd-02",
            "group": "FCD",
            "score": 0.8,
            "metrics": {"residual_p95": 0.75, "roi_to_non_roi_ratio": 1.3},
        },
        {
            "id": "sub-hc-01",
            "group": "HC",
            "score": 0.2,
            "metrics": {"residual_p95": 0.18, "roi_to_non_roi_ratio": 0.7},
        },
    ])

    result = evaluate_output_root(root)

    # p95 should have all 3 subjects
    assert result["metrics"]["residual_p95"]["n_scored"] == 3
    assert result["metrics"]["residual_p95"]["n_missing"] == 0

    # roi_ratio should have only 2 subjects
    assert result["metrics"]["roi_to_non_roi_ratio"]["n_scored"] == 2
    assert result["metrics"]["roi_to_non_roi_ratio"]["n_missing"] == 1

    # residual_mean should have 3 subjects from manifest
    assert result["metrics"]["residual_mean"]["n_scored"] == 3
    assert result["metrics"]["residual_mean"]["n_missing"] == 0
