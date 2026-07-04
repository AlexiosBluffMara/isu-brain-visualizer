"""Audit synthetic-normal geometry and residual interpretability.

This module checks whether a synthetic normal volume is on the same physical
grid as the patient volume and quantifies how much the synthetic reference
differs outside and inside the lesion ROI.  It does not validate diagnosis; it
creates paper-method evidence that subtraction is being interpreted honestly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates


_ROI_HYBRID_OUTSIDE_SIMILARITY_PASS = 0.95
_ROI_HYBRID_OUTSIDE_SIMILARITY_REVIEW = 0.85
_NVIDIA_MIN_INFERENCE_STEPS = 30


def _load_canonical(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data.astype(np.float32), img.affine.astype(np.float64)


def _same_grid(
    shape: tuple[int, int, int],
    affine: np.ndarray,
    ref_shape: tuple[int, int, int],
    ref_affine: np.ndarray,
) -> bool:
    return tuple(shape[:3]) == tuple(ref_shape) and np.allclose(affine, ref_affine, atol=1e-4)


def _resample_to_reference(
    data: np.ndarray,
    src_affine: np.ndarray,
    ref_shape: tuple[int, int, int],
    ref_affine: np.ndarray,
    *,
    order: int,
) -> np.ndarray:
    if _same_grid(data.shape[:3], src_affine, ref_shape, ref_affine):
        return data.astype(np.float32, copy=True)

    src_inv = np.linalg.inv(src_affine)
    out = np.empty(ref_shape, dtype=np.float32)
    xs = np.arange(ref_shape[0], dtype=np.float32)
    ys = np.arange(ref_shape[1], dtype=np.float32)
    for z0 in range(0, ref_shape[2], 16):
        z1 = min(ref_shape[2], z0 + 16)
        zs = np.arange(z0, z1, dtype=np.float32)
        i, j, k = np.meshgrid(xs, ys, zs, indexing="ij")
        ones = np.ones_like(i, dtype=np.float32)
        ref_vox = np.stack([i, j, k, ones], axis=0).reshape(4, -1)
        world = ref_affine @ ref_vox
        src_vox = src_inv @ world
        sampled = map_coordinates(
            data,
            [src_vox[0], src_vox[1], src_vox[2]],
            order=order,
            mode="nearest",
        )
        out[:, :, z0:z1] = sampled.reshape(ref_shape[0], ref_shape[1], z1 - z0)
    return out.astype(np.float32)


def _normalize_for_metrics(data: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
        return np.clip(finite, 0.0, 1.0).astype(np.float32)
    fg = finite[finite > 0]
    if fg.size == 0:
        return np.zeros_like(finite, dtype=np.float32)
    lo = float(np.percentile(fg, 1))
    hi = float(np.percentile(fg, 99))
    if hi <= lo:
        return np.zeros_like(finite, dtype=np.float32)
    return np.clip((finite - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _masked_metrics(patient: np.ndarray, synthetic: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    valid = mask & np.isfinite(patient) & np.isfinite(synthetic)
    n = int(valid.sum())
    if n == 0:
        return {
            "voxel_count": 0,
            "mae": None,
            "rmse": None,
            "pearson_r": None,
            "similarity_1_minus_mae": None,
        }

    diff = patient[valid] - synthetic[valid]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    p = patient[valid]
    s = synthetic[valid]
    if n > 2 and float(p.std()) > 1e-8 and float(s.std()) > 1e-8:
        corr: Optional[float] = float(np.corrcoef(p, s)[0, 1])
    else:
        corr = None
    return {
        "voxel_count": n,
        "mae": mae,
        "rmse": rmse,
        "pearson_r": corr,
        "similarity_1_minus_mae": float(max(0.0, min(1.0, 1.0 - mae))),
    }


def _counterfactual_class(synthetic_normal_source: str) -> str:
    source = synthetic_normal_source.lower()
    if source.startswith("nvidia_rflow"):
        return "unconditional_nvidia_prior_roi_hybrid"
    if source.startswith("control_prior_roi"):
        return "healthy_control_population_prior_roi_hybrid"
    if source.startswith("control_prior_whole"):
        return "healthy_control_population_prior_whole_brain"
    if source.startswith("contralateral_patient_specific"):
        return "patient_specific_contralateral_geometric_roi_completion"
    if source.startswith("placeholder"):
        return "local_smoothing_placeholder"
    if "patient_copy" in source:
        return "patient_copy_degenerate_reference"
    return "unknown_reference"


def _build_subtraction_readiness_gate(
    *,
    source_class: str,
    synthetic_normal_source: str,
    same_grid_before_resample: bool,
    synthetic_resampled_to_patient_grid: bool,
    roi_voxels: int,
    outside: dict[str, object],
    inside: dict[str, object],
    source_metadata: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    """Classify whether a synthetic normal can support subtraction claims."""
    blockers: list[str] = []
    warnings: list[str] = []

    outside_similarity = outside.get("similarity_1_minus_mae")
    outside_voxels = int(outside.get("voxel_count") or 0)
    inside_voxels = int(inside.get("voxel_count") or 0)

    if source_class == "patient_copy_degenerate_reference":
        blockers.append(
            "synthetic normal is a patient-copy reference, so subtraction is degenerate"
        )
    if source_class == "unknown_reference":
        blockers.append(
            f"synthetic_normal_source {synthetic_normal_source!r} is not recognized"
        )
    if synthetic_resampled_to_patient_grid or not same_grid_before_resample:
        warnings.append(
            "synthetic normal was not saved on the patient grid before audit resampling"
        )

    # NVIDIA-specific metadata blockers. Accept either the compact nested
    # manifest form {"nvidia": {...}} or a direct metadata dict for CLI/tests.
    nvidia_shape_matches_config: object = None
    nvidia_steps_observed: object = None
    if source_class == "unconditional_nvidia_prior_roi_hybrid":
        metadata = source_metadata or {}
        nvidia_meta = {}
        if isinstance(metadata, dict):
            nested = metadata.get("nvidia")
            nvidia_meta = nested if isinstance(nested, dict) else metadata
        if not isinstance(nvidia_meta, dict):
            nvidia_meta = {}
        nvidia_steps_observed = nvidia_meta.get("nvidia_steps")
        if nvidia_steps_observed is None:
            blockers.append(
                "NVIDIA prior source_metadata is missing nvidia_steps; "
                f"required >= {_NVIDIA_MIN_INFERENCE_STEPS} for a valid prior"
            )
        else:
            try:
                nvidia_steps_int = int(nvidia_steps_observed)
            except (TypeError, ValueError):
                blockers.append(
                    f"NVIDIA prior nvidia_steps={nvidia_steps_observed!r} is not an integer"
                )
            else:
                if nvidia_steps_int < _NVIDIA_MIN_INFERENCE_STEPS:
                    blockers.append(
                        f"NVIDIA prior nvidia_steps={nvidia_steps_observed} is below the required "
                        f"minimum of {_NVIDIA_MIN_INFERENCE_STEPS} (rflow-mr-brain default)"
                    )
        nvidia_shape_matches_config = nvidia_meta.get(
            "nvidia_generated_shape_matches_config"
        )
        if nvidia_shape_matches_config is False:
            blockers.append(
                "NVIDIA prior nvidia_generated_shape_matches_config is explicitly false"
            )

    if roi_voxels > 0:
        if outside_voxels <= 0:
            blockers.append("ROI-positive scan has no outside-ROI reference voxels")
        elif outside_similarity is None:
            blockers.append("outside-ROI preservation similarity is unavailable")
        elif float(outside_similarity) < _ROI_HYBRID_OUTSIDE_SIMILARITY_REVIEW:
            blockers.append(
                "outside-ROI preservation similarity is below the review threshold"
            )
        elif float(outside_similarity) < _ROI_HYBRID_OUTSIDE_SIMILARITY_PASS:
            warnings.append(
                "outside-ROI preservation similarity is below the pass threshold"
            )
        if inside_voxels <= 0:
            warnings.append("ROI-positive scan has no inside-ROI audit voxels")
    else:
        if source_class.endswith("_roi_hybrid"):
            blockers.append("ROI-hybrid source was used but the audit found no ROI voxels")
        if source_class == "local_smoothing_placeholder":
            blockers.append("placeholder synthetic normal without an ROI is not a healthy reference")

    status = "fail" if blockers else ("review" if warnings else "pass")
    patient_conditioned_counterfactual_claim_allowed = (
        source_class == "patient_specific_contralateral_geometric_roi_completion"
        and roi_voxels > 0
    )
    gaussian_subtraction_claim_allowed = status in {"pass", "review"}
    if source_class.startswith("unconditional_nvidia"):
        claim_class = "research_prior_roi_hybrid_not_patient_conditioned"
    elif source_class.startswith("healthy_control_population"):
        claim_class = "healthy_control_population_reference"
    elif patient_conditioned_counterfactual_claim_allowed:
        claim_class = "patient_specific_geometric_counterfactual"
    elif source_class == "local_smoothing_placeholder":
        claim_class = "placeholder_debug_reference"
    else:
        claim_class = "not_supported_for_patient_specific_counterfactual"

    return {
        "status": status,
        "blockers": blockers,
        "warnings": warnings,
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "thresholds": {
            "roi_hybrid_outside_similarity_pass": _ROI_HYBRID_OUTSIDE_SIMILARITY_PASS,
            "roi_hybrid_outside_similarity_review": _ROI_HYBRID_OUTSIDE_SIMILARITY_REVIEW,
            "nvidia_min_inference_steps": _NVIDIA_MIN_INFERENCE_STEPS,
        },
        "nvidia_steps": nvidia_steps_observed,
        "nvidia_min_inference_steps": _NVIDIA_MIN_INFERENCE_STEPS,
        "nvidia_generated_shape_matches_config": nvidia_shape_matches_config,
        "outside_roi_similarity_1_minus_mae": outside_similarity,
        "roi_voxels": roi_voxels,
        "same_grid_before_resample": same_grid_before_resample,
        "synthetic_resampled_to_patient_grid": synthetic_resampled_to_patient_grid,
        "gaussian_subtraction_claim_allowed": gaussian_subtraction_claim_allowed,
        "patient_conditioned_counterfactual_claim_allowed": (
            patient_conditioned_counterfactual_claim_allowed
        ),
        "claim_class": claim_class,
    }


def audit_synthetic_normal(
    *,
    patient_volume: Path,
    synthetic_normal: Path,
    output_json: Optional[Path] = None,
    roi_mask: Optional[Path] = None,
    synthetic_normal_source: str = "",
    source_metadata: Optional[dict[str, object]] = None,
    roi_dilation_iterations: int = 5,
) -> dict[str, object]:
    """Return and optionally write a synthetic-normal audit dictionary."""
    patient_data, patient_affine = _load_canonical(Path(patient_volume))
    synthetic_data, synthetic_affine = _load_canonical(Path(synthetic_normal))
    patient_shape = tuple(int(x) for x in patient_data.shape[:3])

    same_grid_before = _same_grid(synthetic_data.shape[:3], synthetic_affine, patient_shape, patient_affine)
    synthetic_resampled = _resample_to_reference(
        synthetic_data,
        synthetic_affine,
        patient_shape,
        patient_affine,
        order=1,
    )

    roi_bool = np.zeros(patient_shape, dtype=bool)
    roi_resampled = False
    if roi_mask is not None and Path(roi_mask).is_file():
        roi_data, roi_affine = _load_canonical(Path(roi_mask))
        roi_resampled = not _same_grid(roi_data.shape[:3], roi_affine, patient_shape, patient_affine)
        roi_on_grid = _resample_to_reference(
            (roi_data > 0).astype(np.float32),
            roi_affine,
            patient_shape,
            patient_affine,
            order=0,
        )
        roi_bool = roi_on_grid > 0.5

    patient_norm = _normalize_for_metrics(patient_data)
    synthetic_norm = _normalize_for_metrics(synthetic_resampled)
    brain_mask = np.isfinite(patient_norm) & (patient_norm > 1e-5)
    if int(brain_mask.sum()) < 32:
        brain_mask = np.isfinite(patient_norm)

    if roi_bool.any():
        roi_support = binary_dilation(roi_bool, iterations=roi_dilation_iterations)
        outside_mask = brain_mask & ~roi_support
        inside_mask = brain_mask & roi_bool
    else:
        roi_support = np.zeros(patient_shape, dtype=bool)
        outside_mask = brain_mask
        inside_mask = np.zeros(patient_shape, dtype=bool)

    outside = _masked_metrics(patient_norm, synthetic_norm, outside_mask)
    inside = _masked_metrics(patient_norm, synthetic_norm, inside_mask)
    inside_mae = inside.get("mae")
    outside_mae = outside.get("mae")
    if isinstance(inside_mae, float) and isinstance(outside_mae, float) and outside_mae > 1e-8:
        inside_to_outside = inside_mae / outside_mae
    else:
        inside_to_outside = None

    source_class = _counterfactual_class(synthetic_normal_source)
    patient_conditioned = False
    patient_specific_counterfactual = (
        source_class == "patient_specific_contralateral_geometric_roi_completion"
    )
    if source_class == "patient_copy_degenerate_reference":
        subtraction_interpretation = "degenerate_patient_copy_reference"
    elif patient_specific_counterfactual and roi_bool.any():
        subtraction_interpretation = "patient_specific_contralateral_roi_residual"
    elif roi_bool.any():
        subtraction_interpretation = "roi_hybrid_same_patient_grid_residual"
    else:
        subtraction_interpretation = "whole_brain_same_patient_grid_residual"

    if patient_specific_counterfactual:
        claim_boundary = "patient_specific_geometric_counterfactual_not_learned_inpainting"
    else:
        claim_boundary = (
            "not_patient_conditioned_inpainting"
            if not patient_conditioned
            else "patient_conditioned_inpainting"
        )

    audit: dict[str, object] = {
        "method": "synthetic_normal_geometry_and_claim_audit",
        "patient_volume": str(patient_volume),
        "synthetic_normal": str(synthetic_normal),
        "roi_mask": str(roi_mask) if roi_mask else None,
        "synthetic_normal_source": synthetic_normal_source,
        "counterfactual_class": source_class,
        "patient_conditioned_inpainting": patient_conditioned,
        "patient_specific_counterfactual": patient_specific_counterfactual,
        "claim_boundary": claim_boundary,
        "subtraction_interpretation": subtraction_interpretation,
        "patient_shape": list(patient_shape),
        "synthetic_shape": list(synthetic_data.shape[:3]),
        "same_grid_before_resample": same_grid_before,
        "synthetic_resampled_to_patient_grid": not same_grid_before,
        "roi_voxels": int(roi_bool.sum()),
        "roi_resampled_to_patient_grid": roi_resampled,
        "roi_support_voxels_after_dilation": int(roi_support.sum()),
        "brain_voxels": int(brain_mask.sum()),
        "outside_roi": outside,
        "inside_roi": inside,
        "inside_to_outside_mae_ratio": inside_to_outside,
        "source_metadata": source_metadata or {},
        "limitations": [
            "This audit checks geometry and intensity preservation, not clinical validity.",
            "High outside-ROI similarity can be expected for ROI-hybrid methods because the patient is copied outside the ROI.",
            "Patient-conditioned inpainting is false for current placeholder, control-prior, and NVIDIA image-only modes.",
        ],
    }
    audit["subtraction_readiness_gate"] = _build_subtraction_readiness_gate(
        source_class=source_class,
        synthetic_normal_source=synthetic_normal_source,
        same_grid_before_resample=same_grid_before,
        synthetic_resampled_to_patient_grid=not same_grid_before,
        roi_voxels=int(roi_bool.sum()),
        outside=outside,
        inside=inside,
        source_metadata=source_metadata,
    )

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(audit, indent=2))
    return audit


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit synthetic-normal geometry and claim boundary.")
    p.add_argument("--patient-volume", type=Path, required=True)
    p.add_argument("--synthetic-normal", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--roi-mask", type=Path, default=None)
    p.add_argument("--synthetic-normal-source", default="")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    audit = audit_synthetic_normal(
        patient_volume=args.patient_volume,
        synthetic_normal=args.synthetic_normal,
        output_json=args.output_json,
        roi_mask=args.roi_mask,
        synthetic_normal_source=args.synthetic_normal_source,
    )
    outside = audit["outside_roi"]
    print(
        "Synthetic-normal audit complete: "
        f"class={audit['counterfactual_class']} "
        f"outside_mae={outside.get('mae')}"
    )


if __name__ == "__main__":
    main()
