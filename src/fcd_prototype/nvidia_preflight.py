"""Preflight checks for NVIDIA NV-Generate-MR-Brain integration.

The released rflow-mr-brain workflow can produce a brain MRI prior with
modality, output-size, and spacing controls. The Hugging Face model card lists
multi-modality generation for T1, FLAIR, T2, and SWI, while cross-modality
synthesis is still described as "coming soon"; the NVIDIA inference guide also
states that MR brain is image-only and has no MR ControlNet path. The clinically
important boundary for this project is therefore narrow: the released workflow
does not accept a patient MRI plus lesion mask for learned FCD inpainting. This
module turns that boundary into an explicit readiness report before running the
costly GPU inference path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from fcd_prototype.synthetic_normal import (
    _MODALITY_CODES,
    _REQUIRED_RELATIVE_PATHS,
    _choose_generation_grid,
    _get_nvidia_git_commit,
    _infer_modality_code,
    _load_canonical,
    _voxel_spacing_from_affine,
)


_RECOMMENDED_MR_BRAIN_FOV_MM: dict[str, dict[str, tuple[float, float, float]]] = {
    "t1": {
        "axial": (240.0, 240.0, 174.0),
        "sagittal": (176.0, 250.0, 250.0),
        "coronal": (240.0, 200.0, 240.0),
    },
    "t1w": {
        "axial": (240.0, 240.0, 174.0),
        "sagittal": (176.0, 250.0, 250.0),
        "coronal": (240.0, 200.0, 240.0),
    },
    "t2": {
        "axial": (240.0, 240.0, 158.0),
        "sagittal": (162.0, 240.0, 240.0),
        "coronal": (200.0, 180.0, 200.0),
    },
    "t2w": {
        "axial": (240.0, 240.0, 158.0),
        "sagittal": (162.0, 240.0, 240.0),
        "coronal": (200.0, 180.0, 200.0),
    },
    "flair": {
        "axial": (250.0, 250.0, 175.0),
        "sagittal": (176.0, 250.0, 250.0),
        "coronal": (250.0, 200.0, 250.0),
    },
}

_DOCUMENTED_MODEL_CARD_MODALITIES = ("T1", "FLAIR", "T2", "SWI")
_DOCUMENTED_MAX_OUTPUT_DIM = (512, 512, 256)
_DOCUMENTED_MIN_SPACING_MM = (0.45, 0.45, 0.7)
_DOCUMENTED_MODEL_CARD_URL = "https://huggingface.co/nvidia/NV-Generate-MR-Brain"
_DOCUMENTED_MODEL_CARD_TASK = "unconditional-image-generation"
_DOCUMENTED_MODEL_CARD_RELEASE_DATE = "2026-03-16"
_DOCUMENTED_MODEL_CARD_LICENSE = "NVIDIA Open Model License"
_DOCUMENTED_TRAINING_DATASET = "Forithmus/MR-RATE"
_DOCUMENTED_TRAINING_SCAN_COUNT_APPROX = 28_000
_DOCUMENTED_TESTING_SCAN_COUNT_APPROX = 8_000
_DOCUMENTED_EVALUATION_SCAN_COUNT_APPROX = 4_000
_DOCUMENTED_MODEL_CARD_INPUTS = (
    "number_of_output_samples",
    "modality_label",
    "output_size",
    "spacing",
)
_SUPPORTED_MODALITY_HINTS = tuple(sorted(_MODALITY_CODES))
_PREFERRED_DIMS = {
    (256, 256, 256),
    (128, 256, 256),
    (256, 128, 256),
    (256, 256, 128),
    _DOCUMENTED_MAX_OUTPUT_DIM,
}


def _candidate_nvidia_roots(explicit_root: Optional[Path]) -> list[Path]:
    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(Path(explicit_root).expanduser())
    for env_name in ("NVIDIA_GENERATE_CTMR_ROOT", "NVIDIA_MAISI_ROOT", "NVIDIA_MAISI_WEIGHTS"):
        raw = os.environ.get(env_name)
        if not raw:
            continue
        p = Path(raw).expanduser()
        if p.name == "models":
            p = p.parent
        candidates.append(p)

    deduped: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped


def _infer_plane(shape: tuple[int, int, int], affine: np.ndarray) -> str:
    spacing = _voxel_spacing_from_affine(affine)
    fov = np.asarray(shape[:3], dtype=np.float64) * spacing
    shortest = int(np.argmin(fov))
    return ("sagittal", "coronal", "axial")[shortest]


def _root_status(root: Optional[Path]) -> dict[str, object]:
    if root is None:
        return {
            "available": False,
            "root": None,
            "missing_files": list(_REQUIRED_RELATIVE_PATHS),
            "git_commit": None,
        }
    root = Path(root)
    missing = [rel for rel in _REQUIRED_RELATIVE_PATHS if not (root / rel).is_file()]
    return {
        "available": root.is_dir() and not missing,
        "root": str(root),
        "missing_files": missing,
        "git_commit": _get_nvidia_git_commit(root) if root.is_dir() else None,
    }


def _spacing_status(dim: list[int], spacing: list[float]) -> dict[str, object]:
    dim_arr = np.asarray(dim, dtype=np.int64)
    spacing_arr = np.asarray(spacing, dtype=np.float64)
    dim_divisible = bool(np.all(dim_arr % 16 == 0))
    dim_within_max = bool(np.all(dim_arr <= np.array(_DOCUMENTED_MAX_OUTPUT_DIM)))
    return {
        "dim_divisible_by_16": dim_divisible,
        "dim_preferred_layout": tuple(int(x) for x in dim) in _PREFERRED_DIMS,
        "dim_supported_by_documented_constraints": dim_divisible and dim_within_max,
        "dim_within_model_max": dim_within_max,
        "spacing_within_documented_minimum": bool(np.all(spacing_arr >= np.array(_DOCUMENTED_MIN_SPACING_MM))),
        "spacing_positive": bool(np.all(spacing_arr > 0)),
    }


def preflight_nvidia_mr_brain(
    *,
    patient_volume: Optional[Path] = None,
    roi_mask: Optional[Path] = None,
    modality_hint: Optional[str] = "FLAIR",
    nvidia_root: Optional[Path] = None,
) -> dict[str, object]:
    """Return a JSON-serializable NVIDIA MR-Brain readiness report."""
    roots = _candidate_nvidia_roots(nvidia_root)
    selected_root = next((p for p in roots if p.is_dir()), roots[0] if roots else None)
    root = _root_status(selected_root)

    modality_key = (modality_hint or "FLAIR").lower()
    modality_supported = modality_key in _MODALITY_CODES
    modality_code = _infer_modality_code(modality_hint)

    patient: dict[str, object] = {"provided": False}
    dim: list[int]
    spacing: list[float]
    plane = "unknown"
    recommended_fov: Optional[tuple[float, float, float]] = None
    fov_delta_pct: Optional[list[float]] = None

    if patient_volume is not None:
        data, affine, _ = _load_canonical(Path(patient_volume))
        shape = tuple(int(x) for x in data.shape[:3])
        patient_spacing = _voxel_spacing_from_affine(affine)
        patient_fov = np.asarray(shape, dtype=np.float64) * patient_spacing
        plane = _infer_plane(shape, affine)
        dim, spacing = _choose_generation_grid(shape, affine)
        recommended_fov = _RECOMMENDED_MR_BRAIN_FOV_MM.get(modality_key, {}).get(plane)
        if recommended_fov is not None:
            rec = np.asarray(recommended_fov, dtype=np.float64)
            fov = np.asarray(dim, dtype=np.float64) * np.asarray(spacing, dtype=np.float64)
            fov_delta_pct = (np.abs(fov - rec) / np.maximum(rec, 1e-8) * 100.0).tolist()
        patient = {
            "provided": True,
            "path": str(patient_volume),
            "shape": list(shape),
            "spacing": [float(x) for x in patient_spacing.tolist()],
            "fov_mm": [float(x) for x in patient_fov.tolist()],
            "inferred_plane": plane,
        }
    else:
        dim, spacing = _choose_generation_grid((160, 256, 256), None)
        recommended_fov = _RECOMMENDED_MR_BRAIN_FOV_MM.get(modality_key, {}).get("sagittal")

    generation_fov = [float(d) * float(s) for d, s in zip(dim, spacing)]
    spacing_checks = _spacing_status(dim, spacing)

    roi = {
        "provided": roi_mask is not None,
        "path": str(roi_mask) if roi_mask is not None else None,
        "exists": bool(roi_mask is not None and Path(roi_mask).is_file()),
    }
    if roi["exists"]:
        roi_data = nib.as_closest_canonical(nib.load(str(roi_mask))).get_fdata(dtype=np.float32)
        if roi_data.ndim == 4:
            roi_data = roi_data[..., 0]
        roi["positive_voxels"] = int((roi_data > 0).sum())

    warnings: list[str] = []
    blockers: list[str] = []
    if not root["available"]:
        blockers.append("NVIDIA repo/weights are not ready; see missing_files.")
    if not modality_supported:
        blockers.append(f"Unsupported modality hint {modality_hint!r}; supported: {_SUPPORTED_MODALITY_HINTS}.")
    if not all(bool(spacing_checks[k]) for k in (
        "dim_divisible_by_16",
        "dim_supported_by_documented_constraints",
        "dim_within_model_max",
        "spacing_within_documented_minimum",
        "spacing_positive",
    )):
        blockers.append("Generation dim/spacing fails one or more NVIDIA MR-Brain geometry checks.")
    if fov_delta_pct is not None and max(fov_delta_pct) > 35.0:
        warnings.append(
            "Requested FOV is more than 35% away from the documented MR-Brain training FOV for "
            f"{modality_key}/{plane}; output may be out-of-distribution."
        )
    if roi["provided"] and not roi["exists"]:
        blockers.append("ROI path was provided but does not exist.")
    if roi.get("positive_voxels") == 0:
        warnings.append("ROI exists but contains no positive voxels; NVIDIA mode will not create a lesion-specific hybrid.")

    limitations = [
        "NV-Generate-MR-Brain rflow-mr-brain is image-only/parameter-conditioned in the released repo; the NVIDIA inference guide says there is no MR ControlNet path.",
        "The released MR-Brain path does not provide patient MRI + ROI mask conditioned FCD inpainting.",
        "This project can only use NVIDIA output as an unconditional multi-modality image-only prior plus post-hoc ROI hybridization.",
        "A publishable patient-specific counterfactual claim requires a separate conditioning/fine-tuning method or a defensible cohort-prior baseline.",
    ]

    ready = not blockers
    model_card_contract = {
        "model_card_url": _DOCUMENTED_MODEL_CARD_URL,
        "model_card_task": _DOCUMENTED_MODEL_CARD_TASK,
        "model_card_release_date": _DOCUMENTED_MODEL_CARD_RELEASE_DATE,
        "model_card_license": _DOCUMENTED_MODEL_CARD_LICENSE,
        "input_parameters_documented": list(_DOCUMENTED_MODEL_CARD_INPUTS),
        "output_artifact": "3D synthetic brain MRI NIfTI",
        "supported_modalities_documented": list(_DOCUMENTED_MODEL_CARD_MODALITIES),
        "max_output_dim_documented": list(_DOCUMENTED_MAX_OUTPUT_DIM),
        "min_spacing_mm_documented": list(_DOCUMENTED_MIN_SPACING_MM),
        "training_dataset_documented": _DOCUMENTED_TRAINING_DATASET,
        "training_scan_count_approx": _DOCUMENTED_TRAINING_SCAN_COUNT_APPROX,
        "testing_scan_count_approx": _DOCUMENTED_TESTING_SCAN_COUNT_APPROX,
        "evaluation_scan_count_approx": _DOCUMENTED_EVALUATION_SCAN_COUNT_APPROX,
        "commercial_use_ready_documented": True,
        "hf_inference_provider_deployed": False,
        "local_repo_weights_required": True,
        "clinical_validation_documented": False,
        "clinical_diagnostic_use_allowed": False,
        "cross_modality_synthesis_documented_status": "coming_soon",
        "patient_mri_input_documented": False,
        "roi_mask_input_documented": False,
        "lesion_mask_input_documented": False,
        "patient_conditioned_counterfactual_documented": False,
        "fcd_mask_guided_generation_documented": False,
    }
    local_adapter_contract = {
        "inference_mode": "official_diff_model_infer_image_only_prior_then_posthoc_roi_hybrid",
        "uses_official_diff_model_infer": True,
        "passes_patient_mri_to_nvidia_model": False,
        "passes_roi_mask_to_nvidia_model": False,
        "released_workflow_accepts_patient_mri_or_roi_mask": False,
        "posthoc_roi_hybridization": True,
        "patient_grid_alignment": "patient_centered_fov_affine_resample",
    }
    paper_claim_readiness = {
        "nvidia_inference_claim_allowed": ready,
        "population_prior_residual_claim_allowed": ready,
        "model_card_task": _DOCUMENTED_MODEL_CARD_TASK,
        "model_card_supports_original_pitch": False,
        "model_card_supports_patient_conditioned_counterfactual": False,
        "model_card_supports_fcd_mask_guided_generation": False,
        "released_workflow_accepts_patient_mri_or_roi_mask": False,
        "paper_claim_requires_new_conditioning_method": True,
        "patient_specific_healthy_counterfactual_claim_allowed": False,
        "patient_conditioned_lesion_inpainting_claim_allowed": False,
        "fcd_mask_guided_generation_claim_allowed": False,
        "claim_boundary": "nvidia_image_only_prior_not_patient_conditioned_counterfactual",
        "patient_conditioned_blocker_reason": (
            "Released NV-Generate-MR-Brain exposes parameter-conditioned synthetic "
            "MRI generation, but not patient MRI plus ROI/FCD-mask conditioned "
            "healthy inpainting."
        ),
        "required_for_original_pitch": [
            "patient MRI conditioning",
            "lesion or ROI mask conditioning",
            "validation that the generated ROI is healthy tissue rather than an unrelated population prior",
        ],
        "paper_options": [
            "Use NVIDIA as an unconditional healthy population prior plus post-hoc ROI hybridization.",
            "Use a healthy-control Gaussian prior as the non-NVIDIA baseline.",
            "Implement a new patient-conditioned adapter, fine-tune, or inpainting method before claiming the original pitch.",
        ],
        "recommended_paper_framing": (
            "Use NVIDIA MR-Brain as a healthy population prior or augmentation source; "
            "do not claim it directly creates a patient-specific healthy replacement for FCD."
        ),
    }
    report = {
        "method": "nvidia_mr_brain_preflight",
        "ready_for_inference": ready,
        "ready_for_patient_conditioned_counterfactual": False,
        "root": root,
        "patient": patient,
        "roi": roi,
        "modality_hint": modality_hint,
        "modality_code": modality_code,
        "modality_supported": modality_supported,
        "model_card_url": _DOCUMENTED_MODEL_CARD_URL,
        "model_card_task": _DOCUMENTED_MODEL_CARD_TASK,
        "generation_dim": dim,
        "generation_spacing": spacing,
        "generation_fov_mm": generation_fov,
        "recommended_fov_mm": list(recommended_fov) if recommended_fov else None,
        "fov_delta_pct": fov_delta_pct,
        "geometry_checks": spacing_checks,
        "capability_boundary": "image_only_unconditional_generation_plus_roi_hybrid",
        "model_card_contract": model_card_contract,
        "local_adapter_contract": local_adapter_contract,
        "paper_claim_readiness": paper_claim_readiness,
        "supported_modalities_documented": list(_DOCUMENTED_MODEL_CARD_MODALITIES),
        "max_output_dim_documented": list(_DOCUMENTED_MAX_OUTPUT_DIM),
        "min_spacing_mm_documented": list(_DOCUMENTED_MIN_SPACING_MM),
        "multi_modality_generation_supported": True,
        "cross_modality_synthesis_status": "coming_soon_on_model_card",
        "cross_modality_synthesis_supported": False,
        "patient_mri_conditioning_supported": False,
        "lesion_mask_inpainting_supported": False,
        "patient_conditioned_inpainting_supported": False,
        "fcd_mask_guided_generation_supported": False,
        "clinical_diagnostic_use_allowed": False,
        "paper_claim_requires_new_conditioning_method": True,
        "warnings": warnings,
        "blockers": blockers,
        "limitations": limitations,
        "sources": [
            _DOCUMENTED_MODEL_CARD_URL,
            "https://github.com/NVIDIA-Medtech/NV-Generate-CTMR/blob/main/docs/inference.md",
        ],
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preflight NVIDIA NV-Generate-MR-Brain readiness.")
    p.add_argument("--patient-volume", type=Path, default=None)
    p.add_argument("--roi-mask", type=Path, default=None)
    p.add_argument("--modality-hint", default="FLAIR")
    p.add_argument("--nvidia-root", type=Path, default=None)
    p.add_argument("--output-json", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    report = preflight_nvidia_mr_brain(
        patient_volume=args.patient_volume,
        roi_mask=args.roi_mask,
        modality_hint=args.modality_hint,
        nvidia_root=args.nvidia_root,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
