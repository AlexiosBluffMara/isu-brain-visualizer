"""Synthetic normal brain generation adapter.

Supports four modes:

  placeholder      -- Deterministic local counterfactual.  Fills the ROI region
                      by blending intensity from surrounding tissue, creating a
                      plausible "what-if no lesion" volume.  Explicitly labeled as
                      NOT diagnostic and NOT NVIDIA.

  nvidia           -- Real NV-Generate-CTMR (rflow-mr-brain) integration.
                      Calls the official scripts/diff_model_infer.py via subprocess,
                      then aligns and ROI-hybridizes the generated prior into
                      patient space.  Never fabricates NVIDIA output.

  control-prior    -- Cohort fallback for paper experiments when NVIDIA is not
                      configured or not patient-conditioned.  Healthy controls are
                      affine-resampled into the patient grid, intensity matched,
                      and averaged into a healthy prior. With an ROI, the prior is
                      hybridized into the lesion region. Without an ROI, the prior
                      becomes a whole-brain reference instead of a patient copy.

  contralateral    -- Patient-specific healthy counterfactual for ROI-positive FCD
                      scans. Mirrors the contralateral homologous region across the
                      canonical L-R axis and feathers it into the lesion ROI.
                      Preserves the patient's own skull/ventricle/global anatomy
                      outside the ROI. No external data required.  Requires a non-
                      empty ROI; control subjects (no ROI) receive a patient copy
                      with an explicit skip label. Not diagnostic.

Code repo:  https://github.com/NVIDIA-Medtech/NV-Generate-CTMR.git
Model card: https://huggingface.co/nvidia/NV-Generate-MR-Brain  (weights + license)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal, Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates


SyntheticMode = Literal["placeholder", "nvidia", "control-prior", "contralateral"]
_CONTROL_PRIOR_SIMILARITY_PASS_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# Modality mapping (from configs/modality_mapping.json in the official repo)
# ---------------------------------------------------------------------------
_MODALITY_CODES: dict[str, int] = {
    "flair": 11,
    "t1": 9,
    "t1w": 9,
    "t2": 10,
    "t2w": 10,
    "swi": 20,
}

_REQUIRED_RELATIVE_PATHS = [
    "scripts/diff_model_infer.py",
    "configs/config_network_rflow.json",
    "configs/environment_maisi_diff_model_rflow-mr-brain.json",
    "configs/config_maisi_diff_model_rflow-mr-brain.json",
    "models/autoencoder_v1.pt",
    "models/diff_unet_3d_rflow-mr-brain_v0.pt",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_canonical(path: Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data, img.affine.copy(), img.header.copy()


def _robust_normalize(data: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(data, 1), np.percentile(data, 99)
    if hi <= lo:
        return data.copy()
    return ((data - lo) / (hi - lo)).clip(0.0, 1.0).astype(np.float32)


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
    order: int = 1,
    chunk_slices: int = 16,
) -> np.ndarray:
    """Affine-aware resampling from source voxel grid into reference voxel grid.

    This is intentionally grid/affine based rather than a shape-only zoom.  MRI
    datasets frequently mix voxel sizes and origins; ignoring affine metadata
    can move a lesion or control prior to the wrong physical location.
    """
    if _same_grid(data.shape[:3], src_affine, ref_shape, ref_affine):
        return data.astype(np.float32, copy=True)

    src_inv = np.linalg.inv(src_affine)
    out = np.empty(ref_shape, dtype=np.float32)

    xs = np.arange(ref_shape[0], dtype=np.float32)
    ys = np.arange(ref_shape[1], dtype=np.float32)

    for z0 in range(0, ref_shape[2], chunk_slices):
        z1 = min(ref_shape[2], z0 + chunk_slices)
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


def _match_intensity_to_patient(
    prior: np.ndarray,
    patient: np.ndarray,
    reference_mask: np.ndarray,
) -> np.ndarray:
    """Match prior mean/std to patient over a trusted non-lesion reference mask."""
    mask = reference_mask & np.isfinite(prior) & np.isfinite(patient)
    if int(mask.sum()) < 32:
        mask = np.isfinite(prior) & np.isfinite(patient)
    if int(mask.sum()) < 32:
        return prior.astype(np.float32)

    p_mean = float(patient[mask].mean())
    p_std = float(patient[mask].std())
    pr_mean = float(prior[mask].mean())
    pr_std = float(prior[mask].std())
    if p_std < 1e-6 or pr_std < 1e-6:
        return prior.astype(np.float32)
    matched = (prior - pr_mean) / (pr_std + 1e-8) * p_std + p_mean
    return matched.clip(0.0, 1.0).astype(np.float32)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size < 2 or b.size < 2:
        return None
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std <= 1e-8 or b_std <= 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _reference_similarity_metrics(
    patient: np.ndarray,
    candidate: np.ndarray,
    reference_mask: np.ndarray,
) -> dict[str, object]:
    mask = reference_mask & np.isfinite(patient) & np.isfinite(candidate)
    voxel_count = int(mask.sum())
    if voxel_count < 32:
        return {
            "voxel_count": voxel_count,
            "mae": None,
            "rmse": None,
            "correlation": None,
            "similarity_1_minus_mae": None,
        }
    p = patient[mask].astype(np.float64)
    c = candidate[mask].astype(np.float64)
    diff = c - p
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    return {
        "voxel_count": voxel_count,
        "mae": mae,
        "rmse": rmse,
        "correlation": _safe_corr(p, c),
        "similarity_1_minus_mae": float(max(0.0, min(1.0, 1.0 - mae))),
    }


def _subject_id_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    name = path.name
    if "_FLAIR" in name:
        return name.split("_FLAIR", 1)[0]
    if "_T1w" in name:
        return name.split("_T1w", 1)[0]
    return path.stem.replace(".nii", "")


def _save_float_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header,
    output_path: Path,
) -> None:
    img_out = nib.Nifti1Image(data.astype(np.float32), affine, header)
    img_out.header.set_data_dtype(np.float32)
    nib.save(img_out, str(output_path))


def _voxel_spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).astype(np.float64)
    spacing[spacing <= 1e-8] = 1.0
    return spacing


def _valid_control_paths(control_prior_volumes: Optional[list[Path]]) -> list[Path]:
    return [Path(p) for p in (control_prior_volumes or []) if Path(p).is_file()]


def _patient_brain_mask(patient_norm: np.ndarray) -> np.ndarray:
    brain_mask = np.isfinite(patient_norm) & (patient_norm > 1e-5)
    if int(brain_mask.sum()) < 32:
        brain_mask = np.isfinite(patient_norm)
    return brain_mask


def _build_control_prior_volume(
    patient_norm: np.ndarray,
    patient_shape: tuple[int, int, int],
    patient_affine: np.ndarray,
    control_paths: list[Path],
    reference_mask: np.ndarray,
) -> tuple[np.ndarray, list[Path], list[str], dict[str, object]]:
    prior_sum = np.zeros(patient_shape, dtype=np.float64)
    used_paths: list[Path] = []
    skipped: list[str] = []
    per_control_quality: list[dict[str, object]] = []

    for control_path in control_paths:
        try:
            control_data, control_affine, _ = _load_canonical(control_path)
            control_norm = _robust_normalize(control_data)
            control_resampled = _resample_to_reference(
                control_norm,
                control_affine,
                patient_shape,
                patient_affine,
                order=1,
            )
            control_matched = _match_intensity_to_patient(
                control_resampled,
                patient_norm,
                reference_mask,
            )
            metrics = _reference_similarity_metrics(
                patient_norm,
                control_matched,
                reference_mask,
            )
            per_control_quality.append({
                "subject": _subject_id_from_path(control_path),
                "path": str(control_path),
                **metrics,
            })
            prior_sum += control_matched.astype(np.float64)
            used_paths.append(control_path)
        except Exception as exc:
            skipped.append(f"{control_path}: {exc}")

    if not used_paths:
        raise ValueError(
            "control-prior synthetic mode could not use any supplied healthy-control "
            f"volumes. Skipped: {skipped[:3]}"
        )

    prior = (prior_sum / float(len(used_paths))).astype(np.float32)
    prior_metrics = _reference_similarity_metrics(patient_norm, prior, reference_mask)
    similarities = [
        float(item["similarity_1_minus_mae"])
        for item in per_control_quality
        if item.get("similarity_1_minus_mae") is not None
    ]
    maes = [
        float(item["mae"])
        for item in per_control_quality
        if item.get("mae") is not None
    ]
    corrs = [
        float(item["correlation"])
        for item in per_control_quality
        if item.get("correlation") is not None
    ]
    prior_similarity = prior_metrics.get("similarity_1_minus_mae")
    if prior_similarity is None:
        gate = "insufficient_reference"
    elif float(prior_similarity) >= _CONTROL_PRIOR_SIMILARITY_PASS_THRESHOLD:
        gate = "pass"
    else:
        gate = "review"
    quality = {
        "control_prior_quality_gate": gate,
        "control_prior_quality_gate_metric": "prior_reference_similarity_1_minus_mae",
        "control_prior_quality_gate_threshold": _CONTROL_PRIOR_SIMILARITY_PASS_THRESHOLD,
        "control_prior_reference_voxel_count": prior_metrics.get("voxel_count"),
        "control_prior_reference_mae": prior_metrics.get("mae"),
        "control_prior_reference_rmse": prior_metrics.get("rmse"),
        "control_prior_reference_correlation": prior_metrics.get("correlation"),
        "control_prior_reference_similarity_1_minus_mae": prior_similarity,
        "control_prior_control_similarity_mean": (
            float(np.mean(similarities)) if similarities else None
        ),
        "control_prior_control_similarity_min": (
            float(np.min(similarities)) if similarities else None
        ),
        "control_prior_control_mae_mean": float(np.mean(maes)) if maes else None,
        "control_prior_control_correlation_mean": (
            float(np.mean(corrs)) if corrs else None
        ),
        "control_prior_per_control_quality": per_control_quality,
    }
    return prior, used_paths, skipped, quality


def _control_prior_subjects(paths: list[Path]) -> list[str]:
    return [_subject_id_from_path(path) for path in paths]


def _patient_copy_response(
    patient_data: np.ndarray,
    patient_affine: np.ndarray,
    patient_header: nib.Nifti1Header,
    output_path: Path,
    *,
    source: str,
    skip_reason: str,
    roi_voxels: int,
    notes: list[str],
    limitations: list[str],
) -> dict:
    _save_float_nifti(patient_data, patient_affine, patient_header, output_path)
    return {
        "synthetic_normal_source": source,
        "synthetic_normal_is_diagnostic": False,
        "mode": "control-prior",
        "control_prior_skip_reason": skip_reason,
        "control_prior_count": 0,
        "roi_voxels": roi_voxels,
        "notes": notes,
        "limitations": limitations,
    }


def _whole_brain_control_prior_response(
    patient_data: np.ndarray,
    patient_affine: np.ndarray,
    patient_header: nib.Nifti1Header,
    output_path: Path,
    control_paths: list[Path],
    subject_id: Optional[str],
    *,
    skip_reason: str,
    roi_voxels: int,
) -> dict:
    patient_norm = _robust_normalize(patient_data)
    brain_mask = _patient_brain_mask(patient_norm)
    prior, used_paths, skipped, quality = _build_control_prior_volume(
        patient_norm=patient_norm,
        patient_shape=tuple(patient_norm.shape[:3]),
        patient_affine=patient_affine,
        control_paths=control_paths,
        reference_mask=brain_mask,
    )
    synthetic = patient_norm.copy()
    synthetic[brain_mask] = prior[brain_mask]
    _save_float_nifti(synthetic, patient_affine, patient_header, output_path)

    return {
        "synthetic_normal_source": "control_prior_whole_brain_prior",
        "synthetic_normal_is_diagnostic": False,
        "mode": "control-prior",
        "subject_id": subject_id,
        "control_prior_count": len(used_paths),
        "control_prior_subjects": _control_prior_subjects(used_paths),
        "control_prior_volumes": [str(path) for path in used_paths],
        "control_prior_aggregation": "whole_brain_affine_resampled_intensity_matched_mean",
        "control_prior_alignment": "canonical RAS affine-resampling to patient grid",
        "control_prior_scope": "whole_brain_no_roi",
        "control_prior_skip_reason": skip_reason,
        "control_prior_skipped_count": len(skipped),
        "control_prior_skipped": skipped[:10],
        **quality,
        "roi_voxels": roi_voxels,
        "notes": [
            "CONTROL-PRIOR whole-brain synthetic normal: no usable lesion ROI was available, "
            "so healthy-control volumes were canonicalized, affine-resampled into patient "
            "space, intensity-matched over the patient brain mask, averaged, and written "
            "as a whole-brain healthy reference.",
            "This avoids the degenerate patient-copy residual for uploaded/no-ROI scans.",
            "This is a population healthy prior, not lesion-conditioned NVIDIA inpainting.",
        ],
        "limitations": [
            "Whole-brain control prior is not patient-specific diffusion inpainting.",
            "Current alignment is affine/grid resampling only; deformable registration is not implemented.",
            "Residuals may include normal anatomic mismatch, scanner/protocol effects, and registration error.",
            "The prior quality depends on modality, voxel geometry, age distribution, and intensity normalization.",
            "Not validated for clinical diagnosis.",
        ],
    }


# ---------------------------------------------------------------------------
# Placeholder mode
# ---------------------------------------------------------------------------

def _inpaint_roi_placeholder(
    data: np.ndarray,
    roi_mask: np.ndarray,
    dilation_sigma: float = 6.0,
) -> np.ndarray:
    """Smooth-fill the ROI region from surrounding tissue.

    Strategy: heavily smooth the full volume then blend back inside the ROI.
    This is a deterministic morphological estimate, not a generative model.
    """
    mask = (roi_mask > 0).astype(bool)
    smoothed = gaussian_filter(data, sigma=dilation_sigma).astype(np.float32)

    result = data.copy()
    if mask.any():
        result[mask] = smoothed[mask]
    return result


# ---------------------------------------------------------------------------
# NVIDIA dependency check + root resolution
# ---------------------------------------------------------------------------

class NvidiaDependencyError(RuntimeError):
    """Raised when NV-Generate-MR-Brain is not configured."""


_NVIDIA_INSTRUCTIONS = """
NV-Generate-CTMR (rflow-mr-brain) is not available.

To enable NVIDIA synthetic normal generation:

  1. Clone the code repository (weights are NOT included):
       git clone https://github.com/NVIDIA-Medtech/NV-Generate-CTMR.git

  2. Accept the model license on Hugging Face:
       https://huggingface.co/nvidia/NV-Generate-MR-Brain

  3. Download model weights from inside the cloned repo:
       python -m scripts.download_model_data --version rflow-mr-brain --root_dir ./ --model_only

  4. Install runtime deps into the same environment as fcd-build-end-to-end:
       pip install "huggingface_hub>=0.20.0" "safetensors>=0.4" "einops>=0.7" "fire>=0.5" \
           "tqdm>=4.65" "PyYAML>=6.0" "scikit-image>=0.20.0" "tensorboard>=2.14.0"

  5. Set one of the following environment variables to the clone root:
       NVIDIA_GENERATE_CTMR_ROOT  (preferred)
       NVIDIA_MAISI_ROOT
       NVIDIA_MAISI_WEIGHTS       (clone root or models/ subdirectory accepted)

  6. GPU requirement: Linux + NVIDIA GPU with ≥16 GB VRAM recommended.

This adapter never fabricates NVIDIA output.  Every manifest is labeled:
  synthetic_normal_source: nvidia_rflow-mr-brain_roi_hybrid
  synthetic_normal_is_diagnostic: false
""".strip()


def _resolve_nvidia_root() -> Path:
    """Return the validated NV-Generate-CTMR clone root, or raise NvidiaDependencyError."""
    candidates: list[str] = []
    for var in ("NVIDIA_GENERATE_CTMR_ROOT", "NVIDIA_MAISI_ROOT", "NVIDIA_MAISI_WEIGHTS"):
        val = os.environ.get(var, "").strip()
        if val:
            candidates.append(val)

    for raw in candidates:
        p = Path(raw)
        # Accept clone root directly
        if (p / "scripts" / "diff_model_infer.py").is_file():
            return p
        # Accept models/ subdirectory — walk up one level
        if p.name == "models" and (p.parent / "scripts" / "diff_model_infer.py").is_file():
            return p.parent

    if not candidates:
        raise NvidiaDependencyError(
            "No NVIDIA root env var set (NVIDIA_GENERATE_CTMR_ROOT, NVIDIA_MAISI_ROOT, "
            "or NVIDIA_MAISI_WEIGHTS).\n\n" + _NVIDIA_INSTRUCTIONS
        )
    raise NvidiaDependencyError(
        f"NVIDIA root candidates {candidates!r} do not contain scripts/diff_model_infer.py.\n\n"
        + _NVIDIA_INSTRUCTIONS
    )


def _validate_nvidia_root(root: Path) -> None:
    missing = [rel for rel in _REQUIRED_RELATIVE_PATHS if not (root / rel).is_file()]
    if missing:
        raise NvidiaDependencyError(
            f"NVIDIA repo at {root} is missing required files: {missing}\n\n"
            + _NVIDIA_INSTRUCTIONS
        )


def _check_nvidia_available() -> Path:
    """Return validated NVIDIA root, or raise NvidiaDependencyError."""
    root = _resolve_nvidia_root()
    _validate_nvidia_root(root)
    return root


# ---------------------------------------------------------------------------
# NVIDIA dim/spacing/modality helpers
# ---------------------------------------------------------------------------

def _parse_env_list(var: str, default: list) -> list:
    val = os.environ.get(var, "").strip()
    if val:
        try:
            return [float(x) for x in val.split(",")]
        except ValueError:
            pass
    return default


def _choose_generation_grid(
    patient_shape: tuple,
    patient_affine: Optional[np.ndarray] = None,
) -> tuple[list[int], list[float]]:
    """Choose NVIDIA dim + spacing from patient physical FOV when possible.

    The NVIDIA model's most important geometry knob is FOV = dim * spacing.
    For real MRI volumes, preserve the patient FOV as closely as the released
    rflow-mr-brain dimensional constraints allow.  For tiny synthetic/unit-test
    volumes whose FOV would require invalid spacing, fall back to the historical
    ds004199-friendly default.

    Override with NVIDIA_MAISI_DIM and NVIDIA_MAISI_SPACING env vars.
    """
    # Env overrides (integer dims, float spacing)
    dim_raw = _parse_env_list("NVIDIA_MAISI_DIM", [])
    spc_raw = _parse_env_list("NVIDIA_MAISI_SPACING", [])

    if len(dim_raw) == 3 and len(spc_raw) == 3:
        return [int(d) for d in dim_raw], [float(s) for s in spc_raw]

    if patient_affine is not None:
        shape = np.asarray(patient_shape[:3], dtype=np.float64)
        patient_spacing = _voxel_spacing_from_affine(np.asarray(patient_affine, dtype=np.float64))
        patient_fov = shape * patient_spacing

        # rflow-mr-brain accepts 256^3 and the three common plane-specific
        # layouts with one shorter 128 axis.  Choose 128 for the one axis whose
        # FOV is clearly the slice-stack / shorter anatomical extent.
        dim = np.array([256, 256, 256], dtype=np.int32)
        smallest = int(np.argmin(patient_fov))
        if patient_fov[smallest] < (0.85 * float(np.median(patient_fov))):
            dim[smallest] = 128

        spacing = patient_fov / dim.astype(np.float64)
        if np.all((spacing >= 0.4) & (spacing <= 5.0)):
            return dim.astype(int).tolist(), [float(x) for x in spacing.tolist()]

    # Default heuristic for typical ds004199 1mm FCD FLAIR.
    default_dim = [128, 256, 256]
    default_spc = [1.25, 1.0, 1.0]
    return default_dim, default_spc


def _patient_centered_generated_affine(
    patient_affine: np.ndarray,
    patient_shape: tuple[int, int, int],
    generated_shape: tuple[int, int, int],
    generation_dim: list[int],
    generation_spacing: list[float],
) -> tuple[np.ndarray, list[float]]:
    """Build an affine for an unconditional NVIDIA prior in patient space.

    The raw NVIDIA NIfTI is an unconditional sample and may not carry an affine
    compatible with the patient.  We therefore assign it the requested physical
    FOV, align its center and axes to the patient grid, then resample with the
    same affine-aware code used elsewhere in the pipeline.
    """
    patient_affine = np.asarray(patient_affine, dtype=np.float64)
    patient_spacing = _voxel_spacing_from_affine(patient_affine)
    patient_axes = patient_affine[:3, :3] / patient_spacing.reshape(1, 3)

    requested_fov = (
        np.asarray(generation_dim, dtype=np.float64)
        * np.asarray(generation_spacing, dtype=np.float64)
    )
    generated_shape_arr = np.asarray(generated_shape[:3], dtype=np.float64)
    generated_shape_arr[generated_shape_arr <= 0] = 1.0
    effective_spacing = requested_fov / generated_shape_arr

    patient_center_vox = (np.asarray(patient_shape[:3], dtype=np.float64) - 1.0) / 2.0
    patient_center_world = (
        patient_affine @ np.array([*patient_center_vox.tolist(), 1.0], dtype=np.float64)
    )[:3]

    generated_affine = np.eye(4, dtype=np.float64)
    generated_affine[:3, :3] = patient_axes * effective_spacing.reshape(1, 3)
    generated_center_vox = (generated_shape_arr - 1.0) / 2.0
    generated_affine[:3, 3] = (
        patient_center_world - generated_affine[:3, :3] @ generated_center_vox
    )

    return generated_affine, [float(x) for x in effective_spacing.tolist()]


def _infer_modality_code(modality_hint: Optional[str]) -> int:
    if modality_hint:
        return _MODALITY_CODES.get(modality_hint.lower(), 11)
    return 11  # FLAIR default for this FCD dataset


# ---------------------------------------------------------------------------
# NVIDIA subprocess runner
# ---------------------------------------------------------------------------

def _get_nvidia_git_commit(root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _run_nvidia_generation(
    nvidia_root: Path,
    work_dir: Path,
    dim: list[int],
    spacing: list[float],
    modality: int,
    seed: int,
    num_steps: int,
    cfg_scale: float,
    timeout: int,
) -> Path:
    """Invoke the official diff_model_infer.py and return the path to the generated NIfTI."""
    # Resolve to absolute so paths in config JSONs are valid when cwd=nvidia_root.
    work_dir = work_dir.resolve()
    output_subdir = (work_dir / "output").resolve()
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Patch environment JSON: redirect output_dir and model paths to absolute paths.
    # These are written to absolute paths under work_dir so they remain valid when
    # the subprocess runs with cwd=nvidia_root.
    env_src = nvidia_root / "configs" / "environment_maisi_diff_model_rflow-mr-brain.json"
    with env_src.open() as fh:
        env_cfg = json.load(fh)
    env_cfg["output_dir"] = str(output_subdir.resolve())
    env_cfg["model_dir"] = str((nvidia_root / "models").resolve())
    env_cfg["existing_ckpt_filepath"] = str(
        (nvidia_root / "models" / "diff_unet_3d_rflow-mr-brain_v0.pt").resolve()
    )
    env_cfg["trained_autoencoder_path"] = str(
        (nvidia_root / "models" / "autoencoder_v1.pt").resolve()
    )
    env_cfg["modality_mapping_path"] = str(
        (nvidia_root / "configs" / "modality_mapping.json").resolve()
    )
    patched_env_path = (work_dir / "environment_maisi_diff_model_rflow-mr-brain.json").resolve()
    with patched_env_path.open("w") as fh:
        json.dump(env_cfg, fh, indent=2)

    # Patch diff model config
    diff_src = nvidia_root / "configs" / "config_maisi_diff_model_rflow-mr-brain.json"
    with diff_src.open() as fh:
        diff_cfg = json.load(fh)
    infer = diff_cfg.setdefault("diffusion_unet_inference", {})
    infer["dim"] = dim
    infer["spacing"] = spacing
    infer["modality"] = modality
    infer["random_seed"] = seed
    infer["num_inference_steps"] = num_steps
    infer["cfg_guidance_scale"] = cfg_scale
    patched_diff_path = (work_dir / "config_maisi_diff_model_rflow-mr-brain.json").resolve()
    with patched_diff_path.open("w") as fh:
        json.dump(diff_cfg, fh, indent=2)

    # Copy network config unchanged
    net_src = nvidia_root / "configs" / "config_network_rflow.json"
    patched_net_path = (work_dir / "config_network_rflow.json").resolve()
    shutil.copy2(str(net_src), str(patched_net_path))

    cmd = [
        sys.executable,
        "-m", "scripts.diff_model_infer",
        "--env_config", str(patched_env_path),
        "--model_config", str(patched_diff_path),
        "--model_def", str(patched_net_path),
    ]

    result = subprocess.run(
        cmd,
        cwd=str(nvidia_root),
        capture_output=False,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"NVIDIA diff_model_infer.py failed with exit code {result.returncode}. "
            "Check stderr above for details."
        )

    # Find the generated NIfTI (official script writes unet_3d_*.nii.gz)
    generated = sorted(output_subdir.glob("*.nii.gz"))
    if not generated:
        raise RuntimeError(
            f"NVIDIA inference succeeded (exit 0) but no .nii.gz found in {output_subdir}."
        )
    return generated[-1]  # latest by name sort


# ---------------------------------------------------------------------------
# NVIDIA postprocessing: align prior + ROI hybridization
# ---------------------------------------------------------------------------

def _roi_hybrid_counterfactual(
    patient: np.ndarray,
    prior: np.ndarray,
    roi_mask: np.ndarray,
    feather_sigma: float = 4.0,
    dilation_iterations: int = 5,
) -> np.ndarray:
    """Blend prior into patient inside ROI with Gaussian-feathered boundary.

    Outside support mask: patient strictly unchanged.
    Inside ROI: prior, after local intensity matching to the surrounding shell.
    Boundary: Gaussian-smoothed weight, zeroed outside the dilated support mask.

    Intensity matching uses the shell (dilated_support minus ROI) so the prior
    is matched to surrounding healthy tissue, not the lesion voxels themselves.
    """
    roi_bool = (roi_mask > 0)
    roi_bin = roi_bool.astype(np.float32)

    # Support mask: dilated ROI — blending is confined to this region
    dilated = binary_dilation(roi_bool, iterations=dilation_iterations)
    support = dilated.astype(np.float32)

    # Gaussian-feathered weight, zeroed outside the support
    weight = gaussian_filter(roi_bin, sigma=feather_sigma)
    max_w = weight[dilated].max() if dilated.any() else (weight.max() + 1e-8)
    weight = (weight / (max_w + 1e-8)).astype(np.float32)
    weight = weight * support  # zero outside dilated support — patient untouched there

    # Local intensity matching: match prior to the shell around ROI (healthy tissue).
    # Fall back to ROI voxels if shell is empty (e.g. ROI touches the image boundary).
    shell_bool = dilated & ~roi_bool
    ref_bool = shell_bool if shell_bool.any() else roi_bool
    if ref_bool.any():
        p_mean = patient[ref_bool].mean()
        p_std = patient[ref_bool].std() + 1e-8
        pr_mean = prior[ref_bool].mean()
        pr_std = prior[ref_bool].std() + 1e-8
        prior_matched = (prior - pr_mean) / pr_std * p_std + p_mean
        prior_matched = prior_matched.clip(0.0, 1.0).astype(np.float32)
    else:
        prior_matched = prior

    return (weight * prior_matched + (1.0 - weight) * patient).astype(np.float32)


def _generate_control_prior(
    patient_volume: Path,
    roi_mask: Optional[Path],
    output_path: Path,
    control_prior_volumes: Optional[list[Path]],
    subject_id: Optional[str],
) -> dict:
    """Build a patient-space healthy prior from healthy control volumes."""
    patient_data, patient_affine, patient_header = _load_canonical(patient_volume)
    patient_shape = tuple(patient_data.shape[:3])
    control_paths = _valid_control_paths(control_prior_volumes)

    has_roi = roi_mask is not None and Path(roi_mask).is_file()
    if not has_roi:
        if control_paths:
            return _whole_brain_control_prior_response(
                patient_data=patient_data,
                patient_affine=patient_affine,
                patient_header=patient_header,
                output_path=output_path,
                control_paths=control_paths,
                subject_id=subject_id,
                skip_reason=(
                    "No ROI mask - using whole-brain healthy-control prior instead "
                    "of patient-copy reference."
                ),
                roi_voxels=0,
            )
        return _patient_copy_response(
            patient_data,
            patient_affine,
            patient_header,
            output_path,
            source="control_prior_no_roi_patient_copy",
            skip_reason="No ROI mask and no healthy controls - patient volume used as normal reference.",
            roi_voxels=0,
            notes=[
                "Control-prior mode: no ROI mask and no healthy controls were available, "
                "so the patient volume was saved as the normal reference.",
                "Residual expected near-zero because this is a fallback copy, not a healthy prior.",
            ],
            limitations=[
                "No control-prior reference was computed because no lesion ROI or healthy controls were available.",
                "Not validated for clinical diagnosis.",
            ],
        )

    if not control_paths:
        raise ValueError(
            "control-prior synthetic mode requires at least one healthy-control "
            "volume path in control_prior_volumes."
        )

    roi_arr, roi_affine, _ = _load_canonical(Path(roi_mask))
    roi_resampled = _resample_to_reference(
        roi_arr.astype(np.float32),
        roi_affine,
        patient_shape,
        patient_affine,
        order=0,
    )
    roi_bin = (roi_resampled > 0.5).astype(np.float32)
    roi_bool = roi_bin > 0
    roi_voxels = int(roi_bool.sum())

    if roi_voxels == 0:
        return _whole_brain_control_prior_response(
            patient_data=patient_data,
            patient_affine=patient_affine,
            patient_header=patient_header,
            output_path=output_path,
            control_paths=control_paths,
            subject_id=subject_id,
            skip_reason=(
                "ROI mask exists but contains no positive voxels - using whole-brain "
                "healthy-control prior instead of patient-copy reference."
            ),
            roi_voxels=0,
        )

    patient_norm = _robust_normalize(patient_data)
    brain_mask = np.isfinite(patient_norm) & (patient_norm > 1e-5)
    non_roi_reference = brain_mask & ~binary_dilation(roi_bool, iterations=3)
    if int(non_roi_reference.sum()) < 32:
        non_roi_reference = brain_mask & ~roi_bool
    if int(non_roi_reference.sum()) < 32:
        non_roi_reference = np.isfinite(patient_norm) & ~roi_bool

    prior, used_paths, skipped, quality = _build_control_prior_volume(
        patient_norm=patient_norm,
        patient_shape=patient_shape,
        patient_affine=patient_affine,
        control_paths=control_paths,
        reference_mask=non_roi_reference,
    )
    hybrid = _roi_hybrid_counterfactual(patient_norm, prior, roi_bin)

    _save_float_nifti(hybrid, patient_affine, patient_header, output_path)

    control_subjects = _control_prior_subjects(used_paths)
    source = "control_prior_roi_hybrid"
    return {
        "synthetic_normal_source": source,
        "synthetic_normal_is_diagnostic": False,
        "mode": "control-prior",
        "subject_id": subject_id,
        "control_prior_count": len(used_paths),
        "control_prior_subjects": control_subjects,
        "control_prior_volumes": [str(path) for path in used_paths],
        "control_prior_aggregation": "affine_resampled_intensity_matched_mean",
        "control_prior_alignment": "canonical RAS affine-resampling to patient grid",
        "control_prior_skipped_count": len(skipped),
        "control_prior_skipped": skipped[:10],
        **quality,
        "roi_voxels": roi_voxels,
        "notes": [
            "CONTROL-PRIOR synthetic normal: healthy-control volumes were canonicalized, "
            "affine-resampled into patient space, intensity-matched outside the lesion ROI, "
            "and averaged into a patient-grid healthy prior.",
            "Hybrid counterfactual: patient outside ROI, control-prior inside ROI, "
            "Gaussian-feathered boundary.",
            "This is the non-NVIDIA fallback path for paper experiments when the public "
            "NVIDIA model cannot perform patient-conditioned ROI inpainting.",
        ],
        "limitations": [
            "Population control prior, not true patient-conditioned diffusion inpainting.",
            "Current alignment is affine/grid resampling only; deformable registration is not implemented.",
            "The prior quality depends on modality, voxel geometry, age distribution, and intensity normalization.",
            "Not validated for clinical diagnosis.",
            "Residual concentration near ROI is expected but is not a validated clinical finding.",
        ],
    }


# ---------------------------------------------------------------------------
# Contralateral mode
# ---------------------------------------------------------------------------

def _estimate_x_midline(patient_data: np.ndarray, foreground_mask: np.ndarray) -> float:
    """Estimate the brain x-midline as the intensity-weighted centroid along axis 0.

    Uses the foreground voxels to compute a weighted centroid rather than using
    the raw array centre, which can be biased by asymmetric zero-padding.

    Parameters
    ----------
    patient_data:
        Normalized float32 patient volume, shape (X, Y, Z).
    foreground_mask:
        Boolean mask of brain foreground voxels, same shape.

    Returns
    -------
    float
        Voxel index along axis 0 that best approximates the interhemispheric plane.
    """
    if int(foreground_mask.sum()) < 32:
        return float(patient_data.shape[0] - 1) / 2.0

    x_indices = np.arange(patient_data.shape[0], dtype=np.float64)
    # Per-slice foreground count to find the peak extent on each side
    slice_fg_counts = foreground_mask.sum(axis=(1, 2)).astype(np.float64)
    total = float(slice_fg_counts.sum())
    if total < 1.0:
        return float(patient_data.shape[0] - 1) / 2.0
    midline = float(np.dot(x_indices, slice_fg_counts) / total)
    return midline


def _mirror_x_voxels(
    coords_i: np.ndarray,
    midline_x: float,
) -> np.ndarray:
    """Mirror voxel x-coordinates across the estimated midline.

    Parameters
    ----------
    coords_i:
        1-D array of integer voxel x-coordinates (axis 0).
    midline_x:
        Estimated midline x coordinate (float voxels).

    Returns
    -------
    np.ndarray
        Float array of mirrored x coordinates (may be fractional for interpolation).
    """
    return 2.0 * midline_x - coords_i.astype(np.float64)


def _generate_contralateral(
    patient_volume: Path,
    roi_mask: Optional[Path],
    output_path: Path,
    subject_id: Optional[str],
    feather_sigma: float = 4.0,
    dilation_iterations: int = 5,
) -> dict:
    """Build a patient-specific healthy counterfactual using the contralateral hemisphere.

    For ROI-positive FCD scans:
      1. Load patient + ROI in canonical RAS orientation.
      2. Estimate brain x-midline from foreground (not raw array centre).
      3. For each ROI voxel, look up its mirrored x-coordinate on the
         contralateral side using affine-aware voxel coordinates.
      4. Reject mirrored sources that are out-of-bounds, outside foreground,
         or inside the ROI itself — fall back to a Gaussian-smoothed inpaint
         value for those voxels.
      5. Feather the replacement around an ROI dilation/smoothing boundary so
         the synthetic normal is continuous.
      6. Save in the original patient affine/header convention.

    For control subjects (no ROI or empty ROI), return a patient-copy with a
    clear skip label — does not crash controls.

    Parameters
    ----------
    patient_volume : Path
        Canonical normalized patient NIfTI.
    roi_mask : Optional[Path]
        Binary lesion ROI NIfTI (same space as patient_volume).
    output_path : Path
        Destination NIfTI for the synthetic normal.
    subject_id : Optional[str]
        Subject label for provenance metadata.
    feather_sigma : float
        Gaussian sigma (voxels) used for ROI boundary feathering.
    dilation_iterations : int
        Binary dilation iterations defining the blending support shell.

    Returns
    -------
    dict with provenance, statistics, and paper-safe limitations.
    """
    patient_data, patient_affine, patient_header = _load_canonical(patient_volume)
    patient_shape = tuple(int(s) for s in patient_data.shape[:3])
    patient_norm = _robust_normalize(patient_data)

    # Foreground mask: non-zero finite brain voxels
    foreground = (np.isfinite(patient_norm) & (patient_norm > 1e-5)).astype(bool)
    if int(foreground.sum()) < 32:
        foreground = np.isfinite(patient_norm)

    # -----------------------------------------------------------------------
    # No-ROI path: patient copy with explicit skip label
    # -----------------------------------------------------------------------
    has_roi = roi_mask is not None and Path(roi_mask).is_file()
    if not has_roi:
        _save_float_nifti(patient_norm, patient_affine, patient_header, output_path)
        return {
            "synthetic_normal_source": "contralateral_skipped_no_roi_patient_copy",
            "synthetic_normal_is_diagnostic": False,
            "mode": "contralateral",
            "contralateral_skip_reason": (
                "No ROI mask provided — contralateral substitution requires a "
                "non-empty lesion ROI. Patient volume saved as normal reference. "
                "Residual expected near-zero (control subject or no ROI annotation)."
            ),
            "roi_voxel_count": 0,
            "mirrored_voxel_fraction": 0.0,
            "fallback_voxel_fraction": 0.0,
            "feather_sigma": feather_sigma,
            "dilation_iterations": dilation_iterations,
            "midline_voxel_index": None,
            "midline_world_mm": None,
            "notes": [
                "CONTRALATERAL mode: no ROI — patient volume used as normal reference.",
                "Residual expected near-zero (control subject).",
                "Contralateral substitution was skipped because no ROI was available.",
            ],
            "limitations": [
                "Contralateral mode requires a non-empty lesion ROI; skipped for this subject.",
                "Not validated for clinical diagnosis.",
            ],
        }

    # -----------------------------------------------------------------------
    # Load and resample ROI mask to patient grid
    # -----------------------------------------------------------------------
    roi_arr, roi_affine, _ = _load_canonical(Path(roi_mask))
    roi_resampled = _resample_to_reference(
        roi_arr.astype(np.float32),
        roi_affine,
        patient_shape,
        patient_affine,
        order=0,
    )
    roi_bool = (roi_resampled > 0.5).astype(bool)
    roi_voxel_count = int(roi_bool.sum())

    # -----------------------------------------------------------------------
    # Empty-ROI path: patient copy with explicit skip label
    # -----------------------------------------------------------------------
    if roi_voxel_count == 0:
        _save_float_nifti(patient_norm, patient_affine, patient_header, output_path)
        return {
            "synthetic_normal_source": "contralateral_skipped_empty_roi_patient_copy",
            "synthetic_normal_is_diagnostic": False,
            "mode": "contralateral",
            "contralateral_skip_reason": (
                "ROI mask exists but contains no positive voxels after resampling to "
                "the patient grid. Patient volume saved as normal reference. "
                "Residual expected near-zero."
            ),
            "roi_voxel_count": 0,
            "mirrored_voxel_fraction": 0.0,
            "fallback_voxel_fraction": 0.0,
            "feather_sigma": feather_sigma,
            "dilation_iterations": dilation_iterations,
            "midline_voxel_index": None,
            "midline_world_mm": None,
            "notes": [
                "CONTRALATERAL mode: ROI mask empty — patient volume used as normal reference.",
                "Contralateral substitution was skipped because the ROI contains no voxels.",
            ],
            "limitations": [
                "Contralateral mode requires a non-empty lesion ROI; skipped for this subject.",
                "Not validated for clinical diagnosis.",
            ],
        }

    # -----------------------------------------------------------------------
    # Estimate x-midline from brain foreground
    # -----------------------------------------------------------------------
    midline_x = _estimate_x_midline(patient_norm, foreground)

    # Convert midline voxel index to world mm via affine
    # patient_affine maps voxel coords (i,j,k,1)^T -> world (x,y,z,1)^T
    midline_world_arr = patient_affine @ np.array(
        [midline_x, patient_shape[1] / 2.0, patient_shape[2] / 2.0, 1.0]
    )
    midline_world_mm = float(midline_world_arr[0])

    # -----------------------------------------------------------------------
    # Build a Gaussian-smoothed fallback image for rejected source voxels
    # -----------------------------------------------------------------------
    smooth_sigma = max(feather_sigma, 3.0)
    smoothed_fallback = gaussian_filter(patient_norm, sigma=smooth_sigma).astype(np.float32)

    # -----------------------------------------------------------------------
    # Contralateral sampling for ROI voxels
    # -----------------------------------------------------------------------
    # Find all ROI voxel indices
    roi_vox_i, roi_vox_j, roi_vox_k = np.where(roi_bool)

    # Mirror x-coordinates across midline (float for map_coordinates)
    mirror_i = _mirror_x_voxels(roi_vox_i, midline_x)

    # For affine-aware sampling we compose the reflected voxel coords back
    # through the patient affine so that voxel-size asymmetry is handled.
    # Because we're staying in the same patient grid (same affine), the mirror
    # reflection in voxel space is exact — no extra world-space roundtrip needed.
    # We sample the patient volume at fractional mirrored positions (order-1
    # tri-linear interpolation).
    contralateral_values = map_coordinates(
        patient_norm,
        [mirror_i, roi_vox_j.astype(np.float64), roi_vox_k.astype(np.float64)],
        order=1,
        mode="constant",
        cval=np.nan,
    ).astype(np.float32)

    # -----------------------------------------------------------------------
    # Validity checks: reject mirrored sources that are
    #   (a) out of bounds (NaN from map_coordinates cval=nan),
    #   (b) outside brain foreground (sampling in zero-padded regions),
    #   (c) inside the ROI itself (would recycle the lesion signal).
    # -----------------------------------------------------------------------
    # Build a bool mask indicating which mirrored source positions are "in ROI"
    mirror_i_clipped = np.clip(np.round(mirror_i).astype(np.int32), 0, patient_shape[0] - 1)
    mirror_j_clipped = roi_vox_j.copy()  # j,k unchanged
    mirror_k_clipped = roi_vox_k.copy()

    # Out-of-bounds: mirror_i < 0 or >= shape[0]
    oob = (mirror_i < 0) | (mirror_i >= patient_shape[0])

    # Outside foreground at the mirrored position
    in_foreground_mirror = foreground[mirror_i_clipped, mirror_j_clipped, mirror_k_clipped]
    not_in_foreground = ~in_foreground_mirror

    # Inside ROI at the mirrored position (recycling lesion signal)
    in_roi_mirror = roi_bool[mirror_i_clipped, mirror_j_clipped, mirror_k_clipped]

    # NaN from out-of-bounds interpolation
    is_nan = ~np.isfinite(contralateral_values)

    bad_source = oob | not_in_foreground | in_roi_mirror | is_nan

    # For good sources, use contralateral values; for bad sources use fallback
    fallback_values = smoothed_fallback[roi_vox_i, roi_vox_j, roi_vox_k]
    replacement_values = np.where(bad_source, fallback_values, contralateral_values)

    mirrored_count = int((~bad_source).sum())
    fallback_count = int(bad_source.sum())
    mirrored_fraction = float(mirrored_count) / max(1, roi_voxel_count)
    fallback_fraction = float(fallback_count) / max(1, roi_voxel_count)

    # -----------------------------------------------------------------------
    # Build the replacement volume: copy patient, inject replacement into ROI
    # -----------------------------------------------------------------------
    replacement_vol = patient_norm.copy()
    replacement_vol[roi_vox_i, roi_vox_j, roi_vox_k] = replacement_values

    # -----------------------------------------------------------------------
    # Feather the replacement around the ROI dilation boundary
    # so the synthetic normal is continuous at the edges.
    # Strategy mirrors _roi_hybrid_counterfactual from the existing code.
    # -----------------------------------------------------------------------
    roi_bin = roi_bool.astype(np.float32)
    dilated = binary_dilation(roi_bool, iterations=dilation_iterations)
    support = dilated.astype(np.float32)

    weight = gaussian_filter(roi_bin, sigma=feather_sigma)
    max_w = weight[dilated].max() if dilated.any() else (weight.max() + 1e-8)
    weight = (weight / (max_w + 1e-8)).astype(np.float32)
    weight = weight * support  # zero outside dilated support -> patient untouched there

    synthetic = (weight * replacement_vol + (1.0 - weight) * patient_norm).astype(np.float32)

    # -----------------------------------------------------------------------
    # Save output
    # -----------------------------------------------------------------------
    _save_float_nifti(synthetic, patient_affine, patient_header, output_path)

    return {
        "synthetic_normal_source": "contralateral_patient_specific_roi_completion",
        "synthetic_normal_is_diagnostic": False,
        "mode": "contralateral",
        "subject_id": subject_id,
        "roi_voxel_count": roi_voxel_count,
        "mirrored_voxel_fraction": round(mirrored_fraction, 4),
        "fallback_voxel_fraction": round(fallback_fraction, 4),
        "feather_sigma": feather_sigma,
        "dilation_iterations": dilation_iterations,
        "midline_voxel_index": round(midline_x, 3),
        "midline_world_mm": round(midline_world_mm, 3),
        "midline_estimation": "intensity_weighted_foreground_centroid_axis0",
        "contralateral_axis": "voxel axis 0 (canonical RAS x / left-right after nib.as_closest_canonical)",
        "mirrored_voxel_count": mirrored_count,
        "fallback_voxel_count": fallback_count,
        "fallback_method": "gaussian_smoothed_inpaint",
        "fallback_smooth_sigma": smooth_sigma,
        "notes": [
            "CONTRALATERAL synthetic normal: ROI voxels filled by mirroring the "
            "contralateral homologous region across the estimated interhemispheric plane.",
            "Patient anatomy outside the ROI is preserved exactly.",
            "Mirrored source voxels that are out-of-bounds, outside brain foreground, "
            "or inside the ROI itself are replaced by a local Gaussian-smoothed inpaint "
            f"(fallback fraction: {fallback_fraction:.1%}).",
            "Boundary is feathered using Gaussian-weighted blending over a dilated ROI "
            "support mask to maintain image continuity.",
            "NOT diagnostic. Patient-specific but not a trained generative model.",
        ],
        "limitations": [
            "Contralateral substitution assumes the brain is approximately symmetric; "
            "strong anatomical asymmetry (e.g. asymmetric sulcal patterns, large lesions "
            "with mass effect) may reduce the quality of the counterfactual.",
            "Midline estimation uses a foreground-weighted voxel centroid along axis 0 "
            "(canonical RAS L-R); slight tilt or roll in the scan can shift the estimate.",
            "Mirrored voxels that fall outside foreground or inside the ROI are replaced by "
            "Gaussian smoothing, which may underestimate lesion-size residuals.",
            "Feathering is a post-hoc geometric blend, not a learned inpainting model.",
            "Not validated for clinical diagnosis.",
            "Residual concentration near ROI is expected but is not a validated clinical finding.",
        ],
    }


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def generate_synthetic_normal(
    patient_volume: Path,
    roi_mask: Optional[Path],
    output_path: Path,
    mode: SyntheticMode = "placeholder",
    modality_hint: Optional[str] = None,
    subject_id: Optional[str] = None,
    rng_seed: int = 42,
    control_prior_volumes: Optional[list[Path]] = None,
) -> dict:
    """Generate a synthetic normal brain volume.

    Parameters
    ----------
    patient_volume:
        Normalized patient FLAIR or T1w NIfTI (same space as roi_mask).
    roi_mask:
        Binary ROI mask.  Required for ROI-hybridized synthetic normal.
    output_path:
        Destination NIfTI for the synthetic normal.
    mode:
        "placeholder", "nvidia", or "control-prior".
    modality_hint:
        "FLAIR", "T1w", "T2w", etc.  Used to pick NVIDIA modality code.
    subject_id:
        Subject label used in logs and metadata.
    rng_seed:
        RNG seed forwarded to NVIDIA generation.
    control_prior_volumes:
        Healthy-control NIfTI volumes used by "control-prior" mode.

    Returns
    -------
    dict with source, mode, notes, limitations, and NVIDIA metadata when applicable.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "nvidia":
        return _generate_nvidia(
            patient_volume=Path(patient_volume),
            roi_mask=roi_mask,
            output_path=output_path,
            modality_hint=modality_hint,
            subject_id=subject_id,
            rng_seed=rng_seed,
        )
    if mode == "control-prior":
        return _generate_control_prior(
            patient_volume=Path(patient_volume),
            roi_mask=roi_mask,
            output_path=output_path,
            control_prior_volumes=control_prior_volumes,
            subject_id=subject_id,
        )
    if mode == "contralateral":
        return _generate_contralateral(
            patient_volume=Path(patient_volume),
            roi_mask=roi_mask,
            output_path=output_path,
            subject_id=subject_id,
        )

    # --- placeholder mode ----------------------------------------------------
    data, affine, header = _load_canonical(Path(patient_volume))
    roi = None
    roi_voxels = 0
    if roi_mask is not None and Path(roi_mask).is_file():
        roi_arr, _, _ = _load_canonical(Path(roi_mask))
        roi = roi_arr
        roi_voxels = int((roi_arr > 0).sum())

    synth = _inpaint_roi_placeholder(data, roi if roi is not None else np.zeros_like(data))

    img_out = nib.Nifti1Image(synth, affine, header)
    img_out.header.set_data_dtype(np.float32)
    nib.save(img_out, str(output_path))

    notes = [
        "PLACEHOLDER synthetic normal: ROI filled by Gaussian-smoothed surrounding tissue.",
        "NOT diagnostic. NOT NVIDIA NV-Generate-MR-Brain output.",
        "For NVIDIA output: set NVIDIA_GENERATE_CTMR_ROOT and use --synthetic-mode nvidia.",
    ]
    return {
        "synthetic_normal_source": "placeholder",
        "synthetic_normal_is_diagnostic": False,
        "mode": mode,
        "roi_voxels_inpainted": roi_voxels,
        "notes": notes,
        "limitations": [
            "Placeholder uses Gaussian smoothing, not a trained generative model.",
            "Residual will concentrate near ROI but is not a validated clinical finding.",
            "Controls (no ROI) receive whole-brain smoothing; residual is expected near-zero.",
        ],
    }


def _generate_nvidia(
    patient_volume: Path,
    roi_mask: Optional[Path],
    output_path: Path,
    modality_hint: Optional[str],
    subject_id: Optional[str],
    rng_seed: int,
) -> dict:
    """Real NV-Generate-MR-Brain rflow-mr-brain integration."""
    nvidia_root = _check_nvidia_available()

    patient_data, patient_affine, patient_header = _load_canonical(patient_volume)
    patient_shape = patient_data.shape[:3]

    has_roi = roi_mask is not None and Path(roi_mask).is_file()

    # If no ROI mask (healthy control), skip generation and return patient copy.
    if not has_roi:
        img_out = nib.Nifti1Image(patient_data, patient_affine, patient_header)
        img_out.header.set_data_dtype(np.float32)
        nib.save(img_out, str(output_path))
        return {
            "synthetic_normal_source": "nvidia_no_roi_patient_copy",
            "synthetic_normal_is_diagnostic": False,
            "mode": "nvidia",
            "nvidia_skip_reason": "No ROI mask — healthy control, patient volume used as reference.",
            "roi_voxels": 0,
            "notes": [
                "NVIDIA mode: no ROI mask present — patient volume saved as normal reference.",
                "Residual expected near-zero (control subject).",
            ],
            "limitations": [
                "No generative prior computed for control subjects.",
                "Not a validated clinical finding.",
            ],
        }

    # Resolve generation parameters
    modality = _infer_modality_code(modality_hint)
    dim, spacing = _choose_generation_grid(patient_shape, patient_affine)
    num_steps = int(os.environ.get("NVIDIA_MAISI_STEPS", "30"))
    cfg_scale = float(os.environ.get("NVIDIA_MAISI_CFG_SCALE", "10"))
    timeout = int(os.environ.get("NVIDIA_MAISI_TIMEOUT_SEC", "1800"))

    work_dir = output_path.parent / "nvidia_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build subprocess command summary for metadata
    cmd_summary = (
        f"python -m scripts.diff_model_infer (cwd={nvidia_root}, "
        f"dim={dim}, spacing={spacing}, modality={modality}, seed={rng_seed}, "
        f"steps={num_steps}, cfg_scale={cfg_scale})"
    )

    generated_nifti = _run_nvidia_generation(
        nvidia_root=nvidia_root,
        work_dir=work_dir,
        dim=dim,
        spacing=spacing,
        modality=modality,
        seed=rng_seed,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        timeout=timeout,
    )

    # Save raw prior artifact
    raw_prior_path = output_path.parent / "nvidia_raw_prior.nii.gz"
    shutil.copy2(str(generated_nifti), str(raw_prior_path))

    # Load and align generated prior to patient space
    gen_data, _, _ = _load_canonical(generated_nifti)
    gen_norm = _robust_normalize(gen_data)
    generated_shape = tuple(int(x) for x in gen_norm.shape[:3])
    gen_affine, gen_effective_spacing = _patient_centered_generated_affine(
        patient_affine=patient_affine,
        patient_shape=patient_shape,
        generated_shape=generated_shape,
        generation_dim=dim,
        generation_spacing=spacing,
    )
    gen_aligned = _resample_to_reference(
        gen_norm,
        gen_affine,
        patient_shape,
        patient_affine,
        order=1,
    )

    # Load ROI mask
    roi_arr, _, _ = _load_canonical(Path(roi_mask))
    roi_bin = (roi_arr > 0).astype(np.float32)
    roi_voxels = int(roi_bin.sum())

    # Patient normalized [0,1]
    patient_norm = _robust_normalize(patient_data)

    # ROI-hybrid counterfactual
    hybrid = _roi_hybrid_counterfactual(patient_norm, gen_aligned, roi_bin)

    img_out = nib.Nifti1Image(hybrid, patient_affine, patient_header)
    img_out.header.set_data_dtype(np.float32)
    nib.save(img_out, str(output_path))

    upstream_commit = _get_nvidia_git_commit(nvidia_root)

    return {
        "synthetic_normal_source": "nvidia_rflow-mr-brain_roi_hybrid",
        "synthetic_normal_is_diagnostic": False,
        "mode": "nvidia",
        "nvidia_raw_prior_artifact": raw_prior_path.name,
        "nvidia_modality_code": modality,
        "nvidia_modality_hint": modality_hint,
        "nvidia_generation_dim": dim,
        "nvidia_generation_spacing": spacing,
        "nvidia_generation_fov_mm": [
            float(d) * float(s) for d, s in zip(dim, spacing)
        ],
        "nvidia_patient_fov_mm": [
            float(s) * float(sp)
            for s, sp in zip(patient_shape, _voxel_spacing_from_affine(patient_affine))
        ],
        "nvidia_generated_shape": list(generated_shape),
        "nvidia_generated_shape_matches_config": list(generated_shape) == list(dim),
        "nvidia_generated_effective_spacing": gen_effective_spacing,
        "nvidia_prior_alignment": "patient_centered_fov_affine_resample",
        "nvidia_seed": rng_seed,
        "nvidia_steps": num_steps,
        "nvidia_cfg_scale": cfg_scale,
        "nvidia_upstream_commit": upstream_commit,
        "nvidia_root": str(nvidia_root),
        "nvidia_command_summary": cmd_summary,
        "roi_voxels": roi_voxels,
        "notes": [
            "NVIDIA rflow-mr-brain generation succeeded.",
            f"Raw prior: {raw_prior_path.name} (identity-affine, {dim} voxels).",
            "Prior assigned the requested physical FOV, center-aligned to the patient, "
            "and affine-resampled into the patient grid.",
            "Hybrid counterfactual: patient outside ROI, intensity-matched prior inside ROI, "
            "Gaussian-feathered boundary.",
        ],
        "limitations": [
            "Image-only NVIDIA generation plus ROI hybridization — not true patient-conditioned "
            "diffusion inpainting.",
            "Generated prior is unconditional (not conditioned on this patient's anatomy).",
            "ROI boundary blending is a post-hoc fusion, not a learned inpainting model.",
            "Not validated for clinical diagnosis.",
            "Residual concentration near ROI is expected but is not a validated clinical finding.",
        ],
    }
