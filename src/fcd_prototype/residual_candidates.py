"""Residual-derived lesion candidate extraction.

The pre-subtraction candidate ROI heuristic is intentionally conservative and
has weak recall on ds004199.  This module works later in the pipeline: after a
patient-vs-healthy residual heatmap exists, it turns the strongest connected
residual regions into explicit candidate masks that can be viewed in NiiVue and
evaluated against manual FCD ROIs when available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, center_of_mass, label, map_coordinates


def _load_canonical(path: Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data.astype(np.float32), img.affine.astype(np.float64), img.header.copy()


def _same_grid(
    shape: tuple[int, int, int],
    affine: np.ndarray,
    ref_shape: tuple[int, int, int],
    ref_affine: np.ndarray,
) -> bool:
    return tuple(shape[:3]) == tuple(ref_shape) and np.allclose(affine, ref_affine, atol=1e-4)


def _voxel_spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    spacing = np.linalg.norm(np.asarray(affine, dtype=np.float64)[:3, :3], axis=0)
    spacing = np.where(np.isfinite(spacing) & (spacing > 1e-6), spacing, 1.0)
    return spacing.astype(np.float64)


def _ball_structure(affine: np.ndarray, radius_mm: float) -> np.ndarray:
    spacing = _voxel_spacing_from_affine(affine)
    extents = np.ceil(float(radius_mm) / spacing).astype(int)
    grids = np.meshgrid(
        *[np.arange(-extent, extent + 1, dtype=np.float64) for extent in extents],
        indexing="ij",
    )
    distance = np.zeros_like(grids[0], dtype=np.float64)
    for axis, grid in enumerate(grids):
        distance += (grid * spacing[axis]) ** 2
    structure = np.sqrt(distance) <= float(radius_mm)
    structure[tuple(extents.tolist())] = True
    return structure


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
    out = np.zeros(ref_shape, dtype=np.float32)
    xs = np.arange(ref_shape[0], dtype=np.float32)
    ys = np.arange(ref_shape[1], dtype=np.float32)
    for z0 in range(0, ref_shape[2], 16):
        z1 = min(ref_shape[2], z0 + 16)
        zs = np.arange(z0, z1, dtype=np.float32)
        i, j, k = np.meshgrid(xs, ys, zs, indexing="ij")
        ref_vox = np.stack([i, j, k, np.ones_like(i)], axis=0).reshape(4, -1)
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


def _voxel_to_world(vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    hom = np.concatenate([vox.astype(np.float64), np.ones((len(vox), 1))], axis=1)
    return (affine @ hom.T).T[:, :3]


def _save_like(data: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(np.float32), affine, header)
    img.header.set_data_dtype(np.float32)
    nib.save(img, str(path))


def _component_bbox(indices: np.ndarray) -> dict[str, list[int]]:
    return {
        "min": indices.min(axis=0).astype(int).tolist(),
        "max": indices.max(axis=0).astype(int).tolist(),
    }


def _manual_overlap_metrics(
    candidate_mask: np.ndarray,
    manual_roi: np.ndarray,
) -> dict[str, float | int | bool]:
    cand = candidate_mask > 0
    roi = manual_roi > 0
    cand_n = int(cand.sum())
    roi_n = int(roi.sum())
    inter = int((cand & roi).sum())
    union = int((cand | roi).sum())
    dice = (2.0 * inter / (cand_n + roi_n)) if (cand_n + roi_n) else 1.0
    coverage = (inter / roi_n) if roi_n else 1.0
    precision = (inter / cand_n) if cand_n else 0.0
    jaccard = (inter / union) if union else 1.0
    return {
        "manual_roi_available": bool(roi_n > 0),
        "manual_roi_voxels": roi_n,
        "candidate_voxels": cand_n,
        "intersection_voxels": inter,
        "manual_coverage": float(coverage),
        "candidate_precision": float(precision),
        "dice": float(dice),
        "jaccard": float(jaccard),
    }


def extract_residual_candidates(
    *,
    residual_heatmap: Path,
    output_dir: Path,
    reference_volume: Optional[Path] = None,
    manual_roi: Optional[Path] = None,
    max_candidates: int = 5,
    percentile: float = 99.0,
    min_threshold: float = 0.25,
    min_component_voxels: int = 8,
    dilate_mm: float = 5.0,
    dilate_iterations: Optional[int] = None,
) -> dict[str, object]:
    """Extract connected residual components and write candidate masks.

    Parameters are intentionally simple: residual heatmaps are normalized to
    [0, 1], so thresholding by both percentile and minimum value keeps behavior
    stable across placeholder, control-prior, and NVIDIA-hybrid modes.
    """
    residual_heatmap = Path(residual_heatmap)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    residual, affine, header = _load_canonical(residual_heatmap)
    residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
    residual = np.clip(residual, 0.0, None).astype(np.float32)
    ref_shape = tuple(int(x) for x in residual.shape[:3])
    reference_foreground_voxels: Optional[int] = None

    if reference_volume is not None and Path(reference_volume).is_file():
        ref_data, ref_affine, ref_header = _load_canonical(Path(reference_volume))
        reference_foreground_voxels = int(
            np.count_nonzero(np.isfinite(ref_data) & (ref_data > 1e-6))
        )
        if not _same_grid(ref_shape, affine, ref_data.shape[:3], ref_affine):
            residual = _resample_to_reference(
                residual,
                affine,
                tuple(int(x) for x in ref_data.shape[:3]),
                ref_affine,
                order=1,
            )
            affine = ref_affine
            header = ref_header
            ref_shape = tuple(int(x) for x in residual.shape[:3])

    positive = residual[residual > 0]
    if positive.size:
        percentile_value = float(np.percentile(positive, percentile))
        threshold = max(float(min_threshold), percentile_value)
    else:
        percentile_value = 0.0
        threshold = float(min_threshold)

    binary = residual >= threshold if positive.size else np.zeros(ref_shape, dtype=bool)
    dilation_mode = "none"
    if binary.any():
        if dilate_iterations is not None:
            if dilate_iterations > 0:
                binary = binary_dilation(binary, iterations=int(dilate_iterations))
                dilation_mode = "voxel_iterations"
        elif dilate_mm > 0:
            binary = binary_dilation(
                binary,
                structure=_ball_structure(affine, float(dilate_mm)),
                iterations=1,
            )
            dilation_mode = "physical_mm"

    labeled, n_labels = label(binary)
    components: list[dict[str, object]] = []
    component_masks: list[np.ndarray] = []

    for label_id in range(1, int(n_labels) + 1):
        comp = labeled == label_id
        voxels = int(comp.sum())
        if voxels < min_component_voxels:
            continue
        values = residual[comp]
        indices = np.argwhere(comp)
        center_vox = np.array(center_of_mass(comp), dtype=np.float64).reshape(1, 3)
        center_world = _voxel_to_world(center_vox, affine)[0]
        peak_index = indices[int(np.argmax(values))]
        peak_world = _voxel_to_world(peak_index.reshape(1, 3), affine)[0]
        components.append({
            "label": int(label_id),
            "voxel_count": voxels,
            "score_sum": float(values.sum()),
            "score_mean": float(values.mean()),
            "score_max": float(values.max()),
            "center_voxel": [float(x) for x in center_vox[0].tolist()],
            "center_world_mm": [float(x) for x in center_world.tolist()],
            "peak_voxel": peak_index.astype(int).tolist(),
            "peak_world_mm": [float(x) for x in peak_world.tolist()],
            "bbox_voxel": _component_bbox(indices),
        })
        component_masks.append(comp)

    order = sorted(
        range(len(components)),
        key=lambda idx: (
            float(components[idx]["score_sum"]),
            float(components[idx]["score_max"]),
            int(components[idx]["voxel_count"]),
        ),
        reverse=True,
    )[:max_candidates]

    selected_components: list[dict[str, object]] = []
    union = np.zeros(ref_shape, dtype=np.float32)
    artifacts: dict[str, str] = {}
    voxel_volume_mm3 = float(np.prod(_voxel_spacing_from_affine(affine)))
    for out_idx, comp_idx in enumerate(order):
        comp_mask = component_masks[comp_idx].astype(np.float32)
        union = np.maximum(union, comp_mask)
        fname = f"residual_candidate_{out_idx:02d}.nii.gz"
        _save_like(comp_mask, affine, header, output_dir / fname)
        artifacts[f"residual_candidate_{out_idx:02d}"] = fname
        comp = dict(components[comp_idx])
        comp["rank"] = out_idx
        comp["artifact"] = fname
        comp["volume_mm3"] = float(int(comp["voxel_count"]) * voxel_volume_mm3)
        comp["volume_ml"] = float(comp["volume_mm3"] / 1000.0)
        selected_components.append(comp)

    union_path = output_dir / "residual_candidate_mask.nii.gz"
    _save_like(union, affine, header, union_path)
    artifacts["residual_candidate_mask"] = union_path.name

    manual_metrics: dict[str, object] = {
        "manual_roi_available": False,
    }
    if manual_roi is not None and Path(manual_roi).is_file():
        roi_data, roi_affine, _ = _load_canonical(Path(manual_roi))
        roi_resampled = _resample_to_reference(
            (roi_data > 0).astype(np.float32),
            roi_affine,
            ref_shape,
            affine,
            order=0,
        )
        manual_metrics = _manual_overlap_metrics(union, roi_resampled)
        manual_metrics["manual_roi_resampled_to_residual_grid"] = not _same_grid(
            roi_data.shape[:3], roi_affine, ref_shape, affine
        )

    manifest = {
        "method": "residual_connected_component_candidates",
        "residual_heatmap": str(residual_heatmap),
        "reference_volume": str(reference_volume) if reference_volume else None,
        "manual_roi": str(manual_roi) if manual_roi else None,
        "max_candidates": int(max_candidates),
        "percentile": float(percentile),
        "percentile_value": percentile_value,
        "min_threshold": float(min_threshold),
        "threshold_value": threshold,
        "min_component_voxels": int(min_component_voxels),
        "dilate_mm": float(dilate_mm),
        "dilate_iterations": int(dilate_iterations) if dilate_iterations is not None else None,
        "dilation_mode": dilation_mode,
        "component_count_raw": int(n_labels),
        "component_count_after_size_filter": int(len(components)),
        "selected_count": int(len(selected_components)),
        "selected_components": selected_components,
        "candidate_voxels": int(union.sum()),
        "candidate_volume_mm3": float(union.sum() * voxel_volume_mm3),
        "candidate_volume_ml": float(union.sum() * voxel_volume_mm3 / 1000.0),
        "reference_foreground_voxels": reference_foreground_voxels,
        "candidate_foreground_fraction": (
            float(union.sum() / reference_foreground_voxels)
            if reference_foreground_voxels else None
        ),
        "manual_overlap": manual_metrics,
        "artifacts": artifacts,
        "limitations": [
            "Residual candidates are generated after healthy-reference subtraction; they are not a clinical diagnosis.",
            "Candidate quality depends on the healthy reference, registration, modality normalization, and threshold settings.",
            "For no-ROI uploads, this is the first-pass region proposal for visual review and cohort validation.",
        ],
    }

    manifest_path = output_dir / "residual_candidate_manifest.json"
    artifacts["residual_candidate_manifest"] = manifest_path.name
    manifest["artifacts"] = artifacts
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract lesion candidates from a residual heatmap.")
    p.add_argument("--residual-heatmap", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--reference-volume", type=Path, default=None)
    p.add_argument("--manual-roi", type=Path, default=None)
    p.add_argument("--max-candidates", type=int, default=5)
    p.add_argument("--percentile", type=float, default=99.0)
    p.add_argument("--min-threshold", type=float, default=0.25)
    p.add_argument("--min-component-voxels", type=int, default=8)
    p.add_argument("--dilate-mm", type=float, default=5.0,
                   help="Physically grow candidate masks by this radius in millimeters.")
    p.add_argument("--dilate-iterations", type=int, default=None,
                   help="Optional legacy voxel-iteration dilation; overrides --dilate-mm when set.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = extract_residual_candidates(
        residual_heatmap=args.residual_heatmap,
        output_dir=args.output_dir,
        reference_volume=args.reference_volume,
        manual_roi=args.manual_roi,
        max_candidates=args.max_candidates,
        percentile=args.percentile,
        min_threshold=args.min_threshold,
        min_component_voxels=args.min_component_voxels,
        dilate_mm=args.dilate_mm,
        dilate_iterations=args.dilate_iterations,
    )
    print(
        f"Residual candidates: selected={result['selected_count']} "
        f"threshold={result['threshold_value']:.4f} -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
