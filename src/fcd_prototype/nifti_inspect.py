"""NIfTI volume inspection utilities."""

import numpy as np
import nibabel as nib


def inspect_nifti(path: str) -> dict:
    """Load a NIfTI file and return a dict of key statistics."""
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float64)
    header = img.header
    zooms = header.get_zooms()

    return {
        "shape": data.shape,
        "voxel_spacing_mm": tuple(float(z) for z in zooms),
        "affine": img.affine,
        "dtype": str(header.get_data_dtype()),
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "mean": float(np.nanmean(data)),
        "nan_count": int(np.sum(np.isnan(data))),
        "inf_count": int(np.sum(np.isinf(data))),
    }


def print_inspection(stats: dict) -> None:
    """Pretty-print the output of inspect_nifti."""
    print(f"Shape          : {stats['shape']}")
    print(f"Voxel spacing  : {stats['voxel_spacing_mm']} mm")
    print(f"Dtype          : {stats['dtype']}")
    print(f"Min / Max      : {stats['min']:.4g} / {stats['max']:.4g}")
    print(f"Mean           : {stats['mean']:.4g}")
    print(f"NaN count      : {stats['nan_count']}")
    print(f"Inf count      : {stats['inf_count']}")
    print("Affine matrix  :")
    for row in stats["affine"]:
        print("  ", "  ".join(f"{v:10.4f}" for v in row))
