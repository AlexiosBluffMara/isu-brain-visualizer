"""Generate a synthetic NIfTI brain volume with a known artificial lesion.

Output: outputs/synthetic/synthetic_lesion.nii.gz

The lesion is a small bright ellipsoid embedded in a smoothly varying
background.  It is fully artificial — no real patient data is used.
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def make_synthetic_lesion(
    shape: tuple[int, int, int] = (64, 64, 48),
    lesion_center: tuple[int, int, int] | None = None,
    lesion_radii: tuple[int, int, int] = (4, 4, 3),
    lesion_intensity: float = 0.7,
    rng_seed: int = 42,
) -> nib.Nifti1Image:
    """Return a synthetic NIfTI1 image with one embedded bright ellipsoid."""
    rng = np.random.default_rng(rng_seed)

    # Background: smooth random field simulating grey-matter MRI intensity.
    noise = rng.standard_normal(shape).astype(np.float32)
    background = gaussian_filter(noise, sigma=6.0) * 300 + 600
    background = np.clip(background, 0, None).astype(np.float32)

    if lesion_center is None:
        lesion_center = (shape[0] // 2 + 5, shape[1] // 2 - 3, shape[2] // 2 + 2)

    cx, cy, cz = lesion_center
    rx, ry, rz = lesion_radii
    x, y, z = np.ogrid[: shape[0], : shape[1], : shape[2]]
    ellipsoid_mask = (
        ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2
    ) <= 1.0

    # Lesion sits ~40% above the local background, normalised to [0,1] range.
    local_bg_max = float(background[ellipsoid_mask].max()) if ellipsoid_mask.any() else 600.0
    lesion_val = local_bg_max * (1 + lesion_intensity)
    background[ellipsoid_mask] = lesion_val

    affine = np.diag([1.0, 1.0, 1.2, 1.0])  # 1 mm × 1 mm × 1.2 mm voxels
    img = nib.Nifti1Image(background, affine)
    img.header.set_zooms((1.0, 1.0, 1.2))
    img.header.set_data_dtype(np.float32)
    return img


def main() -> None:
    out_dir = Path("outputs/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_lesion.nii.gz"

    img = make_synthetic_lesion()
    nib.save(img, str(out_path))
    print(f"Saved synthetic lesion NIfTI: {out_path}")
    print(f"  Shape : {img.shape}")
    print(f"  Affine:\n{img.affine}")


if __name__ == "__main__":
    main()
