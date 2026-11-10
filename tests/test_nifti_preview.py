"""Tests for nifti_preview using synthetic NIfTI data."""

import numpy as np
import nibabel as nib
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fcd_prototype.nifti_preview import save_previews, _robust_normalize


@pytest.fixture
def synthetic_volume(tmp_path):
    """Random 20x30x25 volume with realistic-ish intensity range."""
    rng = np.random.default_rng(42)
    data = (rng.standard_normal((20, 30, 25)) * 500 + 1000).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    path = tmp_path / "vol.nii"
    nib.save(img, str(path))
    return str(path)


@pytest.fixture
def constant_volume(tmp_path):
    """All-zero volume — exercises the flat-signal edge case."""
    data = np.zeros((10, 10, 10), dtype=np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    path = tmp_path / "zeros.nii"
    nib.save(img, str(path))
    return str(path)


@pytest.fixture
def volume_4d(tmp_path):
    """4-D volume — should be reduced to 3D by taking first time point."""
    data = np.ones((8, 8, 8, 3), dtype=np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    path = tmp_path / "vol4d.nii"
    nib.save(img, str(path))
    return str(path)


# --- _robust_normalize ---

def test_normalize_range():
    arr = np.linspace(0, 100, 200)
    out = _robust_normalize(arr)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_normalize_flat():
    arr = np.full((10, 10), 5.0)
    out = _robust_normalize(arr)
    assert np.all(out == 0.0)


# --- save_previews ---

def test_saves_three_planes(synthetic_volume, tmp_path):
    out_dir = str(tmp_path / "previews")
    saved = save_previews(synthetic_volume, output_dir=out_dir)
    assert set(saved.keys()) == {"axial", "coronal", "sagittal"}


def test_png_files_exist(synthetic_volume, tmp_path):
    out_dir = str(tmp_path / "previews")
    saved = save_previews(synthetic_volume, output_dir=out_dir)
    for path in saved.values():
        assert Path(path).exists(), f"Expected PNG at {path}"
        assert Path(path).stat().st_size > 0


def test_custom_prefix(synthetic_volume, tmp_path):
    out_dir = str(tmp_path / "previews")
    saved = save_previews(synthetic_volume, output_dir=out_dir, prefix="myscan")
    for plane, path in saved.items():
        assert Path(path).name == f"myscan_{plane}.png"


def test_constant_volume_no_crash(constant_volume, tmp_path):
    out_dir = str(tmp_path / "previews")
    saved = save_previews(constant_volume, output_dir=out_dir)
    for path in saved.values():
        assert Path(path).exists()


def test_4d_volume_reduced(volume_4d, tmp_path):
    out_dir = str(tmp_path / "previews")
    saved = save_previews(volume_4d, output_dir=out_dir)
    assert len(saved) == 3
