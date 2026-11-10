"""Tests for nifti_inspect using synthetic NIfTI data."""

import numpy as np
import nibabel as nib
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fcd_prototype.nifti_inspect import inspect_nifti


@pytest.fixture
def synthetic_nifti(tmp_path):
    """Create a small synthetic NIfTI file."""
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    affine = np.eye(4)
    affine[0, 0] = 1.0  # 1 mm isotropic
    affine[1, 1] = 1.0
    affine[2, 2] = 1.5  # 1.5 mm slice thickness
    img = nib.Nifti1Image(data, affine)
    img.header.set_zooms((1.0, 1.0, 1.5))
    path = tmp_path / "synthetic.nii"
    nib.save(img, str(path))
    return str(path)


@pytest.fixture
def synthetic_nifti_gz(tmp_path):
    """Create a small synthetic .nii.gz file."""
    data = np.ones((5, 5, 5), dtype=np.float32) * 42.0
    img = nib.Nifti1Image(data, np.eye(4))
    img.header.set_zooms((2.0, 2.0, 2.0))
    path = tmp_path / "synthetic.nii.gz"
    nib.save(img, str(path))
    return str(path)


@pytest.fixture
def nifti_with_nan(tmp_path):
    """NIfTI file containing NaN and Inf values."""
    data = np.zeros((4, 4, 4), dtype=np.float32)
    data[0, 0, 0] = np.nan
    data[1, 1, 1] = np.inf
    data[2, 2, 2] = -np.inf
    img = nib.Nifti1Image(data, np.eye(4))
    path = tmp_path / "nan_inf.nii"
    nib.save(img, str(path))
    return str(path)


def test_shape(synthetic_nifti):
    stats = inspect_nifti(synthetic_nifti)
    assert stats["shape"] == (2, 3, 4)


def test_voxel_spacing(synthetic_nifti):
    stats = inspect_nifti(synthetic_nifti)
    assert abs(stats["voxel_spacing_mm"][0] - 1.0) < 1e-5
    assert abs(stats["voxel_spacing_mm"][2] - 1.5) < 1e-5


def test_affine_is_4x4(synthetic_nifti):
    stats = inspect_nifti(synthetic_nifti)
    assert stats["affine"].shape == (4, 4)


def test_min_max_mean(synthetic_nifti):
    stats = inspect_nifti(synthetic_nifti)
    # data is arange(24), so min=0, max=23, mean=11.5
    assert stats["min"] == pytest.approx(0.0)
    assert stats["max"] == pytest.approx(23.0)
    assert stats["mean"] == pytest.approx(11.5)


def test_no_nan_no_inf(synthetic_nifti):
    stats = inspect_nifti(synthetic_nifti)
    assert stats["nan_count"] == 0
    assert stats["inf_count"] == 0


def test_gz_shape(synthetic_nifti_gz):
    stats = inspect_nifti(synthetic_nifti_gz)
    assert stats["shape"] == (5, 5, 5)


def test_nan_and_inf_counts(nifti_with_nan):
    stats = inspect_nifti(nifti_with_nan)
    assert stats["nan_count"] == 1
    assert stats["inf_count"] == 2  # +inf and -inf both count
