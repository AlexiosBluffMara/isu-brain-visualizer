"""Tests for gaussian_seed — uses small NIfTI fixtures."""

from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

from fcd_prototype.gaussian_seed import build_seed, save_seed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_volume(tmp_path: Path) -> Path:
    """32×32×24 synthetic volume with a hot spot."""
    rng = np.random.default_rng(1)
    data = (rng.standard_normal((32, 32, 24)).astype(np.float32) * 100 + 500).clip(0, 4095)
    # Insert a bright region
    data[14:18, 14:18, 10:13] = 2000.0
    affine = np.diag([1.5, 1.5, 2.0, 1.0])
    path = tmp_path / "vol.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


@pytest.fixture()
def small_roi(tmp_path: Path) -> Path:
    """Binary ROI in the bright region."""
    data = np.zeros((32, 32, 24), dtype=np.uint8)
    data[14:18, 14:18, 10:13] = 1
    affine = np.diag([1.5, 1.5, 2.0, 1.0])
    path = tmp_path / "roi.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


# ---------------------------------------------------------------------------
# build_seed
# ---------------------------------------------------------------------------

def test_seed_keys_present(small_volume):
    seed = build_seed(small_volume, max_points=500)
    required = {
        "positions_world", "positions_voxel", "scales",
        "rotations_quat", "opacity", "intensity",
        "affine", "source_shape", "is_roi",
    }
    assert required.issubset(seed.keys())


def test_seed_positions_shape(small_volume):
    seed = build_seed(small_volume, max_points=500)
    n = len(seed["positions_world"])
    assert seed["positions_world"].shape == (n, 3)
    assert seed["positions_voxel"].shape == (n, 3)


def test_seed_scales_shape(small_volume):
    seed = build_seed(small_volume, max_points=500)
    n = len(seed["positions_world"])
    assert seed["scales"].shape == (n, 3)


def test_seed_rotations_identity(small_volume):
    seed = build_seed(small_volume, max_points=500)
    # w=1, x=0, y=0, z=0 for all
    assert np.allclose(seed["rotations_quat"][:, 0], 1.0)
    assert np.allclose(seed["rotations_quat"][:, 1:], 0.0)


def test_seed_opacity_range(small_volume):
    seed = build_seed(small_volume, max_points=500)
    assert seed["opacity"].min() >= 0.0
    assert seed["opacity"].max() <= 1.0


def test_seed_max_points_respected(small_volume):
    seed = build_seed(small_volume, max_points=100)
    assert len(seed["positions_world"]) <= 100


def test_seed_affine_shape(small_volume):
    seed = build_seed(small_volume)
    assert seed["affine"].shape == (4, 4)


def test_seed_source_shape(small_volume):
    seed = build_seed(small_volume)
    assert list(seed["source_shape"]) == [32, 32, 24]


def test_seed_deterministic(small_volume):
    s1 = build_seed(small_volume, max_points=200, rng_seed=7)
    s2 = build_seed(small_volume, max_points=200, rng_seed=7)
    np.testing.assert_array_equal(s1["positions_voxel"], s2["positions_voxel"])


def test_seed_different_seeds(small_volume):
    s1 = build_seed(small_volume, max_points=200, rng_seed=1)
    s2 = build_seed(small_volume, max_points=200, rng_seed=2)
    # Very unlikely to be identical with different seeds
    assert not np.array_equal(s1["positions_voxel"], s2["positions_voxel"])


# ---------------------------------------------------------------------------
# ROI integration
# ---------------------------------------------------------------------------

def test_seed_roi_labels(small_volume, small_roi):
    seed = build_seed(small_volume, roi_path=small_roi, max_points=2000)
    assert seed["is_roi"].any(), "Expected some ROI voxels in seed"


def test_seed_roi_prioritized(small_volume, small_roi):
    """ROI voxels appear in the seed even at tight max_points."""
    seed = build_seed(small_volume, roi_path=small_roi, max_points=50)
    assert seed["is_roi"].any()


def test_seed_without_roi_all_false(small_volume):
    seed = build_seed(small_volume, roi_path=None, max_points=500)
    assert not seed["is_roi"].any()


# ---------------------------------------------------------------------------
# save_seed and round-trip
# ---------------------------------------------------------------------------

def test_save_seed_creates_npz(tmp_path, small_volume):
    seed = build_seed(small_volume, max_points=200)
    out = tmp_path / "seeds" / "test.npz"
    save_seed(seed, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_seed_roundtrip(tmp_path, small_volume):
    seed = build_seed(small_volume, max_points=200)
    out = tmp_path / "test_rt.npz"
    save_seed(seed, out)
    loaded = dict(np.load(str(out)))
    np.testing.assert_array_equal(seed["positions_world"], loaded["positions_world"])
    np.testing.assert_array_equal(seed["affine"], loaded["affine"])


# ---------------------------------------------------------------------------
# Edge case: empty / zero volume
# ---------------------------------------------------------------------------

def test_seed_empty_volume_raises(tmp_path):
    data = np.zeros((16, 16, 16), dtype=np.float32)
    path = tmp_path / "empty.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    with pytest.raises(ValueError, match="No foreground voxels"):
        build_seed(path, max_points=100)
