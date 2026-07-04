"""Tests for real_preview — uses synthetic NIfTI fixtures, no real data."""

from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

from fcd_prototype.real_preview import (
    save_real_preview,
    _load_canonical,
    _world_to_voxel,
    _clamp_voxel,
    _roi_world_centroid,
    _foreground_world_centroid,
)


@pytest.fixture()
def nifti_volume(tmp_path: Path) -> Path:
    """64×64×48 synthetic brain-like volume."""
    rng = np.random.default_rng(0)
    data = (rng.standard_normal((64, 64, 48)).astype(np.float32) * 200 + 600).clip(0, 4095)
    affine = np.diag([1.0, 1.0, 1.2, 1.0])
    img = nib.Nifti1Image(data, affine)
    path = tmp_path / "vol.nii.gz"
    nib.save(img, str(path))
    return path


@pytest.fixture()
def roi_volume(tmp_path: Path) -> Path:
    """Small binary ROI mask — non-zero in a 6×6×4 region at [30:36, 30:36, 20:24]."""
    data = np.zeros((64, 64, 48), dtype=np.uint8)
    data[30:36, 30:36, 20:24] = 1
    affine = np.diag([1.0, 1.0, 1.2, 1.0])
    img = nib.Nifti1Image(data, affine)
    path = tmp_path / "roi.nii.gz"
    nib.save(img, str(path))
    return path


@pytest.fixture()
def offcenter_roi_volume(tmp_path: Path) -> Path:
    """ROI mask placed at voxel corner [55:60, 55:60, 42:46], far from volume centre."""
    data = np.zeros((64, 64, 48), dtype=np.uint8)
    data[55:60, 55:60, 42:46] = 1
    affine = np.diag([1.0, 1.0, 1.2, 1.0])
    img = nib.Nifti1Image(data, affine)
    path = tmp_path / "roi_offcenter.nii.gz"
    nib.save(img, str(path))
    return path


# ---------------------------------------------------------------------------
# Basic preview generation tests (inherited from prior suite)
# ---------------------------------------------------------------------------

def test_save_preview_creates_three_pngs(tmp_path, nifti_volume):
    out_dir = tmp_path / "out"
    saved = save_real_preview(
        t1w_path=nifti_volume,
        flair_path=None,
        roi_path=None,
        output_dir=out_dir,
        subject="sub-00001",
    )
    assert len(saved) == 3
    for plane in ("axial", "coronal", "sagittal"):
        assert plane in saved
        assert saved[plane].exists()
        assert saved[plane].stat().st_size > 0


def test_save_preview_with_flair_creates_two_col(tmp_path, nifti_volume):
    out_dir = tmp_path / "out2"
    saved = save_real_preview(
        t1w_path=nifti_volume,
        flair_path=nifti_volume,
        roi_path=None,
        output_dir=out_dir,
        subject="sub-00002",
    )
    assert len(saved) == 3
    for p in saved.values():
        assert p.exists()


def test_save_preview_with_roi_overlay(tmp_path, nifti_volume, roi_volume):
    out_dir = tmp_path / "out3"
    saved = save_real_preview(
        t1w_path=nifti_volume,
        flair_path=None,
        roi_path=roi_volume,
        output_dir=out_dir,
        subject="sub-00058",
    )
    for p in saved.values():
        assert p.exists()


def test_save_preview_no_volumes_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        save_real_preview(
            t1w_path=None,
            flair_path=None,
            roi_path=None,
            output_dir=tmp_path / "empty",
            subject="sub-00000",
        )


def test_save_preview_missing_t1w_uses_flair(tmp_path, nifti_volume):
    out_dir = tmp_path / "out4"
    saved = save_real_preview(
        t1w_path=None,
        flair_path=nifti_volume,
        roi_path=None,
        output_dir=out_dir,
        subject="sub-00003",
    )
    assert len(saved) == 3


def test_save_preview_output_dir_created(tmp_path, nifti_volume):
    nested = tmp_path / "a" / "b" / "c"
    assert not nested.exists()
    save_real_preview(
        t1w_path=nifti_volume,
        flair_path=None,
        roi_path=None,
        output_dir=nested,
        subject="sub-00004",
    )
    assert nested.exists()


# ---------------------------------------------------------------------------
# World-space slice selection tests
# ---------------------------------------------------------------------------

def test_roi_centroid_targets_offcenter_roi(tmp_path, nifti_volume, offcenter_roi_volume):
    """With an off-center ROI, the slice index should be closer to the ROI than to centre."""
    roi_d, roi_a = _load_canonical(offcenter_roi_volume)
    world_xyz = _roi_world_centroid(roi_d, roi_a)
    vol_d, vol_a = _load_canonical(nifti_volume)
    vox = _world_to_voxel(world_xyz, vol_a)
    ci, cj, ck = _clamp_voxel(vox, vol_d.shape)

    # ROI centroid is around voxel (57, 57, 44); volume centre is (32, 32, 24)
    # The selected index should be closer to 57 than to 32
    assert ci > 32, f"Expected axial index > 32, got {ci}"
    assert cj > 32, f"Expected coronal index > 32, got {cj}"
    assert ck > 24, f"Expected sagittal index > 24, got {ck}"


def test_world_coordinate_maps_consistently_across_different_affines(tmp_path):
    """Two volumes with different shapes/affines must map the same world point to valid voxels."""
    # Volume A: 64×64×48 at 1mm isotropic, origin at (0,0,0)
    shape_a = (64, 64, 48)
    affine_a = np.diag([1.0, 1.0, 1.2, 1.0])

    # Volume B: 32×32×24 at 2mm isotropic, origin at (10, 10, 5)
    affine_b = np.array([
        [2.0, 0, 0, 10],
        [0, 2.0, 0, 10],
        [0, 0, 2.4, 5],
        [0,   0,  0,  1],
    ])
    shape_b = (32, 32, 24)

    world_xyz = np.array([32.0, 32.0, 28.0])  # in mm

    vox_a = _world_to_voxel(world_xyz, affine_a)
    ci_a, cj_a, ck_a = _clamp_voxel(vox_a, shape_a)
    assert 0 <= ci_a < shape_a[0]
    assert 0 <= cj_a < shape_a[1]
    assert 0 <= ck_a < shape_a[2]

    vox_b = _world_to_voxel(world_xyz, affine_b)
    ci_b, cj_b, ck_b = _clamp_voxel(vox_b, shape_b)
    assert 0 <= ci_b < shape_b[0]
    assert 0 <= cj_b < shape_b[1]
    assert 0 <= ck_b < shape_b[2]


def test_foreground_centroid_returns_valid_world_coord(tmp_path, nifti_volume):
    data, affine = _load_canonical(nifti_volume)
    world_xyz = _foreground_world_centroid(data, affine)
    assert world_xyz.shape == (3,)
    vox = _world_to_voxel(world_xyz, affine)
    ci, cj, ck = _clamp_voxel(vox, data.shape)
    assert 0 <= ci < data.shape[0]
    assert 0 <= cj < data.shape[1]
    assert 0 <= ck < data.shape[2]


def test_roi_centroid_world_coord(tmp_path, roi_volume):
    """ROI voxels are at [30:36, 30:36, 20:24]; centroid should be ~(32.5, 32.5, 22) voxels."""
    roi_d, roi_a = _load_canonical(roi_volume)
    world_xyz = _roi_world_centroid(roi_d, roi_a)
    # Map back through same affine
    vox = _world_to_voxel(world_xyz, roi_a)
    assert abs(vox[0] - 32.5) < 1.0
    assert abs(vox[1] - 32.5) < 1.0
    assert abs(vox[2] - 22.0) < 1.0
