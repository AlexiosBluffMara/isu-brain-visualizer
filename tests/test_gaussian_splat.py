"""Tests for gaussian_splat — CPU preview renderer.  No real data needed."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fcd_prototype.gaussian_splat import render_lesion_splat, render_splat


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_ref_volume(tmp_path: Path) -> Path:
    """20×20×16 synthetic reference volume."""
    data = np.random.default_rng(0).random((20, 20, 16)).astype(np.float32)
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    path = tmp_path / "ref.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


@pytest.fixture()
def simple_seed_npz(tmp_path: Path) -> Path:
    """Minimal seed .npz with a handful of points inside the 20×20×16 grid."""
    rng = np.random.default_rng(1)
    n = 50
    positions_voxel = rng.random((n, 3)).astype(np.float32) * np.array([19, 19, 15])
    intensity = rng.random(n).astype(np.float32)
    opacity = rng.random(n).astype(np.float32)
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    seed_path = tmp_path / "seed.npz"
    np.savez_compressed(
        str(seed_path),
        positions_voxel=positions_voxel,
        intensity=intensity,
        opacity=opacity,
        affine=affine,
        positions_world=positions_voxel * 2,
        scales=np.ones((n, 3), dtype=np.float32),
        rotations_quat=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        source_shape=np.array([20, 20, 16], dtype=np.int64),
        is_roi=np.zeros(n, dtype=bool),
    )
    return seed_path


@pytest.fixture()
def out_of_bounds_seed_npz(tmp_path: Path) -> Path:
    """Seed with points mostly outside the 20×20×16 grid — should not crash."""
    rng = np.random.default_rng(2)
    n = 30
    # most coords outside [0..19, 0..19, 0..15]
    positions_voxel = (rng.random((n, 3)) * 200 - 50).astype(np.float32)
    intensity = rng.random(n).astype(np.float32)
    opacity = rng.random(n).astype(np.float32)
    seed_path = tmp_path / "oob_seed.npz"
    np.savez_compressed(
        str(seed_path),
        positions_voxel=positions_voxel,
        intensity=intensity,
        opacity=opacity,
        affine=np.eye(4),
        positions_world=positions_voxel,
        scales=np.ones((n, 3), dtype=np.float32),
        rotations_quat=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        source_shape=np.array([20, 20, 16], dtype=np.int64),
        is_roi=np.zeros(n, dtype=bool),
    )
    return seed_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_render_splat_output_exists(tmp_path, simple_seed_npz, small_ref_volume):
    out = tmp_path / "splat.nii.gz"
    result = render_splat(simple_seed_npz, small_ref_volume, out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_splat_shape_matches_reference(tmp_path, simple_seed_npz, small_ref_volume):
    out = tmp_path / "splat.nii.gz"
    render_splat(simple_seed_npz, small_ref_volume, out)
    img = nib.load(str(out))
    assert img.shape == (20, 20, 16)


def test_render_splat_affine_preserved(tmp_path, simple_seed_npz, small_ref_volume):
    out = tmp_path / "splat.nii.gz"
    render_splat(simple_seed_npz, small_ref_volume, out)
    ref_img = nib.load(str(small_ref_volume))
    ref_img = nib.as_closest_canonical(ref_img)
    out_img = nib.load(str(out))
    np.testing.assert_allclose(out_img.affine, ref_img.affine, atol=1e-5)


def test_render_splat_normalized_zero_to_one(tmp_path, simple_seed_npz, small_ref_volume):
    out = tmp_path / "splat.nii.gz"
    render_splat(simple_seed_npz, small_ref_volume, out)
    data = nib.load(str(out)).get_fdata(dtype=np.float32)
    assert float(data.min()) >= 0.0
    assert float(data.max()) <= 1.0 + 1e-5


def test_render_splat_out_of_bounds_no_crash(tmp_path, out_of_bounds_seed_npz, small_ref_volume):
    out = tmp_path / "oob_splat.nii.gz"
    render_splat(out_of_bounds_seed_npz, small_ref_volume, out)
    assert out.exists()


def test_render_splat_max_points_respected(tmp_path, simple_seed_npz, small_ref_volume):
    out1 = tmp_path / "splat_all.nii.gz"
    out5 = tmp_path / "splat_5.nii.gz"
    render_splat(simple_seed_npz, small_ref_volume, out1)
    render_splat(simple_seed_npz, small_ref_volume, out5, max_points=5)
    # Both outputs should exist and be valid
    for p in (out1, out5):
        assert p.exists()
        img = nib.load(str(p))
        assert img.shape == (20, 20, 16)


def test_render_splat_sigma_zero(tmp_path, simple_seed_npz, small_ref_volume):
    """sigma_vox=0 skips blurring — should still produce valid output."""
    out = tmp_path / "splat_sig0.nii.gz"
    render_splat(simple_seed_npz, small_ref_volume, out, sigma_vox=0.0)
    assert out.exists()
    data = nib.load(str(out)).get_fdata(dtype=np.float32)
    assert float(data.max()) <= 1.0 + 1e-5


def test_render_splat_creates_parent_dirs(tmp_path, simple_seed_npz, small_ref_volume):
    out = tmp_path / "a" / "b" / "splat.nii.gz"
    render_splat(simple_seed_npz, small_ref_volume, out)
    assert out.exists()


# ---------------------------------------------------------------------------
# Fixtures for render_lesion_splat
# ---------------------------------------------------------------------------

@pytest.fixture()
def lesion_seed_npz(tmp_path: Path) -> Path:
    """Seed with ROI-marked points clustered in [7:13, 7:13, 5:11] of a 20×20×16 grid.

    Non-ROI points are scattered uniformly across the whole grid.
    """
    rng = np.random.default_rng(3)
    n_roi = 40
    n_bg = 200

    # ROI points — clustered in center block
    roi_positions = rng.random((n_roi, 3)).astype(np.float32) * np.array([6, 6, 6]) + np.array([7, 7, 5])
    roi_intensity = rng.random(n_roi).astype(np.float32)
    roi_opacity = rng.random(n_roi).astype(np.float32)

    # Background points — scattered everywhere
    bg_positions = rng.random((n_bg, 3)).astype(np.float32) * np.array([19, 19, 15])
    bg_intensity = rng.random(n_bg).astype(np.float32)
    bg_opacity = rng.random(n_bg).astype(np.float32)

    positions_voxel = np.vstack([roi_positions, bg_positions])
    intensity = np.concatenate([roi_intensity, bg_intensity])
    opacity = np.concatenate([roi_opacity, bg_opacity])
    is_roi = np.array([True] * n_roi + [False] * n_bg, dtype=bool)

    seed_path = tmp_path / "lesion_seed.npz"
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    np.savez_compressed(
        str(seed_path),
        positions_voxel=positions_voxel,
        intensity=intensity,
        opacity=opacity,
        affine=affine,
        positions_world=positions_voxel * 2,
        scales=np.ones((len(positions_voxel), 3), dtype=np.float32),
        rotations_quat=np.tile([1, 0, 0, 0], (len(positions_voxel), 1)).astype(np.float32),
        source_shape=np.array([20, 20, 16], dtype=np.int64),
        is_roi=is_roi,
    )
    return seed_path


@pytest.fixture()
def no_roi_seed_npz(tmp_path: Path) -> Path:
    """Seed where all is_roi flags are False."""
    rng = np.random.default_rng(4)
    n = 50
    positions_voxel = rng.random((n, 3)).astype(np.float32) * np.array([19, 19, 15])
    intensity = rng.random(n).astype(np.float32)
    opacity = rng.random(n).astype(np.float32)
    seed_path = tmp_path / "no_roi_seed.npz"
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    np.savez_compressed(
        str(seed_path),
        positions_voxel=positions_voxel,
        intensity=intensity,
        opacity=opacity,
        affine=affine,
        positions_world=positions_voxel * 2,
        scales=np.ones((n, 3), dtype=np.float32),
        rotations_quat=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        source_shape=np.array([20, 20, 16], dtype=np.int64),
        is_roi=np.zeros(n, dtype=bool),
    )
    return seed_path


# ---------------------------------------------------------------------------
# Tests for render_lesion_splat
# ---------------------------------------------------------------------------

def test_render_lesion_splat_output_exists(tmp_path, lesion_seed_npz, small_ref_volume):
    out = tmp_path / "lesion_splat.nii.gz"
    result = render_lesion_splat(lesion_seed_npz, small_ref_volume, out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_lesion_splat_shape_matches_reference(tmp_path, lesion_seed_npz, small_ref_volume):
    out = tmp_path / "lesion_splat.nii.gz"
    render_lesion_splat(lesion_seed_npz, small_ref_volume, out)
    img = nib.load(str(out))
    assert img.shape == (20, 20, 16)


def test_render_lesion_splat_normalized_zero_to_one(tmp_path, lesion_seed_npz, small_ref_volume):
    out = tmp_path / "lesion_splat.nii.gz"
    render_lesion_splat(lesion_seed_npz, small_ref_volume, out)
    data = nib.load(str(out)).get_fdata(dtype=np.float32)
    assert float(data.min()) >= 0.0
    assert float(data.max()) <= 1.0 + 1e-5


def test_render_lesion_splat_spatially_concentrated(tmp_path, lesion_seed_npz, small_ref_volume):
    """Most of the splat energy must lie within a dilated ROI bounding-box.

    ROI seed points are in [7:13, 7:13, 5:11].  We dilate by 3 voxels (sigma=1
    Gaussian radius) and require that >=80% of total nonzero mass is inside.
    """
    out = tmp_path / "lesion_splat.nii.gz"
    render_lesion_splat(lesion_seed_npz, small_ref_volume, out, sigma_vox=1.0)
    data = nib.load(str(out)).get_fdata(dtype=np.float32)

    # Dilated ROI bounding box: center [7:13, 7:13, 5:11] + 3-vox margin
    dil = 3
    roi_box = np.zeros(data.shape, dtype=bool)
    roi_box[
        max(0, 7 - dil): min(data.shape[0], 13 + dil),
        max(0, 7 - dil): min(data.shape[1], 13 + dil),
        max(0, 5 - dil): min(data.shape[2], 11 + dil),
    ] = True

    total_mass = float(data.sum())
    inside_mass = float(data[roi_box].sum())
    assert total_mass > 0, "Splat has no signal"
    fraction_inside = inside_mass / total_mass
    assert fraction_inside >= 0.80, (
        f"Only {fraction_inside:.1%} of splat mass is inside the dilated ROI bbox; "
        f"expected >=80%.  Splat is too spatially diffuse."
    )


def test_render_lesion_splat_no_roi_uses_heatmap_fallback(
    tmp_path, no_roi_seed_npz, small_ref_volume
):
    """When seed has no ROI points, heatmap_path is used as the fallback signal."""
    # Build a tiny heatmap covering a known patch
    heatmap_data = np.zeros((20, 20, 16), dtype=np.float32)
    heatmap_data[8:12, 8:12, 6:10] = 0.8
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    heatmap_path = tmp_path / "heatmap.nii.gz"
    nib.save(nib.Nifti1Image(heatmap_data, affine), str(heatmap_path))

    out = tmp_path / "lesion_splat_fallback.nii.gz"
    render_lesion_splat(no_roi_seed_npz, small_ref_volume, out, heatmap_path=heatmap_path)
    data = nib.load(str(out)).get_fdata(dtype=np.float32)
    assert float(data.max()) > 0, "Heatmap fallback should produce nonzero output"
    # Signal should be present where heatmap was nonzero
    assert data[8:12, 8:12, 6:10].sum() > 0


def test_render_lesion_splat_no_roi_no_heatmap_produces_zeros(
    tmp_path, no_roi_seed_npz, small_ref_volume
):
    """When seed has no ROI points and no heatmap, output should be all-zero."""
    out = tmp_path / "lesion_splat_zeros.nii.gz"
    render_lesion_splat(no_roi_seed_npz, small_ref_volume, out)
    data = nib.load(str(out)).get_fdata(dtype=np.float32)
    assert float(data.max()) == 0.0, "Expected all-zero output with no ROI and no heatmap"


def test_render_lesion_splat_less_diffuse_than_whole_brain(
    tmp_path, lesion_seed_npz, small_ref_volume
):
    """Lesion splat must be more spatially concentrated than whole-brain render_splat.

    Measures concentration as fraction of voxels that are nonzero.  Fewer
    active voxels = more focused overlay.
    """
    out_lesion = tmp_path / "lesion.nii.gz"
    out_whole = tmp_path / "whole.nii.gz"
    render_lesion_splat(lesion_seed_npz, small_ref_volume, out_lesion, sigma_vox=1.0)
    render_splat(lesion_seed_npz, small_ref_volume, out_whole, sigma_vox=1.0)

    lesion_data = nib.load(str(out_lesion)).get_fdata(dtype=np.float32)
    whole_data = nib.load(str(out_whole)).get_fdata(dtype=np.float32)

    lesion_nonzero_frac = np.count_nonzero(lesion_data) / lesion_data.size
    whole_nonzero_frac = np.count_nonzero(whole_data) / whole_data.size
    assert lesion_nonzero_frac < whole_nonzero_frac, (
        f"Lesion splat nonzero fraction ({lesion_nonzero_frac:.3f}) should be "
        f"smaller than whole-brain ({whole_nonzero_frac:.3f})"
    )
