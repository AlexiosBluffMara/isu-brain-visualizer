"""Tests for product_demo — uses synthetic NIfTI fixtures, no real data."""

from pathlib import Path
import json

import nibabel as nib
import numpy as np
import pytest

from fcd_prototype.product_demo import build_product_demo


# ---------------------------------------------------------------------------
# Minimal fake BIDS dataset fixture
# ---------------------------------------------------------------------------

def _make_volume(path: Path, shape=(20, 20, 16), affine=None, dtype=np.float32) -> None:
    if affine is None:
        affine = np.diag([1.0, 1.0, 1.5, 1.0])
    rng = np.random.default_rng(0)
    data = (rng.standard_normal(shape).astype(dtype) * 200 + 600).clip(0, 4095)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _make_roi(path: Path, shape=(20, 20, 16), affine=None) -> None:
    if affine is None:
        affine = np.diag([1.0, 1.0, 1.5, 1.0])
    data = np.zeros(shape, dtype=np.uint8)
    data[8:12, 8:12, 6:10] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


@pytest.fixture()
def fake_dataset_with_roi(tmp_path: Path) -> tuple[Path, str]:
    """Minimal BIDS-like dataset with T1w + FLAIR + ROI for sub-00001."""
    subj = "sub-00001"
    anat = tmp_path / "ds" / subj / "anat"
    anat.mkdir(parents=True)
    _make_volume(anat / f"{subj}_acq-sag_T1w.nii.gz")
    _make_volume(anat / f"{subj}_FLAIR.nii.gz")
    _make_roi(anat / f"{subj}_FLAIR_roi.nii.gz")
    return tmp_path / "ds", subj


@pytest.fixture()
def fake_dataset_no_roi(tmp_path: Path) -> tuple[Path, str]:
    """Minimal BIDS-like dataset with T1w + FLAIR, no ROI, for sub-00002."""
    subj = "sub-00002"
    anat = tmp_path / "ds2" / subj / "anat"
    anat.mkdir(parents=True)
    _make_volume(anat / f"{subj}_acq-sag_T1w.nii.gz")
    _make_volume(anat / f"{subj}_FLAIR.nii.gz")
    return tmp_path / "ds2", subj


@pytest.fixture()
def fake_dataset_t1w_only(tmp_path: Path) -> tuple[Path, str]:
    """Only T1w, no FLAIR, no ROI."""
    subj = "sub-00003"
    anat = tmp_path / "ds3" / subj / "anat"
    anat.mkdir(parents=True)
    _make_volume(anat / f"{subj}_acq-sag_T1w.nii.gz")
    return tmp_path / "ds3", subj


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_demo_creates_manifest(tmp_path, fake_dataset_with_roi):
    ds_dir, subj = fake_dataset_with_roi
    out = tmp_path / "out"
    manifest = build_product_demo(ds_dir, subj, out, max_seed_points=200)
    assert (out / subj / "manifest.json").exists()
    assert manifest["subject"] == subj


def test_build_demo_artifacts_exist(tmp_path, fake_dataset_with_roi):
    ds_dir, subj = fake_dataset_with_roi
    out = tmp_path / "out"
    manifest = build_product_demo(ds_dir, subj, out, max_seed_points=200)
    out_subj = out / subj
    for key, fname in manifest["artifacts"].items():
        assert (out_subj / fname).exists(), f"Missing artifact: {key} -> {fname}"


def test_build_demo_roi_subject_has_roi_artifacts(tmp_path, fake_dataset_with_roi):
    ds_dir, subj = fake_dataset_with_roi
    out = tmp_path / "out"
    manifest = build_product_demo(ds_dir, subj, out, max_seed_points=200)
    assert "lesion_roi" in manifest["artifacts"]
    assert "lesion_heatmap" in manifest["artifacts"]


def test_build_demo_no_roi_omits_roi_artifacts(tmp_path, fake_dataset_no_roi):
    ds_dir, subj = fake_dataset_no_roi
    out = tmp_path / "out"
    manifest = build_product_demo(ds_dir, subj, out, max_seed_points=200)
    assert "lesion_roi" not in manifest["artifacts"]
    assert "lesion_heatmap" not in manifest["artifacts"]


def test_build_demo_splat_shape_matches_base(tmp_path, fake_dataset_with_roi):
    ds_dir, subj = fake_dataset_with_roi
    out = tmp_path / "out"
    manifest = build_product_demo(ds_dir, subj, out, max_seed_points=200)
    out_subj = out / subj

    splat_img = nib.load(str(out_subj / "gaussian_splat.nii.gz"))
    # Find base shape (FLAIR or T1w)
    if "patient_flair" in manifest["artifacts"]:
        base_img = nib.load(str(out_subj / manifest["artifacts"]["patient_flair"]))
    else:
        base_img = nib.load(str(out_subj / manifest["artifacts"]["patient_t1w"]))
    assert splat_img.shape == base_img.shape


def test_build_demo_affine_consistent(tmp_path, fake_dataset_with_roi):
    ds_dir, subj = fake_dataset_with_roi
    out = tmp_path / "out"
    manifest = build_product_demo(ds_dir, subj, out, max_seed_points=200)
    out_subj = out / subj

    if "patient_flair" in manifest["artifacts"]:
        base_img = nib.load(str(out_subj / manifest["artifacts"]["patient_flair"]))
    else:
        base_img = nib.load(str(out_subj / manifest["artifacts"]["patient_t1w"]))
    base_affine = base_img.affine

    for key in ("lesion_roi", "lesion_heatmap", "gaussian_splat"):
        if key not in manifest["artifacts"]:
            continue
        img = nib.load(str(out_subj / manifest["artifacts"][key]))
        np.testing.assert_allclose(
            img.affine, base_affine, atol=1e-5,
            err_msg=f"Affine mismatch for {key}"
        )


def test_build_demo_t1w_only(tmp_path, fake_dataset_t1w_only):
    ds_dir, subj = fake_dataset_t1w_only
    out = tmp_path / "out"
    manifest = build_product_demo(ds_dir, subj, out, max_seed_points=200)
    assert "patient_t1w" in manifest["artifacts"]
    assert "patient_flair" not in manifest["artifacts"]


def test_build_demo_force_rebuilds(tmp_path, fake_dataset_with_roi):
    ds_dir, subj = fake_dataset_with_roi
    out = tmp_path / "out"
    build_product_demo(ds_dir, subj, out, max_seed_points=200)
    manifest_path = out / subj / "manifest.json"
    mtime1 = manifest_path.stat().st_mtime
    build_product_demo(ds_dir, subj, out, max_seed_points=200, force=True)
    mtime2 = manifest_path.stat().st_mtime
    assert mtime2 >= mtime1


def test_build_demo_skips_without_force(tmp_path, fake_dataset_with_roi):
    ds_dir, subj = fake_dataset_with_roi
    out = tmp_path / "out"
    build_product_demo(ds_dir, subj, out, max_seed_points=200)
    manifest_path = out / subj / "manifest.json"
    mtime1 = manifest_path.stat().st_mtime
    build_product_demo(ds_dir, subj, out, max_seed_points=200, force=False)
    assert manifest_path.stat().st_mtime == mtime1


def test_build_demo_manifest_has_notes(tmp_path, fake_dataset_with_roi):
    ds_dir, subj = fake_dataset_with_roi
    out = tmp_path / "out"
    manifest = build_product_demo(ds_dir, subj, out, max_seed_points=200)
    assert isinstance(manifest["notes"], list)
    assert any("CPU splat" in n for n in manifest["notes"])


def test_manifest_group_from_participants_tsv(tmp_path):
    """Group comes from participants.tsv when present, not hard-coded 'unknown'."""
    subj = "sub-00058"
    anat = tmp_path / "ds" / subj / "anat"
    anat.mkdir(parents=True)
    _make_volume(anat / f"{subj}_acq-sag_T1w.nii.gz")
    _make_volume(anat / f"{subj}_FLAIR.nii.gz")
    _make_roi(anat / f"{subj}_FLAIR_roi.nii.gz")

    # Write participants.tsv with group=fcd for this subject
    tsv = tmp_path / "ds" / "participants.tsv"
    tsv.write_text("participant_id\tgroup\n" + f"{subj}\tfcd\n")

    out = tmp_path / "out"
    manifest = build_product_demo(tmp_path / "ds", subj, out, max_seed_points=200)
    assert manifest["group"] == "fcd", f"Expected 'fcd', got {manifest['group']!r}"


def test_manifest_group_unknown_without_participants_tsv(tmp_path):
    """Without participants.tsv the group falls back to 'unknown'."""
    subj = "sub-00001"
    anat = tmp_path / "ds" / subj / "anat"
    anat.mkdir(parents=True)
    _make_volume(anat / f"{subj}_acq-sag_T1w.nii.gz")
    _make_volume(anat / f"{subj}_FLAIR.nii.gz")

    out = tmp_path / "out"
    manifest = build_product_demo(tmp_path / "ds", subj, out, max_seed_points=200)
    assert manifest["group"] == "unknown"


def test_splat_concentrated_near_roi(tmp_path, fake_dataset_with_roi):
    """gaussian_splat overlay must be spatially concentrated near the ROI.

    The fixture ROI is at [8:12, 8:12, 6:10] in a 20×20×16 volume.
    At least 70% of the splat's total signal mass should lie within a
    4-vox dilation of that bounding box.

    splat_sigma=1.5 is used here because the synthetic volume is only 20 vox
    per side — sigma=4.0 (the real-brain default) would diffuse energy across
    the whole grid and make the geometric assertion meaningless at toy scale.
    """
    ds_dir, subj = fake_dataset_with_roi
    out = tmp_path / "out"
    manifest = build_product_demo(ds_dir, subj, out, max_seed_points=500, splat_sigma=1.5)
    out_subj = out / subj

    if "gaussian_splat" not in manifest["artifacts"]:
        pytest.skip("No gaussian_splat artifact — subject has no ROI")

    splat_img = nib.load(str(out_subj / manifest["artifacts"]["gaussian_splat"]))
    data = splat_img.get_fdata(dtype=np.float32)

    # Dilated ROI bounding box: [8:12, 8:12, 6:10] + 4-vox margin
    dil = 4
    roi_box = np.zeros(data.shape, dtype=bool)
    roi_box[
        max(0, 8 - dil): min(data.shape[0], 12 + dil),
        max(0, 8 - dil): min(data.shape[1], 12 + dil),
        max(0, 6 - dil): min(data.shape[2], 10 + dil),
    ] = True

    total_mass = float(data.sum())
    assert total_mass > 0, "gaussian_splat overlay has no signal"
    inside_mass = float(data[roi_box].sum())
    fraction_inside = inside_mass / total_mass
    assert fraction_inside >= 0.70, (
        f"Only {fraction_inside:.1%} of splat mass inside dilated ROI bbox; "
        f"expected >=70%.  Overlay is too diffuse."
    )


def test_no_roi_subject_has_no_gaussian_splat(tmp_path, fake_dataset_no_roi):
    """Subjects without an ROI should not have a gaussian_splat in the manifest."""
    ds_dir, subj = fake_dataset_no_roi
    out = tmp_path / "out"
    manifest = build_product_demo(ds_dir, subj, out, max_seed_points=200)
    assert "gaussian_splat" not in manifest["artifacts"], (
        "gaussian_splat should be omitted when no ROI is present"
    )


def test_roi_resampled_when_affine_differs_but_shape_matches(tmp_path):
    """ROI is resampled when shape matches base but affine is different."""
    subj = "sub-00099"
    anat = tmp_path / "ds" / subj / "anat"
    anat.mkdir(parents=True)

    base_affine = np.diag([1.0, 1.0, 1.5, 1.0])
    # Same shape, different affine (shifted origin)
    roi_affine = base_affine.copy()
    roi_affine[:3, 3] = 10.0  # non-zero origin shift

    _make_volume(anat / f"{subj}_FLAIR.nii.gz", shape=(20, 20, 16), affine=base_affine)

    # ROI: same shape as base but different affine; place mask in a corner
    roi_data = np.zeros((20, 20, 16), dtype=np.uint8)
    roi_data[0:4, 0:4, 0:4] = 1
    roi_path = anat / f"{subj}_FLAIR_roi.nii.gz"
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(roi_data, roi_affine), str(roi_path))

    out = tmp_path / "out"
    manifest = build_product_demo(tmp_path / "ds", subj, out, max_seed_points=200)

    # lesion_roi must exist and carry base_affine (not roi_affine)
    assert "lesion_roi" in manifest["artifacts"]
    roi_out = nib.load(str(out / subj / manifest["artifacts"]["lesion_roi"]))
    np.testing.assert_allclose(
        roi_out.affine, nib.as_closest_canonical(nib.Nifti1Image(np.zeros((20, 20, 16)), base_affine)).affine,
        atol=1e-5,
        err_msg="lesion_roi should carry base affine after resampling"
    )
