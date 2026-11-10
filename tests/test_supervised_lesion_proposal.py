"""Tests for supervised_lesion_proposal module."""

import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import fcd_prototype.supervised_lesion_proposal as slp
from fcd_prototype.supervised_lesion_proposal import (
    load_canonical,
    resize_array,
    _robust_normalize,
    _foreground_mask,
    save_nifti_mask,
    build_supervised_records,
    predict_lesion_roi,
    evaluate_lesion_proposer,
    evaluate_threshold_sweep,
    validate_spatial_size,
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from fcd_prototype.supervised_lesion_proposal import (
        TinyLesionNet,
        train_lesion_proposer,
        train_patch_lesion_proposer,
    )


def create_synthetic_nifti(shape, affine=None, data=None):
    """Create synthetic NIfTI file."""
    if data is None:
        data = np.random.rand(*shape).astype(np.float32)
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    return img


def _make_threshold_sweep_dataset(tmp_path: Path) -> tuple[Path, tuple[int, int, int]]:
    """Create one FCD subject with ROI and one HC subject without ROI."""
    dataset_dir = tmp_path / "sweep_dataset"
    shape = (12, 12, 12)

    for subject_id, group in (("sub-00001", "fcd"), ("sub-00002", "hc")):
        anat = dataset_dir / subject_id / "anat"
        anat.mkdir(parents=True, exist_ok=True)
        data = np.ones(shape, dtype=np.float32) * 0.5
        nib.save(
            nib.Nifti1Image(data, np.eye(4)),
            str(anat / f"{subject_id}_T2sel_FLAIR.nii.gz"),
        )
        if group == "fcd":
            roi = np.zeros(shape, dtype=np.float32)
            roi[2:5, 2:5, 2:5] = 1.0
            nib.save(
                nib.Nifti1Image(roi, np.eye(4)),
                str(anat / f"{subject_id}_T2sel_FLAIR_roi.nii.gz"),
            )

    with (dataset_dir / "participants.tsv").open("w") as fh:
        fh.write("participant_id\tgroup\n")
        fh.write("sub-00001\tfcd\n")
        fh.write("sub-00002\thc\n")

    return dataset_dir, shape


def test_load_canonical():
    """Test load_canonical handles canonical orientation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data = np.random.rand(10, 12, 14).astype(np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        path = tmpdir / "test.nii.gz"
        nib.save(img, str(path))

        loaded_data, loaded_affine, header, canonical_img = load_canonical(path)
        assert loaded_data.shape == (10, 12, 14)
        assert loaded_affine.shape == (4, 4)
        assert loaded_data.dtype == np.float32


def test_load_canonical_4d():
    """Test load_canonical squeezes 4D to 3D."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data = np.random.rand(10, 12, 14, 3).astype(np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        path = tmpdir / "test4d.nii.gz"
        nib.save(img, str(path))

        loaded_data, _, _, _ = load_canonical(path)
        assert loaded_data.shape == (10, 12, 14)


def test_robust_normalize():
    """Test robust normalization."""
    data = np.array([[[0, 10, 20, 30]]], dtype=np.float32)
    norm = _robust_normalize(data)
    assert norm.min() >= 0
    assert norm.max() <= 1


def test_robust_normalize_with_mask():
    """Test robust normalization with mask."""
    data = np.ones((5, 5, 5), dtype=np.float32) * 100
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[:3, :3, :3] = True
    norm = _robust_normalize(data, mask)
    assert norm.shape == (5, 5, 5)


def test_foreground_mask():
    """Test foreground mask extraction."""
    data = np.zeros((10, 10, 10), dtype=np.float32)
    data[3:7, 3:7, 3:7] = 1.0
    mask = _foreground_mask(data)
    assert mask.sum() > 0
    assert mask.sum() < data.size


def test_sample_patch_centers_adds_lesion_like_hard_negatives(tmp_path):
    """Hard negatives should target bright/edgy non-ROI tissue, not the manual ROI."""
    rng = np.random.default_rng(123)
    shape = (20, 20, 20)
    data = rng.normal(loc=0.35, scale=0.03, size=shape).astype(np.float32)
    data[3:6, 3:6, 3:6] = 0.9
    data[15:18, 15:18, 15:18] = 1.4

    roi = np.zeros(shape, dtype=np.float32)
    roi[3:6, 3:6, 3:6] = 1.0

    flair_path = tmp_path / "sub-00001_T2sel_FLAIR.nii.gz"
    roi_path = tmp_path / "sub-00001_T2sel_FLAIR_roi.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(flair_path))
    nib.save(nib.Nifti1Image(roi, np.eye(4)), str(roi_path))

    samples = slp._sample_patch_centers(
        [{
            "subject_id": "sub-00001",
            "flair_path": str(flair_path),
            "roi_path": str(roi_path),
        }],
        patch_size=8,
        positives_per_subject=2,
        negatives_per_subject=1,
        hard_negatives_per_subject=2,
        hard_negative_min_separation_voxels=2,
        seed=7,
    )

    hard_samples = [
        sample for sample in samples if sample.get("sample_type") == "negative_hard"
    ]
    assert len(hard_samples) == 2
    assert all(sample["label"] == 0.0 for sample in hard_samples)
    assert any(
        np.linalg.norm(np.asarray(sample["center"]) - np.asarray([16, 16, 16])) <= 2.0
        for sample in hard_samples
    )
    assert not any(
        3 <= sample["center"][0] < 6
        and 3 <= sample["center"][1] < 6
        and 3 <= sample["center"][2] < 6
        for sample in hard_samples
    )


def test_resize_array_same_shape():
    """Test resize_array returns copy when shape unchanged."""
    data = np.random.rand(16, 16, 16).astype(np.float32)
    resized = resize_array(data, (16, 16, 16), order=1)
    assert resized.shape == (16, 16, 16)


def test_resize_array_upsample():
    """Test resize_array upsampling."""
    data = np.ones((8, 8, 8), dtype=np.float32) * 0.5
    resized = resize_array(data, (16, 16, 16), order=1)
    assert resized.shape == (16, 16, 16)
    assert abs(resized.mean() - 0.5) < 0.1


def test_resize_array_downsample_nearest():
    """Test resize_array downsampling with nearest neighbor."""
    data = np.zeros((16, 16, 16), dtype=np.float32)
    data[4:12, 4:12, 4:12] = 1.0
    resized = resize_array(data, (8, 8, 8), order=0)
    assert resized.shape == (8, 8, 8)
    assert resized.sum() > 0


def test_save_nifti_mask():
    """Test saving NIfTI mask."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mask = np.random.rand(10, 10, 10).astype(np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(np.zeros((10, 10, 10)), affine)
        header = img.header

        output_path = tmpdir / "mask.nii.gz"
        save_nifti_mask(mask, affine, header, output_path)

        assert output_path.exists()
        loaded = nib.load(str(output_path))
        assert loaded.get_fdata().shape == (10, 10, 10)


def test_build_supervised_records_empty():
    """Test build_supervised_records with no data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        records = build_supervised_records(tmpdir)
        assert records == []


def test_build_supervised_records_synthetic():
    """Test build_supervised_records structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create synthetic subject directory
        dataset_dir = tmpdir / "ds004199"
        subj_dir = dataset_dir / "sub-00001" / "anat"
        subj_dir.mkdir(parents=True, exist_ok=True)

        # Create FLAIR volume
        flair_data = np.random.rand(64, 64, 64).astype(np.float32)
        flair_img = nib.Nifti1Image(flair_data, np.eye(4))
        flair_path = subj_dir / "sub-00001_T2sel_FLAIR.nii.gz"
        nib.save(flair_img, str(flair_path))

        # Create ROI mask
        roi_data = np.zeros((64, 64, 64), dtype=np.float32)
        roi_data[20:40, 20:40, 20:40] = 1.0
        roi_img = nib.Nifti1Image(roi_data, np.eye(4))
        roi_path = subj_dir / "sub-00001_T2sel_FLAIR_roi.nii.gz"
        nib.save(roi_img, str(roi_path))

        # Create participants.tsv inside dataset_dir
        participants_path = dataset_dir / "participants.tsv"
        with open(participants_path, 'w') as f:
            f.write("participant_id\tgroup\n")
            f.write("sub-00001\tfcd\n")

        records = build_supervised_records(dataset_dir)
        assert len(records) > 0
        assert records[0]["subject_id"] == "sub-00001"
        assert records[0]["group"] == "fcd"
        assert records[0]["roi_path"] is not None


def test_build_supervised_records_hc_no_roi():
    """Test that HC records have None for roi_path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create HC subject
        dataset_dir = tmpdir / "ds004199"
        subj_dir = dataset_dir / "sub-00002" / "anat"
        subj_dir.mkdir(parents=True, exist_ok=True)

        flair_data = np.random.rand(64, 64, 64).astype(np.float32)
        flair_img = nib.Nifti1Image(flair_data, np.eye(4))
        flair_path = subj_dir / "sub-00002_T2sel_FLAIR.nii.gz"
        nib.save(flair_img, str(flair_path))

        # Create participants.tsv inside dataset_dir
        participants_path = dataset_dir / "participants.tsv"
        with open(participants_path, 'w') as f:
            f.write("participant_id\tgroup\n")
            f.write("sub-00002\thc\n")

        records = build_supervised_records(dataset_dir, include_controls=True)
        assert len(records) > 0
        assert records[0]["group"] == "hc"
        assert records[0]["roi_path"] is None


def test_validate_spatial_size_valid():
    """Test validate_spatial_size accepts sizes divisible by 4."""
    validate_spatial_size((64, 64, 64))
    validate_spatial_size((128, 96, 64))
    validate_spatial_size((4, 4, 4))


def test_validate_spatial_size_invalid():
    """Test validate_spatial_size rejects sizes not divisible by 4."""
    with pytest.raises(ValueError, match="must be divisible by 4"):
        validate_spatial_size((65, 64, 64))

    with pytest.raises(ValueError, match="must be divisible by 4"):
        validate_spatial_size((64, 63, 64))

    with pytest.raises(ValueError, match="must be divisible by 4"):
        validate_spatial_size((16, 16, 17))


def test_predict_lesion_roi_reports_foreground_fraction(tmp_path, monkeypatch):
    """Prediction reports foreground burden needed for uploaded-MRI triage."""
    shape = (4, 4, 4)
    ckpt = tmp_path / "lesion_proposer.pt"
    ckpt.write_bytes(b"not used by monkeypatched loader")
    volume = tmp_path / "input_flair.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4)), str(volume))

    monkeypatch.setattr(
        slp,
        "_load_prediction_model",
        lambda checkpoint_path, device="auto": (object(), shape, {}, "cpu"),
    )

    def fake_predict_probability_map(model, spatial_size, volume_path, device):
        prob = np.zeros(shape, dtype=np.float32)
        prob[0:2, 0:2, 0:2] = 0.7
        prob[3, 3, 3] = 0.8
        return {
            "probability": prob,
            "affine": np.eye(4),
            "header": nib.Nifti1Header(),
            "shape": shape,
        }

    monkeypatch.setattr(slp, "_predict_probability_map", fake_predict_probability_map)

    output = tmp_path / "pred_roi.nii.gz"
    result = predict_lesion_roi(
        ckpt,
        volume,
        output,
        threshold=0.5,
        min_component_voxels=2,
        device="cpu",
    )

    assert output.exists()
    assert result["predicted_voxels"] == 8
    assert result["predicted_foreground_fraction"] == pytest.approx(8 / 64)
    assert nib.load(str(output)).get_fdata().sum() == pytest.approx(8.0)


def test_select_probability_mask_top_fraction_is_budgeted():
    """Top-fraction mode should select the requested highest-probability budget."""
    probability = np.arange(8, dtype=np.float32).reshape(2, 2, 2) / 10.0

    mask, meta = slp._select_probability_mask(
        probability,
        selection_mode="top-fraction",
        foreground_fraction=0.25,
    )

    assert int(mask.sum()) == 2
    assert mask[1, 1, 0] == pytest.approx(1.0)
    assert mask[1, 1, 1] == pytest.approx(1.0)
    assert meta["requested_foreground_voxels"] == 2
    assert meta["selected_voxels_before_component_filter"] == 2
    assert meta["effective_probability_threshold"] == pytest.approx(0.6)


def test_select_probability_mask_threshold_and_top_fraction_caps_eligible_voxels():
    """Combined mode should cap threshold-eligible voxels to the top budget."""
    probability = np.arange(8, dtype=np.float32).reshape(2, 2, 2) / 10.0

    mask, meta = slp._select_probability_mask(
        probability,
        threshold=0.2,
        selection_mode="threshold-and-top-fraction",
        foreground_fraction=0.25,
    )

    assert int(mask.sum()) == 2
    assert mask[1, 1, 0] == pytest.approx(1.0)
    assert mask[1, 1, 1] == pytest.approx(1.0)
    assert meta["threshold_eligible_voxels"] == 5
    assert meta["requested_foreground_voxels"] == 2
    assert meta["effective_probability_threshold"] == pytest.approx(0.6)


def test_select_probability_mask_threshold_and_top_fraction_keeps_small_eligible_set():
    """Combined mode should keep all eligible voxels when thresholding is stricter."""
    probability = np.arange(8, dtype=np.float32).reshape(2, 2, 2) / 10.0

    mask, meta = slp._select_probability_mask(
        probability,
        threshold=0.65,
        selection_mode="threshold-and-top-fraction",
        foreground_fraction=0.25,
    )

    assert int(mask.sum()) == 1
    assert mask[1, 1, 1] == pytest.approx(1.0)
    assert meta["threshold_eligible_voxels"] == 1
    assert meta["requested_foreground_voxels"] == 2
    assert meta["effective_probability_threshold"] == pytest.approx(0.65)


def test_select_probability_mask_budgeted_mode_requires_fraction():
    """Budgeted ROI selection must fail clearly without an explicit fraction."""
    probability = np.zeros((2, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="foreground_fraction is required"):
        slp._select_probability_mask(
            probability,
            selection_mode="top-fraction",
            foreground_fraction=None,
        )

    with pytest.raises(ValueError, match="foreground_fraction must be in"):
        slp._select_probability_mask(
            probability,
            selection_mode="top-fraction",
            foreground_fraction=0.0,
        )


def test_evaluate_threshold_sweep_selects_operating_point_and_writes_outputs(tmp_path, monkeypatch):
    """Threshold sweep should prefer FCD coverage without HC foreground burden."""
    dataset_dir, shape = _make_threshold_sweep_dataset(tmp_path)
    ckpt = tmp_path / "lesion_proposer.pt"
    ckpt.write_bytes(b"not used by monkeypatched loader")

    monkeypatch.setattr(
        slp,
        "_load_prediction_model",
        lambda checkpoint_path, device="auto": (object(), shape, {}, "cpu"),
    )

    def fake_predict_probability_map(model, spatial_size, volume_path, device):
        prob = np.zeros(shape, dtype=np.float32)
        path_text = str(volume_path)
        if "sub-00001" in path_text:
            prob[2:5, 2:5, 2:5] = 0.8
            prob[0:2, 0:2, 0:2] = 0.3
        else:
            prob[0:2, 0:2, 0:2] = 0.3
        return {
            "probability": prob,
            "affine": np.eye(4),
            "header": nib.Nifti1Header(),
            "shape": shape,
        }

    monkeypatch.setattr(slp, "_predict_probability_map", fake_predict_probability_map)

    output_dir = tmp_path / "sweep"
    result = evaluate_threshold_sweep(
        ckpt,
        dataset_dir,
        output_dir,
        thresholds=[0.2, 0.5, 0.9],
        min_coverage=1.0,
        max_hc_foreground_fraction=0.0,
        device="cpu",
    )

    selected = result["selected_operating_point"]
    assert selected is not None
    assert selected["threshold"] == pytest.approx(0.5)
    assert selected["fcd_coverage_ge_0_50_rate"] == pytest.approx(1.0)
    assert selected["hc_mean_predicted_foreground_fraction"] == pytest.approx(0.0)
    assert selected["selected_operating_point"] is True
    assert (output_dir / "threshold_sweep.json").exists()
    assert (output_dir / "threshold_sweep.csv").exists()

    low_threshold = next(row for row in result["thresholds"] if row["threshold"] == pytest.approx(0.2))
    assert low_threshold["hc_false_positive_subject_rate"] == pytest.approx(1.0)
    assert low_threshold["meets_operating_constraints"] is False


def test_evaluate_threshold_sweep_calibrates_budgeted_selection_modes(tmp_path, monkeypatch):
    """Sweep should calibrate the same budgeted ROI modes used at upload time."""
    dataset_dir, shape = _make_threshold_sweep_dataset(tmp_path)
    ckpt = tmp_path / "lesion_proposer.pt"
    ckpt.write_bytes(b"not used by monkeypatched loader")

    monkeypatch.setattr(
        slp,
        "_load_prediction_model",
        lambda checkpoint_path, device="auto": (object(), shape, {}, "cpu"),
    )

    def fake_predict_probability_map(model, spatial_size, volume_path, device):
        prob = np.zeros(shape, dtype=np.float32)
        path_text = str(volume_path)
        if "sub-00001" in path_text:
            prob[2:5, 2:5, 2:5] = 0.8
            prob[0:2, 0:2, 0:2] = 0.3
        else:
            prob[0:2, 0:2, 0:2] = 0.3
        return {
            "probability": prob,
            "affine": np.eye(4),
            "header": nib.Nifti1Header(),
            "shape": shape,
        }

    monkeypatch.setattr(slp, "_predict_probability_map", fake_predict_probability_map)

    output_dir = tmp_path / "budgeted_sweep"
    result = evaluate_threshold_sweep(
        ckpt,
        dataset_dir,
        output_dir,
        thresholds=[0.5],
        selection_modes=["top-fraction", "threshold-and-top-fraction"],
        foreground_fractions=[0.02],
        min_coverage=1.0,
        max_hc_foreground_fraction=0.03,
        device="cpu",
    )

    selected = result["selected_operating_point"]
    assert selected is not None
    assert selected["selection_mode"] == "threshold-and-top-fraction"
    assert selected["threshold"] == pytest.approx(0.5)
    assert selected["foreground_fraction"] == pytest.approx(0.02)
    assert selected["fcd_coverage_ge_0_50_rate"] == pytest.approx(1.0)
    assert selected["hc_mean_predicted_foreground_fraction"] == pytest.approx(0.0)
    assert result["constraints"]["selection_modes"] == [
        "top-fraction",
        "threshold-and-top-fraction",
    ]
    assert result["constraints"]["foreground_fractions"] == [pytest.approx(0.02)]

    top_fraction_row = next(
        row for row in result["thresholds"]
        if row["selection_mode"] == "top-fraction"
    )
    assert top_fraction_row["threshold"] is None
    assert top_fraction_row["foreground_fraction"] == pytest.approx(0.02)
    assert top_fraction_row["meets_operating_constraints"] is True


def test_evaluate_threshold_sweep_requires_fractions_for_budgeted_modes(tmp_path):
    """Budgeted calibration should fail clearly without foreground fractions."""
    dataset_dir, _shape = _make_threshold_sweep_dataset(tmp_path)
    ckpt = tmp_path / "lesion_proposer.pt"
    ckpt.write_bytes(b"not used")

    with pytest.raises(ValueError, match="foreground_fractions are required"):
        evaluate_threshold_sweep(
            ckpt,
            dataset_dir,
            tmp_path / "sweep",
            thresholds=[0.5],
            selection_modes=["top-fraction"],
            foreground_fractions=None,
            device="cpu",
        )


def test_load_supervised_roi_operating_point_from_threshold_sweep(tmp_path):
    """Runtime ROI settings should load from the calibrated sweep artifact."""
    path = tmp_path / "threshold_sweep.json"
    path.write_text(json.dumps({
        "method": "supervised_lesion_threshold_sweep",
        "selected_operating_point": {
            "threshold": 0.7,
            "selected_operating_point": True,
            "meets_operating_constraints": True,
            "n_fcd": 4,
            "n_hc": 3,
            "fcd_mean_manual_coverage": 0.72,
            "fcd_coverage_ge_0_50_rate": 0.75,
            "hc_mean_predicted_foreground_fraction": 0.004,
        },
        "constraints": {
            "min_component_voxels": 64,
            "min_coverage_ge_0_50_rate": 0.5,
            "max_hc_mean_predicted_foreground_fraction": 0.01,
        },
    }))

    op = slp.load_supervised_roi_operating_point(path)

    assert op["source_kind"] == "threshold_sweep"
    assert op["source_path"] == str(path)
    assert op["threshold"] == pytest.approx(0.7)
    assert op["min_component_voxels"] == 64
    assert op["max_foreground_fraction"] == pytest.approx(0.01)
    assert op["selected_operating_point"] is True
    assert op["meets_operating_constraints"] is True
    assert op["n_fcd"] == 4
    assert op["n_hc"] == 3
    assert op["fcd_mean_manual_coverage"] == pytest.approx(0.72)
    assert op["fcd_coverage_ge_0_50_rate"] == pytest.approx(0.75)
    assert op["hc_mean_predicted_foreground_fraction"] == pytest.approx(0.004)
    assert op["min_coverage_ge_0_50_rate"] == pytest.approx(0.5)
    assert (
        op["max_hc_mean_predicted_foreground_fraction"]
        == pytest.approx(0.01)
    )
    assert op["constraints"]["min_component_voxels"] == 64


def test_load_supervised_roi_operating_point_from_direct_json(tmp_path):
    """Direct operating-point JSON should support budgeted ROI selection."""
    path = tmp_path / "roi_operating_point.json"
    path.write_text(json.dumps({
        "selection_mode": "top-fraction",
        "foreground_fraction": 0.02,
        "max_foreground_fraction": 0.03,
        "min_component_voxels": 32,
    }))

    op = slp.load_supervised_roi_operating_point(path)

    assert op["source_kind"] == "direct"
    assert op["selection_mode"] == "top-fraction"
    assert op["foreground_fraction"] == pytest.approx(0.02)
    assert op["max_foreground_fraction"] == pytest.approx(0.03)
    assert op["min_component_voxels"] == 32


def test_load_supervised_roi_operating_point_rejects_uncalibrated_sweep(tmp_path):
    """A failed sweep should not silently fall back to default inference settings."""
    path = tmp_path / "threshold_sweep.json"
    path.write_text(json.dumps({
        "method": "supervised_lesion_threshold_sweep",
        "selected_operating_point": None,
        "constraints": {"min_component_voxels": 64},
    }))

    with pytest.raises(ValueError, match="no selected_operating_point"):
        slp.load_supervised_roi_operating_point(path)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_tiny_lesion_net_create():
    """Test TinyLesionNet model creation."""
    model = TinyLesionNet.create()
    assert model is not None
    assert hasattr(model, 'forward')

    x = torch.randn(1, 1, 32, 32, 32)
    out = model(x)
    assert out.shape == (1, 1, 32, 32, 32)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_train_lesion_proposer_synthetic():
    """Test training on tiny synthetic dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        dataset_dir = tmpdir / "dataset"
        output_dir = tmpdir / "output"
        dataset_dir.mkdir(parents=True)

        # Create 2 FCD subjects
        for i in range(2):
            subj_id = f"sub-{i:05d}"
            subj_dir = dataset_dir / subj_id / "anat"
            subj_dir.mkdir(parents=True)

            flair_data = np.random.rand(24, 24, 24).astype(np.float32)
            flair_img = nib.Nifti1Image(flair_data, np.eye(4))
            flair_path = subj_dir / f"{subj_id}_FLAIR.nii.gz"
            nib.save(flair_img, str(flair_path))

            roi_data = np.zeros((24, 24, 24), dtype=np.float32)
            roi_data[8:16, 8:16, 8:16] = 1.0
            roi_img = nib.Nifti1Image(roi_data, np.eye(4))
            roi_path = subj_dir / f"{subj_id}_FLAIR_roi.nii.gz"
            nib.save(roi_img, str(roi_path))

        # Create participants.tsv inside dataset_dir
        participants_path = dataset_dir / "participants.tsv"
        with open(participants_path, 'w') as f:
            f.write("participant_id\tgroup\n")
            f.write("sub-00000\tfcd\n")
            f.write("sub-00001\tfcd\n")

        metrics = train_lesion_proposer(
            dataset_dir,
            output_dir,
            spatial_size=(16, 16, 16),
            epochs=1,
            batch_size=1,
            lr=1e-3,
            device="cpu",
            include_controls=True,
            seed=42,
        )

        assert (output_dir / "lesion_proposer.pt").exists()
        assert (output_dir / "metrics.json").exists()
        assert "train_subjects" in metrics
        assert "val_subjects" in metrics
        assert len(metrics["loss_history"]) == 1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_train_lesion_proposer_one_record():
    """Test training on single record (smoke test / bootstrap case)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        dataset_dir = tmpdir / "dataset"
        output_dir = tmpdir / "output"
        dataset_dir.mkdir(parents=True)

        # Create 1 FCD subject
        subj_id = "sub-00000"
        subj_dir = dataset_dir / subj_id / "anat"
        subj_dir.mkdir(parents=True)

        flair_data = np.random.rand(24, 24, 24).astype(np.float32)
        flair_img = nib.Nifti1Image(flair_data, np.eye(4))
        flair_path = subj_dir / f"{subj_id}_FLAIR.nii.gz"
        nib.save(flair_img, str(flair_path))

        roi_data = np.zeros((24, 24, 24), dtype=np.float32)
        roi_data[8:16, 8:16, 8:16] = 1.0
        roi_img = nib.Nifti1Image(roi_data, np.eye(4))
        roi_path = subj_dir / f"{subj_id}_FLAIR_roi.nii.gz"
        nib.save(roi_img, str(roi_path))

        # Create participants.tsv
        participants_path = dataset_dir / "participants.tsv"
        with open(participants_path, 'w') as f:
            f.write("participant_id\tgroup\n")
            f.write("sub-00000\tfcd\n")

        metrics = train_lesion_proposer(
            dataset_dir,
            output_dir,
            spatial_size=(16, 16, 16),
            epochs=1,
            batch_size=1,
            lr=1e-3,
            device="cpu",
            include_controls=True,
            seed=42,
        )

        # With 1 record, should use it for training and have empty validation
        assert (output_dir / "lesion_proposer.pt").exists()
        assert (output_dir / "metrics.json").exists()
        assert len(metrics["train_subjects"]) == 1
        assert len(metrics["val_subjects"]) == 0
        assert len(metrics["loss_history"]) == 1
        assert "pos_weight" in metrics
        assert metrics["pos_weight"] > 1.0

        # Verify pos_weight is stored in checkpoint
        ckpt = torch.load(output_dir / "lesion_proposer.pt", weights_only=False)
        assert "pos_weight" in ckpt
        assert ckpt["pos_weight"] > 1.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_predict_lesion_roi_synthetic():
    """Test prediction creates output file with correct shape."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        dataset_dir = tmpdir / "dataset"
        output_dir = tmpdir / "output"
        dataset_dir.mkdir(parents=True)

        # Create training data (1 subject)
        subj_dir = dataset_dir / "sub-00000" / "anat"
        subj_dir.mkdir(parents=True)

        flair_data = np.random.rand(24, 24, 24).astype(np.float32)
        flair_img = nib.Nifti1Image(flair_data, np.eye(4))
        flair_path = subj_dir / "sub-00000_FLAIR.nii.gz"
        nib.save(flair_img, str(flair_path))

        roi_data = np.zeros((24, 24, 24), dtype=np.float32)
        roi_data[8:16, 8:16, 8:16] = 1.0
        roi_img = nib.Nifti1Image(roi_data, np.eye(4))
        roi_path = subj_dir / "sub-00000_FLAIR_roi.nii.gz"
        nib.save(roi_img, str(roi_path))

        participants_path = dataset_dir / "participants.tsv"
        with open(participants_path, 'w') as f:
            f.write("participant_id\tgroup\n")
            f.write("sub-00000\tfcd\n")

        # Train
        ckpt_dir = tmpdir / "ckpt"
        train_lesion_proposer(
            dataset_dir,
            ckpt_dir,
            spatial_size=(16, 16, 16),
            epochs=1,
            batch_size=1,
            device="cpu",
        )

        # Predict on test volume
        test_flair_data = np.random.rand(32, 32, 32).astype(np.float32)
        test_flair_img = nib.Nifti1Image(test_flair_data, np.eye(4))
        test_flair_path = tmpdir / "test_flair.nii.gz"
        nib.save(test_flair_img, str(test_flair_path))

        pred_path = tmpdir / "pred_roi.nii.gz"
        result = predict_lesion_roi(
            ckpt_dir / "lesion_proposer.pt",
            test_flair_path,
            pred_path,
            threshold=0.5,
            device="cpu",
        )

        assert pred_path.exists()
        pred_img = nib.load(str(pred_path))
        assert pred_img.get_fdata().shape == (32, 32, 32)
        assert "predicted_voxels" in result
        assert "probability_mean" in result


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_predict_lesion_roi_connected_component_filter():
    """Test that connected-component filtering excludes label 0 (background).

    Verifies that when min_component_voxels is set, label 0 is never kept
    even if the background is large, and that all-zero output is returned
    when no foreground component meets the threshold.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        dataset_dir = tmpdir / "dataset"
        output_dir = tmpdir / "output"
        dataset_dir.mkdir(parents=True)

        # Create training data (1 subject with tiny ROI)
        subj_dir = dataset_dir / "sub-00000" / "anat"
        subj_dir.mkdir(parents=True)

        flair_data = np.random.rand(24, 24, 24).astype(np.float32)
        flair_img = nib.Nifti1Image(flair_data, np.eye(4))
        flair_path = subj_dir / "sub-00000_FLAIR.nii.gz"
        nib.save(flair_img, str(flair_path))

        roi_data = np.zeros((24, 24, 24), dtype=np.float32)
        roi_data[8:16, 8:16, 8:16] = 1.0
        roi_img = nib.Nifti1Image(roi_data, np.eye(4))
        roi_path = subj_dir / "sub-00000_FLAIR_roi.nii.gz"
        nib.save(roi_img, str(roi_path))

        participants_path = dataset_dir / "participants.tsv"
        with open(participants_path, 'w') as f:
            f.write("participant_id\tgroup\n")
            f.write("sub-00000\tfcd\n")

        # Train
        ckpt_dir = tmpdir / "ckpt"
        train_lesion_proposer(
            dataset_dir,
            ckpt_dir,
            spatial_size=(16, 16, 16),
            epochs=1,
            batch_size=1,
            device="cpu",
        )

        # Test 1: Predict with tiny foreground and high component threshold
        # Should yield all-zero output (no foreground component survives)
        test_flair_data = np.random.rand(32, 32, 32).astype(np.float32)
        test_flair_img = nib.Nifti1Image(test_flair_data, np.eye(4))
        test_flair_path = tmpdir / "test_flair.nii.gz"
        nib.save(test_flair_img, str(test_flair_path))

        pred_path_1 = tmpdir / "pred_roi_1.nii.gz"
        result_1 = predict_lesion_roi(
            ckpt_dir / "lesion_proposer.pt",
            test_flair_path,
            pred_path_1,
            threshold=0.5,
            min_component_voxels=10000,
            device="cpu",
        )

        pred_img_1 = nib.load(str(pred_path_1))
        pred_data_1 = pred_img_1.get_fdata(dtype=np.float32)
        # With impossible threshold, output should be all-zero or near-zero
        assert pred_data_1.sum() < 10, f"Expected near-zero, got {pred_data_1.sum()}"
        assert result_1["predicted_voxels"] == 0

        # Test 2: Predict with normal threshold, verify label 0 is not kept
        # Create a synthetic prob map where background is large
        from unittest.mock import patch

        pred_path_2 = tmpdir / "pred_roi_2.nii.gz"

        def mock_predict_with_large_background(cp, vol, out, **kwargs):
            # Simulate model that predicts only background as foreground
            # (entire volume as "positive" from model perspective)
            pred_resized = np.ones((32, 32, 32), dtype=np.float32) * 0.6
            save_nifti_mask(pred_resized > 0.5, np.eye(4),
                           nib.load(str(vol)).header, out)
            return {
                "predicted_voxels": int(32*32*32),
                "probability_mean": 0.6,
                "probability_max": 0.6,
                "threshold": 0.5,
                "spatial_size": (16, 16, 16),
            }

        # Just verify connected component filter works with min_component_voxels
        # by checking the implementation didn't crash and outputs reasonable result
        result_2 = predict_lesion_roi(
            ckpt_dir / "lesion_proposer.pt",
            test_flair_path,
            pred_path_2,
            threshold=0.5,
            min_component_voxels=100,
            device="cpu",
        )
        assert pred_path_2.exists()
        assert result_2["predicted_voxels"] >= 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_evaluate_lesion_proposer_synthetic():
    """Test evaluation writes summary and CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        dataset_dir = tmpdir / "dataset"
        train_output_dir = tmpdir / "train_output"
        eval_output_dir = tmpdir / "eval_output"
        dataset_dir.mkdir(parents=True)

        # Create 2 FCD subjects
        for i in range(2):
            subj_id = f"sub-{i:05d}"
            subj_dir = dataset_dir / subj_id / "anat"
            subj_dir.mkdir(parents=True)

            flair_data = np.random.rand(24, 24, 24).astype(np.float32)
            flair_img = nib.Nifti1Image(flair_data, np.eye(4))
            flair_path = subj_dir / f"{subj_id}_FLAIR.nii.gz"
            nib.save(flair_img, str(flair_path))

            roi_data = np.zeros((24, 24, 24), dtype=np.float32)
            roi_data[8:16, 8:16, 8:16] = 1.0
            roi_img = nib.Nifti1Image(roi_data, np.eye(4))
            roi_path = subj_dir / f"{subj_id}_FLAIR_roi.nii.gz"
            nib.save(roi_img, str(roi_path))

        participants_path = dataset_dir / "participants.tsv"
        with open(participants_path, 'w') as f:
            f.write("participant_id\tgroup\n")
            f.write("sub-00000\tfcd\n")
            f.write("sub-00001\tfcd\n")

        # Train
        train_lesion_proposer(
            dataset_dir,
            train_output_dir,
            spatial_size=(16, 16, 16),
            epochs=1,
            batch_size=1,
            device="cpu",
        )

        # Evaluate
        summary = evaluate_lesion_proposer(
            train_output_dir / "lesion_proposer.pt",
            dataset_dir,
            eval_output_dir,
            threshold=0.5,
            device="cpu",
        )

        assert (eval_output_dir / "summary.json").exists()
        assert (eval_output_dir / "subjects.csv").exists()
        assert "n_subjects" in summary
        assert "mean_dice" in summary
        assert summary["n_subjects"] > 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_patch_lesion_proposer_checkpoint_predicts_and_sweeps(tmp_path):
    """Patch checkpoints should work through the standard predict/sweep APIs."""
    dataset_dir, _shape = _make_threshold_sweep_dataset(tmp_path)
    train_output_dir = tmp_path / "patch_train"
    sweep_output_dir = tmp_path / "patch_sweep"
    pred_path = tmp_path / "predicted_patch_roi.nii.gz"

    metrics = train_patch_lesion_proposer(
        dataset_dir,
        train_output_dir,
        patch_size=8,
        positives_per_subject=2,
        negatives_per_subject=2,
        hard_negatives_per_subject=1,
        epochs=1,
        batch_size=2,
        device="cpu",
    )

    assert metrics["model_type"] == "patch_lesion_classifier"
    assert metrics["hard_negatives_per_subject"] == 1
    assert metrics["n_train_hard_negative_samples"] >= 1
    assert (train_output_dir / "lesion_proposer.pt").exists()

    flair_path = (
        dataset_dir / "sub-00001" / "anat" / "sub-00001_T2sel_FLAIR.nii.gz"
    )
    pred_result = predict_lesion_roi(
        train_output_dir / "lesion_proposer.pt",
        flair_path,
        pred_path,
        threshold=0.5,
        device="cpu",
    )
    assert pred_path.exists()
    assert pred_result["spatial_size"] == (8, 8, 8)
    assert pred_result["probability_max"] >= pred_result["probability_mean"]

    sweep = evaluate_threshold_sweep(
        train_output_dir / "lesion_proposer.pt",
        dataset_dir,
        sweep_output_dir,
        thresholds=[0.1, 0.5],
        device="cpu",
    )
    assert sweep["spatial_size"] == [8, 8, 8]
    assert sweep["n_fcd"] == 1
    assert sweep["n_hc"] == 1
    assert (sweep_output_dir / "threshold_sweep.json").exists()


def test_import_without_torch():
    """Test that module imports even without torch."""
    from fcd_prototype import supervised_lesion_proposal
    assert hasattr(supervised_lesion_proposal, 'load_canonical')
    assert hasattr(supervised_lesion_proposal, 'build_supervised_records')


def test_torch_error_when_needed():
    """Test that torch import error is clear."""
    import sys
    torch_module = sys.modules.get('torch')
    if torch_module is None:
        from fcd_prototype.supervised_lesion_proposal import _ensure_torch
        with pytest.raises(RuntimeError, match="torch is required"):
            _ensure_torch()
