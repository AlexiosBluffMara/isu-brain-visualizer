"""Tests for synthetic_normal."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from fcd_prototype.synthetic_normal import (
    NvidiaDependencyError,
    _roi_hybrid_counterfactual,
    _robust_normalize,
    generate_synthetic_normal,
    _check_nvidia_available,
    _infer_modality_code,
    _choose_generation_grid,
    _patient_centered_generated_affine,
    _generate_contralateral,
    _estimate_x_midline,
    _mirror_x_voxels,
)


def _make_volume(path: Path, shape=(24, 24, 20), seed=1, affine=None) -> None:
    if affine is None:
        affine = np.diag([1.0, 1.0, 1.5, 1.0])
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal(shape).astype(np.float32) * 0.2 + 0.6).clip(0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _make_roi(path: Path, shape=(24, 24, 20)) -> None:
    data = np.zeros(shape, dtype=np.uint8)
    data[8:14, 8:14, 7:13] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.5, 1.0])), str(path))


def _make_fake_nvidia_root(root: Path) -> None:
    """Create a minimal fake NV-Generate-CTMR repo structure."""
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)

    (root / "scripts" / "diff_model_infer.py").write_text("# fake")
    (root / "models" / "autoencoder_v1.pt").write_bytes(b"\x00")
    (root / "models" / "diff_unet_3d_rflow-mr-brain_v0.pt").write_bytes(b"\x00")

    env_cfg = {
        "data_base_dir": "./data",
        "model_dir": "./models",
        "model_filename": "diff_unet_3d_rflow-mr-brain_v0.pt",
        "output_dir": "./output",
        "output_prefix": "unet_3d",
        "trained_autoencoder_path": "models/autoencoder_v1.pt",
        "existing_ckpt_filepath": "./models/diff_unet_3d_rflow-mr-brain_v0.pt",
        "modality_mapping_path": "./configs/modality_mapping.json",
    }
    (root / "configs" / "environment_maisi_diff_model_rflow-mr-brain.json").write_text(
        json.dumps(env_cfg)
    )
    diff_cfg = {
        "diffusion_unet_inference": {
            "dim": [128, 128, 128],
            "spacing": [1.0, 1.0, 1.0],
            "random_seed": 42,
            "num_inference_steps": 30,
            "modality": 11,
            "cfg_guidance_scale": 10,
        }
    }
    (root / "configs" / "config_maisi_diff_model_rflow-mr-brain.json").write_text(
        json.dumps(diff_cfg)
    )
    (root / "configs" / "config_network_rflow.json").write_text("{}")
    modality_map = {"mri_flair": 11, "mri_t1": 9, "mri_t2": 10}
    (root / "configs" / "modality_mapping.json").write_text(json.dumps(modality_map))


# ---------------------------------------------------------------------------
# Placeholder mode tests (unchanged behaviour)
# ---------------------------------------------------------------------------

def test_placeholder_creates_output(tmp_path):
    vol = tmp_path / "vol.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_volume(vol)
    info = generate_synthetic_normal(vol, None, out, mode="placeholder")
    assert out.exists()
    assert info["synthetic_normal_source"] == "placeholder"
    assert info["synthetic_normal_is_diagnostic"] is False


def test_placeholder_same_shape_as_input(tmp_path):
    vol = tmp_path / "vol.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_volume(vol, shape=(20, 20, 16))
    generate_synthetic_normal(vol, None, out, mode="placeholder")
    orig = nib.load(str(vol))
    synth = nib.load(str(out))
    assert orig.shape == synth.shape


def test_placeholder_with_roi_inpaints_roi_region(tmp_path):
    """Placeholder must modify the ROI region (smooth fill ≠ identity)."""
    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_volume(vol)
    _make_roi(roi)
    generate_synthetic_normal(vol, roi, out, mode="placeholder")
    orig = nib.load(str(vol)).get_fdata(dtype=np.float32)
    synth = nib.load(str(out)).get_fdata(dtype=np.float32)
    roi_mask = nib.load(str(roi)).get_fdata() > 0
    assert not np.allclose(orig[roi_mask], synth[roi_mask])


def test_placeholder_notes_present(tmp_path):
    vol = tmp_path / "vol.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_volume(vol)
    info = generate_synthetic_normal(vol, None, out, mode="placeholder")
    assert isinstance(info["notes"], list) and len(info["notes"]) > 0
    assert any("PLACEHOLDER" in n or "placeholder" in n.lower() for n in info["notes"])


def test_placeholder_limitations_present(tmp_path):
    vol = tmp_path / "vol.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_volume(vol)
    info = generate_synthetic_normal(vol, None, out, mode="placeholder")
    assert isinstance(info["limitations"], list) and len(info["limitations"]) > 0


# ---------------------------------------------------------------------------
# Control-prior mode tests
# ---------------------------------------------------------------------------

def test_control_prior_requires_controls_with_roi(tmp_path):
    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_volume(vol)
    _make_roi(roi)

    with pytest.raises(ValueError, match="healthy-control"):
        generate_synthetic_normal(vol, roi, out, mode="control-prior")


def test_control_prior_no_roi_uses_whole_brain_control_prior(tmp_path):
    vol = tmp_path / "vol.nii.gz"
    control = tmp_path / "control.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_volume(vol, seed=4)
    _make_volume(control, seed=5)

    info = generate_synthetic_normal(
        vol,
        None,
        out,
        mode="control-prior",
        control_prior_volumes=[control],
    )

    assert info["synthetic_normal_source"] == "control_prior_whole_brain_prior"
    assert info["control_prior_scope"] == "whole_brain_no_roi"
    assert info["control_prior_count"] == 1
    assert info["control_prior_quality_gate"] in {
        "pass",
        "review",
        "insufficient_reference",
    }
    assert info["control_prior_reference_similarity_1_minus_mae"] is not None
    assert info["control_prior_per_control_quality"][0]["subject"] == "control"
    orig = nib.load(str(vol)).get_fdata(dtype=np.float32)
    orig_norm = _robust_normalize(orig)
    synth = nib.load(str(out)).get_fdata(dtype=np.float32)
    assert synth.shape == orig.shape
    assert not np.allclose(orig_norm, synth)


def test_control_prior_no_roi_without_controls_returns_patient_copy(tmp_path):
    vol = tmp_path / "vol.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_volume(vol, seed=4)

    info = generate_synthetic_normal(
        vol,
        None,
        out,
        mode="control-prior",
        control_prior_volumes=[],
    )

    assert info["synthetic_normal_source"] == "control_prior_no_roi_patient_copy"
    assert info["control_prior_count"] == 0
    orig = nib.load(str(vol)).get_fdata(dtype=np.float32)
    synth = nib.load(str(out)).get_fdata(dtype=np.float32)
    assert np.allclose(orig, synth)


def test_control_prior_empty_roi_uses_whole_brain_prior(tmp_path):
    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "empty_roi.nii.gz"
    control = tmp_path / "control.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_volume(vol, seed=14)
    _make_volume(control, seed=15)
    empty = np.zeros((24, 24, 20), dtype=np.uint8)
    nib.save(nib.Nifti1Image(empty, np.diag([1.0, 1.0, 1.5, 1.0])), str(roi))

    info = generate_synthetic_normal(
        vol,
        roi,
        out,
        mode="control-prior",
        control_prior_volumes=[control],
    )

    assert info["synthetic_normal_source"] == "control_prior_whole_brain_prior"
    assert "ROI mask exists" in info["control_prior_skip_reason"]
    orig_norm = _robust_normalize(nib.load(str(vol)).get_fdata(dtype=np.float32))
    synth = nib.load(str(out)).get_fdata(dtype=np.float32)
    assert not np.allclose(orig_norm, synth)


def test_control_prior_with_roi_creates_hybrid(tmp_path):
    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    c1 = tmp_path / "sub-control01" / "anat" / "sub-control01_FLAIR.nii.gz"
    c2 = tmp_path / "sub-control02" / "anat" / "sub-control02_FLAIR.nii.gz"
    out = tmp_path / "subject01" / "synthetic_normal.nii.gz"
    _make_volume(vol, seed=10)
    _make_roi(roi)
    _make_volume(c1, seed=11)
    _make_volume(c2, seed=12)

    info = generate_synthetic_normal(
        vol,
        roi,
        out,
        mode="control-prior",
        subject_id="sub-test01",
        control_prior_volumes=[c1, c2],
    )

    assert out.exists()
    assert info["synthetic_normal_source"] == "control_prior_roi_hybrid"
    assert info["control_prior_count"] == 2
    assert info["control_prior_subjects"] == ["sub-control01", "sub-control02"]
    assert info["control_prior_quality_gate_metric"] == "prior_reference_similarity_1_minus_mae"
    assert len(info["control_prior_per_control_quality"]) == 2
    orig = nib.load(str(vol)).get_fdata(dtype=np.float32)
    orig_norm = _robust_normalize(orig)
    synth = nib.load(str(out)).get_fdata(dtype=np.float32)
    roi_mask = nib.load(str(roi)).get_fdata(dtype=np.float32) > 0
    assert not np.allclose(orig_norm[roi_mask], synth[roi_mask])
    assert synth[0, 0, 0] == pytest.approx(orig_norm[0, 0, 0])


def test_control_prior_quality_gate_passes_for_matching_control(tmp_path):
    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    control = tmp_path / "sub-control-pass" / "anat" / "sub-control-pass_FLAIR.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_volume(vol, seed=42)
    _make_volume(control, seed=42)
    _make_roi(roi)

    info = generate_synthetic_normal(
        vol,
        roi,
        out,
        mode="control-prior",
        control_prior_volumes=[control],
    )

    assert info["control_prior_quality_gate"] == "pass"
    assert info["control_prior_reference_similarity_1_minus_mae"] == pytest.approx(1.0)
    assert info["control_prior_reference_mae"] == pytest.approx(0.0, abs=1e-6)
    assert info["control_prior_per_control_quality"][0]["similarity_1_minus_mae"] == pytest.approx(1.0)


def test_control_prior_resamples_affine_to_patient_grid(tmp_path):
    patient_affine = np.diag([1.0, 1.0, 1.5, 1.0])
    control_affine = np.diag([2.0, 2.0, 3.0, 1.0])
    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    control = tmp_path / "sub-control" / "anat" / "sub-control_FLAIR.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_volume(vol, shape=(24, 24, 20), seed=21, affine=patient_affine)
    _make_roi(roi, shape=(24, 24, 20))
    _make_volume(control, shape=(12, 12, 10), seed=22, affine=control_affine)

    info = generate_synthetic_normal(
        vol,
        roi,
        out,
        mode="control-prior",
        control_prior_volumes=[control],
    )

    synth_img = nib.load(str(out))
    patient_img = nib.load(str(vol))
    assert synth_img.shape == patient_img.shape
    assert np.allclose(synth_img.affine, patient_img.affine)
    assert info["control_prior_alignment"] == "canonical RAS affine-resampling to patient grid"


# ---------------------------------------------------------------------------
# NVIDIA missing-dependency tests (updated env var names)
# ---------------------------------------------------------------------------

def test_nvidia_missing_env_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_GENERATE_CTMR_ROOT", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_ROOT", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_WEIGHTS", raising=False)
    with pytest.raises(NvidiaDependencyError) as exc_info:
        _check_nvidia_available()
    msg = str(exc_info.value)
    assert "NVIDIA_GENERATE_CTMR_ROOT" in msg or "NVIDIA_MAISI_ROOT" in msg


def test_nvidia_missing_weights_raises(tmp_path, monkeypatch):
    """Legacy test: NVIDIA_MAISI_WEIGHTS with no valid path still raises."""
    monkeypatch.delenv("NVIDIA_GENERATE_CTMR_ROOT", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_ROOT", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_WEIGHTS", raising=False)
    with pytest.raises(NvidiaDependencyError):
        _check_nvidia_available()


def test_nvidia_missing_weights_in_generate(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_GENERATE_CTMR_ROOT", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_ROOT", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_WEIGHTS", raising=False)
    vol = tmp_path / "vol.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_volume(vol)
    with pytest.raises(NvidiaDependencyError):
        generate_synthetic_normal(vol, None, out, mode="nvidia")


def test_nvidia_nonexistent_dir_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_GENERATE_CTMR_ROOT", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_ROOT", raising=False)
    monkeypatch.setenv("NVIDIA_MAISI_WEIGHTS", str(tmp_path / "nonexistent"))
    with pytest.raises(NvidiaDependencyError):
        _check_nvidia_available()


def test_nvidia_error_message_has_install_instructions(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_GENERATE_CTMR_ROOT", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_ROOT", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_WEIGHTS", raising=False)
    try:
        _check_nvidia_available()
    except NvidiaDependencyError as e:
        assert "huggingface.co/nvidia" in str(e) or "NVIDIA_GENERATE_CTMR_ROOT" in str(e)


# ---------------------------------------------------------------------------
# NVIDIA no-GPU mocked tests (fake root + monkeypatched subprocess)
# ---------------------------------------------------------------------------

def _make_fake_generated_nifti(output_dir: Path, shape=(24, 24, 20)) -> Path:
    """Create a fake generated NIfTI in the output_dir as the official script would."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.random.default_rng(99).standard_normal(shape).astype(np.float32)
    path = output_dir / "unet_3d_generated.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def _fake_subprocess_run(fake_root: Path):
    """Return a monkeypatched subprocess.run that creates a fake NIfTI in nvidia_work/output."""
    def _run(cmd, cwd=None, capture_output=False, timeout=None, **kwargs):
        # Find the environment JSON from cmd args (official flag is --env_config)
        env_file = None
        for i, arg in enumerate(cmd):
            if arg in ("--env_config", "-e") and i + 1 < len(cmd):
                env_file = Path(cmd[i + 1])
        if env_file and env_file.is_file():
            with env_file.open() as fh:
                env_cfg = json.load(fh)
            out_dir = Path(env_cfg.get("output_dir", str(fake_root / "output")))
        else:
            out_dir = fake_root / "nvidia_work" / "output"
        _make_fake_generated_nifti(out_dir)

        class FakeResult:
            returncode = 0
        return FakeResult()

    return _run


def test_nvidia_mode_creates_outputs_with_roi(tmp_path, monkeypatch):
    """NVIDIA mode with ROI: subprocess mocked, check output files and metadata."""
    fake_root = tmp_path / "fake_nvidia"
    _make_fake_nvidia_root(fake_root)
    monkeypatch.delenv("NVIDIA_MAISI_ROOT", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_WEIGHTS", raising=False)
    monkeypatch.setenv("NVIDIA_GENERATE_CTMR_ROOT", str(fake_root))

    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    out = tmp_path / "subject01" / "synthetic_normal.nii.gz"
    _make_volume(vol, shape=(24, 24, 20))
    _make_roi(roi, shape=(24, 24, 20))

    with patch("fcd_prototype.synthetic_normal.subprocess.run", side_effect=_fake_subprocess_run(tmp_path / "subject01")):
        info = generate_synthetic_normal(
            patient_volume=vol,
            roi_mask=roi,
            output_path=out,
            mode="nvidia",
            modality_hint="FLAIR",
            subject_id="sub-test01",
            rng_seed=58058,
        )

    assert out.exists(), "synthetic_normal.nii.gz must be created"
    raw_prior = out.parent / "nvidia_raw_prior.nii.gz"
    assert raw_prior.exists(), "nvidia_raw_prior.nii.gz must be saved"

    assert info["synthetic_normal_source"] == "nvidia_rflow-mr-brain_roi_hybrid"
    assert info["synthetic_normal_is_diagnostic"] is False
    assert info["nvidia_modality_code"] == 11  # FLAIR
    assert info["nvidia_seed"] == 58058
    assert info["nvidia_prior_alignment"] == "patient_centered_fov_affine_resample"
    assert "nvidia_generation_fov_mm" in info
    assert "nvidia_patient_fov_mm" in info
    assert "nvidia_raw_prior_artifact" in info
    limitations_text = " ".join(info["limitations"]).lower()
    assert "nvidia generation" in limitations_text or "image-only" in limitations_text
    assert "not validated" in " ".join(info["limitations"]).lower()


def test_nvidia_mode_output_shape_matches_patient(tmp_path, monkeypatch):
    """Output NIfTI shape must match patient volume shape."""
    fake_root = tmp_path / "fake_nvidia"
    _make_fake_nvidia_root(fake_root)
    monkeypatch.setenv("NVIDIA_GENERATE_CTMR_ROOT", str(fake_root))
    monkeypatch.delenv("NVIDIA_MAISI_ROOT", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_WEIGHTS", raising=False)

    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    out = tmp_path / "sub" / "synthetic_normal.nii.gz"
    _make_volume(vol, shape=(32, 28, 24))
    _make_roi(roi, shape=(32, 28, 24))

    with patch("fcd_prototype.synthetic_normal.subprocess.run", side_effect=_fake_subprocess_run(tmp_path / "sub")):
        generate_synthetic_normal(vol, roi, out, mode="nvidia", rng_seed=1)

    patient_img = nib.load(str(vol))
    synth_img = nib.load(str(out))
    assert patient_img.shape == synth_img.shape


def test_nvidia_mode_no_roi_skips_subprocess(tmp_path, monkeypatch):
    """No ROI mask: subprocess must NOT be called, patient copy returned."""
    fake_root = tmp_path / "fake_nvidia"
    _make_fake_nvidia_root(fake_root)
    monkeypatch.setenv("NVIDIA_GENERATE_CTMR_ROOT", str(fake_root))
    monkeypatch.delenv("NVIDIA_MAISI_ROOT", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_WEIGHTS", raising=False)

    vol = tmp_path / "vol.nii.gz"
    out = tmp_path / "ctrl" / "synthetic_normal.nii.gz"
    _make_volume(vol)

    called = []
    def _spy_run(*args, **kwargs):
        called.append(True)

    with patch("fcd_prototype.synthetic_normal.subprocess.run", side_effect=_spy_run):
        info = generate_synthetic_normal(vol, None, out, mode="nvidia", rng_seed=1)

    assert not called, "subprocess.run must not be called for control (no ROI)"
    assert out.exists()
    assert "patient_copy" in info["synthetic_normal_source"]
    assert info["synthetic_normal_is_diagnostic"] is False


def test_nvidia_mode_uses_maisi_root_env(tmp_path, monkeypatch):
    """NVIDIA_MAISI_ROOT env var should also be accepted."""
    fake_root = tmp_path / "fake_nvidia"
    _make_fake_nvidia_root(fake_root)
    monkeypatch.delenv("NVIDIA_GENERATE_CTMR_ROOT", raising=False)
    monkeypatch.setenv("NVIDIA_MAISI_ROOT", str(fake_root))
    monkeypatch.delenv("NVIDIA_MAISI_WEIGHTS", raising=False)

    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    out = tmp_path / "s" / "synthetic_normal.nii.gz"
    _make_volume(vol)
    _make_roi(roi)

    with patch("fcd_prototype.synthetic_normal.subprocess.run", side_effect=_fake_subprocess_run(tmp_path / "s")):
        info = generate_synthetic_normal(vol, roi, out, mode="nvidia", rng_seed=1)

    assert info["synthetic_normal_source"] == "nvidia_rflow-mr-brain_roi_hybrid"


def test_nvidia_mode_maisi_weights_models_subdir(tmp_path, monkeypatch):
    """NVIDIA_MAISI_WEIGHTS pointing at models/ subdir should walk up to clone root."""
    fake_root = tmp_path / "fake_nvidia"
    _make_fake_nvidia_root(fake_root)
    monkeypatch.delenv("NVIDIA_GENERATE_CTMR_ROOT", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_ROOT", raising=False)
    monkeypatch.setenv("NVIDIA_MAISI_WEIGHTS", str(fake_root / "models"))

    root = _check_nvidia_available()
    assert root == fake_root


# ---------------------------------------------------------------------------
# Modality code inference
# ---------------------------------------------------------------------------

def test_modality_flair_is_11():
    assert _infer_modality_code("FLAIR") == 11
    assert _infer_modality_code("flair") == 11


def test_modality_t1w_is_9():
    assert _infer_modality_code("T1w") == 9
    assert _infer_modality_code("t1") == 9


def test_modality_t2w_is_10():
    assert _infer_modality_code("T2w") == 10


def test_modality_none_defaults_to_flair():
    assert _infer_modality_code(None) == 11


# ---------------------------------------------------------------------------
# Generation grid defaults
# ---------------------------------------------------------------------------

def test_default_grid_for_typical_fcd_flair(monkeypatch):
    monkeypatch.delenv("NVIDIA_MAISI_DIM", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_SPACING", raising=False)
    dim, spacing = _choose_generation_grid((160, 256, 256))
    assert dim == [128, 256, 256]
    assert spacing == [1.25, 1.0, 1.0]


def test_generation_grid_preserves_patient_fov_with_affine(monkeypatch):
    monkeypatch.delenv("NVIDIA_MAISI_DIM", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_SPACING", raising=False)
    affine = np.diag([0.8, 0.8, 0.8, 1.0])

    dim, spacing = _choose_generation_grid((208, 320, 320), affine)

    assert dim == [128, 256, 256]
    assert spacing == pytest.approx([1.3, 1.0, 1.0])
    fov = [d * s for d, s in zip(dim, spacing)]
    assert fov == pytest.approx([166.4, 256.0, 256.0])


def test_generation_grid_tiny_fov_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("NVIDIA_MAISI_DIM", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_SPACING", raising=False)
    affine = np.diag([1.0, 1.0, 1.5, 1.0])

    dim, spacing = _choose_generation_grid((24, 24, 20), affine)

    assert dim == [128, 256, 256]
    assert spacing == [1.25, 1.0, 1.0]


def test_env_override_grid(monkeypatch):
    monkeypatch.setenv("NVIDIA_MAISI_DIM", "128,256,256")
    monkeypatch.setenv("NVIDIA_MAISI_SPACING", "1.25,1,1")
    dim, spacing = _choose_generation_grid((160, 256, 256))
    assert dim == [128, 256, 256]
    assert spacing == [1.25, 1.0, 1.0]


def test_patient_centered_generated_affine_aligns_centers():
    patient_affine = np.diag([1.0, 1.0, 1.5, 1.0])
    patient_shape = (160, 256, 256)
    generated_shape = (128, 256, 256)
    dim = [128, 256, 256]
    spacing = [1.25, 1.0, 1.5]

    gen_affine, effective_spacing = _patient_centered_generated_affine(
        patient_affine,
        patient_shape,
        generated_shape,
        dim,
        spacing,
    )

    patient_center = (
        patient_affine @ np.array([(160 - 1) / 2, (256 - 1) / 2, (256 - 1) / 2, 1.0])
    )[:3]
    generated_center = (
        gen_affine @ np.array([(128 - 1) / 2, (256 - 1) / 2, (256 - 1) / 2, 1.0])
    )[:3]
    assert generated_center == pytest.approx(patient_center)
    assert effective_spacing == pytest.approx([1.25, 1.0, 1.5])


# ---------------------------------------------------------------------------
# Subprocess command flag regression test
# ---------------------------------------------------------------------------

def test_nvidia_subprocess_uses_official_flags(tmp_path, monkeypatch):
    """diff_model_infer.py must be called with -e/--env_config, --model_config, --model_def.
    Regression guard: wrong flags (--environment-file, --config-file, --network-config) must not appear.
    """
    fake_root = tmp_path / "fake_nvidia"
    _make_fake_nvidia_root(fake_root)
    monkeypatch.setenv("NVIDIA_GENERATE_CTMR_ROOT", str(fake_root))
    monkeypatch.delenv("NVIDIA_MAISI_ROOT", raising=False)
    monkeypatch.delenv("NVIDIA_MAISI_WEIGHTS", raising=False)

    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    out = tmp_path / "s" / "synthetic_normal.nii.gz"
    _make_volume(vol)
    _make_roi(roi)

    captured_cmd = []

    def _capture_run(cmd, cwd=None, capture_output=False, timeout=None, **kwargs):
        captured_cmd.extend(cmd)
        # Still need to create the fake output NIfTI
        env_file = None
        for i, arg in enumerate(cmd):
            if arg in ("--env_config", "-e") and i + 1 < len(cmd):
                env_file = Path(cmd[i + 1])
        if env_file and env_file.is_file():
            with env_file.open() as fh:
                env_cfg = json.load(fh)
            out_dir = Path(env_cfg.get("output_dir", str(tmp_path / "s" / "nvidia_work" / "output")))
        else:
            out_dir = tmp_path / "s" / "nvidia_work" / "output"
        _make_fake_generated_nifti(out_dir)

        class FakeResult:
            returncode = 0
        return FakeResult()

    with patch("fcd_prototype.synthetic_normal.subprocess.run", side_effect=_capture_run):
        generate_synthetic_normal(vol, roi, out, mode="nvidia", rng_seed=1)

    assert captured_cmd, "subprocess.run was never called"

    # Module mode: "-m" and "scripts.diff_model_infer" must appear as adjacent args
    assert "-m" in captured_cmd, f"Missing -m flag; got: {captured_cmd}"
    m_idx = captured_cmd.index("-m")
    assert m_idx + 1 < len(captured_cmd) and captured_cmd[m_idx + 1] == "scripts.diff_model_infer", \
        f"'-m' must be immediately followed by 'scripts.diff_model_infer'; got: {captured_cmd}"

    # Official flags
    assert "--env_config" in captured_cmd, f"Missing --env_config flag; got: {captured_cmd}"
    assert "--model_config" in captured_cmd, f"Missing --model_config flag; got: {captured_cmd}"
    assert "--model_def" in captured_cmd, f"Missing --model_def flag; got: {captured_cmd}"

    # All three config paths must be absolute (so they resolve correctly when cwd=nvidia_root)
    for flag in ("--env_config", "--model_config", "--model_def"):
        flag_idx = captured_cmd.index(flag)
        path_val = captured_cmd[flag_idx + 1]
        assert Path(path_val).is_absolute(), \
            f"Value after {flag} must be an absolute path; got: {path_val!r}"

    # env_config output_dir must also be absolute
    env_flag_idx = captured_cmd.index("--env_config")
    env_file = Path(captured_cmd[env_flag_idx + 1])
    with env_file.open() as fh:
        loaded_env = json.load(fh)
    assert Path(loaded_env["output_dir"]).is_absolute(), \
        f"output_dir in env config must be absolute; got: {loaded_env['output_dir']!r}"

    # Old wrong flags must not appear
    assert "--environment-file" not in captured_cmd, "Old flag --environment-file must not appear"
    assert "--config-file" not in captured_cmd, "Old flag --config-file must not appear"
    assert "--network-config" not in captured_cmd, "Old flag --network-config must not appear"
    # Direct file path invocation must not appear
    assert not any("diff_model_infer.py" in arg for arg in captured_cmd), \
        f"Direct file path invocation must not appear; got: {captured_cmd}"


# ---------------------------------------------------------------------------
# ROI hybrid: voxels well outside ROI must be unchanged
# ---------------------------------------------------------------------------

def test_roi_hybrid_outside_support_unchanged():
    """Voxels well outside the dilated ROI support must equal the patient exactly."""
    rng = np.random.default_rng(7)
    shape = (40, 40, 40)
    patient = rng.random(shape).astype(np.float32)
    prior = rng.random(shape).astype(np.float32)

    # Small ROI in the center
    roi = np.zeros(shape, dtype=np.float32)
    roi[18:22, 18:22, 18:22] = 1.0

    result = _roi_hybrid_counterfactual(patient, prior, roi, feather_sigma=2.0, dilation_iterations=3)

    # Voxels far from the ROI (e.g. corners) must be exactly the patient value
    corners = [
        (0, 0, 0), (0, 0, -1), (0, -1, 0), (-1, 0, 0),
        (0, -1, -1), (-1, 0, -1), (-1, -1, 0), (-1, -1, -1),
    ]
    for idx in corners:
        assert result[idx] == patient[idx], \
            f"Corner voxel {idx} changed: patient={patient[idx]:.6f}, result={result[idx]:.6f}"


# ---------------------------------------------------------------------------
# Contralateral mode tests
# ---------------------------------------------------------------------------

def _make_asymmetric_volume(path: Path, shape=(40, 40, 30), seed=77) -> None:
    """Create a volume with a clear left-right asymmetry (bright spot on left side)."""
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal(shape).astype(np.float32) * 0.15 + 0.5).clip(0.0, 1.0)
    # Bright spot on left hemisphere (low x) to simulate lesion signal
    data[4:10, 16:24, 10:20] = 0.9
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.0, 1.0])), str(path))


def _make_left_roi(path: Path, shape=(40, 40, 30)) -> None:
    """ROI on the left hemisphere (low x-axis voxels)."""
    data = np.zeros(shape, dtype=np.uint8)
    data[4:10, 16:24, 10:20] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.0, 1.0])), str(path))


# --- dispatch ---

def test_contralateral_dispatch_via_generate_synthetic_normal(tmp_path):
    """generate_synthetic_normal(mode='contralateral') must dispatch to _generate_contralateral."""
    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_asymmetric_volume(vol)
    _make_left_roi(roi)
    info = generate_synthetic_normal(vol, roi, out, mode="contralateral")
    assert info["mode"] == "contralateral"
    assert info["synthetic_normal_source"] == "contralateral_patient_specific_roi_completion"
    assert out.exists()


# --- ROI-positive case: anatomy outside ROI preserved, ROI changes toward mirror ---

def test_contralateral_roi_positive_changes_roi_toward_mirror(tmp_path):
    """ROI voxels must shift toward the mirrored contralateral value; corners must be unchanged."""
    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_asymmetric_volume(vol, shape=(40, 40, 30))
    _make_left_roi(roi, shape=(40, 40, 30))

    info = generate_synthetic_normal(vol, roi, out, mode="contralateral")

    orig_img = nib.load(str(vol))
    orig_data = orig_img.get_fdata(dtype=np.float32)
    synth_data = nib.load(str(out)).get_fdata(dtype=np.float32)
    roi_mask = nib.load(str(roi)).get_fdata(dtype=np.float32) > 0

    # ROI region must have changed (bright lesion signal replaced by mirror side)
    assert not np.allclose(orig_data[roi_mask], synth_data[roi_mask], atol=1e-4), \
        "ROI voxels must differ after contralateral substitution"

    # Corners well outside the ROI+dilation support must be exactly preserved
    corners = [(0, 0, 0), (0, 0, -1), (0, -1, 0)]
    from fcd_prototype.synthetic_normal import _robust_normalize
    orig_norm = _robust_normalize(orig_data)
    for idx in corners:
        assert synth_data[idx] == pytest.approx(orig_norm[idx], abs=1e-5), \
            f"Corner {idx} changed: orig={orig_norm[idx]:.6f}, synth={synth_data[idx]:.6f}"


def test_contralateral_roi_positive_output_shape_matches_patient(tmp_path):
    """Output shape must match patient volume shape."""
    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_asymmetric_volume(vol, shape=(40, 40, 30))
    _make_left_roi(roi, shape=(40, 40, 30))
    generate_synthetic_normal(vol, roi, out, mode="contralateral")
    assert nib.load(str(vol)).shape == nib.load(str(out)).shape


def test_contralateral_roi_positive_output_not_diagnostic(tmp_path):
    """synthetic_normal_is_diagnostic must be False."""
    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_asymmetric_volume(vol)
    _make_left_roi(roi)
    info = generate_synthetic_normal(vol, roi, out, mode="contralateral")
    assert info["synthetic_normal_is_diagnostic"] is False


# --- metadata keys for ROI-positive case ---

def test_contralateral_roi_positive_metadata_keys(tmp_path):
    """ROI-positive contralateral result must include all required metadata keys."""
    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_asymmetric_volume(vol)
    _make_left_roi(roi)
    info = generate_synthetic_normal(vol, roi, out, mode="contralateral", subject_id="sub-test99")

    assert info["mode"] == "contralateral"
    assert isinstance(info["roi_voxel_count"], int) and info["roi_voxel_count"] > 0
    assert 0.0 <= info["mirrored_voxel_fraction"] <= 1.0
    assert 0.0 <= info["fallback_voxel_fraction"] <= 1.0
    assert pytest.approx(info["mirrored_voxel_fraction"] + info["fallback_voxel_fraction"], abs=1e-4) == 1.0
    assert info["midline_voxel_index"] is not None
    assert isinstance(info["limitations"], list) and len(info["limitations"]) >= 3
    assert any("not validated" in lim.lower() for lim in info["limitations"])
    assert any("contralateral" in lim.lower() or "symmetric" in lim.lower()
               for lim in info["limitations"])
    assert info["subject_id"] == "sub-test99"


# --- empty-ROI skip path ---

def test_contralateral_empty_roi_skips_safely(tmp_path):
    """Empty ROI (all zeros) must return a patient copy with the correct skip source label."""
    vol = tmp_path / "vol.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_volume(vol)
    # Write an ROI file that is all zeros
    empty_roi = tmp_path / "empty_roi.nii.gz"
    zeros = np.zeros((24, 24, 20), dtype=np.uint8)
    nib.save(nib.Nifti1Image(zeros, np.diag([1.0, 1.0, 1.5, 1.0])), str(empty_roi))

    info = generate_synthetic_normal(vol, empty_roi, out, mode="contralateral")

    assert out.exists()
    assert info["synthetic_normal_source"].startswith("contralateral_skipped_empty_roi"), \
        f"Expected source to start with 'contralateral_skipped_empty_roi', got: {info['synthetic_normal_source']!r}"
    assert info["mode"] == "contralateral"
    assert info["roi_voxel_count"] == 0
    assert info["mirrored_voxel_fraction"] == 0.0
    assert info["fallback_voxel_fraction"] == 0.0
    assert info["midline_voxel_index"] is None
    assert isinstance(info["limitations"], list) and len(info["limitations"]) > 0


def test_contralateral_empty_roi_output_is_patient_copy(tmp_path):
    """For empty ROI, the saved NIfTI must be close to the normalized patient volume."""
    vol = tmp_path / "vol.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_volume(vol, seed=3)
    empty_roi = tmp_path / "empty_roi.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((24, 24, 20), dtype=np.uint8),
                              np.diag([1.0, 1.0, 1.5, 1.0])), str(empty_roi))
    generate_synthetic_normal(vol, empty_roi, out, mode="contralateral")
    orig_norm = _robust_normalize(nib.load(str(vol)).get_fdata(dtype=np.float32))
    synth = nib.load(str(out)).get_fdata(dtype=np.float32)
    assert np.allclose(orig_norm, synth, atol=1e-5)


# --- no-ROI skip path ---

def test_contralateral_no_roi_skips_safely(tmp_path):
    """No ROI mask must return a patient copy with the correct skip source label."""
    vol = tmp_path / "vol.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_volume(vol, seed=5)

    info = generate_synthetic_normal(vol, None, out, mode="contralateral")

    assert out.exists()
    assert info["synthetic_normal_source"].startswith("contralateral_skipped_no_roi"), \
        f"Expected source to start with 'contralateral_skipped_no_roi', got: {info['synthetic_normal_source']!r}"
    assert info["mode"] == "contralateral"
    assert info["roi_voxel_count"] == 0
    assert info["mirrored_voxel_fraction"] == 0.0
    assert info["fallback_voxel_fraction"] == 0.0
    assert info["midline_voxel_index"] is None
    assert isinstance(info["limitations"], list) and len(info["limitations"]) > 0


def test_contralateral_no_roi_output_is_patient_copy(tmp_path):
    """For no ROI, the saved NIfTI must be close to the normalized patient volume."""
    vol = tmp_path / "vol.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_volume(vol, seed=6)
    generate_synthetic_normal(vol, None, out, mode="contralateral")
    orig_norm = _robust_normalize(nib.load(str(vol)).get_fdata(dtype=np.float32))
    synth = nib.load(str(out)).get_fdata(dtype=np.float32)
    assert np.allclose(orig_norm, synth, atol=1e-5)


# --- helper unit tests ---

def test_estimate_x_midline_symmetric_volume():
    """Symmetric volume centroid should be close to the centre voxel."""
    shape = (40, 30, 20)
    rng = np.random.default_rng(42)
    data = rng.random(shape).astype(np.float32)
    mask = np.ones(shape, dtype=bool)
    midline = _estimate_x_midline(data, mask)
    assert midline == pytest.approx((shape[0] - 1) / 2.0, abs=1.0)


def test_estimate_x_midline_empty_mask_fallback():
    """Empty foreground mask must fall back to (shape[0]-1)/2."""
    shape = (30, 20, 15)
    data = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)
    midline = _estimate_x_midline(data, mask)
    assert midline == pytest.approx((shape[0] - 1) / 2.0)


def test_mirror_x_voxels_symmetry():
    """Mirroring twice must recover the original coordinates."""
    coords = np.array([5, 10, 15, 20], dtype=np.int32)
    midline = 19.5
    mirrored = _mirror_x_voxels(coords, midline)
    re_mirrored = _mirror_x_voxels(mirrored.astype(np.int32), midline)
    # After two reflections we should recover the original (within float rounding)
    assert np.allclose(re_mirrored, coords.astype(np.float64), atol=0.5)


def test_contralateral_direct_call_returns_dict(tmp_path):
    """Direct _generate_contralateral call must return a dict with mode='contralateral'."""
    vol = tmp_path / "vol.nii.gz"
    roi = tmp_path / "roi.nii.gz"
    out = tmp_path / "synth.nii.gz"
    _make_asymmetric_volume(vol)
    _make_left_roi(roi)
    result = _generate_contralateral(
        patient_volume=vol,
        roi_mask=roi,
        output_path=out,
        subject_id="sub-direct-test",
    )
    assert isinstance(result, dict)
    assert result["mode"] == "contralateral"
    assert result["synthetic_normal_is_diagnostic"] is False
    assert out.exists()
