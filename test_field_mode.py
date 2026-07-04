#!/usr/bin/env python
"""Quick test of field-mode Gaussian residual."""
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import nibabel as nib

try:
    import torch
except ImportError:
    print("PyTorch not installed; skipping test")
    sys.exit(0)

from fcd_prototype.gaussian_residual import compare_gaussian_fields

def make_nifti(path, shape=(12, 12, 10), seed=0):
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal(shape).astype(np.float32) * 0.2 + 0.6).clip(0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.5, 1.0])), str(path))

def make_field_pt(path, n=40, seed=0, intensity_bias=0.0):
    rng = np.random.default_rng(seed)
    means_norm = rng.uniform(-0.8, 0.8, (n, 3)).astype(np.float32)
    log_scales = rng.uniform(-3.0, -1.0, (n, 3)).astype(np.float32)
    quat_wxyz = np.zeros((n, 4), dtype=np.float32)
    quat_wxyz[:, 0] = 1.0
    opacity_logit = rng.uniform(-1.0, 1.0, (n,)).astype(np.float32) + intensity_bias
    colors_raw = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)

    norm_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    norm_scale = np.float32(50.0)

    state = {
        "means_norm": torch.tensor(means_norm),
        "log_scales": torch.tensor(log_scales),
        "quat_wxyz": torch.tensor(quat_wxyz),
        "opacity_logit": torch.tensor(opacity_logit),
        "colors_raw": torch.tensor(colors_raw),
        "norm_center": norm_center,
        "norm_scale": float(norm_scale),
        "affine": np.diag([1.0, 1.0, 1.5, 1.0]).astype(np.float64),
        "num_gaussians": n,
        "backend": "fallback",
        "device": "cpu",
        "seed": seed,
        "jitter_voxels": False,
        "volume_path": "",
        "roi_path": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))

def test_field_mode_identical():
    """Test that identical fields produce near-zero residual."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        p_pt = tmpdir / "patient" / "field.pt"
        n_pt = tmpdir / "normal" / "field.pt"
        vol = tmpdir / "ref.nii.gz"

        make_field_pt(p_pt, n=30, seed=42)
        make_field_pt(n_pt, n=30, seed=42)
        make_nifti(vol)

        out = tmpdir / "out"
        metrics = compare_gaussian_fields(
            p_pt, n_pt, vol, out,
            heatmap_method="field",
            field_chunk_size=16,
        )

        assert metrics["heatmap_method"] == "field", f"Method: {metrics['heatmap_method']}"
        assert "field_residual_mean" in metrics, f"Keys: {metrics.keys()}"
        assert metrics["field_residual_mean"] < 0.01, f"Mean residual: {metrics['field_residual_mean']}"
        print(f"✓ Identical fields: residual_mean={metrics['field_residual_mean']:.6f}")

def test_field_mode_changed():
    """Test that changed intensity is detected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        n_pt = tmpdir / "normal" / "field.pt"
        p_pt = tmpdir / "patient" / "field.pt"
        vol = tmpdir / "ref.nii.gz"

        make_field_pt(n_pt, n=40, seed=50)
        make_field_pt(p_pt, n=40, seed=50, intensity_bias=4.0)
        make_nifti(vol)

        out = tmpdir / "out"
        metrics = compare_gaussian_fields(
            p_pt, n_pt, vol, out,
            heatmap_method="field",
            field_chunk_size=32,
        )

        assert metrics["field_residual_mean"] > 0.0, f"Mean: {metrics['field_residual_mean']}"
        assert metrics["field_residual_max"] > 0.001, f"Max: {metrics['field_residual_max']}"
        print(f"✓ Changed intensity detected: residual_mean={metrics['field_residual_mean']:.6f}, max={metrics['field_residual_max']:.6f}")

def test_scatter_mode_still_works():
    """Verify scatter mode still available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        p_pt = tmpdir / "patient" / "field.pt"
        n_pt = tmpdir / "normal" / "field.pt"
        vol = tmpdir / "ref.nii.gz"

        make_field_pt(p_pt, n=25, seed=60)
        make_field_pt(n_pt, n=25, seed=61)
        make_nifti(vol)

        out = tmpdir / "out"
        metrics = compare_gaussian_fields(
            p_pt, n_pt, vol, out,
            heatmap_method="scatter",
        )

        assert metrics["heatmap_method"] == "scatter", f"Method: {metrics['heatmap_method']}"
        assert "residual_p99" in metrics
        assert "field_residual_mean" not in metrics, "scatter should not have field_* keys"
        print(f"✓ Scatter mode works: p99={metrics['residual_p99']:.6f}")

if __name__ == "__main__":
    try:
        test_field_mode_identical()
        test_field_mode_changed()
        test_scatter_mode_still_works()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
