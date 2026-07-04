"""Tests for splat_ply_export — binary PLY schema and content validation."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fcd_prototype.splat_ply_export import (
    export_lesion_highlighted_splat_ply,
    export_splat_ply,
    _PLY_DTYPE,
    _stratified_weighted_sample_indices,
    _volume_opacity_from_intensity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_volume(path: Path, shape=(24, 24, 20), affine=None) -> None:
    if affine is None:
        affine = np.diag([1.0, 1.0, 1.5, 1.0])
    rng = np.random.default_rng(7)
    data = (rng.standard_normal(shape).astype(np.float32) * 300 + 800).clip(0, 4095)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


@pytest.fixture()
def vol_path(tmp_path: Path) -> Path:
    p = tmp_path / "vol.nii.gz"
    _make_volume(p)
    return p


# ---------------------------------------------------------------------------
# Header / schema tests
# ---------------------------------------------------------------------------

def test_ply_file_created(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    export_splat_ply(vol_path, out, max_points=200)
    assert out.exists()


def test_ply_header_magic(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    export_splat_ply(vol_path, out, max_points=200)
    with open(out, "rb") as fh:
        first_line = fh.readline()
    assert first_line.strip() == b"ply"


def test_ply_binary_little_endian_format(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    export_splat_ply(vol_path, out, max_points=200)
    with open(out, "rb") as fh:
        header = b""
        while True:
            line = fh.readline()
            header += line
            if line.strip() == b"end_header":
                break
    assert b"binary_little_endian" in header


def test_ply_has_expected_properties(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    export_splat_ply(vol_path, out, max_points=200)
    with open(out, "rb") as fh:
        header = b""
        while True:
            line = fh.readline()
            header += line
            if line.strip() == b"end_header":
                break
    header_str = header.decode("ascii")
    for prop in (
        "x", "y", "z", "red", "green", "blue", "opacity",
        "scale_x", "scale_y", "scale_z",
        "rot_w", "rot_x", "rot_y", "rot_z",
    ):
        assert f"property" in header_str and prop in header_str


def test_ply_vertex_count_matches_meta(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    meta = export_splat_ply(vol_path, out, max_points=200)
    n = meta["n_points"]
    with open(out, "rb") as fh:
        header = b""
        while True:
            line = fh.readline()
            header += line
            if line.strip() == b"end_header":
                break
    header_str = header.decode("ascii")
    for line in header_str.splitlines():
        if line.startswith("element vertex"):
            declared_n = int(line.split()[-1])
            break
    assert declared_n == n


# ---------------------------------------------------------------------------
# Content tests
# ---------------------------------------------------------------------------

def _read_ply_array(ply_path: Path) -> np.ndarray:
    """Parse the binary PLY and return structured array."""
    with open(ply_path, "rb") as fh:
        header_lines = []
        while True:
            line = fh.readline()
            header_lines.append(line.decode("ascii", errors="replace").strip())
            if header_lines[-1] == "end_header":
                break
        n = next(int(l.split()[-1]) for l in header_lines if l.startswith("element vertex"))
        raw = fh.read()
    arr = np.frombuffer(raw, dtype=_PLY_DTYPE, count=n)
    return arr


def test_coordinates_finite(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    export_splat_ply(vol_path, out, max_points=300)
    arr = _read_ply_array(out)
    assert np.all(np.isfinite(arr["x"]))
    assert np.all(np.isfinite(arr["y"]))
    assert np.all(np.isfinite(arr["z"]))


def test_point_count_bounded(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    meta = export_splat_ply(vol_path, out, max_points=150)
    assert meta["n_points"] <= 150


def test_nonzero_foreground_coverage(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    meta = export_splat_ply(vol_path, out, max_points=400)
    assert meta["n_points"] > 0


def test_deterministic_export(tmp_path, vol_path):
    """Two calls with same seed must produce identical PLY bytes."""
    out1 = tmp_path / "a.ply"
    out2 = tmp_path / "b.ply"
    export_splat_ply(vol_path, out1, max_points=200, rng_seed=42)
    export_splat_ply(vol_path, out2, max_points=200, rng_seed=42)
    assert out1.read_bytes() == out2.read_bytes()


def test_different_seed_different_result(tmp_path, vol_path):
    out1 = tmp_path / "a.ply"
    out2 = tmp_path / "b.ply"
    export_splat_ply(vol_path, out1, max_points=200, rng_seed=1)
    export_splat_ply(vol_path, out2, max_points=200, rng_seed=2)
    assert out1.read_bytes() != out2.read_bytes()


def test_meta_json_written(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    export_splat_ply(vol_path, out, max_points=200, subject="sub-00001", group="fcd")
    meta_path = tmp_path / "out_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["subject"] == "sub-00001"
    assert meta["group"] == "fcd"
    assert "n_points" in meta
    assert "world_bounds_min" in meta
    assert "world_bounds_max" in meta
    assert meta["gaussian_state_npz"] == "out_state.npz"
    assert meta["gaussian_state_schema"] == "mri_direct_splat_v1"
    contract = meta["coordinate_frame"]
    assert contract["schema"] == "mri_direct_splat_coordinate_frame_v1"
    assert contract["coordinate_frame"] == "canonical_ras_world_mm"
    assert contract["voxel_index_frame"] == "canonical_ijk_zero_based"
    assert contract["source_shape"] == [24, 24, 20]
    assert contract["source_orientation"] == ["R", "A", "S"]
    assert contract["voxel_to_world_max_abs_error_mm"] <= 1e-5
    assert contract["voxel_positions_in_bounds_fraction"] == pytest.approx(1.0)


def test_gaussian_state_npz_matches_direct_ply_schema(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    meta = export_splat_ply(vol_path, out, max_points=240, sampling_mode="volume")
    state_path = tmp_path / "out_state.npz"

    assert state_path.exists()
    with np.load(state_path) as state:
        assert str(state["state_schema"]) == "mri_direct_splat_v1"
        assert (
            str(state["coordinate_frame_schema"])
            == "mri_direct_splat_coordinate_frame_v1"
        )
        assert str(state["source_volume"]) == str(vol_path)
        assert str(state["sampling_mode"]) == "volume"
        assert state["positions_world"].shape == (meta["n_points"], 3)
        assert state["positions_voxel"].shape == (meta["n_points"], 3)
        assert state["scales"].shape == (meta["n_points"], 3)
        assert state["rotations_quat"].shape == (meta["n_points"], 4)
        assert state["opacity"].shape == (meta["n_points"],)
        assert state["intensity"].shape == (meta["n_points"],)
        assert state["color_rgb"].shape == (meta["n_points"], 3)
        assert state["source_orientation"].tolist() == ["R", "A", "S"]
        np.testing.assert_allclose(state["source_voxel_spacing_mm"], [1.0, 1.0, 1.5])
        np.testing.assert_allclose(
            np.linalg.norm(state["rotations_quat"], axis=1),
            1.0,
            atol=1e-5,
        )


def test_lesion_highlighted_export_adds_red_roi_gaussians(tmp_path, vol_path):
    roi = tmp_path / "roi.nii.gz"
    img = nib.load(str(vol_path))
    roi_data = np.zeros(img.shape[:3], dtype=np.uint8)
    roi_data[8:12, 9:13, 7:10] = 1
    nib.save(nib.Nifti1Image(roi_data, img.affine), str(roi))

    out = tmp_path / "lesion_highlighted.ply"
    meta = export_lesion_highlighted_splat_ply(
        vol_path,
        roi,
        out,
        max_anatomy_points=300,
        max_lesion_points=50,
        rng_seed=5,
        sampling_mode="volume",
    )
    arr = _read_ply_array(out)

    assert meta["lesion_highlight"] is True
    assert meta["n_anatomy_points"] <= 300
    assert meta["n_lesion_points"] == int(roi_data.sum())
    assert len(arr) == meta["n_anatomy_points"] + meta["n_lesion_points"]
    lesion_arr = arr[-meta["n_lesion_points"]:]
    assert np.all(lesion_arr["red"] == 255)
    assert np.all(lesion_arr["green"] >= 70)
    assert np.all(lesion_arr["opacity"] > 0.9)

    with np.load(tmp_path / "lesion_highlighted_state.npz") as state:
        assert state["is_lesion"].sum() == meta["n_lesion_points"]
        assert str(state["source_roi"]) == str(roi)


def test_opacity_in_unit_range(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    export_splat_ply(vol_path, out, max_points=300)
    arr = _read_ply_array(out)
    assert float(arr["opacity"].min()) >= 0.0
    assert float(arr["opacity"].max()) <= 1.0


def test_scale_positive(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    export_splat_ply(vol_path, out, max_points=300)
    arr = _read_ply_array(out)
    assert float(arr["scale_x"].min()) > 0
    assert float(arr["scale_y"].min()) > 0
    assert float(arr["scale_z"].min()) > 0


def test_direct_splat_exports_normalized_gradient_quaternions(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    export_splat_ply(vol_path, out, max_points=300)
    arr = _read_ply_array(out)
    q = np.stack([arr["rot_w"], arr["rot_x"], arr["rot_y"], arr["rot_z"]], axis=1)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)
    assert np.any(np.abs(arr["rot_x"]) > 1e-5) or np.any(np.abs(arr["rot_y"]) > 1e-5)
    assert np.all(arr["rot_w"] >= 0.0)


def test_surface_mode_exports_tangent_disk_scales(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    meta = export_splat_ply(vol_path, out, max_points=300, sampling_mode="surface")
    arr = _read_ply_array(out)
    assert meta["gaussian_orientation"] == "local_gradient_tangent_frame"
    assert meta["gaussian_scale_model"] == "gradient_tangent_disk"
    assert float(arr["scale_x"].mean()) > float(arr["scale_z"].mean())
    assert float(arr["scale_y"].mean()) > float(arr["scale_z"].mean())


# ---------------------------------------------------------------------------
# Surface-mode tests
# ---------------------------------------------------------------------------

def test_surface_mode_meta_records_mode(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    meta = export_splat_ply(vol_path, out, max_points=200, sampling_mode="surface")
    assert meta["sampling_mode"] == "surface"


def test_uniform_mode_meta_records_mode(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    meta = export_splat_ply(vol_path, out, max_points=200, sampling_mode="uniform")
    assert meta["sampling_mode"] == "uniform"


def test_volume_mode_meta_records_mode(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    meta = export_splat_ply(vol_path, out, max_points=200, sampling_mode="volume")
    assert meta["sampling_mode"] == "volume"
    assert meta["gaussian_scale_model"] == "coverage_balanced_density_adaptive_volume_kernel"
    assert meta["sampling_weight_model"] == "full_foreground_base_plus_intensity_gradient"
    assert meta["volume_sampling_strategy"] == "foreground_grid_stratified_weighted"
    assert meta["stratified_occupied_cells_total"] >= meta["stratified_cells_sampled"] > 0
    assert meta["foreground_mask_model"] == "largest_connected_tissue_component"
    assert meta["cleaned_foreground_voxels_total"] == meta["foreground_voxels_total"]
    assert meta["density_scale_factor"] >= 1.35
    assert meta["density_scale_factor"] <= 2.25
    assert meta["density_voxels_per_splat"] > 1.0


def test_stratified_volume_sampler_covers_occupied_cells_before_weight_bias():
    """A dense presentation budget should not collapse into one bright region."""
    coords = []
    weights = []
    for x_high in (0, 1):
        for y_high in (0, 1):
            for z_high in (0, 1):
                base = np.array([2 + x_high * 20, 2 + y_high * 20, 2 + z_high * 20])
                for offset in (
                    np.array([0, 0, 0]),
                    np.array([1, 0, 0]),
                    np.array([0, 1, 0]),
                    np.array([0, 0, 1]),
                ):
                    coords.append(base + offset)
                    weights.append(1000.0 if (x_high, y_high, z_high) == (1, 1, 1) else 1.0)
    fg_idx = np.asarray(coords, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()

    selected, meta = _stratified_weighted_sample_indices(
        fg_idx,
        weights,
        n=8,
        rng=np.random.default_rng(5),
        max_cells_per_axis=2,
    )

    selected_coords = fg_idx[selected]
    octants = {
        tuple(int(v) for v in (coord >= 12))
        for coord in selected_coords
    }
    assert len(octants) == 8
    assert meta["stratified_occupied_cells_total"] == 8
    assert meta["stratified_cells_sampled"] == 8


def test_surface_mode_meta_includes_surface_counts(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    meta = export_splat_ply(vol_path, out, max_points=200, sampling_mode="surface")
    assert "surface_voxels_total" in meta
    assert "edge_voxels_total" in meta
    assert meta["surface_voxels_total"] >= 0
    assert meta["edge_voxels_total"] >= 0


def test_surface_mode_meta_has_foreground_total(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    meta = export_splat_ply(vol_path, out, max_points=200, sampling_mode="surface")
    assert "foreground_voxels_total" in meta
    assert meta["foreground_voxels_total"] > 0


def test_surface_mode_meta_note_is_correct(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    meta = export_splat_ply(vol_path, out, max_points=200, sampling_mode="surface")
    assert "dense surface-biased" in meta["note"]
    assert "not optimized neural 3DGS" in meta["note"]


def test_surface_mode_deterministic(tmp_path, vol_path):
    out1 = tmp_path / "s1.ply"
    out2 = tmp_path / "s2.ply"
    export_splat_ply(vol_path, out1, max_points=200, rng_seed=42, sampling_mode="surface")
    export_splat_ply(vol_path, out2, max_points=200, rng_seed=42, sampling_mode="surface")
    assert out1.read_bytes() == out2.read_bytes()


def test_surface_mode_different_from_uniform(tmp_path, vol_path):
    out_s = tmp_path / "surface.ply"
    out_u = tmp_path / "uniform.ply"
    export_splat_ply(vol_path, out_s, max_points=200, rng_seed=42, sampling_mode="surface")
    export_splat_ply(vol_path, out_u, max_points=200, rng_seed=42, sampling_mode="uniform")
    assert out_s.read_bytes() != out_u.read_bytes()


def test_surface_mode_meaningful_surface_representation(tmp_path, vol_path):
    """Surface mode should place a meaningful fraction of points at surface shell."""
    out = tmp_path / "out.ply"
    meta = export_splat_ply(vol_path, out, max_points=500, sampling_mode="surface")
    # surface_voxels_total > 0 means the shell was non-empty
    assert meta.get("surface_voxels_total", 0) > 0


def test_surface_mode_opacity_range(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    export_splat_ply(vol_path, out, max_points=300, sampling_mode="surface")
    arr = _read_ply_array(out)
    # Surface mode uses 0.15 + intensity * 0.5; should not be all 1.0
    assert float(arr["opacity"].max()) <= 1.0
    assert float(arr["opacity"].min()) >= 0.0
    # Verify not degenerate all-ones (surface mode intentionally avoids opacity=1 for all)
    assert float(arr["opacity"].max()) < 1.0 or float(arr["opacity"].min()) < 0.65


def test_volume_mode_uses_soft_accumulation_opacity(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    export_splat_ply(vol_path, out, max_points=300, sampling_mode="volume")
    arr = _read_ply_array(out)
    assert float(arr["opacity"].min()) >= 0.005
    assert float(arr["opacity"].max()) <= 0.55


def test_volume_opacity_curve_suppresses_low_signal_padding():
    opacity = _volume_opacity_from_intensity(np.array([0.0, 0.05, 0.25, 1.0]))

    assert opacity[0] <= 0.011
    assert opacity[1] < opacity[2] < opacity[3]
    assert opacity[-1] <= 0.55


def test_invalid_sampling_mode_raises(tmp_path, vol_path):
    out = tmp_path / "out.ply"
    with pytest.raises(ValueError, match="sampling_mode"):
        export_splat_ply(vol_path, out, max_points=200, sampling_mode="invalid")


# ---------------------------------------------------------------------------
# Effective-threshold / tissue-mask tests
# ---------------------------------------------------------------------------

def test_uniform_mode_effective_threshold_is_requested(tmp_path, vol_path):
    """Uniform mode must not raise the threshold — effective == requested."""
    out = tmp_path / "out.ply"
    meta = export_splat_ply(vol_path, out, max_points=200, sampling_mode="uniform",
                            percentile_threshold=5.0)
    assert meta["percentile_threshold"] == 5.0
    assert meta["effective_percentile_threshold"] == 5.0


def test_volume_mode_uses_tissue_threshold_floor(tmp_path, vol_path):
    """Volume mode should avoid sampling near-zero acquisition padding."""
    out = tmp_path / "out.ply"
    meta = export_splat_ply(vol_path, out, max_points=200, sampling_mode="volume",
                            percentile_threshold=5.0)
    assert meta["percentile_threshold"] == 5.0
    assert meta["effective_percentile_threshold"] >= 20.0
    assert meta["volume_percentile_threshold_floor"] == 20.0
    assert meta["volume_intensity_threshold_floor"] == 0.08
    assert meta["foreground_threshold_intensity_floor"] == 0.08
    assert meta["foreground_threshold_value"] >= 0.08
    assert "surface_voxels_total" not in meta
    assert meta["volume_voxels_total"] == meta["foreground_voxels_total"]


def test_surface_mode_default_effective_threshold_gte_70(tmp_path, vol_path):
    """Surface mode with legacy percentile_threshold=5 must record effective >= 70."""
    out = tmp_path / "out.ply"
    meta = export_splat_ply(vol_path, out, max_points=200, sampling_mode="surface")
    assert meta["percentile_threshold"] == 5.0
    assert meta["effective_percentile_threshold"] >= 70.0


def test_surface_mode_explicit_surface_threshold_respected(tmp_path, vol_path):
    """Explicit surface_percentile_threshold=80 must be recorded as effective."""
    out = tmp_path / "out.ply"
    meta = export_splat_ply(vol_path, out, max_points=200, sampling_mode="surface",
                            surface_percentile_threshold=80.0)
    assert meta["effective_percentile_threshold"] == 80.0


def test_surface_mode_higher_caller_threshold_wins(tmp_path, vol_path):
    """If caller passes percentile_threshold > surface floor, that value wins."""
    out = tmp_path / "out.ply"
    meta = export_splat_ply(vol_path, out, max_points=200, sampling_mode="surface",
                            percentile_threshold=85.0)
    assert meta["effective_percentile_threshold"] == 85.0


def test_surface_mode_lower_caller_threshold_floored(tmp_path, vol_path):
    """If caller passes percentile_threshold < surface floor, floor takes effect."""
    out = tmp_path / "out.ply"
    meta = export_splat_ply(vol_path, out, max_points=200, sampling_mode="surface",
                            percentile_threshold=10.0)
    assert meta["effective_percentile_threshold"] >= 70.0
    assert meta["effective_percentile_threshold"] > meta["percentile_threshold"]
