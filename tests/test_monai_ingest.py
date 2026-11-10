"""Tests for monai_ingest — verifies graceful missing-dependency handling."""

import importlib
import sys
import types
import unittest.mock as mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_fresh(monkeypatch) -> types.ModuleType:
    """Force a fresh import of monai_ingest with MONAI/torch hidden."""
    submod_keys = [k for k in sys.modules if k.startswith("monai.") or k.startswith("torch.")]

    monkeypatch.setitem(sys.modules, "monai", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    for key in submod_keys:
        monkeypatch.setitem(sys.modules, key, None)

    monkeypatch.delitem(sys.modules, "fcd_prototype.monai_ingest", raising=False)

    return importlib.import_module("fcd_prototype.monai_ingest")


def _import_real() -> types.ModuleType:
    """Return a freshly-loaded monai_ingest with real sys.modules (no patches)."""
    # Drop any cached copy that may have been imported while MONAI was hidden
    sys.modules.pop("fcd_prototype.monai_ingest", None)
    return importlib.import_module("fcd_prototype.monai_ingest")


# ---------------------------------------------------------------------------
# Tests: missing MONAI/torch → graceful failure
# ---------------------------------------------------------------------------

def test_monai_unavailable_flag(monkeypatch):
    mod = _import_fresh(monkeypatch)
    # _MONAI_AVAILABLE is now a function; call it while torch/monai are still hidden
    assert mod._MONAI_AVAILABLE() is False


def test_check_monai_exits_with_message(monkeypatch):
    import pytest
    mod = _import_fresh(monkeypatch)
    with pytest.raises(SystemExit) as exc_info:
        mod._check_monai()
    msg = str(exc_info.value)
    assert "monai" in msg.lower() or "torch" in msg.lower() or "pip" in msg.lower()


def test_help_works_without_monai(monkeypatch):
    """--help must exit 0 even when MONAI/torch are not installed."""
    import pytest
    mod = _import_fresh(monkeypatch)
    with pytest.raises(SystemExit) as exc_info:
        mod.main(["--help"])
    assert exc_info.value.code == 0


def test_main_exits_when_monai_absent(monkeypatch):
    import pytest
    mod = _import_fresh(monkeypatch)
    with pytest.raises(SystemExit):
        mod.main([
            "--dataset-dir", "/nonexistent",
            "--output-dir", "/tmp/out",
        ])


# ---------------------------------------------------------------------------
# Tests that run only when MONAI IS available
# (each test re-imports cleanly to avoid cache poisoning from the above tests)
# ---------------------------------------------------------------------------

try:
    import torch
    import monai
    _MONAI_PRESENT = True
except ImportError:
    _MONAI_PRESENT = False

import pytest


@pytest.mark.skipif(not _MONAI_PRESENT, reason="MONAI/torch not installed")
def test_build_transforms_returns_compose():
    from monai.transforms import Compose
    mod = _import_real()
    t = mod.build_transforms(["T1w"])
    assert isinstance(t, Compose)


@pytest.mark.skipif(not _MONAI_PRESENT, reason="MONAI/torch not installed")
def test_build_datalist_empty_dir(tmp_path):
    mod = _import_real()
    result = mod.build_datalist(tmp_path, modality_keys=["T1w"])
    assert result == []


# ---------------------------------------------------------------------------
# Tests for _resolve_subjects — no MONAI/torch required
# ---------------------------------------------------------------------------

def _get_resolve_subjects():
    mod = importlib.import_module("fcd_prototype.monai_ingest")
    return mod._resolve_subjects


def test_resolve_subjects_list():
    fn = _get_resolve_subjects()
    batch = {"subject": ["sub-001", "sub-002"]}
    assert fn(batch, 0, ["T1w"]) == ["sub-001", "sub-002"]


def test_resolve_subjects_tuple():
    fn = _get_resolve_subjects()
    batch = {"subject": ("sub-001", "sub-002")}
    assert fn(batch, 0, ["T1w"]) == ["sub-001", "sub-002"]


def test_resolve_subjects_scalar_string():
    fn = _get_resolve_subjects()
    batch = {"subject": "sub-001"}
    assert fn(batch, 0, ["T1w"]) == ["sub-001"]


def test_resolve_subjects_absent_with_tensor():
    fn = _get_resolve_subjects()
    fake_tensor = mock.MagicMock()
    fake_tensor.shape = (3,)
    batch = {"T1w": fake_tensor}
    result = fn(batch, 5, ["T1w"])
    assert result == ["batch5_0", "batch5_1", "batch5_2"]


def test_resolve_subjects_absent_no_tensor():
    fn = _get_resolve_subjects()
    result = fn({}, 2, ["T1w"])
    assert result == ["batch2_0"]


def test_resolve_subjects_no_unbound_i():
    fn = _get_resolve_subjects()
    result = fn({}, 0, [])
    assert result == ["batch0_0"]


@pytest.mark.skipif(not _MONAI_PRESENT, reason="MONAI/torch not installed")
def test_to_plain_tensor_strips_metatensor():
    """_to_plain_tensor must return a base torch.Tensor, not a subclass."""
    import torch
    mod = _import_real()
    t = torch.zeros(1, 4, 4, 4)
    # Simulate a MetaTensor subclass (MONAI's actual type)
    try:
        from monai.data.meta_tensor import MetaTensor
        meta_t = MetaTensor(t)
    except ImportError:
        pytest.skip("monai.data.meta_tensor not available")
    plain = mod._to_plain_tensor(meta_t)
    assert type(plain) is torch.Tensor, f"Expected torch.Tensor, got {type(plain)}"
    assert plain.shape == (1, 4, 4, 4)


@pytest.mark.skipif(not _MONAI_PRESENT, reason="MONAI/torch not installed")
def test_save_roundtrip_plain_tensors(tmp_path):
    """Saved .pt files must be loadable with weights_only=True (default in PyTorch 2.6+)."""
    import torch
    mod = _import_real()

    # Build a fake batch dict mimicking what MONAI DataLoader produces
    try:
        from monai.data.meta_tensor import MetaTensor
        t1 = MetaTensor(torch.zeros(1, 8, 8, 8))
        fl = MetaTensor(torch.ones(1, 8, 8, 8))
    except ImportError:
        pytest.skip("monai.data.meta_tensor not available")

    item = {
        "T1w": mod._to_plain_tensor(t1),
        "FLAIR": mod._to_plain_tensor(fl),
        "_meta": {"subject": "sub-001", "modalities": ["T1w", "FLAIR"], "source": "/data"},
    }
    out = tmp_path / "sub-001_tensors.pt"
    torch.save(item, out)

    loaded = torch.load(out, weights_only=True)
    assert type(loaded["T1w"]) is torch.Tensor
    assert type(loaded["FLAIR"]) is torch.Tensor
    assert loaded["T1w"].shape == (1, 8, 8, 8)
    assert loaded["_meta"]["subject"] == "sub-001"


@pytest.mark.skipif(not _MONAI_PRESENT, reason="MONAI/torch not installed")
def test_build_datalist_skips_missing_modality(tmp_path):
    import csv
    import nibabel as nib
    import numpy as np

    tsv = tmp_path / "participants.tsv"
    with tsv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["participant_id", "group"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"participant_id": "sub-00001", "group": "fcd"})

    anat = tmp_path / "sub-00001" / "anat"
    anat.mkdir(parents=True)
    data = np.zeros((16, 16, 16), dtype=np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(anat / "sub-00001_T1w.nii.gz"))

    mod = _import_real()
    result = mod.build_datalist(tmp_path, modality_keys=["T1w", "FLAIR"])
    assert result == []

    result = mod.build_datalist(tmp_path, modality_keys=["T1w"])
    assert len(result) == 1
