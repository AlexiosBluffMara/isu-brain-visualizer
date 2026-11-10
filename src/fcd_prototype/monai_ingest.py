"""MONAI ingestion layer for ds004199 NIfTI volumes.

MONAI and torch are optional. Importing this module is fast and side-effect-free;
heavy dependencies are only touched when actual ingestion is requested.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from monai.data import DataLoader
    from monai.transforms import Compose

VOLUME_SIZE = (256, 256, 256)
PIXDIM = (1.0, 1.0, 1.0)


def _MONAI_AVAILABLE() -> bool:
    """Return True iff torch and monai are importable right now."""
    try:
        import torch  # noqa: F401
        import monai  # noqa: F401
        return True
    except ImportError:
        return False


def _check_monai() -> None:
    """Lazily verify torch/MONAI are importable; sys.exit with instructions if not."""
    try:
        import torch  # noqa: F401
        import monai.data  # noqa: F401
        import monai.transforms  # noqa: F401
    except ImportError:
        sys.exit(
            "ERROR: MONAI and/or PyTorch are not installed.\n"
            "Install with:\n"
            "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128\n"
            "  pip install monai"
        )


def build_transforms(keys: list[str]) -> "Compose":
    _check_monai()
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        LoadImaged,
        NormalizeIntensityd,
        Orientationd,
        ResizeWithPadOrCropd,
        Spacingd,
    )
    return Compose([
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=PIXDIM, mode="bilinear"),
        NormalizeIntensityd(keys=keys, nonzero=True),
        ResizeWithPadOrCropd(keys=keys, spatial_size=VOLUME_SIZE),
    ])


def build_datalist(
    dataset_dir: Path,
    modality_keys: list[str],
) -> list[dict[str, str]]:
    """Build MONAI data list by scanning local ds004199 directory."""
    from fcd_prototype.bids_discovery import discover_dataset

    datalist = []
    for rec in discover_dataset(dataset_dir):
        item: dict[str, str] = {}
        skip = False
        for key in modality_keys:
            if key == "T1w":
                path = rec.t1w
            elif key == "FLAIR":
                path = rec.flair
            elif key == "ROI":
                path = rec.flair_roi
            else:
                path = None
            if path is None or not path.is_file():
                skip = True
                break
            item[key] = str(path)
        if not skip:
            item["subject"] = rec.subject_id
            datalist.append(item)
    return datalist


def build_dataloader(
    dataset_dir: Path,
    modality_keys: list[str],
    batch_size: int = 1,
    num_workers: int = 4,
) -> "DataLoader":
    _check_monai()
    import torch
    from monai.data import DataLoader, Dataset

    datalist = build_datalist(dataset_dir, modality_keys)
    if not datalist:
        raise ValueError(f"No valid subjects found under {dataset_dir}")

    transform_keys = [k for k in modality_keys if k != "subject"]
    transforms = build_transforms(transform_keys)

    dataset = Dataset(data=datalist, transform=transforms)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run MONAI ingestion pipeline on a local ds004199 subset."
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Root of the local ds004199 directory.",
    )
    p.add_argument(
        "--modalities",
        nargs="+",
        default=["T1w", "FLAIR"],
        metavar="MOD",
        help="Modality keys to load: T1w, FLAIR, ROI.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/monai_tensors"),
        help="Where to write processed tensors (.pt files).",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    return p


def _to_plain_tensor(t: "torch.Tensor") -> "torch.Tensor":
    """Strip MONAI MetaTensor wrapping → plain torch.Tensor, detached on CPU.

    MetaTensor is a torch.Tensor subclass whose pickle representation references
    monai.data.meta_tensor.MetaTensor.  PyTorch 2.6+ torch.load(weights_only=True)
    rejects that global.  .as_subclass(torch.Tensor) reinterprets the storage as
    a base tensor (zero-copy view) so pickle only sees torch.Tensor.
    """
    import torch
    return t.detach().cpu().as_subclass(torch.Tensor)


def _resolve_subjects(batch: dict, batch_idx: int, modality_keys: list[str]) -> list[str]:
    """Return a list of subject IDs for this batch, one entry per sample.

    MONAI's default collate may produce a list/tuple of strings, a single
    string (batch_size=1), or omit "subject" entirely.
    """
    raw = batch.get("subject")
    if raw is None:
        tensor_key = next((k for k in modality_keys if k in batch), None)
        n = batch[tensor_key].shape[0] if tensor_key is not None else 1
        return [f"batch{batch_idx}_{j}" for j in range(n)]
    if isinstance(raw, str):
        return [raw]
    return list(raw)


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    _check_monai()
    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)

    loader = build_dataloader(
        dataset_dir=args.dataset_dir,
        modality_keys=args.modalities,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Found {len(loader.dataset)} subjects. Processing…")
    for batch_idx, batch in enumerate(loader):
        subjects = _resolve_subjects(batch, batch_idx, args.modalities)
        for i, subject in enumerate(subjects):
            out_path = args.output_dir / f"{subject}_tensors.pt"
            item: dict = {
                k: _to_plain_tensor(batch[k][i])
                for k in args.modalities
                if k in batch
            }
            item["_meta"] = {
                "subject": subject,
                "modalities": [k for k in args.modalities if k in batch],
                "source": str(args.dataset_dir),
            }
            try:
                torch.save(item, out_path)
                print(f"Saved: {out_path}")
            except Exception as exc:
                print(f"WARNING: Failed to save {subject}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
