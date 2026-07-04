"""Supervised lesion proposal baseline for FCD ROI detection.

This module provides a trainable 3D CNN-based approach to predict FCD lesion
ROIs from FLAIR MRI volumes, trained on ds004199 manual ROI labels.

Entry point: fcd-lesion-proposer
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional
import warnings

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, label as nd_label
from scipy.interpolate import RegularGridInterpolator

from fcd_prototype.bids_discovery import discover_dataset


_PREDICTION_SELECTION_MODES = (
    "threshold",
    "top-fraction",
    "threshold-and-top-fraction",
)

_SELECTION_MODE_SORT_RANK = {
    "threshold-and-top-fraction": 0,
    "threshold": 1,
    "top-fraction": 2,
}


def _ensure_torch() -> None:
    """Raise clear error if torch is not installed."""
    try:
        import torch
    except ImportError:
        raise RuntimeError(
            "torch is required for supervised lesion proposal. Install via:\n"
            "  pip install 'fcd-prototype[gaussian]'\n"
            "  or: pip install torch>=2.0"
        )


def load_canonical(path: Path | str) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header, nib.Nifti1Image]:
    """Load NIfTI to canonical orientation.

    Returns:
        (data, affine, header, canonical_img) as float32
    """
    path = Path(path)
    img = nib.load(str(path))
    canonical_img = nib.as_closest_canonical(img)
    data = canonical_img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data, canonical_img.affine.copy(), canonical_img.header.copy(), canonical_img


def _robust_normalize(data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Robust z-score normalize using percentiles."""
    if mask is not None:
        fg = data[mask]
    else:
        fg = data[data > 0]

    if fg.size == 0:
        return np.zeros_like(data)

    p_low, p_high = np.percentile(fg, [2, 98])
    if p_high == p_low:
        # If all values in foreground are equal, return binary/nonzero mask-like array
        result = np.zeros_like(data)
        if mask is not None:
            result[mask] = 1.0
        else:
            result[data > 0] = 1.0
        return result

    norm = (data - p_low) / (p_high - p_low)
    return np.clip(norm, 0, 1)


def _foreground_mask(data: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Simple intensity threshold foreground mask."""
    norm = _robust_normalize(data)
    rough = norm > threshold
    labeled, n = nd_label(rough)
    if n == 0:
        return rough
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax()).astype(bool)


def validate_spatial_size(spatial_size: tuple[int, int, int]) -> None:
    """Validate that spatial_size is divisible by 4 (two MaxPool3d operations).

    Raises:
        ValueError: If any dimension is not divisible by 4.
    """
    for i, dim in enumerate(spatial_size):
        if dim % 4 != 0:
            raise ValueError(
                f"spatial_size dimension {i} ({dim}) must be divisible by 4 "
                f"(TinyLesionNet uses two MaxPool3d layers with stride 2). "
                f"Use spatial_size divisible by 4 (e.g., 64, 128, 256)."
            )


def resize_array(
    data: np.ndarray,
    target_shape: tuple[int, int, int],
    order: int = 1,
) -> np.ndarray:
    """Resize array to target shape via linear (order=1) or nearest (order=0) interpolation."""
    if data.shape == target_shape:
        return data.copy()

    current_shape = data.shape

    coords = np.stack(np.meshgrid(
        np.linspace(0, current_shape[0] - 1, target_shape[0]),
        np.linspace(0, current_shape[1] - 1, target_shape[1]),
        np.linspace(0, current_shape[2] - 1, target_shape[2]),
        indexing='ij'
    ), axis=-1)

    points = coords.reshape(-1, 3)

    if order == 0:
        points = np.round(points).astype(int)
        points = np.clip(points, 0, np.array(current_shape) - 1)
        resized = data[points[:, 0], points[:, 1], points[:, 2]]
    else:
        interp = RegularGridInterpolator(
            (np.arange(current_shape[0]), np.arange(current_shape[1]), np.arange(current_shape[2])),
            data,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        resized = interp(points)

    return resized.reshape(target_shape)


def save_nifti_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header,
    path: Path,
) -> None:
    """Save mask as float32 NIfTI with given affine/header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(mask.astype(np.float32), affine, header)
    img.header.set_data_dtype(np.float32)
    nib.save(img, str(path))


def _component_filter(pred_bin: np.ndarray, min_component_voxels: int = 0) -> np.ndarray:
    """Remove connected components smaller than *min_component_voxels*."""
    pred_bin = (pred_bin > 0).astype(np.float32)
    if min_component_voxels <= 0:
        return pred_bin

    labeled, n = nd_label(pred_bin)
    if n <= 0:
        return np.zeros_like(pred_bin, dtype=np.float32)

    sizes = np.bincount(labeled.ravel())
    valid_labels = np.where(sizes >= min_component_voxels)[0]
    valid_labels = valid_labels[valid_labels > 0]
    if valid_labels.size == 0:
        return np.zeros_like(pred_bin, dtype=np.float32)
    return np.isin(labeled, valid_labels).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_supervised_records(
    dataset_dir: Path | str,
    subjects: Optional[list[str]] = None,
    include_controls: bool = True,
) -> list[dict]:
    """Build training records from ds004199.

    Returns list of dicts with keys:
        subject_id, group, flair_path, roi_path (None for HC or missing ROI)
    """
    dataset_dir = Path(dataset_dir)
    records = []

    subject_set = None
    if subjects is not None:
        subject_set = set(subjects)

    for rec in discover_dataset(dataset_dir):
        if subject_set is not None and rec.subject_id not in subject_set:
            continue

        if not rec.flair:
            continue

        roi_path = None
        group = rec.group.lower() if rec.group else "unknown"

        if group == "fcd":
            if not rec.flair_roi:
                continue
            roi_path = rec.flair_roi
        elif group == "hc":
            if not include_controls:
                continue
            roi_path = None
        else:
            if not include_controls:
                continue
            roi_path = rec.flair_roi if rec.flair_roi else None

        records.append({
            "subject_id": rec.subject_id,
            "group": group,
            "flair_path": str(rec.flair),
            "roi_path": str(roi_path) if roi_path else None,
        })

    return records


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class TinyLesionNet:
    """Simple 3D CNN for lesion ROI prediction. Defined in plain Python, instantiated via torch."""

    @staticmethod
    def create():
        """Create model instance. Must import torch inside."""
        _ensure_torch()
        import torch
        import torch.nn as nn

        class _TinyLesionNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc1 = nn.Sequential(
                    nn.Conv3d(1, 16, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(16, 32, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.pool1 = nn.MaxPool3d(2)

                self.enc2 = nn.Sequential(
                    nn.Conv3d(32, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.pool2 = nn.MaxPool3d(2)

                self.bottleneck = nn.Sequential(
                    nn.Conv3d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(128, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                )

                self.up2 = nn.ConvTranspose3d(64, 64, 2, stride=2)
                self.dec2 = nn.Sequential(
                    nn.Conv3d(128, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(64, 32, 3, padding=1),
                    nn.ReLU(inplace=True),
                )

                self.up1 = nn.ConvTranspose3d(32, 32, 2, stride=2)
                self.dec1 = nn.Sequential(
                    nn.Conv3d(64, 32, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(32, 16, 3, padding=1),
                    nn.ReLU(inplace=True),
                )

                self.out = nn.Conv3d(16, 1, 1)

            def forward(self, x):
                e1 = self.enc1(x)
                p1 = self.pool1(e1)

                e2 = self.enc2(p1)
                p2 = self.pool2(e2)

                b = self.bottleneck(p2)

                u2 = self.up2(b)
                d2 = self.dec2(torch.cat([u2, e2], dim=1))

                u1 = self.up1(d2)
                d1 = self.dec1(torch.cat([u1, e1], dim=1))

                return self.out(d1)

        return _TinyLesionNet()


class PatchLesionNet:
    """Small 3D patch classifier for ROI-free lesion candidate proposal."""

    @staticmethod
    def create():
        _ensure_torch()
        import torch.nn as nn

        class _PatchLesionNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv3d(1, 8, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(2),
                    nn.Conv3d(8, 16, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(2),
                    nn.Conv3d(16, 32, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool3d(1),
                )
                self.classifier = nn.Linear(32, 1)

            def forward(self, x):
                x = self.features(x).flatten(1)
                return self.classifier(x).squeeze(1)

        return _PatchLesionNet()


def _extract_centered_patch(
    data: np.ndarray,
    center: np.ndarray | tuple[int, int, int],
    patch_size: int,
) -> np.ndarray:
    """Extract a cubic patch with zero padding at image boundaries."""
    center = np.asarray(center, dtype=int)
    half = patch_size // 2
    patch = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
    src_slices = []
    dst_slices = []
    for axis, c in enumerate(center):
        src_start = max(0, c - half)
        src_end = min(data.shape[axis], c - half + patch_size)
        dst_start = src_start - (c - half)
        dst_end = dst_start + (src_end - src_start)
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))
    patch[tuple(dst_slices)] = data[tuple(src_slices)]
    return patch


def _lesion_like_hard_negative_score(
    data: np.ndarray,
    eligible_mask: np.ndarray,
) -> np.ndarray:
    """Score non-lesion foreground that resembles lesion candidates."""
    eligible = np.asarray(eligible_mask, dtype=bool)
    if data.size == 0 or not eligible.any():
        return np.zeros_like(data, dtype=np.float32)

    norm = _robust_normalize(data).astype(np.float32)
    smooth = gaussian_filter(norm, sigma=1.0).astype(np.float32)
    smooth_norm = _robust_normalize(smooth, eligible).astype(np.float32)
    gradients = np.gradient(smooth)
    grad_mag = np.sqrt(sum(np.square(g, dtype=np.float32) for g in gradients))
    grad_norm = _robust_normalize(grad_mag, eligible).astype(np.float32)
    score = (0.75 * smooth_norm) + (0.25 * grad_norm)
    score = score.astype(np.float32)
    score[~eligible] = 0.0
    return score


def _sample_hard_negative_centers(
    data: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
    pool_factor: int = 8,
    min_separation_voxels: int = 0,
) -> np.ndarray:
    """Choose high-scoring non-lesion centers while avoiding one tight cluster."""
    count = int(count)
    if count <= 0:
        return np.zeros((0, 3), dtype=int)

    eligible = np.asarray(eligible_mask, dtype=bool)
    coords = np.argwhere(eligible)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=int)

    score = _lesion_like_hard_negative_score(data, eligible)
    coord_scores = score[tuple(coords.T)].astype(np.float64)
    finite = np.isfinite(coord_scores)
    if not finite.any():
        return np.zeros((0, 3), dtype=int)

    coords = coords[finite]
    coord_scores = coord_scores[finite]
    if coords.size == 0:
        return np.zeros((0, 3), dtype=int)

    jitter = rng.random(coord_scores.shape[0]) * 1e-7
    rank_scores = coord_scores + jitter
    pool_size = min(
        coords.shape[0],
        max(count, int(count) * max(1, int(pool_factor))),
    )
    if pool_size >= coords.shape[0]:
        candidate_idxs = np.argsort(-rank_scores)
    else:
        top_idxs = np.argpartition(rank_scores, -pool_size)[-pool_size:]
        candidate_idxs = top_idxs[np.argsort(-rank_scores[top_idxs])]

    min_sep = max(0, int(min_separation_voxels))
    min_dist_sq = float(min_sep * min_sep)
    selected: list[np.ndarray] = []
    selected_keys: set[tuple[int, int, int]] = set()

    for idx in candidate_idxs:
        center = coords[int(idx)].astype(int)
        key = tuple(int(v) for v in center)
        if key in selected_keys:
            continue
        if min_sep > 0 and any(
            float(np.sum((center - prev) ** 2)) < min_dist_sq for prev in selected
        ):
            continue
        selected.append(center)
        selected_keys.add(key)
        if len(selected) >= count:
            break

    if len(selected) < count:
        fallback_idxs = np.argsort(-rank_scores)
        for idx in fallback_idxs:
            center = coords[int(idx)].astype(int)
            key = tuple(int(v) for v in center)
            if key in selected_keys:
                continue
            selected.append(center)
            selected_keys.add(key)
            if len(selected) >= count:
                break

    if not selected:
        return np.zeros((0, 3), dtype=int)
    return np.asarray(selected, dtype=int)


def _interior_center_mask(shape: tuple[int, int, int], margin: int) -> np.ndarray:
    """Return centers far enough from borders to reduce padding artifacts."""
    mask = np.ones(shape, dtype=bool)
    margin = max(0, int(margin))
    if margin <= 0:
        return mask

    for axis, dim in enumerate(shape):
        if dim <= margin * 2:
            continue
        slicer = [slice(None)] * 3
        slicer[axis] = slice(0, margin)
        mask[tuple(slicer)] = False
        slicer[axis] = slice(dim - margin, dim)
        mask[tuple(slicer)] = False
    return mask


def _sample_patch_centers(
    records: list[dict],
    *,
    patch_size: int,
    positives_per_subject: int,
    negatives_per_subject: int,
    seed: int,
    hard_negatives_per_subject: int = 0,
    hard_negative_pool_factor: int = 8,
    hard_negative_min_separation_voxels: Optional[int] = None,
) -> list[dict]:
    """Sample positive lesion-centered and negative foreground patch centers."""
    rng = np.random.default_rng(seed)
    samples: list[dict] = []
    hard_negative_min_separation_voxels = (
        max(1, patch_size // 4)
        if hard_negative_min_separation_voxels is None
        else int(hard_negative_min_separation_voxels)
    )

    for rec in records:
        vol, affine, _header, img = load_canonical(rec["flair_path"])
        fg = _foreground_mask(vol)
        fg_coords = np.argwhere(fg)
        if fg_coords.size == 0:
            continue

        roi_bin = None
        if rec.get("roi_path"):
            roi, roi_affine, _roi_header, _roi_img = load_canonical(rec["roi_path"])
            if roi.shape != vol.shape or not np.allclose(roi_affine, affine):
                roi = resample_from_to(
                    nib.Nifti1Image((roi > 0.5).astype(np.float32), roi_affine),
                    img,
                    order=0,
                ).get_fdata(dtype=np.float32)
            roi_bin = roi > 0.5
            roi_coords = np.argwhere(roi_bin)
            if roi_coords.size > 0:
                take = min(positives_per_subject, len(roi_coords))
                idxs = rng.choice(len(roi_coords), size=take, replace=len(roi_coords) < take)
                for idx in np.atleast_1d(idxs):
                    samples.append({
                        "subject_id": rec["subject_id"],
                        "flair_path": rec["flair_path"],
                        "center": roi_coords[int(idx)].astype(int).tolist(),
                        "label": 1.0,
                        "sample_type": "positive",
                    })

        negative_mask = fg.copy()
        if roi_bin is not None:
            guard = binary_dilation(roi_bin, iterations=max(1, patch_size // 8))
            negative_mask &= ~guard

        hard_negative_mask = negative_mask & _interior_center_mask(
            vol.shape,
            margin=max(1, patch_size // 2),
        )
        if not hard_negative_mask.any():
            hard_negative_mask = negative_mask
        hard_centers = _sample_hard_negative_centers(
            vol,
            hard_negative_mask,
            count=hard_negatives_per_subject,
            rng=rng,
            pool_factor=hard_negative_pool_factor,
            min_separation_voxels=hard_negative_min_separation_voxels,
        )
        hard_center_keys = {
            tuple(int(v) for v in center)
            for center in hard_centers
        }
        for center in hard_centers:
            samples.append({
                "subject_id": rec["subject_id"],
                "flair_path": rec["flair_path"],
                "center": center.astype(int).tolist(),
                "label": 0.0,
                "sample_type": "negative_hard",
            })

        negative_coords = np.argwhere(negative_mask)
        if hard_center_keys and negative_coords.size > 0:
            keep = [
                tuple(int(v) for v in coord) not in hard_center_keys
                for coord in negative_coords
            ]
            negative_coords = negative_coords[np.asarray(keep, dtype=bool)]
        if negative_coords.size == 0:
            negative_coords = fg_coords
        take = min(negatives_per_subject, len(negative_coords))
        idxs = rng.choice(
            len(negative_coords),
            size=take,
            replace=len(negative_coords) < take,
        )
        for idx in np.atleast_1d(idxs):
            samples.append({
                "subject_id": rec["subject_id"],
                "flair_path": rec["flair_path"],
                "center": negative_coords[int(idx)].astype(int).tolist(),
                "label": 0.0,
                "sample_type": "negative_random",
            })

    rng.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lesion_proposer(
    dataset_dir: Path | str,
    output_dir: Path | str,
    subjects: Optional[list[str]] = None,
    val_subjects: Optional[list[str]] = None,
    spatial_size: tuple[int, int, int] = (64, 64, 64),
    epochs: int = 1,
    batch_size: int = 1,
    lr: float = 1e-3,
    device: str = "auto",
    include_controls: bool = True,
    seed: int = 42,
    max_train_subjects: Optional[int] = None,
) -> dict:
    """Train lesion proposer on supervised dataset.

    Returns:
        metrics dict with loss history, val_metrics, train_subjects, val_subjects
    """
    _ensure_torch()
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_spatial_size(spatial_size)

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    records = build_supervised_records(dataset_dir, subjects, include_controls)

    if not records:
        raise ValueError("No training records found")

    # Handle single-record case: use for training, leave validation empty
    if len(records) == 1 and val_subjects is None:
        train_records = records
        val_records = []
    elif val_subjects is None:
        np.random.shuffle(records)
        val_idx = list(range(0, len(records), 5))
        val_records = [records[i] for i in val_idx if i < len(records)]
        train_records = [r for i, r in enumerate(records) if i not in val_idx]
    else:
        val_set = set(val_subjects)
        train_records = [r for r in records if r["subject_id"] not in val_set]
        val_records = [r for r in records if r["subject_id"] in val_set]

    if not train_records:
        raise ValueError("No training records found")

    if max_train_subjects is not None and len(train_records) > max_train_subjects:
        train_records = train_records[:max_train_subjects]

    class SupervisedDataset(Dataset):
        def __init__(self, records, spatial_size):
            self.records = records
            self.spatial_size = spatial_size

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            rec = self.records[idx]
            vol, _, _, _ = load_canonical(rec["flair_path"])
            vol_norm = _robust_normalize(vol)
            vol_resized = resize_array(vol_norm, self.spatial_size, order=1)

            if rec["roi_path"]:
                roi, _, _, _ = load_canonical(rec["roi_path"])
                roi = (roi > 0.5).astype(np.float32)
                roi_resized = resize_array(roi, self.spatial_size, order=0)
            else:
                roi_resized = np.zeros(self.spatial_size, dtype=np.float32)

            vol_tensor = torch.from_numpy(vol_resized[None, ...].astype(np.float32))
            roi_tensor = torch.from_numpy(roi_resized[None, ...].astype(np.float32))

            return vol_tensor, roi_tensor

    train_ds = SupervisedDataset(train_records, spatial_size)
    val_ds = SupervisedDataset(val_records, spatial_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TinyLesionNet.create()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def dice_loss(pred_logits, target):
        pred = torch.sigmoid(pred_logits)
        smooth = 1.0
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1.0 - (2.0 * intersection + smooth) / (union + smooth)

    pos_weight = 1.0
    if len(train_records) > 0:
        total_voxels = spatial_size[0] * spatial_size[1] * spatial_size[2] * len(train_records)
        pos_voxels = 0
        for r in train_records:
            if r["roi_path"]:
                roi, _, _, _ = load_canonical(r["roi_path"])
                roi_binary = (roi > 0.5).astype(np.float32)
                roi_resized = resize_array(roi_binary, spatial_size, order=0)
                pos_voxels += int((roi_resized > 0.5).sum())

        if pos_voxels > 0:
            neg_voxels = total_voxels - pos_voxels
            pos_weight = max(1.0, min(neg_voxels / pos_voxels, 2000.0))
        else:
            pos_weight = 1.0

    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    metrics = {
        "train_subjects": [r["subject_id"] for r in train_records],
        "val_subjects": [r["subject_id"] for r in val_records],
        "spatial_size": spatial_size,
        "pos_weight": float(pos_weight),
        "loss_history": [],
        "val_metrics": [],
    }

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for vol_batch, roi_batch in train_loader:
            vol_batch = vol_batch.to(device)
            roi_batch = roi_batch.to(device)

            pred_logits = model(vol_batch)

            loss = bce_loss(pred_logits, roi_batch) + 0.5 * dice_loss(pred_logits, roi_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        metrics["loss_history"].append(avg_loss)

        model.eval()
        val_dice_list = []
        with torch.no_grad():
            for vol_batch, roi_batch in val_loader:
                vol_batch = vol_batch.to(device)
                roi_batch = roi_batch.to(device)

                pred_logits = model(vol_batch)
                pred = torch.sigmoid(pred_logits)
                pred_bin = (pred > 0.5).float()

                intersection = (pred_bin * roi_batch).sum(dim=(2, 3, 4))
                union = (pred_bin.sum(dim=(2, 3, 4)) + roi_batch.sum(dim=(2, 3, 4)))
                dice = 2.0 * intersection / (union + 1.0)
                val_dice_list.extend(dice.cpu().numpy().flatten())

        avg_val_dice = np.mean(val_dice_list) if val_dice_list else 0.0
        metrics["val_metrics"].append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_dice": float(avg_val_dice),
        })

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "spatial_size": spatial_size,
        "threshold": 0.5,
        "train_subjects": metrics["train_subjects"],
        "val_subjects": metrics["val_subjects"],
        "pos_weight": float(pos_weight),
        "metrics": metrics["val_metrics"],
    }

    ckpt_path = output_dir / "lesion_proposer.pt"
    torch.save(checkpoint, str(ckpt_path))

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def train_patch_lesion_proposer(
    dataset_dir: Path | str,
    output_dir: Path | str,
    subjects: Optional[list[str]] = None,
    val_subjects: Optional[list[str]] = None,
    patch_size: int = 32,
    positives_per_subject: int = 8,
    negatives_per_subject: int = 16,
    hard_negatives_per_subject: int = 0,
    hard_negative_pool_factor: int = 8,
    hard_negative_min_separation_voxels: Optional[int] = None,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "auto",
    include_controls: bool = True,
    seed: int = 42,
    max_train_subjects: Optional[int] = None,
) -> dict:
    """Train a patch classifier that preserves small FCD lesion signal."""
    _ensure_torch()
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if patch_size <= 0 or patch_size % 4 != 0:
        raise ValueError("patch_size must be positive and divisible by 4")
    if positives_per_subject <= 0:
        raise ValueError("positives_per_subject must be positive")
    if negatives_per_subject <= 0:
        raise ValueError("negatives_per_subject must be positive")
    if hard_negatives_per_subject < 0:
        raise ValueError("hard_negatives_per_subject must be non-negative")
    if hard_negative_pool_factor <= 0:
        raise ValueError("hard_negative_pool_factor must be positive")
    if (
        hard_negative_min_separation_voxels is not None
        and hard_negative_min_separation_voxels < 0
    ):
        raise ValueError("hard_negative_min_separation_voxels must be non-negative")

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    records = build_supervised_records(dataset_dir, subjects, include_controls)
    if not records:
        raise ValueError("No training records found")

    if len(records) == 1 and val_subjects is None:
        train_records = records
        val_records = []
    elif val_subjects is None:
        shuffled = list(records)
        np.random.shuffle(shuffled)
        val_idx = list(range(0, len(shuffled), 5))
        val_records = [shuffled[i] for i in val_idx if i < len(shuffled)]
        train_records = [r for i, r in enumerate(shuffled) if i not in val_idx]
    else:
        val_set = set(val_subjects)
        train_records = [r for r in records if r["subject_id"] not in val_set]
        val_records = [r for r in records if r["subject_id"] in val_set]

    if not train_records:
        raise ValueError("No training records found")
    if max_train_subjects is not None and len(train_records) > max_train_subjects:
        train_records = train_records[:max_train_subjects]

    train_samples = _sample_patch_centers(
        train_records,
        patch_size=patch_size,
        positives_per_subject=positives_per_subject,
        negatives_per_subject=negatives_per_subject,
        seed=seed,
        hard_negatives_per_subject=hard_negatives_per_subject,
        hard_negative_pool_factor=hard_negative_pool_factor,
        hard_negative_min_separation_voxels=hard_negative_min_separation_voxels,
    )
    val_samples = _sample_patch_centers(
        val_records,
        patch_size=patch_size,
        positives_per_subject=max(1, positives_per_subject // 2),
        negatives_per_subject=max(1, negatives_per_subject // 2),
        seed=seed + 1,
        hard_negatives_per_subject=(
            0
            if hard_negatives_per_subject == 0
            else max(1, hard_negatives_per_subject // 2)
        ),
        hard_negative_pool_factor=hard_negative_pool_factor,
        hard_negative_min_separation_voxels=hard_negative_min_separation_voxels,
    ) if val_records else []

    if not any(sample["label"] > 0 for sample in train_samples):
        raise ValueError("No positive lesion patch samples found")
    if not any(sample["label"] == 0 for sample in train_samples):
        raise ValueError("No negative patch samples found")

    class PatchDataset(Dataset):
        def __init__(self, samples: list[dict], patch_size: int):
            self.samples = samples
            self.patch_size = patch_size
            self.cache: dict[str, np.ndarray] = {}

        def __len__(self):
            return len(self.samples)

        def _volume(self, path: str) -> np.ndarray:
            if path not in self.cache:
                vol, _affine, _header, _img = load_canonical(path)
                self.cache[path] = _robust_normalize(vol).astype(np.float32)
            return self.cache[path]

        def __getitem__(self, idx):
            sample = self.samples[idx]
            vol = self._volume(sample["flair_path"])
            patch = _extract_centered_patch(vol, sample["center"], self.patch_size)
            x = torch.from_numpy(patch[None, ...].astype(np.float32))
            y = torch.tensor(float(sample["label"]), dtype=torch.float32)
            return x, y

    train_loader = DataLoader(
        PatchDataset(train_samples, patch_size),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        PatchDataset(val_samples, patch_size),
        batch_size=batch_size,
        shuffle=False,
    ) if val_samples else None

    model = PatchLesionNet.create().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_pos = sum(1 for sample in train_samples if sample["label"] > 0)
    n_neg = len(train_samples) - n_pos
    n_random_neg = sum(
        1 for sample in train_samples if sample.get("sample_type") == "negative_random"
    )
    n_hard_neg = sum(
        1 for sample in train_samples if sample.get("sample_type") == "negative_hard"
    )
    n_val_hard_neg = sum(
        1 for sample in val_samples if sample.get("sample_type") == "negative_hard"
    )
    pos_weight = max(1.0, n_neg / max(n_pos, 1))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    metrics = {
        "model_type": "patch_lesion_classifier",
        "train_subjects": [r["subject_id"] for r in train_records],
        "val_subjects": [r["subject_id"] for r in val_records],
        "patch_size": patch_size,
        "positives_per_subject": positives_per_subject,
        "negatives_per_subject": negatives_per_subject,
        "hard_negatives_per_subject": hard_negatives_per_subject,
        "hard_negative_pool_factor": hard_negative_pool_factor,
        "hard_negative_min_separation_voxels": hard_negative_min_separation_voxels,
        "n_train_samples": len(train_samples),
        "n_val_samples": len(val_samples),
        "n_train_positive_samples": n_pos,
        "n_train_random_negative_samples": n_random_neg,
        "n_train_hard_negative_samples": n_hard_neg,
        "n_val_hard_negative_samples": n_val_hard_neg,
        "pos_weight": float(pos_weight),
        "loss_history": [],
        "val_metrics": [],
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": total_loss / max(n_batches, 1),
        }
        if val_loader is not None:
            model.eval()
            correct = 0
            n_seen = 0
            probs: list[float] = []
            labels: list[float] = []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    logits = model(x)
                    prob = torch.sigmoid(logits).cpu().numpy()
                    y_np = y.numpy()
                    correct += int(((prob >= 0.5) == (y_np >= 0.5)).sum())
                    n_seen += len(y_np)
                    probs.extend(float(v) for v in prob)
                    labels.extend(float(v) for v in y_np)
            pos_probs = [p for p, label in zip(probs, labels) if label >= 0.5]
            neg_probs = [p for p, label in zip(probs, labels) if label < 0.5]
            epoch_metrics.update({
                "val_accuracy": correct / max(n_seen, 1),
                "val_positive_probability_mean": _mean_or_none(pos_probs),
                "val_negative_probability_mean": _mean_or_none(neg_probs),
            })
        metrics["loss_history"].append(epoch_metrics["train_loss"])
        metrics["val_metrics"].append(epoch_metrics)

    checkpoint = {
        "model_type": "patch_lesion_classifier",
        "model_state_dict": model.state_dict(),
        "patch_size": patch_size,
        "scan_stride": max(4, patch_size // 2),
        "candidate_radius_mm": 8.0,
        "hard_negatives_per_subject": hard_negatives_per_subject,
        "hard_negative_pool_factor": hard_negative_pool_factor,
        "hard_negative_min_separation_voxels": hard_negative_min_separation_voxels,
        "train_subjects": metrics["train_subjects"],
        "val_subjects": metrics["val_subjects"],
        "pos_weight": float(pos_weight),
        "metrics": metrics["val_metrics"],
    }
    ckpt_path = output_dir / "lesion_proposer.pt"
    torch.save(checkpoint, str(ckpt_path))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _load_prediction_model(checkpoint_path: Path | str, device: str = "auto"):
    """Load a trained lesion proposer once for repeated predictions."""
    _ensure_torch()
    import torch

    checkpoint_path = Path(checkpoint_path)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", "tiny_full_volume")
    if model_type == "patch_lesion_classifier":
        spatial_size = (int(checkpoint.get("patch_size", 32)),) * 3
        model = PatchLesionNet.create()
    else:
        spatial_size = tuple(checkpoint.get("spatial_size", (64, 64, 64)))
        validate_spatial_size(spatial_size)
        model = TinyLesionNet.create()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, spatial_size, checkpoint, device


def _predict_probability_map(
    model,
    spatial_size: tuple[int, int, int],
    volume_path: Path | str,
    device: str,
) -> dict:
    """Return probability map in the input volume's canonical grid."""
    _ensure_torch()
    import torch

    vol, affine, header, _ = load_canonical(volume_path)
    original_shape = vol.shape

    vol_norm = _robust_normalize(vol)
    vol_resized = resize_array(vol_norm, spatial_size, order=1)
    vol_tensor = torch.from_numpy(vol_resized[None, None, ...].astype(np.float32)).to(device)

    with torch.no_grad():
        pred_logits = model(vol_tensor)
        pred_prob = torch.sigmoid(pred_logits).squeeze().cpu().numpy()

    pred_resized = resize_array(pred_prob, original_shape, order=1).astype(np.float32)
    return {
        "probability": pred_resized,
        "affine": affine,
        "header": header,
        "shape": original_shape,
    }


def _predict_patch_probability_map(
    model,
    patch_size: int,
    volume_path: Path | str,
    device: str,
    *,
    stride: int = 16,
    batch_size: int = 64,
) -> dict:
    """Scan a patch classifier over a volume and return a sparse probability map."""
    _ensure_torch()
    import torch

    vol, affine, header, _ = load_canonical(volume_path)
    vol_norm = _robust_normalize(vol).astype(np.float32)
    fg = _foreground_mask(vol)
    coords = np.argwhere(fg)
    probability = np.zeros(vol.shape, dtype=np.float32)

    if coords.size == 0:
        return {
            "probability": probability,
            "affine": affine,
            "header": header,
            "shape": vol.shape,
        }

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    centers: list[tuple[int, int, int]] = []
    stride = max(1, int(stride))
    for i in range(int(mins[0]), int(maxs[0]) + 1, stride):
        for j in range(int(mins[1]), int(maxs[1]) + 1, stride):
            for k in range(int(mins[2]), int(maxs[2]) + 1, stride):
                if fg[i, j, k]:
                    centers.append((i, j, k))

    half_stride = max(1, stride // 2)
    for start in range(0, len(centers), batch_size):
        batch_centers = centers[start:start + batch_size]
        patches = np.stack([
            _extract_centered_patch(vol_norm, center, patch_size)
            for center in batch_centers
        ], axis=0)
        x = torch.from_numpy(patches[:, None, ...].astype(np.float32)).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(x)).cpu().numpy()
        for center, prob in zip(batch_centers, probs):
            i, j, k = center
            i0, i1 = max(0, i - half_stride), min(probability.shape[0], i + half_stride + 1)
            j0, j1 = max(0, j - half_stride), min(probability.shape[1], j + half_stride + 1)
            k0, k1 = max(0, k - half_stride), min(probability.shape[2], k + half_stride + 1)
            probability[i0:i1, j0:j1, k0:k1] = np.maximum(
                probability[i0:i1, j0:j1, k0:k1],
                float(prob),
            )

    probability *= fg.astype(np.float32)
    return {
        "probability": probability,
        "affine": affine,
        "header": header,
        "shape": vol.shape,
    }


def _prediction_probability_for_checkpoint(
    model,
    spatial_size: tuple[int, int, int],
    checkpoint: dict,
    volume_path: Path | str,
    device: str,
) -> dict:
    if checkpoint.get("model_type") == "patch_lesion_classifier":
        patch_size = int(checkpoint.get("patch_size", spatial_size[0]))
        stride = int(checkpoint.get("scan_stride", max(4, patch_size // 2)))
        return _predict_patch_probability_map(
            model,
            patch_size,
            volume_path,
            device,
            stride=stride,
        )
    return _predict_probability_map(model, spatial_size, volume_path, device)


def _requested_foreground_voxels(
    probability_size: int,
    foreground_fraction: Optional[float],
) -> int:
    if foreground_fraction is None:
        raise ValueError("foreground_fraction is required for budgeted ROI selection")
    fraction = float(foreground_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            f"foreground_fraction must be in (0, 1], got {foreground_fraction!r}"
        )
    return max(1, int(np.ceil(float(probability_size) * fraction)))


def _top_k_mask(
    probability: np.ndarray,
    *,
    k: int,
    eligible_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[float]]:
    flat_prob = np.asarray(probability, dtype=np.float32).ravel()
    if flat_prob.size == 0:
        return np.zeros_like(probability, dtype=np.float32), None
    if eligible_mask is None:
        eligible_idx = np.arange(flat_prob.size)
    else:
        eligible_idx = np.flatnonzero(np.asarray(eligible_mask, dtype=bool).ravel())
    if eligible_idx.size == 0:
        return np.zeros_like(probability, dtype=np.float32), None

    k = min(max(1, int(k)), int(eligible_idx.size))
    if k == eligible_idx.size:
        selected = eligible_idx
    else:
        eligible_scores = flat_prob[eligible_idx]
        selected_local = np.argpartition(eligible_scores, -k)[-k:]
        selected = eligible_idx[selected_local]

    out = np.zeros(flat_prob.size, dtype=np.float32)
    out[selected] = 1.0
    cutoff = float(flat_prob[selected].min()) if selected.size else None
    return out.reshape(probability.shape).astype(np.float32), cutoff


def _select_probability_mask(
    probability: np.ndarray,
    *,
    threshold: float = 0.5,
    selection_mode: str = "threshold",
    foreground_fraction: Optional[float] = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Convert a probability map to a binary mask with an optional ROI budget."""
    if selection_mode not in _PREDICTION_SELECTION_MODES:
        raise ValueError(
            "selection_mode must be one of "
            f"{', '.join(_PREDICTION_SELECTION_MODES)}, got {selection_mode!r}"
        )
    probability = np.asarray(probability, dtype=np.float32)
    meta: dict[str, object] = {
        "selection_mode": selection_mode,
        "threshold": float(threshold),
        "foreground_fraction": (
            None if foreground_fraction is None else float(foreground_fraction)
        ),
        "requested_foreground_voxels": None,
        "effective_probability_threshold": None,
    }

    if selection_mode == "threshold":
        mask = (probability > float(threshold)).astype(np.float32)
        selected = int(mask.sum())
        meta["requested_foreground_voxels"] = selected
        meta["selected_voxels_before_component_filter"] = selected
        if selected:
            meta["effective_probability_threshold"] = float(
                probability[mask > 0].min()
            )
        return mask, meta

    requested_voxels = _requested_foreground_voxels(
        int(probability.size),
        foreground_fraction,
    )
    meta["requested_foreground_voxels"] = requested_voxels

    if selection_mode == "top-fraction":
        mask, cutoff = _top_k_mask(probability, k=requested_voxels)
    else:
        eligible = probability > float(threshold)
        eligible_voxels = int(eligible.sum())
        meta["threshold_eligible_voxels"] = eligible_voxels
        if eligible_voxels <= requested_voxels:
            mask = eligible.astype(np.float32)
            cutoff = float(threshold) if eligible_voxels else None
        else:
            mask, cutoff = _top_k_mask(
                probability,
                k=requested_voxels,
                eligible_mask=eligible,
            )

    meta["effective_probability_threshold"] = cutoff
    meta["selected_voxels_before_component_filter"] = int(mask.sum())
    return mask.astype(np.float32), meta


def _numeric_sort_value(value: object, fallback: float) -> float:
    if value is None:
        return fallback
    return float(value)


def predict_lesion_roi(
    checkpoint_path: Path | str,
    volume_path: Path | str,
    output_path: Path | str,
    threshold: float = 0.5,
    min_component_voxels: int = 0,
    device: str = "auto",
    selection_mode: str = "threshold",
    foreground_fraction: Optional[float] = None,
) -> dict:
    """Predict lesion ROI on a single volume.

    Returns dict with predicted_voxels, probability_mean/max, threshold, spatial_size.
    """
    volume_path = Path(volume_path)
    output_path = Path(output_path)

    model, spatial_size, _checkpoint, device = _load_prediction_model(checkpoint_path, device=device)
    pred_info = _prediction_probability_for_checkpoint(
        model, spatial_size, _checkpoint, volume_path, device
    )
    pred_resized = pred_info["probability"]
    pred_bin, selection_meta = _select_probability_mask(
        pred_resized,
        threshold=threshold,
        selection_mode=selection_mode,
        foreground_fraction=foreground_fraction,
    )
    pred_bin = _component_filter(pred_bin, min_component_voxels=min_component_voxels)
    predicted_voxels = int(pred_bin.sum())
    predicted_foreground_fraction = float(predicted_voxels / pred_bin.size) if pred_bin.size else 0.0

    save_nifti_mask(pred_bin, pred_info["affine"], pred_info["header"], output_path)

    return {
        "predicted_voxels": predicted_voxels,
        "predicted_foreground_fraction": predicted_foreground_fraction,
        "probability_mean": float(pred_resized.mean()),
        "probability_max": float(pred_resized.max()),
        "threshold": threshold,
        "spatial_size": spatial_size,
        "selection_mode": selection_mode,
        "foreground_fraction": foreground_fraction,
        "selected_voxels_after_component_filter": predicted_voxels,
        **selection_meta,
    }


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def _mean_or_none(values: list[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _median_or_none(values: list[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


def _rate_ge(values: list[float], threshold: float) -> Optional[float]:
    if not values:
        return None
    return float(sum(1 for value in values if value >= threshold) / len(values))


def _manual_overlap_metrics(
    pred_bin: np.ndarray,
    pred_affine: np.ndarray,
    manual_roi_path: Path | str,
) -> dict[str, float]:
    """Compute Dice, manual coverage, and precision against a manual ROI."""
    manual_roi, manual_affine, _manual_header, _manual_img = load_canonical(manual_roi_path)
    manual_bin = manual_roi > 0.5
    pred_bool = pred_bin > 0.5

    if pred_bool.shape != manual_bin.shape or not np.allclose(pred_affine, manual_affine):
        pred_resampled = resample_from_to(
            nib.Nifti1Image(pred_bool.astype(np.float32), pred_affine),
            nib.Nifti1Image(manual_roi.astype(np.float32), manual_affine),
            order=0,
        ).get_fdata(dtype=np.float32)
        pred_bool = pred_resampled > 0.5

    intersection = int((pred_bool & manual_bin).sum())
    pred_voxels = int(pred_bool.sum())
    manual_voxels = int(manual_bin.sum())
    dice = 2.0 * intersection / (pred_voxels + manual_voxels + 1e-7)
    coverage = intersection / max(manual_voxels, 1)
    precision = intersection / max(pred_voxels, 1)
    return {
        "dice": float(dice),
        "manual_coverage": float(coverage),
        "candidate_precision": float(precision),
        "predicted_voxels": float(pred_voxels),
        "manual_voxels": float(manual_voxels),
        "intersection_voxels": float(intersection),
    }


def evaluate_threshold_sweep(
    checkpoint_path: Path | str,
    dataset_dir: Path | str,
    output_dir: Path | str,
    subjects: Optional[list[str]] = None,
    thresholds: Optional[list[float]] = None,
    selection_modes: Optional[list[str]] = None,
    foreground_fractions: Optional[list[float]] = None,
    min_component_voxels: int = 0,
    min_coverage: float = 0.50,
    max_hc_foreground_fraction: float = 0.01,
    device: str = "auto",
) -> dict:
    """Calibrate supervised ROI thresholds against FCD coverage and HC burden.

    The sweep is the paper/product gate for ROI-free uploaded scans: a threshold
    should preserve meaningful coverage of known FCD lesions while bounding
    false-positive candidate burden in healthy controls.
    """
    checkpoint_path = Path(checkpoint_path)
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if thresholds is None:
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    thresholds = sorted({float(t) for t in thresholds})
    if not thresholds:
        raise ValueError("thresholds must contain at least one value")
    if selection_modes is None:
        selection_modes = ["threshold"]
    selection_modes = list(dict.fromkeys(selection_modes))
    invalid_modes = [
        mode for mode in selection_modes
        if mode not in _PREDICTION_SELECTION_MODES
    ]
    if invalid_modes:
        raise ValueError(
            "selection_modes must be drawn from "
            f"{', '.join(_PREDICTION_SELECTION_MODES)}, got {invalid_modes!r}"
        )

    needs_fraction = any(
        mode in {"top-fraction", "threshold-and-top-fraction"}
        for mode in selection_modes
    )
    if needs_fraction:
        if not foreground_fractions:
            raise ValueError(
                "foreground_fractions are required for budgeted selection modes"
            )
        foreground_fractions = sorted({float(v) for v in foreground_fractions})
        for value in foreground_fractions:
            _requested_foreground_voxels(1, value)
    else:
        foreground_fractions = []

    sweep_configs: list[dict[str, object]] = []
    for mode in selection_modes:
        if mode == "threshold":
            for threshold in thresholds:
                sweep_configs.append({
                    "selection_mode": mode,
                    "threshold": threshold,
                    "foreground_fraction": None,
                })
        elif mode == "top-fraction":
            for fraction in foreground_fractions:
                sweep_configs.append({
                    "selection_mode": mode,
                    "threshold": None,
                    "foreground_fraction": fraction,
                })
        else:
            for threshold in thresholds:
                for fraction in foreground_fractions:
                    sweep_configs.append({
                        "selection_mode": mode,
                        "threshold": threshold,
                        "foreground_fraction": fraction,
                    })

    records = build_supervised_records(dataset_dir, subjects, include_controls=True)
    model, spatial_size, checkpoint, device = _load_prediction_model(
        checkpoint_path,
        device=device,
    )

    accum: dict[int, dict[str, list[float]]] = {
        idx: {
            "fcd_coverage": [],
            "fcd_precision": [],
            "fcd_dice": [],
            "hc_predicted_voxels": [],
            "hc_foreground_fraction": [],
        }
        for idx in range(len(sweep_configs))
    }

    n_fcd = 0
    n_hc = 0
    skipped_records: list[dict[str, str]] = []

    for rec in records:
        group = (rec.get("group") or "unknown").lower()
        flair_path = rec.get("flair_path")
        roi_path = rec.get("roi_path")
        if not flair_path:
            skipped_records.append({"subject_id": rec.get("subject_id", "unknown"), "reason": "missing_flair"})
            continue

        pred_info = _prediction_probability_for_checkpoint(
            model, spatial_size, checkpoint, flair_path, device
        )
        prob = pred_info["probability"]
        pred_affine = pred_info["affine"]

        is_fcd = group == "fcd" and roi_path is not None
        is_hc = group in {"hc", "control", "healthy", "nc"} or roi_path is None
        if is_fcd:
            n_fcd += 1
        elif is_hc:
            n_hc += 1
        else:
            skipped_records.append({
                "subject_id": rec.get("subject_id", "unknown"),
                "reason": f"unsupported_group:{group}",
            })
            continue

        for idx, config in enumerate(sweep_configs):
            threshold_value = (
                0.5 if config["threshold"] is None else float(config["threshold"])
            )
            pred_raw, _selection_meta = _select_probability_mask(
                prob,
                threshold=threshold_value,
                selection_mode=str(config["selection_mode"]),
                foreground_fraction=(
                    None if config["foreground_fraction"] is None
                    else float(config["foreground_fraction"])
                ),
            )
            pred_bin = _component_filter(
                pred_raw,
                min_component_voxels=min_component_voxels,
            )
            pred_voxels = int(pred_bin.sum())
            foreground_fraction = float(pred_voxels / pred_bin.size) if pred_bin.size else 0.0

            if is_fcd:
                overlap = _manual_overlap_metrics(pred_bin, pred_affine, roi_path)
                accum[idx]["fcd_coverage"].append(overlap["manual_coverage"])
                accum[idx]["fcd_precision"].append(overlap["candidate_precision"])
                accum[idx]["fcd_dice"].append(overlap["dice"])
            else:
                accum[idx]["hc_predicted_voxels"].append(float(pred_voxels))
                accum[idx]["hc_foreground_fraction"].append(foreground_fraction)

    rows: list[dict[str, object]] = []
    for idx, config in enumerate(sweep_configs):
        data = accum[idx]
        fcd_coverage = data["fcd_coverage"]
        fcd_precision = data["fcd_precision"]
        fcd_dice = data["fcd_dice"]
        hc_voxels = data["hc_predicted_voxels"]
        hc_fraction = data["hc_foreground_fraction"]
        hc_false_positive_rate = (
            float(sum(1 for value in hc_voxels if value > 0) / len(hc_voxels))
            if hc_voxels else None
        )
        coverage_ge_50_rate = _rate_ge(fcd_coverage, 0.50)
        mean_hc_fraction = _mean_or_none(hc_fraction)
        meets_constraints = (
            coverage_ge_50_rate is not None
            and mean_hc_fraction is not None
            and coverage_ge_50_rate >= min_coverage
            and mean_hc_fraction <= max_hc_foreground_fraction
        )
        rows.append({
            "selection_mode": config["selection_mode"],
            "threshold": config["threshold"],
            "foreground_fraction": config["foreground_fraction"],
            "n_fcd": n_fcd,
            "n_hc": n_hc,
            "fcd_mean_manual_coverage": _mean_or_none(fcd_coverage),
            "fcd_median_manual_coverage": _median_or_none(fcd_coverage),
            "fcd_mean_precision": _mean_or_none(fcd_precision),
            "fcd_median_precision": _median_or_none(fcd_precision),
            "fcd_mean_dice": _mean_or_none(fcd_dice),
            "fcd_median_dice": _median_or_none(fcd_dice),
            "fcd_coverage_ge_0_10_rate": _rate_ge(fcd_coverage, 0.10),
            "fcd_coverage_ge_0_25_rate": _rate_ge(fcd_coverage, 0.25),
            "fcd_coverage_ge_0_50_rate": coverage_ge_50_rate,
            "hc_mean_predicted_voxels": _mean_or_none(hc_voxels),
            "hc_median_predicted_voxels": _median_or_none(hc_voxels),
            "hc_mean_predicted_foreground_fraction": mean_hc_fraction,
            "hc_median_predicted_foreground_fraction": _median_or_none(hc_fraction),
            "hc_false_positive_subject_rate": hc_false_positive_rate,
            "meets_operating_constraints": bool(meets_constraints),
            "selected_operating_point": False,
        })

    candidate_rows = [row for row in rows if row["meets_operating_constraints"]]
    selected = None
    if candidate_rows:
        selected = sorted(
            candidate_rows,
            key=lambda row: (
                -_numeric_sort_value(row["fcd_coverage_ge_0_50_rate"], 0.0),
                _numeric_sort_value(row["hc_mean_predicted_foreground_fraction"], 1.0),
                -_numeric_sort_value(row["fcd_mean_precision"], 0.0),
                _SELECTION_MODE_SORT_RANK[str(row["selection_mode"])],
                _numeric_sort_value(row["foreground_fraction"], 1.0),
                -_numeric_sort_value(row["threshold"], 0.0),
            ),
        )[0]
        selected["selected_operating_point"] = True

    result = {
        "method": "supervised_lesion_threshold_sweep",
        "checkpoint": str(checkpoint_path),
        "dataset_dir": str(dataset_dir),
        "spatial_size": list(spatial_size),
        "thresholds": rows,
        "selected_operating_point": selected,
        "constraints": {
            "min_coverage_ge_0_50_rate": min_coverage,
            "max_hc_mean_predicted_foreground_fraction": max_hc_foreground_fraction,
            "min_component_voxels": min_component_voxels,
            "selection_modes": selection_modes,
            "foreground_fractions": foreground_fractions,
        },
        "n_fcd": n_fcd,
        "n_hc": n_hc,
        "skipped_records": skipped_records,
        "limitations": [
            "Threshold sweep calibrates an ROI proposal gate, not a clinical detector.",
            "FCD coverage and HC burden must be validated on held-out subjects.",
            "A high FCD coverage threshold can still oversegment; precision and HC foreground fraction must be reported.",
        ],
    }

    json_path = output_dir / "threshold_sweep.json"
    json_path.write_text(json.dumps(result, indent=2))

    csv_path = output_dir / "threshold_sweep.csv"
    with csv_path.open("w", newline="") as fh:
        fieldnames = list(rows[0].keys()) if rows else ["threshold"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return result


def load_supervised_roi_operating_point(path: Path | str) -> dict[str, object]:
    """Load calibrated supervised-ROI runtime settings from JSON.

    Accepts either the ``threshold_sweep.json`` written by
    :func:`evaluate_threshold_sweep` or a small direct JSON object.  A sweep
    without a selected operating point is rejected so the upload pipeline cannot
    accidentally run with uncalibrated defaults.
    """
    source_path = Path(path)
    with source_path.open() as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("supervised ROI operating point JSON must be an object")

    constraints = payload.get("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}

    if "selected_operating_point" in payload:
        selected = payload.get("selected_operating_point")
        if not isinstance(selected, dict):
            raise ValueError(
                "threshold_sweep JSON has no selected_operating_point; "
                "calibrate a valid operating point before using supervised ROI"
            )
        source_kind = "threshold_sweep"
        values = dict(selected)
        if "min_component_voxels" not in values and "min_component_voxels" in constraints:
            values["min_component_voxels"] = constraints["min_component_voxels"]
        if (
            "max_foreground_fraction" not in values
            and "max_hc_mean_predicted_foreground_fraction" in constraints
        ):
            values["max_foreground_fraction"] = constraints[
                "max_hc_mean_predicted_foreground_fraction"
            ]
    else:
        source_kind = "direct"
        values = payload

    result: dict[str, object] = {
        "source_path": str(source_path),
        "source_kind": source_kind,
    }
    key_map = {
        "threshold": "threshold",
        "selection_mode": "selection_mode",
        "foreground_fraction": "foreground_fraction",
        "min_component_voxels": "min_component_voxels",
        "max_foreground_fraction": "max_foreground_fraction",
        "max_hc_mean_predicted_foreground_fraction": "max_foreground_fraction",
    }
    for src_key, dst_key in key_map.items():
        if src_key in values and values[src_key] is not None:
            value = values[src_key]
            if dst_key == "selection_mode":
                if value not in _PREDICTION_SELECTION_MODES:
                    raise ValueError(
                        "selection_mode must be one of "
                        f"{', '.join(_PREDICTION_SELECTION_MODES)}, got {value!r}"
                    )
                result[dst_key] = str(value)
            elif dst_key == "min_component_voxels":
                result[dst_key] = int(value)
            else:
                result[dst_key] = float(value)

    if source_kind == "threshold_sweep":
        result["constraints"] = dict(constraints)
        bool_keys = ("selected_operating_point", "meets_operating_constraints")
        for key in bool_keys:
            if key in values and values[key] is not None:
                result[key] = bool(values[key])
        int_keys = ("n_fcd", "n_hc")
        for key in int_keys:
            if key in values and values[key] is not None:
                result[key] = int(values[key])
        metric_keys = (
            "fcd_mean_manual_coverage",
            "fcd_median_manual_coverage",
            "fcd_mean_precision",
            "fcd_median_precision",
            "fcd_mean_dice",
            "fcd_median_dice",
            "fcd_coverage_ge_0_10_rate",
            "fcd_coverage_ge_0_25_rate",
            "fcd_coverage_ge_0_50_rate",
            "hc_mean_predicted_voxels",
            "hc_median_predicted_voxels",
            "hc_mean_predicted_foreground_fraction",
            "hc_median_predicted_foreground_fraction",
            "hc_false_positive_subject_rate",
        )
        for key in metric_keys:
            if key in values and values[key] is not None:
                result[key] = float(values[key])
        constraint_float_keys = (
            "min_coverage_ge_0_50_rate",
            "max_hc_mean_predicted_foreground_fraction",
        )
        for key in constraint_float_keys:
            if key in constraints and constraints[key] is not None:
                result[key] = float(constraints[key])

    if "threshold" not in result and "selection_mode" not in result:
        raise ValueError(
            "supervised ROI operating point must provide at least threshold "
            "or selection_mode"
        )
    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_lesion_proposer(
    checkpoint_path: Path | str,
    dataset_dir: Path | str,
    output_dir: Path | str,
    subjects: Optional[list[str]] = None,
    threshold: float = 0.5,
    device: str = "auto",
    selection_mode: str = "threshold",
    foreground_fraction: Optional[float] = None,
) -> dict:
    """Evaluate on FCD subjects with manual ROI.

    Writes predictions to output_dir/<subject>/predicted_lesion_roi.nii.gz
    and summary.json + subjects.csv
    """
    checkpoint_path = Path(checkpoint_path)
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_supervised_records(dataset_dir, subjects, include_controls=False)
    fcd_records = [r for r in records if r["group"] == "fcd" and r["roi_path"]]

    results = []

    for rec in fcd_records:
        subj_id = rec["subject_id"]
        flair_path = Path(rec["flair_path"])
        roi_path = Path(rec["roi_path"])

        subj_output_dir = output_dir / subj_id
        subj_output_dir.mkdir(parents=True, exist_ok=True)
        pred_path = subj_output_dir / "predicted_lesion_roi.nii.gz"

        pred_result = predict_lesion_roi(
            checkpoint_path, flair_path, pred_path,
            threshold=threshold,
            device=device,
            selection_mode=selection_mode,
            foreground_fraction=foreground_fraction,
        )

        manual_roi, manual_affine, manual_header, manual_img = load_canonical(roi_path)
        manual_bin = (manual_roi > 0.5).astype(np.float32)

        pred_bin, pred_affine, pred_header, pred_img = load_canonical(pred_path)
        pred_bin = (pred_bin > 0.5).astype(np.float32)

        # Resample if shape OR affine differs
        if manual_bin.shape != pred_bin.shape or not np.allclose(pred_affine, manual_affine):
            pred_bin_resampled = resample_from_to(
                nib.Nifti1Image(pred_bin, pred_affine),
                nib.Nifti1Image(manual_roi, manual_affine),
                order=0
            ).get_fdata(dtype=np.float32)
            pred_bin_resampled = (pred_bin_resampled > 0.5).astype(bool)
        else:
            pred_bin_resampled = pred_bin > 0.5

        manual_bin_bool = manual_bin > 0.5
        intersection = (pred_bin_resampled & manual_bin_bool).sum()
        union = (pred_bin_resampled | manual_bin_bool).sum()
        dice = 2.0 * intersection / (pred_bin_resampled.sum() + manual_bin_bool.sum() + 1e-7)

        manual_coverage = intersection / max(manual_bin_bool.sum(), 1)
        candidate_precision = intersection / max(pred_bin_resampled.sum(), 1)

        results.append({
            "subject_id": subj_id,
            "dice": float(dice),
            "manual_coverage": float(manual_coverage),
            "candidate_precision": float(candidate_precision),
            "predicted_voxels": int(pred_bin_resampled.sum()),
            "manual_voxels": int(manual_bin_bool.sum()),
            "intersection_voxels": int(intersection),
        })

    summary = {
        "n_subjects": len(results),
        "mean_dice": float(np.mean([r["dice"] for r in results]) if results else 0.0),
        "mean_coverage": float(np.mean([r["manual_coverage"] for r in results]) if results else 0.0),
        "mean_precision": float(np.mean([r["candidate_precision"] for r in results]) if results else 0.0),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    csv_path = output_dir / "subjects.csv"
    if results:
        with open(csv_path, 'w') as f:
            f.write("subject_id,dice,manual_coverage,candidate_precision,predicted_voxels,manual_voxels,intersection_voxels\n")
            for r in results:
                f.write(
                    f"{r['subject_id']},"
                    f"{r['dice']:.4f},"
                    f"{r['manual_coverage']:.4f},"
                    f"{r['candidate_precision']:.4f},"
                    f"{r['predicted_voxels']},"
                    f"{r['manual_voxels']},"
                    f"{r['intersection_voxels']}\n"
                )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Main entry point with argparse subcommands."""
    parser = argparse.ArgumentParser(
        description="Supervised lesion proposal baseline for FCD detection."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train lesion proposer")
    train_parser.add_argument("dataset_dir", help="Path to ds004199 root")
    train_parser.add_argument("output_dir", help="Output directory for checkpoint")
    train_parser.add_argument("--subjects", nargs="+", default=None, help="Limit to subjects")
    train_parser.add_argument("--val-subjects", nargs="+", default=None, help="Validation subjects")
    train_parser.add_argument("--spatial-size", type=int, nargs=3, default=[64, 64, 64])
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--include-controls", action="store_true", default=True)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--max-train-subjects", type=int, default=None)

    # Patch-train subcommand
    patch_parser = subparsers.add_parser(
        "train-patch",
        help="Train patch classifier lesion proposer",
    )
    patch_parser.add_argument("dataset_dir", help="Path to ds004199 root")
    patch_parser.add_argument("output_dir", help="Output directory for checkpoint")
    patch_parser.add_argument("--subjects", nargs="+", default=None, help="Limit to subjects")
    patch_parser.add_argument("--val-subjects", nargs="+", default=None, help="Validation subjects")
    patch_parser.add_argument("--patch-size", type=int, default=32)
    patch_parser.add_argument("--positives-per-subject", type=int, default=8)
    patch_parser.add_argument("--negatives-per-subject", type=int, default=16)
    patch_parser.add_argument(
        "--hard-negatives-per-subject",
        type=int,
        default=0,
        help="Additional lesion-like non-ROI foreground patches per subject.",
    )
    patch_parser.add_argument(
        "--hard-negative-pool-factor",
        type=int,
        default=8,
        help="Candidate pool multiplier used before diversity filtering hard negatives.",
    )
    patch_parser.add_argument(
        "--hard-negative-min-separation-voxels",
        type=int,
        default=None,
        help="Minimum voxel distance between hard-negative centers.",
    )
    patch_parser.add_argument("--epochs", type=int, default=5)
    patch_parser.add_argument("--batch-size", type=int, default=16)
    patch_parser.add_argument("--lr", type=float, default=1e-3)
    patch_parser.add_argument("--device", default="auto")
    patch_parser.add_argument("--include-controls", action="store_true", default=True)
    patch_parser.add_argument("--seed", type=int, default=42)
    patch_parser.add_argument("--max-train-subjects", type=int, default=None)

    # Predict subcommand
    pred_parser = subparsers.add_parser("predict", help="Predict lesion ROI")
    pred_parser.add_argument("checkpoint", help="Path to checkpoint")
    pred_parser.add_argument("volume", help="Input FLAIR volume")
    pred_parser.add_argument("output", help="Output mask path")
    pred_parser.add_argument("--threshold", type=float, default=0.5)
    pred_parser.add_argument("--min-component-voxels", type=int, default=0)
    pred_parser.add_argument(
        "--selection-mode",
        choices=_PREDICTION_SELECTION_MODES,
        default="threshold",
        help="How to binarize probability maps.",
    )
    pred_parser.add_argument(
        "--foreground-fraction",
        type=float,
        default=None,
        help="Max foreground fraction for top-fraction selection modes.",
    )
    pred_parser.add_argument("--device", default="auto")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate on dataset")
    eval_parser.add_argument("checkpoint", help="Path to checkpoint")
    eval_parser.add_argument("dataset_dir", help="Path to ds004199 root")
    eval_parser.add_argument("output_dir", help="Output directory for results")
    eval_parser.add_argument("--subjects", nargs="+", default=None)
    eval_parser.add_argument("--threshold", type=float, default=0.5)
    eval_parser.add_argument(
        "--selection-mode",
        choices=_PREDICTION_SELECTION_MODES,
        default="threshold",
    )
    eval_parser.add_argument("--foreground-fraction", type=float, default=None)
    eval_parser.add_argument("--device", default="auto")

    # Threshold sweep subcommand
    sweep_parser = subparsers.add_parser("sweep", help="Calibrate thresholds on FCD/HC subjects")
    sweep_parser.add_argument("checkpoint", help="Path to checkpoint")
    sweep_parser.add_argument("dataset_dir", help="Path to ds004199 root")
    sweep_parser.add_argument("output_dir", help="Output directory for threshold_sweep.json/csv")
    sweep_parser.add_argument("--subjects", nargs="+", default=None)
    sweep_parser.add_argument("--thresholds", type=float, nargs="+",
                              default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    sweep_parser.add_argument(
        "--selection-modes",
        choices=_PREDICTION_SELECTION_MODES,
        nargs="+",
        default=["threshold"],
        help="Mask-selection modes to calibrate.",
    )
    sweep_parser.add_argument(
        "--foreground-fractions",
        type=float,
        nargs="+",
        default=None,
        help="Foreground budgets for top-fraction selection modes.",
    )
    sweep_parser.add_argument("--min-component-voxels", type=int, default=0)
    sweep_parser.add_argument("--min-coverage", type=float, default=0.50)
    sweep_parser.add_argument("--max-hc-foreground-fraction", type=float, default=0.01)
    sweep_parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    if args.command == "train":
        metrics = train_lesion_proposer(
            args.dataset_dir,
            args.output_dir,
            subjects=args.subjects,
            val_subjects=args.val_subjects,
            spatial_size=tuple(args.spatial_size),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            include_controls=args.include_controls,
            seed=args.seed,
            max_train_subjects=args.max_train_subjects,
        )
        print(f"Training complete. Checkpoint: {Path(args.output_dir) / 'lesion_proposer.pt'}")

    elif args.command == "train-patch":
        metrics = train_patch_lesion_proposer(
            args.dataset_dir,
            args.output_dir,
            subjects=args.subjects,
            val_subjects=args.val_subjects,
            patch_size=args.patch_size,
            positives_per_subject=args.positives_per_subject,
            negatives_per_subject=args.negatives_per_subject,
            hard_negatives_per_subject=args.hard_negatives_per_subject,
            hard_negative_pool_factor=args.hard_negative_pool_factor,
            hard_negative_min_separation_voxels=args.hard_negative_min_separation_voxels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            include_controls=args.include_controls,
            seed=args.seed,
            max_train_subjects=args.max_train_subjects,
        )
        print(f"Patch training complete. Checkpoint: {Path(args.output_dir) / 'lesion_proposer.pt'}")
        print(f"  Training samples: {metrics['n_train_samples']}")

    elif args.command == "predict":
        result = predict_lesion_roi(
            args.checkpoint,
            args.volume,
            args.output,
            threshold=args.threshold,
            min_component_voxels=args.min_component_voxels,
            device=args.device,
            selection_mode=args.selection_mode,
            foreground_fraction=args.foreground_fraction,
        )
        print(f"Prediction saved to {args.output}")
        print(f"  Predicted voxels: {result['predicted_voxels']}")
        print(f"  Probability mean: {result['probability_mean']:.4f}")

    elif args.command == "evaluate":
        summary = evaluate_lesion_proposer(
            args.checkpoint,
            args.dataset_dir,
            args.output_dir,
            subjects=args.subjects,
            threshold=args.threshold,
            device=args.device,
            selection_mode=args.selection_mode,
            foreground_fraction=args.foreground_fraction,
        )
        print(f"Evaluation complete ({summary['n_subjects']} subjects)")
        print(f"  Mean Dice: {summary['mean_dice']:.4f}")
        print(f"  Mean coverage: {summary['mean_coverage']:.4f}")
        print(f"  Mean precision: {summary['mean_precision']:.4f}")

    elif args.command == "sweep":
        result = evaluate_threshold_sweep(
            args.checkpoint,
            args.dataset_dir,
            args.output_dir,
            subjects=args.subjects,
            thresholds=args.thresholds,
            selection_modes=args.selection_modes,
            foreground_fractions=args.foreground_fractions,
            min_component_voxels=args.min_component_voxels,
            min_coverage=args.min_coverage,
            max_hc_foreground_fraction=args.max_hc_foreground_fraction,
            device=args.device,
        )
        selected = result.get("selected_operating_point")
        print(f"Threshold sweep complete ({result['n_fcd']} FCD, {result['n_hc']} HC)")
        if selected:
            print(f"  Selected threshold: {selected['threshold']}")
        else:
            print("  No threshold met the operating constraints.")


if __name__ == "__main__":
    main()
