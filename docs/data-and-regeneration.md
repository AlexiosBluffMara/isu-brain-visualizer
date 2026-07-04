# Data And Regeneration

## Dataset

This project uses OpenNeuro dataset `ds004199`.

Citation:

> Schuch F et al. "An open presurgery MRI dataset of people with epilepsy and
> focal cortical dysplasia type II". Scientific Data 10, 475 (2023).
> DOI: https://doi.org/10.1038/s41597-023-02386-7
> OpenNeuro: https://openneuro.org/datasets/ds004199

## Local Output Folder

The viewer expects processed data at:

```text
outputs/end_to_end_fcd85_hd
```

The current verified local output contains:

```text
subjects=85
ok=85
bad=0
total_splat_points=7557985
```

## Why The Data Is Not In Git

The generated viewer output is about 15-16 GiB. GitHub source repositories are
not a good place for that much generated binary data. Keep the Git repository
focused on source code and instructions.

Recommended sharing paths:

1. Regenerate locally from OpenNeuro.
2. Share the generated output folder through private object storage.
3. Use release assets only for a compact demo bundle.

## Regenerate Browser Splats

Run from the repository root:

```bash
python scripts/regenerate_browser_splats.py \
  --root outputs/end_to_end_fcd85_hd \
  --anatomy-points 85000 \
  --lesion-points 35000 \
  --percentile-threshold 10
```

Verify:

```bash
python scripts/verify_viewer_dataset.py --root outputs/end_to_end_fcd85_hd
```

## What Gets Regenerated

For each subject, the script writes:

```text
patient_splat_lesion_highlighted_browser.ply
patient_splat_lesion_highlighted_browser_meta.json
patient_splat_lesion_highlighted_browser_state.npz
```

These files are made from:

- the subject MRI foreground anatomy,
- the provided expert lesion mask.

They are tuned for browser responsiveness.
