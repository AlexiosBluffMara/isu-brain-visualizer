# Data Directory

## OpenNeuro ds004199

**Title:** An open presurgery MRI dataset of people with epilepsy and focal cortical
dysplasia type II

**Citation:**
> Schuch F, Walger L, Schmitz M, et al. "An open presurgery MRI dataset of people
> with epilepsy and focal cortical dysplasia type II". *Scientific Data* **10**, 475
> (2023). DOI: [10.1038/s41597-023-02386-7](https://doi.org/10.1038/s41597-023-02386-7)

**OpenNeuro accession:** ds004199 v1.0.6
**OpenNeuro DOI:** [doi:10.18112/openneuro.ds004199.v1.0.6](https://openneuro.org/datasets/ds004199)
**License:** [CC0 1.0 Universal (Public Domain)](https://creativecommons.org/publicdomain/zero/1.0/)

This dataset is public and de-identified, but it is still medical research data.
Handle it carefully, do not attempt to re-identify anyone, and cite the dataset in
any report or paper that uses it.

## What the dataset contains

- **170 subjects total:** 85 with focal cortical dysplasia type II + 85 healthy controls.
- **Format:** BIDS v1.6.0 layout.
- **NIfTI volumes** (`.nii.gz`) - 3D T1w and 3D FLAIR anatomical scans per subject.
- **JSON sidecars** (`.json`) - scan acquisition metadata (scanner, field strength, TR, TE, etc.).
- **TSV files** (`participants.tsv`) - de-identified participant demographics and clinical fields.
- **Expert ROI masks** (`*_roi.nii.gz`) - binary NIfTI lesion masks drawn by a neurologist;
  present for all 85 FCD subjects, absent for healthy controls.

## Why it is local-only (not committed to this repository)

`data/openneuro/ds004199/` is listed in `.gitignore`. The local full cohort is
about 4.15 GB, which is large enough to make normal repository cloning and review
painful. The data is public CC0 and redistribution is permitted - it is excluded
here only for repository size reasons.

`data/README.md` and `data/.gitkeep` remain tracked so collaborators know where to
place the data and how to download it.

## What is committed vs local-only

| Path | Committed | Notes |
|------|-----------|-------|
| `data/README.md` | YES | This file - citation and setup notes |
| `data/.gitkeep` | YES | Keeps the directory present in git |
| `data/openneuro/ds004199/` | **NO** | Local-only; public CC0 but too large for git |
| `data/raw/` | NO | Local archives |
| `data/nifti/` | NO | Converted volumes outside the BIDS tree |
| `data/processed/` | NO | Preprocessed tensors |
| `outputs/` | NO (except small manifests) | Regenerable pipeline outputs |

## Downloading examples

No AWS account is needed. OpenNeuro is publicly accessible.

### Option 1 - project download helper

```bash
# FCD subject with expert lesion ROI mask
fcd-download-ds \
    --destination data/openneuro/ds004199 \
    --subjects sub-00058 \
    --modalities T1w FLAIR \
    --include-roi

# Healthy control subject (no ROI mask)
fcd-download-ds \
    --destination data/openneuro/ds004199 \
    --subjects sub-00124 \
    --modalities T1w FLAIR
```

### Option 2 - OpenNeuro CLI

```bash
pip install openneuro-py
openneuro download --dataset ds004199 --target data/openneuro/ds004199
```

### Option 3 - DataLad

```bash
datalad install -r https://github.com/OpenNeuroDatasets/ds004199
```

After downloading, run the end-to-end pipeline to regenerate viewer manifests under
`outputs/`. See `docs/demo-runbook.md` for the full walkthrough.

## Other public datasets (not included)

| Dataset | License | Source |
|---------|---------|--------|
| IXI healthy controls | CC BY-SA | https://brain-development.org/ixi-dataset/ |
| NFBS skull-strip repo | CC BY | https://www.nitrc.org/projects/nfbs/ |
