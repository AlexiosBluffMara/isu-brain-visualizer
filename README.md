# ISU Brain Visualizer

Educational MRI viewer for public Focal Cortical Dysplasia data.

The project takes OpenNeuro MRI data with expert-marked FCD lesion masks and
shows the same subject in two coordinated ways:

1. A medical MRI slice viewer powered by NiiVue.
2. A browser 3D Gaussian splat viewer powered by Three.js.

The red lesion highlight comes from the provided expert lesion mask in the
dataset. It is not an algorithm guess.

This is for education, research communication, and prototype visualization. It
is not a diagnostic medical device.

## What The Viewer Shows

The main page is:

```text
http://127.0.0.1:8081/viewer/app.html?subject=sub-00058&mode=sync
```

It has three tabs:

| Tab | What it does |
| --- | --- |
| MRI Viewer | Shows the real NIfTI MRI slices and the provided expert lesion mask. |
| 3D Lesion Splat | Shows the same MRI anatomy as a 3D Gaussian splat with the expert lesion mask highlighted. |
| Synced MRI + 3D | Shows both views for the same subject and provides remote controls for the 3D splat camera. |

The right panel shows:

- lesion subtype,
- lesion volume,
- lesion voxel count,
- lesion center,
- lesion bounds,
- lesion data source,
- similar comparison subjects,
- plain-language definitions.

The left panel lets users browse all FCD lesion subjects and group them by
histopathology subtype.

## Dataset

The project uses OpenNeuro dataset `ds004199`.

> Schuch F et al. "An open presurgery MRI dataset of people with epilepsy and
> focal cortical dysplasia type II". Scientific Data 10, 475 (2023).
> DOI: https://doi.org/10.1038/s41597-023-02386-7
> OpenNeuro: https://openneuro.org/datasets/ds004199

The dataset contains public, de-identified MRI data, healthy controls, FCD
subjects, clinical metadata, and expert lesion masks.

## Current Local Data Shape

The current local viewer output folder is:

```text
outputs/end_to_end_fcd85_hd
```

Current verified viewer data:

```text
subjects=85
ok=85
bad=0
total_splat_points=7557985
```

The generated local output is about 15-16 GiB. It is not committed into Git.
Git should hold the source code and regeneration instructions. Large generated
MRI outputs should be regenerated locally or shared through object storage.

## Install Locally

These commands assume Linux or WSL Ubuntu.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
npm install
```

Run tests:

```bash
npm test
```

Start the viewer:

```bash
python -m http.server 8081
```

Open:

```text
http://127.0.0.1:8081/viewer/app.html?subject=sub-00058&mode=sync
```

## Regenerate Browser Splats

The browser splats are made from two source-data pieces:

1. the subject MRI foreground anatomy,
2. the provided expert lesion mask.

They do not use residuals, heatmaps, or candidate-region guesses.

Regenerate all 85 FCD browser splats:

```bash
python scripts/regenerate_browser_splats.py \
  --root outputs/end_to_end_fcd85_hd \
  --anatomy-points 85000 \
  --lesion-points 35000 \
  --percentile-threshold 10
```

Verify the generated viewer data:

```bash
python scripts/verify_viewer_dataset.py --root outputs/end_to_end_fcd85_hd
```

## Sharing The Full Project

The private GitHub repository should contain:

- source code,
- viewer code,
- tests,
- regeneration instructions,
- dataset attribution.

The private GitHub repository should not contain:

- generated 15-16 GiB MRI output folders,
- local virtual environments,
- credentials,
- API keys,
- cloud tokens.

Recommended data-sharing options:

1. Regenerate locally from OpenNeuro using the commands above.
2. Share `outputs/end_to_end_fcd85_hd` through private object storage.
3. Publish a split archive as release assets if a compact demo bundle is needed.

GitHub's regular Git storage is not a good place for 16 GiB of generated binary
viewer outputs.

## Important Files

| Path | Purpose |
| --- | --- |
| `viewer/app.html` | Main educational product shell. |
| `viewer/index.html` | NiiVue MRI slice viewer. |
| `viewer/splat.html` | Three.js 3D lesion splat viewer. |
| `scripts/regenerate_browser_splats.py` | Regenerates responsive browser splats from MRI plus expert lesion mask. |
| `scripts/verify_viewer_dataset.py` | Verifies that all viewer subjects have required artifacts. |
| `src/fcd_prototype/ds004199_download.py` | OpenNeuro dataset download helper. |

## Validation

Current test command:

```bash
npm test
```

Current expected result:

```text
688 passed
subjects=85
ok=85
bad=0
total_splat_points=7557985
```
