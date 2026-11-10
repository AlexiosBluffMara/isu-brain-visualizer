# Data, hosting, downloads, and regeneration

## What is hosted where

| Material | Location | Why |
| --- | --- | --- |
| Viewer, processing code, tests, instructions | Public GitHub repository | Small, reviewable, versioned source |
| Two prepared review cases | v0.2.0 GitHub Release demo ZIP | Quick start; about 57 MiB |
| All 85 prepared FCD viewer cases | Nine independent GitHub Release ZIP packs | About 2.5 GiB unpacked; no giant Git checkout |
| Original MRI, controls, expert masks, clinical metadata | OpenNeuro ds004199 | Authoritative source data |
| Training checkpoints, synthetic normals, residual experiments | Existing local research outputs | Not needed to show the expert-labeled viewer |

The release packages contain everything used by the three teaching views:
normalized FLAIR, normalized T1w where available, the expert ROI, the
lesion-highlighted browser PLY, a minimal manifest, and public case metadata.
They omit redundant training states and machine-specific source paths.

GitHub documents that each release asset must be under 2 GiB and that releases
have no total size or bandwidth limit:
[About releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
Each of our packs is well below that per-file limit.

## Where the source came from

- [OpenNeuro ds004199 v1.0.6](https://openneuro.org/datasets/ds004199/versions/1.0.6)
- [Dataset DOI](https://doi.org/10.18112/openneuro.ds004199.v1.0.6)
- [Original dataset description, including CC0 license](https://github.com/OpenNeuroDatasets/ds004199/blob/1.0.6/dataset_description.json)
- [Schuch et al., Scientific Data 10, 475 (2023)](https://doi.org/10.1038/s41597-023-02386-7)

The source contains 85 FCD participants and 85 healthy controls.
Our prepared teaching set is the 85 FCD cases, for which the expert lesions
can be reviewed. Controls remain available from the source.

The data is in **BIDS** (Brain Imaging Data Structure): each subject has an
`anat` folder holding MRI volumes and scan metadata. MRI files use
`.nii.gz` (compressed NIfTI). Files ending `_roi.nii.gz` hold expert lesion masks.
`participants.tsv` describes the source participants.

## The simplest download

From the code folder:

```powershell
py -3 scripts/run_viewer.py --download-demo
# Or every FCD viewer case:
py -3 scripts/run_viewer.py --download-all
```

On Mac/Linux use `python3` in place of `py -3`.
See [the illustrated run guide](running-the-project.md).

The [release](https://github.com/AlexiosBluffMara/isu-brain-visualizer/releases/tag/v0.2.0)
also has manual ZIP downloads, `SHA256SUMS.txt`, and a copy of the download
catalog. [data/cases.csv](../data/cases.csv) lists each case and its pack.
[data/release-assets.json](../data/release-assets.json) records exact sizes,
subject IDs, and SHA-256 fingerprints.

## Download original MRI from OpenNeuro without code

1. Open the version 1.0.6 page above.
2. Use OpenNeuro's **Download** control and its offered download instructions.
3. Keep `dataset_description.json` and `participants.tsv` with the subject folders.
4. Select the entire dataset for all FCD cases **and healthy controls**, or a
   subject subset if you only need a few original scans.

These originals do not yet contain our browser PLYs or viewer manifests.
Use the ready-made release packs to open the viewer immediately.

## Download source MRI using the project helper

Create and activate a Python virtual environment, then install the processing
package and AWS CLI in it:

```bash
python -m pip install -e ".[dev]"
python -m pip install awscli
```

No AWS account or key is needed: this uses the public OpenNeuro S3 mirror.

Two source subjects:

```bash
fcd-download-ds --destination data/openneuro/ds004199 --subjects sub-00058 sub-00100 --modalities T1w FLAIR --include-roi
```

All source FCD cases:

```bash
fcd-download-ds --destination data/openneuro/ds004199 --all-participants --group fcd --modalities T1w FLAIR --include-roi
```

The complete source cohort, including controls:

```bash
fcd-download-ds --destination data/openneuro/ds004199 --all-participants --group all --modalities T1w FLAIR --include-roi
```

The S3 mirror can move forward as OpenNeuro is updated. For an exact pinned
v1.0.6 source download, follow that version's OpenNeuro instructions instead.
Budget additional disk space for raw scans and processing intermediates.

## Rebuild viewer data from source

Processing requires the Python dependencies; opening an existing release does
not. The direct browser-splat path below does not require CUDA training.

For the two example subjects, run:

```bash
fcd-build-end-to-end --dataset-dir data/openneuro/ds004199 --subjects sub-00058 sub-00100 --output-dir outputs/end_to_end_fcd85_hd --synthetic-mode placeholder --roi-strategy provided
python scripts/regenerate_browser_splats.py --root outputs/end_to_end_fcd85_hd --anatomy-points 85000 --lesion-points 25000 --percentile-threshold 10 --rng-seed 42
```

The first command also produces experimental pipeline artifacts.
The teaching viewer uses only the MRI and expert mask, together with the
browser splat made by the second command. GPU-optimized research outputs are
not required by this path. Historical GPU batch scripts are machine-specific
research records rather than the portable setup route.

If rebuilding a different subset, the launcher reads the index generated by
the pipeline. Remove an old `outputs/fcd85_subjects.txt` only if it belongs to
an unrelated previous dataset, or keep separate checkouts for different sets.

## Rebuild the release packages

After all 85 finished viewer cases exist:

```bash
python scripts/package_release_data.py --root outputs/end_to_end_fcd85_hd --output-dir dist/release-data
```

This builds the two-case demo, nine full-data packs, checksums, the download
catalog, and the case CSV. Pack checksums change when ZIPs are rebuilt; publish
the generated catalog and matching ZIPs together under a new release version
when the contents change. Do not replace published files silently.

## Publication order

1. Validate all 85 source manifests and viewer artifacts.
2. Build and check the demo plus full-case packs.
3. Update the repository's download catalog to match those exact files.
4. Test a fresh code copy with the prepared demo.
5. Commit the viewer and documentation changes.
6. Upload the data as draft release assets, then publish the code and release.
7. Test a download using the public URLs.

The code repository is a source distribution. It is not a publicly hosted live
viewer: people run the viewer locally using the release data.
