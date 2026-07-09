# Product Overview

ISU Brain Visualizer is an educational browser tool for viewing public Focal
Cortical Dysplasia MRI data.

The product shows each subject in three linked forms:

1. 2D MRI slices in NiiVue.
2. A direct 3D MRI volume render in NiiVue, using the isotropic T1w anatomy scan when available.
3. A lesion-highlighted 3D Gaussian splat in Three.js.

The lesion highlight comes from the expert lesion mask included in OpenNeuro
dataset `ds004199`. It is not an automatic diagnosis and not an algorithmic
lesion guess.

## Main User Flow

1. Open `viewer/app.html`.
2. Select a subject from the left panel.
3. Choose one of three tabs:
   - 2D MRI Viewer
   - 3D MRI View
   - 3D Gaussian Splat View
4. Read the left-side subject, acquisition, lesion ROI, and 3D splat metadata.
5. Compare similar subjects with the dataset/subtype dropdown.
6. Export a teaching snapshot if needed.

## Viewer Components

| Component | File | Purpose |
| --- | --- | --- |
| Product shell | `viewer/app.html` | Coordinates subject selection, tabs, metadata, definitions, color legend, and export. |
| MRI viewer | `viewer/index.html` | Displays NIfTI MRI slices, direct 3D MRI volume rendering, and the provided expert lesion mask. |
| 3D Gaussian splat viewer | `viewer/splat.html` | Displays the browser-ready lesion-highlighted Gaussian splat. |

## Data Source Rule

The current product uses source data only:

- MRI anatomy from the subject MRI.
- Lesion highlight from the provided expert lesion mask.
- Direct 3D MRI view from the subject NIfTI voxel data, preferring isotropic T1w anatomy when available.
- 3D Gaussian splat view as a converted visualization of the same source data.

The current product does not use generated residual overlays, heatmaps, or
algorithm-proposed candidate regions in the teaching viewer.
