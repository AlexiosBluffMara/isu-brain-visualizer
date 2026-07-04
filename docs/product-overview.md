# Product Overview

ISU Brain Visualizer is an educational browser tool for viewing public Focal
Cortical Dysplasia MRI data.

The product shows each subject in two linked forms:

1. MRI slices in NiiVue.
2. A 3D lesion-highlighted Gaussian splat in Three.js.

The lesion highlight comes from the expert lesion mask included in OpenNeuro
dataset `ds004199`. It is not an automatic diagnosis and not an algorithmic
lesion guess.

## Main User Flow

1. Open `viewer/app.html`.
2. Select a subject from the left panel.
3. Choose one of three tabs:
   - MRI Viewer
   - 3D Lesion Splat
   - Synced MRI + 3D
4. Read the lesion measurement card.
5. Compare similar subjects in the comparison gallery.
6. Export a teaching snapshot if needed.

## Viewer Components

| Component | File | Purpose |
| --- | --- | --- |
| Product shell | `viewer/app.html` | Coordinates subject selection, tabs, measurement card, comparison gallery, and export. |
| MRI viewer | `viewer/index.html` | Displays NIfTI MRI slices and the provided expert lesion mask. |
| 3D viewer | `viewer/splat.html` | Displays the browser-ready lesion-highlighted Gaussian splat. |

## Data Source Rule

The current product uses source data only:

- MRI anatomy from the subject MRI.
- Lesion highlight from the provided expert lesion mask.

The current product does not use generated residual overlays, heatmaps, or
algorithm-proposed candidate regions in the teaching viewer.
