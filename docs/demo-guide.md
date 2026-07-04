# Demo Guide

## Start

Run:

```bash
python -m http.server 8081
```

Open:

```text
http://127.0.0.1:8081/viewer/app.html?subject=sub-00058&mode=sync
```

## Demo Structure

### 1. Explain The Subject List

Show the left panel.

Say:

> The left panel lists public FCD MRI subjects from OpenNeuro. We can group
> them by histopathology subtype, so Type IIa, Type IIb, and subjects without a
> listed subtype are easier to compare.

### 2. Show The MRI

Click `MRI Viewer`.

Say:

> This is the real MRI in a medical-image viewer called NiiVue. The red overlay
> is the expert lesion mask that came with the dataset.

### 3. Show The 3D Lesion Splat

Click `3D Lesion Splat`.

Say:

> This is the same subject translated into a browser 3D Gaussian splat. The
> pale tissue gives anatomy context. The red-orange highlight comes from the
> same expert lesion mask.

### 4. Show Synced MRI + 3D

Click `Synced MRI + 3D`.

Say:

> This tab keeps the MRI and the 3D splat on the same subject. The remote
> controls move the 3D camera while the MRI stays available as the source view.

### 5. Show Measurements

Point to the lesion measurement card.

Say:

> These values come from the dataset and the expert mask: subtype, lesion
> volume, voxel count, center, and bounds. This makes the visual easier to
> explain without asking someone to interpret raw metadata.

### 6. Show Similar Subjects

Point to the comparison gallery.

Say:

> These are nearby examples ranked by shared subtype, lobe, and side. This is
> useful for teaching because one lesion can be compared against similar cases.

### 7. Export

Click `Export Snapshot`.

Say:

> The export button captures the current teaching screen and saves a metadata
> file with the subject, lesion measurements, and viewer links.

The browser will ask which tab or window to capture. Choose the current viewer
tab.
