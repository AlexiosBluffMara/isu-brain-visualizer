# Demo Guide

## Start

Run:

```bash
python -m http.server 8081
```

Open:

```text
http://127.0.0.1:8081/viewer/app.html?subject=sub-00058&mode=mri2d
```

## Demo Structure

### 1. Explain The Case Data Panel

Show the left panel.

Say:

> The left panel is the case data panel. It shows the selected public OpenNeuro
> subject, the dataset, scanner and acquisition metadata, lesion ROI
> measurements, and the 3D Gaussian splat point count. The subtype dropdown
> lets us filter similar MRI examples by Type IIa, Type IIb, or subtype not
> listed.

### 2. Show The 2D MRI

Click `2D MRI Viewer`.

Say:

> This is the real MRI in a medical-image viewer called NiiVue. It shows the
> coronal, sagittal, and axial slice views. The red overlay is the expert
> lesion mask that came with the dataset.

### 3. Show The 3D MRI View

Click `3D MRI View`.

Say:

> This is still NiiVue. It renders a direct MRI volume from NIfTI voxel data.
> For many subjects, this 3D view uses the T1w anatomy scan because it is a
> smoother 1 mm 3D scan. The expert lesion mask still comes from the FLAIR-space
> dataset label and is placed by MRI coordinates. This is not a Gaussian splat.

### 4. Show The 3D Gaussian Splat View

Click `3D Gaussian Splat View`.

Say:

> This is the same subject converted into a browser 3D Gaussian splat. The
> orange highlight comes from the same expert lesion mask. This view is useful
> as an alternate 3D teaching visualization, but it is not more medically
> authoritative than the NiiVue MRI views.

### 5. Show Measurements

Point to the left-side lesion ROI and 3D Gaussian Splat cards.

Say:

> These values come from the dataset and the expert mask: subtype, lesion
> volume, voxel count, center, bounds, and splat point count. This makes the
> visual easier to explain without asking someone to interpret raw metadata.

### 6. Show Similar Subjects

Use the subtype dropdown and point to the similar MRI examples list.

Say:

> Changing the subtype filter immediately changes which public MRI examples are
> visible. This is useful for teaching because one lesion can be compared
> against similar cases.

### 7. Export

Click `Export Snapshot`.

Say:

> The export button captures the current teaching screen and saves a metadata
> file with the subject, lesion measurements, and viewer links.

The browser will ask which tab or window to capture. Choose the current viewer
tab.
