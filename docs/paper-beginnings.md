# ISU Brain Visualizer: Educational 2D, 3D MRI, and 3D Gaussian Splat Visualization of Expert-Labeled Focal Cortical Dysplasia Lesions

## Working Title

ISU Brain Visualizer: A Browser-Based Educational Viewer for Expert-Labeled Focal Cortical Dysplasia MRI Data Using NiiVue and 3D Gaussian Splatting

## Plain-English Core Idea

Focal cortical dysplasia is a small area of abnormal brain development that can cause epilepsy. The difficult part is that the abnormal area can be subtle on MRI. It may look almost like normal brain tissue, especially near the boundary between gray matter and white matter. If clinicians cannot clearly see the lesion, surgery planning becomes harder.

This project does not try to diagnose new patients. Instead, it starts with public MRI cases where experts have already marked the lesion. The viewer then shows the same case in three ways:

1. Real 2D MRI slices, so the original medical image can be inspected.
2. A direct 3D MRI volume render, so the lesion can be understood as part of the whole head or brain volume.
3. A 3D Gaussian Splat view, so the expert lesion mask can be shown as a separate, interactive 3D teaching object.

The purpose is education and research communication: make the known lesion easier to see, easier to explain, and easier to compare across subjects.

## Abstract Draft

Focal cortical dysplasia (FCD) is a malformation of cortical development and an important cause of drug-resistant focal epilepsy. Detection on standard magnetic resonance imaging (MRI) remains challenging because FCD-related abnormalities can be small, spatially complex, and visually subtle. Imaging features such as cortical thickening, gray-white matter junction blurring, abnormal gyration, and FLAIR hyperintensity may be difficult to identify consistently, even for trained readers. Publicly available, expert-labeled MRI datasets provide an opportunity to develop transparent educational and research tools that improve visual understanding of these lesions without making unsupported diagnostic claims.

We present the beginning of ISU Brain Visualizer, a browser-based educational tool for viewing OpenNeuro dataset `ds004199`, an open presurgery MRI dataset containing 85 people with epilepsy due to FCD type II and 85 healthy controls. The dataset includes 3D-T1, 3D-FLAIR, expert-labeled lesion regions of interest, and clinical metadata. Our prototype presents each FCD subject using three coordinated views: NiiVue-based 2D MRI slice review, NiiVue-based direct 3D MRI volume rendering, and a Three.js-based 3D Gaussian Splat representation derived from the source MRI anatomy and the provided expert lesion mask. The system emphasizes provenance: red indicates the dataset-provided expert lesion mask in the MRI view, while orange indicates the same expert mask after conversion into a 3D Gaussian Splat teaching visualization.

This work is not a diagnostic model and does not infer new lesions. Its contribution is a reproducible visual interface for teaching, stakeholder communication, and early-stage exploration of how expert-labeled FCD lesions can be represented across standard medical-image viewing and newer interactive 3D rendering methods.

## 1. Introduction Draft

Focal cortical dysplasia is a localized abnormality in brain cortical development. It is strongly associated with drug-resistant epilepsy, particularly focal epilepsy. In plain terms, this means that the affected brain area can repeatedly trigger seizures, and medication may not be enough to control them. In these cases, epilepsy surgery can be considered, but surgery depends on finding the relevant brain area accurately.

MRI is central to this process because it gives clinicians a noninvasive way to inspect brain structure. However, FCD can be difficult to see. The abnormalities may be subtle: the cortex may be slightly thickened, the normal boundary between gray matter and white matter may be blurred, the folds of the brain may look slightly unusual, or a signal change may extend through white matter toward the ventricle. These visual clues are medically meaningful, but they can be easy to miss when the lesion is small or when image quality and slice thickness vary.

This challenge is reflected in the clinical literature. Schuch et al. describe FCD as a frequent cause of drug-resistant focal epilepsy and note that FCD abnormalities often escape conventional neuroradiological diagnosis because they can be very subtle. A practical neurology review similarly reports that almost one-third of patients with FCD type II may remain MRI-negative, in some cases because imaging is inadequate. In a 2022 Journal of Clinical Neurology study of 87 postsurgical FCD cases, radiologists identified MRI findings correlated with FCD in 34 cases, or 39.1%. A 2025 reader-performance benchmark using openly available FCD data reported expert detection of 68% overall and 49% on a difficult subset. These figures do not mean that all FCD lesions are invisible. They mean that visual detection is variable enough that clearer educational tools and reproducible viewing workflows matter.

Our project begins from that practical problem. We do not claim that a visualization tool can replace neuroradiology review, surgical planning, electroencephalography, or clinical judgment. Instead, we ask a narrower question: if an expert lesion mask is already available, can we make it easier for students, researchers, and stakeholders to understand where that lesion is, how large it is, and how it sits inside the MRI volume?

## 2. Dataset Selection Draft

We selected OpenNeuro dataset `ds004199`, published by Schuch et al. as "An open presurgery MRI dataset of people with epilepsy and focal cortical dysplasia type II." This dataset is well suited for the current project for five reasons.

First, the dataset is public and de-identified, which allows the tool to be demonstrated and reproduced without using private hospital data. Second, it is directly focused on the condition of interest: FCD type II. Third, it includes both FCD subjects and healthy controls. Fourth, it includes MRI modalities relevant to lesion visualization, including T1-weighted and FLAIR imaging. Fifth, and most important for this prototype, it includes manually labeled lesion regions of interest. Those labels allow us to show a known expert-marked lesion instead of pretending that our prototype has discovered one.

The current viewer output uses the FCD subject set and exposes metadata for each case, including subject identifier, modality availability, acquisition metadata, voxel spacing, lesion volume, lesion voxel count, lesion center, lesion bounds, subtype when available, and 3D Gaussian Splat point counts. This makes the viewer useful not only as a picture viewer but as a case-review surface.

## 3. Visualization Method Draft

The viewer has three tabs because each tab answers a different teaching question.

The first tab is the 2D MRI Viewer. It shows axial, coronal, and sagittal slice views using NiiVue. This is the closest view to normal medical-image review. It is important because it keeps the original MRI visible. The red overlay is the dataset-provided expert lesion mask. This view answers: "Where is the lesion on the real MRI slices?"

The second tab is the 3D MRI View. It also uses NiiVue, but instead of only showing 2D slices, it uses direct 3D volume rendering from the NIfTI voxel data. For many subjects, this 3D view uses the isotropic T1w anatomy scan when available because it is often smoother and more complete for whole-volume viewing than thick-slice FLAIR. The lesion mask remains the expert dataset label, placed by MRI coordinates. This view answers: "Where is the lesion in the full head or brain volume?"

The third tab is the 3D Gaussian Splat View. This view converts MRI-derived anatomy and the expert lesion mask into a browser-rendered 3D Gaussian Splat representation. A Gaussian splat is a soft 3D point with position, opacity, color, and size. Many of these points together can form a navigable 3D object. In our viewer, the anatomy is shown lightly for context, while the lesion splat is orange. This view answers: "What does the expert-marked lesion look like as an isolated interactive 3D teaching object?"

This use of 3D Gaussian Splatting is intentionally conservative. The original 3DGS method was introduced for real-time radiance-field rendering, where scenes are represented using many anisotropic 3D Gaussians and rendered efficiently. Our prototype borrows the visualization idea, but it does not claim to reconstruct a photorealistic patient brain from camera views. Instead, it uses source MRI voxel coordinates and the expert lesion mask to create an alternate 3D representation for education.

## 4. Educational Advantage Draft

The educational value comes from showing the same lesion in multiple forms without changing the source truth.

In 2D, students can inspect the real MRI slices and learn that FCD findings can be subtle. In 3D MRI, they can understand where the lesion sits in the larger anatomy. In 3D Gaussian Splat form, they can rotate the lesion-focused representation freely and see the expert mask as a spatial object. The combination is useful because each view has a different weakness. A 2D slice can be clinically familiar but mentally difficult to assemble into 3D. A 3D volume render is spatially intuitive but depends heavily on opacity and contrast settings. A 3D Gaussian Splat view is interactive and lightweight, but it is a conversion, not the original medical image.

By making those differences explicit, the viewer teaches a careful lesson: visualizations are tools for understanding, not replacements for source data. The source MRI and expert mask remain the authoritative materials. The 3D views help people reason about those materials.

## 5. Current Prototype Scope Draft

The current prototype supports:

- Loading the OpenNeuro FCD subject set.
- Viewing 2D MRI slices with the expert lesion mask.
- Viewing direct 3D MRI volume rendering.
- Viewing lesion-highlighted 3D Gaussian Splats.
- Displaying subject, acquisition, geometry, lesion, and splat metadata.
- Filtering subjects by FCD subtype.
- Exporting teaching snapshot metadata.
- Using consistent color semantics: red for expert lesion ROI, orange for 3D Gaussian Splat, and blue for subject/acquisition metadata.

The current prototype does not:

- Diagnose new lesions.
- Generate clinical reports.
- Replace neuroradiology review.
- Infer heatmaps, residuals, or candidate lesion regions.
- Use synthetic-normal subtraction as a validated detection method.

## 6. References For The Draft

Schuch F, Walger L, Schmitz MH, et al. An open presurgery MRI dataset of people with epilepsy and focal cortical dysplasia type II. Scientific Data. 2023;10:475. https://doi.org/10.1038/s41597-023-02386-7

OpenNeuro dataset ds004199. An open presurgery MRI dataset of people with epilepsy and focal cortical dysplasia type II. https://openneuro.org/datasets/ds004199

Balestrini S, Barba C, Thom M, Guerrini R. Focal cortical dysplasia: a practical guide for neurologists. Practical Neurology. 2023. https://discovery.ucl.ac.uk/id/eprint/10166624/

Lee SK, et al. Lesion Detection Through MRI Postprocessing in Pathology-Proven Focal Cortical Dysplasia: Morphometric Analysis Program. Journal of Clinical Neurology. 2022. https://doi.org/10.3988/jcn.2022.0317

Walger L, Schmitz MH, Bauer T, et al. A public benchmark for human performance in the detection of focal cortical dysplasia. Epilepsia Open. 2025;10(3):778-786. https://doi.org/10.1002/epi4.70028

EpilepsyDiagnosis.org. Focal cortical dysplasia imaging. https://www.epilepsydiagnosis.org/aetiology/focal-cortical-dysplasia-imaging.html

Kerbl B, Kopanas G, Leimkühler T, Drettakis G. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics. 2023. https://arxiv.org/abs/2308.04079

NiiVue documentation. Loading volumes and meshes; colormaps. https://niivue.com/docs/loading/ and https://niivue.com/docs/colormaps2/
