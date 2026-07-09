import {
  DEFAULT_ROOT,
  DEFAULT_SUBJECT,
  DEMO_SUBJECTS,
  TERM_EXPLANATIONS,
  expandGroup,
  expandHistopathology,
  filterSubjects,
  formatLocation,
  formatPoints,
  formatVolume,
  manifestUrl,
  nextSubjectId,
  niivueUrl,
  normalizeSubjects,
  parseSubjectText,
  splatUrl,
} from "./app-core.js?v=20260705-t1w-3d-mri";

const params = new URLSearchParams(window.location.search);
const requestedMode = params.get("mode");
const modeAliases = {
  mri: "mri2d",
  mri2d: "mri2d",
  mri3d: "mri3d",
  splat: "splat",
};

const state = {
  root: new URLSearchParams(window.location.search).get("root") || DEFAULT_ROOT,
  subject: new URLSearchParams(window.location.search).get("subject") || DEFAULT_SUBJECT,
  mode: modeAliases[requestedMode] || "mri2d",
  groupFilter: "all",
  subjects: [],
  manifests: {},
  filtered: [],
};

const els = {
  search: document.getElementById("subject-search"),
  list: document.getElementById("subject-list"),
  status: document.getElementById("status-line"),
  niivueFrame: document.getElementById("niivue-frame"),
  splatFrame: document.getElementById("splat-frame"),
  niivueWrap: document.getElementById("niivue-wrap"),
  splatWrap: document.getElementById("splat-wrap"),
  shared3dControls: document.getElementById("shared-3d-controls"),
  niivueLoading: document.getElementById("niivue-loading"),
  splatLoading: document.getElementById("splat-loading"),
  viewerGrid: document.getElementById("viewer-grid"),
  viewerTitle: document.getElementById("viewer-title"),
  viewerSubtitle: document.getElementById("viewer-subtitle"),
  mriSourceLabel: document.getElementById("mri-source-label"),
  groupFilterSelect: document.getElementById("group-filter"),
  subjectCardTitle: document.getElementById("subject-card-title"),
  subjectDataset: document.getElementById("subject-dataset"),
  subjectSubtype: document.getElementById("subject-subtype"),
  subjectLocation: document.getElementById("subject-location"),
  subjectModalities: document.getElementById("subject-modalities"),
  acquisitionManufacturer: document.getElementById("acquisition-manufacturer"),
  acquisitionModel: document.getElementById("acquisition-model"),
  acquisitionField: document.getElementById("acquisition-field"),
  acquisitionProtocol: document.getElementById("acquisition-protocol"),
  acquisitionSpacing: document.getElementById("acquisition-spacing"),
  lesionCardTitle: document.getElementById("lesion-card-title"),
  lesionVolume: document.getElementById("lesion-volume"),
  lesionVoxels: document.getElementById("lesion-voxels"),
  lesionCenter: document.getElementById("lesion-center"),
  lesionBounds: document.getElementById("lesion-bounds"),
  lesionSource: document.getElementById("lesion-source"),
  splatPoints: document.getElementById("splat-points"),
  splatSource: document.getElementById("splat-source"),
  viewGuideTitle: document.getElementById("view-guide-title"),
  viewGuideCopy: document.getElementById("view-guide-copy"),
  viewGuideList: document.getElementById("view-guide-list"),
  openCurrent: document.getElementById("open-current"),
  openCurrentButton: document.getElementById("open-current-button"),
  subjectPanel: document.getElementById("subject-panel"),
  guidePanel: document.getElementById("guide-panel"),
  toggleSubjectPanel: document.getElementById("toggle-subject-panel"),
  toggleGuidePanel: document.getElementById("toggle-guide-panel"),
  main: document.querySelector(".main"),
};

function setStatus(text) {
  els.status.textContent = text;
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  return response.json();
}

async function loadSubjects() {
  setStatus("Loading the 85-subject FCD dataset list...");
  let subjectIds = [];
  try {
    const textResponse = await fetch("/outputs/fcd85_subjects.txt", { cache: "no-store" });
    if (textResponse.ok) subjectIds = parseSubjectText(await textResponse.text());
  } catch (_) {}

  if (!subjectIds.length) {
    const index = await fetchJson(`${state.root}/index.json`);
    subjectIds = (index.subjects || []).map((entry) => entry.subject || entry);
  }

  const demoFirst = [
    ...DEMO_SUBJECTS,
    ...subjectIds.filter((subject) => !DEMO_SUBJECTS.includes(subject)),
  ];

  await Promise.all(demoFirst.map(async (subject) => {
    try {
      state.manifests[subject] = await fetchJson(manifestUrl({ root: state.root, subject }));
    } catch (error) {
      console.warn(`Manifest failed for ${subject}`, error);
    }
  }));

  state.subjects = normalizeSubjects(demoFirst, state.manifests);
  state.filtered = state.subjects;
  renderSubjectList();
  renderTermList();
  setStatus(`Loaded ${state.subjects.length} FCD subjects from ${state.root}.`);
}

function renderTermList() {
  const list = document.getElementById("term-list");
  list.innerHTML = Object.entries(TERM_EXPLANATIONS)
    .map(([term, explanation]) => `
      <div class="term" title="${escapeHtml(explanation)}">
        <strong>${escapeHtml(term)}</strong>
        <p>${escapeHtml(explanation)}</p>
      </div>
    `)
    .join("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderSubjectList() {
  const query = els.search.value;
  const searched = filterSubjects(state.subjects, query);
  state.filtered = searched.filter((subject) => {
    if (state.groupFilter === "all") return true;
    if (state.groupFilter === "unknown") return !subject.histopathology;
    return subject.histopathology === state.groupFilter;
  });

  const grouped = groupSubjectsForList(state.filtered);
  els.list.innerHTML = grouped.map((group) => `
    <div class="subject-group">
      <div class="subject-group-title">${escapeHtml(group.label)}</div>
      ${group.subjects.map((subject) => renderSubjectButton(subject)).join("")}
    </div>
  `).join("");

  for (const button of els.list.querySelectorAll("button[data-subject]")) {
    button.addEventListener("click", () => selectSubject(button.dataset.subject));
  }
}

function groupSubjectsForList(subjects) {
  const buckets = new Map();
  for (const subject of subjects || []) {
    const label = expandHistopathology(subject.histopathology);
    if (!buckets.has(label)) buckets.set(label, []);
    buckets.get(label).push(subject);
  }
  return [...buckets.entries()].map(([label, groupSubjects]) => ({ label, subjects: groupSubjects }));
}

function renderSubjectButton(subject) {
    const active = subject.id === state.subject ? " active" : "";
    const location = formatLocation(subject.hemisphere, subject.lobe);
    const group = expandGroup(subject.group);
    const histopathology = expandHistopathology(subject.histopathology);
    return `
      <button class="subject-button${active}" data-subject="${subject.id}" title="Open ${subject.id} in the MRI and lesion-splat viewers.">
        <span class="subject-id">${subject.id}</span>
        <span class="subject-meta">${escapeHtml(location)} | ${escapeHtml(group)} | ${escapeHtml(histopathology)} | ${formatVolume(subject.lesionVolumeMm3)}</span>
      </button>
    `;
}

function currentSubjectMeta() {
  return state.subjects.find((subject) => subject.id === state.subject) || {};
}

function known(value, fallback = "unknown") {
  return value == null || value === "" ? fallback : String(value);
}

function formatTesla(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(2)} T` : "unknown";
}

function formatSpacing(value) {
  return Array.isArray(value)
    ? `${value.map((item) => Number(item).toFixed(2)).join(" x ")} mm`
    : "unknown";
}

const MRI_BASE_LABELS = {
  flair: "FLAIR",
  t1w: "T1w",
  t2w: "T2w",
};

function artifactsForManifest(manifest) {
  return manifest?.artifacts || manifest || {};
}

function hasMriArtifact(manifest, base) {
  return Boolean(artifactsForManifest(manifest)[`patient_${base}`]);
}

function resolveMriBase(manifest, preferredBase) {
  if (hasMriArtifact(manifest, preferredBase)) return preferredBase;
  for (const fallback of ["flair", "t2w", "t1w"]) {
    if (hasMriArtifact(manifest, fallback)) return fallback;
  }
  return preferredBase;
}

function preferredMriBaseForMode(mode) {
  return mode === "mri3d" ? "t1w" : "flair";
}

function mriSourceLabelForMode(mode, manifest) {
  if (mode === "splat") {
    return "3D Gaussian Splat source: MRI anatomy plus the provided expert lesion mask.";
  }

  const preferred = preferredMriBaseForMode(mode);
  const actual = resolveMriBase(manifest, preferred);
  const label = MRI_BASE_LABELS[actual] || actual.toUpperCase();
  const fallback = actual !== preferred
    ? ` ${MRI_BASE_LABELS[preferred] || preferred.toUpperCase()} was not available for this subject, so the viewer is using the available ${label} scan.`
    : "";

  if (mode === "mri3d") {
    return `MRI source: ${label}. This 3D MRI view uses the anatomy scan loaded from the subject NIfTI file.${fallback}`;
  }
  return `MRI source: ${label}. This 2D MRI view is the lesion-review scan; the red overlay is the provided expert lesion mask.${fallback}`;
}

function renderMetrics() {
  const subject = currentSubjectMeta();
  const manifest = state.manifests[state.subject] || {};
  const viewerMeta = manifest.viewer_meta || {};
  const scan = viewerMeta.scan || {};
  const geometry = viewerMeta.geometry || {};
  const roi = manifest.viewer_meta?.roi || {};
  const location = formatLocation(subject.hemisphere, subject.lobe);

  els.subjectCardTitle.textContent = state.subject;
  els.subjectDataset.textContent = "OpenNeuro ds004199";
  els.subjectSubtype.textContent = expandHistopathology(subject.histopathology);
  els.subjectLocation.textContent = location;
  els.subjectModalities.textContent = (manifest.modalities || subject.modalities || [scan.modality])
    .filter(Boolean)
    .join(", ") || "unknown";

  els.acquisitionManufacturer.textContent = known(scan.Manufacturer);
  els.acquisitionModel.textContent = known(scan.ManufacturersModelName);
  els.acquisitionField.textContent = formatTesla(scan.MagneticFieldStrength);
  els.acquisitionProtocol.textContent = known(scan.ProtocolName || scan.SeriesDescription);
  els.acquisitionSpacing.textContent = formatSpacing(geometry.voxel_spacing_mm);

  const copy = {
    mri2d: {
      title: `${state.subject} 2D MRI Viewer`,
      subtitle: "Three 2D slice views from the real NIfTI MRI: coronal, sagittal, and axial. The red overlay is the confirmed expert lesion mask.",
      guideTitle: "2D MRI Viewer",
      guideCopy: "This is the safest starting point. It shows the actual MRI slices and the lesion label from the public dataset.",
      guideItems: [
        "Best for checking the source MRI data directly.",
        "Best for seeing exactly where the red expert mask sits on the scan.",
        "Limited because you mentally assemble the 3D shape from separate 2D slices.",
      ],
    },
    mri3d: {
      title: `${state.subject} 3D MRI View`,
      subtitle: "NiiVue renders a direct 3D MRI volume from NIfTI voxel data. This view uses the isotropic T1w anatomy scan when available, with the expert lesion mask placed by MRI coordinates.",
      guideTitle: "3D MRI View",
      guideCopy: "This view is a direct MRI volume render, not a Gaussian splat. For many subjects, T1w is better for full 3D anatomy because it is a 1 mm 3D scan, while FLAIR is often thick-sliced.",
      guideItems: [
        "Best for understanding the lesion's position on the full head or brain volume.",
        "Uses the source NIfTI voxels directly; it is not reconstructed from splat points.",
        "T1w anatomy is used for the 3D volume when available because it avoids the half-skull look caused by thick-slice 2D FLAIR.",
        "Rendering opacity and contrast affect how the volume looks, so it is still a visualization.",
      ],
    },
    splat: {
      title: `${state.subject} 3D Gaussian Splat View`,
      subtitle: "Three.js shows a converted 3D Gaussian splat representation. Orange marks the expert lesion mask after conversion; this is an alternate visualization, not the source MRI.",
      guideTitle: "3D Gaussian Splat View",
      guideCopy: "This converts the MRI-derived anatomy and lesion mask into many soft 3D points. It is useful for teaching and future experiments, but it is not more medically authoritative than the MRI.",
      guideItems: [
        "Useful for seeing a light, browser-friendly 3D representation.",
        "Useful for explaining how lesion shape can be translated into a different 3D format.",
        "Less direct than NiiVue because it is converted from the MRI instead of rendering the NIfTI voxels directly.",
      ],
    },
  };
  const activeCopy = copy[state.mode] || copy.mri2d;
  els.viewerTitle.textContent = activeCopy.title;
  els.viewerSubtitle.textContent = activeCopy.subtitle;
  els.mriSourceLabel.textContent = mriSourceLabelForMode(state.mode, manifest);
  els.mriSourceLabel.classList.toggle("splat-source", state.mode === "splat");
  els.viewGuideTitle.textContent = activeCopy.guideTitle;
  els.viewGuideCopy.textContent = activeCopy.guideCopy;
  els.viewGuideList.innerHTML = activeCopy.guideItems
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("");
  renderLesionCard(subject, manifest, roi);
}

function formatNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString() : "unknown";
}

function formatIjk(value) {
  return Array.isArray(value) ? value.map((item) => Number(item).toFixed(0)).join(", ") : "unknown";
}

function renderLesionCard(subject, manifest, roi) {
  const profile = manifest.browser_splat_profile || {};
  els.lesionCardTitle.textContent = `${state.subject} expert lesion mask`;
  els.lesionVolume.textContent = formatVolume(subject.lesionVolumeMm3);
  els.lesionVoxels.textContent = formatNumber(roi.voxel_count);
  els.lesionCenter.textContent = formatIjk(roi.center_ijk);
  const min = formatIjk(roi.bounds_min_ijk);
  const max = formatIjk(roi.bounds_max_ijk);
  els.lesionBounds.textContent = min === "unknown" || max === "unknown" ? "unknown" : `${min} to ${max}`;
  els.lesionSource.textContent = profile.source_truth_for_lesion === "provided_expert_lesion_mask"
    ? "Provided expert lesion mask"
    : "Expert mask";
  els.splatPoints.textContent = formatPoints(subject.splatPoints);
  els.splatSource.textContent = profile.source_truth_for_lesion === "provided_expert_lesion_mask"
    ? "MRI anatomy plus provided expert lesion mask"
    : "MRI-derived conversion";
}

function setMode(mode) {
  state.mode = modeAliases[mode] || "mri2d";
  for (const button of document.querySelectorAll("[data-mode]")) {
    button.setAttribute("aria-pressed", button.dataset.mode === state.mode ? "true" : "false");
  }

  els.niivueWrap.classList.toggle("hidden", state.mode === "splat");
  els.splatWrap.classList.toggle("hidden", state.mode !== "splat");
  els.shared3dControls.classList.toggle("hidden", !is3dMode());
}

function is3dMode() {
  return state.mode === "mri3d" || state.mode === "splat";
}

function active3dFrame() {
  return state.mode === "splat" ? els.splatFrame : els.niivueFrame;
}

function sendCameraCommand(command) {
  if (!is3dMode()) return;
  const frame = active3dFrame();
  frame.contentWindow?.postMessage({
    type: "isu-brain-visualizer:camera-command",
    command,
  }, window.location.origin);
}

function cameraCommandFromKey(event) {
  const key = event.key.toLowerCase();
  if (key === "w" || event.key === "ArrowUp") return "rotate-up";
  if (key === "s" || event.key === "ArrowDown") return "rotate-down";
  if (key === "a" || event.key === "ArrowLeft") return "rotate-left";
  if (key === "d" || event.key === "ArrowRight") return "rotate-right";
  if (key === "q") return "zoom-out";
  if (key === "e") return "zoom-in";
  return null;
}

function refreshFrames() {
  renderMetrics();
  renderSubjectList();
  const manifest = state.manifests[state.subject] || {};
  const mriLayout = state.mode === "mri3d" ? "4" : "3-2d";
  const mriBase = resolveMriBase(manifest, preferredMriBaseForMode(state.mode));
  const mri = niivueUrl({ root: state.root, subject: state.subject, embed: true, layout: mriLayout, base: mriBase });
  const splat = splatUrl({ root: state.root, subject: state.subject, splat: "lesion-highlighted", embed: true });
  const mriFull = niivueUrl({ root: state.root, subject: state.subject, layout: mriLayout, base: mriBase });
  const splatFull = splatUrl({ root: state.root, subject: state.subject, splat: "lesion-highlighted" });
  els.niivueLoading.classList.add("active");
  els.splatLoading.classList.add("active");
  els.niivueFrame.src = mri;
  els.splatFrame.src = splat;
  const activeUrl = state.mode === "splat" ? splatFull : mriFull;
  const activeLabel = state.mode === "mri2d"
    ? "Open 2D MRI in Full Viewer"
    : state.mode === "mri3d"
      ? "Open 3D MRI in Full Viewer"
      : "Open 3D Gaussian Splat in Full Viewer";
  els.openCurrent.href = activeUrl;
  els.openCurrentButton.textContent = activeLabel;
  els.openCurrentButton.title = `Open the current ${activeLabel.replace("Open ", "").replace(" in Full Viewer", "")} as a separate full-page browser view.`;
  window.history.replaceState(null, "", `/viewer/app.html?root=${encodeURIComponent(state.root)}&subject=${state.subject}&mode=${state.mode}`);
}

function selectSubject(subject) {
  state.subject = subject;
  refreshFrames();
}

function togglePanel(panel, button, mainClass, hiddenLabel) {
  const collapsed = !panel.classList.contains("collapsed");
  panel.classList.toggle("collapsed", collapsed);
  els.main.classList.toggle(mainClass, collapsed);
  button.textContent = collapsed ? "Show" : "Hide";
  button.setAttribute("aria-expanded", collapsed ? "false" : "true");
  button.title = collapsed ? `Show ${hiddenLabel}` : `Hide ${hiddenLabel}`;
}

function wireEvents() {
  els.search.addEventListener("input", renderSubjectList);
  els.niivueFrame.addEventListener("load", () => els.niivueLoading.classList.remove("active"));
  els.splatFrame.addEventListener("load", () => els.splatLoading.classList.remove("active"));
  els.toggleSubjectPanel.addEventListener("click", () => {
    togglePanel(els.subjectPanel, els.toggleSubjectPanel, "subjects-collapsed", "OpenNeuro lesion subjects");
  });
  els.toggleGuidePanel.addEventListener("click", () => {
    togglePanel(els.guidePanel, els.toggleGuidePanel, "guide-collapsed", "plain-language guide");
  });

  for (const button of document.querySelectorAll("[data-mode]")) {
    button.addEventListener("click", () => {
      setMode(button.dataset.mode);
      refreshFrames();
    });
  }

  for (const button of document.querySelectorAll("[data-camera-command]")) {
    button.addEventListener("click", () => sendCameraCommand(button.dataset.cameraCommand));
  }

  document.addEventListener("keydown", (event) => {
    if (!is3dMode()) return;
    const tag = event.target?.tagName?.toLowerCase();
    if (tag === "input" || tag === "select" || tag === "textarea") return;
    const command = cameraCommandFromKey(event);
    if (!command) return;
    sendCameraCommand(command);
    event.preventDefault();
  });

  els.groupFilterSelect.addEventListener("change", () => {
    state.groupFilter = els.groupFilterSelect.value;
    renderSubjectList();
  });

  document.getElementById("export-snapshot").addEventListener("click", exportTeachingSnapshot);

  document.getElementById("prev-subject").addEventListener("click", () => {
    selectSubject(nextSubjectId(state.subjects, state.subject, -1));
  });
  document.getElementById("next-subject").addEventListener("click", () => {
    selectSubject(nextSubjectId(state.subjects, state.subject, 1));
  });
}

async function exportTeachingSnapshot() {
  setStatus("Choose this browser tab or window to export the current teaching snapshot.");
  if (!navigator.mediaDevices?.getDisplayMedia) {
    downloadSnapshotMetadata();
    setStatus("Screen capture is not available in this browser. Exported metadata instead.");
    return;
  }
  let stream = null;
  try {
    stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
    const video = document.createElement("video");
    video.srcObject = stream;
    await video.play();
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);
    const a = document.createElement("a");
    a.download = `${state.subject}-${state.mode}-teaching-snapshot.png`;
    a.href = canvas.toDataURL("image/png");
    a.click();
    downloadSnapshotMetadata();
    setStatus(`Exported snapshot for ${state.subject}.`);
  } catch (error) {
    downloadSnapshotMetadata();
    setStatus(`Screenshot was not captured. Exported metadata: ${error.message}`);
  } finally {
    stream?.getTracks().forEach((track) => track.stop());
  }
}

function downloadSnapshotMetadata() {
  const subject = currentSubjectMeta();
  const manifest = state.manifests[state.subject] || {};
  const mriLayout = state.mode === "mri3d" ? "4" : "3-2d";
  const mriBase = resolveMriBase(manifest, preferredMriBaseForMode(state.mode));
  const payload = {
    title: "ISU Brain Visualizer teaching snapshot",
    subject: state.subject,
    mode: state.mode,
    location: formatLocation(subject.hemisphere, subject.lobe),
    histopathology: expandHistopathology(subject.histopathology),
    lesionVolume: formatVolume(subject.lesionVolumeMm3),
    splatPoints: formatPoints(subject.splatPoints),
    lesionSource: manifest.browser_splat_profile?.source_truth_for_lesion || "provided expert lesion mask",
    mriUrl: niivueUrl({ root: state.root, subject: state.subject, layout: mriLayout, base: mriBase }),
    splatUrl: splatUrl({ root: state.root, subject: state.subject, splat: "lesion-highlighted" }),
    generatedAt: new Date().toISOString(),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.download = `${state.subject}-${state.mode}-teaching-snapshot.json`;
  a.href = URL.createObjectURL(blob);
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

wireEvents();
setMode(state.mode);
loadSubjects()
  .then(refreshFrames)
  .catch((error) => {
    console.error(error);
    setStatus(`Could not load the dataset list: ${error.message}`);
    refreshFrames();
  });
