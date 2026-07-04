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
} from "./app-core.js?v=20260704-ui-clarity";

const state = {
  root: new URLSearchParams(window.location.search).get("root") || DEFAULT_ROOT,
  subject: new URLSearchParams(window.location.search).get("subject") || DEFAULT_SUBJECT,
  mode: new URLSearchParams(window.location.search).get("mode") || "mri",
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
  niivueLoading: document.getElementById("niivue-loading"),
  splatLoading: document.getElementById("splat-loading"),
  viewerGrid: document.getElementById("viewer-grid"),
  viewerTitle: document.getElementById("viewer-title"),
  viewerSubtitle: document.getElementById("viewer-subtitle"),
  syncControls: document.getElementById("sync-controls"),
  metricCount: document.getElementById("metric-count"),
  metricPoints: document.getElementById("metric-points"),
  metricVolume: document.getElementById("metric-volume"),
  metricLocation: document.getElementById("metric-location"),
  lesionCardTitle: document.getElementById("lesion-card-title"),
  lesionSubtype: document.getElementById("lesion-subtype"),
  lesionVolume: document.getElementById("lesion-volume"),
  lesionVoxels: document.getElementById("lesion-voxels"),
  lesionCenter: document.getElementById("lesion-center"),
  lesionBounds: document.getElementById("lesion-bounds"),
  lesionSource: document.getElementById("lesion-source"),
  galleryList: document.getElementById("gallery-list"),
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
      <div class="subject-group-title">${escapeHtml(group.label)} <span>${group.subjects.length}</span></div>
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
        <span class="subject-points">${formatPoints(subject.splatPoints)}</span>
        <span class="subject-meta">${escapeHtml(location)} | ${escapeHtml(group)} | ${escapeHtml(histopathology)} | ${formatVolume(subject.lesionVolumeMm3)}</span>
      </button>
    `;
}

function currentSubjectMeta() {
  return state.subjects.find((subject) => subject.id === state.subject) || {};
}

function renderMetrics() {
  const subject = currentSubjectMeta();
  const manifest = state.manifests[state.subject] || {};
  const roi = manifest.viewer_meta?.roi || {};
  const location = formatLocation(subject.hemisphere, subject.lobe);
  els.metricCount.textContent = `${state.subjects.length}`;
  els.metricPoints.textContent = formatPoints(subject.splatPoints);
  els.metricVolume.textContent = formatVolume(subject.lesionVolumeMm3);
  els.metricLocation.textContent = location;
  const copy = {
    mri: {
      title: `${state.subject} MRI Viewer`,
      subtitle: "NiiVue displays the real NIfTI MRI slices. The red overlay is the provided OpenNeuro expert lesion mask for Focal Cortical Dysplasia.",
    },
    splat: {
      title: `${state.subject} 3D Lesion Splat`,
      subtitle: "Three.js renders MRI-derived Gaussian splats. The highlighted color comes from the OpenNeuro expert lesion mask, so this screen focuses only on the known lesion shape in 3D.",
    },
    sync: {
      title: `${state.subject} synced MRI and 3D lesion splat`,
      subtitle: "Both panels use the same subject. The MRI shows the source slices and expert mask; the 3D remote controls adjust the matching lesion-highlighted splat camera.",
    },
  };
  els.viewerTitle.textContent = copy[state.mode]?.title || copy.mri.title;
  els.viewerSubtitle.textContent = copy[state.mode]?.subtitle || copy.mri.subtitle;
  renderLesionCard(subject, manifest, roi);
  renderComparisonGallery(subject);
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
  els.lesionSubtype.textContent = expandHistopathology(subject.histopathology);
  els.lesionVolume.textContent = formatVolume(subject.lesionVolumeMm3);
  els.lesionVoxels.textContent = formatNumber(roi.voxel_count);
  els.lesionCenter.textContent = formatIjk(roi.center_ijk);
  const min = formatIjk(roi.bounds_min_ijk);
  const max = formatIjk(roi.bounds_max_ijk);
  els.lesionBounds.textContent = min === "unknown" || max === "unknown" ? "unknown" : `${min} to ${max}`;
  els.lesionSource.textContent = profile.source_truth_for_lesion === "provided_expert_lesion_mask"
    ? "Provided expert lesion mask"
    : "Expert mask";
}

function renderComparisonGallery(subject) {
  const candidates = state.subjects
    .filter((candidate) => candidate.id !== subject.id)
    .map((candidate) => ({
      candidate,
      score:
        (candidate.histopathology && candidate.histopathology === subject.histopathology ? 4 : 0) +
        (candidate.lobe && candidate.lobe === subject.lobe ? 2 : 0) +
        (candidate.hemisphere && candidate.hemisphere === subject.hemisphere ? 1 : 0),
    }))
    .sort((a, b) => b.score - a.score || a.candidate.id.localeCompare(b.candidate.id))
    .slice(0, 5)
    .map(({ candidate }) => candidate);

  els.galleryList.innerHTML = candidates.map((candidate) => `
    <button type="button" data-subject="${candidate.id}" class="gallery-item" title="Open ${candidate.id} for comparison.">
      <strong>${escapeHtml(candidate.id)}</strong>
      <span>${escapeHtml(expandHistopathology(candidate.histopathology))}</span>
      <span>${escapeHtml(formatLocation(candidate.hemisphere, candidate.lobe))}</span>
      <span>${formatVolume(candidate.lesionVolumeMm3)}</span>
    </button>
  `).join("");

  for (const button of els.galleryList.querySelectorAll("button[data-subject]")) {
    button.addEventListener("click", () => selectSubject(button.dataset.subject));
  }
}

function setMode(mode) {
  state.mode = mode;
  for (const button of document.querySelectorAll("[data-mode]")) {
    button.setAttribute("aria-pressed", button.dataset.mode === mode ? "true" : "false");
  }

  els.viewerGrid.classList.toggle("both", mode === "sync");
  els.niivueWrap.classList.toggle("hidden", mode === "splat");
  els.splatWrap.classList.toggle("hidden", mode === "mri");
  els.syncControls.classList.toggle("hidden", mode !== "sync");
}

function refreshFrames() {
  renderMetrics();
  renderSubjectList();
  const mri = niivueUrl({ root: state.root, subject: state.subject });
  const splat = splatUrl({ root: state.root, subject: state.subject, splat: "lesion-highlighted" });
  els.niivueLoading.classList.add("active");
  els.splatLoading.classList.add("active");
  els.niivueFrame.src = mri;
  els.splatFrame.src = splat;
  const activeUrl = state.mode === "mri" ? mri : splat;
  const activeLabel = state.mode === "mri" ? "Open MRI Viewer" : "Open 3D Lesion Splat";
  els.openCurrent.href = activeUrl;
  els.openCurrentButton.textContent = activeLabel;
  els.openCurrentButton.title = `Open the current ${state.mode === "mri" ? "MRI viewer" : "3D lesion splat"} in a separate browser tab.`;
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

  for (const button of document.querySelectorAll("[data-group-filter]")) {
    button.addEventListener("click", () => {
      state.groupFilter = button.dataset.groupFilter;
      for (const candidate of document.querySelectorAll("[data-group-filter]")) {
        candidate.setAttribute("aria-pressed", candidate === button ? "true" : "false");
      }
      renderSubjectList();
    });
  }

  for (const button of document.querySelectorAll("[data-splat-command]")) {
    button.addEventListener("click", () => sendSplatCommand(button.dataset.splatCommand));
  }

  document.getElementById("export-snapshot").addEventListener("click", exportTeachingSnapshot);

  document.getElementById("prev-subject").addEventListener("click", () => {
    selectSubject(nextSubjectId(state.subjects, state.subject, -1));
  });
  document.getElementById("next-subject").addEventListener("click", () => {
    selectSubject(nextSubjectId(state.subjects, state.subject, 1));
  });
}

function sendSplatCommand(command) {
  els.splatFrame.contentWindow?.postMessage({
    type: "isu-brain-visualizer:splat-command",
    command,
  }, window.location.origin);
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
  const payload = {
    title: "ISU Brain Visualizer teaching snapshot",
    subject: state.subject,
    mode: state.mode,
    location: formatLocation(subject.hemisphere, subject.lobe),
    histopathology: expandHistopathology(subject.histopathology),
    lesionVolume: formatVolume(subject.lesionVolumeMm3),
    splatPoints: formatPoints(subject.splatPoints),
    lesionSource: manifest.browser_splat_profile?.source_truth_for_lesion || "provided expert lesion mask",
    mriUrl: niivueUrl({ root: state.root, subject: state.subject }),
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
