const DEFAULT_ROOT = "/outputs/end_to_end_fcd85_hd";
const DEFAULT_SUBJECT = "sub-00058";
const DEFAULT_SPLAT = "lesion-highlighted";

const TERM_EXPLANATIONS = {
  MRI: "The original brain scan. Think of it as a stack of thin picture slices through the head.",
  FLAIR: "Fluid-Attenuated Inversion Recovery. This MRI setting suppresses normal fluid so some abnormal tissue can stand out more clearly.",
  "Expert Lesion Mask": "The red label that came with the OpenNeuro dataset. A human expert marked this as the known Focal Cortical Dysplasia lesion.",
  "Lesion Splat": "The orange 3D Gaussian Splat version of that same expert lesion mask, with the MRI anatomy shown lightly for context.",
  "3D MRI View": "A direct 3D volume rendering of the NIfTI MRI voxels. It comes from the MRI itself, not from Gaussian splatting.",
  "3DGS": "3D Gaussian Splatting. It draws a 3D object using many small soft blobs instead of a solid mesh. Here it is an alternate visualization, not the source MRI.",
  NiiVue: "The medical image viewer used here to show real NIfTI MRI slices, overlays, and direct 3D volume rendering.",
  Black: "Empty space, masked background, or low MRI signal. Black does not mean lesion by itself.",
  "Focal Cortical Dysplasia": "A developmental difference in the brain cortex that can cause drug-resistant epilepsy.",
  "Balloon cells": "Large abnormal cells sometimes seen in Focal Cortical Dysplasia Type IIb tissue. They help pathologists tell Type IIb apart from Type IIa.",
  "Type IIa": "A Focal Cortical Dysplasia tissue subtype. It means abnormal cortical cells are present, without balloon cells.",
  "Type IIb": "A Focal Cortical Dysplasia tissue subtype. It means abnormal cortical cells and balloon cells are present.",
};

const GROUP_LABELS = {
  fcd: "Focal Cortical Dysplasia",
  hc: "Healthy Control",
};

const HEMISPHERE_LABELS = {
  L: "Left",
  R: "Right",
};

const LOBE_LABELS = {
  FL: "Frontal Lobe",
  PL: "Parietal Lobe",
  TL: "Temporal Lobe",
  OL: "Occipital Lobe",
  IL: "Insular Lobe",
};

const HISTOPATHOLOGY_LABELS = {
  IIa: "Focal Cortical Dysplasia Type IIa",
  IIb: "Focal Cortical Dysplasia Type IIb",
};

const DEMO_SUBJECTS = ["sub-00058", "sub-00043", "sub-00133", "sub-00120", "sub-00060"];

function cleanRoot(root) {
  const value = String(root || DEFAULT_ROOT).trim() || DEFAULT_ROOT;
  return value.endsWith("/") ? value.slice(0, -1) : value;
}

function normalizeSubjectId(subject) {
  const value = String(subject || DEFAULT_SUBJECT).trim();
  return /^sub-\d{5}$/.test(value) ? value : DEFAULT_SUBJECT;
}

function manifestUrl({ root = DEFAULT_ROOT, subject = DEFAULT_SUBJECT } = {}) {
  return `${cleanRoot(root)}/${normalizeSubjectId(subject)}/manifest.json`;
}

function niivueUrl({ root = DEFAULT_ROOT, subject = DEFAULT_SUBJECT, embed = false, layout = "", base = "" } = {}) {
  const params = new URLSearchParams();
  params.set("root", cleanRoot(root));
  params.set("subject", normalizeSubjectId(subject));
  if (embed) params.set("embed", "1");
  if (["1", "2", "3", "3-2d", "3-3d", "4"].includes(String(layout))) params.set("layout", String(layout));
  if (["flair", "t1w", "t2w"].includes(String(base))) params.set("base", String(base));
  params.set("v", "20260705-t1w-3d-mri");
  return `/viewer/?${params.toString()}`;
}

function splatUrl({
  root = DEFAULT_ROOT,
  subject = DEFAULT_SUBJECT,
  splat = DEFAULT_SPLAT,
  embed = false,
  view = "",
} = {}) {
  const params = new URLSearchParams();
  params.set("root", cleanRoot(root));
  params.set("subject", normalizeSubjectId(subject));
  params.set("splat", splat || DEFAULT_SPLAT);
  if (embed) params.set("embed", "1");
  if (["front", "left", "right", "top", "zoom"].includes(String(view))) params.set("view", String(view));
  return `/viewer/splat.html?${params.toString()}`;
}

function parseSubjectText(text) {
  return String(text || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => /^sub-\d{5}$/.test(line));
}

function normalizeSubjects(subjects, manifests = {}) {
  const seen = new Set();
  return (subjects || [])
    .map((entry) => {
      const subject = typeof entry === "string" ? entry : entry?.subject;
      const id = normalizeSubjectId(subject);
      if (!id || seen.has(id)) return null;
      seen.add(id);
      const manifest = manifests[id] || (typeof entry === "object" ? entry.manifest : null) || {};
      const viewerMeta = manifest.viewer_meta || {};
      const participant = viewerMeta.participant || {};
      const roi = viewerMeta.roi || {};
      const audit = manifest.splat_reconstruction_audit_meta || {};
      const browserSplatMeta = manifest.patient_splat_lesion_highlighted_meta || {};
      const browserSplatProfile = manifest.browser_splat_profile || {};
      return {
        id,
        group: manifest.group || entry?.group || participant.group || "fcd",
        modalities: manifest.modalities || entry?.modalities || [],
        roiAvailable: Boolean(manifest.roi_available ?? entry?.roi_available ?? true),
        lobe: participant.lobe || "",
        hemisphere: participant.hemisphere || "",
        histopathology: participant.histopathology || "",
        lesionVolumeMm3: roi.volume_mm3 ?? null,
        splatPoints: browserSplatMeta.n_points ?? browserSplatProfile.total_points ?? audit.total_points ?? null,
        readiness: audit.mri_to_3dgs_readiness_status || "",
      };
    })
    .filter(Boolean);
}

function expandGroup(group) {
  const value = String(group || "").trim();
  return GROUP_LABELS[value.toLowerCase()] || value || "Unknown Group";
}

function expandHemisphere(hemisphere) {
  const value = String(hemisphere || "").trim();
  return HEMISPHERE_LABELS[value] || value || "Unknown Side";
}

function expandLobe(lobe) {
  const value = String(lobe || "").trim();
  if (!value) return "Unknown Lobe";
  return value
    .split(",")
    .map((part) => LOBE_LABELS[part.trim()] || part.trim())
    .filter(Boolean)
    .join(" and ");
}

function expandHistopathology(histopathology) {
  const value = String(histopathology || "").trim();
  return HISTOPATHOLOGY_LABELS[value] || (value ? `Focal Cortical Dysplasia Type ${value}` : "Histopathology not listed");
}

function formatLocation(hemisphere, lobe) {
  const side = expandHemisphere(hemisphere);
  const region = expandLobe(lobe);
  if (side === "Unknown Side" && region === "Unknown Lobe") return "Location not listed";
  if (side === "Unknown Side") return region;
  if (region === "Unknown Lobe") return side;
  return `${side} ${region}`;
}

function formatVolume(mm3) {
  if (mm3 == null || mm3 === "") return "unknown";
  const value = Number(mm3);
  if (!Number.isFinite(value)) return "unknown";
  return `${value.toLocaleString(undefined, { maximumFractionDigits: 0 })} mm3 (${(value / 1000).toFixed(2)} cm3)`;
}

function formatPoints(points) {
  if (points == null || points === "") return "pending";
  const value = Number(points);
  if (!Number.isFinite(value)) return "pending";
  return `${value.toLocaleString()} points`;
}

function subjectSearchText(subject) {
  return [
    subject.id,
    subject.group,
    expandGroup(subject.group),
    subject.modalities?.join(" "),
    subject.lobe,
    expandLobe(subject.lobe),
    subject.hemisphere,
    expandHemisphere(subject.hemisphere),
    subject.histopathology,
    expandHistopathology(subject.histopathology),
    subject.readiness,
  ].filter(Boolean).join(" ").toLowerCase();
}

function filterSubjects(subjects, query) {
  const q = String(query || "").trim().toLowerCase();
  if (!q) return subjects || [];
  return (subjects || []).filter((subject) => subjectSearchText(subject).includes(q));
}

function nextSubjectId(subjects, currentId, delta) {
  if (!subjects?.length) return DEFAULT_SUBJECT;
  const index = Math.max(0, subjects.findIndex((subject) => subject.id === currentId));
  const next = Math.min(subjects.length - 1, Math.max(0, index + delta));
  return subjects[next].id;
}

function buildDemoLinks(root = DEFAULT_ROOT) {
  return DEMO_SUBJECTS.map((subject) => ({
    subject,
    splatUrl: splatUrl({ root, subject }),
    niivueUrl: niivueUrl({ root, subject }),
  }));
}

function explanationFor(term) {
  return TERM_EXPLANATIONS[term] || "";
}

export {
  DEFAULT_ROOT,
  DEFAULT_SUBJECT,
  DEFAULT_SPLAT,
  DEMO_SUBJECTS,
  TERM_EXPLANATIONS,
  buildDemoLinks,
  cleanRoot,
  explanationFor,
  expandGroup,
  expandHemisphere,
  expandHistopathology,
  expandLobe,
  filterSubjects,
  formatLocation,
  formatPoints,
  formatVolume,
  manifestUrl,
  nextSubjectId,
  niivueUrl,
  normalizeSubjectId,
  normalizeSubjects,
  parseSubjectText,
  splatUrl,
};
