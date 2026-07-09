import test from "node:test";
import assert from "node:assert/strict";

import {
  DEFAULT_ROOT,
  DEMO_SUBJECTS,
  buildDemoLinks,
  cleanRoot,
  explanationFor,
  expandHistopathology,
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
} from "../viewer/app-core.js";

test("URL builders target the finished FCD85 dataset and lesion-highlighted splat", () => {
  assert.equal(manifestUrl({ subject: "sub-00058" }), `${DEFAULT_ROOT}/sub-00058/manifest.json`);
  assert.equal(niivueUrl({ subject: "sub-00058" }), `/viewer/?root=%2Foutputs%2Fend_to_end_fcd85_hd&subject=sub-00058&v=20260705-t1w-3d-mri`);
  assert.equal(
    splatUrl({ subject: "sub-00058" }),
    `/viewer/splat.html?root=%2Foutputs%2Fend_to_end_fcd85_hd&subject=sub-00058&splat=lesion-highlighted`,
  );
  assert.equal(
    niivueUrl({ subject: "sub-00058", embed: true, layout: "3-2d" }),
    `/viewer/?root=%2Foutputs%2Fend_to_end_fcd85_hd&subject=sub-00058&embed=1&layout=3-2d&v=20260705-t1w-3d-mri`,
  );
  assert.equal(
    niivueUrl({ subject: "sub-00058", embed: true, layout: "4", base: "t1w" }),
    `/viewer/?root=%2Foutputs%2Fend_to_end_fcd85_hd&subject=sub-00058&embed=1&layout=4&base=t1w&v=20260705-t1w-3d-mri`,
  );
  assert.equal(
    splatUrl({ subject: "sub-00058", embed: true, view: "left" }),
    `/viewer/splat.html?root=%2Foutputs%2Fend_to_end_fcd85_hd&subject=sub-00058&splat=lesion-highlighted&embed=1&view=left`,
  );
  assert.equal(cleanRoot("/outputs/custom/"), "/outputs/custom");
  assert.equal(cleanRoot("   "), DEFAULT_ROOT);
  assert.equal(
    splatUrl({ root: "/outputs/custom/", subject: "not-valid", splat: "" }),
    "/viewer/splat.html?root=%2Foutputs%2Fcustom&subject=sub-00058&splat=lesion-highlighted",
  );
});

test("subject parsing accepts only OpenNeuro subject IDs", () => {
  assert.deepEqual(parseSubjectText("sub-00058\nbad\n sub-00043 \n"), ["sub-00058", "sub-00043"]);
  assert.equal(normalizeSubjectId("not-a-subject"), "sub-00058");
});

test("subject metadata normalization exposes demo metrics", () => {
  const subjects = normalizeSubjects(["sub-00058"], {
    "sub-00058": {
      group: "fcd",
      modalities: ["FLAIR", "T1w"],
      roi_available: true,
      viewer_meta: {
        participant: { lobe: "FL", hemisphere: "L", histopathology: "IIb" },
        roi: { volume_mm3: 19545 },
      },
      patient_splat_lesion_highlighted_meta: { n_points: 118000 },
      splat_reconstruction_audit_meta: { total_points: 919545, mri_to_3dgs_readiness_status: "review" },
    },
  });
  assert.equal(subjects[0].id, "sub-00058");
  assert.equal(subjects[0].lobe, "FL");
  assert.equal(subjects[0].splatPoints, 118000);
  assert.equal(formatLocation(subjects[0].hemisphere, subjects[0].lobe), "Left Frontal Lobe");
  assert.equal(expandHistopathology(subjects[0].histopathology), "Focal Cortical Dysplasia Type IIb");
});

test("subject normalization handles object entries, duplicate IDs, and missing metadata", () => {
  const subjects = normalizeSubjects([
    { subject: "sub-00058", group: "fcd", modalities: ["FLAIR"], roi_available: false },
    { subject: "sub-00058", group: "duplicate" },
    { subject: "bad-id", group: "bad" },
    { subject: "sub-00043", manifest: { viewer_meta: { participant: { group: "fcd", lobe: "TL" } } } },
  ]);
  assert.equal(subjects.length, 2);
  assert.equal(subjects[0].id, "sub-00058");
  assert.equal(subjects[0].roiAvailable, false);
  assert.equal(subjects[1].id, "sub-00043");
  assert.equal(subjects[1].lobe, "TL");
  assert.equal(subjects[1].group, "fcd");
});

test("filtering and navigation are deterministic", () => {
  const subjects = normalizeSubjects(["sub-00058", "sub-00043"], {
    "sub-00043": { viewer_meta: { participant: { hemisphere: "R", lobe: "FL" } } },
  });
  assert.equal(filterSubjects(subjects, "right").length, 1);
  assert.equal(filterSubjects(subjects, "frontal lobe").length, 1);
  assert.equal(filterSubjects(subjects, "").length, 2);
  assert.deepEqual(filterSubjects(null, ""), []);
  assert.equal(filterSubjects(subjects, "sub-00043")[0].id, "sub-00043");
  assert.equal(nextSubjectId(subjects, "sub-00058", 1), "sub-00043");
  assert.equal(nextSubjectId(subjects, "sub-00043", 1), "sub-00043");
  assert.equal(nextSubjectId([], "sub-00043", 1), "sub-00058");
  assert.equal(nextSubjectId(subjects, "missing", -1), "sub-00058");
});

test("formatters and five demo links are stable", () => {
  assert.equal(formatVolume(19545), "19,545 mm3 (19.55 cm3)");
  assert.equal(formatVolume(8186.97), "8,187 mm3 (8.19 cm3)");
  assert.equal(formatVolume("nope"), "unknown");
  assert.equal(formatPoints(919545), "919,545 points");
  assert.equal(formatPoints(null), "pending");
  const links = buildDemoLinks();
  assert.equal(links.length, 5);
  assert.deepEqual(links.map((link) => link.subject), DEMO_SUBJECTS);
  assert.ok(links.every((link) => link.splatUrl.includes("splat=lesion-highlighted")));
});

test("plain-language explanations are available and safe for unknown terms", () => {
  assert.match(explanationFor("Expert Lesion Mask"), /OpenNeuro dataset/);
  assert.match(explanationFor("3DGS"), /Gaussian Splatting/);
  assert.equal(explanationFor("not-a-term"), "");
});
