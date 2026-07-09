import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");

function readProjectFile(path) {
  return readFileSync(resolve(root, path), "utf8");
}

function selectBlock(html, id) {
  const match = html.match(new RegExp(`<select[^>]+id="${id}"[\\s\\S]*?<\\/select>`));
  assert.ok(match, `Expected select#${id} to exist`);
  return match[0];
}

function optionValues(selectHtml) {
  return [...selectHtml.matchAll(/<option[^>]+value="([^"]+)"/g)].map((match) => match[1]);
}

function cameraCommands(html) {
  return [...html.matchAll(/data-camera-command="([^"]+)"/g)].map((match) => match[1]);
}

test("main app exposes 2D MRI, 3D MRI, and 3D Gaussian splat tabs", () => {
  const html = readProjectFile("viewer/app.html");

  assert.match(html, /data-mode="mri2d"/);
  assert.match(html, /data-mode="mri3d"/);
  assert.match(html, /data-mode="splat"/);
  assert.doesNotMatch(html, /data-mode="sync"/);
  assert.match(html, /2D MRI Viewer/);
  assert.match(html, /3D MRI View/);
  assert.match(html, /3D Gaussian Splat View/);
  assert.match(html, /NiiVue MRI viewer/);
  assert.match(html, /Three\.js lesion-highlighted Gaussian splat viewer/);
});

test("main app keeps dataset, subtype grouping, and teaching export visible", () => {
  const html = readProjectFile("viewer/app.html");

  assert.match(html, /OpenNeuro ds004199/);
  assert.match(html, /id="group-filter"/);
  assert.match(html, /value="IIa"/);
  assert.match(html, /value="IIb"/);
  assert.match(html, /Subtype not listed/);
  assert.match(html, /id="export-snapshot"/);
  assert.match(html, /id="splat-points"[^>]*>pending/);
  assert.match(html, /id="acquisition-manufacturer"/);
  assert.match(html, /Plain-language definitions/);
});

test("main app displays the active MRI source modality under the view title", () => {
  const html = readProjectFile("viewer/app.html");
  const appJs = readProjectFile("viewer/app.js");
  const css = readProjectFile("viewer/app.css");

  assert.match(html, /id="mri-source-label"/);
  assert.match(html, /MRI source: loading/);
  assert.match(css, /\.source-pill/);
  assert.match(appJs, /function mriSourceLabelForMode/);
  assert.match(appJs, /MRI source: \$\{label\}/);
  assert.match(appJs, /resolveMriBase\(manifest, preferredMriBaseForMode\(state\.mode\)\)/);
});

test("left rail puts active case cards before dataset browsing controls", () => {
  const html = readProjectFile("viewer/app.html");
  const subjectIndex = html.indexOf('data-active-card="subject"');
  const acquisitionIndex = html.indexOf('data-active-card="acquisition"');
  const lesionIndex = html.indexOf('data-active-card="lesion"');
  const splatIndex = html.indexOf('data-active-card="splat"');
  const browserIndex = html.indexOf('class="case-card browser-card"');
  const similarIndex = html.indexOf("Similar MRI examples");

  assert.ok(subjectIndex > -1);
  assert.ok(acquisitionIndex > subjectIndex);
  assert.ok(lesionIndex > acquisitionIndex);
  assert.ok(splatIndex > lesionIndex);
  assert.ok(browserIndex > splatIndex);
  assert.ok(similarIndex > browserIndex);
});

test("left rail cards use fixed semantic colors", () => {
  const html = readProjectFile("viewer/app.html");
  const css = readProjectFile("viewer/app.css");

  assert.match(html, /active-card subject-card/);
  assert.match(html, /active-card acquisition-card/);
  assert.match(html, /active-card lesion-card/);
  assert.match(html, /active-card splat-card/);
  assert.match(css, /--subject-blue-rgb: 10, 132, 255/);
  assert.match(css, /--lesion-red-rgb: 238, 45, 63/);
  assert.match(css, /--splat-orange-rgb: 255, 149, 0/);
  assert.match(css, /\.subject-card,\n\.acquisition-card/);
  assert.match(css, /rgba\(var\(--subject-blue-rgb\)/);
  assert.match(css, /\.lesion-card/);
  assert.match(css, /rgba\(var\(--lesion-red-rgb\)/);
  assert.match(css, /\.splat-card/);
  assert.match(css, /rgba\(var\(--splat-orange-rgb\)/);
  assert.match(css, /\.swatch\.red \{ background: var\(--semantic-lesion-red\); \}/);
  assert.match(css, /\.swatch\.orange \{ background: var\(--semantic-splat-orange\); \}/);
  assert.match(css, /\.mode-row button\[data-mode="splat"\]\[aria-pressed="true"\]/);
  assert.doesNotMatch(css, /body\[data-viewer-mode=.*\] \.(active-card|lesion-card|splat-card)/);
});

test("standalone MRI and splat viewers keep red for lesion ROI and orange for 3DGS", () => {
  const niivue = readProjectFile("viewer/index.html");
  const splat = readProjectFile("viewer/splat.html");

  assert.match(niivue, /Red &mdash; Expert Lesion Mask/);
  assert.match(niivue, /Orange &mdash; 3D Lesion Splat/);
  assert.match(niivue, /background:#ff9500/);
  assert.doesNotMatch(niivue, /Orange \/ red/);
  assert.doesNotMatch(niivue, /linear-gradient\(to right, #fa0, #f60, #e00\)/);
  assert.match(splat, /accent-color: #ff9500/);
  assert.match(splat, /button\.act \{ border-color: #ff9500; background: #ff9500/);
});

test("MRI viewer separates 2D slice layout from the NiiVue 3D render panel", () => {
  const html = readProjectFile("viewer/index.html");
  const appJs = readProjectFile("viewer/app.js");

  const layoutSelect = selectBlock(html, "layout-select");
  assert.deepEqual(optionValues(layoutSelect), ["3-2d", "3-3d", "1", "4"]);
  assert.match(layoutSelect, /3-plane MRI only/);
  assert.match(layoutSelect, /3-plane \+ 3D MRI view/);
  assert.match(layoutSelect, /3D render only/);
  assert.match(html, /SHOW_RENDER\.NEVER/);
  assert.match(html, /multiplanarShowRender/);
  assert.match(html, /initialLayout/);
  assert.match(html, /get\("layout"\)/);
  assert.match(html, /get\("embed"\) === "1"/);
  assert.match(html, /html\.embed header/);
  assert.match(html, /applyMriViewLayout\(\);/);
  assert.match(html, /get\("base"\)/);
  assert.match(html, /base-select"\)\.value = initialBase/);
  assert.match(appJs, /function preferredMriBaseForMode/);
  assert.match(appJs, /return mode === "mri3d" \? "t1w" : "flair"/);
  assert.match(appJs, /base: mriBase/);
});

test("MRI viewer overlays only the provided expert lesion mask", () => {
  const html = readProjectFile("viewer/index.html");

  const overlayDefs = html.match(/const overlayDefs = \[[\s\S]*?\];/);
  assert.ok(overlayDefs, "Expected overlayDefs block");
  assert.match(overlayDefs[0], /key: "lesion_roi"/);
  assert.match(overlayDefs[0], /colormap: "red"/);
  assert.doesNotMatch(overlayDefs[0], /heatmap|candidate|residual|z_residual/i);
});

test("MRI base volume lets NiiVue auto-window MR intensities", () => {
  const html = readProjectFile("viewer/index.html");
  const loadVolumesBlock = html.match(/await nv\.loadVolumes\(\[[\s\S]*?\]\);/);

  assert.ok(loadVolumesBlock, "Expected source MRI loadVolumes block");
  assert.match(loadVolumesBlock[0], /colormap: "gray"/);
  assert.doesNotMatch(loadVolumesBlock[0], /cal_min: 0/);
  assert.doesNotMatch(loadVolumesBlock[0], /cal_max: 1/);
});

test("3D splat page exposes only the lesion-highlighted artifact selector", () => {
  const html = readProjectFile("viewer/splat.html");

  const artifactSelect = selectBlock(html, "sel-artifact");
  assert.deepEqual(optionValues(artifactSelect), ["lesion-highlighted"]);
  assert.match(artifactSelect, /3D lesion splat/);
  assert.doesNotMatch(artifactSelect, /residual|heatmap|candidate/i);
  assert.match(html, /This view does not use residuals, heatmaps, or candidate guesses/);
});

test("3D splat page has fixed MRI scale and explicit camera controls", () => {
  const html = readProjectFile("viewer/splat.html");

  assert.match(html, /Fixed MRI scale/);
  assert.match(html, /OrbitControls/);
  assert.match(html, /function wireCameraShortcuts\(\)/);
  assert.match(html, /function applyInitialCameraViewFromUrl\(\)/);
  assert.match(html, /get\("view"\)/);
  assert.match(html, /get\("embed"\) === "1"/);
  assert.match(html, /html\.embed header/);
  assert.match(html, /applyInitialCameraViewFromUrl\(\);/);
  assert.match(html, /key === "w"/);
  assert.match(html, /key === "a"/);
  assert.match(html, /key === "s"/);
  assert.match(html, /key === "d"/);
  assert.match(html, /key === "q"/);
  assert.match(html, /key === "e"/);
  assert.match(html, /event\.key === "ArrowUp"/);
  assert.match(html, /event\.key === "ArrowLeft"/);
});

test("3D MRI and 3D Gaussian splat expose the same camera controls", () => {
  const app = readProjectFile("viewer/app.html");
  const appJs = readProjectFile("viewer/app.js");
  const niivue = readProjectFile("viewer/index.html");
  const splat = readProjectFile("viewer/splat.html");
  const expected = [
    "front",
    "left",
    "top",
    "reset",
    "rotate-left",
    "rotate-right",
    "rotate-up",
    "rotate-down",
    "zoom-out",
    "zoom-in",
  ];

  assert.deepEqual(cameraCommands(app), expected);
  assert.deepEqual(cameraCommands(niivue), expected);
  assert.deepEqual(cameraCommands(splat), expected);
  assert.match(appJs, /isu-brain-visualizer:camera-command/);
  assert.match(niivue, /isu-brain-visualizer:camera-command/);
  assert.match(splat, /isu-brain-visualizer:camera-command/);
});

test("3D MRI and 3D Gaussian splat use the same wheel zoom direction", () => {
  const niivue = readProjectFile("viewer/index.html");
  const splat = readProjectFile("viewer/splat.html");

  assert.match(niivue, /addEventListener\("wheel"/);
  assert.match(niivue, /event\.deltaY < 0 \? "zoom-in" : "zoom-out"/);
  assert.match(niivue, /passive: false/);
  assert.match(splat, /addEventListener\("wheel"/);
  assert.match(splat, /event\.deltaY < 0 \? 0\.88 : 1\.12/);
  assert.match(splat, /passive: false/);
});
