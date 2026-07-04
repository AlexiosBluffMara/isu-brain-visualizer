"""Static regression checks for the Three.js Gaussian splat viewer."""
from __future__ import annotations

from pathlib import Path


def test_viewer_uses_instanced_covariance_ellipses():
    viewer = Path("viewer/splat.html").read_text(encoding="utf-8")
    assert "new THREE.InstancedBufferGeometry()" in viewer
    assert "new THREE.InstancedBufferAttribute(centers, 3)" in viewer
    assert "attribute vec2  aCorner" in viewer
    assert "attribute vec3  aCenter" in viewer
    assert "lambdaMajor" in viewer
    assert "lambdaMinor" in viewer
    assert "clipCenter + vec4(ndcOffset * clipCenter.w" in viewer


def test_viewer_no_longer_uses_point_sprite_coordinates():
    viewer = Path("viewer/splat.html").read_text(encoding="utf-8")
    assert "new THREE.Points" not in viewer
    assert "gl_PointCoord" not in viewer
    assert "gl_PointSize" not in viewer


def test_visible_splat_artifact_picker_is_limited_to_lesion_view():
    viewer = Path("viewer/splat.html").read_text(encoding="utf-8")
    artifact_select = viewer.split('<select id="sel-artifact">', 1)[1].split("</select>", 1)[0]
    assert 'option value="lesion-highlighted"' in artifact_select
    assert "paired-" not in artifact_select
    assert "control-prior" not in artifact_select
    assert "optimized" not in artifact_select


def test_splat_loader_forces_lesion_artifact_even_when_old_urls_are_used():
    viewer = Path("viewer/splat.html").read_text(encoding="utf-8")
    assert 'document.getElementById("sel-artifact").value = "lesion-highlighted";' in viewer
    assert 'let artifactPref = document.getElementById("sel-artifact").value;' in viewer
    assert 'artifactPref === "lesion-highlighted"' in viewer


def test_splat_viewer_keeps_only_lesion_view_as_product_handoff():
    viewer = Path("viewer/splat.html").read_text(encoding="utf-8")
    assert 'id="sel-artifact"' in viewer
    assert 'Open MRI Viewer' in viewer
    assert "3D Lesion Splat" in viewer
    assert "patient-versus-control" not in viewer


def test_viewer_ignores_artifact_dropdown_values_from_url():
    viewer = Path("viewer/splat.html").read_text(encoding="utf-8")
    assert "sel-artifact" in viewer
    assert "allowedSplatParams" not in viewer


def test_viewer_depth_sorts_transparent_gaussian_instances():
    viewer = Path("viewer/splat.html").read_text(encoding="utf-8")
    assert "sortSplatsByDepth" in viewer
    assert "sourceSplatAttributes" in viewer
    assert "depth[b] - depth[a]" in viewer
    assert "far-to-near for alpha blending" in viewer
    assert "camera.matrixWorldInverse.copy(camera.matrixWorld).invert()" in viewer
    assert "geo.getAttribute(attrName).needsUpdate = true" in viewer
    assert "depth-sorted alpha" in viewer


def test_viewer_does_not_present_comparison_scene_labels():
    viewer = Path("viewer/splat.html").read_text(encoding="utf-8")
    assert "makePairedComparisonGroup" in viewer
    assert "currentSplatMeshes" in viewer
    assert 'id="compare-labels"' in viewer
    assert 'document.getElementById("compare-labels").style.display = "none";' in viewer


def test_viewer_has_keyboard_and_preset_camera_controls():
    viewer = Path("viewer/splat.html").read_text(encoding="utf-8")
    assert "wireCameraShortcuts" in viewer
    assert 'key === "w"' in viewer
    assert 'key === "q"' in viewer
    assert "zoomCamera" in viewer
    assert "setCameraPreset" in viewer


def test_niivue_viewer_loads_only_expert_lesion_mask_overlay():
    viewer = Path("viewer/index.html").read_text(encoding="utf-8")
    assert 'id="chk-roi" checked' in viewer
    assert 'key: "lesion_roi"' in viewer
    assert 'residual_candidate_mask' not in viewer
    assert 'lesion_heatmap' not in viewer


def test_niivue_viewer_does_not_expose_residual_overlay():
    viewer = Path("viewer/index.html").read_text(encoding="utf-8")
    assert 'control_prior_z_residual_heatmap' not in viewer
    assert '"chk-residual"' not in viewer
    assert "overlays: ${overlayLabel}" in viewer


def test_niivue_to_splat_handoff_opens_control_prior_comparison():
    viewer = Path("viewer/index.html").read_text(encoding="utf-8")
    assert "function chooseSplatViewArtifact(arts)" in viewer
    assert 'return "lesion-highlighted";' in viewer
    assert 'params.set("splat", chooseSplatViewArtifact(arts));' in viewer


def test_niivue_to_splat_handoff_accepts_uploaded_subject_ids():
    viewer = Path("viewer/index.html").read_text(encoding="utf-8")
    assert '([^\\/]+)\\/manifest\\.json' in viewer
    assert 'const subj = m ? m[2] : (manifest?.subject || "");' in viewer
    assert 'params.set("subject", subj);' in viewer


def test_niivue_viewer_explains_overlay_terms_in_plain_language():
    viewer = Path("viewer/index.html").read_text(encoding="utf-8")
    assert "OpenNeuro manual label" in viewer
    assert "source lesion data" in viewer
    assert "expert lesion mask translated into browser-rendered 3D Gaussian splats" in viewer
    assert "Dark areas are <em>not</em> a lesion marker by themselves" in viewer


def test_niivue_viewer_uses_safe_overlay_defaults():
    viewer = Path("viewer/index.html").read_text(encoding="utf-8")
    assert 'id="chk-roi" checked' in viewer
    assert 'id="chk-heatmap"' not in viewer
    assert 'id="chk-residual"' not in viewer
    assert 'id="chk-candidates"' not in viewer
    assert 'id="chk-splat"' not in viewer


def test_niivue_viewer_has_dataset_subject_browser():
    viewer = Path("viewer/index.html").read_text(encoding="utf-8")
    assert 'id="subj-select"' in viewer
    assert 'id="subj-prev"' in viewer
    assert 'id="subj-next"' in viewer
    assert '"/outputs/end_to_end_fcd85_hd"' in viewer
    assert '"/outputs/fcd85_subjects.txt"' in viewer
    assert "entry.roi_available" in viewer
    assert "entry.group" in viewer
