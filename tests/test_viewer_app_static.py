"""Static contract checks for the productized viewer shell."""

from pathlib import Path


def _html() -> str:
    return Path("viewer/app.html").read_text(encoding="utf-8")


def _js() -> str:
    return Path("viewer/app.js").read_text(encoding="utf-8")


def _core() -> str:
    return Path("viewer/app-core.js").read_text(encoding="utf-8")


def test_app_embeds_existing_niivue_and_splat_viewers():
    html = _html()
    js = _js()
    assert "ISU Brain Visualizer" in html
    assert 'id="niivue-frame"' in html
    assert 'id="splat-frame"' in html
    assert "niivueUrl" in js
    assert "splatUrl" in js
    assert "lesion-highlighted" in js


def test_app_has_three_tab_product_model_without_synced_mode():
    html = _html()
    js = _js()
    assert 'data-mode="mri2d"' in html
    assert 'data-mode="mri3d"' in html
    assert 'data-mode="splat"' in html
    assert 'data-mode="sync"' not in html
    assert "Synced MRI + 3D" not in html
    assert "sync-controls" not in html
    assert "2D MRI Viewer" in html
    assert "3D MRI View" in html
    assert "3D Gaussian Splat View" in html
    assert "Open 2D MRI in Full Viewer" in js
    assert "Open 3D MRI in Full Viewer" in js
    assert "Open 3D Gaussian Splat in Full Viewer" in js


def test_app_has_left_side_metadata_and_dropdown_filter():
    html = _html()
    js = _js()
    assert 'id="dataset-select"' in html
    assert 'id="group-filter"' in html
    assert 'id="acquisition-manufacturer"' in html
    assert 'id="lesion-card"' in html
    assert 'id="splat-points"' in html
    assert "groupFilterSelect" in js
    assert "mri3d" in js


def test_app_shows_active_mri_source_modality_label():
    html = _html()
    js = _js()
    assert 'id="mri-source-label"' in html
    assert "MRI source: loading" in html
    assert "mriSourceLabelForMode" in js
    assert "MRI source: ${label}" in js
    assert "resolveMriBase" in js
    assert "preferredMriBaseForMode" in js


def test_app_defaults_to_finished_fcd85_hd_dataset():
    core = _core()
    assert 'DEFAULT_ROOT = "/outputs/end_to_end_fcd85_hd"' in core
    assert 'DEFAULT_SUBJECT = "sub-00058"' in core
    assert 'DEFAULT_SPLAT = "lesion-highlighted"' in core


def test_app_loads_all_fcd85_subjects_text_list():
    js = _js()
    assert "/outputs/fcd85_subjects.txt" in js
    assert "parseSubjectText" in js
    assert "normalizeSubjects" in js


def test_app_has_plain_language_explanations_for_required_terms():
    html = _html()
    core = _core()
    for term in ["Expert Lesion Mask", "Lesion Splat", "NiiVue", "3DGS", "Balloon cells"]:
        assert term in core
    assert "Plain-language definitions" in html
    assert "Color legend" in html
    assert "OpenNeuro dataset" in core
    assert "MRI anatomy shown lightly for context" in core
    assert "Black does not mean lesion" in core


def test_app_has_safe_subject_switching_controls():
    html = _html()
    js = _js()
    assert 'id="prev-subject"' in html
    assert 'id="next-subject"' in html
    assert "nextSubjectId" in js
    assert "refreshFrames" in js
    assert "history.replaceState" in js


def test_app_has_quiz_and_reveal_lesion_modes():
    html = _html()
    js = _js()
    core = _core()
    assert 'data-lesion-visibility="hidden"' in html
    assert 'data-lesion-visibility="shown"' in html
    assert 'id="quiz-state-label"' in html
    assert "Quiz mode: expert lesion hidden" in js
    assert "Answer mode: expert lesion shown" in js
    assert "Hidden for quiz" in js
    assert "lesionVisibility" in js
    assert "normalizeLesionVisibility" in core
    assert 'params.set("lesion", normalizeLesionVisibility(lesion));' in core


def test_app_has_collapsible_side_panels_and_full_dataset_terms():
    html = _html()
    js = _js()
    core = _core()
    assert 'id="toggle-subject-panel"' in html
    assert 'id="toggle-guide-panel"' in html
    assert "togglePanel" in js
    assert "Left" in core
    assert "Frontal Lobe" in core
    assert "Focal Cortical Dysplasia Type IIb" in core
    assert "Histopathology not listed" in core
