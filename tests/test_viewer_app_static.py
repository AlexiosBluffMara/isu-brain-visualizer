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
    core = _core()
    for term in ["Expert Lesion Mask", "Lesion Splat", "NiiVue", "3DGS", "Balloon cells"]:
        assert term in core
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
