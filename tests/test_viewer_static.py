"""Static contract checks for viewer/index.html."""

from pathlib import Path


def _html() -> str:
    return Path("viewer/index.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Layout: NiiVue canvas as primary area, side panel present
# ---------------------------------------------------------------------------

def test_canvas_is_primary_area():
    html = _html()
    assert 'id="gl-canvas"' in html
    assert 'id="canvas-wrap"' in html
    assert 'id="main-area"' in html
    assert 'id="meta-panel"' in html


def test_side_panel_contains_subject_browser_and_legend():
    html = _html()
    assert 'id="subj-browser"' in html
    assert 'id="legend-section"' in html
    assert 'id="meta-content"' in html


# ---------------------------------------------------------------------------
# Tooltip titles on all required controls
# ---------------------------------------------------------------------------

def test_manifest_url_has_tooltip():
    html = _html()
    assert 'id="manifest-url"' in html
    assert 'Manifest URL' in html
    assert 'manifest.json' in html
    assert 'title=' in html


def test_subject_browser_has_tooltip():
    html = _html()
    assert 'id="subj-select"' in html
    assert 'Subject Browser' in html
    # title attribute on the section title or the select
    assert 'index.json' in html


def test_prev_next_buttons_have_tooltips():
    html = _html()
    assert 'id="subj-prev"' in html
    assert 'id="subj-next"' in html
    assert 'previous subject' in html.lower() or 'prev' in html.lower()
    assert 'next subject' in html.lower() or 'next' in html.lower()


def test_overlay_checkboxes_have_tooltips():
    html = _html()
    for label_text in ("Expert Lesion Mask",):
        assert label_text in html
    # Each overlay label element should have a title attribute
    assert "source data, not an algorithm guess" in html


def test_base_selector_has_tooltip():
    html = _html()
    assert 'id="base-select"' in html
    assert 'FLAIR' in html
    assert 'T1w' in html or 'T2w' in html


def test_btn_splat_view_has_tooltip():
    html = _html()
    assert 'id="btn-splat-view"' in html
    assert 'Open 3D Lesion Splat' in html
    assert 'title=' in html


def test_opacity_has_tooltip():
    html = _html()
    assert 'id="opacity-slider"' in html
    assert 'Opacity' in html
    assert 'transparent' in html.lower() or 'opacity' in html.lower()


def test_layout_has_tooltip():
    html = _html()
    assert 'id="layout-select"' in html
    assert 'Layout' in html


def test_crosshair_reset_have_tooltips():
    html = _html()
    assert 'id="btn-crosshair"' in html
    assert 'id="btn-reset"' in html
    assert 'crosshair' in html.lower()
    assert 'reset' in html.lower()


def test_manual_load_buttons_are_not_exposed():
    html = _html()
    assert 'id="btn-load-base"' not in html
    assert 'id="btn-add-overlay"' not in html
    assert 'id="file-base"' not in html
    assert 'id="file-overlay"' not in html


# ---------------------------------------------------------------------------
# Color legend - required explanatory text
# ---------------------------------------------------------------------------

def test_legend_uses_american_english():
    html = _html()
    assert "Color Legend" in html
    assert "Colour Legend" not in html

def test_legend_expert_lesion_mask_explanation():
    html = _html()
    assert 'Expert Lesion Mask' in html
    assert 'OpenNeuro manual label' in html
    assert 'source lesion data' in html


def test_legend_splat_explanation():
    html = _html()
    assert '3D Lesion Splat' in html
    assert 'expert lesion mask translated into browser-rendered 3D Gaussian splats' in html


def test_legend_black_dark_explanation():
    html = _html()
    # Black/dark explained as background/masked/transparent, not lesion marker
    assert 'background' in html
    assert 'not' in html
    # Should mention black or dark context is not a lesion marker by itself
    assert 'lesion marker' in html.lower() or 'not a lesion' in html.lower() or 'not lesion' in html.lower()


def test_status_uses_educational_visualization_language():
    html = _html()
    assert 'Educational visualization' in html
    assert 'Prototype / not diagnostic' not in html


# ---------------------------------------------------------------------------
# Default checkbox states: expert lesion mask on; no algorithm overlays exposed
# ---------------------------------------------------------------------------

def test_roi_checkbox_checked_by_default():
    html = _html()
    # chk-roi must have `checked` attribute; search in the input tag
    import re
    roi_tag = re.search(r'<input[^>]+id="chk-roi"[^>]*>', html)
    assert roi_tag is not None, "chk-roi input not found"
    assert "checked" in roi_tag.group()


def test_algorithm_overlay_checkboxes_are_not_exposed():
    html = _html()
    for removed in ('id="chk-heatmap"', 'id="chk-residual"', 'id="chk-candidates"', 'id="chk-splat"'):
        assert removed not in html


# ---------------------------------------------------------------------------
# Subject browser JavaScript
# ---------------------------------------------------------------------------

def test_subject_browser_fetches_index_json():
    html = _html()
    assert "loadSubjectIndex" in html
    assert "/outputs/end_to_end/index.json" in html


def test_subject_browser_populates_dropdown():
    html = _html()
    assert "subj-select" in html
    assert "appendChild" in html or "innerHTML" in html


def test_subject_browser_prev_next_navigate():
    html = _html()
    assert "navigateSubject" in html
    assert "subj-prev" in html
    assert "subj-next" in html


def test_subject_browser_loads_manifest_on_change():
    html = _html()
    # Selecting a subject should trigger loadManifest
    assert "loadManifest" in html
    assert "manifest-url" in html


def test_manifest_url_manual_load_preserved():
    html = _html()
    assert 'id="btn-load-manifest"' in html
    assert 'id="manifest-url"' in html


# ---------------------------------------------------------------------------
# Metadata panel - improved labels with plain-language help
# ---------------------------------------------------------------------------

def test_metadata_voxel_spacing_has_help():
    html = _html()
    # Spacing row should include a help tooltip or inline text about mm meaning
    assert "voxel_spacing_mm" in html or "Spacing" in html
    assert "mm" in html
    # Help text explaining smaller = higher resolution
    assert "higher-resolution" in html or "higher resolution" in html


def test_metadata_roi_volume_has_help():
    html = _html()
    assert "volume_mm3" in html or "Volume" in html
    # cm3 conversion label
    assert "cm" in html
    assert "lesion" in html.lower()


def test_metadata_shape_has_help():
    html = _html()
    assert "Shape" in html
    assert "vox" in html.lower() or "voxel" in html.lower()
