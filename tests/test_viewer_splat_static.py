"""Static contract checks for the Three.js Gaussian splat viewer."""

from pathlib import Path


def _viewer_html() -> str:
    return Path("viewer/splat.html").read_text(encoding="utf-8")


def test_splat_viewer_uses_instanced_gaussian_billboards():
    html = _viewer_html()

    assert "new THREE.InstancedBufferGeometry()" in html
    assert "aScaleVec" in html
    assert "aQuat" in html
    assert "projected 3D covariance" in html
    assert "uScaleBoost" in html
    assert "scaleBoostFromMeta" in html
    assert "new THREE.Points" not in html


def test_splat_viewer_exposes_product_render_modes():
    html = _viewer_html()

    assert 'id="sel-render"' in html
    for mode in ("volume", "audit", "radiograph"):
        assert f'value="{mode}"' in html
    assert 'option value="residual"' not in html
    assert "uRenderMode" in html
    assert "uCoverageBoost" in html
    assert "setMaterialBlending" in html


def test_splat_viewer_reports_sampling_and_audit_quality():
    html = _viewer_html()

    assert "artifactQualityLabel" in html
    assert "density_voxels_per_splat" in html
    assert "mri_to_3dgs_claim_allowed" in html
    assert "3D Gaussian Splatting audit" in html
    assert "visualization only" in html


def test_visible_artifact_picker_only_exposes_lesion_splat():
    html = _viewer_html()

    artifact_select = html.split('<select id="sel-artifact">', 1)[1].split("</select>", 1)[0]
    assert 'value="lesion-highlighted"' in artifact_select
    assert "control-prior" not in artifact_select
    assert "paired-" not in artifact_select
    assert "optimized" not in artifact_select


def test_splat_viewer_exposes_visible_camera_controls():
    html = _viewer_html()

    assert "3D controls" in html
    assert "Mouse drag rotates freely" in html
    assert "wireCameraShortcuts" in html
    assert "rotateCamera" in html
    assert "zoomCamera" in html
    assert 'id="btn-view-front"' in html
    assert 'id="btn-view-left"' in html
    assert 'id="btn-view-top"' in html


def test_splat_viewer_fetches_regenerated_ply_without_cache():
    html = _viewer_html()

    assert 'fetch(plyUrl, { cache: "no-store" })' in html
