from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest


def test_ds004199_download_main_wires_arguments(monkeypatch, tmp_path):
    from fcd_prototype import ds004199_download as mod

    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(mod, "download", fake_download)

    mod.main([
        "--destination", str(tmp_path),
        "--subjects", "sub-00058", "sub-00043",
        "--modalities", "T1w", "FLAIR",
        "--include-roi",
        "--dry-run",
    ])

    assert calls == [{
        "destination": tmp_path,
        "subjects": ["sub-00058", "sub-00043"],
        "modalities": ["T1w", "FLAIR"],
        "include_roi": True,
        "dry_run": True,
        "all_participants": False,
        "group": "all",
        "limit": None,
    }]


def test_ds004199_download_all_participants_dry_run_missing_tsv(monkeypatch, tmp_path, capsys):
    from fcd_prototype import ds004199_download as mod

    calls = []
    monkeypatch.setattr(
        mod,
        "download_dataset_files",
        lambda destination, dry_run: calls.append((destination, dry_run)),
    )

    mod.download(
        destination=tmp_path,
        subjects=None,
        modalities=["T1w"],
        include_roi=False,
        dry_run=True,
        all_participants=True,
        group="fcd",
        limit=2,
    )

    assert calls == [(tmp_path, True)]
    assert "Subject expansion requires a local participants.tsv" in capsys.readouterr().out


def test_ds004199_download_all_participants_fetches_metadata_then_subjects(monkeypatch, tmp_path):
    from fcd_prototype import ds004199_download as mod

    calls = []
    monkeypatch.setattr(
        mod,
        "download_dataset_files",
        lambda destination, dry_run: calls.append(("dataset", destination, dry_run)),
    )
    monkeypatch.setattr(
        mod,
        "parse_participants_tsv",
        lambda tsv_path, group: ["sub-00058", "sub-00043", "sub-00133"],
    )
    monkeypatch.setattr(
        mod,
        "download_subject",
        lambda subject, destination, modalities, include_roi, dry_run: calls.append(
            ("subject", subject, destination, modalities, include_roi, dry_run)
        ),
    )

    mod.download(
        destination=tmp_path,
        subjects=None,
        modalities=["FLAIR"],
        include_roi=True,
        dry_run=False,
        all_participants=True,
        group="fcd",
        limit=2,
    )

    assert calls[:2] == [("dataset", tmp_path, False), ("dataset", tmp_path, False)]
    assert calls[2:] == [
        ("subject", "sub-00058", tmp_path, ["FLAIR"], True, False),
        ("subject", "sub-00043", tmp_path, ["FLAIR"], True, False),
    ]


def test_ds004199_download_requires_subject_source(tmp_path):
    from fcd_prototype import ds004199_download as mod

    try:
        mod.download(
            destination=tmp_path,
            subjects=None,
            modalities=["T1w"],
            include_roi=False,
            dry_run=True,
            all_participants=False,
        )
    except ValueError as exc:
        assert "Either subjects or all_participants=True" in str(exc)
    else:
        raise AssertionError("download should require explicit subjects or all participants")


def test_ds004199_download_direct_subjects_route(monkeypatch, tmp_path):
    from fcd_prototype import ds004199_download as mod

    calls = []
    monkeypatch.setattr(
        mod,
        "download_dataset_files",
        lambda destination, dry_run: calls.append(("dataset", destination, dry_run)),
    )
    monkeypatch.setattr(
        mod,
        "download_subject",
        lambda subject, destination, modalities, include_roi, dry_run: calls.append(
            ("subject", subject, destination, modalities, include_roi, dry_run)
        ),
    )

    mod.download(
        destination=tmp_path,
        subjects=["sub-00058"],
        modalities=["T1w"],
        include_roi=True,
        dry_run=True,
        all_participants=False,
    )

    assert calls == [
        ("dataset", tmp_path, True),
        ("subject", "sub-00058", tmp_path, ["T1w"], True, True),
    ]


def test_mri_data_audit_main_dataset_branch(monkeypatch, tmp_path, capsys):
    from fcd_prototype import mri_data_audit as mod

    calls = []

    def fake_audit_dataset(dataset_dir, *, subjects, output_json, output_csv):
        calls.append((dataset_dir, subjects, output_json, output_csv))
        return {"summary": {"status": "ok", "subject_count": 2}}

    monkeypatch.setattr(mod, "audit_dataset", fake_audit_dataset)
    mod.main([
        "--dataset-dir", str(tmp_path / "ds"),
        "--subjects", "sub-00058", "sub-00043",
        "--output-json", str(tmp_path / "audit.json"),
        "--output-csv", str(tmp_path / "audit.csv"),
    ])

    assert calls == [(
        tmp_path / "ds",
        ["sub-00058", "sub-00043"],
        tmp_path / "audit.json",
        tmp_path / "audit.csv",
    )]
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_mri_data_audit_main_single_subject_branch(monkeypatch, tmp_path, capsys):
    from fcd_prototype import mri_data_audit as mod

    def fake_build_subject_data_quality(**kwargs):
        assert kwargs["flair_path"] == tmp_path / "flair.nii.gz"
        return {"status": "warning", "warning_count": 1}

    monkeypatch.setattr(mod, "build_subject_data_quality", fake_build_subject_data_quality)
    out_json = tmp_path / "out" / "audit.json"

    mod.main(["--flair", str(tmp_path / "flair.nii.gz"), "--output-json", str(out_json)])

    assert json.loads(out_json.read_text())["status"] == "warning"
    assert json.loads(capsys.readouterr().out)["warning_count"] == 1


def test_gaussian_field_main_wires_training_options(monkeypatch, tmp_path, capsys):
    from fcd_prototype import gaussian_field as mod

    calls = []

    def fake_check_torch(device):
        calls.append(("check", device))

    def fake_train_gaussian_field(**kwargs):
        calls.append(("train", kwargs))
        return {"backend": "fake", "num_gaussians": kwargs["num_gaussians"], "final_loss": 0.125}

    monkeypatch.setattr(mod, "_check_torch", fake_check_torch)
    monkeypatch.setattr(mod, "train_gaussian_field", fake_train_gaussian_field)

    mod.main([
        "--volume", str(tmp_path / "vol.nii.gz"),
        "--roi", str(tmp_path / "roi.nii.gz"),
        "--output-dir", str(tmp_path / "field"),
        "--num-gaussians", "12",
        "--steps", "3",
        "--image-size", "16",
        "--views", "axial,coronal",
        "--device", "cpu",
        "--seed", "7",
        "--learning-rate", "0.02",
        "--preview-every", "2",
        "--no-jitter",
        "--volume-loss-weight", "0.4",
        "--volume-sample-count", "32",
    ])

    assert calls[0] == ("check", "cpu")
    _, kwargs = calls[1]
    assert kwargs["volume_path"] == tmp_path / "vol.nii.gz"
    assert kwargs["roi_path"] == tmp_path / "roi.nii.gz"
    assert kwargs["num_gaussians"] == 12
    assert kwargs["jitter_voxels"] is False
    assert kwargs["volume_loss_weight"] == 0.4
    assert "Training complete. Backend=fake" in capsys.readouterr().out


def test_cohort_evaluation_main_wires_gate_thresholds(monkeypatch, tmp_path, capsys):
    from fcd_prototype import cohort_evaluation as mod

    calls = []

    def fake_evaluate_output_root(**kwargs):
        calls.append(kwargs)
        return {"subjects": [{"subject": "sub-00058"}], "summary": {"status": "ok"}}

    monkeypatch.setattr(mod, "evaluate_output_root", fake_evaluate_output_root)

    mod.main([
        str(tmp_path / "outputs"),
        "--output-json", str(tmp_path / "out.json"),
        "--output-csv", str(tmp_path / "out.csv"),
        "--score-metric", "auto",
        "--expected-fcd", "85",
        "--expected-hc", "85",
        "--auroc-min", "0.9",
        "--control-prior-similarity-min", "0.7",
        "--splat-foreground-coverage-min", "0.5",
        "--splat-mip-mean-mae-max", "0.1",
        "--optimized-field-foreground-mae-max", "0.2",
        "--optimized-field-all-mae-max", "0.3",
        "--paired-gaussian-min-points", "100",
        "--control-gaussian-min-points", "200",
        "--control-gaussian-min-controls", "5",
        "--optimized-gaussian-min-points", "300",
        "--localization-mean-manual-coverage-min", "0.6",
        "--hc-candidate-foreground-fraction-max", "0.02",
    ])

    kwargs = calls[0]
    assert kwargs["output_root"] == tmp_path / "outputs"
    assert kwargs["output_json"] == tmp_path / "out.json"
    assert kwargs["output_csv"] == tmp_path / "out.csv"
    assert kwargs["gate_thresholds"]["expected_fcd"] == 85
    assert kwargs["gate_thresholds"]["auroc_min"] == 0.9
    assert kwargs["gate_thresholds"]["control_gaussian_min_controls"] == 5
    out = json.loads(capsys.readouterr().out)
    assert out == {"summary": {"status": "ok"}}


def test_control_gaussian_prior_main_wires_compute(monkeypatch, tmp_path, capsys):
    from fcd_prototype import control_gaussian_prior as mod

    calls = []

    def fake_build_control_gaussian_prior_residual(**kwargs):
        calls.append(kwargs)
        return {"n_gaussians": 44, "n_controls": 2, "residual_z_p99": 1.25}

    monkeypatch.setattr(
        mod,
        "build_control_gaussian_prior_residual",
        fake_build_control_gaussian_prior_residual,
    )

    mod.main([
        "--patient-volume", str(tmp_path / "patient.nii.gz"),
        "--control-volume", str(tmp_path / "c1.nii.gz"),
        "--control-volume", str(tmp_path / "c2.nii.gz"),
        "--output-dir", str(tmp_path / "prior"),
        "--max-points", "44",
        "--min-controls", "2",
        "--heatmap-percentile", "99.5",
        "--smooth-sigma", "1.25",
        "--heatmap-method", "scatter",
        "--heatmap-gaussian-exact-max-points", "12",
        "--std-floor-fraction", "0.2",
        "--std-floor-absolute", "0.01",
        "--subject", "sub-00058",
        "--group", "fcd",
    ])

    kwargs = calls[0]
    assert kwargs["patient_volume"] == tmp_path / "patient.nii.gz"
    assert kwargs["control_volumes"] == [tmp_path / "c1.nii.gz", tmp_path / "c2.nii.gz"]
    assert kwargs["max_points"] == 44
    assert kwargs["min_controls"] == 2
    assert kwargs["std_floor_fraction"] == 0.2
    assert kwargs["subject"] == "sub-00058"
    assert "Control Gaussian prior complete: N=44 controls=2" in capsys.readouterr().out


def test_supervised_lesion_proposer_main_train_predict_evaluate_and_sweep(monkeypatch, tmp_path, capsys):
    from fcd_prototype import supervised_lesion_proposal as mod

    calls = []

    monkeypatch.setattr(
        mod,
        "train_lesion_proposer",
        lambda *args, **kwargs: calls.append(("train", args, kwargs)) or {"loss": 1.0},
    )
    monkeypatch.setattr(
        mod,
        "train_patch_lesion_proposer",
        lambda *args, **kwargs: calls.append(("train-patch", args, kwargs)) or {"n_train_samples": 9},
    )
    monkeypatch.setattr(
        mod,
        "predict_lesion_roi",
        lambda *args, **kwargs: calls.append(("predict", args, kwargs)) or {
            "predicted_voxels": 5,
            "probability_mean": 0.25,
        },
    )
    monkeypatch.setattr(
        mod,
        "evaluate_lesion_proposer",
        lambda *args, **kwargs: calls.append(("evaluate", args, kwargs)) or {
            "n_subjects": 2,
            "mean_dice": 0.1,
            "mean_coverage": 0.2,
            "mean_precision": 0.3,
        },
    )
    monkeypatch.setattr(
        mod,
        "evaluate_threshold_sweep",
        lambda *args, **kwargs: calls.append(("sweep", args, kwargs)) or {
            "n_fcd": 1,
            "n_hc": 1,
            "selected_operating_point": {"threshold": 0.4},
        },
    )

    commands = [
        [
            "prog", "train", str(tmp_path / "ds"), str(tmp_path / "out"),
            "--subjects", "sub-00058", "--val-subjects", "sub-00043",
            "--spatial-size", "8", "9", "10", "--epochs", "2",
            "--batch-size", "3", "--lr", "0.02", "--device", "cpu",
            "--max-train-subjects", "4",
        ],
        [
            "prog", "train-patch", str(tmp_path / "ds"), str(tmp_path / "patch"),
            "--subjects", "sub-00058", "--val-subjects", "sub-00043",
            "--patch-size", "20", "--positives-per-subject", "2",
            "--negatives-per-subject", "3", "--hard-negatives-per-subject", "1",
            "--hard-negative-pool-factor", "5",
            "--hard-negative-min-separation-voxels", "6",
            "--epochs", "2", "--batch-size", "4", "--lr", "0.03",
            "--device", "cpu", "--max-train-subjects", "5",
        ],
        [
            "prog", "predict", str(tmp_path / "ckpt.pt"), str(tmp_path / "vol.nii.gz"),
            str(tmp_path / "pred.nii.gz"), "--threshold", "0.4",
            "--min-component-voxels", "7", "--selection-mode", "top-fraction",
            "--foreground-fraction", "0.03", "--device", "cpu",
        ],
        [
            "prog", "evaluate", str(tmp_path / "ckpt.pt"), str(tmp_path / "ds"),
            str(tmp_path / "eval"), "--subjects", "sub-00058", "--threshold", "0.6",
            "--selection-mode", "top-fraction", "--foreground-fraction", "0.02",
            "--device", "cpu",
        ],
        [
            "prog", "sweep", str(tmp_path / "ckpt.pt"), str(tmp_path / "ds"),
            str(tmp_path / "sweep"), "--subjects", "sub-00058",
            "--thresholds", "0.2", "0.4", "--selection-modes", "threshold",
            "--min-component-voxels", "8", "--min-coverage", "0.55",
            "--max-hc-foreground-fraction", "0.04", "--device", "cpu",
        ],
    ]

    for argv in commands:
        monkeypatch.setattr(sys, "argv", argv)
        mod.main()

    assert [call[0] for call in calls] == [
        "train", "train-patch", "predict", "evaluate", "sweep",
    ]
    assert calls[0][2]["spatial_size"] == (8, 9, 10)
    assert calls[1][2]["hard_negative_min_separation_voxels"] == 6
    assert calls[2][2]["selection_mode"] == "top-fraction"
    assert calls[3][2]["foreground_fraction"] == 0.02
    assert calls[4][2]["thresholds"] == [0.2, 0.4]
    output = capsys.readouterr().out
    assert "Training complete" in output
    assert "Patch training complete" in output
    assert "Prediction saved" in output
    assert "Evaluation complete" in output
    assert "Threshold sweep complete" in output


def test_anomaly_pipeline_main_wires_placeholder_pipeline(monkeypatch, tmp_path, capsys):
    from fcd_prototype import anomaly_pipeline as mod

    calls = []

    def fake_run_anomaly_pipeline(**kwargs):
        calls.append(kwargs)
        return {
            "patient": tmp_path / "patient.nii.gz",
            "baseline": tmp_path / "baseline.nii.gz",
            "anomaly": tmp_path / "anomaly.nii.gz",
        }

    monkeypatch.setattr(mod, "run_anomaly_pipeline", fake_run_anomaly_pipeline)
    monkeypatch.setattr(sys, "argv", [
        "prog",
        str(tmp_path / "input.nii.gz"),
        "--output-dir", str(tmp_path / "out"),
        "--smooth-sigma", "3.5",
        "--diff-sigma", "1.5",
        "--threshold-percentile", "97",
    ])

    mod.main()

    assert calls == [{
        "input_path": str(tmp_path / "input.nii.gz"),
        "output_dir": str(tmp_path / "out"),
        "smooth_sigma": 3.5,
        "diff_sigma": 1.5,
        "threshold_percentile": 97.0,
    }]
    out = capsys.readouterr().out
    assert "patient:" in out
    assert "baseline:" in out
    assert "anomaly:" in out


def test_nifti_inspect_pretty_print(capsys):
    from fcd_prototype.nifti_inspect import print_inspection

    print_inspection({
        "shape": (2, 3, 4),
        "voxel_spacing_mm": (1.0, 1.5, 2.0),
        "dtype": "float32",
        "min": 0.0,
        "max": 1.0,
        "mean": 0.5,
        "nan_count": 0,
        "inf_count": 0,
        "affine": np.eye(4),
    })

    out = capsys.readouterr().out
    assert "Shape" in out
    assert "Voxel spacing" in out
    assert "Affine matrix" in out


def test_gaussian_seed_main_wires_build_and_save(monkeypatch, tmp_path):
    from fcd_prototype import gaussian_seed as mod

    calls = []

    def fake_build_seed(**kwargs):
        calls.append(("build", kwargs))
        return {
            "positions_world": np.zeros((2, 3), dtype=np.float32),
            "is_roi": np.array([True, False]),
        }

    def fake_save_seed(seed, output_path):
        calls.append(("save", seed, output_path))

    monkeypatch.setattr(mod, "build_seed", fake_build_seed)
    monkeypatch.setattr(mod, "save_seed", fake_save_seed)

    mod.main([
        str(tmp_path / "volume.nii.gz"),
        "--roi", str(tmp_path / "roi.nii.gz"),
        "--output", str(tmp_path / "seed.npz"),
        "--max-points", "77",
        "--percentile-threshold", "6.5",
        "--rng-seed", "9",
    ])

    assert calls[0][0] == "build"
    assert calls[0][1]["volume_path"] == tmp_path / "volume.nii.gz"
    assert calls[0][1]["roi_path"] == tmp_path / "roi.nii.gz"
    assert calls[0][1]["max_points"] == 77
    assert calls[0][1]["percentile_threshold"] == 6.5
    assert calls[0][1]["rng_seed"] == 9
    assert calls[1][0] == "save"
    assert calls[1][2] == tmp_path / "seed.npz"


def test_gaussian_splat_main_wires_renderer(monkeypatch, tmp_path):
    from fcd_prototype import gaussian_splat as mod

    calls = []

    def fake_render_splat(**kwargs):
        calls.append(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr(mod, "render_splat", fake_render_splat)

    mod.main([
        str(tmp_path / "seed.npz"),
        str(tmp_path / "ref.nii.gz"),
        "--output", str(tmp_path / "splat.nii.gz"),
        "--max-points", "11",
        "--sigma-vox", "2.25",
    ])

    assert calls == [{
        "seed_path": tmp_path / "seed.npz",
        "reference_path": tmp_path / "ref.nii.gz",
        "output_path": tmp_path / "splat.nii.gz",
        "max_points": 11,
        "sigma_vox": 2.25,
    }]


def test_candidate_roi_evaluation_main_single_and_sweep(monkeypatch, tmp_path, capsys):
    from fcd_prototype import candidate_roi_evaluation as mod

    calls = []

    def fake_evaluate_candidate_recall(**kwargs):
        calls.append(("single", kwargs))
        return {"mode": "single"}

    def fake_sweep_candidate_recall(**kwargs):
        calls.append(("sweep", kwargs))
        return {"mode": "sweep"}

    monkeypatch.setattr(mod, "evaluate_candidate_recall", fake_evaluate_candidate_recall)
    monkeypatch.setattr(mod, "sweep_candidate_recall", fake_sweep_candidate_recall)

    mod.main([
        str(tmp_path / "ds"),
        str(tmp_path / "out"),
        "--subjects", "sub-00058,sub-00043",
        "--max-candidates", "4",
        "--radius-mm", "7.5",
        "--percentile", "98.5",
        "--min-separation-mm", "12",
        "--proposal-mode", "brightness-gradient",
        "--force",
    ])
    mod.main([
        str(tmp_path / "ds"),
        str(tmp_path / "out"),
        "--max-candidates", "4,5",
        "--radius-mm", "7.5",
        "--percentile", "98.5",
        "--proposal-mode", "brightness-gradient",
    ])

    assert calls[0][0] == "single"
    assert calls[0][1]["subjects"] == ["sub-00058", "sub-00043"]
    assert calls[0][1]["max_candidates"] == 4
    assert calls[0][1]["radius_mm"] == 7.5
    assert calls[0][1]["force"] is True
    assert calls[1][0] == "sweep"
    assert calls[1][1]["max_candidates_values"] == (4, 5)
    assert '"mode": "single"' in capsys.readouterr().out


def test_candidate_roi_evaluation_main_rejects_invalid_modes(tmp_path):
    from fcd_prototype import candidate_roi_evaluation as mod

    try:
        mod.main([
            str(tmp_path / "ds"),
            str(tmp_path / "out"),
            "--proposal-mode", "not-a-mode",
        ])
    except ValueError as exc:
        assert "proposal-mode" in str(exc)
    else:
        raise AssertionError("invalid proposal mode should raise")


def test_candidate_roi_evaluation_main_rejects_empty_mode(tmp_path):
    from fcd_prototype import candidate_roi_evaluation as mod

    try:
        mod.main([
            str(tmp_path / "ds"),
            str(tmp_path / "out"),
            "--proposal-mode", ",,,",
        ])
    except ValueError as exc:
        assert "--proposal-mode" in str(exc)
    else:
        raise AssertionError("empty proposal mode should raise")


def test_real_preview_main_uses_discovered_subject_when_paths_omitted(monkeypatch, tmp_path):
    from fcd_prototype import real_preview as mod
    from fcd_prototype import bids_discovery

    calls = []

    monkeypatch.setattr(
        bids_discovery,
        "discover_subject",
        lambda dataset_dir, subject: SimpleNamespace(
            t1w=tmp_path / "disc_t1w.nii.gz",
            flair=tmp_path / "disc_flair.nii.gz",
            flair_roi=tmp_path / "disc_roi.nii.gz",
        ),
    )
    monkeypatch.setattr(
        mod,
        "save_real_preview",
        lambda **kwargs: calls.append(kwargs) or {"axial": tmp_path / "axial.png"},
    )

    mod.main([
        "--dataset-dir", str(tmp_path / "ds"),
        "--subject", "sub-00058",
        "--output-dir", str(tmp_path / "previews"),
    ])

    assert calls == [{
        "t1w_path": tmp_path / "disc_t1w.nii.gz",
        "flair_path": tmp_path / "disc_flair.nii.gz",
        "roi_path": tmp_path / "disc_roi.nii.gz",
        "output_dir": tmp_path / "previews" / "sub-00058",
        "subject": "sub-00058",
    }]


def test_monai_ingest_main_saves_plain_tensors(monkeypatch, tmp_path, capsys):
    torch = pytest.importorskip("torch")
    from fcd_prototype import monai_ingest as mod

    class FakeLoader:
        dataset = [object()]

        def __iter__(self):
            yield {
                "subject": ["sub-00058"],
                "T1w": torch.ones((1, 1, 2, 2, 2)),
                "FLAIR": torch.zeros((1, 1, 2, 2, 2)),
            }

    monkeypatch.setattr(mod, "_check_monai", lambda: None)
    monkeypatch.setattr(mod, "build_dataloader", lambda **kwargs: FakeLoader())

    mod.main([
        "--dataset-dir", str(tmp_path / "ds"),
        "--modalities", "T1w", "FLAIR",
        "--output-dir", str(tmp_path / "tensors"),
        "--batch-size", "1",
        "--num-workers", "0",
    ])

    out_path = tmp_path / "tensors" / "sub-00058_tensors.pt"
    assert out_path.exists()
    saved = torch.load(out_path, weights_only=True)
    assert saved["_meta"]["subject"] == "sub-00058"
    assert saved["_meta"]["modalities"] == ["T1w", "FLAIR"]
    assert "Saved:" in capsys.readouterr().out


def test_monai_ingest_main_warns_when_save_fails(monkeypatch, tmp_path, capsys):
    torch = pytest.importorskip("torch")
    from fcd_prototype import monai_ingest as mod

    class FakeLoader:
        dataset = [object()]

        def __iter__(self):
            yield {
                "subject": ["sub-00058"],
                "T1w": torch.ones((1, 1, 2, 2, 2)),
            }

    monkeypatch.setattr(mod, "_check_monai", lambda: None)
    monkeypatch.setattr(mod, "build_dataloader", lambda **kwargs: FakeLoader())
    monkeypatch.setattr(torch, "save", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("disk full")))

    mod.main([
        "--dataset-dir", str(tmp_path / "ds"),
        "--modalities", "T1w",
        "--output-dir", str(tmp_path / "tensors"),
    ])

    assert "WARNING: Failed to save sub-00058" in capsys.readouterr().err


def test_mri_data_audit_viewer_meta_roi_error_branch(monkeypatch, tmp_path):
    from fcd_prototype import mri_data_audit as mod

    roi = tmp_path / "broken_roi.nii.gz"
    roi.write_text("not nifti", encoding="utf-8")
    monkeypatch.setattr(mod.nib, "load", lambda path: (_ for _ in ()).throw(RuntimeError("bad nifti")))

    meta = mod.build_viewer_metadata(
        subject_record=SimpleNamespace(subject_id="sub-00058", group="fcd", metadata={}),
        base_nifti_path=None,
        base_modality="FLAIR",
        roi_nifti_path=roi,
        participant_metadata={"group": "fcd"},
        sidecar_fields={},
    )

    assert meta["roi"]["error"] == "could not compute ROI stats"
