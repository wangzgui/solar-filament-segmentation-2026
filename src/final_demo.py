"""Entry point for the final Kaggle demonstration, using the original H6a policy."""
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ASSETS = {
    "mul2": "zhuowamg/mul2out/versions/1",
    "s0_weights": "dariushafshar/solar-filament-unet-weights/versions/2",
    "s0_test_maps": "lojhhggg/sanxiandata/versions/2",
}
FULL_ASSETS = {
    "a1": "zhuowamg/a1data/versions/1",
    "v1": "zhuowamg/v1output/versions/1",
    "github": "ameypawar123456789/filament-seg-weights",
    "blended": "zhuowamg/v3output",
    "g": "wanhty/g-data/versions/1",
    "q": "lojhhggg/v4output/versions/2",
}
MODES = ("verified_replay", "fresh_s0", "full_inference")
REPLAY_PACKAGES = {
    "numpy": "numpy==2.4.6",
    "scipy": "scipy==1.17.1",
    "skimage": "scikit-image==0.26.0",
    "sklearn": "scikit-learn==1.9.0",
    "PIL": "Pillow==12.3.0",
    "pycocotools": "pycocotools==2.0.11",
    "kagglehub": "kagglehub==1.0.2",
}


def check_environment(mode):
    packages = REPLAY_PACKAGES.copy()
    if mode != "verified_replay":
        packages.update({"torch": "torch==2.10.0", "timm": "timm==1.0.29"})
    if mode == "full_inference":
        packages.update({"torchvision": "torchvision==0.25.0", "cv2": "opencv-python-headless==4.13.0.92",
                         "albumentations": "albumentations==2.0.8", "pandas": "pandas==3.0.3",
                         "joblib": "joblib==1.5.3", "psutil": "psutil==7.2.2", "numba": "numba==0.68.1",
                         "threadpoolctl": "threadpoolctl==3.6.0"})
    missing = [pin for name, pin in packages.items() if importlib.util.find_spec(name) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
    return {pin.split("==")[0]: version(pin.split("==")[0]) for pin in packages.values()}


def acquire(names):
    import kagglehub

    paths = {}
    for name in names:
        handle = (ASSETS | FULL_ASSETS)[name]
        try:
            path = Path(kagglehub.dataset_download(handle))
        except Exception as exc:
            raise RuntimeError(
                f"Could not automatically access {handle}. Check the dataset version/access "
                "from the Kaggle session. Attach all sources by running the notebook once "
                "in an interactive session before Save Version."
            ) from exc
        if not path.is_dir():
            raise FileNotFoundError(f"{handle} did not resolve to a dataset directory: {path}")
        paths[name] = path
        print(f"{name}: {handle} → {path}", flush=True)
    return paths


def verified_ensemble(weights, *, verify_files):
    from audit_assets import S0_FOLDS, S0_RECEIPT
    from mul2_assets import sha256
    from probability import discover_039

    receipt, _, models, _ = discover_039(weights, "")
    if sha256(receipt) != S0_RECEIPT or sha256(receipt.parent / "folds.csv") != S0_FOLDS:
        raise ValueError("Use the signed V2 S0 author receipt and fold table")
    if verify_files:
        for model in models:
            path = receipt.parent / model["file"]
            if not path.is_file() or sha256(path) != model["sha256"]:
                raise ValueError(f"S0 checkpoint missing or SHA-256 mismatch: {model['file']}")
    return receipt, models


def fresh_s0(test_images, weights, run_dir):
    from audit_assets import S0_FOLDS, S0_RECEIPT
    from mul2_assets import sha256
    from probability import infer_pending, metadata_for

    # infer_pending verifies all ten files before loading the models.
    receipt, models = verified_ensemble(weights, verify_files=False)
    photos = {p.stem: p for p in test_images.iterdir() if p.suffix.lower() in (".jpeg", ".jpg")}
    if len(photos) != 180:
        raise ValueError(f"Expected 180 test images; found {len(photos)}")
    meta = {stem: metadata_for(stem, sha256(path), models, None, S0_RECEIPT, S0_FOLDS)
            for stem, path in photos.items()}
    root = run_dir / "fresh_probability"
    cache = root / "Solar_S1_0/p039_cache" / f"{S0_RECEIPT[:16]}_{S0_FOLDS[:12]}"
    cache.mkdir(parents=True, exist_ok=True)
    infer_pending(cache, sorted(photos), photos, meta, receipt, models,
                  {"seed": 2026, "max_gpus": 2, "scratch_root": str(run_dir / "scratch")})
    return root


def full_inference(test_images, paths, run_dir, result, smoke_photos):
    from audit_assets import S0_FOLDS, S0_RECEIPT
    from mul2_assets import canonical, load_test, sha256
    from predict_champion import EXPECTED, publish_signed, select_photo
    from probability import infer_pending, load_s0, metadata_for
    from raw_mul2 import forward
    from sprint import write_submission

    receipt, models = verified_ensemble(paths["s0_weights"], verify_files=False)
    images = {p.stem: p for p in test_images.iterdir() if p.suffix.lower() in (".jpeg", ".jpg")}
    if len(images) != 180:
        raise ValueError(f"Expected 180 test images; found {len(images)}")
    if smoke_photos:
        images = {stem: images[stem] for stem in sorted(images)[:smoke_photos]}
    meta = {stem: metadata_for(stem, sha256(path), models, None, S0_RECEIPT, S0_FOLDS)
            for stem, path in images.items()}
    maps = run_dir / "fresh_s0"
    maps.mkdir()
    # Load the Afshar models first, before placing the independent Mul2 models on CUDA.
    runtime, _, native = infer_pending(maps, sorted(images), images, meta, receipt, models,
                                       {"seed": 2026, "max_gpus": 2, "scratch_root": str(run_dir / "scratch")})
    (run_dir / "s0_runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    import torch
    torch.cuda.empty_cache()
    spec = json.loads((Path(__file__).parent / "h6a/fusion_reference.json").read_text(encoding="utf-8"))["fit_models"]
    predictions = {}
    for stem, base, views in forward(images, paths, run_dir):
        donors = [[(canonical(row["rle"]), float(row["score"])) for row in group] for group in views]
        predictions[stem] = select_photo([canonical(mask) for mask in base], donors,
                                         load_s0(native[stem], meta[stem]), spec)
    if set(predictions) != set(images):
        raise ValueError("Fresh Mul2 did not return every requested test photo")
    if smoke_photos:
        write_submission(result, predictions)
        # Only compare after generating predictions, never pass reference masks to inference.
        reference = Path(__file__).resolve().parents[1] / "results/submission.csv"
        if sha256(reference) != EXPECTED:
            raise ValueError("The diagnostic reference CSV is not the signed champion")
        roster = sorted(p.stem for p in test_images.iterdir() if p.suffix.lower() in (".jpeg", ".jpg"))
        baseline = load_test(reference, roster)
        comparison = {stem: {"mask_sequence_equal": predictions[stem] == baseline[stem],
                             "generated_instances": len(predictions[stem]),
                             "archived_instances": len(baseline[stem])}
                      for stem in images}
        (run_dir / "smoke_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    else:
        publish_signed(result, predictions)


def run(mode, competition_root, output_root, *, smoke_photos=0):
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}. Choose from {MODES}.")
    if smoke_photos and (mode != "full_inference" or not 1 <= smoke_photos < 180):
        raise ValueError("Use smoke_photos=0, or 1 through 179 with full_inference")
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run this competition demo inside Kaggle. Official data stays on Kaggle.")
    test_dirs = [p for p in competition_root.rglob("test_images") if p.is_dir() and p.parent.name == "test"]
    if len(test_dirs) != 1:
        raise FileNotFoundError("Join the competition and attach its official dataset. "
                                "Expected exactly one test/test_images directory.")
    packages = check_environment(mode)
    if mode != "verified_replay":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(f"{mode} requires a Kaggle CUDA GPU. Use verified_replay for CPU.")
    handles = {"verified_replay": tuple(ASSETS), "fresh_s0": ("mul2", "s0_weights"),
               "full_inference": (*FULL_ASSETS, "s0_weights")}
    paths = acquire(handles[mode])

    sys.path.insert(0, str(Path(__file__).resolve().parent / "h6a"))
    from predict_champion import EXPECTED, run as predict
    from mul2_assets import sha256

    output_root.mkdir(parents=True, exist_ok=True)
    # Each mode writes its own file. An unsuccessful rerun cannot publish a CSV.
    run_dir = Path(tempfile.mkdtemp(prefix=mode + "-", dir=output_root))
    manifest = {"status": "running", "mode": mode, "packages": packages,
                "datasets": {name: (ASSETS | FULL_ASSETS)[name] for name in paths}, "training": False}
    log = run_dir / "run_manifest.json"
    log.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Run outputs: {run_dir}", flush=True)
    try:
        result = run_dir / ("smoke_predictions.csv" if smoke_photos else "submission.csv")
        if mode == "full_inference":
            full_inference(test_dirs[0], paths, run_dir, result, smoke_photos)
        else:
            if mode == "verified_replay":
                verified_ensemble(paths["s0_weights"], verify_files=True)
            probability_root = (paths["s0_test_maps"] if mode == "verified_replay"
                                else fresh_s0(test_dirs[0], paths["s0_weights"], run_dir))
            predict(competition_root, result, mul2_root=paths["mul2"], weights_root=paths["s0_weights"],
                    probability_root=probability_root, native_only=mode == "fresh_s0")
        manifest.update(status="smoke_complete" if smoke_photos else "complete", sha256=sha256(result),
                        expected_sha256=None if smoke_photos else EXPECTED, smoke_photos=smoke_photos)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        log.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result
