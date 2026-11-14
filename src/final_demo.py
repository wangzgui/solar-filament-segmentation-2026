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
    if mode == "fresh_s0":
        packages.update({"torch": "torch==2.10.0", "timm": "timm==1.0.29"})
    missing = [pin for name, pin in packages.items() if importlib.util.find_spec(name) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
    return {pin.split("==")[0]: version(pin.split("==")[0]) for pin in packages.values()}


def acquire(names):
    import kagglehub

    paths = {}
    for name in names:
        handle = ASSETS[name]
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


def run(mode, competition_root, output_root):
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}. Choose from {MODES}.")
    if mode == "full_inference":
        raise NotImplementedError(
            "The historical Mul2 test path uses archived A1 seed masks, an archived "
            "contour reference, and G3/Q1 intermediates. The complete replacement of "
            "these inputs from raw photos is not implemented or verified. "
            "See docs/reproduction.md. No cache fallback was run."
        )
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run this competition demo inside Kaggle. Official data stays on Kaggle.")
    test_dirs = [p for p in competition_root.rglob("test_images") if p.is_dir() and p.parent.name == "test"]
    if len(test_dirs) != 1:
        raise FileNotFoundError("Join the competition and attach its official dataset. "
                                "Expected exactly one test/test_images directory.")
    packages = check_environment(mode)
    if mode == "fresh_s0":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("fresh_s0 requires a Kaggle CUDA GPU. Use verified_replay for CPU.")
    paths = acquire(ASSETS if mode == "verified_replay" else ("mul2", "s0_weights"))

    sys.path.insert(0, str(Path(__file__).resolve().parent / "h6a"))
    from predict_champion import EXPECTED, run as predict
    from mul2_assets import sha256

    output_root.mkdir(parents=True, exist_ok=True)
    # Each mode writes its own file. An unsuccessful rerun cannot publish a CSV.
    run_dir = Path(tempfile.mkdtemp(prefix=mode + "-", dir=output_root))
    manifest = {"status": "running", "mode": mode, "packages": packages,
                "datasets": {name: ASSETS[name] for name in paths}, "training": False}
    log = run_dir / "run_manifest.json"
    log.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Run outputs: {run_dir}", flush=True)
    try:
        if mode == "verified_replay":
            verified_ensemble(paths["s0_weights"], verify_files=True)
        probability_root = (paths["s0_test_maps"] if mode == "verified_replay"
                            else fresh_s0(test_dirs[0], paths["s0_weights"], run_dir))
        result = run_dir / "submission.csv"
        predict(competition_root, result, mul2_root=paths["mul2"], weights_root=paths["s0_weights"],
                probability_root=probability_root, native_only=mode == "fresh_s0")
        manifest.update(status="complete", expected_sha256=EXPECTED, sha256=sha256(result))
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        log.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result
