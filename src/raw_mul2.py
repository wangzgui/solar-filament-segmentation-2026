"""Rebuild the original Mul2 inference stages from images and signed weights."""
import ast
import base64
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


MUL2_CODE = "e182a7008aaf7e7481718c9e78c508b87dbd64228179d54cf9679bf92c4f9272"
REPO = Path(__file__).resolve().parents[1]


def unpack_runtime(destination):
    notebook = REPO / "notebooks/03_mul2_multiview.ipynb"
    source = "".join(json.loads(notebook.read_text(encoding="utf-8"))["cells"][0]["source"])
    keys = {"MODULE_PAYLOAD", "FUSION", "FUSION_CODE"}
    values = {node.targets[0].id: ast.literal_eval(node.value)
              for node in ast.parse(source).body
              if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in keys}
    raw = gzip.decompress(base64.b64decode(values["MODULE_PAYLOAD"]))
    if hashlib.sha256(raw).hexdigest() != MUL2_CODE:
        raise ValueError("Original Mul2 source payload has changed")
    for name, text in json.loads(raw).items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    sys.path.insert(0, str(destination))
    return values["FUSION"], values["FUSION_CODE"]


def only(root, filename, predicate=lambda _: True):
    paths = [p for p in root.rglob(filename) if p.is_file() and predicate(p)]
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one {filename} in {root}, found {len(paths)}")
    return paths[0]


def load_recipe(paths, fusion):
    """Read signed recipe manifests, independent of G3/Q1 photo predictions."""
    from solar_a_line.run import FROZEN_SHA
    from solar_v15.common import digest, read_json

    manifests = {}
    for role in ("g", "q"):
        root = paths[role]
        summary = read_json(only(root, "run_manifest.json", lambda p: p.parent.name == f"solar_gq_{role}"))
        if (summary["status"] != "complete" or summary["experiment_signature"] != fusion["expected_experiments"][role]
                or summary["frozen_signature"] != FROZEN_SHA
                or any(summary["artifact_checksums"].get(name) != expected
                       for name, expected in fusion["expected_reference_hashes"][role].items())):
            raise ValueError(f"Incorrect signed {role} release")
        manifests[role] = summary
    manifest = read_json(only(paths["g"], "frozen_cache_manifest.json",
                              lambda p: p.parent.name == "solar_gq_g"))
    if manifest["signature"] != FROZEN_SHA or digest(manifest["contract"]) != FROZEN_SHA:
        raise ValueError("Incorrect combined model and preprocessing contract")
    manifests["g_frozen"] = manifest
    return SimpleNamespace(manifests=manifests)


class SemanticRoots:
    """Give the original weight loader two automatically resolved dataset roots."""
    def __init__(self, paths):
        self.paths = paths

    def rglob(self, name):
        for path in self.paths:
            yield from path.rglob(name)


def forward(photos, paths, working):
    import cv2
    import torch

    fusion, fusion_code = unpack_runtime(working / "original_mul2_source")
    from solar_a1_multiview.run import A1_MANIFEST_SHA, mirror_masks
    from solar_a_line.a101.run import A1_MODEL, A1_QUALITY, load_quality
    from solar_a_line.a101.validation import load_semantic, semantic_photo
    from solar_a_line.detector import architecture, model_contract
    from solar_a_line.run import checkpoint_infer, match_source
    from solar_gq.config import Config as GraphConfig
    from solar_gq.prepare import Prepared
    from solar_mul2.masks import predict_variants, rotated_masks
    from solar_v1.config import seed_all
    from solar_v1.models import load_model
    from solar_v1.preprocess import SolarPreprocessor, read_gray
    from solar_v15.common import digest, file_hash, read_json
    from solar_v15.config import Config as OriginalConfig

    assets = load_recipe(paths, fusion)
    a1 = only(paths["a1"], "run_manifest.json", lambda p: file_hash(p) == A1_MANIFEST_SHA).parent
    signed = read_json(a1 / "run_manifest.json")["artifact_sha256"]
    for name in ("config.json", "training_receipt.json", "baseline_audit.json", "best_detector.pth",
                 "baseline_replay/receipt.json", "baseline_replay/C_HGB.joblib"):
        if file_hash(a1 / name) != signed[name]:
            raise ValueError(f"Signed A1 asset changed: {name}")
    if signed["best_detector.pth"] != A1_MODEL or signed["baseline_replay/C_HGB.joblib"] != A1_QUALITY:
        raise ValueError("The A1 detector or upstream quality model is not the submitted inference asset")
    original_v1 = only(paths["v1"], "normalization.json",
                       lambda p: (p.parent / "checkpoints/refiner_best.pt").is_file()).parent
    v1, norm, _, hashes = match_source(assets, paths["v1"], str(original_v1), working)
    if hashes != read_json(a1 / "baseline_audit.json")["v1_checkpoint_sha256"]:
        raise ValueError("V1 weights differ from the signed A1 run")
    quality = load_quality(a1, assets, fusion, fusion_code)
    cv2.setNumThreads(0)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    seed_all(fusion["seed"])
    cfg = replace(assets.base, OUTPUT_ROOT=str(working), DETECTOR_IMAGE_SIZE=1536, AMP=True)
    detector = architecture(cfg, "R50", norm)
    checkpoint = torch.load(a1 / "best_detector.pth", map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != 3 or checkpoint["signature"] != read_json(a1 / "training_receipt.json")["signature"]:
        raise ValueError("A1 checkpoint metadata changed")
    detector.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    if digest(model_contract(detector)) != digest(read_json(a1 / "config.json")["detector"]):
        raise ValueError("A1 architecture differs from the signed run")
    detector.to(device).eval().requires_grad_(False)
    refiner, _ = load_model(assets.base, norm, "refiner", device, v1 / "checkpoints/refiner_best.pt")
    semantic, missing = load_semantic(SemanticRoots([paths["github"], paths["blended"]]), assets, device)
    if semantic is None:
        raise FileNotFoundError("Missing or invalid signed Mul2 semantic weights: " + "; ".join(missing))
    graph = GraphConfig(**assets.manifests["g"]["config"]).validate()
    identity = digest([MUL2_CODE, A1_MODEL, A1_QUALITY, hashes])
    prepared = Prepared(graph, working / "fresh_g3", identity)
    preprocessor = SolarPreprocessor(working / "preprocessed", assets.base.PREPROCESS)
    original, features = OriginalConfig(), SimpleNamespace(**fusion)

    for i, (stem, path) in enumerate(sorted(photos.items()), 1):
        data = preprocessor.process(read_gray(path))
        photo = semantic_photo(path, data, semantic, working, identity)
        seeds = checkpoint_infer(detector, refiner, photo, path, cfg, assets.base, assets,
                                 original, prepared, features, quality, device)[-1]["final"]
        views = [mirror_masks(detector, refiner, data, cfg, assets.base, device, axis) for axis in (1, 0)]
        views += [rotated_masks(detector, refiner, data, cfg, assets.base, device, k) for k in (1, 3)]
        masks, _ = predict_variants(seeds, views, photo["maps"]["pf"], photo["disk"])
        yield stem, masks["vote_both"], views
        print(f"Fresh Mul2 {i}/{len(photos)}: {stem}", flush=True)
