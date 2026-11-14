"""Dariush 8TTA inference and validated probability cache inputs."""

from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from threading import Lock
from zipfile import BadZipFile

import numpy as np
from PIL import Image

from reconstruct import check_probability


SHAPE = (2048, 2048)
CROP = (88, 1960, 88, 1960)
D4 = [(k, flip) for k in (0, 2, 1, 3) for flip in (False, True)]
ENCODERS = {"resnet34", "tu-tf_efficientnetv2_s"}
LEGACY_RECIPE = "dariush_8d4_native_disk950_amp16_permodel_u8_and_float32_v1"
RECIPE = "dariush_8d4_native_disk950_amp16_permodel_u8_fp32_pixel_repair_v2"
BAD_CACHE = (ValueError, TypeError, KeyError, OSError, EOFError, BadZipFile)
BLEND_VERSION = 344704584
BLEND_KERNEL = "anthonytherrien/magfilo-blended-tile-u-net-for-solar-filaments"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    temp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def discover_039(root, receipt_setting):
    paths = [Path(receipt_setting)] if receipt_setting else sorted(root.rglob("solar_unet_receipt.json"))
    if len(paths) != 1 or not paths[0].is_file():
        raise FileNotFoundError(f"Mount Dariush solar-filament-unet-weights V2 (receipt found at {paths})")
    path = paths[0]
    receipt = read_json(path)
    models = receipt["ensemble"]["models"]
    if len(models) != 10 or {(m["encoder"], int(m["fold"])) for m in models} != {
            (encoder, fold) for encoder in ENCODERS for fold in range(5)}:
        raise ValueError("The 0.39 receipt must contain both encoders for each of the five folds")
    for model in models:
        model["fold"] = int(model["fold"])
        expected = f"unet_{model['encoder']}_fold{model['fold']}.pt"
        if model["file"] != expected or len(model["sha256"]) != 64:
            raise ValueError(f"Unexpected checkpoint in the receipt: {model}")
    fold_file = path.parent / "folds.csv"
    with fold_file.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    folds = {row["stem"]: int(row["fold"]) for row in rows}
    if len(rows) != 707 or len(folds) != 707 or set(folds.values()) != set(range(5)):
        raise ValueError("Expected the author's 707 physical-photo fold assignments")
    return path, receipt, models, folds


def gray_image(path):
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if gray.shape != SHAPE:
        raise ValueError(f"Unexpected photo size: {path}, {gray.shape}")
    return gray


def metadata_for(stem, image_sha, models, fold, receipt_sha, folds_sha):
    chosen = [m["sha256"] for m in models if fold is None or m["fold"] == fold]
    return {"stem": stem, "image_sha256": image_sha, "recipe": RECIPE, "shape": list(SHAPE),
            "fold": fold, "receipt_sha256": receipt_sha, "folds_sha256": folds_sha, "models": chosen}


def validate_cache(pack, expected, path):
    actual = json.loads(pack["provenance"].item())
    # Old full caches passed the same finite/range check. Their untouched maps are reusable.
    if actual != expected and actual != (expected | {"recipe": LEGACY_RECIPE}):
        raise ValueError(f"Cache provenance mismatch: {path}")


def validate_original(original, path):
    if original.shape != SHAPE or original.dtype != np.uint8:
        raise ValueError(f"Invalid original 0.39 probability bytes in {path}")
    return original


def load_p039(path, expected):
    with np.load(path, allow_pickle=False) as pack:
        validate_cache(pack, expected, path)
        return check_probability(pack["p039"], SHAPE), validate_original(pack["baseline_u8"], path)


def load_s0(path, expected):
    with np.load(path, allow_pickle=False) as pack:
        validate_cache(pack, expected, path)
        return validate_original(pack["baseline_u8"], path)


def cache_record(path, has_float):
    with np.load(path, allow_pickle=False) as pack:
        events = json.loads(pack["precision_events"].item()) if "precision_events" in pack else []
        return {"path": str(path), **json.loads(pack["provenance"].item()),
                "has_float": has_float, "precision_events": events}


def resolve_cached(stems, cache, mounted, meta):
    floats, native = {}, {}
    for stem in stems:
        roots = [cache] if mounted is None else [cache, mounted]
        for base in roots:
            path = base / f"{stem}.npz"
            if path.is_file():
                try:
                    load_p039(path, meta[stem])
                except BAD_CACHE as exc:
                    print(f"Ignoring unusable float cache {path}: {exc}", flush=True)
                else:
                    floats[stem] = native[stem] = path
                    break
            if stem in native:
                continue
            for s0_path in (base / f"{stem}.s0.npz", path):
                if s0_path.is_file():
                    try:
                        load_s0(s0_path, meta[stem])
                    except BAD_CACHE as exc:
                        print(f"Ignoring unusable baseline cache {s0_path}: {exc}", flush=True)
                    else:
                        native[stem] = s0_path
                        break
    return floats, native


def ensure_p039(cache, photos, meta, receipt_path, models, settings):
    cache.mkdir(parents=True, exist_ok=True)
    mounted = Path(settings["p039_input_cache"]) if settings["p039_input_cache"] else None
    floats, native = resolve_cached(photos, cache, mounted, meta)
    pending = [stem for stem in photos if stem not in floats]
    replayed = len(floats)
    print(f"P039: {replayed} cached, {len(pending)} missing ({len(native) - replayed} with S0 only)", flush=True)
    runtime = {}
    if pending and not settings["replay_only"]:
        runtime, new_floats, new_native = infer_pending(cache, pending, photos, meta, receipt_path, models, settings)
        floats.update(new_floats)
        native.update(new_native)
    runtime.update(replayed=replayed, computed=len(floats) - replayed,
                   missing_float=[s for s in photos if s not in floats],
                   missing_s0=[s for s in photos if s not in native])
    return floats, native, runtime


def invalid_pixels(prob):
    return ~np.isfinite(prob) | (prob < 0) | (prob > 1)


def repair_probability(raw, retry_fp32, oom_error):
    """Repair only invalid FP16 pixels; preserve the existing uint8 S0 path."""
    invalid = invalid_pixels(raw)
    count = int(invalid.sum())
    if not count:
        return raw, None
    event = {"initial_invalid": count, "nonfinite": int((~np.isfinite(raw)).sum()),
             "repaired": 0, "remaining": count, "retry": "fp32_8tta_single_view"}
    try:
        alternative = retry_fp32()
    except oom_error:
        event["retry"] = "fp32_out_of_memory"
        return raw, event
    except RuntimeError as exc:
        event.update(retry="fp32_runtime_error", error=str(exc))
        return raw, event
    valid = invalid & ~invalid_pixels(alternative)
    np.copyto(raw, alternative, where=valid)
    event["repaired"] = int(valid.sum())
    event["remaining"] -= event["repaired"]
    return raw, event


def save_completed_photo(cache, stem, meta, float_sum, integer_sum, count, events):
    original = np.rint(integer_sum.astype(np.float32) / count).astype(np.uint8)
    raw = None
    if not any(event["remaining"] for event in events):
        raw = (float_sum / count).astype(np.float32)
        try:
            check_probability(raw, SHAPE)
        except ValueError:
            events = [*events, {"retry": "ensemble_validation_failed", "remaining": int(invalid_pixels(raw).sum())}]
            raw = None
    payload = {"baseline_u8": original, "provenance": np.array(json.dumps(meta, sort_keys=True)),
               "precision_events": np.array(json.dumps(events, sort_keys=True))}
    if raw is None:
        path = cache / f"{stem}.s0.npz"
        save_npz(path, **payload)
        print(f"{stem}: float map unavailable; saved original S0 bytes in {path.name}", flush=True)
        return None, path
    path = cache / f"{stem}.npz"
    save_npz(path, p039=raw, **payload)
    return path, path


def predict_tta(torch, model, photo, device, chunk, amp=True):
    pixels, mean, std = photo
    with torch.inference_mode():
        x = torch.from_numpy(np.array(pixels, dtype=np.uint8, copy=True)).to(device).float().div_(255.)
        z = ((x - mean) / std)[None, None]
        acc = torch.zeros(SHAPE, dtype=torch.float32, device=device)
        for j in range(0, len(D4), chunk):
            ops = D4[j:j + chunk]
            inp = torch.cat([torch.rot90(z.flip((3,)) if flip else z, k, (2, 3)) for k, flip in ops])
            inp = inp.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(inp)
            for p, (k, flip) in zip(logits.float().sigmoid(), ops, strict=True):
                p = torch.rot90(p[0], -k, (0, 1))
                acc += p.flip((1,)) if flip else p
        acc /= len(D4)
        if not amp:
            return acc.cpu().numpy()
        return acc.cpu().numpy(), (acc * 255.).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()


def infer_pending(cache, pending, photos, meta, receipt_path, models, settings):
    import importlib
    import time
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU (T4 x2 recommended) or mount the complete P039 cache")
    base = receipt_path.parent
    for model in models:
        if sha256(base / model["file"]) != model["sha256"]:
            raise ValueError(f"Dariush checkpoint SHA-256 differs from receipt: {model['file']}")
    try:
        version = importlib.metadata.version("segmentation-models-pytorch")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version != "0.5.0":
        wheel = base / "segmentation_models_pytorch-0.5.0-py3-none-any.whl"
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)], check=True)
        importlib.invalidate_caches()
    import segmentation_models_pytorch as smp
    import timm

    seed = settings["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # Match the author. Record environment; byte parity is not promised.
    yy, xx = np.ogrid[:SHAPE[0], :SHAPE[1]]
    disk = (yy - (SHAPE[0] - 1) / 2)**2 + (xx - (SHAPE[1] - 1) / 2)**2 <= 950**2

    def prepare(stem):
        pixels = gray_image(photos[stem])
        region = pixels[disk].astype(np.float32) / 255.
        return pixels, float(region.mean()), float(np.float32(max(float(region.std()), 1e-3)))

    with ThreadPoolExecutor(8) as pool:
        inputs = list(pool.map(prepare, pending))
    scratch = Path(settings["scratch_root"])
    scratch.mkdir(parents=True, exist_ok=True)
    bytes_needed = len(pending) * int(np.prod(SHAPE)) * (8 + 2)
    if shutil.disk_usage(scratch).free < bytes_needed + 256 * 1024**2:
        raise OSError(f"Temporary inference needs approximately {bytes_needed / 1e9:.1f} GB free in {scratch}")
    devices = [torch.device(f"cuda:{i}") for i in range(min(settings["max_gpus"], torch.cuda.device_count()))]
    if not devices:
        raise ValueError("max_gpus must be positive")

    def predict_retry(model, photo, device, chunk):
        while True:
            try:
                return predict_tta(torch, model, photo, device, chunk), chunk
            except torch.cuda.OutOfMemoryError:
                if chunk == 1:
                    raise
            chunk //= 2
            torch.cuda.empty_cache()
            print(f"{device}: retry with {chunk} views per pass, still eight views total", flush=True)

    lock = Lock()
    todo = iter(models)
    counts = np.zeros(len(pending), np.uint8)
    expected = [len(meta[s]["models"]) for s in pending]
    logs, new_floats, new_native = {}, {}, {}
    events = {stem: [] for stem in pending}
    with tempfile.TemporaryDirectory(prefix="p039_", dir=scratch) as tmp:
        floats = np.memmap(Path(tmp) / "float.dat", mode="w+", dtype=np.float64, shape=(len(pending), *SHAPE))
        integers = np.memmap(Path(tmp) / "integer.dat", mode="w+", dtype=np.uint16, shape=(len(pending), *SHAPE))
        floats[:] = 0
        integers[:] = 0

        def worker(device):
            while True:
                with lock:
                    row = next(todo, None)
                if row is None:
                    return
                start = time.monotonic()
                chosen = [i for i, s in enumerate(pending) if meta[s]["fold"] in (None, row["fold"])]
                if not chosen:
                    continue
                model = smp.Unet(encoder_name=row["encoder"], encoder_weights=None, in_channels=1, classes=1)
                model.load_state_dict(torch.load(base / row["file"], map_location="cpu", weights_only=True), strict=True)
                model = model.to(device).to(memory_format=torch.channels_last).eval()
                chunk = 4
                for i in chosen:
                    stem = pending[i]
                    (raw, u8), chunk = predict_retry(model, inputs[i], device, chunk)
                    raw, event = repair_probability(raw, lambda: predict_tta(torch, model, inputs[i], device, 1, amp=False),
                                                    torch.cuda.OutOfMemoryError)
                    if event is not None:
                        event.update(stem=stem, checkpoint=row["file"], device=str(device), original_amp_batch=chunk)
                        print(f"P039 precision recovery: {event}", flush=True)
                        if event["retry"] == "fp32_out_of_memory":
                            torch.cuda.empty_cache()
                    with lock:
                        if event is not None:
                            events[stem].append(event)
                        if not any(item["remaining"] for item in events[stem]):
                            floats[i] += raw
                        integers[i] += u8
                        counts[i] += 1
                        complete = counts[i] == expected[i]
                    if complete:
                        p039_path, s0_path = save_completed_photo(cache, stem, meta[stem], floats[i], integers[i],
                                                                expected[i], events[stem])
                        with lock:
                            new_native[stem] = s0_path
                            if p039_path is not None:
                                new_floats[stem] = p039_path
                del model
                torch.cuda.empty_cache()
                with lock:
                    logs[row["file"]] = {"photos": len(chosen), "chunk": chunk, "device": str(device),
                                         "seconds": round(time.monotonic() - start, 1)}
                print(f"P039 {row['file']}: {logs[row['file']]}", flush=True)

        with ThreadPoolExecutor(len(devices)) as pool:
            jobs = [pool.submit(worker, device) for device in devices]
            for job in jobs:
                job.result()
        if list(counts) != expected or set(new_native) != set(pending):
            raise RuntimeError("The P039 inference coverage does not match its receipt")
        del floats, integers
    runtime = {"recipe": RECIPE, "smp": smp.__version__, "timm": timm.__version__, "torch": torch.__version__,
               "cuda": torch.version.cuda, "gpu": [torch.cuda.get_device_name(d) for d in devices], "models": logs,
               "precision_events": [event for stem in pending for event in sorted(events[stem], key=lambda e: e["checkpoint"])]}
    return runtime, new_floats, new_native


def blend_contract(path):
    data = read_json(path)
    expected = {"kernel": BLEND_KERNEL, "script_version_id": BLEND_VERSION, "canvas": list(SHAPE),
                "crop_yxyx": list(CROP), "coordinate_order": "yx", "probability_kind": "pre_threshold_foreground",
                "dtype": "float32", "tta_views": 8, "tile_shape": [1056, 1056], "tile_weighting": "cosine"}
    if any(data.get(key) != val for key, val in expected.items()):
        raise ValueError("Blend manifest does not describe native-coordinate V10 pre-threshold 8TTA probabilities")
    if not data.get("source_url") or not data.get("provenance_note") or not isinstance(data.get("images"), dict):
        raise ValueError("Blend needs a source URL, provenance note, and image roster")
    return data


def load_blend(base, item, image_sha):
    if item["image_sha256"] != image_sha:
        raise ValueError("Blend photo bytes differ from the official photo")
    name = item["file"]
    if Path(name).name != name or not name.endswith(".npy"):
        raise ValueError("Blend manifest requires a sibling .npy per photo")
    path = base / name
    if sha256(path) != item["probability_sha256"]:
        raise ValueError(f"Blend probability SHA-256 differs: {path}")
    pb = check_probability(np.load(path, allow_pickle=False), (CROP[1]-CROP[0], CROP[3]-CROP[2]))
    if not np.any((pb > 0) & (pb < 1)):
        raise ValueError("Blend input is binary or saturated, not a usable probability map")
    return pb
