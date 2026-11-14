import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from pycocotools import mask as cm
from probability import SHAPE


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path, data):
    staged = path.with_suffix(".tmp")
    staged.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staged.replace(path)

def read_csv(path, header=None):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        if header and tuple(rows.fieldnames or ()) != header:
            raise ValueError(f"Invalid header: {path}")
        return list(rows)

def write_csv(path, rows, fields=None):
    fields = fields or (list(rows[0]) if rows else ("stem",))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def unique(paths, description):
    found = sorted(set(paths))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one {description}: {found}")
    return found[0]

def canonical(mask):
    size, counts = mask["size"], mask["counts"]
    if tuple(size) != SHAPE:
        raise ValueError(f"Unexpected RLE shape {size}")
    if isinstance(counts, list):
        return cm.frPyObjects({"size": list(size), "counts": counts}, *SHAPE)
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    if not counts or any(not 48 <= x <= 111 for x in counts):
        raise ValueError("Invalid compressed COCO RLE")
    return {"size": list(SHAPE), "counts": counts}

def iou(left, right):
    if not left or not right:
        return np.zeros((len(left), len(right)))
    return cm.iou(left, right, [0] * len(right))

def legal(masks):
    if masks:
        area = cm.area(masks)
        if not area.all() or int(area.sum()) != int(cm.area(cm.merge(masks))):
            raise ValueError("Empty or overlapping output instance")

def host_counts(ious):
    hit = ious > .5
    return np.array([ious[hit].sum(), hit.sum(), (~hit.any(axis=0)).sum(), (~hit.any(axis=1)).sum()], float)

def metrics(counts):
    value, tp, fp, fn = counts
    denominator = tp + .5 * (fp + fn)
    return {"pq": float(value / denominator) if denominator else 0.,
            "sq": float(value / tp) if tp else 0., "rq": float(tp / denominator) if denominator else 0.,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "iou_sum": float(value)}

def score(views, masks):
    return sum((host_counts(iou(truth, masks)) for truth in views.values()), np.zeros(4))

def assert_replay(stem, arm, masks, views, archived):
    actual = score(views, masks)
    target = [float(archived[k]) for k in ("iou_sum", "tp", "fp", "fn")]
    if not np.allclose(actual, target, rtol=0, atol=1e-9):
        raise ValueError(f"Cached prediction fails archived metric replay for {stem}, {arm}")

def load_test(path, stems):
    photos = {stem: [] for stem in stems}
    for row in read_csv(path, HEADER):
        stem, raw_index = row["filament_id"].rsplit("_", 1)
        if stem not in photos or raw_index != str(len(photos[stem]) + 1):
            raise ValueError(f"Unexpected test ID: {row['filament_id']}")
        photos[stem].append(canonical({"size": list(SHAPE), "counts": row["segmentation_rle"]}))
    for masks in photos.values():
        legal(masks)
    return photos

HEADER = ("filament_id", "segmentation_rle")

ANNOTATION = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

ANNOTATION_SHA = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"

MUL2_SHA = "ba78fdd88a08ec280197f4e2120a9abf850798de4983195c11f80f9c5fd795dd"

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def read_receipt(root, scope, stem, image, identity, include_contour=False):
    path = unique((p for p in (root / "photo_receipts" / scope).glob(f"{stem}.json*")
                   if not p.name.endswith(".tmp")), f"{scope} receipt for {stem}")
    if path.is_dir():
        path = unique((p for p in path.rglob("*") if p.is_file()), "expanded cached receipt")
    with path.open("rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8") as stream:
        receipt = json.load(stream)
    payload = receipt["payload"]
    if (receipt["key"] != digest([identity, scope, stem, sha256(image)])
            or receipt["payload_sha256"] != digest(payload)):
        raise ValueError(f"Cached Mul2 predictions fail image or content identity: {scope}/{stem}")
    if len(payload["views"]) != 4:
        raise ValueError(f"Expected four cached Mul2 views: {stem}")
    base, views = [canonical(mask) for mask in payload["masks"]["vote_both"]], [
        [(canonical(row["rle"]), float(row["score"])) for row in rows] for rows in payload["views"]
    ]
    if include_contour:
        return base, views, [canonical(mask) for mask in payload["masks"]["contour"]]
    return base, views

def input_assets(root):
    baseline = unique((p for p in root.rglob("submissionlast.csv")
                       if (p.parent / "run_identity.json").is_file() and sha256(p) == MUL2_SHA),
                      "signed Mul2 vote_both export from zhuowamg/mul2out")
    archive = baseline.parent
    audit = read_json(archive / "submission_audit.json")
    if audit["submissionlast"] != "vote_both" or audit["submissionlast_sha256"] != MUL2_SHA:
        raise ValueError("Mul2 submission audit disagrees with the pinned winner")
    identity = read_json(archive / "run_identity.json")["identity"]
    annotation = unique(root.rglob(ANNOTATION), "official competition train annotation JSON")
    if sha256(annotation) != ANNOTATION_SHA:
        raise ValueError("The competition annotation JSON differs from Mul2")
    image_dirs = {"train": annotation.parent / "train_images",
                  "test": annotation.parent.parent / "test/test_images"}
    photos = {role: {p.stem: p for p in folder.iterdir() if p.suffix.lower() in (".jpeg", ".jpg")}
              for role, folder in image_dirs.items()}
    if len(photos["test"]) != 180 or set(photos["test"]) & photos["train"].keys():
        raise ValueError("The official test roster should contain 180 disjoint photos")
    roster = read_json(archive / "validation_roster.json")
    groups = {}
    expected = {}
    for role, count in (("selection", 22), ("audit", 21)):
        if len(set(roster[role])) != count:
            raise ValueError(f"Unexpected original Mul2 {role} photo count")
        rows = [r for r in read_csv(archive / f"{role}_per_photo.csv") if r["arm"] == "vote_both"]
        if len(rows) != count or {r["stem"] for r in rows} != set(roster[role]):
            raise ValueError(f"Original Mul2 {role} metrics do not match its roster")
        for row in rows:
            groups[row["stem"]] = row["group"]
            expected[row["stem"]] = row
    if {groups[s] for s in roster["selection"]} & {groups[s] for s in roster["audit"]}:
        raise ValueError("A photo/date group crosses the development/audit split")
    return archive, baseline, identity, annotation, photos, roster, groups, expected