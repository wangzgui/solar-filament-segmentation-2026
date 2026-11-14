"""Generate H6a masks from signed intermediate predictions on Kaggle."""
import argparse
import json
from pathlib import Path

from audit_assets import MUL2_ID, S0_FOLDS, S0_RECEIPT
from candidate import bank_for, predict
from fusion_policy import soft
from mul2_assets import read_receipt, sha256, unique
from probability import discover_039
from reconstruct import KEEP_039, base_cc
from rules import LOCAL, describe_pairs, materialize
from sprint import read_test_map, test_index, write_submission


EXPECTED = "0d170beedd870d5375cb83c3c83d1bc57a3dd852fd715be5b5bec9292a6e5dee"


def rebuild_f4(base, probability):
    donor = base_cc(probability, KEEP_039, quantized=True)
    edits = describe_pairs(base, donor, None, probability, {"conservative": LOCAL["conservative"]})
    masks, _ = materialize(base, edits, {"edit": "conservative"})
    return masks


def run(root, output):
    reference = Path(__file__).with_name("fusion_reference.json")
    spec = json.loads(reference.read_text(encoding="utf-8"))["fit_models"]
    archive = unique(
        (p.parent for p in root.rglob("run_identity.json")
         if (p.parent / "photo_receipts" / "test").is_dir()
         and json.loads(p.read_text(encoding="utf-8")).get("identity") == MUL2_ID),
        "original Mul2 archive identity",
    )
    receipt, _, models, _ = discover_039(root, "")
    if sha256(receipt) != S0_RECEIPT or sha256(receipt.parent / "folds.csv") != S0_FOLDS:
        raise ValueError("Use the original V2 S0 receipt and fold table")
    test = unique((p for p in root.rglob("test_images") if p.parent.name == "test" and p.is_dir()),
                  "competition test_images directory")
    photos = {p.stem: p for p in test.iterdir() if p.suffix.lower() in (".jpeg", ".jpg")}
    if len(photos) != 180:
        raise ValueError(f"Expected 180 test images; found {len(photos)}")
    index = test_index(root)
    suffix = f"Solar_S1_0/p039_cache/{S0_RECEIPT[:16]}_{S0_FOLDS[:12]}"
    predictions = {}
    for k, (stem, photo) in enumerate(sorted(photos.items()), 1):
        base, views = read_receipt(archive, "test", stem, photo, MUL2_ID)
        maps = {stem: [p for p in index.get(stem, []) if p.name == f"{stem}.npz"
                       and p.parent.as_posix().endswith("/" + suffix)]}
        probability, _ = read_test_map(maps, stem, photo, models)
        bank = bank_for(rebuild_f4(base, probability), views, probability)
        scores = predict(bank, spec["selection_full_models"])
        selected, _ = soft(bank, scores, spec["pq_price"], False)
        predictions[stem] = [bank.masks[i] for i in selected]
        if k % 30 == 0:
            print(f"Recomputed {k}/180 photos", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".pending")
    try:
        write_submission(temporary, predictions)
        actual = sha256(temporary)
        if actual != EXPECTED:
            raise ValueError(f"Generated CSV SHA-256 {actual}; expected {EXPECTED}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Verified regenerated H6a: {output} ({actual})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/sub_H6a.csv"))
    args = parser.parse_args()
    run(args.input_root, args.output)
