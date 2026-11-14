"""Frozen Mul2 × S0 contour ablations and signed Kaggle CSVs."""

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import shutil
import time
import traceback

import numpy as np

from contours import ARMS, FEATURES, build_bank, estimates, fit, label_bank, replace_oracle
from diagnostics import interval
from mul2_assets import (HEADER, MUL2_SHA, assert_replay, input_assets, iou, legal, load_test, metrics,
                         read_receipt, score, sha256, write_json)
from probability import BAD_CACHE, gray_image, load_s0, metadata_for
from reconstruct import KEEP_039, base_cc
from audit_assets import (MUL2_ID, S0_ARCHIVE_REPLAY, S0_FOLDS, S0_RECEIPT, S0_RECEIPT_V3, lock_s0,
                          pair_roster, physical_views, prepare_s0, table)


ANCHOR = "mul2_exact"


def labelled_rows(records, images, maps, truth, gt_ids, archive, identity, expected):
    rows = {}
    for r in records:
        s = r["stem"]
        image = images[s]
        a, _ = read_receipt(archive, r["scope"], s, image, identity)
        assert_replay(s, "vote_both", a, truth[s], expected[s])
        probability = maps.pop(s)
        b = base_cc(probability, KEEP_039, quantized=True)
        bank = build_bank(a, b, probability, gray_image(image))
        delta, terms, counts = label_bank(bank, truth[s])
        rows[s] = {"stem": s, "role": r["scope"], "date": r["date"], "bank": bank, "truth": truth[s],
                   "gt_ids": gt_ids[s], "delta": delta, "terms": terms, "counts": counts}
    return rows


def all_models(rows, penalty):
    return {name: fit(rows, dim, penalty) if dim else None for name, dim in ARMS.items()}


def group_oof(rows, penalty):
    dates = sorted({r["date"] for r in rows.values()})
    if len(dates) < 3:
        raise ValueError("Selection must have at least three observation dates")
    result, provenance = {}, {}
    for date in dates:
        train = [r for r in rows.values() if r["date"] != date]
        models = all_models(train, penalty)
        held = [r for r in rows.values() if r["date"] == date]
        result.update({r["stem"]: models for r in held})
        provenance[date] = {"fit_stems": [r["stem"] for r in train],
                            "heldout_stems": [r["stem"] for r in held],
                            "fit_pairs": models["pair_geometry"]["fit_pairs"]}
    return result, provenance


def score_scope(rows, models_for, role):
    totals = defaultdict(dict)
    decision_rows, photo_rows, flip_rows = [], [], []
    swaps = defaultdict(int)
    dates = {s: r["date"] for s, r in rows.items()}
    baseline = metrics(sum((r["counts"] for r in rows.values()), np.zeros(4)))
    q = baseline["pq"]
    for stem, r in rows.items():
        b = r["bank"]
        models = models_for(stem)
        before = {view: iou(gt, b.base).max(axis=1, initial=0.) for view, gt in r["truth"].items()}
        for arm, model in models.items():
            estimated = estimates(b, model)
            selected = estimated > 0.
            result = b.materialize(selected)
            vals = score(r["truth"], result)
            totals[arm][stem] = vals
            swaps[arm] += int(selected.sum())
            photo_rows.append({"scope": role, "stem": stem, "date": r["date"], "arm": arm, "swaps": int(selected.sum()),
                               "eligible_pairs": len(b.pairs), "donors_blocked_by_other_anchors": b.blocked_donors,
                               **metrics(vals)})
            if arm == ANCHOR:
                continue
            if selected.any():
                for view, gt in r["truth"].items():
                    prior = before[view]
                    new = iou(gt, result).max(axis=1, initial=0.)
                    for i in np.flatnonzero((prior > .5) != (new > .5)):
                        flip_rows.append({"scope": role, "stem": stem, "arm": arm, "view": view,
                                          "gt_id": r["gt_ids"][view][i],
                                          "before_iou": float(prior[i]), "after_iou": float(new[i]),
                                          "tp_flip": int(new[i] > .5) - int(prior[i] > .5)})
            for i, (a, d) in enumerate(b.pairs):
                true_margin = (r["delta"][i, 0] - q * r["delta"][i, 1]) / len(r["truth"])
                decision_rows.append({"scope": role, "stem": stem, "date": r["date"], "arm": arm,
                                      "mul2_index": a, "s0_index": d, "selected": bool(selected[i]),
                                      "estimated_margin": float(estimated[i]), "gt_margin_per_view": float(true_margin),
                                      **dict(zip(FEATURES, map(float, b.features[i]), strict=True))})
        totals["s0_oof_reference"][stem] = score(r["truth"], b.donor)
        photo_rows.append({"scope": role, "stem": stem, "date": r["date"], "arm": "s0_oof_reference",
                           "swaps": None, "eligible_pairs": len(b.pairs),
                           "donors_blocked_by_other_anchors": b.blocked_donors,
                           **metrics(totals["s0_oof_reference"][stem])})
    out = {}
    for arm, by_photo in totals.items():
        m = metrics(sum(by_photo.values(), np.zeros(4)))
        local = [metrics(by_photo[s])["pq"] - metrics(totals[ANCHOR][s])["pq"] for s in rows]
        out[arm] = m | {"delta_pq": m["pq"] - q, "photos": len(rows),
                        "positive_photos": sum(v > 1e-10 for v in local), "negative_photos": sum(v < -1e-10 for v in local),
                        "date_bootstrap_95": interval(totals[ANCHOR], by_photo, dates),
                        "swaps": swaps[arm] if arm in ARMS else None}
    return out, photo_rows, decision_rows, flip_rows


def rank(results):
    return sorted(ARMS, key=lambda arm: (-round(results[arm]["pq"], 12), tuple(ARMS).index(arm)))


def test_index(root):
    keys = {f"{digest[:16]}_{S0_FOLDS[:12]}" for digest in (S0_RECEIPT, S0_RECEIPT_V3)}
    maps = defaultdict(list)
    for path in sorted(root.rglob("*.npz")):
        if path.parent.name in keys:
            stem = path.name.removesuffix(".npz").removesuffix(".s0")
            maps[stem].append(path)
    return maps


def read_test_map(index, stem, image, models):
    expected = metadata_for(stem, sha256(image), models, None, S0_RECEIPT, S0_FOLDS)
    diagnostics = []
    for path in index.get(stem, []):
        for digest in (S0_RECEIPT, S0_RECEIPT_V3):
            try:
                return load_s0(path, expected | {"receipt_sha256": digest}), str(path)
            except BAD_CACHE as exc:
                diagnostics.append(f"{path}: {exc}")
    raise FileNotFoundError(f"Missing signed 10-model S0 test map for {stem}. "
                            f"Mount sanxiandata Version 2. {diagnostics[:2]}")


def write_submission(path, predictions):
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(HEADER)
        for stem, masks in sorted(predictions.items()):
            legal(masks)
            writer.writerows((f"{stem}_{i}", mask["counts"].decode("ascii"))
                             for i, mask in enumerate(masks, 1))
    replay = load_test(path, sorted(predictions))
    if replay != predictions:
        raise ValueError(f"Submission round trip failed: {path}")


def export(root, out, test, baseline_file, models, trained, ranked):
    a = load_test(baseline_file, sorted(test))
    if len(a) != 180 or sum(map(len, a.values())) != 1233 or sha256(baseline_file) != MUL2_SHA:
        raise ValueError("The exact Mul2 180-photo/1233-instance baseline is required")
    paths = test_index(root)
    missing = sorted(set(test) - paths.keys())
    if missing:
        raise FileNotFoundError(f"Mount complete sanxiandata V2 test maps. Missing {len(missing)}, e.g. {missing[:4]}")
    output = {arm: {} for arm in ARMS}
    s0 = {}
    choices, by_photo, cache_sources = [], [], []
    for k, (stem, image) in enumerate(sorted(test.items()), 1):
        p, source = read_test_map(paths, stem, image, models)
        donor = base_cc(p, KEEP_039, quantized=True)
        s0[stem] = donor
        cache_sources.append({"stem": stem, "source": source})
        bank = build_bank(a[stem], donor, p, gray_image(image))
        for arm, model in trained.items():
            estimated = estimates(bank, model)
            selected = estimated > 0.
            masks = bank.materialize(selected)
            if len(masks) != len(a[stem]):
                raise ValueError(f"Contour-only submission changed instance count on {stem}")
            output[arm][stem] = masks
            by_photo.append({"stem": stem, "arm": arm, "instances": len(masks), "eligible_pairs": len(bank.pairs),
                             "swaps": int(selected.sum()), "s0_instances": len(donor),
                             "donors_blocked_by_other_anchors": bank.blocked_donors})
            if arm == ANCHOR:
                continue
            for i, (ai, si) in enumerate(bank.pairs):
                choices.append({"stem": stem, "arm": arm, "mul2_index": ai, "s0_index": si,
                                "estimated_margin": float(estimated[i]), "selected": bool(selected[i]),
                                **dict(zip(FEATURES, map(float, bank.features[i]), strict=True))})
        if k % 30 == 0:
            print(f"Frozen test replay {k}/180", flush=True)
    if output[ANCHOR] != a:
        raise ValueError("Mul2 anchor masks changed")
    folder = out / "submissions"
    folder.mkdir(exist_ok=True)
    s0_ref = folder / "submission_s0_reference.csv"
    staged = s0_ref.with_suffix(".staged")
    write_submission(staged, s0)
    if sha256(staged) != S0_ARCHIVE_REPLAY:
        raise ValueError("S0 frozen test decode differs from the pinned archived 8TTA replay CSV")
    staged.replace(s0_ref)
    manifest = {}
    for arm, masks in output.items():
        path = folder / f"submission_{arm}.csv"
        if arm == ANCHOR:
            shutil.copyfile(baseline_file, path)
            if sha256(path) != MUL2_SHA:
                raise ValueError("The copied Mul2 anchor bytes changed")
        else:
            write_submission(path, masks)
        manifest[arm] = {"file": str(path.relative_to(out)), "sha256": sha256(path), "photos": len(masks),
                         "rows": sum(map(len, masks.values())), "changed_photos": sum(masks[s] != a[s] for s in test),
                         "changed_masks": sum(r["swaps"] for r in by_photo if r["arm"] == arm)}
    best_experimental = next(arm for arm in ranked if arm != ANCHOR)
    for name, arm in (("submissionlast.csv", ranked[0]), ("sublast.csv", ranked[0]),
                      ("submission_best_experimental.csv", best_experimental)):
        dest = out / name
        shutil.copyfile(out / manifest[arm]["file"], dest)
        if sha256(dest) != manifest[arm]["sha256"]:
            raise ValueError(f"Alias hash differs: {name}")
    table(out / "test_per_photo.csv", by_photo)
    table(out / "test_decisions.csv", choices)
    table(out / "test_cache_sources.csv", cache_sources)
    return {"arms": manifest, "submissionlast": ranked[0], "best_experimental": best_experimental,
            "original_s0_reference_archive_sha256": sha256(s0_ref), "test_forwards": 0}


def write_report(out, selection, audit, ranked, deployment, oracle):
    lines = ["# Solar 040 · 同实例轮廓替换", "",
             f"Selection 日期折外锁定 **{ranked[0]}**。"
             f"正式 CSV 为 `submissionlast.csv`，最佳实验对照为 `submission_best_experimental.csv`。", "",
             "| 实验 | Selection OOF PQ | Audit PQ | Audit ΔPQ | Audit 替换数 | Audit TP/FP/FN | 测试改动掩码 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, s in selection.items():
        a = audit[arm]
        test = deployment["arms"].get(arm, {}).get("changed_masks")
        lines.append(f"| {arm} | {s['pq']:.5f} | {a['pq']:.5f} | {a['delta_pq']:+.5f} | "
                     f"{a['swaps'] if a['swaps'] is not None else '—'} | {a['tp']}/{a['fp']}/{a['fn']} | "
                     f"{test if test is not None else '—'} |")
    lines += ["", f"同一候选池的 GT Oracle 在 Selection 和 Audit 分别为 "
              f"{oracle['selection']['delta_pq']:+.5f} 和 {oracle['audit']['delta_pq']:+.5f}。"
              "Oracle 只用于历史诊断。", "",
              "同实例关联由最大像素 IoU 与唯一锚点决定。可替换的 S0 掩码与任何其他原 Mul2 掩码零交叠。"
              "合法替换之后逐图实例数等于 Mul2 原值。选型按 Selection 池化折外 PQ，Audit 不参与拟合和排名。", "",
              "Audit 已用于此前的研究方向诊断。两批历史图均有 Mul2 上游曝光。"
              "S0 历史折外使用两模型，测试用十模型的签名概率图。"
              "本地增益和 Public LB 需分别验证。"]
    (out / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(cfg, code_sha, out):
    root = Path(cfg["input_root"])
    archive, baseline, identity, annotation, photos, roster, _, expected = input_assets(root)
    if identity != MUL2_ID:
        raise ValueError("The Mul2 run identity differs from the audited vote_both")
    receipt, models, folds, source = lock_s0(root, cfg, photos)
    all_records = pair_roster(roster, folds)
    records = [r for r in all_records if r["paired"]]
    actual = {role: sum(r["scope"] == role for r in records) for role in ("selection", "audit")}
    if actual != {"selection": 16, "audit": 19}:
        raise ValueError(f"Fold and date roster differs from the audit: {actual}")
    table(out / "pair_roster_and_exposure.csv", all_records)
    write_json(out / "asset_identity.json", {"code_sha256": code_sha, "mul2_run_identity": identity,
               "mul2_sha256": MUL2_SHA, "s0": source, "s0_checkpoint_signatures": models,
               "photo_counts": actual, "test_labels_used": False})
    # Persist the signed incumbent before scoring experimental arms.
    reference = out / "submission_mul2_exact_reference.csv"
    shutil.copyfile(baseline, reference)
    if sha256(reference) != MUL2_SHA:
        raise ValueError("The exact Mul2 reference failed its hash check")
    images = {r["stem"]: photos["train"][r["stem"]] for r in records}
    probabilities = prepare_s0(root, out, cfg, receipt, models, folds, images)
    truth, gt_ids = physical_views(annotation, images)
    rows = labelled_rows(records, images, probabilities, truth, gt_ids, archive, identity, expected)
    selection = {s: r for s, r in rows.items() if r["role"] == "selection"}
    audit = {s: r for s, r in rows.items() if r["role"] == "audit"}
    oof, folds_used = group_oof(selection, cfg["ridge_penalty"])
    selected, sel_photo, sel_decisions, sel_flips = score_scope(selection, oof.__getitem__, "selection_oof")
    ranked = rank(selected)
    trained = all_models(list(selection.values()), cfg["ridge_penalty"])
    write_json(out / "selection_lock.json", {"criterion": "pooled date-OOF PQ, exact ties in declared arm order",
               "winner": ranked[0], "ranking": ranked, "selection_scores": selected,
               "date_folds": folds_used, "audit_used_for_selection": False})
    write_json(out / "fit_parameters.json", trained)
    print(f"Locked {ranked[0]} from Selection date-OOF PQ", flush=True)
    audited, audit_photo, audit_decisions, audit_flips = score_scope(audit, lambda _: trained, "audit")
    oracle = {key: replace_oracle(list(part.values())) for key, part in (("selection", selection), ("audit", audit))}
    table(out / "validation_summary.csv", [{"scope": role, "arm": a, **m}
          for role, group in (("selection_oof", selected), ("audit", audited)) for a, m in group.items()])
    table(out / "validation_per_photo.csv", sel_photo + audit_photo)
    table(out / "validation_decisions.csv", sel_decisions + audit_decisions)
    table(out / "validation_gt_flips.csv", sel_flips + audit_flips)
    write_json(out / "validation_oracle.json", oracle)
    deployment = export(root, out, photos["test"], baseline, models, trained, ranked)
    write_json(out / "submission_manifest.json", deployment)
    write_report(out, selected, audited, ranked, deployment, oracle)
    print(f"Validated submission: {out / 'submissionlast.csv'}", flush=True)


def run(cfg, code_sha):
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = Path(cfg["output_root"]) / f"run_{now}"
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", cfg)
    start = time.monotonic()
    try:
        execute(cfg, code_sha, out)
    except Exception as exc:
        write_json(out / "status.json", {"status": "failed", "error": str(exc),
                   "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - start})
        raise
    write_json(out / "status.json", {"status": "complete", "elapsed_seconds": time.monotonic() - start})
