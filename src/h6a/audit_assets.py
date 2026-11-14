"""Read-only Mul2 × S0 error audit and Oracle diagnostics."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
from pathlib import Path
import json
import time
import traceback

import numpy as np
from pycocotools import mask as cm
from pycocotools.coco import COCO

from diagnostics import (complement, concentration, geometry, grouped_errors, prediction_links,
                         signal_audit, view_errors)
from mul2_assets import (MUL2_SHA, assert_replay, canonical, input_assets, iou, legal, load_test, metrics,
                         read_receipt, score, sha256, write_csv, write_json)
from oracle import OracleProblem, candidate_terms, make_options, observable_effects, pooled_oracle
from probability import (BAD_CACHE, cache_record, discover_039, gray_image, infer_pending, load_s0,
                         metadata_for)
from reconstruct import KEEP_039, base_cc


MUL2_ID = "c7ee748b082c1dc0bffe09218b4a8291a5e287281c639eeaf1c2f0f2019bea64"
S0_RECEIPT = "4e7786a7260fb1c3787f573f86b52998134073a416ef3c5cd310016bf2994d78"
S0_RECEIPT_V3 = "40634b7106d309c48c1aa049410a65b85c4a0306f477f36b4bb37dcab035a7e6"
S0_FOLDS = "d850879ac8303f862a94d7265550bd688c07790373aeacbf0c778bdae9e6631a"
S0_SUBMITTED = "eb87c29dfb233d3d60ae24d805096f59f5d63995949ce63880c426beeeddea47"
S0_ARCHIVE_REPLAY = "4e7187f9bd998efad6cefc471ee62f5f5acf9f208d1cadc900d640d8e9d1b519"
SCOPES = ("audit", "selection", "paired_all")
METHODS = {"rescue": {"rescue"}, "replace": {"replace"}, "split": {"split"},
           "joint": {"rescue", "replace", "split"}}
MODELS = ("mul2", "s0")


def table(path, rows):
    columns = tuple(dict.fromkeys(key for row in rows for key in row))
    write_csv(path, rows, columns)


def physical_views(annotation, stems):
    catalog = COCO(str(annotation))
    views = {s: {} for s in stems}
    ids = {s: {} for s in stems}
    for view, image in catalog.imgs.items():
        stem = Path(image["file_name"]).stem
        if stem in views:
            records = [(ann["id"], canonical(catalog.annToRLE(ann))) for ann in catalog.imgToAnns[view]]
            valid = [(a, r) for a, r in records if int(cm.area(r))]
            views[stem][str(view)] = [r for _, r in valid]
            ids[stem][str(view)] = [a for a, _ in valid]
    if any(not v for v in views.values()):
        raise ValueError("A matched photo has no official annotator view")
    return views, ids


def lock_s0(root, cfg, photos):
    receipt_path, receipt, models, folds = discover_039(root, cfg["s0_receipt"])
    # The author added a learned variant in V3. Its v6 ensemble contract is byte-for-byte
    # identical as parsed metadata. Only the two known receipts and original fold table qualify.
    if sha256(receipt_path) not in (S0_RECEIPT, S0_RECEIPT_V3) or sha256(receipt_path.parent / "folds.csv") != S0_FOLDS:
        raise ValueError("S0 needs the pinned original ensemble receipt and fold roster")
    if set(folds) != set(photos["train"]):
        raise ValueError("The original fold roster and the official physical-photo roster differ")
    if (receipt["runs"]["v6"]["submission_file"]["sha256"] != S0_SUBMITTED
            or receipt["ensemble"]["inference"]["tta_views"] != 8
            or tuple(receipt["ensemble"]["keep_rule"][key]
                     for key in ("t_lo", "t_hi", "min_area", "min_meanp")) != KEEP_039):
        raise ValueError("The S0 recipe and submitted file identity disagree")
    for model in models:
        path = receipt_path.parent / model["file"]
        if sha256(path) != model["sha256"]:
            raise ValueError(f"Incorrect S0 checkpoint bytes: {path}")
    csv_receipt = {"original_submission_id": 57275747, "original_submission_sha256": S0_SUBMITTED,
                   "original_submission_rows": 1177, "original_submission_nonempty_photos": 177,
                   "original_submission_bytes_verified": False,
                   "archive_replay_sha256_expected": S0_ARCHIVE_REPLAY,
                   "archive_replay_bytes_verified": False,
                   "leaderboard_display": .39, "test_predictions_scored": 0}
    if cfg["s0_original_submission"]:
        path = Path(cfg["s0_original_submission"])
        if sha256(path) != S0_SUBMITTED:
            raise ValueError("s0_original_submission is not the author's actual submitted CSV")
        csv_receipt.update(original_submission_bytes_verified=True, original_path=str(path))
    pinned_replay = [Path(cfg["s0_archive_replay"])] if cfg["s0_archive_replay"] else sorted(
        p for p in root.rglob("*.csv") if p.name.lower() in
        {"submission_s0.csv", "submission.csv", "s0_submission.csv", "sub_f0_s.csv", "submissionlast.csv"})
    for path in pinned_replay:
        found = sha256(path)
        if found == S0_SUBMITTED:
            csv_receipt.update(original_submission_bytes_verified=True, original_path=str(path))
        if found == S0_ARCHIVE_REPLAY:
            csv_receipt.update(archive_replay_bytes_verified=True, replay_path=str(path))
        elif cfg["s0_archive_replay"]:
            raise ValueError("s0_archive_replay differs from the archived 8TTA S0 CSV")
    return receipt_path, models, folds, csv_receipt


def pair_roster(roster, folds):
    by_day = defaultdict(set)
    for stem, fold in folds.items():
        by_day[stem[:8]].add(fold)
    exposure = roster["upstream_exposure"]
    records = []
    for scope in ("audit", "selection"):
        for s in sorted(roster[scope]):
            # Unknown training exposure is never treated as clean.
            observed = exposure[s]
            fully_documented_clean = bool(observed) and all(value is False for value in observed.values())
            date_clean = by_day[s[:8]] == {folds[s]}
            records.append({"stem": s, "scope": scope, "date": s[:8], "s0_heldout_fold": folds[s],
                            "s0_other_training_fold_same_date": not date_clean, "paired": date_clean,
                            "mul2_selection_exposed": scope == "selection",
                            "mul2_has_upstream_exposure": any(value is True for value in observed.values()),
                            "mul2_has_unknown_exposure": any(value is None for value in observed.values()),
                            "both_checkpoint_exposure_clean_documented": date_clean and fully_documented_clean,
                            "s0_rule_uses_author_oof_history": True,
                            **observed})
    actual_dates = {r["date"] for r in records if r["scope"] == "audit"}
    if actual_dates & {r["date"] for r in records if r["scope"] == "selection"}:
        raise ValueError("An observation date crosses Mul2 Audit and Selection")
    return records


def prepare_s0(root, output, cfg, receipt, models, folds, images):
    key = f"{S0_RECEIPT[:16]}_{S0_FOLDS[:12]}"
    key_v3 = f"{S0_RECEIPT_V3[:16]}_{S0_FOLDS[:12]}"
    work = output / "p039_holdout" / key
    work.mkdir(parents=True, exist_ok=True)
    roots = [work] + [Path(p) for p in cfg["s0_oof_cache_dirs"]]
    if any(not p.is_dir() for p in roots):
        raise FileNotFoundError(f"A configured S0 held-out cache directory does not exist: {roots}")
    for prefix in (root, Path(cfg["output_root"]).parent):
        for signature in (key, key_v3):
            roots.extend(sorted(p for p in prefix.rglob(signature) if p.is_dir()))
    roots = list(dict.fromkeys(roots))
    meta = {s: metadata_for(s, sha256(path), models, folds[s], S0_RECEIPT, S0_FOLDS)
            for s, path in images.items()}
    result, result_meta, rejected = {}, {}, []
    for s in images:
        paths = [directory / filename for directory in roots
                 for filename in (f"{s}.npz", f"{s}.s0.npz")]
        suffix = cfg.get("preferred_s0_cache_suffixes", {}).get(s)
        if suffix:
            paths = [path for path in paths if path.as_posix().endswith("/" + suffix)]
        for path in paths:
            if not path.is_file():
                continue
            for candidate in (meta[s], meta[s] | {"receipt_sha256": S0_RECEIPT_V3}):
                try:
                    load_s0(path, candidate)
                except BAD_CACHE as exc:
                    problem = str(exc)
                else:
                    result[s] = path
                    result_meta[s] = candidate
                    break
            if s not in result:
                rejected.append({"stem": s, "file": str(path), "reason": problem})
                continue
            break
    pending = [s for s in images if s not in result]
    receipt_json = {"cached": len(result), "missing_before_inference": pending,
                    "cache_dirs": [str(x) for x in roots], "rejected": rejected, "test_forwards": 0}
    write_json(output / "s0_cache_manifest.json", receipt_json)
    print(f"S0 date-held-out maps: {len(result)} cached, {len(pending)} missing", flush=True)
    if pending:
        if cfg["replay_only"]:
            preferred = cfg.get("preferred_s0_cache_suffixes", {})
            raise FileNotFoundError(
                "Missing original F4 S0 OOF maps. Mount llooppoo/errordata and lojhhggg/sanxiandata V2. "
                f"Missing or rejected {len(pending)} photos. Examples: {[preferred.get(s, s) for s in pending[:3]]}. "
                "See s0_cache_manifest.json for the full list and rejected provenance.")
        runtime, _, fresh = infer_pending(work, pending, images, meta, receipt, models, cfg)
        result.update(fresh)
        result_meta.update({s: meta[s] for s in fresh})
        receipt_json.update(runtime=runtime, computed=len(fresh))
    if set(result) != set(images):
        raise RuntimeError("Incomplete paired S0 coverage")
    receipt_json["photos"] = {s: cache_record(p, p.name.endswith(".npz") and not p.name.endswith(".s0.npz"))
                              for s, p in result.items()}
    write_json(output / "s0_cache_manifest.json", receipt_json)
    return {s: load_s0(path, result_meta[s]) for s, path in result.items()}


def prediction_geometry(a, s, view_proposals, prob):
    result = {"mul2": [geometry(m) for m in a], "s0": [geometry(m) for m in s]}
    agreement = np.zeros(len(a), np.int32)
    for view in view_proposals:
        agreement += iou(a, [m for m, _ in view]).max(axis=1, initial=0) >= .30
    for row, support in zip(result["mul2"], agreement, strict=True):
        row["native_support"] = float(support / 4)
    for row, mask in zip(result["s0"], s, strict=True):
        row["native_support"] = float(prob[cm.decode(mask).astype(bool)].mean() / 255)
    return result


def scan_photo(stem, role, image, a, s, a_views, prob, truth, ids):
    photo = {"stem": stem, "scope": role, "date": stem[:8]}
    predictions = dict(zip(MODELS, (a, s), strict=True))
    geo = prediction_geometry(a, s, a_views, prob)
    pairs, prediction_pair_rows = prediction_links(a, s, photo)
    gt_rows, pred_rows, links, complementary = [], [], [], []
    ambiguity = {m: 0 for m in MODELS}
    for view, gt in truth.items():
        meta = photo | {"view": view}
        gg = [geometry(r) for r in gt]
        gts = []
        for name, masks in predictions.items():
            errors, fps, associations, hit = view_errors(masks, gt, meta, name, ids[view], gg, geo[name])
            gt_rows.extend(errors)
            pred_rows.extend(fps)
            links.extend(associations)
            gts.append(errors)
            ambiguity[name] += int((hit.sum(axis=0) > 1).sum() + (hit.sum(axis=1) > 1).sum())
        complementary.extend(complement(*gts, pairs, meta))
    return {"meta": photo, "image": image, "predictions": predictions, "geometry": geo,
            "truth": truth, "ids": ids, "gt_rows": gt_rows, "pred_rows": pred_rows,
            "links": links, "pred_pairs": prediction_pair_rows, "complement": complementary,
            "counts": {m: score(truth, masks) for m, masks in predictions.items()}, "ambiguity": ambiguity}


def effects_with_features(row, donor_geo, base_geo):
    for item in row:
        ds = [int(d) for d in item["donor_indices"].split(",") if d]
        d = [donor_geo[i] for i in ds]
        if not d:
            continue
        nearest = min((np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])
                       for a in d for b in base_geo), default=None)
        yield item | {"donor_model_support": float(np.mean([a["native_support"] for a in d])),
                      "donor_elongation": float(np.mean([a["elongation"] for a in d])),
                      "nearest_anchor_centroid_px": float(nearest) if nearest is not None else None,
                      "abs_log_area_change": abs(float(np.log(item["donor_to_anchor_area"])))
                      if item.get("donor_to_anchor_area") else None}


def oracle_study(rows, cfg):
    analyses, per_photo, decisions, flips, signals, bank_rows = [], [], [], [], [], []
    for anchor, donor in (MODELS, MODELS[::-1]):
        banks = {}
        for s, record in rows.items():
            if any(record["ambiguity"].values()):
                continue  # Both anchors share the same strict comparison cohort.
            base, other = record["predictions"][anchor], record["predictions"][donor]
            options, observable, omitted = make_options(base, other, record["truth"], anchor == "s0",
                                                        cfg["split_neighbors"], cfg["split_children"])
            stats, hits = candidate_terms(options, record["truth"])
            banks[s] = (base, options, observable, stats, hits)
            bank_rows.append(record["meta"] | {"anchor": anchor, "options": len(options),
                             "eligible": sum(o.eligible and not h[1] for o, h in zip(options, hits, strict=True)),
                             "split_neighbors_over_budget": omitted})
        for scope in SCOPES:
            stems = [s for s in banks if scope == "paired_all" or rows[s]["meta"]["scope"] == scope]
            if not stems:
                continue
            base_counts = {s: rows[s]["counts"][anchor] for s in stems}
            q = metrics(sum(base_counts.values(), np.zeros(4)))["pq"]
            groups = {s: s[:8] for s in stems}
            for s in stems:
                _, options, observable, stats, hits = banks[s]
                strict_actions = [(i, features) for i, features in observable if not hits[i][1]]
                effects = observable_effects(options, strict_actions, stats, q)
                for v in effects_with_features(effects, rows[s]["geometry"][donor], rows[s]["geometry"][anchor]):
                    signals.append(rows[s]["meta"] | v | {"scope": scope, "anchor": anchor, "donor": donor})
            for method, allowed in METHODS.items():
                if anchor == "mul2" and method == "split":
                    continue
                problems = {s: OracleProblem(banks[s][0], banks[s][1], banks[s][3], banks[s][4],
                                             rows[s]["truth"], allowed) for s in stems}
                solution, iterations = pooled_oracle(problems, q)
                terms = {s: v for s, (_, v) in solution.items()}
                gain = concentration(base_counts, terms, groups)
                margins = gain.pop("photo_pooled_margins")
                action_counts = Counter()
                for s, (picks, value) in solution.items():
                    old = metrics(base_counts[s])
                    new = metrics(value)
                    options = banks[s][1]
                    chosen = [options[i] for i in picks if options[i].op != "keep"]
                    action_counts.update(o.op for o in chosen)
                    per_photo.append({"scope": scope, "stem": s, "date": s[:8], "anchor": anchor,
                                      "oracle": method, "pooled_margin": margins[s], "anchor_pq": old["pq"],
                                      "delta_pq": new["pq"] - old["pq"], "delta_tp": new["tp"] - old["tp"],
                                      "delta_fp": new["fp"] - old["fp"], "delta_fn": new["fn"] - old["fn"],
                                      "delta_sq": new["sq"] - old["sq"], **new})
                    if not chosen:
                        continue
                    masks = [r for i in picks for r in options[i].masks]
                    for i in picks:
                        o = options[i]
                        if o.op != "keep":
                            decisions.append({"scope": scope, "stem": s, "anchor": anchor, "oracle": method,
                                              "option": i, "operation": o.op, "anchor_slot": o.slot,
                                              "donor_indices": ",".join(map(str, o.donor)), "recipe": o.mode})
                    if scope == "audit":
                        rows[s].setdefault("oracle_examples", {})[anchor, method] = masks
                    for view, gt in rows[s]["truth"].items():
                        pre = iou(gt, rows[s]["predictions"][anchor]).max(axis=1, initial=0)
                        post = iou(gt, masks).max(axis=1, initial=0)
                        for i, (p, n) in enumerate(zip(pre, post, strict=True)):
                            if (p > .5) != (n > .5) or abs(p - n) >= .10:
                                flips.append({"scope": scope, "stem": s, "view": view, "anchor": anchor,
                                              "oracle": method, "gt_id": rows[s]["ids"][view][i],
                                              "before_iou": float(p), "after_iou": float(n),
                                              "tp_flip": int(n > .5) - int(p > .5)})
                analyses.append({"scope": scope, "anchor": anchor, "oracle": method, "photos": len(stems),
                                 "baseline_pq": q, **metrics(sum(terms.values(), np.zeros(4))),
                                 **gain, "iterations": iterations, "selected": dict(action_counts)})
    return analyses, per_photo, decisions, flips, signals, bank_rows


def summarize(rows, oracles, signals):
    result = {}
    for scope in SCOPES:
        records = [r for r in rows.values() if scope == "paired_all" or r["meta"]["scope"] == scope]
        if not records:
            continue
        result[scope] = section = {"photos": len(records), "dates": len({r["meta"]["date"] for r in records}),
                                  "strict_oracle_photos": sum(not any(r["ambiguity"].values()) for r in records)}
        section["baselines"] = {}
        for model in MODELS:
            counts = sum((r["counts"][model] for r in records), np.zeros(4))
            q = metrics(counts)
            den = q["tp"] + .5 * (q["fp"] + q["fn"])
            budget = {"missed_gt": .5 * q["fn"], "false_predictions": .5 * q["fp"],
                      "matched_contour": q["tp"] - q["iou_sum"]}
            categories = defaultdict(lambda: {"n": 0, "units": 0.})
            for record in records:
                for row in (*record["gt_rows"], *record["pred_rows"]):
                    if row["model"] == model and row["pq_loss_units"] > 0:
                        label = row["primary"] if row["primary"] != "strong_tp" else "remaining_contour_tp"
                        categories[label]["n"] += 1
                        categories[label]["units"] += row["pq_loss_units"]
            ambiguity = sum(r["ambiguity"][model] for r in records)
            section["baselines"][model] = q | {"dq": q["rq"], "loss_budget": budget,
                "loss_fraction_of_pq_denominator": {k: v / den for k, v in budget.items()} if den else {},
                "top_error_categories": sorted(categories.items(), key=lambda a: -a[1]["units"])[:3],
                "all_error_categories": dict(categories), "ambiguous_official_relations": ambiguity,
                "raw_error_budget_reconciles": bool(np.isclose(sum(v["units"] for v in categories.values()),
                                                               sum(budget.values())))}
        joint = [c for r in records for c in r["complement"]]
        section["paired_states"] = dict(Counter(c["state"] for c in joint))
        section["paired_reasons"] = dict(Counter(c["state"] + "/" + c["reason"] for c in joint))
        section["common_failures"] = dict(Counter(c["mul2_cause"] + "/" + c["s0_cause"] for c in joint
                                                  if c["state"] == "both_fail"))
        section["paired_by_area"] = {label: dict(Counter(c["state"] for c in joint if c["area_bin"] == label))
                                     for label in sorted({c["area_bin"] for c in joint})}
    audit_ops = [v for v in oracles if v["scope"] == "audit" and v["oracle"] != "joint" and v["delta_pq"] > 1e-9]
    actionable = {(v["anchor"], v["operation"]) for v in signals if v["scope"] == "audit"
                  and v["auc_date_low"] is not None and v["auc_date_low"] > .50 and v["positives"] >= 5}
    qualified = [v for v in audit_ops if v["photos_positive_local_pq"] >= 3
                 and v["top3_positive_pooled_margin_share"] is not None
                 and v["top3_positive_pooled_margin_share"] < .75 and (v["anchor"], v["oracle"]) in actionable]
    ranked = sorted(qualified or audit_ops, key=lambda v: (-v["delta_pq"], -v["photos_positive_local_pq"]))
    if ranked:
        top = ranked[0]
        priority = {"anchor": top["anchor"], "operation": top["oracle"], "audit_delta_pq_upper_bound": top["delta_pq"],
                    "photos_positive_local_pq": top["photos_positive_local_pq"],
                    "top3_margin_share": top["top3_positive_pooled_margin_share"],
                    "status": "cross_photo_and_signal_supported" if qualified else "gt_only_opportunity_unvalidated"}
    else:
        priority = {"status": "no_supported_cross_model_operation",
                    "next_question": "Measure the most frequent common failure first on an independently labeled roster."}
    return result, priority


def visual_cases(out, rows, oracle_photos):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = [r for record in rows.values() for r in record["complement"]]
    candidates.sort(key=lambda r: (r["scope"] != "audit", -abs(r["mul2_iou"] - r["s0_iou"])))
    case_groups = {"mul2_only": lambda c: c["state"] == "mul2_only",
                   "s0_only": lambda c: c["state"] == "s0_only",
                   "both_fail": lambda c: c["state"] == "both_fail",
                   "near_miss": lambda c: c["reason"] == "near_miss_to_hit",
                   "contour": lambda c: "better_contour" in c["reason"],
                   "merge": lambda c: "overmerge" in c["mul2_cause"] or "overmerge" in c["s0_cause"],
                   "fragmentation": lambda c: "fragmentation" in c["mul2_cause"] or "fragmentation" in c["s0_cause"]}
    examples = [(label, next((c for c in candidates if condition(c)), None), None)
                for label, condition in case_groups.items()]
    for method in ("rescue", "replace", "split", "joint"):
        gained = [r for r in oracle_photos if r["scope"] == "audit" and r["oracle"] == method
                  and (r["anchor"], method) in rows[r["stem"]].get("oracle_examples", {}) and r["delta_pq"] > 0]
        if gained:
            r = max(gained, key=lambda v: v["pooled_margin"])
            before = rows[r["stem"]]["predictions"][r["anchor"]]
            after = rows[r["stem"]]["oracle_examples"][r["anchor"], method]
            same_photo = [c for c in candidates if c["stem"] == r["stem"]]
            focus = max(same_photo, key=lambda c: (iou([rows[c["stem"]]["truth"][c["view"]][c["gt_index"]]], after)
                                                  .max(initial=0) - iou([rows[c["stem"]]["truth"][c["view"]][c["gt_index"]]],
                                                                       before).max(initial=0)), default=None)
            examples.append(("oracle_" + method, focus, (r["anchor"], method)))
    plots = out / "figures"
    plots.mkdir(exist_ok=True)
    manifest = []
    colors = ("#34d399", "#60a5fa", "#fbbf24", "#fb7185", "#c084fc")
    for name, case, oracle in examples:
        if case is None:
            continue
        record = rows[case["stem"]]
        gt = record["truth"][case["view"]]
        photo = gray_image(record["image"])
        x, y, w, h = map(int, cm.toBbox(gt[case["gt_index"]]))
        padding = max(35, min(180, max(w, h) // 2))
        roi = (slice(max(0, y-padding), min(photo.shape[0], y+h+padding)),
               slice(max(0, x-padding), min(photo.shape[1], x+w+padding)))
        image = photo[roi]
        panels = [("Annotation view", gt), (f"Mul2 · IoU {case['mul2_iou']:.3f}", record["predictions"]["mul2"]),
                  (f"S0 OOF · IoU {case['s0_iou']:.3f}", record["predictions"]["s0"])]
        if oracle:
            panels.append((f"Oracle {oracle[1]} · {oracle[0]}", record["oracle_examples"][oracle]))
        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.3))
        for ax, (title, masks) in zip(axes, panels, strict=True):
            ax.imshow(image, cmap="gray", vmin=np.percentile(image, 2), vmax=np.percentile(image, 98))
            for i, rle in enumerate(masks):
                binary = cm.decode(rle)[roi]
                if binary.any() and not binary.all():
                    ax.contour(binary, [.5], colors=[colors[i % len(colors)]], linewidths=1.)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"{name.replace('_', ' ')} · {case['stem']} · view {case['view']} · GT {case['gt_id']}")
        fig.tight_layout()
        filename = f"{name}.png"
        fig.savefig(plots / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        manifest.append(case | {"figure": f"figures/{filename}"})
    return manifest


def save_labelled_masks(path, rows):
    def native(masks):
        return [rle | {"counts": rle["counts"].decode("ascii")} for rle in masks]

    with gzip.open(path, "wt", encoding="utf-8", compresslevel=3) as stream:
        for stem, record in rows.items():
            payload = {name: native(masks) for name, masks in record["predictions"].items()}
            edits = {f"{anchor}_{method}": native(masks)
                     for (anchor, method), masks in record.get("oracle_examples", {}).items()}
            stream.write(json.dumps({"stem": stem, "scope": record["meta"]["scope"], "baseline": payload,
                                     "audit_oracle_edits": edits}, allow_nan=False) + "\n")


def render_report(out, summary, oracles, signals, priority, s0_id, excluded):
    lines = ["# Solar Error Audit · Mul2 vote_both × S0 OOF", "",
             "主分析使用原 Mul2 Audit。Selection 和合并样本只作历史敏感性分析。"
             "S0 每张图使用原留出折的两套权重和原八视角流程。同日其他图跨折的照片已排除。"
             "Mul2 继承模型见过本次配对照片。S0 固定沿用原作者依据历史 OOF 确定的部署规则。"
             "两套方法的比较属于历史同图错误归因。", "",
             f"排除同日跨折照片 {len(excluded)} 张。S0 原作者实际提交文件字节核验为 "
             f"{s0_id['original_submission_bytes_verified']}，独立归档复现 CSV 字节核验为 "
             f"{s0_id['archive_replay_bytes_verified']}。排行榜显示精度为 0.39。", ""]
    for scope in SCOPES:
        if scope not in summary:
            continue
        v = summary[scope]
        lines += [f"## {scope} · {v['photos']} 张图，{v['dates']} 个日期，严格 Oracle {v['strict_oracle_photos']} 张", "",
                  "| 基线 | PQ | DQ | SQ | TP | FP | FN |", "|---|---:|---:|---:|---:|---:|---:|"]
        for model, m in v["baselines"].items():
            lines.append(f"| {model} | {m['pq']:.5f} | {m['dq']:.5f} | {m['sq']:.5f} | {m['tp']} | {m['fp']} | {m['fn']} |")
        lines += ["", "按 PQ 分母核算缺口。FN 和 FP 每个计 0.5，已匹配轮廓计 1−IoU。"
                  "合并或拆分可以同时造成 FN 和 FP，错误表保留两种计数及对应标记。", ""]
        for model, m in v["baselines"].items():
            top = "、".join(f"{k} {d['n']} 个／{d['units']:.2f}" for k, d in m["top_error_categories"])
            budget = "、".join(f"{k} {x:.1%}" for k, x in sorted(
                m["loss_fraction_of_pq_denominator"].items(), key=lambda kv: -kv[1]))
            lines.append(f"- **{model}** 的三类最大损失为 {top or '无'}。总体缺口为 {budget}。")
        states = "、".join(f"{k} {n}" for k, n in sorted(v["paired_states"].items()))
        common = "、".join(f"{k} {n}" for k, n in sorted(v["common_failures"].items(), key=lambda kv: -kv[1])[:3])
        reasons = "、".join(f"{k} {n}" for k, n in sorted(v["paired_reasons"].items()) if "_only/" in k)
        contours = "、".join(f"{k.split('/', 1)[1]} {n}" for k, n in sorted(v["paired_reasons"].items())
                             if k.startswith("both_hit/"))
        lines += ["", f"逐 GT 标注视角的命中关系为 {states or '无'}。主要共同失败为 {common or '无'}。",
                  f"各自独有命中的结构归因为 {reasons or '无'}。",
                  f"共同命中轮廓差异为 {contours or '无'}。", "",
                  "| 锚点 | Oracle | PQ | ΔPQ | ΔTP | ΔFP | ΔFN | 改善照片 | 前三图正贡献占比 | 固定解日期重采样 95% 区间 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in oracles:
            if r["scope"] != scope:
                continue
            share = r["top3_positive_pooled_margin_share"]
            share = "—" if share is None else f"{share:.0%}"
            low, high = r["fixed_solution_date_bootstrap_95"]
            band = "—" if low is None else f"[{low:+.5f}, {high:+.5f}]"
            lines.append(f"| {r['anchor']} | {r['oracle']} | {r['pq']:.5f} | {r['delta_pq']:+.5f} | "
                         f"{r['delta_tp']:+d} | {r['delta_fp']:+d} | {r['delta_fn']:+d} | "
                         f"{r['photos_positive_local_pq']} | {share} | {band} |")
        lines.append("")
    lines += ["## 单一攻关点", ""]
    if "operation" in priority:
        issue = {"rescue": "识别确实缺失且能合法共存的独立实例",
                 "replace": "判断已命中实例的哪一份轮廓更可靠",
                 "split": "识别并拆开 S0 错误合并的相邻实例"}[priority["operation"]]
        lines += [f"优先研究 **{issue}**（{priority['anchor']} 锚点）。"
                  f"Audit 的 GT 约束上限为 {priority['audit_delta_pq_upper_bound']:+.5f}，"
                  f"{priority['photos_positive_local_pq']} 张图的逐图 PQ 改善。"]
        if priority["status"] == "gt_only_opportunity_unvalidated":
            lines.append("当前只得到 GT 辅助机会。跨图一致性或预测时可见信号未通过本报告的证据标准。")
    else:
        lines.append("现有候选未给出正向的单操作 Audit 上限。下一步先复核双方共同失败最多的类型。")
    lines += ["", "## 预测时可见的证据", ""]
    signal_groups = defaultdict(list)
    for row in signals:
        if row["scope"] == "audit" and row["auc"] is not None:
            signal_groups[row["anchor"], row["operation"]].append(row)
    for (anchor, op), group in sorted(signal_groups.items()):
        best = max(group, key=lambda r: (r["auc_date_low"] or 0, r["auc"]))
        lower = best["auc_date_low"]
        supports = lower is not None and lower > .50 and best["positives"] >= 5
        lines.append(f"- {anchor} / {op} 的 {best['feature']} AUC {best['auc']:.3f}，"
                     f"{best['n']} 个可单独合法执行的候选中 {best['positives']} 个正边际。"
                     f"{'有初步排序信号。' if supports else '这批图尚不足以支持可靠判断。'}")
    if not signal_groups:
        lines.append("Audit 缺少同时含正负边际的操作候选，无法估计排序可靠性。")
    lines += ["", "IoU 命中、漏检和真实合并关系需要标注。可见特征包括预测之间的相交、形状、邻居距离和模型支持。"
              "AUC 仅衡量预先指定的单特征方向，正例定义为同一锚点 PQ 下的正净边际。"
              "多个特征在同一批照片上诊断，结果仍需独立验证。", "",
              "## 错误分类口径", "",
              "官方命中要求 IoU 严格大于 0.5。Near-miss 为 0.30–0.50，完整漏检为最佳 IoU 小于 0.10。"
              "已命中且 IoU 小于 0.65 记为低质量轮廓。"
              "合并标记要求同一预测分别覆盖至少两个 GT 各自至少 20%，且每个 GT 占预测至少 2%。"
              "拆分要求至少两个片段各覆盖 GT 至少 10%，GT 覆盖各片段至少 30%，联合覆盖 GT 至少 45%。"
              "误检与已命中 GT 的 IoU 达 0.10 时记为重复。分层表保留面积、二阶矩细长度和距盘心的位置。", "",
              "## Oracle 定义", "",
              "- Rescue 固定原实例，仅允许加入至少在一个标注视角补回漏检的原始掩码。",
              "- Replace 固定原实例槽位，仅允许用存在共同命中且几何相交的另一模型原掩码替换。",
              "- Split 以 S0 为锚点，用 2–4 个相邻 Mul2 实例作原掩码替代，或作 S0 原区域内的最近标记分区。"
              "每个 S0 种子最多考虑 8 个按覆盖度排序的邻居。每组在至少一个标注视角命中不同的真实实例。",
              "- Joint 在同一个候选集合中联合求解，保留其余锚点，约束像素互斥、候选唯一使用和每视角一对一。"
              "若原基线存在官方多重命中歧义，该图只保留在基线诊断。Oracle 直接最大化跨照片池化 PQ，增益不可相加。",
              "", "数值适用于报告中的候选池和历史配对样本。日期重采样对已选好的 Oracle 解重计分，"
              "区间衡量样本波动。去掉贡献最高的三张图后的固定解增益另见 oracle_summary.csv。"
              "训练曝光、S0 折外模型数量及候选约束都限制对排行榜的外推。"]
    (out / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(root, out, cfg, code_sha):
    archive, submitted, identity, annotation, photos, roster, groups, expected = input_assets(root)
    if identity != MUL2_ID:
        raise ValueError("Mul2 archive is not the signed vote_both run")
    mul2_test = load_test(submitted, list(photos["test"]))
    receipt, models, folds, s0_identity = lock_s0(root, cfg, photos)
    records = pair_roster(roster, folds)
    paired = [r for r in records if r["paired"]]
    if not paired:
        raise RuntimeError("The mounted assets contain no same-date-isolated S0 held-out photos with Mul2 receipts")
    write_json(out / "identity_and_scope.json", {"code_sha256": code_sha, "mul2_run_identity": identity,
               "mul2_submitted_sha256": MUL2_SHA, "mul2_test_roster_size": len(mul2_test),
               "mul2_test_instances": sum(map(len, mul2_test.values())),
               "mul2_archive": str(archive), "mul2_original_runtime": json.loads(
                   (archive / "run_identity.json").read_text(encoding="utf-8")),
               "mul2_settings_and_lineage": json.loads((archive / "run_summary.json").read_text(encoding="utf-8")),
               "s0": s0_identity, "s0_receipt_sha256": sha256(receipt), "s0_folds_sha256": S0_FOLDS,
               "s0_checkpoints": models, "s0_views": 8, "s0_keep_rule": list(KEEP_039),
               "s0_models_per_paired_photo": 2, "independent_end_to_end_evaluation": False,
               "both_checkpoint_exposure_clean_documented_photos": sum(
                   r["both_checkpoint_exposure_clean_documented"] for r in paired),
               "paired_counts": dict(Counter(r["scope"] for r in paired)), "test_metrics_computed": False})
    table(out / "pair_roster_and_exposure.csv", records)
    images = {r["stem"]: photos["train"][r["stem"]] for r in paired}
    truth, gt_ids = physical_views(annotation, images)
    a_cache = {}
    for r in paired:
        s, scope = r["stem"], r["scope"]
        a, view_proposals = read_receipt(archive, scope, s, images[s], identity)
        legal(a)
        assert_replay(s, "vote_both", a, truth[s], expected[s])
        a_cache[s] = a, view_proposals
    print(f"Verified Mul2 replay on {len(a_cache)} physical photos before S0 processing", flush=True)
    prob = prepare_s0(root, out, cfg, receipt, models, folds, images)
    rows = {}
    for index, r in enumerate(paired, 1):
        s = r["stem"]
        q = prob.pop(s)
        a, view_proposals = a_cache.pop(s)
        b = base_cc(q, KEEP_039, quantized=True)
        legal(b)
        rows[s] = scan_photo(s, r["scope"], images[s], a, b, view_proposals, q, truth[s], gt_ids[s])
        if index % 5 == 0 or index == len(paired):
            print(f"Instance audit: {index}/{len(paired)}", flush=True)
    gt_rows = [v for r in rows.values() for v in r["gt_rows"]]
    pred_rows = [v for r in rows.values() for v in r["pred_rows"]]
    complements = [v for r in rows.values() for v in r["complement"]]
    for name, data in (("gt_errors", gt_rows), ("prediction_errors", pred_rows), ("complementarity", complements),
                       ("paired_tp_flips", [c for c in complements if c["mul2_hit"] != c["s0_hit"]]),
                       ("candidate_gt_links", [v for r in rows.values() for v in r["links"]]),
                       ("candidate_source_pairs", [v for r in rows.values() for v in r["pred_pairs"]]),
                       ("stratified_errors", grouped_errors(gt_rows, pred_rows))):
        table(out / f"{name}.csv", data)
    photo_metrics = []
    for r in rows.values():
        p = {m: metrics(r["counts"][m]) for m in MODELS}
        photo_metrics.append(r["meta"] | {"mul2_group": groups[r["meta"]["stem"]],
                            "annotator_views": len(r["truth"]), "strict_oracle_eligible": not any(r["ambiguity"].values()),
                            **{f"{m}_ambiguous_official_relations": v for m, v in r["ambiguity"].items()},
                            **{f"{m}_{k}": x for m in MODELS for k, x in p[m].items()},
                            **{f"delta_s0_minus_mul2_{k}": p["s0"][k] - p["mul2"][k]
                               for k in ("pq", "tp", "fp", "fn", "sq", "rq")}})
    table(out / "per_photo_metrics.csv", photo_metrics)
    print("Optimizing legal Oracle candidate sets on labelled photos", flush=True)
    oracles, by_photo, decisions, flips, evidence, banks = oracle_study(rows, cfg)
    signals = signal_audit(evidence)
    for name, data in (("oracle_summary", oracles), ("oracle_per_photo", by_photo),
                       ("oracle_decisions", decisions), ("oracle_tp_flips", flips),
                       ("observable_candidate_evidence", evidence), ("observable_signals", signals),
                       ("oracle_candidate_bank", banks)):
        table(out / f"{name}.csv", data)
    summary, priority = summarize(rows, oracles, signals)
    write_json(out / "run_summary.json", {"scopes": summary, "oracles": oracles, "single_priority": priority})
    excluded = [r for r in records if not r["paired"]]
    render_report(out, summary, oracles, signals, priority, s0_identity, excluded)
    table(out / "representative_cases.csv", visual_cases(out, rows, by_photo))
    save_labelled_masks(out / "labelled_diagnostic_masks.jsonl.gz", rows)
    for scope, v in summary.items():
        print(f"{scope}: {v['photos']} photos | Mul2 PQ {v['baselines']['mul2']['pq']:.5f} | "
              f"S0 OOF PQ {v['baselines']['s0']['pq']:.5f}", flush=True)
    print(f"Audit report: {out / 'experiment_summary.md'}", flush=True)


def run(cfg, code_sha):
    start = time.monotonic()
    root = Path(cfg["input_root"])
    output = Path(cfg["output_root"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = output / f"run_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", cfg)
    try:
        execute(root, out, cfg, code_sha)
    except Exception as exc:
        write_json(out / "status.json", {"status": "failed", "error": str(exc),
                   "traceback": traceback.format_exc(), "seconds": time.monotonic() - start})
        raise
    write_json(out / "status.json", {"status": "complete", "seconds": time.monotonic() - start})
