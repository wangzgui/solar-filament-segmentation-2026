"""Run the F4 Candidate V1 ablations from archived predictions."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from scipy.optimize import linear_sum_assignment

from audit_assets import MUL2_ID, S0_FOLDS, S0_RECEIPT, lock_s0, pair_roster, physical_views, prepare_s0, table
from candidate import ARMS, MODELS, RULES, bank_for, decide, fit, predict, target
from diagnostics import geometry
from f4_original import PRIOR
from frozen import RECEIPT
from mul2_assets import (MUL2_SHA, assert_replay, input_assets, iou, load_test, metrics, read_receipt,
                         score, sha256, write_json)
from reconstruct import KEEP_039, base_cc
from rules import LOCAL, describe_pairs, materialize
from sprint import read_test_map, test_index, write_submission


F4 = "F1_conservative"
TERMS = ("iou_sum", "tp", "fp", "fn")


def champion_file(root):
    expected = PRIOR["hashes"][F4]
    names = ("sub_F4_conservative.csv", "f4_conservative_original.csv", "submissionlast.csv")
    for name in names:
        for path in sorted(root.rglob(name)):
            if sha256(path) == expected:
                return path
    raise FileNotFoundError("Mount the original F4 Conservative CSV from zhuowamg/fas-v2out")


def replay_anchor(base, probability, pairs):
    donor = base_cc(probability, KEEP_039, quantized=True)
    edits = describe_pairs(base, donor, pairs, probability, {"conservative": LOCAL["conservative"]})
    masks, _ = materialize(base, edits, {"edit": "conservative"})
    return masks


def historical_banks(records, photos, archive, identity, maps, truth, expected):
    result, controls = {}, []
    for rec in records:
        stem, role = rec["stem"], rec["scope"]
        base, views = read_receipt(archive, role, stem, photos[stem], identity)
        assert_replay(stem, "vote_both", base, truth[stem], expected[stem])
        probability = maps.pop(stem)
        anchors = replay_anchor(base, probability, RECEIPT["validation"][stem]["pairs"])
        observed = score(truth[stem], anchors)
        if not np.allclose(observed, PRIOR["validation"][stem][F4], rtol=0., atol=1e-9):
            raise ValueError(f"F4 validation replay failed for {stem}: {observed.tolist()}")
        controls.append({"stem": stem, "scope": role, **dict(zip(TERMS, observed, strict=True))})
        result[stem] = bank_for(anchors, views, probability)
    return result, controls


def baseline_price(banks, truth, stems):
    counts = sum((score(truth[s], banks[s].masks[:banks[s].n_anchor]) for s in stems), np.zeros(4))
    return .5 * metrics(counts)["pq"]


def fit_selection(banks, truth, groups):
    targets = {s: target(bank, truth[s]) for s, bank in banks.items()}
    dates = sorted(set(groups.values()))
    if len(dates) < 3:
        raise ValueError("At least three distinct Selection dates are required for grouped cross-fitting")
    estimate, prices, folds = {}, {}, []
    for date in dates:
        train = [s for s, d in groups.items() if d != date]
        hold = [s for s, d in groups.items() if d == date]
        models = fit(banks, targets, train)
        price = baseline_price(banks, truth, train)
        folds.append({"holdout_date": date, "train_stems": train, "held_out_stems": hold, "pq_price": price,
                      "models": models})
        for s in hold:
            estimate[s], prices[s] = predict(banks[s], models), price
    return estimate, prices, folds, fit(banks, targets, list(banks)), baseline_price(banks, truth, list(banks))


def changes(bank, picked):
    anchors = set(range(bank.n_anchor))
    removed, added = sorted(anchors - set(picked)), sorted(set(picked) - anchors)
    pairs = []
    if removed and added:
        local = bank.overlap[np.ix_(removed, added)]
        ii, jj = linear_sum_assignment(local, maximize=True)
        pairs = [(removed[i], added[j]) for i, j in zip(ii, jj, strict=True) if local[i, j] > 0.]
    return removed, added, pairs


def choice_log(meta, bank, picked, estimates, eligible, observed=None):
    removed, added, pairs = changes(bank, picked)
    replaced = dict(pairs)
    reverse = {j: i for i, j in pairs}
    chosen = set(picked)
    removed_set = set(removed)
    for i, node in enumerate(bank.nodes):
        selected = i in chosen
        action = ("keep" if selected else "replace_out" if i in replaced else "drop") if i < bank.n_anchor else (
            "replace_in" if i in reverse else "rescue" if selected else "unused")
        if meta["scope"] == "test" and action in ("keep", "unused"):
            continue
        blockers = [j for j in picked if bank.overlap[i, j] > 0.]
        yield {**meta, "node": i, "source": node["source"], "selected": selected, "action": action,
               "eligible_for_solver": i in eligible,
               "decision_context": "selected" if selected else "preselection_gate" if i not in eligible else
               "overlap_with_selected" if blockers else "local_solution_or_low_utility",
               "overlaps_selected_nodes": " ".join(map(str, blockers)),
               "swap_partner": replaced.get(i, reverse.get(i)), "conflict_anchors_removed": " ".join(
                   str(j) for j in sorted(removed_set) if bank.overlap[i, j] > 0.),
               "origin_views": " ".join(map(str, sorted(node["views"]))), "receipt_slots": json.dumps(node["refs"]),
               "area": node["area"], "elongation": node["elongation"], "radial": node["radial"],
               **{f"feature_{key}": node[key] for key in MODELS["relational"]},
               **{f"expected_iou_{name}": float(values[i]) for name, values in estimates.items()},
               **({"observed_iou_contribution": float(observed[i])} if observed is not None else {})}


def evaluate_scope(scope, banks, truth, gt_ids, dates, estimates, prices, arms, combo=None, frozen_decisions=None, baseline_arm='C0'):
    per_photo, instances, swaps_log, per_view, gt_changes, edge_rows, prediction_rows = [], [], [], [], [], [], []
    results = {}
    for stem, bank in banks.items():
        gt = truth[stem]
        original = bank.masks[:bank.n_anchor]
        base = metrics(score(gt, original))
        view_iou = {v: iou(masks, bank.masks) for v, masks in gt.items()}
        observed = target(bank, gt)
        gt_shapes = {v: [geometry(mask) for mask in masks] for v, masks in gt.items()}
        edge_rows.extend({"scope": scope, "stem": stem, **row} for row in bank.edges)
        for arm in arms:
            meta = {"scope": scope, "stem": stem, "date": dates[stem], "arm": arm}
            if frozen_decisions is None:
                picked, graph = decide(bank, estimates[stem], prices[stem], arm, combo)
                eligible = graph.pop('eligible_nodes')
            else:
                picked, graph = frozen_decisions[stem][arm]
                eligible = graph.get('eligible_nodes', set(picked))
                graph = {k: v for k, v in graph.items() if k != 'eligible_nodes'}
            removed, added, pairs = changes(bank, picked)
            selected_masks = [bank.masks[i] for i in picked]
            m = metrics(score(gt, selected_masks))
            action_counts = Counter()
            for view, matrix in view_iou.items():
                hit = matrix > .5
                after = hit[:, picked].any(axis=1)
                before = hit[:, :bank.n_anchor].any(axis=1)
                by_node = hit.any(axis=0)
                for i in picked:
                    node = bank.nodes[i]
                    area, elong, radial = node["area"], node["elongation"], node["radial"]
                    prediction_rows.append({**meta, "view": view, "node": i, "tp": int(by_node[i]), "fp": int(not by_node[i]),
                        "area_bin": "<250" if area < 250 else "250–999" if area < 1000 else "1000–3999" if area < 4000 else ">=4000",
                        "elongation_bin": "<3" if elong < 3 else "3–7" if elong < 8 else ">=8",
                        "radial_bin": "center" if radial < .5 else "middle" if radial < .8 else "limb"})
                action_counts.update(added_tp=int(by_node[added].sum()), added_fp=int((~by_node[added]).sum()),
                                     removed_tp=int(by_node[removed].sum()), removed_fp=int((~by_node[removed]).sum()),
                                     gt_gained=int((after & ~before).sum()), gt_lost=int((before & ~after).sum()))
                row_metrics = metrics(score({view: gt[view]}, selected_masks))
                per_view.append({**meta, "view": view, **row_metrics, "dq": row_metrics["rq"]})
                after_iou = matrix[:, picked].max(axis=1, initial=0.)
                before_iou = matrix[:, :bank.n_anchor].max(axis=1, initial=0.)
                for k, (prior, current) in enumerate(zip(before, after, strict=True)):
                    gt_changes.append({**meta, "view": view, "gt_id": gt_ids[stem][view][k],
                                       "before_tp": int(prior), "after_tp": int(current),
                                       "tp_flip": int(current) - int(prior), "before_iou": float(before_iou[k]),
                                       "after_iou": float(after_iou[k]), **gt_shapes[view][k]})
                for i, j in pairs:
                    gained = [gt_ids[stem][view][k] for k in np.flatnonzero(hit[:, j] & ~before)]
                    lost = [gt_ids[stem][view][k] for k in np.flatnonzero(hit[:, i] & ~after)]
                    swaps_log.append({**meta, "view": view, "old_node": i, "new_node": j, "iou_between_masks": bank.overlap[i, j],
                                      "old_tp": int(by_node[i]), "new_tp": int(by_node[j]), "gt_gained": json.dumps(gained),
                                      "gt_lost": json.dumps(lost), "tp_flip": len(gained) - len(lost)})
            per_photo.append({**meta, **m, "dq": m["rq"], "delta_pq": m["pq"] - base["pq"], "n_out": len(picked),
                              "dropped": len(removed) - len(pairs), "rescued": len(added) - len(pairs), "replaced": len(pairs),
                              **graph, **action_counts})
            instances.extend(choice_log(meta, bank, picked, estimates[stem], eligible, observed))
    base = metrics(sum((np.array([r[key] for key in TERMS]) for r in per_photo if r["arm"] == baseline_arm), np.zeros(4)))
    for arm in arms:
        rows = [r for r in per_photo if r["arm"] == arm]
        m = metrics(sum((np.array([r[k] for k in TERMS]) for r in rows), np.zeros(4)))
        results[arm] = {"scope": scope, "arm": arm, **m, "dq": m["rq"],
                        **{f"delta_{k}": m[k] - base[k] for k in ("pq", "tp", "fp", "fn")},
                        "negative_photos": sum(r["delta_pq"] < -1e-10 for r in rows),
                        "positive_dates": len({r["date"] for r in rows if r["delta_pq"] > 1e-10}),
                        **{k: sum(r[k] for r in rows) for k in ("dropped", "rescued", "replaced", "added_tp", "added_fp",
                                                                "removed_tp", "removed_fp", "gt_gained", "gt_lost")}}
    return results, per_photo, instances, swaps_log, per_view, gt_changes, edge_rows, prediction_rows


def save_scope(out, scope, result):
    _, photos, choices, swaps, views, targets, edges, predictions = result
    for name, rows in zip(("per_photo", "decisions", "swaps", "per_view", "gt_changes", "edges", "prediction_groups"),
                          (photos, choices, swaps, views, targets, edges, predictions), strict=True):
        table(out / f"{scope}_{name}.csv", rows)
    groups = defaultdict(Counter)
    for kind in ("area_bin", "elongation_bin", "radial_bin"):
        for row in targets:
            key = (row["arm"], kind, row[kind])
            groups[key].update(gt=1, gt_matched=row["after_tp"], gt_gained=int(row["tp_flip"] > 0), gt_lost=int(row["tp_flip"] < 0))
        for row in predictions:
            key = (row["arm"], kind, row[kind])
            groups[key].update(prediction_tp=row["tp"], prediction_fp=row["fp"])
    table(out / f"{scope}_groups.csv", [{"arm": arm, "group": group, "value": value, **counts}
                                         for (arm, group, value), counts in sorted(groups.items())])
    bucket = defaultdict(Counter)
    for row in choices:
        key = (row["arm"], row["source"], row["decision_context"])
        observed = row["observed_iou_contribution"]
        bucket[key].update(candidates=1, matched_any_view=int(observed > 0.), observed_iou_sum=observed)
        for name in MODELS:
            bucket[key][f"{name}_absolute_error"] += abs(row[f"expected_iou_{name}"] - observed)
    table(out / f"{scope}_candidate_diagnostics.csv", [
        {"arm": arm, "source": source, "decision_context": context, "candidates": counts["candidates"],
         "matched_any_view": counts["matched_any_view"],
         "mean_observed_iou_contribution": counts["observed_iou_sum"] / counts["candidates"],
         **{f"{name}_mae": counts[f"{name}_absolute_error"] / counts["candidates"] for name in MODELS}}
        for (arm, source, context), counts in sorted(bucket.items())])


def combination(selection):
    drop = max(("C1a", "C1b"), key=lambda a: selection[a]["pq"])
    add = max(("C2a", "C2b"), key=lambda a: selection[a]["pq"])
    return (drop, add) if all(selection[a]["delta_pq"] > 1e-10 and selection[a]["positive_dates"] >= 2
                              for a in (drop, add)) else None


def export(out, root, test, archive, identity, base_path, c0_path, models, s0_models, arms, combo, price, winner):
    test_ids = sorted(test)
    baseline, frozen = load_test(base_path, test_ids), load_test(c0_path, test_ids)
    index = test_index(root)
    reference_directory = f"Solar_S1_0/p039_cache/{S0_RECEIPT[:16]}_{S0_FOLDS[:12]}"
    index = {stem: [p for p in paths if p.parent.as_posix().endswith("/" + reference_directory) and p.name == f"{stem}.npz"]
             for stem, paths in index.items()}
    missing = [stem for stem in test_ids if not index.get(stem)]
    if missing:
        raise FileNotFoundError(f"Mount the original sanxiandata V2 test caches, missing {len(missing)}: {missing[:4]}")
    predictions, logs, photo_rows = {arm: {} for arm in arms}, [], []
    for step, (stem, image) in enumerate(sorted(test.items()), 1):
        base, views = read_receipt(archive, "test", stem, image, identity)
        if base != baseline[stem]:
            raise ValueError(f"Mul2 test receipt differs from its signed submission: {stem}")
        probability, path = read_test_map(index, stem, image, s0_models)
        anchors = replay_anchor(base, probability, RECEIPT["test"][stem])
        if anchors != frozen[stem]:
            raise ValueError(f"F4 test replay differs from the original champion CSV: {stem}")
        bank = bank_for(anchors, views, probability)
        estimates = predict(bank, models)
        for arm in arms:
            picked, graph = decide(bank, estimates, price, arm, combo)
            eligible = graph.pop("eligible_nodes")
            predictions[arm][stem] = [bank.masks[i] for i in picked]
            removed, added, pairs = changes(bank, picked)
            meta = {"scope": "test", "stem": stem, "arm": arm}
            logs.extend(choice_log(meta, bank, picked, estimates, eligible))
            photo_rows.append({**meta, **graph, "s0_cache": path, "n_out": len(picked), "dropped": len(removed) - len(pairs),
                               "rescued": len(added) - len(pairs), "replaced": len(pairs)})
        if step % 20 == 0 or step == len(test):
            print(f"Test {step}/{len(test)}", flush=True)
    if predictions["C0"] != frozen:
        raise ValueError("C0 changed the frozen test masks")
    hashes = {}
    for arm, images in predictions.items():
        path = out / f"sub_{arm}.csv"
        if arm == "C0":
            shutil.copyfile(c0_path, path)
        else:
            write_submission(path, images)
        hashes[arm] = sha256(path)
    backup = out / "f4_conservative_original.csv"
    shutil.copyfile(c0_path, backup)
    if hashes["C0"] != PRIOR["hashes"][F4] or sha256(backup) != hashes["C0"]:
        raise ValueError("F4 submission SHA-256 changed")
    for name in ("sublast.csv", "submissionlast.csv"):
        path = out / name
        shutil.copyfile(out / f"sub_{winner}.csv", path)
        if sha256(path) != hashes[winner]:
            raise ValueError(f"{name} differs from Selection winner")
    table(out / "test_decisions.csv", logs)
    table(out / "test_per_photo.csv", photo_rows)
    return hashes


def run(config, code_hash):
    start = time.monotonic()
    root = Path(config["input_root"])
    out = Path(config["output_root"]) / datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    phase = "input validation"
    try:
        write_json(out / "experiment_config.json", {"code_sha256": code_hash, "config": config, "rules": RULES,
                   "models": MODELS, "candidate_contours": "frozen F4 and raw archived donor contours",
                   "training": "Selection only, leave-one-date-out", "test_gt_access": False})
        archive, base, identity, annotation, photos, roster, _, expected = input_assets(root)
        champion = champion_file(root)
        receipt, s0_models, folds, s0_meta = lock_s0(root, config, photos)
        records = [r for r in pair_roster(roster, folds) if r["paired"]]
        cache_reference = json.loads(Path(__file__).with_name("reference_cache.json").read_text(encoding="utf-8"))
        if (identity != MUL2_ID or sha256(base) != MUL2_SHA or LOCAL != PRIOR["f4_parameters"] or
                {r["stem"]: r["scope"] for r in records} != {s: r["role"] for s, r in RECEIPT["validation"].items()} or
                set(photos["test"]) != set(RECEIPT["test"]) or set(cache_reference["suffixes"]) != {r["stem"] for r in records}):
            raise ValueError("F4 reference assets or historical splits differ from the signed replay")
        table(out / "photo_roster.csv", records)
        write_json(out / "input_identity.json", {"mul2_identity": identity, "mul2_sha256": sha256(base),
                   "f4_sha256": sha256(champion), "f4_path": str(champion), "s0": s0_meta,
                   "historical_s0_receipt_sha256": cache_reference["source_sha256"]})
        phase = "Selection cross-fitting"
        split = {r["scope"]: [] for r in records}
        for row in records:
            split[row["scope"]].append(row)
        dates = {r["stem"]: r["date"] for r in records}
        if {r["date"] for r in split["selection"]} & {r["date"] for r in split["audit"]}:
            raise ValueError("Selection and Audit dates overlap")
        scope_data = {}

        def prepare(scope):
            cache_out = out / scope
            cache_out.mkdir()
            subset = {r["stem"]: photos["train"][r["stem"]] for r in split[scope]}
            truth, ids = physical_views(annotation, subset)
            maps = prepare_s0(root, cache_out, config | {"replay_only": True,
                              "preferred_s0_cache_suffixes": cache_reference["suffixes"]}, receipt, s0_models, folds, subset)
            banks, controls = historical_banks(split[scope], subset, archive, identity, maps, truth, expected)
            table(cache_out / "replay_controls.csv", controls)
            return banks, truth, ids

        banks, truth, ids = prepare("selection")
        estimates, prices, oof, trained, price = fit_selection(banks, truth, {s: dates[s] for s in banks})
        write_json(out / "fit_models.json", {"folds": oof, "selection_full_models": trained, "pq_price": price})
        sel = evaluate_scope("selection", banks, truth, ids, dates, estimates, prices, ARMS)
        combo = combination(sel[0])
        arms = (*ARMS, "C6") if combo else ARMS
        if combo:
            sel = evaluate_scope("selection", banks, truth, ids, dates, estimates, prices, arms, combo)
        ordered = sorted(arms, key=lambda arm: (-sel[0][arm]["pq"], arms.index(arm)))
        winner = ordered[0]
        write_json(out / "selection_lock.json", {"winner": winner, "ordered": ordered, "C6": combo,
                   "C6_status": "active" if combo else "no_independent_selection_gain",
                   "tie_break": "exact pooled PQ, then declared arm order", "audit_used": False})
        table(out / "selection_ranking.csv", [{"rank": rank, **sel[0][arm]} for rank, arm in enumerate(ordered, 1)])
        save_scope(out, "selection", sel)
        scope_data["selection"] = sel[0]
        print(f"Selection locked: {winner}. Starting Audit.", flush=True)
        phase = "locked Audit evaluation"
        banks, truth, ids = prepare("audit")
        audit = evaluate_scope("audit", banks, truth, ids, dates, {s: predict(bank, trained) for s, bank in banks.items()},
                               dict.fromkeys(banks, price), arms, combo)
        scope_data["audit"] = audit[0]
        save_scope(out, "audit", audit)
        table(out / "selection_audit.csv", [scope_data[scope][arm] for scope in ("selection", "audit") for arm in ordered])
        phase = "test exports"
        hashes = export(out, root, photos["test"], archive, identity, base, champion, trained, s0_models, arms, combo, price, winner)
        write_json(out / "submission_manifest.json", {"winner": winner, "arm_sha256": hashes,
                   "sublast_sha256": hashes[winner], "original_f4_sha256": hashes["C0"], "test_gt_access": False})
        report = ["# F4-Candidate V1", "", f"Selection winner **{winner}**. C6 combination: {combo}.", "",
                  "| Arm | Selection PQ | Audit PQ | Audit ΔTP | Audit ΔFP | Audit rescued / dropped / swapped |",
                  "|---|---:|---:|---:|---:|---:|"]
        for arm in ordered:
            a, b = scope_data["selection"][arm], scope_data["audit"][arm]
            report.append(f"| {arm} | {a['pq']:.6f} | {b['pq']:.6f} | {b['delta_tp']:+} | {b['delta_fp']:+} | "
                          f"{b['rescued']} / {b['dropped']} / {b['replaced']} |")
        selected_audit = scope_data["audit"][winner]
        report.extend(["", f"For {winner}, Audit gained {selected_audit['gt_gained']} GT-view matches and lost "
                       f"{selected_audit['gt_lost']}. Added masks match in {selected_audit['added_tp']} annotator views "
                       f"and are FP in {selected_audit['added_fp']}. Removed masks were TP in {selected_audit['removed_tp']} "
                       f"views and FP in {selected_audit['removed_fp']}. {selected_audit['negative_photos']} Audit photos lost PQ.",
                       "Use the candidate diagnostics to compare the matched candidates gated out before selection, "
                       "candidates blocked by a selected mask, and candidates left out by the local objective."])
        report.extend(["", "Selection uses date-held-out predictions. Audit and test use the refit on all Selection dates.",
                       "These historical photos have repeated downstream exposure and potential upstream exposure. "
                       "Historical PQ gains need external validation.", "",
                       "All physical masks are shared across annotators. A swap is attributed by maximum-IoU one-to-one pairing "
                       "of removed anchors and selected donors. Actual pixel conflicts are listed in the decision table.", "",
                       "No Oracle choices or Audit labels enter the exported predictions. New masks retain their cached contours."])
        (out / "experiment_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    except Exception as exc:
        write_json(out / "status.json", {"status": "failed", "phase": phase, "error": str(exc),
                   "traceback": traceback.format_exc(), "seconds": time.monotonic() - start})
        raise
    write_json(out / "status.json", {"status": "complete", "winner": winner, "seconds": time.monotonic() - start})
    print(out / "experiment_summary.md", flush=True)
