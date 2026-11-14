"""Exact Candidate V1 replay, blind decision comparison, and Fusion V1 exports."""

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import time
import traceback

import numpy as np

from audit_assets import MUL2_ID, S0_FOLDS, S0_RECEIPT, lock_s0, pair_roster, physical_views, prepare_s0, table
from candidate import bank_for, predict
from fusion_policy import ARMS, BASE, NEW, PARAMS, controls, fusion, touching
from frozen import RECEIPT
from mul2_assets import MUL2_SHA, input_assets, load_test, metrics, read_receipt, sha256, write_json
from run_candidate import TERMS, changes, choice_log, evaluate_scope, replay_anchor, save_scope
from sprint import read_test_map, test_index, write_submission


ARCHIVED = dict(zip(BASE, ("C0", "C1b", "C5"), strict=True))


def mask_keys(bank):
    return [hashlib.sha256(mask["counts"]).hexdigest() for mask in bank.masks]


def comparison(stem, scope, bank, estimates, original):
    anchors = set(range(bank.n_anchor))
    one, two = (set(original[arm][0]) for arm in ("H1", "H2"))
    d1, d2, donors = anchors - one, anchors - two, two - anchors
    hashes = mask_keys(bank)
    vals = estimates["relational"]
    rows = []
    for i, node in enumerate(bank.nodes):
        if i not in anchors and scope == "test" and i not in donors and not touching(bank, i, d1):
            continue
        touched = touching(bank, i, d1)
        rows.append({"stem": stem, "scope": scope, "node": i, "stable_id": f"{stem}:{hashes[i]}",
                     "mask_sha256": hashes[i], "source": node["source"], "receipt_slots": json.dumps(node["refs"]),
                     "f4": i in anchors, "c1b": i in one, "c5": i in two, "shared_relational_score": float(vals[i]),
                     "c1b_deleted_neighbors": " ".join(hashes[j] for j in sorted(touched)),
                     "c1b_deleted_max_iou": max((float(bank.overlap[i, j]) for j in touched), default=0.),
                     "c5_operation": ("swap_candidate" if touching(bank, i, d2) else "rescue") if i in donors else
                                     "remove" if i in d2 else "keep" if i in anchors else "unused"})
    union = len(d1 | d2)
    return {"stem": stem, "scope": scope, "c1b_removed": len(d1), "c5_removed": len(d2), "common_removed": len(d1 & d2),
            "c1b_only_removed": len(d1 - d2), "c5_only_removed": len(d2 - d1), "c5_donors": len(donors),
            "c5_donors_touching_c1b_removals": sum(bool(touching(bank, i, d1)) for i in donors),
            "c5_other_donors": sum(not touching(bank, i, d1) for i in donors), "removal_jaccard": len(d1 & d2) / union if union else None,
            "c1b_changed": bool(d1), "c5_changed": bool(d2 or donors),
            "control_relational_score_corr": 1. if len(vals) > 1 and np.ptp(vals) > 0. else None}, rows


def check_models(spec, records):
    selection = {r["stem"]: r["date"] for r in records if r["scope"] == "selection"}
    all_dates = set(selection.values())
    folds = {row["holdout_date"]: row for row in spec["folds"]}
    if len(folds) != len(spec["folds"]) or set(folds) != all_dates:
        raise ValueError("Archived Candidate V1 OOF dates differ from the signed roster")
    for date, row in folds.items():
        if (set(row["held_out_stems"]) != {s for s, d in selection.items() if d == date} or
                set(row["train_stems"]) != {s for s, d in selection.items() if d != date}):
            raise ValueError(f"Archived Candidate V1 training crosses its held-out date: {date}")
    return folds


def verify_metric_controls(result, expected):
    for row in result[1]:
        arm = row["arm"]
        if arm in BASE:
            prior = expected[row["stem"]][ARCHIVED[arm]]
            if not np.allclose([row[k] for k in TERMS], prior, atol=1e-9, rtol=0.):
                raise ValueError(f"Historical control terms changed: {row['scope']} {row['stem']} {arm}")


def case_table(result):
    hits = defaultdict(dict)
    for row in result[5]:
        hits[row["stem"], row["view"], row["gt_id"]][row["arm"]] = row
    cases = []
    for variants in hits.values():
        origin = variants["H0"]["after_tp"]
        h1, h2 = (variants[a]["after_tp"] for a in ("H1", "H2"))
        for arm in NEW:
            row = variants[arm]
            value = row["after_tp"]
            if origin and not value:
                label = "lost_f4_tp"
            elif h2 and not value and not origin:
                label = "lost_c5_rescue"
            elif value and not (origin or h1 or h2):
                label = "fusion_only_recovery"
            else:
                continue
            cases.append({**row, "case": label, "h1_tp": h1, "h2_tp": h2})
    return cases


def paired_diagnostics(result, banks, predictions):
    by_photo = defaultdict(dict)
    dates = defaultdict(dict)
    for row in result[1]:
        by_photo[row["stem"]][row["arm"]] = row
        arm_rows = dates[row["date"]].setdefault(row["arm"], [])
        arm_rows.append(np.asarray([row[key] for key in TERMS]))
    paired = [{**r, "delta_vs_h1": r["pq"] - rows["H1"]["pq"], "delta_vs_h2": r["pq"] - rows["H2"]["pq"],
               "both_controls_positive_fusion_negative": rows["H1"]["delta_pq"] > 1e-10 and
               rows["H2"]["delta_pq"] > 1e-10 and r["delta_pq"] < -1e-10}
              for rows in by_photo.values() for r in rows.values()]
    date_rows = []
    for date, arms in dates.items():
        scores = {a: metrics(sum(rows, np.zeros(4))) for a, rows in arms.items()}
        date_rows.extend({"date": date, "arm": a, **m, "delta_vs_h0": m["pq"] - scores["H0"]["pq"],
                          "delta_vs_h1": m["pq"] - scores["H1"]["pq"], "delta_vs_h2": m["pq"] - scores["H2"]["pq"]}
                         for a, m in scores.items())
    nodes = []
    keys = {s: mask_keys(b) for s, b in banks.items()}
    for row in result[2]:
        arm, stem, i = (row[k] for k in ("arm", "stem", "node"))
        if arm not in NEW or row["source"] != "multiview":
            continue
        old = predictions[stem]
        bank = banks[stem]
        matching = row["observed_iou_contribution"] > 0.
        removed_h1 = set(range(bank.n_anchor)) - set(old["H1"][0])
        duplicates = [j for j in removed_h1 if bank.overlap[i, j] >= PARAMS["duplicate_veto_iou"]]
        if row["selected"] and not matching:
            label = "unmatched_donor_similar_to_c1b_removal" if duplicates else "unmatched_selected_donor"
        elif not row["selected"] and matching and i in old["H2"][0]:
            label = "h2_donor_matching_a_view_removed"
        else:
            continue
        nodes.append({**row, "case": label, "stable_id": f"{stem}:{keys[stem][i]}",
                      "similar_removed_mask_ids": " ".join(keys[stem][j] for j in sorted(duplicates))})
    return paired, date_rows, nodes


def write_report(out, ranked, selection, audit, diagnostics, hashes, changed):
    lines = ["# F4-Candidate Fusion V1", "", f"Selection winner **{ranked[0]}**. Known Public LB champion **H2 / C5**.",
             "H1 and H2 use the same archived relational Ridge estimates. The soft variants change how existing risk is priced.",
             "Historical Selection/Audit contain repeated experimental exposure. Public LB deltas are unknown.", "",
             "| Split | C1b removed | C5 removed | Both | C1b-only | C5-only | C5 donors touching C1b removals | Same edited photos |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for scope in ("selection", "audit", "test"):
        rows = [r for r in diagnostics if r["scope"] == scope]
        fields = ("c1b_removed", "c5_removed", "common_removed", "c1b_only_removed", "c5_only_removed", "c5_donors_touching_c1b_removals")
        items = [sum(r[k] for r in rows) for k in fields] + [sum(r["c1b_changed"] and r["c5_changed"] for r in rows)]
        lines.append("| " + " | ".join([scope, *map(str, items)]) + " |")
    lines.extend(["", "| Arm | Selection PQ | Audit PQ | Audit ΔPQ vs H2 | Added TP/FP | Removed TP/FP | Test photos changed vs H2 |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for arm in ranked:
        s, a = selection[arm], audit[arm]
        lines.append(f"| {arm} | {s['pq']:.6f} | {a['pq']:.6f} | {a['pq']-audit['H2']['pq']:+.6f} | "
                     f"{a['added_tp']}/{a['added_fp']} | {a['removed_tp']}/{a['removed_fp']} | {changed[arm]} |")
    aliases = defaultdict(list)
    for arm, digest in hashes.items():
        aliases[digest].append(arm)
    lines.extend(["", "Identical test outputs: " + str([v for v in aliases.values() if len(v) > 1]) + ".",
                  "The per-photo operation table records why a policy changes or keeps the local solution. "
                  "The failure table shows lost F4 matches, lost C5 rescues, and fusion-only recoveries.",
                  "F4 and donor contours are unchanged. Every arm uses the same signed cache. Oracle choices are excluded."])
    (out / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config, code_hash):
    start = time.monotonic()
    here, root = Path(__file__).parent, Path(config["input_root"])
    out = Path(config["output_root"]) / datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    phase = "preflight"
    try:
        reference = json.loads((here / "fusion_reference.json").read_text(encoding="utf-8"))
        cache = json.loads((here / "reference_cache.json").read_text(encoding="utf-8"))
        spec = reference["fit_models"]
        archive, mul2_file, identity, annotation, photos, roster, _, _ = input_assets(root)
        receipt, s0_models, fold_map, s0_meta = lock_s0(root, config, photos)
        records = [r for r in pair_roster(roster, fold_map) if r["paired"]]
        dates = {r["stem"]: r["date"] for r in records}
        if (identity != MUL2_ID or sha256(mul2_file) != MUL2_SHA or set(photos["test"]) != set(RECEIPT["test"]) or
                {r["stem"]: r["scope"] for r in records} != {s: r["role"] for s, r in RECEIPT["validation"].items()} or
                set(cache["suffixes"]) != set(dates)):
            raise ValueError("F4/Candidate V1 input identity differs")
        folds = check_models(spec, records)
        baseline, frozen = load_test(mul2_file, sorted(photos["test"])), {}
        for arm in BASE:
            path = here / f"reference_{arm}.csv"
            if sha256(path) != reference["sha256"][arm]:
                raise ValueError(f"Archived CSV hash differs for {arm}")
            frozen[arm] = load_test(path, sorted(photos["test"]))
        index = test_index(root)
        suffix = f"Solar_S1_0/p039_cache/{S0_RECEIPT[:16]}_{S0_FOLDS[:12]}"
        index = {s: [p for p in index.get(s, []) if p.name == f"{s}.npz" and p.parent.as_posix().endswith("/" + suffix)]
                 for s in photos["test"]}
        missing = [s for s, paths in index.items() if not paths]
        if missing:
            raise FileNotFoundError(f"Mount the original sanxiandata V2 test maps: {missing[:4]}")
        for scope, stems in [("test", photos["test"]), *[(s, [r["stem"] for r in records if r["scope"] == s]) for s in ("selection", "audit")]]:
            for stem in stems:
                if not any(p for p in (archive / "photo_receipts" / scope).glob(f"{stem}.json*") if not p.name.endswith(".tmp")):
                    raise FileNotFoundError(f"Mount Mul2 receipt {scope}/{stem}")
        write_json(out / "experiment_config.json", {"code_sha256": code_hash, "fusion_parameters": PARAMS,
                   "candidate_v1": reference["experiment_config"], "s0": s0_meta, "test_gt_access": False,
                   "c5_known_public_lb_champion": True, "h7": "H6b followed by H4b, declared before scoring"})
        write_json(out / "candidate_v1_fit_models.json", spec)
        table(out / "photo_roster.csv", records)
        all_data, comparisons, identities = {}, [], []
        phase = "prediction-only control replay and diagnosis"
        for scope in ("selection", "audit", "test"):
            is_test = scope == "test"
            subset = photos["test"] if is_test else {r["stem"]: photos["train"][r["stem"]] for r in records if r["scope"] == scope}
            scope_out = out / scope
            scope_out.mkdir()
            maps = {} if is_test else prepare_s0(root, scope_out, config | {"replay_only": True,
                          "preferred_s0_cache_suffixes": cache["suffixes"]}, receipt, s0_models, fold_map, subset)
            all_data[scope] = {}
            for k, (stem, photo) in enumerate(sorted(subset.items()), 1):
                base, views = read_receipt(archive, scope, stem, photo, identity)
                if is_test and base != baseline[stem]:
                    raise ValueError(f"Mul2 test receipt changed: {stem}")
                probability = read_test_map(index, stem, photo, s0_models)[0] if is_test else maps.pop(stem)
                pairs = RECEIPT["test"][stem] if is_test else RECEIPT["validation"][stem]["pairs"]
                bank = bank_for(replay_anchor(base, probability, pairs), views, probability)
                model, price = spec["selection_full_models"], spec["pq_price"]
                if scope == "selection":
                    fold = folds[dates[stem]]
                    model, price = fold["models"], fold["pq_price"]
                estimates = predict(bank, model)
                original = controls(bank, estimates, price)
                for arm, (picked, _) in original.items():
                    same = ([bank.masks[i] for i in picked] == frozen[arm][stem]) if is_test else (
                        picked == reference["historical_nodes"][scope][stem][ARCHIVED[arm]])
                    if not same:
                        raise ValueError(f"Exact Candidate V1 replay failed for {scope}/{stem}/{arm}")
                row, nodes = comparison(stem, scope, bank, estimates, original)
                comparisons.append(row)
                identities.extend(nodes)
                all_data[scope][stem] = (bank, model, estimates, price, original)
                if k % 30 == 0 or k == len(subset):
                    print(f"Frozen {scope} replay {k}/{len(subset)}", flush=True)
        table(out / "control_overlap_by_photo.csv", comparisons)
        table(out / "control_instance_identity.csv", identities)
        phase = "fusion and Selection lock"
        reports, decisions, operations = {}, {}, []
        for scope in ("selection", "audit", "test"):
            predictions = {}
            for stem, (bank, model, estimates, price, original) in all_data[scope].items():
                predictions[stem] = fusion(bank, model, estimates, price, original)
                for arm, (picked, stats) in predictions[stem].items():
                    removed, added, pairs = changes(bank, picked)
                    operations.append({"scope": scope, "stem": stem, "date": dates.get(stem, stem[:8]), "arm": arm,
                                       "dropped": len(removed) - len(pairs), "rescued": len(added) - len(pairs),
                                       "swapped": len(pairs), "net_instances": len(picked) - bank.n_anchor,
                                       "n_out": len(picked), **{k: v for k, v in stats.items() if k != "eligible_nodes"}})
            decisions[scope] = predictions
            if scope == "test":
                break
            banks = {s: row[0] for s, row in all_data[scope].items()}
            truth, ids = physical_views(annotation, {s: photos["train"][s] for s in banks})
            result = evaluate_scope(scope, banks, truth, ids, dates, {s: row[2] for s, row in all_data[scope].items()},
                                    {s: row[3] for s, row in all_data[scope].items()}, ARMS,
                                    frozen_decisions=predictions, baseline_arm="H0")
            verify_metric_controls(result, reference["historical_counts"][scope])
            reports[scope] = result[0]
            save_scope(out, scope, result)
            table(out / f"{scope}_failure_cases.csv", case_table(result))
            for name, rows in zip(("paired_deltas", "date_deltas", "donor_failure_cases"),
                                  paired_diagnostics(result, banks, predictions), strict=True):
                table(out / f"{scope}_{name}.csv", rows)
            if scope == "selection":
                ranked = sorted(ARMS, key=lambda a: (-result[0][a]["pq"], ARMS.index(a)))
                winner = ranked[0]
                write_json(out / "selection_lock.json", {"winner": winner, "ordered": ranked, "audit_used": False})
                print(f"Selection locked {winner}. Starting Audit.", flush=True)
                phase = "locked Audit and test"
        table(out / "operations_by_photo.csv", operations)
        table(out / "selection_audit.csv", [reports[s][a] for s in ("selection", "audit") for a in ranked])
        table(out / "selection_ranking.csv", [{"rank": k, **reports["selection"][a]} for k, a in enumerate(ranked, 1)])
        phase = "test submission verification"
        hashes, changed = {}, Counter()
        logs = []
        for arm in ARMS:
            path = out / f"sub_{arm}.csv"
            images = {}
            for stem, (bank, _, estimates, _, _) in all_data["test"].items():
                picked, details = decisions["test"][stem][arm]
                images[stem] = [bank.masks[i] for i in picked]
                changed[arm] += int(images[stem] != frozen["H2"][stem])
                keys = mask_keys(bank)
                for row in choice_log({"scope": "test", "stem": stem, "arm": arm}, bank, picked, estimates, details["eligible_nodes"]):
                    logs.append({**row, "stable_id": f"{stem}:{keys[row['node']]}", "mask_sha256": keys[row["node"]]})
            if arm in BASE:
                path.write_bytes((here / f"reference_{arm}.csv").read_bytes())
            else:
                write_submission(path, images)
            hashes[arm] = sha256(path)
        for arm, name in zip(BASE, ("F4_Conservative_Original.csv", "C1b_Original.csv", "C5_Public_LB_Champion.csv"), strict=True):
            (out / name).write_bytes((here / f"reference_{arm}.csv").read_bytes())
        table(out / "test_decisions.csv", logs)
        for name in ("sublast.csv", "submissionlast.csv"):
            path = out / name
            path.write_bytes((out / f"sub_{winner}.csv").read_bytes())
            if sha256(path) != hashes[winner]:
                raise ValueError(f"Submission winner SHA differs: {name}")
        write_json(out / "submission_manifest.json", {"winner": winner, "hashes": hashes, "known_public_lb_champion": "H2",
                   "changed_test_photos_vs_h2": dict(changed), "test_gt_access": False})
        write_report(out, ranked, reports["selection"], reports["audit"], comparisons, hashes, changed)
    except Exception as exc:
        write_json(out / "status.json", {"status": "failed", "phase": phase, "error": str(exc), "traceback": traceback.format_exc()})
        raise
    write_json(out / "status.json", {"status": "complete", "winner": winner, "seconds": time.monotonic() - start})
    print(out / "experiment_summary.md", flush=True)
