"""Per-instance evidence, paired statistics, and visual review."""

from collections import Counter, defaultdict

import numpy as np
from pycocotools import mask as cm
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

from mul2_assets import iou, metrics
from oracle import overlaps


def geometry(mask):
    y, x = np.nonzero(cm.decode(mask))
    points = np.stack((x, y), axis=1).astype(float)
    center = points.mean(axis=0)
    zero = points - center
    eigen = np.linalg.eigvalsh(zero.T @ zero / len(points))
    area = len(points)
    length = float(np.sqrt(12 * eigen[-1] + 1))
    width = float(np.sqrt(12 * eigen[0] + 1))
    radius = float(np.linalg.norm(center - (np.array(mask["size"])[::-1] - 1) / 2) / 950)
    return {"area": area, "cx": float(center[0]), "cy": float(center[1]),
            "major_px": length, "minor_px": width, "elongation": length / width, "radius_r950": radius,
            "area_bin": ("<250" if area < 250 else "250–999" if area < 1000 else
                         "1000–3999" if area < 4000 else ">=4000"),
            "elongation_bin": ("<3" if length / width < 3 else "3–7" if length / width < 8 else ">=8"),
            "radial_bin": ("center" if radius < .5 else "middle" if radius < .8 else "limb")}


def view_errors(base, gt, meta, model, gt_ids, g_geometry, p_geometry):
    j, inter, g_frac, p_frac = overlaps(gt, base)
    hit = j > .5
    merge_link = (g_frac >= .20) & (p_frac >= .02)
    split_link = (g_frac >= .10) & (p_frac >= .30)
    merged = merge_link.sum(axis=0) >= 2
    split = (split_link.sum(axis=1) >= 2) & ((g_frac * split_link).sum(axis=1) >= .45)
    gt_rows = []
    pred_rows = []
    for k in range(len(gt)):
        best = float(j[k].max(initial=0))
        idx = int(np.argmax(j[k])) if best else None
        matched = bool(hit[k].any())
        merged_here = bool((merge_link[k] & merged).any())
        if matched:
            cause = "weak_boundary_tp" if best < .65 else "strong_tp"
        elif merged_here and split[k]:
            cause = "merge_and_split_fn"
        elif merged_here:
            cause = "overmerge_fn"
        elif split[k]:
            cause = "fragmentation_fn"
        elif best >= .30:
            cause = "near_miss_fn"
        elif best >= .10:
            cause = "weak_overlap_fn"
        else:
            cause = "complete_miss_fn"
        gt_rows.append(meta | {"model": model, "gt_id": gt_ids[k], "gt_index": k, **g_geometry[k],
                               "primary": cause, "hit": matched, "best_prediction": idx, "best_iou": best,
                               "near_miss": bool(.30 <= best <= .50), "split_flag": bool(split[k]),
                               "merge_flag": merged_here, "pq_loss_units": .5 if not matched else 1 - best})
    for k in range(len(base)):
        best = float(j[:, k].max(initial=0))
        matched = bool(hit[:, k].any())
        duplicate = bool(((j[:, k] >= .10) & hit.any(axis=1)).any()) and not matched
        split_here = bool((split & split_link[:, k]).any())
        if matched:
            cause = "tp"
        elif duplicate:
            cause = "duplicate_fp"
        elif merged[k]:
            cause = "overmerge_fp"
        elif split_here:
            cause = "fragment_fp"
        elif best >= .10:
            cause = "near_gt_fp"
        else:
            cause = "unmatched_fp"
        pred_rows.append(meta | {"model": model, "prediction": k, **p_geometry[k], "primary": cause,
                                 "hit": matched, "best_gt_id": gt_ids[int(np.argmax(j[:, k]))] if best else None,
                                 "best_iou": best, "duplicate_flag": duplicate, "split_flag": split_here,
                                 "merge_flag": bool(merged[k]), "pq_loss_units": 0. if matched else .5})
    links = [meta | {"model": model, "gt_id": gt_ids[g], "prediction": int(p),
                     "iou": float(j[g, p]), "intersection": int(inter[g, p]),
                     "gt_coverage": float(g_frac[g, p]), "pred_coverage": float(p_frac[g, p]),
                     "official_hit": bool(hit[g, p])}
             for g, p in zip(*np.where(inter > 0))]
    return gt_rows, pred_rows, links, hit


def prediction_links(a, s, meta):
    j, inter, af, sf = overlaps(a, s)
    i, k = linear_sum_assignment(j, maximize=True)
    paired = {(x, y) for x, y in zip(i, k) if j[x, y] > 0}
    return j, [meta | {"mul2_index": int(x), "s0_index": int(y), "iou": float(j[x, y]),
                       "intersection": int(inter[x, y]), "mul2_fraction": float(af[x, y]),
                       "s0_fraction": float(sf[x, y]), "best_total_iou_pair": (x, y) in paired}
               for x, y in zip(*np.where(inter > 0))]


def complement(a, s, pair_j, meta):
    result = []
    for left, right in zip(a, s, strict=True):
        state, reason, coexist = "both_fail", "shared_failure", None
        if left["hit"] and right["hit"]:
            state = "both_hit"
            diff = left["best_iou"] - right["best_iou"]
            reason = ("mul2_better_contour" if diff >= .10 else
                      "s0_better_contour" if diff <= -.10 else "contour_gap_under_0.10")
        elif left["hit"] or right["hit"]:
            won, lost = (left, right) if left["hit"] else (right, left)
            state = won["model"] + "_only"
            wi = won["best_prediction"]
            coexist = not bool((pair_j[wi] if left["hit"] else pair_j[:, wi]).any())
            if lost["merge_flag"]:
                reason = "separates_merge"
            elif lost["split_flag"]:
                reason = "repairs_fragmentation"
            elif lost["best_iou"] >= .30:
                reason = "near_miss_to_hit"
            elif coexist:
                reason = "independent_missing_instance"
            else:
                reason = "low_overlap_or_conflict"
        result.append(meta | {"gt_id": left["gt_id"], "gt_index": left["gt_index"], "state": state, "reason": reason,
                              "mul2_hit": left["hit"], "s0_hit": right["hit"],
                              "mul2_iou": left["best_iou"], "s0_iou": right["best_iou"],
                              "mul2_cause": left["primary"], "s0_cause": right["primary"],
                              "mul2_prediction": left["best_prediction"], "s0_prediction": right["best_prediction"],
                              "winner_can_coexist": coexist, "area_bin": left["area_bin"],
                              "elongation_bin": left["elongation_bin"], "radial_bin": left["radial_bin"]})
    return result


def grouped_errors(gt_rows, pred_rows):
    out = []
    for dimension in ("area_bin", "elongation_bin", "radial_bin"):
        bins = defaultdict(Counter)
        for kind, rows in (("GT", gt_rows), ("prediction", pred_rows)):
            for r in rows:
                key = r["scope"], r["model"], kind, r[dimension]
                bucket = bins[key]
                bucket["count"] += 1
                bucket["matched"] += r["hit"]
                bucket["iou_sum_matched"] += r["best_iou"] if r["hit"] else 0
                bucket["merge"] += r["merge_flag"]
                bucket["split"] += r["split_flag"]
                bucket["near_miss"] += r.get("near_miss", False)
                bucket["loss_units"] += r["pq_loss_units"]
        for (scope, model, kind, label), counts in sorted(bins.items()):
            out.append({"scope": scope, "model": model, "unit": kind, "feature": dimension, "bin": label,
                        **counts, "matched_rate": counts["matched"] / counts["count"]})
    return out


def interval(before, after, groups, draws=2000, seed=2026):
    dates = sorted(set(groups.values()))
    if len(dates) < 2:
        return [None, None]
    a = np.array([sum((v for s, v in before.items() if groups[s] == d), np.zeros(4)) for d in dates])
    b = np.array([sum((v for s, v in after.items() if groups[s] == d), np.zeros(4)) for d in dates])
    ids = np.random.default_rng(seed).integers(0, len(dates), (draws, len(dates)))
    diffs = []
    for values in (a, b):
        z = values[ids].sum(axis=1)
        d = z[:, 1] + .5 * (z[:, 2] + z[:, 3])
        diffs.append(np.divide(z[:, 0], d, out=np.zeros_like(d), where=d > 0))
    return np.quantile(diffs[1] - diffs[0], [.025, .975]).tolist()


def concentration(before, after, groups):
    keys = sorted(after)
    a, b = (sum((v for s, v in values.items() if s in after), np.zeros(4)) for values in (before, after))
    q = metrics(a)["pq"]
    gains = {}
    photo_pq = {}
    for s in keys:
        ds, tp, fp, fn = after[s] - before[s]
        gains[s] = float(ds - q * (tp + .5 * (fp + fn)))
        photo_pq[s] = metrics(after[s])["pq"] - metrics(before[s])["pq"]
    positive = sorted((g for g in gains.values() if g > 1e-10), reverse=True)
    good = {s for s in keys if photo_pq[s] > 1e-10}
    top = sorted((s for s in keys if gains[s] > 1e-10), key=lambda s: (-gains[s], s))[:3]
    rest = [s for s in keys if s not in top]
    outside = ([metrics(sum((v[s] for s in rest), np.zeros(4)))["pq"] for v in (before, after)]
               if rest else None)
    return {"delta_pq": metrics(b)["pq"] - q, "delta_tp": int(b[1] - a[1]), "delta_fp": int(b[2] - a[2]),
            "delta_fn": int(b[3] - a[3]), "delta_sq": metrics(b)["sq"] - metrics(a)["sq"],
            "photos_positive_local_pq": len(good),
            "photos_negative_local_pq": sum(p < -1e-10 for p in photo_pq.values()),
            "dates_positive_local_pq": len({groups[s] for s in good}),
            "top3_positive_pooled_margin_share": sum(positive[:3]) / sum(positive) if positive else None,
            "top3_positive_photos": top,
            "delta_pq_without_top3_at_fixed_solution": outside[1] - outside[0] if outside else None,
            "photo_pooled_margins": gains, "fixed_solution_date_bootstrap_95": interval(
                {s: before[s] for s in keys}, after, {s: groups[s] for s in keys})}


FEATURES = {
    "rescue": {"donor_area": 1, "donor_elongation": 1, "donor_model_support": 1,
               "nearest_anchor_centroid_px": 1},
    "replace": {"pair_iou": 1, "pair_min_containment": 1, "donor_model_support": 1,
                "abs_log_area_change": -1},
    "split": {"seed_coverage": 1, "min_child_inside_seed": 1, "child_area_ratio": 1, "children": -1},
}


def signal_audit(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["scope"], row["anchor"], row["operation"]].append(row)
    result = []
    for (scope, model, op), items in sorted(groups.items()):
        for key, direction in FEATURES[op].items():
            observed = [(direction * r[key], r["positive"], r["date"]) for r in items if r.get(key) is not None]
            if not observed:
                continue
            x, y, dates = map(np.asarray, zip(*observed))

            def auc(v, label):
                p, n = int(label.sum()), int((~label).sum())
                return float((rankdata(v)[label].sum() - p * (p + 1) / 2) / (p * n)) if p and n else None

            value = auc(x, y)
            distinct = np.unique(dates)
            bands = []
            if len(distinct) >= 2 and value is not None:
                indices = [np.flatnonzero(dates == date) for date in distinct]
                rng = np.random.default_rng(2026)
                for draw in rng.integers(0, len(indices), (400, len(indices))):
                    ix = np.concatenate([indices[i] for i in draw])
                    estimate = auc(x[ix], y[ix])
                    if estimate is not None:
                        bands.append(estimate)
            threshold = np.quantile(x, .75)
            upper = y[x >= threshold]
            result.append({"scope": scope, "anchor": model, "operation": op, "feature": key,
                           "direction": direction, "n": len(x), "dates": len(distinct), "positives": int(y.sum()),
                           "positive_rate": float(y.mean()), "auc": value,
                           "auc_date_low": float(np.quantile(bands, .025)) if bands else None,
                           "auc_date_high": float(np.quantile(bands, .975)) if bands else None,
                           "top_quartile_with_ties_n": len(upper),
                           "top_quartile_with_ties_precision": float(upper.mean())})
    return result
