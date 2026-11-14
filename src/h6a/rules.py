"""Interventions on the archived Contour Sprint pair roster."""

import numpy as np
from pycocotools import mask as cm
from scipy import ndimage as ndi

from contours import union_crop
from mul2_assets import iou, legal
from reconstruct import S8, encode_roi


F0, F1 = "F0_mul2", "F1_pair_probability"
THRESHOLDS = {"F2_t005": .005, "F2_t010": .010, "F2_t015": .015, "F2_t020": .020}
# Size and shape flags are evaluated jointly with pixel evidence.
STRUCTURE = {
    "conservative": dict(coverage=.75, length_ratio=1.35, elongation_ratio=1.7, area_ratio=1.6,
                         deep_loss=.09, bad_pixels=.24, mean_evidence=.14),
    "moderate": dict(coverage=.63, length_ratio=1.55, elongation_ratio=2.0, area_ratio=1.9,
                     deep_loss=.15, bad_pixels=.38, mean_evidence=.08),
}
LOCAL = {
    "conservative": dict(band=3., add_p=.55, remove_p=.22),
    "moderate": dict(band=6., add_p=.45, remove_p=.30),
}


def geometry(mask):
    coords = np.argwhere(mask).astype(float)
    coords -= coords.mean(axis=0)
    eig = np.linalg.eigvalsh(coords.T @ coords / len(coords))
    return float(np.sqrt(12 * eig[1] + 1)), float(np.sqrt((eig[1] + 1) / (eig[0] + 1)))


def protected(stats, cfg):
    def ratio_change(x):
        return max(x, 1 / x)

    geometry_flag = (min(stats["cover_mul2"], stats["cover_s0"]) < cfg["coverage"]
                     or ratio_change(stats["length_ratio"]) > cfg["length_ratio"]
                     or ratio_change(stats["elongation_ratio"]) > cfg["elongation_ratio"]
                     or ratio_change(stats["area_ratio"]) > cfg["area_ratio"]
                     or stats["deep_loss_fraction"] > cfg["deep_loss"])
    weak_evidence = (stats["dispute_bad_fraction"] > cfg["bad_pixels"]
                     or stats["dispute_mean_evidence"] < cfg["mean_evidence"])
    if stats["s0_components"] > stats["mul2_components"]:
        return "connectivity_worsens"
    if geometry_flag and weak_evidence:
        return "deformation_with_weak_pixel_evidence"
    return "accept"


def describe_pairs(base, donor, archived, probability, local_rules=LOCAL):
    legal(base)
    legal(donor)
    if probability.dtype != np.uint8:
        raise ValueError("Expected original uint8 S0 map")
    j = iou(base, donor)
    eligible = []
    for a, row in enumerate(j):
        best = row.max(initial=0.)
        ties = np.flatnonzero(row == best)
        if 0. < best < 1. and len(ties) == 1 and np.count_nonzero(j[:, ties[0]]) == 1:
            eligible.append((a, int(ties[0])))
    if set(eligible) != {(a, s) for a, s, _ in archived} or len(eligible) != len(archived):
        raise ValueError("Archived pairs differ from the exact upstream candidate pool")
    result = []
    for ai, si, margin in archived:
        a, b = base[ai], donor[si]
        box = union_crop(a, b, probability.shape, padding=8)
        x, y = cm.decode(a)[box].astype(bool), cm.decode(b)[box].astype(bool)
        p = probability[box].astype(np.float32) / 255.
        add, remove = y & ~x, x & ~y
        core = x & y
        inward = ndi.distance_transform_edt(np.pad(x, 1))[1:-1, 1:-1]
        outward = ndi.distance_transform_edt(~x)
        la, ea = geometry(x)
        lb, eb = geometry(y)
        nx, ny = (int(ndi.label(z, structure=S8)[1]) for z in (x, y))
        support = np.r_[p[add] - .55, .30 - p[remove]]
        stats = {"mul2_slot": ai, "s0_slot": si, "margin": margin, "f1_selected": margin > 0.,
                 "pair_iou": float(j[ai, si]),
                 "area_ratio": float(y.sum() / x.sum()), "cover_mul2": float(core.sum() / x.sum()),
                 "cover_s0": float(core.sum() / y.sum()), "length_ratio": lb / la, "elongation_ratio": eb / ea,
                 "mul2_components": nx, "s0_components": ny, "deep_loss_fraction": float((remove & (inward > 4)).sum() / x.sum()),
                 "dispute_bad_fraction": float((support < 0).mean()), "dispute_mean_evidence": float(support.mean()),
                 "s0_mean_probability": float(p[y].mean()), "added_pixels": int(add.sum()), "removed_pixels": int(remove.sum())}
        edits = {"whole": (b, int(add.sum()), int(remove.sum()), "applied")}
        for level, cfg in local_rules.items():
            trimmed = x & ~(remove & (inward <= cfg["band"]) & (p <= cfg["remove_p"]))
            grown = add & (outward <= cfg["band"]) & (p >= cfg["add_p"])
            proposed = ndi.binary_propagation(trimmed, structure=S8, mask=trimmed | grown)
            if not proposed.any() or not proposed[core].all():
                raise ValueError(f"Local {level} unexpectedly lost the shared core at pair {ai}/{si}")
            if ndi.label(proposed, structure=S8)[1] != nx:
                edits[level] = (a, 0, 0, "topology_or_core_rejected")
                continue
            ap, rp = int((proposed & ~x).sum()), int((x & ~proposed).sum())
            edits[level] = (encode_roi(proposed, box, probability.shape) if ap or rp else a,
                            ap, rp, "applied" if ap or rp else "no_supported_pixels")
        result.append((stats, edits))
    return result


def arms():
    return {F0: {"edit": "none"}, F1: {"edit": "whole", "threshold": 0.},
            **{name: {"edit": "whole", "threshold": value} for name, value in THRESHOLDS.items()},
            **{f"F3_{level}": {"edit": "whole", "threshold": 0., "structure": level} for level in STRUCTURE},
            **{f"F4_{level}": {"edit": level} for level in LOCAL}}


def materialize(base, candidates, recipe):
    masks = list(base)
    decisions = []
    for stats, edits in candidates:
        slot = stats["mul2_slot"]
        mode = recipe["edit"]
        reason = "baseline" if mode == "none" else "eligible"
        if mode != "none" and "threshold" in recipe and stats["margin"] <= recipe["threshold"]:
            reason = "quality_threshold"
        elif mode != "none" and "structure" in recipe:
            verdict = protected(stats, STRUCTURE[recipe["structure"]])
            if verdict != "accept":
                reason = verdict
        added, removed = 0, 0
        if reason == "eligible":
            masks[slot], added, removed, reason = edits[mode]
        decisions.append({"mul2_slot": slot, "s0_slot": stats["s0_slot"], "decision": reason,
                          "added_pixels": added, "removed_pixels": removed})
    legal(masks)
    return masks, decisions
