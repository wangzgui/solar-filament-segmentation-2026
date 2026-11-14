"""Prediction-only interventions on the signed Candidate V1 decision policy."""

import numpy as np

from candidate import Bank, decide, predict
from mul2_assets import legal


BASE = ("H0", "H1", "H2")
NEW = ("H3a", "H3b", "H4a", "H4b", "H5", "H6a", "H6b", "H7")
ARMS = BASE + NEW
PARAMS = {"duplicate_veto_iou": .70, "duplicate_override_iou_gain": .08, "release_iou_min": .04,
          "anchor_post_floor": .065, "anchor_post_original_floor": .020,
          "donor_post_surplus": .10, "donor_post_s0_strong": .58,
          "donor_post_s0_weak": .52, "consensus_surplus": .07, "consensus_swap_margin": .045,
          "soft_anchor_max": .035, "soft_donor_max": .025}


def touching(bank, i, ids):
    return {j for j in ids if bank.overlap[i, j] > 0.}


def context_bank(bank, anchors, ids):
    """Recompute relational features in the active context from cached geometry."""
    active = sorted(anchors) + sorted(set(ids) - set(anchors))
    n_anchor = len(anchors)
    back = {node: k for k, node in enumerate(active)}
    nodes = [dict(bank.nodes[i]) for i in active]
    masks = [bank.masks[i] for i in active]
    overlap = bank.overlap[np.ix_(active, active)]
    areas = np.asarray([n["area"] for n in nodes], dtype=float)
    inter = overlap * (areas[:, None] + areas) / (1. + overlap)
    fractions = inter / areas[:, None]
    edges = [{**edge, "i": back[edge["i"]], "j": back[edge["j"]]} for edge in bank.edges
             if edge["i"] in back and edge["j"] in back]
    adjacent = [[] for _ in nodes]
    for edge in edges:
        i, j = edge["i"], edge["j"]
        adjacent[i].append((j, edge["relation"]))
        adjacent[j].append((i, edge["relation"]))
    for i, node in enumerate(nodes):
        related = adjacent[i]
        node.update(anchor_iou=float(overlap[i, :n_anchor].max(initial=0.)),
                    inside_anchor=float(fractions[i, :n_anchor].max(initial=0.)),
                    contains_anchor=float(fractions[:n_anchor, i].max(initial=0.)),
                    anchor_touch_count=int((overlap[i, :n_anchor] > 0.).sum()),
                    duplicate_donors=sum(j >= n_anchor and r == "duplicate" for j, r in related),
                    fragment_edges=sum(r == "fragment" for _, r in related),
                    extension_edges=sum(r == "extension" for _, r in related),
                    anchor_fragment_edges=sum(j < n_anchor and r == "fragment" for j, r in related),
                    stronger_competitor=max((nodes[j]["det_score"] * overlap[i, j] for j, r in related
                                            if r in ("duplicate", "fragment", "competition")), default=0.))
        # The trained role feature always identifies an original F4 mask.
    return Bank(masks, n_anchor, nodes, overlap, edges), active


def weak_or_contested(node):
    return ((node["support_views"] <= 1 and node["s0_mean"] <= .43) or
            node["fragment_edges"] > 0 or node["stronger_competitor"] >= .40)


def predelete(bank, models, initial, price, h1, h2, enhanced):
    all_ids = set(range(len(bank.nodes)))
    anchors = set(range(bank.n_anchor))
    removed = anchors - set(h1)
    if not removed:
        return list(h2[0]), {**h2[1], "released_eligible": 0, "context_candidates": len(bank.nodes)}
    surviving = anchors - removed
    base_eligible = h2[1]["eligible_nodes"]
    candidates = set(surviving)
    for i in all_ids - anchors:
        related = touching(bank, i, removed)
        if any(bank.overlap[i, j] >= PARAMS["duplicate_veto_iou"] and
               initial["relational"][i] - initial["relational"][j] < PARAMS["duplicate_override_iou_gain"] for j in related):
            continue
        unblocked = related and not touching(bank, i, surviving)
        reopened = enhanced and any(bank.overlap[i, j] >= PARAMS["release_iou_min"] for j in related)
        if i in base_eligible or unblocked or reopened:
            candidates.add(i)
    local, index = context_bank(bank, surviving, all_ids - removed)
    allowed = {k for k, i in enumerate(index) if i in candidates}
    picked, stats = decide(local, predict(local, models), price, "C5", donor_allow=allowed)
    return sorted(index[i] for i in picked), {"released_eligible": sum(bool(touching(bank, i, removed)) and
        i not in base_eligible for i in candidates - anchors), "context_candidates": len(index),
        "hard_edges": stats["hard_edges"], "eligible_nodes": {index[k] for k in stats["eligible_nodes"]}}


def postcheck(bank, models, initial, price, picked, h1, check_donors):
    anchors = set(range(bank.n_anchor))
    current = set(picked)
    local, index = context_bank(bank, current, range(len(bank.nodes)))
    after = predict(local, models)["relational"]
    before = initial["relational"]
    cut_anchors, cut_donors = set(), set()
    for k, i in enumerate(index[:local.n_anchor]):
        node = local.nodes[k]
        if i in anchors:
            if (after[k] < price - PARAMS["anchor_post_floor"] and before[i] < price - PARAMS["anchor_post_original_floor"]
                    and weak_or_contested(node)):
                cut_anchors.add(i)
        elif check_donors:
            weak_views = node["support_views"] <= 1
            fragile = node["s0_mean"] < PARAMS["donor_post_s0_weak"] or (
                node["anchor_fragment_edges"] > 0 and node["s0_mean"] < PARAMS["donor_post_s0_strong"])
            if weak_views and fragile and min(before[i], after[k]) <= price + PARAMS["donor_post_surplus"]:
                cut_donors.add(i)
    final = current - cut_anchors - cut_donors
    restored = set()
    for i in sorted(cut_donors):
        for j in sorted(touching(bank, i, (anchors - current) & set(h1)), key=lambda j: (-bank.overlap[i, j], j)):
            if before[j] >= price and not touching(bank, j, final):
                final.add(j)
                restored.add(j)
    return sorted(final), {"postcheck_anchor_drop": len(cut_anchors), "postcheck_donor_drop": len(cut_donors),
                           "restored_f4_after_rejection": len(restored), "eligible_nodes": current | restored}


def consensus(bank, initial, price, h1, h2):
    anchors = set(range(bank.n_anchor))
    common_drop = anchors - set(h1) - set(h2)
    final = anchors - common_drop
    values = initial["relational"]
    for i in sorted(set(h2) - anchors):
        conflicts = touching(bank, i, anchors)
        if (conflicts and conflicts <= common_drop and not touching(bank, i, final) and
                values[i] >= price + PARAMS["consensus_surplus"] and
                values[i] - max(values[j] for j in conflicts) >= PARAMS["consensus_swap_margin"]):
            final.add(i)
    return sorted(final), {"consensus_anchor_drop": len(common_drop), "eligible_nodes": anchors | (set(h2) - anchors)}


def soft(bank, initial, price, typed):
    adjusted = {name: values.copy() for name, values in initial.items()}
    values = adjusted["relational"]
    touched = 0
    for i, node in enumerate(bank.nodes):
        if i < bank.n_anchor and weak_or_contested(node):
            risk = float(np.clip((price + .02 - values[i]) / .16, 0., 1.))
            if typed:
                risk *= 1. if (node["duplicate_donors"] or node["fragment_edges"]) else .45
                if node["support_views"] >= 3 and node["s0_mean"] >= .58:
                    risk = 0.
            values[i] -= PARAMS["soft_anchor_max"] * risk
            touched += int(risk > 0.)
        elif typed and i >= bank.n_anchor and node["support_views"] <= 1:
            risk = float(np.clip((.62 - node["s0_mean"]) / .22, 0., 1.))
            values[i] -= PARAMS["soft_donor_max"] * risk
            touched += int(risk > 0.)
    picked, stats = decide(bank, adjusted, price, "C5")
    return picked, {**stats, "soft_risk_candidates": touched}


def controls(bank, initial, price):
    return {arm: decide(bank, initial, price, candidate) for arm, candidate in zip(BASE, ("C0", "C1b", "C5"), strict=True)}


def fusion(bank, models, initial, price, original):
    h1, h2 = original["H1"][0], original["H2"]
    predicted = dict(original)
    predicted["H3a"] = predelete(bank, models, initial, price, h1, h2, False)
    predicted["H3b"] = predelete(bank, models, initial, price, h1, h2, True)
    predicted["H4a"] = postcheck(bank, models, initial, price, h2[0], h1, False)
    predicted["H4b"] = postcheck(bank, models, initial, price, h2[0], h1, True)
    predicted["H5"] = consensus(bank, initial, price, h1, h2[0])
    predicted["H6a"] = soft(bank, initial, price, False)
    predicted["H6b"] = soft(bank, initial, price, True)
    predicted["H7"] = postcheck(bank, models, initial, price, predicted["H6b"][0], h1, True)
    for arm, (picked, _) in predicted.items():
        if len(picked) != len(set(picked)) or any(i < 0 or i >= len(bank.nodes) for i in picked):
            raise ValueError(f"Invalid selected nodes for {arm}")
        legal([bank.masks[i] for i in picked])
    return predicted
