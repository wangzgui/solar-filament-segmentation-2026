"""Prediction-only features and legal local selection for the F4 candidate pool."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from pycocotools import mask as cm
from scipy import ndimage as ndi
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from skimage.morphology import skeletonize

from mul2_assets import iou, legal


ARMS = ("C0", "C1a", "C1b", "C2a", "C2b", "C3", "C4a", "C4b", "C5")
FEATURES = {
    "base": ("anchor", "det_score", "log_area", "log_elongation", "radial", "component_count", "support_views", "support_iou"),
    "s0": ("s0_mean", "s0_fraction", "s0_contrast"),
    "relation": ("anchor_iou", "inside_anchor", "contains_anchor", "duplicate_donors", "fragment_edges",
                 "extension_edges", "anchor_fragment_edges", "stronger_competitor", "anchor_touch_count"),
}
MODELS = {"visual": FEATURES["base"], "semantic": FEATURES["base"] + FEATURES["s0"],
          "relational": FEATURES["base"] + FEATURES["s0"] + FEATURES["relation"]}
RULES = {"ridge_alpha": 12., "drop_conservative": .11, "drop_relational": .065, "add_conservative": .055,
         "add_semantic": .035, "add_joint": .020, "drop_joint": .060, "swap_margin": .045,
         "relation_penalty": .070, "duplicate_iou": .55, "min_donor_area": 96,
         "max_relation_gap": 36., "min_endpoint_alignment": .80, "milp_seconds": 45.}


@dataclass
class Bank:
    masks: list
    n_anchor: int
    nodes: list
    overlap: np.ndarray
    edges: list


def mask_properties(mask, probability):
    x, y, w, h = map(int, cm.toBbox(mask))
    height, width = mask["size"]
    lo_y, hi_y = max(0, y - 4), min(height, y + h + 4)
    lo_x, hi_x = max(0, x - 4), min(width, x + w + 4)
    crop = cm.decode(mask)[lo_y:hi_y, lo_x:hi_x].astype(bool)
    if not crop.any():
        raise ValueError("Candidate mask is empty")
    yx = np.argwhere(crop).astype(float)
    center = yx.mean(axis=0)
    relative = yx - center
    eigen, directions = np.linalg.eigh(relative.T @ relative / len(yx))
    axis = directions[:, -1]
    skeleton = skeletonize(crop)
    neighborhood = ndi.convolve(skeleton.astype(np.int16), np.ones((3, 3), np.int16), mode="constant")
    ends = np.argwhere(skeleton & (neighborhood == 2))
    projection = (ends - center) @ axis if len(ends) else np.empty(0)
    if len(ends) > 2:
        ends = ends[[projection.argmin(), projection.argmax()]]
    shift = np.array([lo_y, lo_x])
    patch = probability[lo_y:hi_y, lo_x:hi_x]
    inside = patch[crop]
    ring = ndi.binary_dilation(crop, iterations=3) & ~crop
    mean = float(inside.mean()) / 255.
    length, thickness = np.sqrt(12 * eigen[::-1] + 1)
    radius = float(np.linalg.norm(center + shift - (np.array(mask["size"]) - 1) / 2.) / 950.)
    return {"area": len(yx), "log_area": float(np.log1p(len(yx))), "log_elongation": float(np.log(length / thickness)),
            "elongation": float(length / thickness), "radial": radius,
            "component_count": int(ndi.label(crop, structure=np.ones((3, 3)))[1]), "s0_mean": mean,
            "s0_fraction": float(np.mean(inside >= 128)),
            "s0_contrast": mean - float(patch[ring].mean()) / 255. if ring.any() else 0.,
            "box": (x, y, w, h), "axis": axis, "ends": ends + shift}


def endpoint_gap(left, right):
    if not len(left["ends"]) or not len(right["ends"]):
        return np.inf, 0.
    delta = right["ends"][None, :, :] - left["ends"][:, None, :]
    distance = np.linalg.norm(delta, axis=2)
    a, b = np.unravel_index(distance.argmin(), distance.shape)
    gap = float(distance[a, b])
    direction = delta[a, b] / max(gap, 1.)
    alignment = min(abs(left["axis"] @ direction), abs(right["axis"] @ direction),
                    abs(left["axis"] @ right["axis"]))
    return gap, float(alignment)


def relation_edges(nodes, overlap, fractions):
    edges = []
    for i, j in combinations(range(len(nodes)), 2):
        left, right = nodes[i], nodes[j]
        ax, ay, aw, ah = left["box"]
        bx, by, bw, bh = right["box"]
        box_gap = max(ax - bx - bw, bx - ax - aw, ay - by - bh, by - ay - ah, 0)
        if box_gap > RULES["max_relation_gap"]:
            continue
        gap, align = endpoint_gap(left, right)
        ij, ji = float(fractions[i, j]), float(fractions[j, i])
        ratio = min(left["area"], right["area"]) / max(left["area"], right["area"])
        value = float(overlap[i, j])
        continuous = gap <= RULES["max_relation_gap"] and align >= RULES["min_endpoint_alignment"]
        if value >= RULES["duplicate_iou"] or min(ij, ji) >= .72:
            relation = "duplicate"
        elif (max(ij, ji) >= .75 and ratio <= .64) or (value == 0. and continuous and ratio <= .40):
            relation = "fragment"
        elif value > 0.:
            relation = "competition"
        elif continuous:
            relation = "extension"
        else:
            relation = "nearby"
        edges.append({"i": i, "j": j, "relation": relation, "iou": value,
                      "left_fraction": ij, "right_fraction": ji, "area_ratio": ratio,
                      "endpoint_gap": gap if np.isfinite(gap) else None, "alignment": align,
                      "hard_conflict": value > 0.})
    return edges


def bank_for(anchors, views, probability):
    legal(anchors)
    n_anchor = len(anchors)
    masks, nodes = list(anchors), [{"source": "f4", "views": set(), "refs": [], "det_score": 0.} for _ in anchors]
    anchor_keys = {m["counts"]: k for k, m in enumerate(anchors)}
    lookup = dict(anchor_keys)
    for view, detections in enumerate(views):
        for slot, (mask, score) in enumerate(detections):
            if mask["size"] != list(probability.shape) or not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Candidate canvas or detector score is invalid")
            area = int(cm.area(mask))
            if area == 0:
                raise ValueError("Cached candidate is empty")
            if area < RULES["min_donor_area"]:
                continue
            key = mask["counts"]
            if key not in lookup:
                lookup[key] = len(masks)
                masks.append(mask)
                nodes.append({"source": "multiview", "views": set(), "refs": [], "det_score": 0.})
            node = nodes[lookup[key]]
            node["views"].add(view)
            node["refs"].append((view, slot))
            node["det_score"] = max(node["det_score"], score)
    if any(mask["size"] != list(probability.shape) for mask in masks):
        raise ValueError("Inconsistent F4 canvas")
    areas = np.array([int(cm.area(mask)) for mask in masks], dtype=float)
    overlap = iou(masks, masks)
    np.fill_diagonal(overlap, 0.)
    inter = overlap * (areas[:, None] + areas) / (1. + overlap)
    fractions = inter / areas[:, None]
    for node, mask in zip(nodes, masks, strict=True):
        node.update(mask_properties(mask, probability))
    edges = relation_edges(nodes, overlap, fractions)
    adjacent = [[] for _ in nodes]
    for edge in edges:
        adjacent[edge["i"]].append((edge["j"], edge))
        adjacent[edge["j"]].append((edge["i"], edge))
    cached_scores = [node["det_score"] for node in nodes]
    for i, node in enumerate(nodes):
        matching = [j for j in range(len(nodes)) if overlap[i, j] >= .5 and nodes[j]["refs"]]
        per_view = {v: max((overlap[i, j] for j in matching if v in nodes[j]["views"]), default=0.) for v in range(4)}
        for v in node["views"]:
            per_view[v] = 1.
        values = [v for v in per_view.values() if v]
        competitors = [j for j, e in adjacent[i] if e["relation"] in ("duplicate", "fragment", "competition")]
        strength = max((cached_scores[j] * overlap[i, j] for j in competitors), default=0.)
        # The same candidate score has the same meaning for anchors and donors.
        if i < n_anchor:
            node["det_score"] = max([cached_scores[i], *(cached_scores[j] for j in matching)])
        node.update(anchor=int(i < n_anchor), support_views=len(values), support_iou=float(np.mean(values)) if values else 0.,
                    anchor_iou=float(overlap[i, :n_anchor].max(initial=0.)),
                    inside_anchor=float(fractions[i, :n_anchor].max(initial=0.)),
                    contains_anchor=float(fractions[:n_anchor, i].max(initial=0.)),
                    anchor_touch_count=int((overlap[i, :n_anchor] > 0.).sum()),
                    duplicate_donors=sum(j >= n_anchor and e["relation"] == "duplicate" for j, e in adjacent[i]),
                    fragment_edges=sum(e["relation"] == "fragment" for _, e in adjacent[i]),
                    extension_edges=sum(e["relation"] == "extension" for _, e in adjacent[i]),
                    anchor_fragment_edges=sum(j < n_anchor and e["relation"] == "fragment" for j, e in adjacent[i]),
                    stronger_competitor=strength)
    return Bank(masks, n_anchor, nodes, overlap, edges)


def matrix(bank, columns):
    return np.array([[node[key] for key in columns] for node in bank.nodes], dtype=float).reshape(-1, len(columns))


def target(bank, truth):
    # One deployable score per physical mask. Average its matched IoU over annotators.
    best = np.stack([iou(bank.masks, masks).max(axis=1, initial=0.) for masks in truth.values()])
    return np.where(best > .5, best, 0.).mean(axis=0)


def fit(banks, targets, stems):
    weights = np.concatenate([np.full(len(banks[s].nodes), 1. / max(len(banks[s].nodes), 1)) for s in stems])
    y = np.concatenate([targets[s] for s in stems])
    if not len(y):
        raise ValueError("Selection training has no cached candidates")
    weights *= len(weights) / weights.sum()
    models = {}
    for name, columns in MODELS.items():
        x = np.concatenate([matrix(banks[s], columns) for s in stems])
        scale = StandardScaler().fit(x, sample_weight=weights)
        reg = Ridge(alpha=RULES["ridge_alpha"]).fit(scale.transform(x), y, sample_weight=weights)
        models[name] = {"columns": columns, "mean": scale.mean_.tolist(), "scale": scale.scale_.tolist(),
                        "coef": reg.coef_.tolist(), "intercept": float(reg.intercept_)}
    return models


def predict(bank, models):
    return {name: np.clip(((matrix(bank, model["columns"]) - model["mean"]) / model["scale"]) @ model["coef"] +
                          model["intercept"], 0., 1.) for name, model in models.items()}


def solve_local(bank, utility, fixed, allowed, groups=(), penalties=(), exact_group=True):
    """Optimize separate connected components of the exact pixel/soft relation graph."""
    active = sorted(set(fixed) | set(allowed))
    if not active:
        return [], 0
    index = {node: k for k, node in enumerate(active)}
    hard = [(i, j) for i, j in combinations(active, 2) if bank.overlap[i, j] > 0.]
    soft = [(i, j, v) for i, j, v in penalties if i in index and j in index and (i not in fixed or j not in fixed)]
    edges = hard + [(i, j) for i, j, _ in soft] + [(group[0], j) for group in groups for j in group[1:]]
    links = [(index[i], index[j]) for i, j in edges]
    ii, jj = zip(*links) if links else ([], [])
    graph = coo_matrix((np.ones(len(links)), (ii, jj)), shape=(len(active), len(active))).tocsr()
    count, labels = connected_components(graph, directed=False)
    selected = []
    for component in range(count):
        items = [active[k] for k in np.flatnonzero(labels == component)]
        if len(items) == 1:
            i = items[0]
            if i in fixed or utility[i] > 0.:
                selected.append(i)
            continue
        local = {node: k for k, node in enumerate(items)}
        pairs = [(i, j, v) for i, j, v in soft if i in local and j in local]
        rows, cols, data, lo, hi = [], [], [], [], []

        def constraint(ids, coefficients, low, high):
            rows.extend([len(lo)] * len(ids))
            cols.extend(ids)
            data.extend(coefficients)
            lo.append(low)
            hi.append(high)

        for i, j in hard:
            if i in local and j in local:
                constraint([local[i], local[j]], [1., 1.], 0., 1.)
        for group in groups:
            if group[0] in local:
                constraint([local[i] for i in group], np.ones(len(group)), 1., 1. if exact_group else np.inf)
        for k, (i, j, _) in enumerate(pairs, len(items)):
            constraint([local[i], local[j], k], [1., 1., -1.], -np.inf, 1.)
        cost = np.r_[-utility[items], [v for _, _, v in pairs]]
        lower = np.r_[[int(i in fixed) for i in items], np.zeros(len(pairs))]
        design = coo_matrix((data, (rows, cols)), shape=(len(lo), len(cost))).tocsc()
        solution = milp(cost, integrality=np.r_[np.ones(len(items)), np.zeros(len(pairs))],
                        bounds=Bounds(lower, np.ones(len(cost))), constraints=LinearConstraint(design, lo, hi),
                        options={"time_limit": RULES["milp_seconds"], "mip_rel_gap": 0.})
        if not solution.success:
            raise RuntimeError(f"Local exact selection failed for {len(items)} candidates: {solution.message}")
        selected.extend(i for i, x in zip(items, solution.x[:len(items)], strict=True) if x > .5)
    return sorted(selected), count


def decide(bank, estimates, pq_price, arm, combo=None, donor_allow=None):
    anchors = set(range(bank.n_anchor))
    if arm == "C0":
        return sorted(anchors), {"active": len(anchors), "components": len(anchors), "hard_edges": 0, "soft_edges": 0,
                                 "eligible_nodes": anchors}
    if arm == "C6":
        if not combo:
            raise ValueError("C6 requires two independently selected arms")
        left, dl = decide(bank, estimates, pq_price, combo[0])
        right, dr = decide(bank, estimates, pq_price, combo[1])
        eligible = dl.pop("eligible_nodes") | dr.pop("eligible_nodes")
        return sorted(set(left) | (set(right) - anchors)), {**{k: dl[k] + dr[k] for k in dl},
            "active": len(eligible), "eligible_nodes": eligible}
    mode = {"C1a": "semantic", "C1b": "relational", "C2a": "visual", "C2b": "semantic", "C3": "semantic",
            "C4a": "relational", "C4b": "semantic", "C5": "relational"}[arm]
    value = estimates[mode].copy() - pq_price
    value[:bank.n_anchor] += 1e-7  # Break exact ties in favor of the frozen anchor.
    eligible, droppable = set(), set()
    swaps = {}
    for i, node in enumerate(bank.nodes):
        supported = node["support_views"] >= 2
        semantic = node["s0_mean"] >= .48 and node["s0_fraction"] >= .38
        if i in anchors:
            weak = node["support_views"] <= 1 and node["s0_mean"] <= .43
            contested = node["fragment_edges"] > 0 or node["stronger_competitor"] >= .40
            if arm == "C1a" and weak and value[i] < -RULES["drop_conservative"]:
                droppable.add(i)
            if arm == "C1b" and (weak or contested) and value[i] < -RULES["drop_relational"]:
                droppable.add(i)
            if arm in ("C4b", "C5") and (weak or (arm == "C5" and contested)) and value[i] < -RULES["drop_joint"]:
                droppable.add(i)
            continue
        if donor_allow is not None and i not in donor_allow:
            continue
        blocked = np.flatnonzero(bank.overlap[i, :bank.n_anchor] > 0.).tolist()
        if arm == "C2a" and not blocked and supported and value[i] > RULES["add_conservative"]:
            eligible.add(i)
            value[i] -= RULES["add_conservative"]
        if arm in ("C2b", "C3") and not blocked and semantic and value[i] > RULES["add_semantic"]:
            eligible.add(i)
            value[i] -= RULES["add_semantic"]
        if arm == "C4a" and len(blocked) == 1 and (supported or semantic):
            j = blocked[0]
            if (bank.overlap[i, j] >= .16 and max(node["inside_anchor"], node["contains_anchor"]) >= .42 and
                    value[i] > value[j] + RULES["swap_margin"]):
                swaps.setdefault(j, []).append(i)
                eligible.add(i)
                value[i] -= RULES["swap_margin"]
        if arm in ("C4b", "C5") and (supported or semantic) and value[i] > RULES["add_joint"]:
            if not blocked or (node["anchor_iou"] >= .10 and node["anchor_touch_count"] <= 3):
                eligible.add(i)
                value[i] -= RULES["add_joint"]
    if arm in ("C4b", "C5"):
        # Only an eligible competitor may displace a protected anchor. A
        # singleton exact group preserves it if no competing donor is chosen.
        for j in anchors - droppable:
            competitors = sorted(i for i in eligible if bank.overlap[i, j] > 0.)
            if competitors:
                swaps[j] = competitors
                value[j] += RULES["swap_margin"]
    fixed = anchors - droppable - swaps.keys()
    groups = [[j, *options] for j, options in sorted(swaps.items())]
    if arm in ("C3", "C5"):
        # For two otherwise admissible competing donors, prefer a fuller mask
        # when the extra area is supported. Cap the deduction across duplicates.
        deductions = np.zeros(len(bank.nodes))
        for edge in bank.edges:
            i, j = edge["i"], edge["j"]
            if i in eligible and j in eligible and edge["hard_conflict"] and edge["relation"] in ("duplicate", "fragment"):
                small, large = sorted((i, j), key=lambda k: bank.nodes[k]["area"])
                if (edge["area_ratio"] <= .78 and value[large] >= .04 and
                        bank.nodes[large]["s0_mean"] >= bank.nodes[small]["s0_mean"] - .08):
                    deductions[small] = .5 * RULES["relation_penalty"]
        value -= deductions
    penalty = [(e["i"], e["j"], RULES["relation_penalty"]) for e in bank.edges
               if arm in ("C3", "C5") and not e["hard_conflict"] and e["relation"] == "fragment"
               and (arm == "C5" or (e["i"] in eligible and e["j"] in eligible))]
    active = anchors | eligible
    soft_edges = sum(i in active and j in active and (i not in fixed or j not in fixed) for i, j, _ in penalty)
    picked, count = solve_local(bank, value, fixed, active, groups, penalty, exact_group=arm == "C4a")
    if not fixed.issubset(picked):
        raise ValueError("Protected F4 anchors changed")
    if arm == "C4a" and len(picked) != bank.n_anchor:
        raise ValueError("Swap-only must preserve the instance count")
    legal([bank.masks[i] for i in picked])
    return picked, {"active": len(active), "components": count, "hard_edges": sum(
        e["hard_conflict"] and e["i"] in active and e["j"] in active for e in bank.edges), "soft_edges": soft_edges,
        "eligible_nodes": active}
