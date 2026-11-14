"""Semantic fusion and disjoint instance reconstruction for Solar S1.0."""

from dataclasses import dataclass, field

import numpy as np
from pycocotools import mask as coco
from scipy import ndimage as ndi


S8 = np.ones((3, 3), bool)
KEEP_039 = (.35, .99, 200, .85)


def check_probability(array, shape):
    if array.dtype != np.float32 or array.shape != shape:
        raise ValueError(f"Expected float32 probability {shape}, got {array.dtype} {array.shape}")
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError("Probability contains nonfinite or out-of-range values")
    return array


def feather_weights(shape, crop, width):
    y0, y1, x0, x1 = crop
    if not (0 <= y0 < y1 <= shape[0] and 0 <= x0 < x1 <= shape[1] and width > 0):
        raise ValueError("Invalid Blend crop or feather width")
    y, x = np.ogrid[:y1 - y0, :x1 - x0]
    distance = np.minimum(np.minimum(y, y1 - y0 - 1 - y), np.minimum(x, x1 - x0 - 1 - x))
    return (.5 - .5 * np.cos(np.pi * np.minimum(distance / width, 1))).astype(np.float32)


def fuse(p039, pb, crop, feather, alpha):
    y0, y1, x0, x1 = crop
    if not 0 <= alpha <= 1 or feather.shape != pb.shape:
        raise ValueError("Invalid blend weight or feather shape")
    out = p039.copy()
    center = out[y0:y1, x0:x1]
    weight = np.float32(alpha) * feather
    valid = weight > 0
    center[valid] = (1 - weight[valid]) * center[valid] + weight[valid] * pb[valid]
    return out


def labels_at(prob, threshold):
    labels, count = ndi.label(prob > threshold, structure=S8)
    foreground = labels > 0
    ids, values = labels[foreground], prob[foreground]
    areas = np.bincount(ids, minlength=count + 1)
    sums = np.bincount(ids, weights=values, minlength=count + 1)
    peaks = np.zeros(count + 1, dtype=prob.dtype)
    np.maximum.at(peaks, ids, values)
    return labels, areas, sums / np.maximum(areas, 1), peaks


def encode_roi(mask, box, shape):
    """Encode a bounded region directly in native COCO column-major coordinates."""
    height, width = shape
    y, x = box
    edges = np.diff(np.pad(mask, ((1, 1), (0, 0))).T, axis=1)
    cols, rows = np.nonzero(edges)
    runs = ((cols + x.start) * height + rows + y.start).reshape(-1, 2)
    if not len(runs):
        raise ValueError("Cannot encode an empty candidate")
    join = runs[:-1, 1] == runs[1:, 0]
    starts = runs[np.r_[True, ~join], 0]
    ends = runs[np.r_[~join, True], 1]
    boundaries = np.r_[0, np.column_stack((starts, ends)).ravel()]
    if ends[-1] != height * width:
        boundaries = np.r_[boundaries, height * width]
    counts = np.diff(boundaries).tolist()
    return coco.frPyObjects({"size": list(shape), "counts": counts}, height, width)


def base_cc(prob, keep, quantized=False):
    low, high, min_area, min_mean = keep
    scale = 255 if quantized else 1
    low, high = (round(low * 255), round(high * 255)) if quantized else (low, high)
    labels, areas, means, peaks = labels_at(prob, low)
    boxes = ndi.find_objects(labels)
    ids = np.flatnonzero((areas >= min_area) & (means / scale >= min_mean) & (peaks > high))
    return [encode_roi(labels[boxes[i - 1]] == i, boxes[i - 1], prob.shape) for i in ids]


@dataclass
class Candidate:
    key: str
    level: int
    threshold: float
    box: tuple
    mask: np.ndarray
    area: int
    mean: float
    peak: float
    q90: float
    high_fraction: float
    length: float
    width: float
    elongation: float
    contrast: float
    agreement: float | None
    agreement_coverage: float
    rle: dict
    parent: int | None
    children: list = field(default_factory=list)
    stability: float = 0.
    basic: float = 0.
    quality: float = 0.


def expand(box, shape, padding):
    return tuple(slice(max(0, axis.start - padding), min(size, axis.stop + padding))
                 for axis, size in zip(box, shape))


def relative(box, outer):
    return tuple(slice(axis.start - base.start, axis.stop - base.start) for axis, base in zip(box, outer))


def candidates(prob, settings, p039=None, pb=None, crop=None):
    all_nodes, previous_labels, previous_nodes = [], None, {}
    thresholds = settings["thresholds"]
    if not thresholds or sorted(set(thresholds)) != list(thresholds):
        raise ValueError("Candidate thresholds must be strictly increasing")
    for level, threshold in enumerate(thresholds):
        labels, areas, means, peaks = labels_at(prob, threshold)
        boxes = ndi.find_objects(labels)
        current = {}
        for label in np.flatnonzero(areas >= settings["candidate_min_area"]):
            box = boxes[label - 1]
            mask = labels[box] == label
            area = int(areas[label])
            ys, xs = np.nonzero(mask)
            abs_y, abs_x = ys + box[0].start, xs + box[1].start
            parent = None if previous_labels is None else previous_nodes[int(previous_labels[abs_y[0], abs_x[0]])]
            var_y, var_x = np.var(ys), np.var(xs)
            covariance = np.mean((ys - ys.mean()) * (xs - xs.mean()))
            span = np.hypot(var_y - var_x, 2 * covariance)
            length = float(np.sqrt(12 * (var_y + var_x + span) / 2 + 1))
            width = float(np.sqrt(12 * max(0, (var_y + var_x - span) / 2) + 1))
            outer = expand(box, prob.shape, 3)
            local = np.zeros(prob[outer].shape, bool)
            local[relative(box, outer)] = mask
            ring = ndi.binary_dilation(local, structure=S8, iterations=3) & ~local
            values = prob[box][mask]
            mean = float(means[label])
            contrast = mean - float(prob[outer][ring].mean()) if ring.any() else 0.
            agreement, coverage = None, 0.
            if pb is not None:
                y0, y1, x0, x1 = crop
                valid = (abs_y >= y0) & (abs_y < y1) & (abs_x >= x0) & (abs_x < x1)
                coverage = float(valid.mean())
                if valid.any():
                    vy, vx = abs_y[valid], abs_x[valid]
                    agreement = float(1 - np.abs(p039[vy, vx] - pb[vy-y0, vx-x0]).mean())
            node = Candidate(f"{level}:{label}", level, threshold, box, mask, area, mean, float(peaks[label]),
                             float(np.quantile(values, .9)), float((values >= .75).mean()), length, width,
                             length / width, contrast, agreement, coverage, encode_roi(mask, box, prob.shape), parent)
            index = len(all_nodes)
            if parent is not None:
                all_nodes[parent].children.append(index)
            current[int(label)] = index
            all_nodes.append(node)
        previous_labels, previous_nodes = labels, current

    for node in all_nodes:
        ratios = [all_nodes[i].area / node.area for i in node.children]
        if node.parent is not None:
            ratios.append(node.area / all_nodes[node.parent].area)
        node.stability = max(ratios, default=0.)
        size = float(np.clip(np.log2(node.area / 200) / 3, -1, 1))
        aspect = float(np.clip(np.log2(node.elongation) / 3, 0, 1))
        # Fixed training-free utility. The per-instance cost prevents automatic reward for splitting.
        node.basic = .55 * node.mean + .30 * node.q90 + .15 * node.peak + .08 * size - settings["instance_cost"]
        agreement = 0. if node.agreement is None else (node.agreement - .80) * node.agreement_coverage
        node.quality = (node.basic + .08 * (node.stability - .5) + .08 * (node.contrast - .30)
                        + .04 * (node.high_fraction - .45) + .05 * (aspect - .5) + .08 * agreement)
    return all_nodes


def greedy(nodes, shape):
    occupied = np.zeros(shape, bool)
    selected = []
    for node in sorted(nodes, key=lambda n: (-n.basic, -n.area, n.key)):
        if node.basic <= 0:
            break
        region = occupied[node.box]
        if not np.any(region & node.mask):
            selected.append(node)
            region |= node.mask
    return selected


def tree_select(nodes, score, agreement_weight=1.):
    """Maximum utility antichain over sampled level-set trees, with an empty option."""
    solutions = [None] * len(nodes)
    for i in range(len(nodes) - 1, -1, -1):
        node = nodes[i]
        total = sum(solutions[j][0] for j in node.children)
        own = getattr(node, score)
        if score == "quality" and node.agreement is not None:
            own -= (1 - agreement_weight) * .08 * (node.agreement - .80) * node.agreement_coverage
        if own > 0 and own >= total:
            solutions[i] = own, [node]
        else:
            solutions[i] = total, [n for j in node.children for n in solutions[j][1]]
    return [n for i, node in enumerate(nodes) if node.parent is None for n in solutions[i][1]]


def close_selected(prob, selected, radius, selective, settings):
    occupied = np.zeros(prob.shape, bool)
    for node in selected:
        occupied[node.box] |= node.mask
    result = []
    for node in sorted(selected, key=lambda n: (-n.quality, -n.area, n.key)):
        box = expand(node.box, prob.shape, radius + 1)
        original = np.zeros(prob[box].shape, bool)
        original[relative(node.box, box)] = node.mask
        region = occupied[box]
        other = region & ~original
        candidate = ndi.binary_closing(original, structure=np.ones((2 * radius + 1,) * 2, bool))
        addition = candidate & ~original & ~other
        if selective:
            addition &= (prob[box] >= settings["repair_min_probability"])
            addition &= ~ndi.binary_dilation(other, structure=S8)
        # Only accept small additions that stay attached to their original instance.
        limit = max(1, int(node.area * settings["repair_max_fraction"]))
        size = int(addition.sum())
        if 0 < size <= limit:
            grown = ndi.binary_propagation(original, structure=S8, mask=original | addition)
            region |= grown
            result.append((node, encode_roi(grown, box, prob.shape), int(grown.sum()) - node.area))
        else:
            result.append((node, node.rle, 0))
    return result


def structural(prob, settings, p039=None, pb=None, crop=None):
    nodes = candidates(prob, settings, p039, pb, crop)
    basic_tree = tree_select(nodes, "basic")
    selected = tree_select(nodes, "quality")
    plain = lambda choices: [(node, node.rle, 0) for node in choices]
    stages = {
        "multithreshold": plain(greedy(nodes, prob.shape)),
        "tree": plain(basic_tree),
        "quality": plain(selected),
        "small_close": close_selected(prob, selected, 1, False, settings),
        "selective_close": close_selected(prob, selected, 2, True, settings),
    }
    if pb is not None:
        stages["quality_no_agreement"] = plain(tree_select(nodes, "quality", agreement_weight=0.))
    return nodes, stages
