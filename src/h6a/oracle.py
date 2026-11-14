"""Constrained, GT-only counterfactuals over an explicit mask bank."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from pycocotools import mask as cm
from scipy import ndimage as ndi
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from mul2_assets import host_counts, iou, legal, metrics, score
from reconstruct import encode_roi


@dataclass(frozen=True)
class Option:
    slot: int
    op: str
    masks: tuple
    donor: tuple = ()
    mode: str = ""
    eligible: bool = True


def overlaps(left, right):
    a, b = np.asarray(cm.area(left), float), np.asarray(cm.area(right), float)
    j = iou(left, right)
    inter = np.rint(j * (a[:, None] + b[None, :]) / (1 + j))
    return j, inter, inter / np.maximum(a[:, None], 1), inter / np.maximum(b[None, :], 1)


def partition(parent, markers):
    """Assign all S0 seed pixels to the nearest in-seed Mul2 marker pixels."""
    x, y, w, h = np.asarray(cm.toBbox(parent), int)
    roi = (slice(y, y+h), slice(x, x+w))
    region = cm.decode(parent)[roi].astype(bool)
    seeds = np.zeros(region.shape, np.int32)
    for i, mask in enumerate(markers, 1):
        seeds[region & (cm.decode(mask)[roi] > 0)] = i
    nearest = ndi.distance_transform_edt(seeds == 0, return_distances=False, return_indices=True)
    labels = seeds[tuple(nearest)]
    return tuple(encode_roi(region & (labels == i), roi, tuple(parent["size"]))
                 for i in range(1, len(markers) + 1))


def make_options(base, donor, views, allow_split, max_split_neighbors=8, max_split_children=4):
    j, inter, by_base, by_donor = overlaps(base, donor)
    records = [(overlaps(gt, base), iou(gt, donor) > .5, gt) for gt in views.values()]
    h_base = [record[0][0] > .5 for record in records]
    h_donor = [record[1] for record in records]
    options = [Option(i, "keep", (mask,)) for i, mask in enumerate(base)]
    observable = []
    omitted_neighbors = 0

    def add(option, legal_alone, features):
        options.append(option)
        if legal_alone:
            observable.append((len(options) - 1, features))

    donor_area = cm.area(donor)
    for d, mask in enumerate(donor):
        touched = inter[:, d] > 0
        rescue = any(np.any(right[:, d] & ~left.any(axis=1)) for left, right in zip(h_base, h_donor))
        common = {"donor_area": int(donor_area[d]), "max_anchor_iou": float(j[:, d].max(initial=0)),
                  "max_donor_containment": float(by_donor[:, d].max(initial=0)),
                  "touched_anchors": int(touched.sum())}
        add(Option(len(base) + d, "rescue", (mask,), (d,), "whole", rescue), not touched.any(), common)
        for b in np.flatnonzero((j[:, d] >= .10) | (by_base[:, d] >= .5) | (by_donor[:, d] >= .5)):
            shared = any(np.any(left[:, b] & right[:, d]) for left, right in zip(h_base, h_donor))
            add(Option(int(b), "replace", (mask,), (d,), "whole", shared), touched.sum() == 1,
                common | {"pair_iou": float(j[b, d]), "donor_to_anchor_area": float(donor_area[d] / cm.area(base[b])),
                          "pair_min_containment": float(min(by_base[b, d], by_donor[b, d]))})

    if allow_split:
        for b, parent in enumerate(base):
            near = sorted(np.flatnonzero((by_donor[b] >= .10) & (inter[b] >= 10)).tolist(),
                          key=lambda d: (-by_donor[b, d], -inter[b, d], d))
            omitted_neighbors += max(0, len(near) - max_split_neighbors)
            near = near[:max_split_neighbors]
            for n in range(2, min(max_split_children, len(near)) + 1):
                for ds in combinations(near, n):
                    masks = tuple(donor[d] for d in ds)
                    features = {"children": n, "seed_coverage": float(by_base[b, list(ds)].sum()),
                                "min_child_inside_seed": float(by_donor[b, list(ds)].min()),
                                "child_area_ratio": float(min(donor_area[list(ds)]) / max(donor_area[list(ds)]))}
                    partitioned = partition(parent, masks)
                    for mode, proposed in (("whole_donor", masks), ("partition_seed", partitioned)):
                        eligible = False
                        for (_, _, gt_fraction, seed_fraction), _, gt in records:
                            relevant = (gt_fraction[:, b] >= .20) & (seed_fraction[:, b] >= .02)
                            hit = (iou(gt, list(proposed)) > .5) & relevant[:, None]
                            if np.all(hit.sum(axis=0) == 1) and np.all(hit.sum(axis=1) <= 1):
                                eligible = True
                                break
                        other = np.delete(inter[:, list(ds)], b, axis=0)
                        add(Option(b, "split", proposed, ds, mode, eligible),
                            mode == "partition_seed" or not other.any(), features)
    return options, observable, omitted_neighbors


def candidate_terms(options, views):
    # Each selected mask contributes one instance in every annotator view.
    stats = np.zeros((len(options), 4), float)
    hits = []
    for i, option in enumerate(options):
        keys = []
        ambiguous = False
        for view, gt in views.items():
            matrix = iou(gt, list(option.masks))
            h = matrix > .5
            ambiguous |= bool(np.any(h.sum(axis=0) > 1) or np.any(h.sum(axis=1) > 1))
            keys.extend((view, int(k)) for k in np.flatnonzero(h.any(axis=1)))
            term = host_counts(matrix)
            term[3] -= len(gt)
            stats[i] += term
        hits.append((frozenset(keys), ambiguous))
    return stats, hits


class OracleProblem:
    def __init__(self, base, options, stats, hits, views, allowed):
        self.options = options
        self.const = np.array([0., 0., 0., sum(map(len, views.values()))])
        self.views = views
        self.base = tuple(range(len(base)))
        self.ids = [i for i, v in enumerate(options)
                    if v.op == "keep" or (v.op in allowed and v.eligible and not hits[i][1])]
        self.stats = stats[self.ids]
        self.choices = [options[i] for i in self.ids]
        slot = defaultdict(list)
        used = defaultdict(list)
        for i, option in enumerate(self.choices):
            slot[option.slot].append(i)
            for d in option.donor:
                used[d].append(i)
        constraints = [(xs, int(k < len(base)), 1) for k, xs in slot.items()]
        constraints += [(xs, 0, 1) for xs in used.values() if len(xs) > 1]
        for i, a in enumerate(self.choices):
            for k in range(i):
                b = self.choices[k]
                if a.slot == b.slot or (set(a.donor) & set(b.donor)):
                    continue
                if hits[self.ids[i]][0] & hits[self.ids[k]][0] or iou(list(a.masks), list(b.masks)).any():
                    constraints.append(([i, k], 0, 1))
        self.constraint = None
        if constraints:
            cols = [j for xs, _, _ in constraints for j in xs]
            rows = [i for i, (xs, _, _) in enumerate(constraints) for _ in xs]
            mat = coo_matrix((np.ones(len(cols)), (rows, cols)),
                             shape=(len(constraints), len(self.ids))).tocsc()
            self.constraint = LinearConstraint(mat, [a for _, a, _ in constraints],
                                               [b for _, _, b in constraints])

    def select(self, ratio):
        if all(v.op == "keep" for v in self.choices):
            return self.base, self.const + self.stats.sum(axis=0)
        v = self.stats
        denom = v[:, 1] + .5 * (v[:, 2] + v[:, 3])
        solved = milp(ratio * denom - v[:, 0], integrality=np.ones(len(v)),
                      bounds=Bounds(0, 1), constraints=self.constraint,
                      options={"mip_rel_gap": 0.0})
        if solved.status != 0 or solved.x is None:
            raise RuntimeError(f"Oracle optimization failed: {solved.message}")
        chosen = np.flatnonzero(solved.x > .5)
        picks = tuple(self.ids[i] for i in chosen)
        return picks, self.const + v[chosen].sum(axis=0)

    def validate(self, picks, terms):
        masks = [r for i in picks for r in self.options[i].masks]
        legal(masks)
        for gt in self.views.values():
            h = iou(gt, masks) > .5
            if (h.sum(axis=0) > 1).any() or (h.sum(axis=1) > 1).any():
                raise ValueError("Oracle one-to-one assignment failed")
        if not np.allclose(score(self.views, masks), terms, rtol=0, atol=1e-10):
            raise ValueError("Oracle additive and official PQ terms disagree")
        return masks


def pooled_oracle(problems, baseline_ratio):
    """Dinkelbach iterations solve the pooled PQ ratio, not mean photo PQ."""
    q = baseline_ratio
    for iteration in range(1, 41):
        solution = {stem: problem.select(q) for stem, problem in problems.items()}
        total = sum((terms for _, terms in solution.values()), np.zeros(4))
        next_q = metrics(total)["pq"]
        if abs(q - next_q) < 1e-10:
            for stem, (picks, terms) in solution.items():
                problems[stem].validate(picks, terms)
            return solution, iteration
        q = next_q
    raise RuntimeError("Oracle ratio optimization did not converge")


def observable_effects(options, observable, stats, ratio):
    for i, features in observable:
        option = options[i]
        prior = stats[option.slot] if option.op != "rescue" else np.zeros(4)
        ds, tp, fp, fn = stats[i] - prior
        margin = ds - ratio * (tp + .5 * (fp + fn))
        yield {"option": i, "operation": option.op, "mode": option.mode, "slot": option.slot,
               "donor_indices": ",".join(map(str, option.donor)), **features, "delta_tp": int(tp),
               "delta_fp": int(fp), "delta_fn": int(fn), "delta_iou_sum": float(ds),
               "gt_pq_margin": float(margin), "positive": bool(margin > 1e-10),
               "oracle_candidate_eligible": option.eligible}
