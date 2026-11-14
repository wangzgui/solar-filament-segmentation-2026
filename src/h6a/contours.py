"""Whole-mask replacements with one incumbent identity and no pixel conflict."""

from dataclasses import dataclass

import numpy as np
from pycocotools import mask as cm
from scipy import ndimage as ndi

from mul2_assets import iou, legal, score, metrics
from oracle import overlaps
from reconstruct import S8


FEATURES = ("pair_iou", "minimum_coverage", "signed_log_area", "abs_log_area", "other_s0_coverage_of_mul2",
            "s0_foreground_support", "disputed_probability_margin",
            "local_dark_contrast_delta", "boundary_gradient_delta")
ARMS = {"mul2_exact": 0, "pair_geometry": 5, "pair_probability": 7, "pair_image": 9}


@dataclass
class Bank:
    base: list
    donor: list
    pairs: list
    features: np.ndarray
    blocked_donors: int

    def materialize(self, decisions):
        masks = list(self.base)
        for k in np.flatnonzero(decisions):
            a, b = self.pairs[k]
            masks[a] = self.donor[b]
        legal(masks)
        return masks


def union_crop(a, b, shape, padding=6):
    u = cm.merge([a, b])
    x, y, w, h = map(int, cm.toBbox(u))
    return (slice(max(0, y-padding), min(shape[0], y+h+padding)),
            slice(max(0, x-padding), min(shape[1], x+w+padding)))


def appearance(a, b, p, gray):
    union = a | b
    signed_pixels = b.astype(np.int8) - a.astype(np.int8)
    probability_margin = float(np.sum(signed_pixels * (2 * p - 1)) / union.sum())
    scale = max(1., float(np.percentile(gray, 95) - np.percentile(gray, 5)))
    gradient = ndi.gaussian_gradient_magnitude(gray.astype(np.float32), sigma=1.25) / scale

    def image_evidence(mask):
        ring = ndi.binary_dilation(mask, structure=S8, iterations=4) & ~union
        inside = mask & ~ndi.binary_erosion(mask, structure=S8)
        surround = float(gray[ring].mean()) if ring.any() else float(np.median(gray))
        return (surround - float(gray[mask].mean())) / scale, float(gradient[inside].mean())

    delta = np.subtract(image_evidence(b), image_evidence(a))
    return float(p[b].mean()), probability_margin, float(delta[0]), float(delta[1])


def build_bank(base, donor, probability_u8, image):
    legal(base)
    legal(donor)
    if probability_u8.shape != image.shape or probability_u8.dtype != np.uint8 or image.dtype != np.uint8:
        raise ValueError("Contour inputs must share the native uint8 image geometry")
    j, pixels, cov_a, cov_b = overlaps(base, donor)
    blocked = int((np.count_nonzero(pixels, axis=0) > 1).sum())
    pairs, values = [], []
    for ai, row in enumerate(j):
        best = row.max(initial=0.)
        if best <= 0. or best == 1.:
            continue
        peers = np.flatnonzero(row == best)
        if len(peers) != 1:
            continue
        si = int(peers[0])
        # A selected S0 mask must fit with every other retained Mul2 instance.
        if np.count_nonzero(pixels[:, si]) != 1:
            continue
        a, b = base[ai], donor[si]
        roi = union_crop(a, b, image.shape)
        a_mask = cm.decode(a)[roi].astype(bool)
        b_mask = cm.decode(b)[roi].astype(bool)
        signed_area = float(np.log(b_mask.sum() / a_mask.sum()))
        other_coverage = float(cov_a[ai].sum() - cov_a[ai, si])
        extras = appearance(a_mask, b_mask, probability_u8[roi].astype(np.float32) / 255., image[roi])
        values.append((float(best), float(min(cov_a[ai, si], cov_b[ai, si])),
                       signed_area, abs(signed_area), other_coverage, *extras))
        pairs.append((ai, si))
    return Bank(base, donor, pairs, np.asarray(values, float).reshape(-1, len(FEATURES)), blocked)


def match_terms(views, masks):
    """Additive official numerator and denominator terms for disjoint predictions."""
    terms = np.zeros((len(masks), 2))
    for truth in views.values():
        values = iou(truth, masks)
        hit = values > .5
        terms[:, 0] += np.where(hit, values, 0.).sum(axis=0)
        terms[:, 1] += .5 * np.maximum(1, hit.sum(axis=0))
    return terms


def label_bank(bank, truth):
    old = match_terms(truth, bank.base)
    new = match_terms(truth, [bank.donor[b] for _, b in bank.pairs])
    delta = new - old[[a for a, _ in bank.pairs]]
    terms = old.sum(axis=0) + [0., .5 * sum(map(len, truth.values()))]
    actual = score(truth, bank.base)
    if not np.allclose(terms, [actual[0], actual[1] + .5 * (actual[2] + actual[3])], atol=1e-10, rtol=0):
        raise ValueError("Official terms and baseline metrics disagree")
    all_swaps = score(truth, bank.materialize(np.ones(len(bank.pairs), bool)))
    if not np.allclose(terms + delta.sum(axis=0),
                       [all_swaps[0], all_swaps[1] + .5 * (all_swaps[2] + all_swaps[3])],
                       atol=1e-10, rtol=0):
        raise ValueError("Official terms and legal replacement metrics disagree")
    return delta, terms, actual


def fit(rows, num_features, penalty):
    available = [r for r in rows if len(r["bank"].pairs)]
    if not available:
        raise ValueError("No eligible replacement pairs in a fitting fold")
    a, b = np.sum([r["terms"] for r in rows], axis=0)
    q = a / b if b else 0.
    x = np.vstack([r["bank"].features[:, :num_features] for r in available])
    y = np.concatenate([(r["delta"][:, 0] - q * r["delta"][:, 1]) / len(r["truth"]) for r in available])
    weights = np.concatenate([np.full(len(r["delta"]), 1 / len(r["delta"])) for r in available])
    weights /= weights.sum()
    mean = weights @ x
    std = np.sqrt(weights @ (x - mean)**2).clip(.025)
    z = (x - mean) / std
    center = float(weights @ y)
    coef = np.linalg.solve(z.T @ (weights[:, None] * z) + penalty * np.eye(num_features),
                           z.T @ (weights * (y - center)))
    return {"features": list(FEATURES[:num_features]), "mean": mean.tolist(), "std": std.tolist(),
            "center": center, "coef": coef.tolist(), "fit_baseline_pq": float(q),
            "fit_photos_with_pairs": len(available), "fit_pairs": len(x)}


def estimates(bank, model):
    if model is None:
        return np.zeros(len(bank.pairs), float)
    x = bank.features[:, :len(model["coef"])]
    return model["center"] + ((x - model["mean"]) / model["std"]) @ model["coef"]


def replace_oracle(rows):
    if not rows:
        return {"pq": 0., "delta_pq": 0., "swaps": 0}
    base = np.sum([r["terms"] for r in rows], axis=0)
    q0 = base[0] / base[1] if base[1] else 0.
    q = q0
    for _ in range(40):
        choices = [r["delta"][:, 0] - q * r["delta"][:, 1] > 1e-12 for r in rows]
        total = base + sum((r["delta"][i].sum(axis=0) for r, i in zip(rows, choices, strict=True)), np.zeros(2))
        updated = total[0] / total[1] if total[1] else 0.
        if abs(q - updated) <= 1e-10:
            values = sum((score(r["truth"], r["bank"].materialize(i))
                          for r, i in zip(rows, choices, strict=True)), np.zeros(4))
            official = metrics(values)["pq"]
            if abs(official - updated) > 1e-10:
                raise ValueError("Replacement Oracle differs from official scorer")
            return {"pq": float(official), "delta_pq": float(official - q0),
                    "swaps": sum(int(i.sum()) for i in choices)}
        q = updated
    raise RuntimeError("Replacement Oracle did not converge")
