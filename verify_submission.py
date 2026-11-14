"""Validate the archived competition submission without accessing annotations."""
import argparse
from collections import defaultdict
import csv
import hashlib
from pathlib import Path

import numpy as np
from pycocotools import mask


SHA256 = "0d170beedd870d5375cb83c3c83d1bc57a3dd852fd715be5b5bec9292a6e5dee"


def verify(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"Unexpected H6a SHA-256 {digest}")
    groups = defaultdict(list)
    with path.open(newline="", encoding="ascii") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("Submission columns differ")
        for row in reader:
            stem, index = row["filament_id"].rsplit("_", 1)
            if int(index) != len(groups[stem]) + 1:
                raise ValueError(f"Non-sequential instance ID for {stem}")
            groups[stem].append({"size": [2048, 2048], "counts": row["segmentation_rle"].encode("ascii")})
    size = sum(len(group) for group in groups.values())
    if size != 1268:
        raise ValueError(f"Unexpected instance count {size}")
    for stem, group in groups.items():
        areas = np.asarray(mask.area(group))
        if (areas == 0).any() or int(mask.area(mask.merge(group))) != areas.sum():
            raise ValueError(f"Empty or overlapping instance in {stem}")
    print(f"Verified SHA-256 {digest}: {size} disjoint instances on {len(groups)} nonempty photos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=Path(__file__).parent / "results/submission.csv")
    verify(parser.parse_args().csv)
