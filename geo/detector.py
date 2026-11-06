#!/usr/bin/env python3
"""Score detector circles on the same pixel grid, cells and metrics as the networks in tanks.py.

`grid` writes the pixel grid as a GeoTIFF for geo/detect.py. `score` reads its GeoPackages.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tanks  # noqa: E402


def partition_at(target, transform, points):
    """The partition of each (x, y) map point: train, val, test, or empty outside all cells."""
    from rasterio.transform import rowcol

    if not points:
        return []
    rows, cols = rowcol(transform, [p.x for p in points], [p.y for p in points])
    h, w = target["cell"].shape
    names = []
    for r, c in zip(rows, cols):
        inside = 0 <= r < h and 0 <= c < w and target["cell"][r, c] > 0
        names.append(str(target["splits"][target["cell"][r, c] - 1]) if inside else "")
    return names


def circles_mask(circles, shape, transform):
    """The map of the kept circles. rasterize refuses an empty list, and no kept circle is a valid result."""
    from rasterio.features import rasterize

    if not circles:
        return np.zeros(shape, bool)
    return rasterize(circles, shape, transform=transform, fill=0, default_value=1, dtype="uint8") > 0


def match(preds, scores, truths, pred_part, truth_part, conf, part):
    """Greedy one-to-one match by score at IoU >= 0.5, for one partition.
    Returns AP50 over all scores, and the object counts at score >= conf."""
    used = np.zeros(len(truths), bool)
    hits, misses_at, tp_conf, fp_conf = [], [], 0, 0
    total = truth_part.count(part)
    for i in np.argsort(-scores):
        best, at = 0.0, -1
        for j, t in enumerate(truths):
            if not used[j] and preds[i].intersects(t):
                iou = preds[i].intersection(t).area / preds[i].union(t).area
                if iou > best:
                    best, at = iou, j
        if best >= 0.5:
            used[at] = True
            if truth_part[at] != part:
                continue
            hit = 1
        elif pred_part[i] == part:
            hit = 0
        else:
            continue
        hits.append(hit)
        if scores[i] >= conf:
            tp_conf += hit
            fp_conf += 1 - hit
    tp = np.cumsum(hits) if hits else np.zeros(0)
    recall, precision = tp / max(total, 1), tp / np.arange(1, len(tp) + 1)
    r = np.concatenate([[0], recall, [1]])
    p = np.maximum.accumulate(np.concatenate([[1], precision, [0]])[::-1])[::-1]
    pr, rc = tp_conf / max(tp_conf + fp_conf, 1), tp_conf / max(total, 1)
    return {"ap50": round(float(np.sum((r[1:] - r[:-1]) * p[1:])), 4), "f1": round(2 * pr * rc / max(pr + rc, 1e-9), 4),
            "tp": tp_conf, "fp": fp_conf, "fn": total - tp_conf}


def grid(a):
    import rasterio

    data = Path(a.data)
    transform, width, height, _ = tanks.target_grid(data)
    image = tanks.load_target()["image"]
    assert image.shape[1:] == (height, width), (image.shape, height, width)
    with rasterio.open(a.out, "w", driver="GTiff", width=width, height=height, count=3, dtype="uint8", crs="EPSG:4326", transform=transform) as d:
        d.write(image)
    print(a.out, width, height)


def score(a):
    import geopandas as gpd

    data = Path(a.data)
    transform, width, height, _ = tanks.target_grid(data)
    target = tanks.load_target()
    truths = gpd.read_file(data / "Tanks.gdb", layer="Tanks").to_crs(4326).geometry.to_list()
    # shortcut: this checks shape and tank count, not file hashes; upgrade if a second dataset is ever used
    assert target["cell"].shape == (height, width) and target["inst"].max() == len(truths), "run prepare on this --data"
    truth_part = partition_at(target, transform, [t.centroid for t in truths])
    results = {}
    for path in a.pred:
        gdf = gpd.read_file(path).to_crs(4326)
        preds, scores = gdf.geometry.to_list(), gdf["score"].to_numpy(float)
        pred_part = partition_at(target, transform, [p.centroid for p in preds])
        kept = [p for p, s in zip(preds, scores) if s >= a.conf]
        mask = circles_mask(kept, target["cell"].shape, transform)
        row = {"detections": len(preds), "kept": len(kept), "conf": a.conf, "masks": tanks.scored(mask, target),
               "circles": {part: match(preds, scores, truths, pred_part, truth_part, a.conf, part) for part in ("train", "val", "test")}}
        results[Path(path).stem] = row
        print(Path(path).stem, json.dumps(row), flush=True)
    (tanks.OUT / "detector.json").write_text(json.dumps(results, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("grid")
    p.add_argument("--data", required=True, help="folder with Tanks.tif and 20150520_075001.tif")
    p.add_argument("--out", required=True)
    p.set_defaults(func=grid)
    p = sub.add_parser("score")
    p.add_argument("--data", required=True, help="folder with Tanks.tif, 20150520_075001.tif and Tanks.gdb")
    p.add_argument("--conf", type=float, default=0.70, help="fixed from the 2015 image; do not tune it on these cells")
    p.add_argument("pred", nargs="+")
    p.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
