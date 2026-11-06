"""Tiled storage-tank detection on a GeoTIFF with a DOTA-pretrained LibreYOLO OBB model.

Writes <out>.npz (pixel-space circles: cx, cy, d, score) and <out>.gpkg (WGS84 circle polygons).
"""

import argparse
import time

import geopandas as gpd
import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.windows import Window
from shapely.geometry import Point
from torchvision.ops import nms

from libreyolo import LibreYOLO


def tiles(width, height, size, stride):
    xs = list(range(0, max(width - size, 0) + 1, stride))
    ys = list(range(0, max(height - size, 0) + 1, stride))
    if xs[-1] + size < width:
        xs.append(width - size)
    if ys[-1] + size < height:
        ys.append(height - size)
    return [(x, y) for y in ys for x in xs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raster", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--scale", type=float, default=1.0, help="resample factor; 0.5 = half resolution")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--overlap", type=int, default=256, help="tile overlap in model-input pixels")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--nms", type=float, default=0.3)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    t0 = time.time()
    model = LibreYOLO(a.model, device="cuda")
    tank = [k for k, v in model.names.items() if "storage" in str(v).lower() and "tank" in str(v).lower()]
    assert len(tank) == 1, model.names
    tank = tank[0]

    src = rasterio.open(a.raster)
    size = round(a.imgsz / a.scale)  # tile size in source pixels
    stride = size - round(a.overlap / a.scale)
    windows = tiles(src.width, src.height, size, stride)
    edge = 3 / a.scale  # a box this close to an inner tile border is a cut-off tank

    dets = []  # x1, y1, x2, y2, score in source pixels
    n_skipped = 0
    for i in range(0, len(windows), a.batch):
        chunk, imgs = [], []
        for x, y in windows[i : i + a.batch]:
            w = Window(x, y, size, size)
            if not src.read_masks(1, window=w, boundless=True).any():
                n_skipped += 1
                continue
            arr = src.read([1, 2, 3], window=w, boundless=True).transpose(1, 2, 0)
            im = Image.fromarray(np.ascontiguousarray(arr))
            if a.scale != 1.0:
                im = im.resize((a.imgsz, a.imgsz), Image.Resampling.BILINEAR)
            chunk.append((x, y))
            imgs.append(im)
        if not imgs:
            continue
        results = model(imgs, conf=a.conf, classes=[tank], batch=len(imgs), max_det=1000)
        for (x, y), r in zip(chunk, results):
            if r.obb is None or len(r.obb.data) == 0:
                continue
            c = r.obb.xyxyxyxy.cpu().numpy() / a.scale  # (n, 4, 2) tile pixels
            conf = r.obb.conf.cpu().numpy()
            x1, y1 = c[..., 0].min(1), c[..., 1].min(1)
            x2, y2 = c[..., 0].max(1), c[..., 1].max(1)
            cut = (
                ((x1 < edge) & (x > 0))
                | ((y1 < edge) & (y > 0))
                | ((x2 > size - edge) & (x + size < src.width))
                | ((y2 > size - edge) & (y + size < src.height))
            )
            for j in np.flatnonzero(~cut):
                dets.append((x1[j] + x, y1[j] + y, x2[j] + x, y2[j] + y, conf[j]))

    d = np.array(dets, dtype=np.float64).reshape(-1, 5)
    keep = nms(torch.from_numpy(d[:, :4]), torch.from_numpy(d[:, 4]), a.nms).numpy()
    d = d[keep]
    cx, cy = (d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2
    diam = ((d[:, 2] - d[:, 0]) + (d[:, 3] - d[:, 1])) / 2
    np.savez(a.out + ".npz", cx=cx, cy=cy, d=diam, score=d[:, 4])

    T = src.transform
    geoms = []
    for x, y, r in zip(cx, cy, diam / 2):
        gx, gy = T * (x, y)
        geoms.append(Point(gx, gy).buffer(r * T.a, 32))  # square pixels in degrees
    gdf = gpd.GeoDataFrame(
        {"score": d[:, 4], "diameter_px": diam, "diameter_m": diam * abs(T.a) * 111320 * np.sqrt(np.cos(np.radians(T.f)))},
        geometry=geoms,
        crs=src.crs,
    )
    gdf.to_file(a.out + ".gpkg", driver="GPKG")
    print(
        f"{a.model} scale={a.scale}: {len(windows)} tiles ({n_skipped} empty), "
        f"{len(d)} tanks at conf>={a.conf}, {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
