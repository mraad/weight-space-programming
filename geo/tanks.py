#!/usr/bin/env python3
"""Repair a ResNet18 U-Net for tanks in Tanks.tif with LLM weight edits, and compare it with plain fixes.

The network trains on the 2015 image and then meets Tanks.tif. Commands run on gc1, except `wsp.py ask`. Test cells stay
sealed until `final`. Prompts hold only numbers from train cells. See geo/README.md.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from scipy import ndimage as ni

GEO = Path(__file__).resolve().parent
sys.path.insert(0, str(GEO.parent))
import wsp  # noqa: E402

sys.path.insert(0, str(GEO))
from tiny import TinyUNet  # noqa: E402

OUT = GEO.parent / "results" / "geo"
SIZE = 256
MIN_AREA = 50  # pixels, under half of the smallest tank (9.6 m across); fixed before any result
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # the encoder's own values, as the existing R34 model uses
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
HEAD = "segmentation_head.0"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --- data ----------------------------------------------------------------------------------------

def prepare(a):
    """Put Tanks.tif on the pixel grid of the 2015 chips, so scale is not the fault, and save both data sets."""
    import geopandas as gpd
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_bounds

    data = Path(a.data)
    OUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(data / "20150520_075001.tif") as d:
        res = d.res[0]
        assert d.crs.to_epsg() == 4326
    with rasterio.open(data / "Tanks.tif") as d:
        left, bottom, right, top = transform_bounds(d.crs, "EPSG:4326", *d.bounds)
    width, height = int((right - left) / res), int((top - bottom) / res)
    transform = from_origin(left, top, res, res)

    def warp(path):
        with rasterio.open(path) as d, WarpedVRT(d, crs="EPSG:4326", transform=transform, width=width, height=height,
                                                 resampling=Resampling.average) as v:
            return v.read((1, 2, 3))

    image, old = warp(data / "Tanks.tif"), warp(data / "20150520_075001.tif")
    cells = gpd.read_file(data / "split_cells.geojson").to_crs(4326).reset_index(drop=True)
    tanks = gpd.read_file(data / "Tanks.gdb", layer="Tanks").to_crs(4326).reset_index(drop=True)
    shape = (height, width)
    cell = rasterize(((g, i + 1) for i, g in enumerate(cells.geometry)), shape, transform=transform, dtype="int32")
    inst = rasterize(((g, i + 1) for i, g in enumerate(tanks.geometry)), shape, transform=transform, dtype="int32")
    np.savez(OUT / "target.npz", image=image, cell=cell, inst=inst, splits=cells.split.to_numpy(dtype=str))

    # 2015 on the same grid and cells. Role 1 trains, role 2 selects the epoch, role 0 is hidden: target val and test
    # places, and pixels with no 2015 data. Every 6th train cell selects, so target val stays out of the source network.
    ids = cells.index[cells.split == "train"].to_numpy() + 1
    role = rasterize_roles(cell, ids, old.any(0))
    np.savez(OUT / "source.npz", image=old, role=role)
    print(json.dumps({"grid": list(shape), "res_deg": res, "tanks": len(tanks), "cells": cells.split.value_counts().to_dict(),
                      "pixels_by_role": np.bincount(role.ravel(), minlength=3).tolist(), "tif_mean": image.mean((1, 2)).round(1).tolist(),
                      "2015_mean": old[:, role == 1].mean(1).round(1).tolist(), "params": sum(p.numel() for p in build().parameters())}, indent=1))


def rasterize_roles(cell, train_ids, has_data):
    """Per pixel: 1 trains the source network, 2 selects its epoch, 0 is hidden."""
    role = np.zeros(cell.shape, np.uint8)
    role[np.isin(cell, train_ids)] = 1
    role[np.isin(cell, train_ids[::6])] = 2
    role[~has_data] = 0
    return role


def load_target():
    with np.load(OUT / "target.npz") as d:
        return {k: d[k] for k in d.files}


def part_cells(target, part):
    return np.where(target["splits"] == part)[0] + 1


def train_pixels(target):
    return np.isin(target["cell"], part_cells(target, "train"))


def blank(target):
    """Hide all pixels outside train cells, so no val or test pixel can reach a fit or a prompt through a tile's context."""
    return blank_outside(target["image"], train_pixels(target))


def blank_outside(image, keep):
    image = image.copy()
    for band in image:
        band[~keep] = np.uint8(band[keep].mean())
    return image


# --- network -------------------------------------------------------------------------------------

def build(weights=None):
    return smp.Unet("resnet18", encoder_weights=weights, classes=1)


def load_model(path):
    model = build()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(DEV).eval()


def weights_of(model):
    return {k: v.cpu().numpy().copy() for k, v in model.state_dict().items() if v.is_floating_point()}


def set_weights(model, net):
    state = model.state_dict()
    for name, value in net.items():
        state[name].copy_(torch.from_numpy(value).to(state[name].device))
    return model


def normal(x):
    return (x.float() / 255 - MEAN.to(x.device)) / STD.to(x.device)


def starts(n):
    return sorted({*range(0, max(n - SIZE, 0), SIZE // 2), max(n - SIZE, 0)})


@torch.no_grad()
def forward_map(model, image, features=False, batch=32):
    """Logits for a whole image. Each tile gives only its centre, except at the image edge. Optionally the head's input."""
    model.eval()
    h, w = image.shape[1:]
    logits = np.zeros((h, w), np.float32)
    channels = model.segmentation_head[0].in_channels
    feats = np.zeros((channels, h, w), np.float32) if features else None
    seen = []
    hook = model.segmentation_head.register_forward_pre_hook(lambda _m, args: seen.append(args[0]))
    margin = SIZE // 4
    corners = [(y, x) for y in starts(h) for x in starts(w)]
    for i in range(0, len(corners), batch):
        part = corners[i:i + batch]
        x = torch.from_numpy(np.stack([image[:, y:y + SIZE, x:x + SIZE] for y, x in part])).to(DEV)
        seen.clear()
        out = model(normal(x))[:, 0].cpu().numpy()
        f = seen[0].cpu().numpy()
        for k, (y, x0) in enumerate(part):
            y0, y1 = 0 if y == 0 else margin, SIZE if y + SIZE >= h else SIZE - margin
            x1, x2 = 0 if x0 == 0 else margin, SIZE if x0 + SIZE >= w else SIZE - margin
            logits[y + y0:y + y1, x0 + x1:x0 + x2] = out[k, y0:y1, x1:x2]
            if features:
                feats[:, y + y0:y + y1, x0 + x1:x0 + x2] = f[k, :, y0:y1, x1:x2]
    hook.remove()
    return (logits, feats) if features else logits


# --- scoring -------------------------------------------------------------------------------------

def counts(pred, target):
    """Per cell: pixel tp, fp, fn, then object tp, fp, fn at IoU >= 0.5. A pair of cell lists gives any metric or bootstrap."""
    cell, inst = target["cell"], target["inst"]
    out = np.zeros((len(target["splits"]) + 1, 6), np.int64)
    truth = inst > 0
    for col, m in enumerate((pred & truth, pred & ~truth, ~pred & truth)):
        out[:, col] = np.bincount(cell[m], minlength=len(out))

    comps, n = ni.label(pred)
    size = np.bincount(comps.ravel(), minlength=n + 1)
    size[0] = 0
    kept = np.where(size >= MIN_AREA)[0]
    lookup = np.zeros(n + 1, np.int64)
    lookup[kept] = np.arange(1, len(kept) + 1)
    comps = lookup[comps]
    tanks = int(inst.max()) + 1
    overlap = np.bincount(comps.ravel() * tanks + inst.ravel(), minlength=(len(kept) + 1) * tanks).reshape(-1, tanks)
    pred_size, tank_size = overlap.sum(1), overlap.sum(0)
    union = pred_size[:, None] + tank_size[None, :] - overlap
    iou = np.divide(overlap, union, out=np.zeros(overlap.shape), where=union > 0)
    iou[0, :] = 0
    iou[:, 0] = 0
    used_pred, used_tank = set(), set()
    for p, t in sorted(zip(*np.where(iou >= 0.5)), key=lambda pt: -iou[pt]):
        if p not in used_pred and t not in used_tank:
            used_pred.add(int(p))
            used_tank.add(int(t))

    def cell_at(labels, ids):
        centres = ni.center_of_mass(np.ones(labels.shape), labels, ids)
        return [cell[int(round(y)), int(round(x))] for y, x in centres]

    present = [t for t in range(1, tanks) if tank_size[t]]
    for t, c in zip(present, cell_at(inst, present)):
        out[c, 3 if t in used_tank else 5] += 1
    ids = list(range(1, len(kept) + 1))
    for p, c in zip(ids, cell_at(comps, ids)):
        if p not in used_pred:
            out[c, 4] += 1
    return out


def summary(table, cells):
    tp, fp, fn, otp, ofp, ofn = table[cells].sum(0)
    p, r = otp / max(otp + ofp, 1), otp / max(otp + ofn, 1)
    return {"iou": round(float(tp / max(tp + fp + fn, 1)), 4), "obj_precision": round(float(p), 4), "obj_recall": round(float(r), 4),
            "obj_f1": round(float(2 * p * r / max(p + r, 1e-9)), 4), "tp": int(otp), "fp": int(ofp), "fn": int(ofn)}


def score_map(pred, target, parts=("train", "val")):
    table = counts(pred, target)
    return {part: summary(table, part_cells(target, part)) for part in parts}


# --- training ------------------------------------------------------------------------------------

def augment(x, y, w, rng, jitter):
    # shortcut: one flip and turn per batch, not per sample; enough for this small test
    k = int(rng.integers(4))
    x, y, w = (torch.rot90(t, k, (-2, -1)) for t in (x, y, w))
    if rng.integers(2):
        x, y, w = (t.flip(-1) for t in (x, y, w))
    x = x.float()
    if jitter:
        gain = torch.empty(len(x), 1, 1, 1, device=x.device).uniform_(0.7, 1.3)
        shift = torch.empty(len(x), 1, 1, 1, device=x.device).uniform_(-30, 30)
        x = (x * gain + shift).clamp(0, 255)
    return x, y, w


def loss_on(logits, y, w):
    y, w = y.float(), w.float()
    bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).sum() / w.sum().clamp(min=1)
    p = torch.sigmoid(logits) * w
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + (y * w).sum() + 1)
    return bce + dice


def fit(sample, select, steps, epochs, out, jitter, seed, make=lambda: build("imagenet"), pre=normal):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-4, total_steps=steps * epochs)
    best = -1.0
    for epoch in range(epochs):
        model.train()
        for _ in range(steps):
            x, y, w = augment(*sample(rng), rng, jitter)
            loss = loss_on(model(pre(x))[:, 0], y, w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        score = select(model)
        if score > best:
            best = score
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out)
        print(f"epoch {epoch} loss {loss.item():.4f} select_iou {score:.4f} best {best:.4f}", flush=True)
    return best


def train(a):
    """Both networks train on crops of one whole image, on the same cells, so the date is the only difference between them."""
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    target = load_target()
    truth = target["inst"] > 0
    if a.kind == "source":
        with np.load(OUT / "source.npz") as d:
            image, role = d["image"], d["role"]
        keep = role == 1
        shown = blank_outside(image, role > 0)

        def select(model):
            pred, held = forward_map(model, shown) > 0, role == 2
            return (pred & truth & held).sum() / max(((pred | truth) & held).sum(), 1)

        name = f"source{'-jitter' if a.jitter else ''}-s{a.seed}"
    else:  # the upper bound: real training on the new image
        image, keep = target["image"], train_pixels(target)
        logits_of = tiny_map if a.kind == "tiny" else forward_map

        def select(model):
            return score_map(logits_of(model, image) > 0, target, ("val",))["val"]["iou"]

        name = f"{'tiny-trained' if a.kind == 'tiny' else 'target'}-s{a.seed}"
    pixels = torch.from_numpy(blank_outside(image, keep)).to(DEV)  # pixels that select the epoch or score it never reach training
    labels, mask = torch.from_numpy(truth).to(DEV), torch.from_numpy(keep).to(DEV)
    h, w = keep.shape
    where = np.argwhere(keep)

    def sample(rng):
        xs, ys, ws = [], [], []
        for cy, cx in where[rng.integers(len(where), size=32)]:
            y, x = min(max(cy - SIZE // 2, 0), h - SIZE), min(max(cx - SIZE // 2, 0), w - SIZE)
            xs.append(pixels[:, y:y + SIZE, x:x + SIZE])
            ys.append(labels[y:y + SIZE, x:x + SIZE])
            ws.append(mask[y:y + SIZE, x:x + SIZE])
        return torch.stack(xs), torch.stack(ys), torch.stack(ws)

    extra = {"make": TinyUNet, "pre": lambda x: x.float() / 255} if a.kind == "tiny" else {}
    print(name, "best", fit(sample, select, 25, a.epochs, OUT / "models" / f"{name}.pt", a.jitter, a.seed, **extra))


# --- evidence for the prompt and for the fair baseline ---------------------------------------------

def adabn(model, image):
    """Recompute all BatchNorm statistics on this image, with no labels."""
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    h, w = image.shape[1:]
    corners = [(y, x) for y in starts(h) for x in starts(w)]
    with torch.no_grad():
        for i in range(0, len(corners), 32):
            x = torch.from_numpy(np.stack([image[:, y:y + SIZE, x:x + SIZE] for y, x in corners[i:i + 32]])).to(DEV)
            model(normal(x))
    return model.eval()


def evidence(path, target):
    """Everything the LLM and the LDA baseline may know. It reads train-cell pixels only."""
    keep = train_pixels(target)
    image = blank(target)
    truth = target["inst"] > 0
    model = load_model(path)
    logits, feats = forward_map(model, image, features=True)
    inner, outer = ni.binary_erosion(truth, iterations=3), ni.binary_dilation(truth, iterations=3)
    masks = {"tank interior": inner & keep, "tank edge": outer & ~inner & keep, "background": ~outer & keep}
    channels = {name: (feats[:, m].mean(1), feats[:, m].std(1)) for name, m in masks.items()}
    pos, neg = feats[:, truth & keep], feats[:, ~truth & keep]
    within = (np.cov(pos[:, ::5]) * pos.shape[1] + np.cov(neg[:, ::5]) * neg.shape[1]) / (pos.shape[1] + neg.shape[1])

    before = weights_of(model)
    after = weights_of(adabn(load_model(path), image))
    bn = [name[:-len(".running_mean")] for name in before if name.endswith("running_mean")]
    shift = {n: float(np.mean(np.abs(after[f"{n}.running_mean"] - before[f"{n}.running_mean"]) /
                              np.sqrt(np.maximum(before[f"{n}.running_var"], 1e-8)))) for n in bn}
    per_channel = {n: {"source_mean": before[f"{n}.running_mean"], "source_var": before[f"{n}.running_var"],
                       "new_mean": after[f"{n}.running_mean"], "new_var": after[f"{n}.running_var"]}
                   for n in (bn[0], bn[-1])}
    table = counts(logits > 0, target)
    return {"channels": channels, "class_pixels": {k: int(m.sum()) for k, m in masks.items()},
            "pos_mean": pos.mean(1), "neg_mean": neg.mean(1), "within": within, "prior": float(truth[keep].mean()),
            "logit": {k: (float(logits[m].mean()), float(logits[m].std())) for k, m in masks.items()},
            "metrics": summary(table, part_cells(target, "train")), "shift": shift, "per_channel": per_channel,
            "weight": before[f"{HEAD}.weight"], "bias": before[f"{HEAD}.bias"],
            "shapes": {k: list(v.shape) for k, v in before.items()}}


def lda_edits(ev):
    """The closed-form head from the evidence table, as an edits block, so it reaches the network the way a reply does."""
    n = len(ev["pos_mean"])
    diff = ev["pos_mean"] - ev["neg_mean"]
    w = np.linalg.solve(ev["within"] + 1e-6 * np.trace(ev["within"]) / n * np.eye(n), diff)
    b = -w @ (ev["pos_mean"] + ev["neg_mean"]) / 2 + np.log(ev["prior"] / (1 - ev["prior"]))
    centre = ev["weight"].shape[-1] // 2
    return ("```edits\n" f"{HEAD}.weight[0, :, :, :] = 0\n"
            f"{HEAD}.weight[0, 0:{n}, {centre}, {centre}] = {json.dumps([round(float(v), 6) for v in w])}\n"
            f"{HEAD}.bias[0] = {float(b):.6f}\n```")


def num(v):
    return f"{float(v):.3g}"


def render(ev, shuffle_seed=None):
    """The evidence as text. The control permutes the channel rows, so the row-to-channel mapping in the text is false."""
    n = len(ev["pos_mean"])
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    perm = rng.permutation(n) if rng is not None else np.arange(n)
    lines = ["CURRENT RESULT ON THE TRAIN AREA: " + json.dumps(ev["metrics"]),
             f"Tank share of train-area pixels: {ev['prior']:.4f}.", "",
             "CURRENT LOGIT (mean, std) by pixel class:"]
    lines += [f"  {k}: {num(m)}, {num(s)}, {ev['class_pixels'][k]} pixels" for k, (m, s) in ev["logit"].items()]
    lines += ["", "HEAD INPUT CHANNELS (16 channels; each cell is mean, std):",
              "channel | " + " | ".join(ev["channels"])]
    for row in range(n):
        c = perm[row]
        lines.append(f"{row} | " + " | ".join(f"{num(m[c])}, {num(s[c])}" for m, s in ev["channels"].values()))
    lines += ["", "WITHIN-CLASS COVARIANCE of the head input channels (tank vs not tank, pooled; rows and columns are channels):"]
    lines += ["  " + " ".join(num(ev["within"][perm[i], perm[j]]) for j in range(n)) for i in range(n)]
    lines += ["", "CURRENT HEAD, the only true mapping of channels to weights:",
              f"{HEAD}.weight, shape {list(ev['weight'].shape)}; each row is one channel's 3 x 3 kernel, row by row:"]
    lines += [f"  channel {c}: " + " ".join(num(v) for v in ev["weight"][0, c].ravel()) for c in range(n)]
    lines += [f"{HEAD}.bias: {num(ev['bias'][0])}", "",
              "BATCHNORM SHIFT on the new image, by layer: mean of |new mean - source mean| / source std over the channels:"]
    lines += [f"  {k}: {num(v)}" for k, v in ev["shift"].items()]
    for name, table in ev["per_channel"].items():
        size = len(table["source_mean"])
        order = rng.permutation(size) if rng is not None else np.arange(size)
        lines += ["", f"BATCHNORM {name}, per channel: source running_mean, source running_var, new mean, new var:"]
        lines += [f"  {row}: " + " ".join(num(v[order[row]]) for v in table.values()) for row in range(size)]
    lines += ["", "ALL WEIGHT TENSORS (name, shape):"] + [f"  {k} {v}" for k, v in ev["shapes"].items()]
    return "\n".join(lines)


def report(a):
    target = load_target()
    ev = evidence(OUT / "models" / "source-s0.pt", target)
    template = (GEO / "prompt_repair.md").read_text()
    (OUT / "prompts").mkdir(parents=True, exist_ok=True)
    (OUT / "prompts" / "repair.txt").write_text(template.replace("{{EVIDENCE}}", render(ev)))
    (OUT / "prompts" / "repair-shuffled.txt").write_text(template.replace("{{EVIDENCE}}", render(ev, a.seed)))
    print(f"prompts written to {OUT / 'prompts'}")


# --- baselines -----------------------------------------------------------------------------------

def hist_lut(src, dst):
    """Per band, the lookup table that gives dst's values the distribution of src."""
    lut = []
    for band in range(3):
        a, b = (np.bincount(v[band].ravel(), minlength=256).cumsum() for v in (src, dst))
        lut.append(np.searchsorted(a / a[-1], b / b[-1]).clip(0, 255).astype(np.uint8))
    return lut


def filter_features(image):
    image = image.astype(np.float32)
    gray = image.mean(0)
    feats = [gray, image[2], image[0] - image[2]]
    for sigma in (10, 20, 40, 80):  # about 8 to 64 m on this grid
        smooth = ni.gaussian_filter(gray, sigma)
        feats += [smooth, smooth - ni.gaussian_filter(gray, sigma * 2),
                  -np.sqrt(np.maximum(ni.gaussian_filter(gray * gray, sigma) - smooth ** 2, 0))]
    return np.stack(feats, -1)


def fit_filter(target):
    """A closed-form linear fit on blur and contrast filters. It sees train pixels only, because a blur would otherwise
    carry val and test pixels into train features."""
    keep = train_pixels(target)
    truth = target["inst"] > 0
    X = filter_features(blank(target))
    mean, std = X[keep].mean(0), X[keep].std(0) + 1e-6
    X = (X - mean) / std
    pos, neg = X[keep & truth], X[keep & ~truth]
    w = np.linalg.solve(np.cov(neg[::7].T) + 1e-3 * np.eye(X.shape[-1]), pos.mean(0) - neg.mean(0))
    tr, y = (X @ w)[keep], truth[keep]
    best = max(np.quantile(tr, np.linspace(0.5, 0.995, 60)), key=lambda v: ((tr > v) & y).sum() / max(((tr > v) | y).sum(), 1))
    return mean, std, w, best


def filter_lda(target):
    """The floor, with no network: the fit runs on the full image, so its blurs see the whole scene, as a network does."""
    mean, std, w, best = fit_filter(target)
    return (filter_features(target["image"]) - mean) / std @ w > best


def retrain_head(path, target):
    """Real gradients on the same scope a reply can edit cheaply, with the same train pixels as the evidence."""
    torch.manual_seed(0)
    keep, truth = torch.from_numpy(train_pixels(target)).to(DEV), torch.from_numpy(target["inst"] > 0).to(DEV)
    model = load_model(path)
    _, feats = forward_map(model, blank(target), features=True)
    x = torch.from_numpy(feats).unsqueeze(0).to(DEV)
    head = model.segmentation_head[0]
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(300):
        loss = loss_on(head(x)[0, 0], truth, keep)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def model_for(name):
    return load_model(OUT / "models" / f"{name}.pt")


def predict(name, target):
    """The tank map of one comparator, from its saved state, on the full image."""
    if name == "filter-lda":
        return filter_lda(target)
    if name == "hist-match":
        with np.load(OUT / "source.npz") as d:
            lut = hist_lut(d["image"][:, d["role"] == 1], target["image"][:, train_pixels(target)])
        image = np.stack([lut[b][target["image"][b]] for b in range(3)])
        return forward_map(model_for("source-s0"), image) > 0
    return forward_map(model_for(name), target["image"]) > 0


def baselines(a):
    target = load_target()
    path = OUT / "models" / "source-s0.pt"
    ev = evidence(path, target)
    torch.save({k: v.cpu() for k, v in adabn(load_model(path), blank(target)).state_dict().items()}, OUT / "models" / "adabn.pt")
    torch.save({k: v.cpu() for k, v in retrain_head(path, target).state_dict().items()}, OUT / "models" / "head-retrain.pt")

    block = lda_edits(ev)
    (OUT / "lda_edits.md").write_text(block)
    base = weights_of(load_model(path))
    net, edit = wsp.apply_edits(block, base=base)
    assert edit["errors"] == [], edit["errors"]
    model = set_weights(load_model(path), net)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, OUT / "models" / "lda-head.pt")
    # The edits must reach the network: the same head applied directly gives the same map, except at tile seams.
    _, feats = forward_map(load_model(path), target["image"], features=True)
    direct = model.segmentation_head[0](torch.from_numpy(feats).unsqueeze(0).to(DEV))[0, 0].detach().cpu().numpy() > 0
    via_edits = forward_map(model, target["image"]) > 0
    agreement = float((direct == via_edits).mean())
    assert agreement > 0.995, f"LDA head reached the network badly: {agreement}"

    names = [p.stem for p in sorted((OUT / "models").glob("source*.pt"))] + ["adabn", "hist-match", "lda-head", "head-retrain", "filter-lda"]
    names += [p.stem for p in sorted((OUT / "models").glob("target*.pt"))]
    results = {"lda_agreement": agreement}
    for name in names:
        results[name] = score_map(predict(name, target), target)
        print(name, json.dumps(results[name]), flush=True)
    fault = results["source-s0"]["val"]["iou"]
    upper = results["target-s0"]["val"]["iou"]
    results["gate0"] = {"no_edit_val_iou": fault, "upper_val_iou": upper, "fault_exists": fault < 0.9 * upper}
    print("GATE 0", json.dumps(results["gate0"]))
    (OUT / "baselines.json").write_text(json.dumps(results, indent=1))


# --- Stage C: a tiny U-Net written from nothing -----------------------------------------------------

def tiny_map(model, image):
    """Logits for a whole image, as the prompt says: the edge repeats until 16 divides each side, and the padding is cut."""
    model.eval()
    h, w = image.shape[1:]
    x = torch.from_numpy(image[None]).float().to(DEV) / 255
    with torch.no_grad():
        return model(F.pad(x, (0, -w % 16, 0, -h % 16), mode="replicate"))[0, 0, :h, :w].cpu().numpy()


def tiny_shapes():
    return {k: tuple(v.shape) for k, v in TinyUNet().state_dict().items()}


def tiny_from(reply):
    net, edit = wsp.apply_edits(reply, tiny_shapes())
    edit["nonfinite"] = sum(int(np.count_nonzero(~np.isfinite(v))) for v in net.values())
    return set_weights(TinyUNet().to(DEV), net), edit


def facts(target):
    """The fact sheet for the prompt and the plain rule. It reads train-cell pixels only."""
    keep = train_pixels(target)
    image = blank(target).astype(np.float32) / 255
    truth = target["inst"] > 0
    inner, outer = ni.binary_erosion(truth, iterations=3), ni.binary_dilation(truth, iterations=3)
    masks = {"tank interior": inner & keep, "tank edge": outer & ~inner & keep, "background": ~outer & keep}
    gray = image.mean(0)
    local = np.sqrt(np.maximum(ni.uniform_filter(gray * gray, 5) - ni.uniform_filter(gray, 5) ** 2, 0))
    ids = np.unique(target["inst"][truth & keep])
    centres = ni.center_of_mass(np.ones(truth.shape), target["inst"], ids)
    inside = [i for i, (y, x) in zip(ids, centres) if keep[int(round(y)), int(round(x))]]
    area = np.bincount(target["inst"].ravel())[inside]
    return {"masks": {k: {"pixels": int(m.sum()), "mean": image[:, m].mean(1), "std": image[:, m].std(1),
                          "gray": float(gray[m].mean()), "local_std": float(local[m].mean())} for k, m in masks.items()},
            "share": float(truth[keep].mean()), "tanks": len(inside),
            "diameter": np.percentile(2 * np.sqrt(area / np.pi), [0, 25, 50, 75, 100])}


def render_facts(f):
    lines = [f"Tank share of train-area pixels: {f['share']:.4f}.",
             f"Tanks in the train area: {f['tanks']}. Tank diameter in pixels, min / 25% / median / 75% / max: " +
             " / ".join(f"{v:.0f}" for v in f["diameter"]) + ".", "",
             "Band values in [0, 1], by pixel class. Each cell is mean, std. Local std is the std of gray (mean of the bands) in a 5 x 5 window:",
             "class | red | green | blue | gray mean | mean local std | pixels"]
    for name, m in f["masks"].items():
        bands = " | ".join(f"{m['mean'][b]:.3f}, {m['std'][b]:.3f}" for b in range(3))
        lines.append(f"{name} | {bands} | {m['gray']:.3f} | {m['local_std']:.4f} | {m['pixels']}")
    return "\n".join(lines)


def fact_rule(f):
    """A plain program with the same facts: gray above the midpoint of two class means. It goes through the edits path, as a reply does."""
    t = (f["masks"]["tank interior"]["gray"] + f["masks"]["background"]["gray"]) / 2
    return ("```edits\n"
            "e0a.weight[0, 0:3, 1, 1] = [0.333333, 0.333333, 0.333333]\n"
            "e0b.weight[0, 0, 1, 1] = 1\n"
            "d0.weight[0, 8, 1, 1] = 1\n"  # channel 8 of d0's input is channel 0 of e0: up(d1) comes first
            "out.weight[0, 0, 0, 0] = 20\n"
            f"out.bias[0] = {-20 * t:.6f}\n```")


def scored(pred, target):
    """Train and val. Test cells are added only after `final` opened them, and then they only report and select nothing."""
    table = counts(pred, target)
    opened = (OUT / "final.json").exists()
    result = {part: summary(table, part_cells(target, part)) for part in ("train", "val", "test")[:3 if opened else 2]}
    if opened:
        cells = part_cells(target, "test")
        rng = np.random.default_rng(0)
        draws = [rng.choice(cells, len(cells)) for _ in range(2000)]
        for m in ("iou", "obj_f1"):
            result["test"][f"{m}_ci"] = np.percentile([summary(table, d)[m] for d in draws], [2.5, 97.5]).round(3).tolist()
    return result


def write_report(a):
    target = load_target()
    f = facts(target)
    (OUT / "prompts").mkdir(parents=True, exist_ok=True)
    (OUT / "prompts" / "write.txt").write_text((GEO / "prompt_write.md").read_text()
                                               .replace("{{SOURCE}}", (GEO / "tiny.py").read_text().strip())
                                               .replace("{{FACTS}}", render_facts(f)))
    rows = {}
    model, edit = tiny_from(fact_rule(f))
    assert edit["errors"] == [] and edit["nonzero"] == 7, edit
    rows["fact-rule"] = scored(tiny_map(model, target["image"]) > 0, target)
    rows["zero"] = scored(tiny_map(tiny_from("")[0], target["image"]) > 0, target)
    for seed in range(3):
        torch.manual_seed(seed)
        rows[f"random-s{seed}"] = scored(tiny_map(TinyUNet().to(DEV), target["image"]) > 0, target)
    rows["filter-lda"] = scored(filter_lda(target), target)
    trained = TinyUNet().to(DEV)
    trained.load_state_dict(torch.load(OUT / "models" / "tiny-trained-s0.pt", map_location=DEV))
    rows["tiny-trained-s0"] = scored(tiny_map(trained, target["image"]) > 0, target)
    (OUT / "write_baselines.json").write_text(json.dumps(rows, indent=1))
    for name, row in rows.items():
        print(name, json.dumps(row), flush=True)


def write_score(a):
    run = Path(a.run)
    record = json.loads((run / "run.json").read_text())
    problem = wsp.run_problem(record)
    if problem:
        sys.exit(f"refusing to score: {problem}")
    target = load_target()
    model, result = tiny_from((run / "answer.md").read_text())
    result.update(scored(tiny_map(model, target["image"]) > 0, target))
    (run / "score.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("applied", "statements", "nonzero", "nonfinite", "cut_off", "train", "val", "test") if k in result}))
    if result["errors"]:
        print(f"{len(result['errors'])} statements skipped; see score.json")


# --- a reply -------------------------------------------------------------------------------------

def edited_model(run):
    record = json.loads((Path(run) / "run.json").read_text())
    problem = wsp.run_problem(record)
    if problem:
        sys.exit(f"refusing to score: {problem}")
    path = OUT / "models" / "source-s0.pt"
    base = weights_of(load_model(path))
    net, edit = wsp.apply_edits((Path(run) / "answer.md").read_text(), base=base)
    changed = sum(int(np.count_nonzero(net[k] != base[k])) for k in net)
    total = sum(v.size for v in net.values())
    edit.update(changed=changed, changed_share=changed / total,
                nonfinite=sum(int(np.count_nonzero(~np.isfinite(v))) for v in net.values()))
    return set_weights(load_model(path), net), edit


def score(a):
    target = load_target()
    model, result = edited_model(a.run)
    result.update(score_map(forward_map(model, target["image"]) > 0, target))
    (Path(a.run) / "score.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("applied", "statements", "changed", "changed_share", "nonfinite", "train", "val")}, indent=1))
    if result["errors"]:
        print(f"{len(result['errors'])} statements skipped; see score.json")


def final(a):
    """Open the test cells, once. A second run needs a deliberate rm of final.json, and a new test area in honesty."""
    out = OUT / "final.json"
    if out.exists():
        sys.exit(f"test cells already used: {out}")
    target = load_target()
    cells = part_cells(target, "test")
    rng = np.random.default_rng(0)
    draws = [rng.choice(cells, len(cells)) for _ in range(a.boot)]
    results = {"boot": a.boot, "test_cells": len(cells), "models": {}}
    reference = None
    for name in a.names:
        pred = forward_map(edited_model(name)[0], target["image"]) > 0 if Path(name).is_dir() else predict(name, target)
        table = counts(pred, target)
        boot = {m: np.array([summary(table, d)[m] for d in draws]) for m in ("iou", "obj_f1")}
        reference = boot if reference is None else reference
        row = {**summary(table, cells)}
        for m in boot:
            row[f"{m}_ci"] = np.percentile(boot[m], [2.5, 97.5]).round(3).tolist()
            row[f"{m}_minus_first_ci"] = np.percentile(boot[m] - reference[m], [2.5, 97.5]).round(3).tolist()
        results["models"][Path(name).name] = row
        print(Path(name).name, json.dumps(row), flush=True)
    out.write_text(json.dumps(results, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", required=True, help="folder with Tanks.tif, 20150520_075001.tif, Tanks.gdb, split_cells.geojson")
    p.set_defaults(func=prepare)
    p = sub.add_parser("train")
    p.add_argument("kind", choices=["source", "target", "tiny"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jitter", action="store_true", help="colour jitter, the plain way to prevent this fault")
    p.add_argument("--epochs", type=int, default=40)
    p.set_defaults(func=train)
    p = sub.add_parser("report")
    p.add_argument("--seed", type=int, default=0, help="shuffle seed of the control")
    p.set_defaults(func=report)
    sub.add_parser("baselines").set_defaults(func=baselines)
    p = sub.add_parser("score")
    p.add_argument("run")
    p.set_defaults(func=score)
    sub.add_parser("write-report", help="Stage C: prompt and baselines").set_defaults(func=write_report)
    p = sub.add_parser("write-score", help="Stage C: score one reply")
    p.add_argument("run")
    p.set_defaults(func=write_score)
    p = sub.add_parser("final")
    p.add_argument("names", nargs="+", help="comparators and run folders; the first is the reference for the differences")
    p.add_argument("--boot", type=int, default=2000)
    p.set_defaults(func=final)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
