#!/usr/bin/env python3
"""Write docs/tanks.json and the tank figures from the private results in results/geo.

The public build needs only those outputs. The figures use no pixels of the private imagery: the networks run on a synthetic scene.

Run with the geo venv: python docs/tank_figures.py --gdb <folder>/Tanks.gdb
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
GEO = ROOT / "results" / "geo"
sys.path.insert(0, str(ROOT / "geo"))
import tanks  # noqa: E402

# Measured once on the Mac and on gc1, with plain programs and no LLM, after the Stage B replies (val cells).
BN1_CHECKS = {"true": [0.5535, 0.449], "variance": [0.5076, 0.3902], "mean": [0.1006, 0.0877], "scale": [0.4995, 0.3077],
              "shuffle_skip": [[0.4918, 0.2222], [0.5589, 0.3684], [0.6813, 0.6667], [0.5278, 0.2667], [0.2444, 0.0851]],
              "shuffle_all": [[0.0931, 0.0169], [0.3097, 0.0476], [0.0877, 0.0556], [0.0433, 0.0], [0.0449, 0.0028]]}
PLAIN_RULES = {"band": [0.1454, 0.039, 0.156, 0.0512], "band_share": [0.3845, 0.0822, 0.4271, 0.1831], "index": [0.3007, 0.0913, 0.3354, 0.1176]}


def load(path):
    return json.loads((GEO / path).read_text())


def row(group, label, val, test):
    return {"group": group, "label": label, "val_iou": val["iou"], "val_f1": val["obj_f1"], "iou": test["iou"], "iou_ci": test["iou_ci"],
            "f1": test["obj_f1"], "f1_ci": test["obj_f1_ci"], "tp": test["tp"], "fp": test["fp"], "fn": test["fn"]}


def replies(pattern):
    out = {}
    for folder in sorted(glob.glob(str(GEO / pattern))):
        name = folder[-5:]
        record, score = (json.loads((Path(folder) / f).read_text()) for f in ("run.json", "score.json"))
        out[name] = {"folder": folder, "score": score, "tokens": record["audit"]["usage"]["output_tokens"], "seconds": record["wall_s"], "cost": record["audit"].get("cost_usd")}
    return out


def data(gdb):
    final, val = load("final.json")["models"], load("baselines.json")
    write, refit, det = load("write_baselines.json"), load("filter_lda_refit.json"), load("detector.json")
    def base(group, label, name):
        return row(group, label, val[name]["val"], final[name])
    repair, control, claude, astra = (replies(f"tanks-{p}-*") for p in ("repair-claude", "repair-shuffled", "write-claude", "write-gpt"))
    def stage_b(group, label, rep):
        return row(group, label, rep["score"]["val"], next(v for k, v in final.items() if k.endswith(Path(rep["folder"]).name)))
    def stage_c(group, label, rep):
        return row(group, label, rep["score"]["val"], rep["score"]["test"])
    rows = [base("base", "No edit, seed 0", "source-s0"), base("base", "No edit, seed 1", "source-s1"), base("base", "No edit, seed 2", "source-s2"),
            row("base", "Fact rule", write["fact-rule"]["val"], write["fact-rule"]["test"]),
            base("base", "LDA head", "lda-head"), base("base", "Head retrained by gradient", "head-retrain"),
            row("base", "Filter LDA, no network", refit["val"], refit["test"]),
            stage_c("llm", "Claude write 54466", claude["54466"]), stage_c("llm", "Claude write 54464", claude["54464"]), stage_c("llm", "Claude write 54468", claude["54468"]),
            stage_c("llm", "Astra write 60743", astra["60743"]), stage_c("llm", "Astra write 60745", astra["60745"]), stage_c("llm", "Astra write 60748", astra["60748"]),
            stage_b("llm", "Claude repair (3 replies, same effect)", repair["41904"]),
            stage_b("llm", "Claude control 41910 and 41912", control["41910"]), stage_b("llm", "Claude control 41914", control["41914"]),
            base("base", "AdaBN", "adabn"), base("base", "Histogram match", "hist-match"), base("best", "Colour jitter in training", "source-jitter-s0"),
            row("best", "Tiny U-Net, trained", write["tiny-trained-s0"]["val"], write["tiny-trained-s0"]["test"]),
            base("best", "ResNet18 U-Net, trained", "target-s0"),
            row("best", "RT-DETRv2-S, native pixels", det["rtdetrv2s_native_s1.0"]["masks"]["val"], det["rtdetrv2s_native_s1.0"]["masks"]["test"]),
            row("best", "RT-DETRv2-S, fixed settings", det["rtdetrv2s_grid_s1.5"]["masks"]["val"], det["rtdetrv2s_grid_s1.5"]["masks"]["test"])]
    def calls(group):
        return {name: {"applied": r["score"]["applied"], "statements": r["score"]["statements"], "tokens": r["tokens"], "seconds": r["seconds"],
                       "cost": r["cost"], "changed": r["score"].get("changed"), "nonzero": r["score"]["nonzero"]} for name, r in group.items()}
    return {"rows": rows, "calls": {"repair": calls(repair), "control": calls(control), "claude": calls(claude), "astra": calls(astra)},
            "bn1": BN1_CHECKS, "plain": PLAIN_RULES, "circles": {k: det["rtdetrv2s_grid_s1.5"]["circles"][k] for k in ("train", "val", "test")},
            "sizes": sizes(gdb)}


def sizes(gdb):
    import geopandas as gpd
    truth = gpd.read_file(gdb, layer="Tanks").to_crs(3857)
    preds = gpd.read_file(GEO / "rtdetr" / "rtdetrv2s_grid_s1.5.gpkg").to_crs(3857)
    diameter = 2 * np.sqrt(truth.geometry.area / np.pi)

    def best(t, keep):
        return max([t.intersection(p).area / t.union(p).area for p in preds.geometry[keep] if t.intersects(p)] + [0])

    at_70 = np.array([best(t, preds.score >= 0.7) for t in truth.geometry]) >= 0.5
    at_any = np.array([best(t, preds.score >= 0) for t in truth.geometry]) >= 0.5
    out = []
    for label, low, high in (("Under 20 m", 0, 20), ("20 to 40 m", 20, 40), ("40 to 80 m", 40, 80), ("80 m or more", 80, 1e9)):
        m = (diameter >= low) & (diameter < high)
        out.append({"label": label, "tanks": int(m.sum()), "at_70": int(at_70[m].sum()), "any": int(at_any[m].sum())})
    return out


def synthetic_scene(h=384, w=512):
    """Pale discs for tanks, with other objects that a brightness rule can mistake for tanks. Colours come from the fact sheet."""
    rng = np.random.default_rng(7)
    img = Image.new("RGB", (w, h), (137, 130, 97))
    draw = ImageDraw.Draw(img)
    truth = Image.new("L", (w, h), 0)
    mark = ImageDraw.Draw(truth)
    draw.line([(0, 330), (512, 250)], fill=(205, 200, 180), width=10)  # a pale road
    draw.rectangle([30, 30, 130, 80], fill=(226, 222, 206))  # a pale roof
    draw.rectangle([390, 300, 490, 370], fill=(120, 140, 200))  # a blue roof
    for i in range(40):  # small pale buildings
        x, y = int(rng.integers(150, 330)), int(rng.integers(280, 370))
        draw.rectangle([x, y, x + 8, y + 5], fill=(200, 195, 175))
    for x, y, r in [(95, 170, 62), (250, 110, 40), (360, 90, 26), (420, 190, 48), (250, 215, 16), (300, 200, 10), (450, 50, 12)]:
        draw.ellipse([x - r - 3, y - r - 3, x + r + 3, y + r + 3], fill=(95, 90, 75))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(224, 222, 206))
        mark.ellipse([x - r, y - r, x + r, y + r], fill=255)
    noise = rng.normal(0, 7, (h, w, 3))
    return np.clip(np.array(img, np.float32) + noise, 0, 255).astype(np.uint8), np.array(truth) > 0


def caption(array, text):
    image = Image.fromarray(array).convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 20), (255, 255, 255))
    canvas.paste(image, (0, 20))
    ImageDraw.Draw(canvas).text((4, 4), text, fill=(20, 30, 28))
    return canvas


def mosaic(panels, columns):
    w, h = panels[0].size
    sheet = Image.new("RGB", (columns * w + (columns - 1) * 8, (len(panels) + columns - 1) // columns * (h + 8) - 8), (255, 255, 255))
    for i, panel in enumerate(panels):
        sheet.paste(panel, (i % columns * (w + 8), i // columns * (h + 8)))
    return sheet


def scene_figure(claude, astra):
    image, truth = synthetic_scene()
    chw = image.transpose(2, 0, 1)
    gray = lambda m: np.stack([np.where(m, 255, 40)] * 3, -1).astype(np.uint8)

    def net(reply):
        model, _ = tanks.tiny_from(Path(reply["folder"], "answer.md").read_text())
        return gray(tanks.tiny_map(model, chw) > 0)

    trained = tanks.TinyUNet().to(tanks.DEV)
    trained.load_state_dict(tanks.torch.load(GEO / "models" / "tiny-trained-s0.pt", map_location=tanks.DEV))
    panels = [caption(image, "Synthetic scene"), caption(gray(truth), "True tanks"),
              caption(net(claude["54468"]), "Claude reply 54468"), caption(net(astra["60748"]), "Astra reply 60748"),
              caption(net(claude["54464"]), "Claude reply 54464"), caption(gray(tanks.tiny_map(trained, chw) > 0), "Same network, trained")]
    mosaic(panels, 2).save(ROOT / "docs" / "img" / "tank-scene.jpg", quality=82)


def weights_panel(reply, title):
    model, _ = tanks.tiny_from(Path(reply["folder"], "answer.md").read_text())
    state = {k: v.cpu().numpy() for k, v in model.state_dict().items() if k.endswith(".weight")}
    cell, images = 3, []
    for name, w in state.items():
        o, i = w.shape[:2]
        k = w.shape[2]
        grid = np.zeros((o * k, i * k))
        for a in range(o):
            for b in range(i):
                grid[a * k:(a + 1) * k, b * k:(b + 1) * k] = w[a, b]
        scale = np.abs(grid).max() or 1
        v = grid / scale
        pixel = np.ones(grid.shape + (3,), np.float32)
        pos, neg = v > 0, v < 0
        pixel[pos] = 1 - v[pos, None] * (1 - np.array([0.92, 0.41, 0.2]))
        pixel[neg] = 1 + v[neg, None] * (1 - np.array([0.1, 0.65, 0.5]))
        pixel = (pixel * 255).astype(np.uint8)
        images.append((name.split(".")[0], Image.fromarray(pixel).resize((grid.shape[1] * cell, grid.shape[0] * cell), Image.NEAREST)))
    W, x, y, row_h = 335, 4, 24, 0
    sheet = Image.new("RGB", (W, 1200), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), title, fill=(20, 30, 28))
    for name, im in images:
        if x + im.width > W - 4:
            x, y, row_h = 4, y + row_h + 18, 0
        draw.text((x, y), name, fill=(70, 80, 76))
        sheet.paste(im, (x, y + 12))
        draw.rectangle([x - 1, y + 11, x + im.width, y + 12 + im.height], outline=(200, 205, 202))
        x, row_h = x + im.width + 8, max(row_h, im.height)
    return sheet.crop((0, 0, W, y + row_h + 18))


def weights_figure(claude, astra):
    a, b = weights_panel(claude["54468"], "Claude reply 54468: weights of each layer"), weights_panel(astra["60748"], "Astra reply 60748: weights of each layer")
    sheet = Image.new("RGB", (a.width + b.width + 8, max(a.height, b.height)), (255, 255, 255))
    sheet.paste(a, (0, 0))
    sheet.paste(b, (a.width + 8, 0))
    sheet.save(ROOT / "docs" / "img" / "tank-weights.jpg", quality=86)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gdb", required=True)
    a = p.parse_args()
    out = data(a.gdb)
    (ROOT / "docs" / "tanks.json").write_text(json.dumps(out, indent=1) + "\n")
    claude, astra = replies("tanks-write-claude-*"), replies("tanks-write-gpt-*")
    scene_figure(claude, astra)
    weights_figure(claude, astra)
    print("docs/tanks.json, docs/img/tank-scene.jpg, docs/img/tank-weights.jpg")


if __name__ == "__main__":
    main()
