"""Run on gc1 with the geo venv: python -m unittest geo/test_tanks.py. CPU is enough."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detector
import tanks


def disc(shape, y, x, r):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (yy - y) ** 2 + (xx - x) ** 2 <= r ** 2


def scene(size=512):
    """Four cells, one per quarter plus a second train cell: train, train, val, test."""
    half = size // 2
    cell = np.ones((size, size), np.int32)
    cell[:half, half:] = 2
    cell[half:, :half] = 3
    cell[half:, half:] = 4
    inst = np.zeros((size, size), np.int32)
    centres = [(120, 120), (120, 380), (380, 120), (380, 380)]
    image = np.full((3, size, size), 100, np.uint8)
    for i, (y, x) in enumerate(centres):
        m = disc((size, size), y, x, 40)
        inst[m] = i + 1
        image[:, m] = 220
    return {"image": image, "inst": inst, "cell": cell, "splits": np.array(["train", "train", "val", "test"])}


class Scoring(unittest.TestCase):
    def test_perfect_map(self):
        t = scene()
        result = tanks.score_map(t["inst"] > 0, t, ("train", "val", "test"))
        for part in result.values():
            self.assertEqual((part["iou"], part["obj_f1"]), (1.0, 1.0))
        self.assertEqual((result["train"]["tp"], result["val"]["tp"]), (2, 1))

    def test_empty_map(self):
        t = scene()
        part = tanks.score_map(np.zeros_like(t["inst"], bool), t)["val"]
        self.assertEqual((part["iou"], part["obj_f1"], part["fn"]), (0.0, 0.0, 1))

    def test_false_object_counts_in_the_cell_of_its_centre(self):
        t = scene()
        pred = (t["inst"] > 0) | disc(t["inst"].shape, 300, 60, 10)  # val cell, no tank there
        result = tanks.score_map(pred, t, ("train", "val"))
        self.assertEqual((result["val"]["fp"], result["train"]["fp"]), (1, 0))

    def test_specks_are_not_objects(self):
        t = scene()
        pred = (t["inst"] > 0).copy()
        pred[300, 60] = True
        self.assertEqual(tanks.score_map(pred, t)["val"]["fp"], 0)

    def test_half_a_tank_is_not_a_match(self):
        t = scene()
        pred = (t["inst"] > 0) & (np.arange(512)[None, :] < 380)
        self.assertEqual(tanks.score_map(pred, t, ("test",))["test"]["tp"], 0)


class Partitions(unittest.TestCase):
    def test_blank_hides_all_but_train_cells(self):
        t = scene()
        other = {**t, "image": t["image"].copy()}
        other["image"][:, 256:, :] = 7
        self.assertTrue((tanks.blank(t) == tanks.blank(other)).all())

    def test_changing_val_and_test_pixels_and_labels_keeps_the_prompt(self):
        import torch
        t = scene()
        torch.manual_seed(0)
        model = tanks.build()
        changed = {**t, "image": t["image"].copy(), "inst": t["inst"].copy()}
        changed["image"][:, 256:, :] = np.random.default_rng(1).integers(0, 255, (3, 256, 512), dtype=np.uint8)
        changed["inst"][380 - 40:380 + 40, 120 - 40:120 + 40] = 0
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save(model.state_dict(), path)
            self.assertEqual(tanks.render(tanks.evidence(path, t)), tanks.render(tanks.evidence(path, changed)))


class Tiny(unittest.TestCase):
    def test_size_is_in_the_design_range_and_matches_the_prompt(self):
        total = sum(p.numel() for p in tanks.TinyUNet().parameters())
        self.assertEqual(total, 16961)
        self.assertIn(f"{total:,} weights", (Path(tanks.GEO) / "prompt_write.md").read_text())

    def test_receptive_field_covers_the_largest_tank(self):
        import torch
        model = tanks.TinyUNet().double()  # float32 rounding hides the outer ring of a small change
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(0.2)
        x = torch.ones(1, 3, 512, 512, dtype=torch.float64)
        y = x.clone()
        y[0, :, 256, 256] = 2
        with torch.no_grad():
            changed = (model(x) != model(y))[0, 0]
        rows, cols = torch.where(changed)
        self.assertGreaterEqual(int(rows.max() - rows.min()) + 1, 164)
        self.assertGreaterEqual(int(cols.max() - cols.min()) + 1, 164)

    def test_zero_network_finds_nothing(self):
        t = scene()
        model, _ = tanks.tiny_from("")
        self.assertFalse((tanks.tiny_map(model, t["image"]) > 0).any())

    def test_fact_rule_reaches_the_output_through_the_edits(self):
        t = scene()
        f = tanks.facts(t)
        model, edit = tanks.tiny_from(tanks.fact_rule(f))
        self.assertEqual((edit["errors"], edit["nonzero"]), ([], 7))
        pred = tanks.tiny_map(model, t["image"]) > 0
        self.assertEqual(tanks.score_map(pred, t, ("val",))["val"]["iou"], 1.0)

    def test_changing_val_and_test_pixels_and_labels_keeps_the_facts(self):
        t = scene()
        changed = {**t, "image": t["image"].copy(), "inst": t["inst"].copy()}
        changed["image"][:, 256:, :] = np.random.default_rng(1).integers(0, 255, (3, 256, 512), dtype=np.uint8)
        changed["inst"][380 - 40:380 + 40, 120 - 40:120 + 40] = 0
        self.assertEqual(tanks.render_facts(tanks.facts(t)), tanks.render_facts(tanks.facts(changed)))


class FilterFit(unittest.TestCase):
    def test_val_and_test_pixels_do_not_change_the_fit(self):
        t = scene(1024)
        t["inst"] = np.zeros_like(t["inst"])
        for i, (y, x) in enumerate([(120, 120), (120, 800), (400, 300), (800, 120), (800, 800)]):
            t["inst"][disc(t["inst"].shape, y, x, 40)] = i + 1
        t["image"] = np.random.default_rng(0).integers(60, 140, t["image"].shape, dtype=np.uint8)
        t["image"][:, t["inst"] > 0] = 220
        t["cell"] = np.ones_like(t["cell"])
        t["cell"][:, 512:] = 2
        t["cell"][512:, :512] = 3
        t["cell"][512:, 512:] = 4
        changed = {**t, "image": t["image"].copy()}
        changed["image"][:, 512:, :] = 255
        for a, b in zip(tanks.fit_filter(t), tanks.fit_filter(changed)):
            self.assertTrue(np.array_equal(a, b))


class DetectorMatch(unittest.TestCase):
    def setUp(self):
        from shapely.geometry import Point
        self.point = Point
        self.truths = [Point(0, 0).buffer(10), Point(100, 0).buffer(10), Point(200, 0).buffer(10)]
        self.parts = ["val", "val", "test"]

    def run_match(self, preds, scores, pred_part, conf=0.7, part="val"):
        return detector.match(preds, np.array(scores), self.truths, pred_part, self.parts, conf, part)

    def test_perfect_circles_in_one_partition(self):
        r = self.run_match(self.truths, [0.9, 0.8, 0.9], ["val", "val", "test"])
        self.assertEqual((r["ap50"], r["f1"], r["tp"], r["fp"], r["fn"]), (1.0, 1.0, 2, 0, 0))

    def test_low_score_counts_for_ap_but_not_at_the_threshold(self):
        r = self.run_match(self.truths[:2], [0.9, 0.5], ["val", "val"])
        self.assertEqual((r["ap50"], r["tp"], r["fn"]), (1.0, 1, 1))

    def test_false_circle_only_counts_in_its_partition(self):
        false = self.point(50, 50).buffer(5)
        preds, scores = self.truths[:2] + [false], [0.9, 0.9, 0.8]
        self.assertEqual(self.run_match(preds, scores, ["val", "val", "val"])["fp"], 1)
        self.assertEqual(self.run_match(preds, scores, ["val", "val", "test"])["fp"], 0)

    def test_two_circles_on_one_tank_give_one_match(self):
        r = self.run_match([self.truths[0], self.truths[0]], [0.9, 0.8], ["val", "val"])
        self.assertEqual((r["tp"], r["fp"], r["fn"]), (1, 1, 1))

    def test_a_poor_overlap_is_not_a_match(self):
        r = self.run_match([self.point(15, 0).buffer(10)], [0.9], ["val"])
        self.assertEqual((r["tp"], r["fp"]), (0, 1))


class SealedTest(unittest.TestCase):
    def test_test_cells_report_only_after_final(self):
        t = scene()
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(tanks, "OUT", Path(folder)):
            self.assertNotIn("test", tanks.scored(t["inst"] > 0, t))
            (Path(folder) / "final.json").write_text("{}")
            self.assertEqual(tanks.scored(t["inst"] > 0, t)["test"]["iou"], 1.0)


class Roles(unittest.TestCase):
    def test_roles_hide_other_cells_and_missing_data(self):
        cell = np.array([[1, 2, 3, 0]] * 2)
        has_data = np.ones_like(cell, bool)
        has_data[1, 1] = False
        role = tanks.rasterize_roles(cell, np.array([1, 2]), has_data)
        self.assertEqual(role.tolist(), [[2, 1, 0, 0], [2, 0, 0, 0]])


class Evidence(unittest.TestCase):
    def test_control_changes_channel_rows_only(self):
        n = 16
        ev = {"pos_mean": np.arange(n, dtype=float), "neg_mean": np.zeros(n), "within": np.eye(n), "prior": 0.1,
              "metrics": {"iou": 0.1}, "class_pixels": {"tank interior": 1}, "logit": {"tank interior": (0.0, 1.0)},
              "channels": {"tank interior": (np.arange(n, dtype=float), np.ones(n))}, "shift": {"a": 0.5},
              "per_channel": {"a": {"source_mean": np.arange(4.0), "source_var": np.ones(4), "new_mean": np.zeros(4), "new_var": np.ones(4)}},
              "weight": np.arange(n * 9, dtype=np.float32).reshape(1, n, 3, 3), "bias": np.zeros(1), "shapes": {"x": [1]}}
        true, control = tanks.render(ev), tanks.render(ev, 0)
        self.assertNotEqual(true, control)
        head = lambda text: text[text.index("CURRENT HEAD"):text.index("BATCHNORM SHIFT")]
        self.assertEqual(head(true), head(control))

    def test_lda_points_at_the_tank_channel(self):
        n = 16
        pos, neg = np.zeros(n), np.zeros(n)
        pos[3] = 2.0
        ev = {"pos_mean": pos, "neg_mean": neg, "within": np.eye(n), "prior": 0.5, "weight": np.ones((1, n, 3, 3), np.float32), "bias": np.ones(1, np.float32)}
        base = {f"{tanks.HEAD}.weight": ev["weight"], f"{tanks.HEAD}.bias": ev["bias"]}
        net, report = tanks.wsp.apply_edits(tanks.lda_edits(ev), base=base)
        self.assertEqual(report["errors"], [])
        w = net[f"{tanks.HEAD}.weight"][0]
        self.assertEqual(w[3, 1, 1], max(w.ravel()))
        self.assertEqual(w[3, 0, 0], 0)
        self.assertAlmostEqual(float(net[f"{tanks.HEAD}.bias"][0]), -2.0, places=4)

    def test_histogram_lookup_moves_dst_to_src(self):
        rng = np.random.default_rng(0)
        src = rng.integers(150, 200, (3, 200, 200), dtype=np.uint8)
        dst = rng.integers(50, 100, (3, 200, 200), dtype=np.uint8)
        lut = tanks.hist_lut(src, dst)
        self.assertAlmostEqual(float(lut[0][dst[0]].mean()), float(src[0].mean()), delta=2)

    def test_shape_size_of_the_network(self):
        model = tanks.build()
        total = sum(p.numel() for p in model.parameters())
        self.assertEqual(model.segmentation_head[0].weight.shape[1], 16)
        self.assertEqual(model.segmentation_head[0].kernel_size, (3, 3))
        print(json.dumps({"params": total, "head": model.segmentation_head[0].weight.numel()}))


if __name__ == "__main__":
    unittest.main()
