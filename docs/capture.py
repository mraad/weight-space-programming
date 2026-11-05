#!/usr/bin/env python3
"""Capture the page screenshots for the manual, or the animated GIF for the README, into docs/img.

Run: .venv/bin/python docs/capture.py [--gif]
"""
import io
import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
PAGE = (ROOT / "lander" / "index.html").as_uri()
OUT = ROOT / "docs" / "img"
OUT.mkdir(exist_ok=True)
JPEG = {"type": "jpeg", "quality": 78}

# Seeded randomness and a virtual clock: the real page runs one tick per step, so the GIF is the same on every run.
# Only the page's own loop (named `frame`) is held back; Playwright needs the real requestAnimationFrame.
VIRTUAL_CLOCK = """
(() => {
  let a = 20261105, t = 0, next = null;
  Math.random = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let x = a;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
  performance.now = () => t;
  const real = window.requestAnimationFrame.bind(window);
  window.requestAnimationFrame = (cb) => { if (cb.name === 'frame') { next = cb; return 0; } return real(cb); };
  window.step = () => { t += 1000 / 30 + 0.001; const cb = next; next = null; if (cb) cb(t); };
})();
"""
TICKS_PER_FRAME = 6  # at 100 ms a frame, this plays at twice the real speed of 30 ticks a second
HOLD_TICKS = 36  # how long the outcome message stays on screen
GIF_COLORS = 48


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 900, "height": 900})
        page.goto(PAGE)
        page.select_option("#pilot", "1")  # Claude reply B
        page.keyboard.press("3")
        # A new flight toward pad 3, caught in the air with the engine firing.
        page.click("#restart")
        page.wait_for_function("Number(document.getElementById('in1').textContent) < 0.45", timeout=60_000)
        page.screenshot(path=OUT / "page-flight.jpg", clip={"x": 0, "y": 0, "width": 900, "height": 760}, **JPEG)
        page.locator("#weightsPanel").screenshot(path=OUT / "page-weights.jpg", **JPEG)

        phone = browser.new_page(viewport={"width": 390, "height": 760}, device_scale_factor=2)
        phone.goto(PAGE)
        phone.wait_for_function("Number(document.getElementById('in1').textContent) < 0.6", timeout=60_000)
        phone.screenshot(path=OUT / "page-phone.jpg", **JPEG)
        browser.close()


def gif():
    """Claude reply A lands on pad 2, lands on pad 3, then changes course when the target changes in flight."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 720, "height": 760}, color_scheme="light")
        page.add_init_script(VIRTUAL_CLOCK)
        page.goto(PAGE)
        assert page.evaluate("document.getElementById('pilot').selectedOptions[0].text").startswith("Claude reply A")
        box = page.locator("#status").bounding_box()
        clip = {"x": 0, "y": 0, "width": 720, "height": box["y"] + box["height"] + 16}
        flame = page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--flame').trim()")
        frames = []

        def advance(ticks):
            for tick in range(1, ticks + 1):
                page.evaluate("window.step()")
                if tick % TICKS_PER_FRAME == 0:
                    frames.append(Image.open(io.BytesIO(page.screenshot(clip=clip))).convert("RGB"))

        def fly(pad, change_at=None, change_to=None):
            page.keyboard.press(str(pad))
            page.keyboard.press("r")
            ticks = 0
            while not page.inner_text("#status"):
                advance(TICKS_PER_FRAME)
                ticks += TICKS_PER_FRAME
                if ticks == change_at:
                    page.keyboard.press(str(change_to))
            assert page.inner_text("#status") == "Landed on the target pad.", page.inner_text("#status")
            advance(HOLD_TICKS)

        fly(2)
        fly(3)
        fly(1, change_at=72, change_to=3)
        browser.close()

    # One palette for every frame keeps the file small and the colors steady.
    # The flame covers few pixels, so it gets its own band or the palette would drop it.
    sample = Image.new("RGB", (frames[0].width, frames[0].height * 3 + 200), flame)
    for i, frame in enumerate((frames[0], frames[len(frames) // 2], frames[-1])):
        sample.paste(frame, (0, i * frames[0].height))
    palette = sample.quantize(colors=GIF_COLORS, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    out = OUT / "three-pads.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
    print(f"{out.name}: {len(frames)} frames, {frames[0].size}, {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    gif() if "--gif" in sys.argv else main()
