#!/usr/bin/env python3
"""Capture the page screenshots for the documentation into docs/img.

Run: .venv/bin/python docs/capture.py
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
PAGE = (ROOT / "lander" / "index.html").as_uri()
OUT = ROOT / "docs" / "img"
OUT.mkdir(exist_ok=True)
JPEG = {"type": "jpeg", "quality": 78}


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


if __name__ == "__main__":
    main()
