#!/usr/bin/env python3
"""Browser test for lander/index.html, opened from file:// with no server.

Run: .venv/bin/python tests/e2e_lander.py [screenshot_dir]
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGE = (Path(__file__).resolve().parent.parent / "lander" / "index.html").as_uri()
FLIGHT_TIMEOUT_MS = 60_000


def label(page):
    return page.eval_on_selector("#pilot", "e => e.options[e.selectedIndex].text")


def pick(page, name):
    value = page.eval_on_selector("#pilot", "(e, n) => [...e.options].find(o => o.text.startsWith(n)).value", name)
    page.select_option("#pilot", value)


def main(shots):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1100, "height": 1000})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(PAGE)

        options = page.eval_on_selector_all("#pilot option", "os => os.map(o => o.text)")
        print("pilots:", options)
        assert any(o.startswith("You") for o in options) and any(o.startswith("Plain program") for o in options)
        assert page.locator("#pads button").count() == 3

        # Every network pilot and the plain program must finish flights with a real outcome message.
        for name in [o.split(" (")[0] for o in options if not o.startswith(("You", "Random"))]:
            pick(page, name)
            page.wait_for_function("document.getElementById('status').textContent !== ''", timeout=FLIGHT_TIMEOUT_MS)
            print(f"{name}: {page.inner_text('#status')}")
            assert page.locator("#in7").inner_text() in ("0.000", "1.000")
            if shots:
                page.screenshot(path=f"{shots}/{name.replace(' ', '-')}.png", full_page=True)

        # Pad keys move the target, and the one-hot inputs follow.
        pick(page, "Plain program")
        page.keyboard.press("3")
        page.wait_for_function("document.getElementById('in7').textContent === '1.000'")
        assert page.get_attribute("#pads button:nth-child(3)", "aria-pressed") == "true"
        assert page.get_attribute("#pads button:nth-child(1)", "aria-pressed") == "false"

        # Taking over keeps the flight in progress: fuel must not jump back to full.
        page.wait_for_function("Number(document.getElementById('in4').textContent) < 0.9", timeout=FLIGHT_TIMEOUT_MS)
        fuel_before = float(page.inner_text("#in4"))
        page.keyboard.press("ArrowUp")
        page.wait_for_timeout(100)
        assert label(page).startswith("You"), label(page)
        assert float(page.inner_text("#in4")) <= fuel_before, "pressing an engine restarted the flight"

        # Pressing an engine takes over from the pilot and burns fuel; thrust slows the fall.
        page.keyboard.press("r")
        page.keyboard.down("ArrowUp")
        page.wait_for_function("Number(document.getElementById('in3').textContent) > 0.1", timeout=10_000)
        page.keyboard.up("ArrowUp")
        assert label(page).startswith("You"), label(page)
        assert float(page.inner_text("#in4")) < 1
        assert page.locator("#weightsPanel").is_hidden()

        # After choosing a pilot in the list, the arrow keys fly the craft and leave the list alone.
        page.focus("#pilot")
        pick(page, "You")
        page.keyboard.press("r")
        page.wait_for_timeout(100)
        start_vx, start_vy = float(page.inner_text("#in2")), float(page.inner_text("#in3"))
        page.keyboard.down("ArrowUp")
        page.keyboard.down("ArrowRight")
        # A gain of 0.1 always fits inside the world; a fixed speed target ran out of world in 2% of flights.
        page.wait_for_function(f"Number(document.getElementById('in2').textContent) > {start_vx + 0.1}", timeout=10_000)
        assert float(page.inner_text("#in3")) > start_vy - 0.1, "holding two keys must fire both engines in turn"
        page.keyboard.up("ArrowUp")
        page.keyboard.up("ArrowRight")
        assert label(page).startswith("You"), label(page)
        page.keyboard.press("r")

        # With no input, a human flight ends and stays ended until New flight.
        page.wait_for_function("document.getElementById('status').textContent !== ''", timeout=FLIGHT_TIMEOUT_MS)
        ended = page.inner_text("#status")
        print("no input:", ended)
        assert "Landed" not in ended
        page.wait_for_timeout(2000)
        assert page.inner_text("#status") == ended
        page.click("#restart")
        assert page.inner_text("#status") == ""

        # A pad choice survives from one network flight to the next.
        # Pinned randomness: a random pick would choose pad 3, so this cannot pass by luck.
        page.evaluate("() => { window.realRandom = Math.random; Math.random = () => 0.9; }")
        pick(page, "Plain program")
        page.keyboard.press("1")
        page.wait_for_function("document.getElementById('status').textContent !== ''", timeout=FLIGHT_TIMEOUT_MS)
        page.wait_for_function("document.getElementById('status').textContent === ''", timeout=10_000)
        assert page.get_attribute("#pads button:nth-child(1)", "aria-pressed") == "true", "the next flight changed the pad"
        page.evaluate("() => { Math.random = window.realRandom; }")
        pick(page, "You")

        # On-screen engine buttons work with a pointer.
        box = page.locator("#controls [data-key=thrust]").bounding_box()
        fuel = float(page.inner_text("#in4"))
        page.mouse.move(box["x"] + 5, box["y"] + 5)
        page.mouse.down()
        page.wait_for_timeout(500)
        page.mouse.up()
        assert float(page.inner_text("#in4")) < fuel

        # Phone width: no horizontal page scroll.
        page.set_viewport_size({"width": 375, "height": 800})
        page.wait_for_timeout(200)
        overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        assert overflow <= 0, f"page overflows by {overflow}px at phone width"
        if shots:
            page.screenshot(path=f"{shots}/phone.png", full_page=True)

        assert not errors, errors
        browser.close()
    print("e2e ok")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
