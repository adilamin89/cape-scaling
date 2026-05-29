#!/usr/bin/env python3
"""
test_dashboard.py
=================
Playwright end-to-end tests for the CAPE dashboard (index.html).

Tests every tab, verifies all key numbers are visible, checks for JS errors,
and validates frontier model data.

Usage:
    pip install playwright
    playwright install chromium
    python scripts/test_dashboard.py

    # Test live GitHub Pages version:
    python scripts/test_dashboard.py --url https://adilamin89.github.io/cape-scaling

    # Test local file:
    python scripts/test_dashboard.py  # defaults to ../index.html
"""

import os
import sys
import time
import argparse
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright, expect
except ImportError:
    print("ERROR: playwright not installed.")
    print("Run: pip install playwright && playwright install chromium")
    sys.exit(1)

# ── Ground truth numbers that must appear in the dashboard ─────────────────
REQUIRED_NUMBERS = [
    ("r = −0.989",    "r=−0.989",       "Core anticorrelation"),
    ("3.5B",          "3.5B",           "Critical scale Nc"),
    ("0.629",         "A=0.629",        "SFEE slope A"),
    ("-5.886",        "B=−5.886",       "SFEE intercept B"),
    ("0.000",         "OLMo γ₁₂=0",    "OLMo zero-param confirmation"),
    ("0.238",         "α=0.238",        "Loss exponent alpha"),
    ("0.40",          "β=0.40",         "Order parameter beta"),
    ("1.254",         "d_eff=1.254",    "Effective dimension at Nc"),
    ("1.35",          "Gi=1.35",        "Ginzburg number"),
    ("0.767",         "κ=0.767",        "GL parameter (Type II)"),
    ("3.08",          "χ₂=3.08",        "Susceptibility peak"),
    ("38.8",          "θ*=38.8°",       "TRSB angle"),
    ("10×",           "10x data lever", "Data quality multiplier"),
    ("130B",          "validity end",   "Framework breaks at ~130B"),
    ("5.6%",          "hold-out MAE",   "Cross-family prediction error"),
    ("3.6%",          "ODE error",      "Discovered ODE trajectory error"),
    ("+0.34",         "frontier r",     "Frontier cooperative coupling"),
]

# ── Frontier models that must appear ───────────────────────────────────────
FRONTIER_MODELS = [
    "Sonnet 4.5", "Sonnet 4.6", "Opus 4.6",
    "GPT-5.2 Pro", "Gemini 3 Flash", "Gemini 3 Pro", "Gemini 3.1 Pro",
    "DeepSeek V3.2", "Kimi K2.5", "Qwen3.5-72B",
]

# ── Pythia / base models ────────────────────────────────────────────────────
BASE_MODELS = [
    "Pythia-70M", "Pythia-160M", "Pythia-410M", "Pythia-1B",
    "Pythia-6.9B", "Pythia-12B", "OLMo-7B",
    "Llama-2-7B", "Llama-2-13B", "Llama-2-70B",
    "Mistral-7B", "Gemma-7B",
]

# ── H-field values ──────────────────────────────────────────────────────────
H_FIELDS = [
    ("-13.9", "Sonnet 4.6 local tax excursion"),
    ("2.2",   "Opus 4.6 cooperative"),
    ("4.0",   "GPT-5.2 h-field"),
]

TABS = ["explore", "frontier", "physics", "how", "paper"]


def run_tests(url: str, headless: bool = True):
    results = []

    def ok(name, detail=""):
        results.append(("PASS", name, detail))
        print(f"  ✓  {name}" + (f"  [{detail}]" if detail else ""))

    def fail(name, detail=""):
        results.append(("FAIL", name, detail))
        print(f"  ✗  {name}" + (f"  [{detail}]" if detail else ""))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()

        # Track console errors
        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg) if msg.type == "error" else None)

        print(f"\n{'='*60}")
        print(f"CAPE DASHBOARD TESTS")
        print(f"URL: {url}")
        print(f"{'='*60}\n")

        # ── Load ───────────────────────────────────────────────────────────
        print("[ LOADING ]")
        try:
            page.goto(url, wait_until="networkidle", timeout=15000)
            ok("Page loads without timeout")
        except Exception as e:
            fail("Page loads", str(e))
            browser.close()
            return results

        time.sleep(1)  # Let React render

        # ── Check title ────────────────────────────────────────────────────
        title = page.title()
        if "CAPE" in title or "Alignment" in title or "Phase" in title:
            ok("Page title contains CAPE/Alignment/Phase", title[:50])
        else:
            fail("Page title", f"Got: {title[:50]}")

        # ── Check no crash (app rendered) ────────────────────────────────
        # Try React #root first, then vanilla .app container
        root_el = page.locator("#root")
        app_el = page.locator(".app")
        if root_el.count() > 0:
            root_content = root_el.inner_text(timeout=5000)
        elif app_el.count() > 0:
            root_content = app_el.inner_text(timeout=5000)
        else:
            root_content = page.locator("body").inner_text(timeout=5000)
        if len(root_content) > 100:
            ok("App rendered (has content)")
        else:
            fail("App rendered", f"Only {len(root_content)} chars")

        # ── Tab navigation ─────────────────────────────────────────────────
        print("\n[ TABS ]")
        for tab_id in TABS:
            try:
                # Click tab button — try multiple selectors
                clicked = False
                for selector in [
                    f'[data-tab="{tab_id}"]',
                    f'button:has-text("{tab_id.upper()}")',
                    f'[onclick*="{tab_id}"]',
                ]:
                    try:
                        el = page.locator(selector).first
                        if el.is_visible():
                            el.click()
                            clicked = True
                            break
                    except:
                        pass

                if not clicked:
                    # Try clicking by partial text match
                    tab_labels = {"explore": "EXPLORER", "frontier": "FRONTIER",
                                  "physics": "OBSERV", "how": "HOW IT", "paper": "KEY"}
                    label = tab_labels.get(tab_id, tab_id.upper())
                    page.get_by_text(label, exact=False).first.click()
                    clicked = True

                time.sleep(0.3)
                ok(f"Tab '{tab_id}' clickable")
            except Exception as e:
                fail(f"Tab '{tab_id}' clickable", str(e)[:60])

        # ── KEY RESULTS tab — verify all numbers ──────────────────────────
        print("\n[ KEY NUMBERS — KEY RESULTS TAB ]")
        # Navigate to paper/results tab
        try:
            page.get_by_text("KEY", exact=False).first.click()
            time.sleep(0.5)
        except:
            pass

        full_text = page.content()  # Full HTML
        for text, label, description in REQUIRED_NUMBERS:
            if text in full_text:
                ok(f"{label}", description)
            else:
                fail(f"{label} ('{text}' missing)", description)

        # ── FRONTIER TAB — all 10 models ──────────────────────────────────
        print("\n[ FRONTIER MODELS ]")
        try:
            page.get_by_text("FRONTIER", exact=False).first.click()
            time.sleep(0.5)
        except:
            pass
        frontier_text = page.content()
        for model in FRONTIER_MODELS:
            if model in frontier_text:
                ok(f"Frontier model: {model}")
            else:
                fail(f"Frontier model: {model}")

        # ── H-FIELD VALUES ─────────────────────────────────────────────────
        print("\n[ H-FIELD VALUES ]")
        for val, desc in H_FIELDS:
            if val in frontier_text:
                ok(f"h-field {val}", desc)
            else:
                fail(f"h-field {val}", desc)

        # ── EXPLORER TAB — base models ────────────────────────────────────
        print("\n[ BASE MODELS — EXPLORER TAB ]")
        try:
            page.get_by_text("EXPLORER", exact=False).first.click()
            time.sleep(0.5)
        except:
            pass
        explorer_text = page.content()
        for model in BASE_MODELS:
            if model in explorer_text:
                ok(f"Base model: {model}")
            else:
                fail(f"Base model: {model}")

        # ── GITHUB LINK ────────────────────────────────────────────────────
        print("\n[ LINKS ]")
        if "github.com/adilamin89/cape-scaling" in full_text:
            ok("GitHub link present")
        else:
            fail("GitHub link present")

        if "adilamin@uwm.edu" in full_text or "adil89aminx@gmail.com" in full_text:
            ok("Email contact present")
        else:
            fail("Email contact present")

        if "arXiv" in full_text or "arxiv" in full_text:
            ok("arXiv reference present")
        else:
            fail("arXiv reference present")

        # ── CONSOLE ERRORS ─────────────────────────────────────────────────
        print("\n[ JAVASCRIPT HEALTH ]")
        errors = [e for e in console_errors if "favicon" not in str(e.text).lower()]
        if len(errors) == 0:
            ok("No JavaScript console errors")
        else:
            fail(f"{len(errors)} console errors", errors[0].text[:100] if errors else "")

        # Check canvases rendered
        canvas_count = page.locator("canvas").count()
        if canvas_count >= 3:
            ok(f"Canvas charts rendered ({canvas_count} canvases)")
        else:
            fail(f"Canvas charts ({canvas_count} found, expected ≥3)")

        browser.close()

    # ── Summary ───────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r[0] == "PASS")
    failed = sum(1 for r in results if r[0] == "FAIL")
    total = len(results)

    print(f"\n{'='*60}")
    print(f"RESULT: {passed}/{total} passed  |  {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED ✓")
    else:
        print("FAILED TESTS:")
        for r in results:
            if r[0] == "FAIL":
                print(f"  ✗  {r[1]}" + (f": {r[2]}" if r[2] else ""))
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CAPE dashboard")
    parser.add_argument("--url", default=None,
                        help="URL to test (default: local index.html)")
    parser.add_argument("--live", action="store_true",
                        help="Test adilamin89.github.io/cape-scaling")
    parser.add_argument("--show", action="store_true",
                        help="Show browser (not headless)")
    args = parser.parse_args()

    if args.live:
        url = "https://adilamin89.github.io/cape-scaling"
    elif args.url:
        url = args.url
    else:
        # Local file
        script_dir = Path(__file__).parent
        local_html = (script_dir.parent / "index.html").resolve()
        if not local_html.exists():
            print(f"ERROR: {local_html} not found")
            print("Run from repo root or pass --url")
            sys.exit(1)
        url = f"file://{local_html}"

    results = run_tests(url, headless=not args.show)
    failed = sum(1 for r in results if r[0] == "FAIL")
    sys.exit(1 if failed > 0 else 0)
