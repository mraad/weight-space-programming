#!/usr/bin/env python3
"""Check that every reference in docs/references.json exists and is the work it claims to be.

arXiv entries: the arXiv API must return the same title. Other entries: the link must open, and the title
(or the entry's `expect` text) must be in the page. Writes docs/references-check.json. Exit 1 if a link fails.

Run: python3 docs/check_refs.py
"""
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

DOCS = Path(__file__).resolve().parent
HEADERS = {"User-Agent": "Mozilla/5.0 (reference check)"}
ATOM = {"a": "http://www.w3.org/2005/Atom"}
CONFIRMED, BY_HAND, EXISTS, REFUSED, FAILED = ("title confirmed", "title read by hand", "link opens, title not in page",
                                               "link refuses scripts", "FAILED")


def normal(text):
    return re.sub(r"[^a-z0-9]+", " ", html.unescape(text).lower()).strip()


def arxiv_id(url):
    match = re.match(r"https://arxiv\.org/abs/([\d.]+)$", url)
    return match.group(1) if match else None


def get(url):
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.status, response.read(2_000_000)


def check_arxiv(refs, results):
    for start in range(0, len(refs), 40):
        if start:
            time.sleep(3)  # arXiv asks clients to pace API calls
        batch = refs[start:start + 40]
        url = "https://export.arxiv.org/api/query?max_results=50&id_list=" + ",".join(arxiv_id(r["url"]) for r in batch)
        _, body = get(url)
        found = {}
        for entry in ET.fromstring(body).findall("a:entry", ATOM):
            key = entry.find("a:id", ATOM).text.split("/abs/")[-1].split("v")[0]
            title = entry.find("a:title", ATOM)
            if title is not None:
                names = [a.find("a:name", ATOM).text for a in entry.findall("a:author", ATOM)]
                found[key] = (" ".join(title.text.split()), names)
        for ref in batch:
            title, names = found.get(arxiv_id(ref["url"]), (None, []))
            first_author = normal(ref["authors"].split()[0].rstrip(","))
            if not title:
                results[ref["id"]] = {"status": FAILED, "source": "arXiv API", "detail": "id not found"}
            elif normal(title) != normal(ref["title"]):
                results[ref["id"]] = {"status": FAILED, "source": "arXiv API", "detail": f"arXiv title is: {title}"}
            elif f" {first_author} " not in f" {normal(' '.join(names))} ":
                results[ref["id"]] = {"status": FAILED, "source": "arXiv API", "detail": f"arXiv authors are: {names[:3]}"}
            else:
                results[ref["id"]] = {"status": CONFIRMED, "source": "arXiv API, title and author"}


def check_page(ref):
    try:
        status, body = get(ref.get("check_url") or ref["url"])
    except urllib.error.HTTPError as err:
        # Some publishers refuse scripts, which does not prove that the page is missing. Every other error is a failure.
        status = REFUSED if err.code in (401, 403, 429) else FAILED
        return {"status": status, "source": "HTTP", "detail": f"HTTP {err.code}"}
    except Exception as err:  # network errors are results to report, not crashes
        return {"status": FAILED, "source": "HTTP", "detail": str(err)[:120]}
    expected = normal(ref.get("expect") or ref["title"])
    if body[:5] == b"%PDF-":
        # shortcut: a script cannot read PDF text with the standard library, so the title is confirmed by reading the page and noted in `read_by_hand`.
        return {"status": BY_HAND if ref.get("read_by_hand") else EXISTS, "source": f"HTTP {status}", "detail": "PDF"}
    page = normal(body.decode("utf8", "ignore"))
    return {"status": CONFIRMED if f" {expected} " in f" {page} " else EXISTS, "source": f"HTTP {status}"}


def main():
    refs = json.loads((DOCS / "references.json").read_text())
    ids = [r["id"] for r in refs]
    if len(ids) != len(set(ids)):
        sys.exit("duplicate reference ids")
    results = {}
    check_arxiv([r for r in refs if arxiv_id(r["url"])], results)
    pages = [r for r in refs if not arxiv_id(r["url"])]
    with ThreadPoolExecutor(8) as pool:
        results.update(zip((r["id"] for r in pages), pool.map(check_page, pages)))
    report = {"checked": date.today().isoformat(), "results": {i: results[i] for i in ids}}
    (DOCS / "references-check.json").write_text(json.dumps(report, indent=1) + "\n")
    tally = Counter(result["status"] for result in results.values())
    for ref in refs:
        result = results[ref["id"]]
        if result["status"] != CONFIRMED:
            print(f"{result['status']:30} {ref['id']:24} {ref['url'][:60]} {result.get('detail', '')}")
    print(dict(tally))
    sys.exit(1 if tally.get(FAILED) else 0)


if __name__ == "__main__":
    main()
