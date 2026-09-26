"""Fetch out-of-construction baseline sentences from the RNC (НКРЯ) API.

For every unique anchor wordform observed in data/items.jsonl, request
sentences from the main corpus containing that exact form. Raw responses are
cached one file per form in data/raw/rnc/<form>.jsonl; re-running never
re-downloads. Sentences where the form co-occurs with all other anchors of
some construction it belongs to are dropped ("out of construction" rule).

Output: data/rnc_items.jsonl, one row per (sentence, target word).
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pymorphy3
import requests

MORPH = pymorphy3.MorphAnalyzer()

ROOT = Path(__file__).resolve().parent.parent
ITEMS = ROOT / "data" / "items.jsonl"
CACHE_DIR = ROOT / "data" / "raw" / "rnc"
OUT_PATH = ROOT / "data" / "rnc_items.jsonl"
LOG_PATH = ROOT / "data" / "rnc_fetch.log"

API = "https://ruscorpora.ru/api/v1/lex-gramm/concordance"
TARGET_PER_FORM = 50
DOCS_PER_PAGE = 50
MAX_PAGES = 3
PAUSE_S = 1.5


def norm(s: str) -> str:
    return s.lower().replace("ё", "е")


def sentence_keys(text: str) -> set[str]:
    """Wordforms and lemmas of a sentence, for matching anchor keys."""
    keys = set()
    for t in text.split():
        w = norm(t.strip('.,;:!?«»"()—–…'))
        if not w:
            continue
        keys.add(w)
        keys.add(MORPH.parse(w)[0].normal_form.replace("ё", "е"))
    return keys


def load_env_token() -> str:
    token = os.environ.get("RNC_API_TOKEN")
    if token:
        return token
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("RNC_API_TOKEN="):
                return line.split("=", 1)[1].strip()
    sys.exit("RNC_API_TOKEN not found")


def query_page(session: requests.Session, form: str, page: int) -> dict:
    body = {
        "corpus": {"type": "MAIN"},
        "lexGramm": {
            "sectionValues": [
                {
                    "subsectionValues": [
                        {
                            "conditionValues": [
                                {"fieldName": "form", "text": {"v": form}}
                            ]
                        }
                    ]
                }
            ]
        },
        "params": {
            "pageParams": {
                "page": page,
                "docsPerPage": DOCS_PER_PAGE,
                "snippetsPerDoc": 1,
            }
        },
    }
    for attempt in range(5):
        r = session.post(API, json=body, timeout=60)
        if r.status_code == 200:
            return r.json()
        wait = PAUSE_S * (2 ** (attempt + 1))
        print(
            f"  {form} page {page}: HTTP {r.status_code}, retry in {wait:.0f}s",
            flush=True,
        )
        time.sleep(wait)
    raise RuntimeError(
        f"RNC request failed for {form!r} page {page}: HTTP {r.status_code}"
    )


def snippet_to_sentence(snippet: dict) -> tuple[str, int, int, str] | None:
    """Rebuild the sentence containing the hit; return (text, hit_start, hit_end, doc_id)."""
    words = [w for seq in snippet["sequences"] for w in seq["words"]]
    hit_sent = None
    for w in words:
        if w.get("displayParams", {}).get("hit"):
            hit_sent = w.get("source", {}).get("sentId")
            break
    if hit_sent is None:
        return None
    text = ""
    hit_start = hit_end = None
    in_sent = False
    for w in words:
        sid = w.get("source", {}).get("sentId")
        if w["type"] == "WORD":
            in_sent = sid == hit_sent
        if not in_sent:
            continue
        if (
            w["type"] == "WORD"
            and w.get("displayParams", {}).get("hit")
            and hit_start is None
        ):
            hit_start = len(text)
            text += w["text"]
            hit_end = len(text)
        else:
            text += w["text"]
    if hit_start is None:
        return None
    lead = len(text) - len(text.lstrip())
    text = text.strip()
    doc_id = snippet.get("source", {}).get("docSource", {}).get("docId", "")
    return text, hit_start - lead, hit_end - lead, doc_id


def fetch_form(session: requests.Session, form: str) -> list[dict]:
    cache = CACHE_DIR / f"{form}.jsonl"
    if cache.exists():
        return [
            json.loads(line)
            for line in cache.read_text(encoding="utf-8").splitlines()
            if line
        ]
    rows = []
    seen = set()
    for page in range(MAX_PAGES):
        data = query_page(session, form, page)
        time.sleep(PAUSE_S)
        n_docs = 0
        for group in data.get("groups", []):
            for doc in group.get("docs", []):
                n_docs += 1
                for sg in doc.get("snippetGroups", []):
                    for sn in sg.get("snippets", []):
                        res = snippet_to_sentence(sn)
                        if res is None:
                            continue
                        text, s, e, doc_id = res
                        if text in seen:
                            continue
                        seen.add(text)
                        rows.append(
                            {
                                "form": form,
                                "text": text,
                                "char_start": s,
                                "char_end": e,
                                "doc_id": doc_id,
                            }
                        )
        if n_docs < DOCS_PER_PAGE or len(rows) >= TARGET_PER_FORM * 2:
            break
    cache.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    return rows


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    items = [
        json.loads(line) for line in ITEMS.read_text(encoding="utf-8").splitlines()
    ]

    # observed anchor forms; for each, the constructions it anchors and which
    # anchor key it realises there (inflected variants of one anchor share a key)
    form_records: dict[str, set[int]] = defaultdict(set)
    form_key: dict[tuple[str, int], str] = {}
    record_keys: dict[int, set[str]] = defaultdict(set)
    for it in items:
        if it["type"] == "anchor":
            f = norm(it["text"])
            form_records[f].add(it["record"])
            form_key[(f, it["record"])] = it["anchor_key"]
            record_keys[it["record"]].add(it["anchor_key"])

    forms = sorted(form_records)
    print(f"{len(forms)} unique observed anchor forms")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {load_env_token()}"

    log = []
    n_out = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for i, form in enumerate(forms, 1):
            try:
                rows = fetch_form(session, form)
            except RuntimeError as exc:
                log.append(f"{form}: FAILED {exc}")
                print(f"[{i}/{len(forms)}] {form}: FAILED", flush=True)
                continue
            # out-of-construction filter
            other_sets = [
                record_keys[r] - {form_key[(form, r)]} for r in form_records[form]
            ]
            other_sets = [s for s in other_sets if s]
            kept, dropped = [], 0
            for r in rows:
                toks = sentence_keys(r["text"])
                if any(s <= toks for s in other_sets):
                    dropped += 1
                    continue
                kept.append(r)
            kept = kept[:TARGET_PER_FORM]
            single_anchor = not other_sets
            for r in kept:
                out.write(
                    json.dumps(
                        {
                            **r,
                            "records": sorted(form_records[form]),
                            "filter_applicable": not single_anchor,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            n_out += len(kept)
            msg = (
                f"{form}: fetched={len(rows)} dropped_in_construction={dropped} kept={len(kept)} "
                f"filter={'n/a (single-anchor)' if single_anchor else 'applied'}"
            )
            log.append(msg)
            print(f"[{i}/{len(forms)}] {msg}", flush=True)
    LOG_PATH.write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"wrote {n_out} sentences to {OUT_PATH}")


if __name__ == "__main__":
    main()
