"""Collect ordinary corpus contexts for every observed anchor form.

The controls match word identity. They are not a random sample of all Russian
sentences, and the automatic filter cannot prove that they contain no construction.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import requests

from prepare import norm, sentence_key, write_jsonl, original

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream"))
import fetch_rnc as api

ORIGINAL_SNIPPET_PARSER = api.snippet_to_sentence
PARSER_EXCLUSIONS = Counter()


def parse_snippet(snippet):
    """Reject incomplete API snippets instead of inventing missing sentence text."""
    words = [word for sequence in snippet["sequences"] for word in sequence["words"]]
    if any(not isinstance(word.get("text"), str) for word in words):
        PARSER_EXCLUSIONS["snippet_missing_text"] += 1
        return None
    return ORIGINAL_SNIPPET_PARSER(snippet)


def load_token():
    """Read the credential without printing or copying it into experiment outputs."""
    token = os.environ.get("RNC_API_TOKEN", "").strip()
    path = ROOT / ".env"
    if not token and path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("RNC_API_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("\"'")
    if not token:
        raise SystemExit("Set RNC_API_TOKEN or add it to the ignored .env file, then rerun.")
    return token


def filter_rows(rows, form, other_sets, construction_sentences, seen):
    """Reject wrong hits, reused sentences, and detectable construction contexts."""
    kept, rejected = [], Counter()
    for row in rows:
        sentence = row["text"]
        target = sentence[row["char_start"]:row["char_end"]]
        key = sentence_key(sentence)
        if norm(target) != form:
            rejected["wrong_form_or_offset"] += 1
            continue
        if key in construction_sentences or (form, key) in seen:
            rejected["duplicate_or_overlap"] += 1
            continue
        words = original.split_words(sentence)
        keys = {norm(w) for _, _, w in words} | {original.lemma(w) for _, _, w in words}
        if any(s and s <= keys for s in other_sets):
            rejected["coanchor_match"] += 1
            continue
        seen.add((form, key))
        kept.append(row)
    return kept, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-form", type=int, default=50)
    parser.add_argument("--max-pages", type=int, default=3)
    args = parser.parse_args()
    if args.per_form < 1 or args.max_pages < 1:
        parser.error("Collection limits must be positive")
    token = load_token()
    items_path = ROOT / "data/items.jsonl"
    items = [json.loads(x) for x in items_path.read_text().splitlines()]
    inventory = [json.loads(x) for x in (ROOT / "data/items_all.jsonl").read_text().splitlines()]
    scopes = json.loads((ROOT / "data/scope_index.json").read_text())
    from audit_spans import row_key
    forms = sorted({norm(r["text"]) for r in items if r["type"] == "anchor"
                    and scopes[row_key(r)]["within"] is not False})
    form_records, record_keys, form_keys = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in inventory:
        if row["type"] == "anchor":
            form, record = norm(row["text"]), row["record"]
            form_records[form].add(record)
            record_keys[record].add(row["anchor_key"])
            form_keys[form, record].add(row["anchor_key"])
    sentences = {sentence_key(r["sentence"]) for r in inventory}
    # A configuration-specific cache makes interrupted collection resumable.
    api.CACHE_DIR = ROOT / f"data/raw/rnc_live_{args.per_form}_{args.max_pages}"
    api.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    api.TARGET_PER_FORM, api.MAX_PAGES = args.per_form, args.max_pages
    api.snippet_to_sentence = parse_snippet
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    output, coverage, seen = [], [], set()
    for index, form in enumerate(forms, 1):
        # Stop on a failed request. Never publish an apparently complete partial baseline.
        PARSER_EXCLUSIONS.clear()
        rows = api.fetch_form(session, form)
        # Keep exclusion counts alongside the cache so resuming preserves this audit.
        parse_audit = api.CACHE_DIR / f"{form}.audit.json"
        if not parse_audit.exists():
            parse_audit.write_text(json.dumps(dict(PARSER_EXCLUSIONS)))
        parse_exclusions = json.loads(parse_audit.read_text())
        sets = [record_keys[r] - form_keys[form, r] for r in form_records[form]]
        kept, rejected = filter_rows(rows, form, sets, sentences, seen)
        kept = kept[:args.per_form]
        for row in kept:
            sentence = row["text"]
            output.append({**row, "sentence": sentence,
                           "text": sentence[row["char_start"]:row["char_end"]],
                           "record": -1, "example_idx": len(output), "type": "rnc",
                           "sentence_id": hashlib.sha256(sentence_key(sentence).encode()).hexdigest()[:20],
                           "filter_applicable": bool(sets) and all(bool(s) for s in sets),
                           "source": "RNC_live_API", "records": sorted(form_records[form])})
        coverage.append(dict(form=form, fetched=len(rows), retained=len(kept), exclusions=dict(rejected),
                             parser_exclusions=parse_exclusions))
        print(f"{index}/{len(forms)} forms; {len(output)} retained targets", flush=True)
    audit = dict(source="RNC_live_API", completed_utc=datetime.now(timezone.utc).isoformat(),
                 input_sha256=hashlib.sha256(items_path.read_bytes()).hexdigest(),
                 requested_forms=len(forms), covered_forms=sum(r["retained"] > 0 for r in coverage),
                 targets=len(output), per_form=args.per_form, max_pages=args.max_pages, forms=coverage)
    if not output:
        raise RuntimeError("No usable RNC targets; inspect API responses before continuing")
    # Write only after every form has been attempted successfully.
    target = ROOT / "data/rnc_items.jsonl"
    temporary = target.with_suffix(".partial")
    write_jsonl(temporary, output)
    temporary.replace(target)
    (ROOT / "results/rnc_collection.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
