"""Join HF construction boundaries to source slot/anchor annotations by text."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import pandas as pd

from prepare import sentence_key

ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r"[а-яёa-z0-9]+", re.I)


def extract_scope(raw):
    clean, inside, depth = [], [], 0
    for c in raw:
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                return None
        else:
            clean.append(c)
            inside.append(depth > 0)
    if depth or not any(inside):
        return None
    clean = "".join(clean)
    flags = tuple(any(inside[m.start() : m.end()]) for m in WORD.finditer(clean))
    return sentence_key(clean), flags


def row_key(r):
    return f"{r['record']}:{r['example_idx']}:{r['char_start']}:{r['char_end']}"


def main():
    hf = pd.concat([pd.read_parquet(ROOT / f"data/raw/hf/{s}.parquet") for s in ["train", "validation", "test"]])
    spans = defaultdict(set)
    for r in hf.itertuples():
        scope = extract_scope(r.example)
        if scope:
            key, flags = scope
            spans[(int(r.pattern_id), key)].add(flags)
    rows = [json.loads(s) for s in (ROOT / "data/items.jsonl").read_text().splitlines()]
    index, counts, examples = {}, Counter(), defaultdict(set)
    for r in rows:
        key = (r["record"], sentence_key(r["sentence"]))
        candidates = spans.get(key, set())
        status = "unmatched" if not candidates else "ambiguous" if len(candidates) != 1 else "matched"
        within = None
        if status == "matched":
            flags = next(iter(candidates))
            words = list(WORD.finditer(r["sentence"]))
            if len(words) != len(flags):
                raise ValueError("Canonical text join changed word count")
            overlapping = [i for i, w in enumerate(words) if w.start() < r["char_end"] and w.end() > r["char_start"]]
            within = all(flags[i] for i in overlapping) if overlapping else None
        index[row_key(r)] = dict(status=status, within=within)
        counts[f"{r['type']}:{status}:{within}"] += 1
        examples[status].add((r["record"], r["example_idx"]))
    (ROOT / "data/scope_index.json").write_text(json.dumps(index))
    audit = dict(
        word_counts=dict(counts),
        example_counts={k: len(v) for k, v in examples.items()},
        rule="For text-matched HF examples, retain words inside braces; unmatched/ambiguous rows retained with a flag. No scores consulted.",
    )
    (ROOT / "results/span_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
