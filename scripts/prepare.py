"""Audit both datasets and select a deterministic, larger construction sample."""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream"))
import parse_constructicon as original

original.lemma = lru_cache(maxsize=None)(original.lemma)
original.is_citation_content = lru_cache(maxsize=None)(original.is_citation_content)
FUNC = {"PREP", "CONJ", "PRCL", "NPRO", "INTJ", "PRED", "ADVB_PRO"}


def norm(s):
    return " ".join(s.lower().replace("ё", "е").split())


def sentence_key(s):
    return " ".join(re.findall(r"[а-яёa-z0-9]+", norm(s)))


def anchors(name):
    # Cyrillic lexical material only; Latin grammatical tags are not anchors.
    return sorted(set(re.findall(r"[а-яё]+(?:-[а-яё]+)*", name.lower())))


def write_jsonl(path, rows):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--constructions", type=int, default=300, help="0 selects every eligible construction")
    p.add_argument("--seed", type=int, default=20260926)
    args = p.parse_args()
    all_rows, excluded, records, source_sentences = [], [], {}, set()
    for path in sorted((ROOT / "data/raw/constructicon").glob("*.yml")):
        rec = yaml.safe_load(path.read_text())
        rid = int(rec["record"])
        aa = {norm(a) for a in anchors(rec["name"])}
        al = {original.lemma(a) for a in aa if original.is_citation_content(a)}
        classes = {original.MORPH.parse(a)[0].tag.POS in FUNC for a in aa}
        kind = "mixed" if len(classes) == 2 else "func" if True in classes else "content"
        meta = dict(
            record=rid,
            name=rec["name"],
            anchors=sorted(aa),
            kind=kind,
            syn="; ".join(str(x or "unknown") for x in (rec.get("syntactic_type_of_construction") or ["unknown"])),
            declared_anchor_pos=rec.get("part_of_speech_of_anchor") or [],
        )
        records[rid] = meta
        for ei, raw in enumerate(rec.get("examples") or []):
            clean, slots = original.strip_brackets(raw.strip())
            source_sentences.add((rid, sentence_key(clean)))
            reason = None
            if not slots:
                reason = "no_annotated_slots"
            elif any(c in clean for c in "[]{}"):
                reason = "residual_markup"
            words = original.classify_words(original.split_words(clean), slots, aa, al)
            if not any(w["type"] == "anchor" for w in words):
                reason = reason or "no_anchor_match"
            if reason:
                excluded.append(dict(record=rid, example_idx=ei, reason=reason))
                continue
            for w in words:
                all_rows.append(
                    {
                        **meta,
                        "example_idx": ei,
                        "sentence": clean,
                        "sentence_id": hashlib.sha256(sentence_key(clean).encode()).hexdigest()[:20],
                        **w,
                    }
                )
        source_sentences.add((rid, sentence_key(rec.get("illustration") or "")))

    hf = pd.concat(
        [pd.read_parquet(ROOT / f"data/raw/hf/{s}.parquet").assign(split=s) for s in ["train", "validation", "test"]],
        ignore_index=True,
    )
    hf_keys = [(int(r.pattern_id), sentence_key(re.sub(r"[{}]", "", r.example))) for r in hf.itertuples()]
    hf_audit = dict(
        rows=len(hf),
        constructions=int(hf.pattern_id.nunique()),
        columns=list(hf.columns),
        exact_normalized_source_matches=sum(k in source_sentences for k in hf_keys),
        unique_record_sentence_pairs=len(set(hf_keys)),
        split_sizes=hf.groupby("split").size().to_dict(),
        example_types={str(k): int(v) for k, v in hf.example_type.value_counts().items()},
    )
    write_jsonl(ROOT / "data/items_all.jsonl", all_rows)
    eligible = sorted({r["record"] for r in all_rows})
    chosen = sorted(random.Random(args.seed).sample(eligible, min(args.constructions or len(eligible), len(eligible))))
    selected = set(chosen)
    rows = [r for r in all_rows if r["record"] in selected]
    # Identical sentences across records are not independent observations. Keep one.
    owner = {}
    kept = []
    for r in rows:
        key = (r["record"], r["example_idx"])
        owner.setdefault(r["sentence_id"], key)
        if owner[r["sentence_id"]] == key:
            kept.append(r)
    write_jsonl(ROOT / "data/items.jsonl", kept)
    (ROOT / "data/selected_constructions.json").write_text(
        json.dumps([records[r] for r in chosen], ensure_ascii=False, indent=2)
    )
    write_jsonl(ROOT / "data/parse_exclusions.jsonl", excluded)
    # Refresh the original RNC co-anchor filter against the EXPANDED inventory.
    # Retain a separately flagged subset in which every target construction has co-anchors.
    form_records, record_keys = defaultdict(set), defaultdict(set)
    form_keys = defaultdict(set)
    for r in all_rows:
        if r["type"] == "anchor":
            form = norm(r["text"])
            form_records[form].add(r["record"])
            record_keys[r["record"]].add(r["anchor_key"])
            form_keys[(form, r["record"])].add(r["anchor_key"])
    in_keys = {sentence_key(r["sentence"]) for r in all_rows}
    rnc, seen, rejected = [], set(), Counter()
    selected_forms = {norm(w["text"]) for w in kept if w["type"] == "anchor"}
    for line in (ROOT / "data/raw/rnc_original.jsonl").read_text().splitlines():
        r = json.loads(line)
        form = norm(r["form"])
        if form not in selected_forms:
            rejected["unmatched_form"] += 1
            continue
        key = (sentence_key(r["text"]), r["char_start"], r["char_end"])
        if key in seen or key[0] in in_keys:
            rejected["duplicate_or_overlap"] += 1
            continue
        seen.add(key)
        words = original.split_words(r["text"])
        keys = {norm(w) for _, _, w in words} | {original.lemma(w) for _, _, w in words}
        sets = [record_keys[rid] - form_keys[(form, rid)] for rid in form_records[form]]
        if any(s and s <= keys for s in sets):
            rejected["coanchor_match"] += 1
            continue
        rnc.append(
            {
                **r,
                "filter_applicable": all(bool(s) for s in sets),
                "sentence": r["text"],
                "text": r["text"][r["char_start"] : r["char_end"]],
                "type": "rnc",
                "record": -1,
                "example_idx": len(rnc),
                "sentence_id": hashlib.sha256(key[0].encode()).hexdigest()[:20],
                "source": "upstream_cached_RNC",
                "records": sorted(form_records[form]),
            }
        )
    write_jsonl(ROOT / "data/rnc_items.jsonl", rnc)
    audit = dict(
        seed=args.seed,
        source_constructions=len(records),
        eligible_constructions=len(eligible),
        eligible_examples=len({(r["record"], r["example_idx"]) for r in all_rows}),
        selected_constructions=len(chosen),
        selected_examples=len(owner),
        selected_word_counts=dict(Counter(r["type"] for r in kept)),
        excluded_examples=dict(Counter(r["reason"] for r in excluded)),
        hf=hf_audit,
        rnc_targets=len(rnc),
        rnc_forms=len({r["form"] for r in rnc}),
        rnc_filter_applicable_targets=sum(r["filter_applicable"] for r in rnc),
        rnc_exclusions=dict(rejected),
    )
    (ROOT / "results/data_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
