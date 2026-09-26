"""Parse the 30 selected constructicon records into data/items.jsonl.

For each example sentence:
1. Strip bracket markup `[text]Role`, tracking char spans of slot fillers.
2. Split into words (whitespace-separated, punctuation stripped from ends,
   hyphenated words kept as one word).
3. Classify each word as anchor / filler / other:
   - filler: char span overlaps a slot span.
   - anchor: outside slots and matches one of the construction's fixed
     `anchors` (by form, by lemma for citation-form content anchors, or as
     a hyphenated compound).
   - other: everything else. Not scored by the model.
The >512-token check happens at scoring time against the real tokenizer.

Output: one JSON object per word; only anchor/filler words are scored.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pymorphy3

MORPH = pymorphy3.MorphAnalyzer()

ROOT = Path(__file__).resolve().parent.parent
SELECTED = ROOT / "data" / "selected_constructions.json"
YAML_DIR = ROOT / "data" / "constructicon_yml"
OUT_PATH = ROOT / "data" / "items.jsonl"
EXCLUSIONS_PATH = ROOT / "data" / "parse_exclusions.log"

BRACKET_RE = re.compile(r"\[([^\[\]]+)\](\w+)")
PUNCT_STRIP = " \t\n\r.,;:!?—–‒―−()\"'«»„“”…"


def strip_brackets(text: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Remove [text]Role markup, return (clean_text, [(start, end, role), ...])."""
    parts = []
    slots = []
    last_end = 0
    cur_len = 0
    for m in BRACKET_RE.finditer(text):
        pre = text[last_end : m.start()]
        parts.append(pre)
        cur_len += len(pre)
        inner = m.group(1)
        slot_start = cur_len
        parts.append(inner)
        cur_len += len(inner)
        slot_end = cur_len
        slots.append((slot_start, slot_end, m.group(2)))
        last_end = m.end()
    tail = text[last_end:]
    parts.append(tail)
    clean = "".join(parts)
    return clean, slots


def split_words(clean_text: str) -> list[tuple[int, int, str]]:
    """Whitespace-split, strip leading/trailing punctuation, keep char spans."""
    words = []
    for m in re.finditer(r"\S+", clean_text):
        start, end = m.start(), m.end()
        raw = clean_text[start:end]
        lstripped = len(raw) - len(raw.lstrip(PUNCT_STRIP))
        rstripped = len(raw) - len(raw.rstrip(PUNCT_STRIP))
        new_start = start + lstripped
        new_end = end - rstripped
        if new_start >= new_end:
            continue
        words.append((new_start, new_end, clean_text[new_start:new_end]))
    return words


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and a_end > b_start


CONTENT_POS = {"NOUN", "ADJF", "ADJS", "VERB", "INFN", "PRTF", "PRTS", "GRND"}


def lemma(word: str) -> str:
    return MORPH.parse(word.lower().replace("ё", "е"))[0].normal_form.replace("ё", "е")


def is_citation_content(anchor: str) -> bool:
    """Content-word anchor given in dictionary form (e.g. 'стать', 'легкий')."""
    a = anchor.lower().replace("ё", "е")
    p = MORPH.parse(a)[0]
    return p.tag.POS in CONTENT_POS and p.normal_form.replace("ё", "е") == a


def match_anchor(
    word: str, anchors_lower: set[str], anchor_lemmas: set[str]
) -> tuple[str, str] | None:
    """Return (match mode, anchor key) or None. Mode is 'form' / 'lemma' /
    'compound'; the key identifies which anchor of the construction matched,
    so inflected variants of one anchor share a key.

    The `name` field of this constructicon never uses the '-Tag' notation for
    inflectable anchors (e.g. 'стать', 'счесть', 'легкий' are citation forms
    that surface inflected), so lemma matching is used as a fallback whenever
    exact wordform matching fails. Hyphenated compounds like 'худо-бедно',
    stored as separate anchors, match when every part is an anchor.
    """
    w = word.lower().replace("ё", "е")
    if w in anchors_lower:
        return "form", w
    lem = lemma(w)
    if lem in anchor_lemmas:
        return "lemma", lem
    if "-" in w:
        parts = [p for p in w.split("-") if p]
        if parts and all(
            p in anchors_lower or lemma(p) in anchor_lemmas for p in parts
        ):
            return "compound", w
    return None


def classify_words(
    words: list[tuple[int, int, str]],
    slots: list[tuple[int, int, str]],
    anchors_lower: set[str],
    anchor_lemmas: set[str],
) -> list[dict]:
    out = []
    for start, end, text in words:
        word_slots = [role for (s, e, role) in slots if overlaps(start, end, s, e)]
        match_mode = anchor_key = None
        if word_slots:
            wtype = "filler"
            role = word_slots[0]
        else:
            m = match_anchor(text, anchors_lower, anchor_lemmas)
            if m:
                match_mode, anchor_key = m
            wtype = "anchor" if m else "other"
            role = None
        out.append(
            {
                "char_start": start,
                "char_end": end,
                "text": text,
                "type": wtype,
                "role": role,
                "anchor_match": match_mode,
                "anchor_key": anchor_key,
            }
        )
    return out


def main() -> None:
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    n_items = 0
    n_examples = 0
    n_constructions_no_slots = 0
    exclusions = []

    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for rec in selected:
            record_id = rec["record"]
            anchors_lower = {a.lower().replace("ё", "е") for a in rec["anchors"]}
            anchor_lemmas = {lemma(a) for a in anchors_lower if is_citation_content(a)}
            yml_path = YAML_DIR / f"{record_id}.yml"
            if not yml_path.exists():
                exclusions.append(f"record {record_id}: yaml file missing")
                continue
            import yaml

            data = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
            examples = data.get("examples") or []
            construction_has_slot = False
            for ex_idx, raw_example in enumerate(examples):
                text = raw_example.strip()
                if not text:
                    continue
                clean, slots = strip_brackets(text)
                if not slots:
                    exclusions.append(
                        f"record {record_id} example {ex_idx}: no bracketed slots found"
                    )
                else:
                    construction_has_slot = True
                words = split_words(clean)
                classified = classify_words(words, slots, anchors_lower, anchor_lemmas)
                has_anchor = any(w["type"] == "anchor" for w in classified)
                if not has_anchor:
                    exclusions.append(
                        f"record {record_id} example {ex_idx}: no anchor word matched "
                        f"anchors={sorted(anchors_lower)} in text={clean!r}"
                    )
                n_examples += 1
                for w in classified:
                    n_items += 1
                    out_f.write(
                        json.dumps(
                            {
                                "record": record_id,
                                "name": rec["name"],
                                "kind": rec["kind"],
                                "syn": rec["syn"],
                                "example_idx": ex_idx,
                                "sentence": clean,
                                **w,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            if not construction_has_slot:
                n_constructions_no_slots += 1

    EXCLUSIONS_PATH.write_text("\n".join(exclusions) + "\n", encoding="utf-8")
    print(f"wrote {n_items} words from {n_examples} examples to {OUT_PATH}")
    print(f"constructions with zero slotted examples: {n_constructions_no_slots}")
    print(f"exclusions logged: {len(exclusions)} -> {EXCLUSIONS_PATH}")


if __name__ == "__main__":
    main()
