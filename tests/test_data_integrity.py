"""Tests for alignment and scope rules that directly affect the estimates."""

import json
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_spans import extract_scope, row_key


def test_scope_offsets_and_malformed_braces():
    assert extract_scope("Он {всё-таки пришёл}!") == ("он все таки пришел", (False, True, True, True))
    assert extract_scope("{Он {пришёл}}") == ("он пришел", (True, True))
    assert extract_scope("Он }пришёл{") is None
    assert extract_scope("Он {пришёл") is None
    assert extract_scope("Он пришёл") is None


def test_target_offsets_and_sentence_deduplication():
    rows = [json.loads(s) for s in (ROOT / "data/items.jsonl").read_text().splitlines()]
    owners = {}
    keys = set()
    for row in rows:
        assert row["sentence"][row["char_start"] : row["char_end"]] == row["text"]
        assert row_key(row) not in keys
        keys.add(row_key(row))
        owner = (row["record"], row["example_idx"])
        assert owners.setdefault(row["sentence_id"], owner) == owner


def test_outside_targets_are_outside_and_sampling_is_bounded():
    scopes = json.loads((ROOT / "data/scope_index.json").read_text())
    counts = {}
    for line in (ROOT / "data/outside_items.jsonl").read_text().splitlines():
        row = json.loads(line)
        assert scopes[row_key(row)] == {"status": "matched", "within": False}
        counts[row["sentence_id"]] = counts.get(row["sentence_id"], 0) + 1
    assert counts and max(counts.values()) <= 2


def test_rnc_offsets_are_exact_targets():
    if not (ROOT / "data/rnc_items.jsonl").exists():
        pytest.skip("Live RNC collection has not finished")
    for line in (ROOT / "data/rnc_items.jsonl").read_text().splitlines():
        row = json.loads(line)
        target = row["sentence"][row["char_start"] : row["char_end"]]
        assert target == row["text"]
        assert target.lower().replace("ё", "е") == row["form"].lower().replace("ё", "е")
