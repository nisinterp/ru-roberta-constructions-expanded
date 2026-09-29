"""Validate scored targets and make an explicitly requested cutoff auditable.

A cutoff is allowed only for RNC, only with --cutoff, and only when the current
inputs and outputs exactly match the saved user-requested snapshot.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_snapshot(root=ROOT):
    snapshot = json.loads((root / "results/cutoff_snapshot.json").read_text())
    for relative, entry in snapshot["files"].items():
        if file_hash(root / relative) != entry["sha256"]:
            raise ValueError(f"Cutoff snapshot changed: {relative}")
    return snapshot


def target_key(row):
    return tuple(row[name] for name in ("record", "example_idx", "char_start", "char_end", "type"))


def check_coverage(dataset, input_name, *, allow_cutoff=False, root=ROOT):
    if allow_cutoff and dataset != "rnc":
        raise ValueError("Only the RNC baseline may be incomplete at the cutoff")
    input_path = root / "data" / input_name
    metadata = root / "results" / f"affinity_{dataset}.metadata.json"
    if not metadata.exists() or json.loads(metadata.read_text())["input_sha256"] != file_hash(input_path):
        raise ValueError(f"{dataset} scores do not match the current inputs")
    rows = [json.loads(s) for s in input_path.read_text().splitlines()]
    expected_list = [target_key(row) for row in rows if row["type"] != "other"]
    expected = set(expected_list)
    output_path = root / "results" / f"affinity_{dataset}.jsonl"
    with output_path.open() as source:
        output = [target_key(json.loads(line)) for line in source]
    excluded_path = root / "results" / f"exclusions_{dataset}.jsonl"
    exclusions = (
        [tuple(json.loads(line)["key"]) for line in excluded_path.read_text().splitlines()]
        if excluded_path.exists()
        else []
    )
    excluded = set(exclusions)
    if (
        len(expected) != len(expected_list)
        or len(output) != len(set(output))
        or len(excluded) != len(exclusions)
        or set(output) & excluded
        or not (set(output) | excluded) <= expected
    ):
        raise ValueError(f"Duplicate, conflicting, or unexpected {dataset} targets")
    missing = expected - set(output) - excluded
    if missing and not allow_cutoff:
        raise ValueError(f"Incomplete {dataset} scores: {len(missing)} targets not scored")
    if allow_cutoff:
        verify_snapshot(root)
    return dict(
        expected=len(expected),
        scored=len(output),
        excluded=len(excluded),
        unscored=len(missing),
        eligible=len(expected) - len(excluded),
        complete=not missing,
        scored_fraction=len(output) / (len(expected) - len(excluded)),
    )
