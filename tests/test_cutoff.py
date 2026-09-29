"""The cutoff exception must not silently accept changed or duplicated predictions."""

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from study_coverage import check_coverage, file_hash


def frozen_sample(root):
    (root / "data").mkdir()
    (root / "results").mkdir()
    rows = [dict(record=-1, example_idx=i, char_start=0, char_end=1, type="rnc") for i in range(2)]
    source = root / "data/rnc_items.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = root / "results/affinity_rnc.jsonl"
    output.write_text(json.dumps(rows[0]) + "\n")
    (root / "results/affinity_rnc.metadata.json").write_text(json.dumps(dict(input_sha256=file_hash(source))))
    snapshot = dict(files={str(p.relative_to(root)): dict(sha256=file_hash(p)) for p in [source, output]})
    (root / "results/cutoff_snapshot.json").write_text(json.dumps(snapshot))
    return output


def test_partial_coverage_requires_explicit_cutoff(tmp_path):
    frozen_sample(tmp_path)
    with pytest.raises(ValueError, match="Incomplete rnc"):
        check_coverage("rnc", "rnc_items.jsonl", root=tmp_path)
    result = check_coverage("rnc", "rnc_items.jsonl", allow_cutoff=True, root=tmp_path)
    assert result["scored"] == 1 and result["unscored"] == 1
    assert result["complete"] is False


def test_cutoff_rejects_changed_predictions(tmp_path):
    output = frozen_sample(tmp_path)
    row = json.loads(output.read_text())
    row["correct"] = 1
    output.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Cutoff snapshot changed"):
        check_coverage("rnc", "rnc_items.jsonl", allow_cutoff=True, root=tmp_path)


def test_cutoff_rejects_duplicate_predictions(tmp_path):
    output = frozen_sample(tmp_path)
    output.write_text(output.read_text() * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        check_coverage("rnc", "rnc_items.jsonl", allow_cutoff=True, root=tmp_path)


def test_cutoff_cannot_relax_construction_coverage(tmp_path):
    with pytest.raises(ValueError, match="Only the RNC"):
        check_coverage("constructicon", "items.jsonl", allow_cutoff=True, root=tmp_path)


def test_length_cells_are_weighted_within_forms():
    import pandas as pd
    from analyze_cutoff import matched_effect

    # Form a has two length cells, form b has one. Equal cell weighting would
    # give 2/3; the intended equal-form effect is (1 + 0) / 2 = 1/2.
    anchors = pd.DataFrame(dict(form=["a", "a", "b"], length_bin=[0, 1, 0], correct=[1, 1, 0]))
    controls = anchors.assign(correct=0)
    result, _ = matched_effect(anchors, controls, ["form", "length_bin"], "correct")
    assert result["mean_diff"] == 0.5
    assert result["n_pairs"] == 2
