"""Audit the stopped RNC sample without collecting text or running the model.

Length matching is a sensitivity check, not a correction for missing predictions.
Every estimate remains conditional on the saved, nonrandom cutoff sample.
"""

import json
from pathlib import Path

import pandas as pd

from analyze_expanded import paired_summary
from study_coverage import check_coverage, target_key, verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def describe(values):
    values = pd.Series(values).dropna()
    return dict(
        n=len(values),
        mean=float(values.mean()) if len(values) else None,
        median=float(values.median()) if len(values) else None,
        q25=float(values.quantile(0.25)) if len(values) else None,
        q75=float(values.quantile(0.75)) if len(values) else None,
        minimum=float(values.min()) if len(values) else None,
        maximum=float(values.max()) if len(values) else None,
    )


def matched_effect(anchors, controls, keys, metric):
    """Average matched length cells within a form, then give each form equal weight."""
    a = anchors.groupby(keys)[metric].mean().rename("mean_in")
    b = controls.groupby(keys)[metric].mean().rename("mean_out")
    pairs = pd.concat([a, b], axis=1).dropna()
    by_form = pairs.groupby("form").mean()
    effect = paired_summary(by_form.mean_in - by_form.mean_out)
    return dict(
        **effect,
        matched_cells=len(pairs),
        mean_in=float(by_form.mean_in.mean()) if len(by_form) else None,
        mean_out=float(by_form.mean_out.mean()) if len(by_form) else None,
    ), by_form


def main():
    snapshot = verify_snapshot(ROOT)
    coverage = check_coverage("rnc", "rnc_items.jsonl", allow_cutoff=True, root=ROOT)
    con = pd.read_csv(RES / "construction_scores.csv.gz")
    rnc = pd.read_json(RES / "affinity_rnc.jsonl", lines=True)
    outside = pd.read_json(RES / "affinity_outside.jsonl", lines=True)
    inputs = [json.loads(line) for line in (ROOT / "data/rnc_items.jsonl").read_text().splitlines()]
    exclusions = {tuple(json.loads(line)["key"]) for line in (RES / "exclusions_rnc.jsonl").read_text().splitlines()}
    eligible = [row for row in inputs if target_key(row) not in exclusions]
    expected = pd.Series([r["form"] for r in eligible]).value_counts().rename("eligible")
    scored = rnc.form.value_counts().rename("scored")
    forms = pd.concat([expected, scored], axis=1).fillna(0).astype(int)
    forms["remaining"] = forms.eligible - forms.scored
    forms["complete"] = forms.remaining.eq(0)
    forms.index.name = "form"
    forms.to_csv(RES / "tables/rnc_cutoff_form_coverage.csv")
    completed = set(forms.index[forms.complete])
    scored_keys = {target_key(row) for row in rnc.to_dict("records")}
    # Whitespace length can be calculated for unscored inputs without model inference.
    lengths = pd.DataFrame(
        {
            "scored": [target_key(r) in scored_keys for r in eligible],
            "whitespace_words": [len(r["sentence"].split()) for r in eligible],
        }
    )
    result = dict(
        cutoff_utc=snapshot["cutoff_utc"],
        coverage=coverage,
        forms=dict(
            eligible=len(forms),
            scored=int(forms.scored.gt(0).sum()),
            fully_scored=int(forms.complete.sum()),
            without_scores=int(forms.scored.eq(0).sum()),
        ),
        sentence_lengths=dict(
            scored_rnc_tokens=describe(rnc.sentence_tokens),
            construction_tokens=describe(con.sentence_tokens),
            scored_rnc_whitespace_words=describe(lengths.loc[lengths.scored, "whitespace_words"]),
            unscored_rnc_whitespace_words=describe(lengths.loc[~lengths.scored, "whitespace_words"]),
        ),
        sensitivity={},
        probability_descriptives={},
    )
    anchors = con[con.type.eq("anchor")].copy()
    # Token-length bins retain overlap without treating cells as independent forms.
    bins = [0, 8, 16, 24, 32, 48, 64, 128, 512]
    anchors["length_bin"] = pd.cut(anchors.sentence_tokens, bins, labels=False)
    rnc["length_bin"] = pd.cut(rnc.sentence_tokens, bins, labels=False)
    for label, a, b, keys in [
        ("fully_scored_forms", anchors[anchors.form.isin(completed)], rnc[rnc.form.isin(completed)], ["form"]),
        ("same_form_and_length_bin", anchors, rnc, ["form", "length_bin"]),
        ("same_form_and_exact_token_length", anchors, rnc, ["form", "sentence_tokens"]),
        ("exclude_under_five_tokens", anchors[anchors.sentence_tokens >= 5], rnc[rnc.sentence_tokens >= 5], ["form"]),
    ]:
        result["sensitivity"][label] = {}
        for metric in ["correct", "p_chain"]:
            effect, pairs = matched_effect(a, b, keys, metric)
            result["sensitivity"][label][metric] = effect
            pairs.to_csv(RES / f"tables/cutoff_{label}_{metric}.csv")
    # These occurrence-level summaries complement the form-weighted comparisons.
    for label, frame in [("rnc", rnc), ("outside", outside)]:
        result["probability_descriptives"][label] = {
            metric: describe(frame[metric])
            for metric in ["correct", "p_single", "p_chain", "p_lemma", "log_p_chain", "top5_single"]
            if metric in frame
        }
    length_table = lengths.groupby(["scored", "whitespace_words"]).size().rename("targets").reset_index()
    length_table.to_csv(RES / "tables/rnc_cutoff_length_counts.csv", index=False)
    (RES / "cutoff_analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != "probability_descriptives"}, indent=2))


if __name__ == "__main__":
    main()
