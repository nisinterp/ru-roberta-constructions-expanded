"""Within-sentence comparisons with sampled words outside annotated spans."""

import json
import numpy as np
import pandas as pd

from analyze_expanded import ROOT, RES, TABLES, cluster_rate, paired_summary, require_complete
from audit_spans import row_key


def main():
    require_complete("outside", "outside_items.jsonl")
    require_complete("constructicon", "items.jsonl")
    inside = pd.read_json(RES / "affinity_constructicon.jsonl", lines=True)
    scopes = json.loads((ROOT / "data/scope_index.json").read_text())
    inside = inside[[scopes[row_key(r)]["within"] is True for r in inside.to_dict("records")]].copy()
    outside = pd.read_json(RES / "affinity_outside.jsonl", lines=True)
    outside["form"] = outside.text.str.lower().str.replace("ё", "е")
    joined = pd.concat([inside, outside], ignore_index=True)
    summary = {
        "counts": {
            "outside_words": len(outside),
            "outside_sentences": int(outside.sentence_id.nunique()),
            "outside_constructions": int(outside.record.nunique()),
        },
        "rates": {role: cluster_rate(g) for role, g in joined.groupby("type")},
    }
    # Bootstrap construction-level mean sentence differences to respect nesting.
    comparisons = []
    for role in ["anchor", "filler", "all_inside"]:
        a = inside if role == "all_inside" else inside[inside.type == role]
        for matching, keys in [
            ("sentence", ["record", "sentence_id"]),
            ("sentence_and_pos", ["record", "sentence_id", "pos"]),
        ]:
            for metric in ["correct", "p_chain"]:
                pair = (
                    a.groupby(keys)[metric]
                    .mean()
                    .to_frame("inside")
                    .join(outside.groupby(keys)[metric].mean().rename("outside"), how="inner")
                )
                pair["diff"] = pair.inside - pair.outside
                # Give each sentence equal weight, even when it has several matched POS groups.
                sentence_means = pair.groupby(["record", "sentence_id"]).mean()
                construction_means = sentence_means.groupby("record").mean()
                construction_diffs = construction_means["diff"]
                entry = dict(
                    role=role,
                    matching=matching,
                    metric=metric,
                    matched_cells=len(pair),
                    **paired_summary(construction_diffs),
                    mean_in=float(construction_means.inside.mean()),
                    mean_out=float(construction_means.outside.mean()),
                )
                comparisons.append(entry)
                pair.to_csv(TABLES / f"outside_{role}_{matching}_{metric}.csv")
    summary["comparisons"] = comparisons
    summary["interpretation"] = (
        "Outside means outside the annotated target span, not absence of all constructions. "
        "Up to two words uniformly sampled per eligible sentence before scoring. "
        "Comparisons average sentence (or sentence-POS) differences within construction, "
        "then weight constructions equally; CI resamples constructions. "
        "Word identities are not controlled; use RNC matched-form analysis alongside this."
    )
    columns = [
        "record",
        "example_idx",
        "sentence_id",
        "type",
        "form",
        "pos",
        "n_tokens",
        "p_chain",
        "correct",
        "sentence_tokens",
    ]
    outside[columns].to_csv(RES / "outside_scores.csv.gz", index=False)
    # JSON excludes non-finite values for interoperability.
    (RES / "outside_summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda x: float(x) if isinstance(x, np.generic) else x)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
