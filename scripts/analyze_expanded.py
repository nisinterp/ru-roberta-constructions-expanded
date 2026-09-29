"""Compare lexical roles and matched forms using construction-aware uncertainty.

Complete coverage is required unless the explicit, hash-verified RNC cutoff is
requested. Bootstrap units are constructions or matched forms, never raw words.
"""

import json
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream"))
import analyze as original
from audit_spans import row_key

RES = ROOT / "results"
TABLES = RES / "tables"
SEED = 20260926
NBOOT = 10000


def require_complete(dataset, input_name):
    """Keep strict completeness as the default for ordinary experiment runs."""
    from study_coverage import check_coverage

    return check_coverage(dataset, input_name, root=ROOT)


def boot_mean(values):
    """Percentile interval for equally weighted paired construction/form effects."""
    x = np.asarray(values, dtype=float)
    if not len(x):
        return [None, None]
    rng = np.random.default_rng(SEED)
    means = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(NBOOT)]
    return np.quantile(means, [0.025, 0.975]).tolist()


def cluster_rate(df, col="correct", cluster="record"):
    """Resample whole clusters, retaining occurrence weighting within each draw."""
    sub = df[df[col].notna()]
    if not len(sub):
        return dict(n=0, clusters=0, mean=None, ci95=[None, None])
    g = sub.groupby(cluster)[col].agg(["sum", "count"])
    rng = np.random.default_rng(SEED)
    arr = g.to_numpy()
    estimates = []
    for _ in range(NBOOT):
        sums = arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
        estimates.append(sums[0] / sums[1])
    return dict(
        n=len(sub), clusters=len(g), mean=float(sub[col].mean()), ci95=np.quantile(estimates, [0.025, 0.975]).tolist()
    )


def paired_summary(d):
    """Summarize paired effects; ties stay in the mean but leave the sign test."""
    d = np.asarray(d, dtype=float)
    nz = d[d != 0]
    return dict(
        n_pairs=len(d),
        mean_diff=float(d.mean()) if len(d) else None,
        ci95=boot_mean(d),
        positive=int((d > 0).sum()),
        ties=int((d == 0).sum()),
        wilcoxon_p=float(stats.wilcoxon(d).pvalue) if len(nz) >= 5 else None,
        sign_p=float(stats.binomtest((nz > 0).sum(), len(nz)).pvalue) if len(nz) else None,
    )


def pos_model(df, formula, cluster):
    fit = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df[cluster]})
    conf = fit.conf_int()
    return fit, pd.DataFrame(
        dict(
            term=fit.params.index,
            coefficient=fit.params.values,
            se=fit.bse.values,
            p=fit.pvalues.values,
            ci_low=conf[0].values,
            ci_high=conf[1].values,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", action="store_true", help="Analyze the verified user-requested RNC cutoff")
    parser.add_argument(
        "--skip-legacy", action="store_true", help="Skip duplicate upstream analysis; primary statistics are unchanged"
    )
    args = parser.parse_args()
    from study_coverage import check_coverage

    TABLES.mkdir(exist_ok=True)
    collection = json.loads((RES / "rnc_collection.json").read_text())
    input_hash = hashlib.sha256((ROOT / "data/items.jsonl").read_bytes()).hexdigest()
    if collection["source"] != "RNC_live_API" or collection["input_sha256"] != input_hash:
        raise ValueError("Collect a complete live RNC baseline for the current inventory first")
    require_complete("constructicon", "items.jsonl")
    rnc_coverage = check_coverage("rnc", "rnc_items.jsonl", allow_cutoff=args.cutoff, root=ROOT)
    con = pd.read_json(RES / "affinity_constructicon.jsonl", lines=True)
    rnc = pd.read_json(RES / "affinity_rnc.jsonl", lines=True)
    scopes = json.loads((ROOT / "data/scope_index.json").read_text())
    con["scope_status"] = [scopes[row_key(r)]["status"] for r in con.to_dict("records")]
    con["within_scope"] = [scopes[row_key(r)]["within"] for r in con.to_dict("records")]
    raw_con = con.copy()
    con = con[con.within_scope != False].copy()  # noqa: E712
    con["form"] = con.text.str.lower().str.replace("ё", "е")
    for df in (con, rnc):
        df["pos_class"] = np.where(df.pos.isin(original.FUNC_POS), "func", "content")
    summary = {
        "analysis_mode": "user_requested_cutoff" if args.cutoff else "complete_run",
        "rnc_scoring_coverage": rnc_coverage,
        "rnc_collection": {k: v for k, v in collection.items() if k != "forms"},
        "counts": dict(
            constructions=int(con.record.nunique()),
            examples=int(con[["record", "example_idx"]].drop_duplicates().shape[0]),
            anchors=int((con.type == "anchor").sum()),
            fillers=int((con.type == "filler").sum()),
            rnc_targets=len(rnc),
            rnc_forms=int(rnc.form.nunique()),
        ),
    }
    summary["scope_filter"] = dict(
        excluded_scored_targets=len(raw_con) - len(con), retained_unmatched=int((con.scope_status != "matched").sum())
    )
    # Reproduce the source pipeline, with its output in a separate namespace.
    original.ROOT = ROOT
    original.RES = RES
    original.TABLES = RES / "original_tables"
    if not args.skip_legacy:
        original.main()
        (RES / "summary.json").rename(RES / "original_metrics.json")
    for metric in ["correct", "p_chain", "p_single", "top5_single"]:
        entry = {}
        for role in ["anchor", "filler"]:
            entry[role] = cluster_rate(con[con.type == role], metric)
        paired = con.groupby(["record", "type"])[metric].mean().unstack().dropna()
        entry["paired_construction_difference"] = paired_summary(paired.anchor - paired.filler)
        paired.assign(diff=paired.anchor - paired.filler).to_csv(TABLES / f"construction_{metric}.csv")
        summary[metric] = entry
    summary["probability_descriptives"] = {}
    for metric in ["p_single", "p_chain", "p_lemma", "log_p_chain"]:
        summary["probability_descriptives"][metric] = {}
        for role in ["anchor", "filler"]:
            values = con.loc[con.type == role, metric].dropna()
            summary["probability_descriptives"][metric][role] = dict(
                n=len(values),
                mean=float(values.mean()) if len(values) else None,
                median=float(values.median()) if len(values) else None,
                q25=float(values.quantile(0.25)) if len(values) else None,
                q75=float(values.quantile(0.75)) if len(values) else None,
            )
    summary["tokenization"] = {
        role: {
            label: cluster_rate(group[group.n_tokens.gt(1) == multi])
            for label, multi in [("single", False), ("multi", True)]
        }
        for role, group in con.groupby("type")
    }
    pos_rows = []
    for (pos, role), g in con.groupby(["pos", "type"]):
        r = cluster_rate(g)
        pos_rows.append(
            dict(
                pos=pos,
                role=role,
                n=r["n"],
                constructions=r["clusters"],
                accuracy=r["mean"],
                ci_low=r["ci95"][0],
                ci_high=r["ci95"][1],
                mean_affinity=float(g.p_chain.mean()),
                single_accuracy=float(g[g.n_tokens == 1].correct.mean()),
            )
        )
    pd.DataFrame(pos_rows).to_csv(TABLES / "pos_accuracy.csv", index=False)
    matched_pos = []
    for pos, g in con.groupby("pos"):
        pair = g.groupby(["record", "type"]).correct.mean().unstack().reindex(columns=["anchor", "filler"]).dropna()
        if len(pair):
            s = paired_summary(pair.anchor - pair.filler)
            matched_pos.append(dict(pos=pos, **s))
    pos_tests = pd.DataFrame(matched_pos)
    valid = pos_tests.sign_p.notna()
    pos_tests.loc[valid, "sign_p_holm"] = multipletests(pos_tests.loc[valid, "sign_p"], method="holm")[1]
    pos_tests.to_csv(TABLES / "within_pos_construction_differences.csv", index=False)
    # LPM coefficients are percentage-point associations, not causal effects.
    anc = con[con.type == "anchor"].copy()
    counts = anc.pos.value_counts()
    anc["pos_group"] = anc.pos.where(anc.pos.map(counts) >= 30, "OTHER_RARE")
    anc["log_frequency"] = np.log1p(anc.form.map(con.form.value_counts()))
    fit, coefs = pos_model(
        anc,
        "correct ~ C(pos_group) + np.log1p(n_tokens) + np.log1p(sentence_tokens) + word_chars + log_frequency",
        "record",
    )
    coefs.to_csv(TABLES / "anchor_pos_adjusted_model.csv", index=False)
    ix = [i for i, s in enumerate(fit.params.index) if s.startswith("C(pos_group)")]
    restriction = np.eye(len(fit.params))[ix]
    test = fit.wald_test(restriction, scalar=True)
    summary["pos_association"] = dict(
        model="linear probability, construction-cluster robust SE",
        n=int(fit.nobs),
        r_squared=float(fit.rsquared),
        joint_pos_wald_p=float(test.pvalue),
        joint_pos_statistic=float(test.statistic),
        rare_pos_threshold=30,
        caveat="Observed associations; sample word frequency is not an independent corpus frequency estimate; POS tags are context-free pymorphy3.",
    )
    con["is_anchor"] = (con.type == "anchor").astype(int)
    con["log_frequency"] = np.log1p(con.form.map(con.form.value_counts()))
    _, coefs = pos_model(
        con,
        "correct ~ is_anchor + C(pos) + np.log1p(n_tokens) + np.log1p(sentence_tokens) + word_chars + log_frequency",
        "record",
    )
    coefs.to_csv(TABLES / "anchor_advantage_adjusted_model.csv", index=False)
    summary["adjusted_anchor_advantage"] = coefs[coefs.term == "is_anchor"].iloc[0].to_dict()
    # Match form identity before comparing contexts. Context-free POS matching
    # cannot reliably disambiguate homographs and is only a limited sensitivity check.
    summary["in_vs_out"] = {}
    for label, c, r, keys in [
        ("all_cached_controls", con, rnc, ["form"]),
        ("filter_applicable", con, rnc[rnc.filter_applicable], ["form"]),
        ("same_form_and_pos", con, rnc, ["form", "pos"]),
        ("at_least_five_each", con, rnc, ["form"]),
    ]:
        summary["in_vs_out"][label] = {}
        for metric in ["correct", "p_chain", "p_single", "p_lemma"]:
            a = c[c.type == "anchor"].groupby(keys)[metric].agg(["mean", "count"])
            b = r.groupby(keys)[metric].agg(["mean", "count"])
            pair = a.join(b, lsuffix="_in", rsuffix="_out", how="inner").dropna()
            if label == "at_least_five_each":
                pair = pair[(pair.count_in >= 5) & (pair.count_out >= 5)]
            pair["diff"] = pair.mean_in - pair.mean_out
            pair.to_csv(TABLES / f"inout_{label}_{metric}.csv")
            summary["in_vs_out"][label][metric] = {
                **paired_summary(pair["diff"]),
                "mean_in": float(pair.mean_in.mean()) if len(pair) else None,
                "mean_out": float(pair.mean_out.mean()) if len(pair) else None,
            }
    summary["kind_accuracy"] = {
        str(k): {role: cluster_rate(g[g.type == role]) for role in ["anchor", "filler"]} for k, g in con.groupby("kind")
    }
    summary["single_token_accuracy"] = {
        role: cluster_rate(con[(con.type == role) & (con.n_tokens == 1)]) for role in ["anchor", "filler"]
    }
    summary["pos_class_accuracy"] = {
        str(k): {role: cluster_rate(g[g.type == role]) for role in ["anchor", "filler"]}
        for k, g in con.groupby("pos_class")
    }
    (TABLES / "qualitative.md").write_text(original.qualitative(con), encoding="utf-8")
    summary["scope_sensitivity"] = {}
    for label, frame in [("original_rule", raw_con), ("hf_matched_only", con[con.scope_status == "matched"])]:
        pair = frame.groupby(["record", "type"]).correct.mean().unstack().dropna()
        summary["scope_sensitivity"][label] = paired_summary(pair.anchor - pair.filler)
    # Preserve a compact, redistributable word-level numeric table without corpus sentence text.
    columns = [
        "record",
        "example_idx",
        "char_start",
        "char_end",
        "sentence_id",
        "type",
        "form",
        "pos",
        "n_tokens",
        "p_single",
        "p_chain",
        "log_p_chain",
        "p_lemma",
        "correct",
        "top5_single",
        "sentence_tokens",
        "word_chars",
        "scope_status",
    ]
    con[columns].to_csv(RES / "construction_scores.csv.gz", index=False)
    rnc[[c for c in columns + ["doc_id", "filter_applicable"] if c in rnc]].to_csv(
        RES / "rnc_scores.csv.gz", index=False
    )
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
