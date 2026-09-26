"""Statistical analysis of global affinity: anchors vs fillers, in vs out of
construction, multitoken effects, slices, qualitative extremes.

Writes CSV/Markdown tables to results/tables/ and headline numbers to
results/summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TABLES = RES / "tables"
N_BOOT = 10_000
RNG = np.random.default_rng(20250923)

FUNC_POS = {"PREP", "CONJ", "PRCL", "NPRO", "INTJ", "PRED", "ADVB_PRO"}


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """P(X > Y) - P(X < Y), computed via Mann-Whitney U."""
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return 2 * u / (len(x) * len(y)) - 1


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = pd.read_json(RES / "affinity_constructicon.jsonl", lines=True)
    rnc_path = RES / "affinity_rnc.jsonl"
    rnc = pd.read_json(rnc_path, lines=True) if rnc_path.exists() else pd.DataFrame()
    con["form"] = con["text"].str.lower().str.replace("ё", "е")
    con["pos_class"] = np.where(con["pos"].isin(FUNC_POS), "func", "content")
    return con, rnc


def pooled_test(df: pd.DataFrame, col: str) -> dict:
    a = df.loc[(df.type == "anchor") & df[col].notna(), col].to_numpy()
    f = df.loc[(df.type == "filler") & df[col].notna(), col].to_numpy()
    mw = stats.mannwhitneyu(a, f, alternative="two-sided")
    return {
        "n_anchor": len(a),
        "n_filler": len(f),
        "median_anchor": float(np.median(a)),
        "median_filler": float(np.median(f)),
        "mean_anchor": float(a.mean()),
        "mean_filler": float(f.mean()),
        "share_gt_05_anchor": float((a > 0.5).mean()),
        "share_gt_05_filler": float((f > 0.5).mean()),
        "mannwhitney_U": float(mw.statistic),
        "mannwhitney_p": float(mw.pvalue),
        "cliffs_delta": cliffs_delta(a, f),
    }


def construction_diffs(df: pd.DataFrame, col: str) -> pd.DataFrame:
    sub = df[df.type.isin(["anchor", "filler"]) & df[col].notna()]
    g = sub.groupby(["record", "type"])[col].mean().unstack()
    g = g.dropna(subset=[c for c in ("anchor", "filler") if c in g])
    g["diff"] = g["anchor"] - g["filler"]
    return g


def bootstrap_by_construction(df: pd.DataFrame, col: str) -> dict:
    """Resample constructions with replacement; CI for Cliff's delta and for
    the mean construction-level difference."""
    sub = df[df.type.isin(["anchor", "filler"]) & df[col].notna()]
    groups = {
        rec: (
            g.loc[g.type == "anchor", col].to_numpy(),
            g.loc[g.type == "filler", col].to_numpy(),
        )
        for rec, g in sub.groupby("record")
    }
    recs = list(groups)
    diffs = construction_diffs(df, col)["diff"]
    deltas, mean_diffs = [], []
    for _ in range(N_BOOT):
        sample = RNG.choice(recs, size=len(recs), replace=True)
        a = np.concatenate([groups[r][0] for r in sample])
        f = np.concatenate([groups[r][1] for r in sample])
        if len(a) and len(f):
            deltas.append(cliffs_delta(a, f))
        d = diffs.reindex(sample).dropna()
        if len(d):
            mean_diffs.append(d.mean())
    return {
        "cliffs_delta_ci95": [
            float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)),
        ],
        "mean_constr_diff_ci95": [
            float(np.percentile(mean_diffs, 2.5)),
            float(np.percentile(mean_diffs, 97.5)),
        ],
    }


def construction_level(df: pd.DataFrame, col: str) -> dict:
    d = construction_diffs(df, col)["diff"]
    n_pos = int((d > 0).sum())
    n = int(len(d))
    sign = stats.binomtest(n_pos, n, 0.5) if n else None
    return {
        "n_constructions": n,
        "n_diff_positive": n_pos,
        "share_diff_positive": n_pos / n if n else float("nan"),
        "mean_constr_diff": float(d.mean()),
        "median_constr_diff": float(d.median()),
        "sign_test_p": float(sign.pvalue) if sign else float("nan"),
    }


def slice_table(df: pd.DataFrame, col: str, by: str) -> pd.DataFrame:
    rows = []
    sub = df[df.type.isin(["anchor", "filler"]) & df[col].notna()]
    for key, g in sub.groupby(by):
        a = g.loc[g.type == "anchor", col]
        f = g.loc[g.type == "filler", col]
        rows.append(
            {
                by: key,
                "n_anchor": len(a),
                "n_filler": len(f),
                "median_anchor": a.median() if len(a) else np.nan,
                "median_filler": f.median() if len(f) else np.nan,
                "share_gt_05_anchor": (a > 0.5).mean() if len(a) else np.nan,
                "share_gt_05_filler": (f > 0.5).mean() if len(f) else np.nan,
                "cliffs_delta": cliffs_delta(a.to_numpy(), f.to_numpy())
                if len(a) and len(f)
                else np.nan,
                "mannwhitney_p": stats.mannwhitneyu(a, f).pvalue
                if len(a) >= 3 and len(f) >= 3
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def in_vs_out(
    con: pd.DataFrame, rnc: pd.DataFrame, col: str
) -> tuple[pd.DataFrame, dict]:
    a = (
        con[(con.type == "anchor") & con[col].notna()]
        .groupby("form")[col]
        .agg(["mean", "count"])
    )
    r = rnc[rnc[col].notna()].groupby("form")[col].agg(["mean", "count"])
    m = a.join(r, lsuffix="_in", rsuffix="_out", how="inner")
    m["diff"] = m["mean_in"] - m["mean_out"]
    w = stats.wilcoxon(m["mean_in"], m["mean_out"]) if len(m) >= 5 else None
    # bootstrap over forms for the mean difference
    boots = [
        RNG.choice(m["diff"].to_numpy(), size=len(m), replace=True).mean()
        for _ in range(N_BOOT)
    ]
    summary = {
        "n_forms": int(len(m)),
        "n_forms_in_higher": int((m["diff"] > 0).sum()),
        "median_in": float(m["mean_in"].median()),
        "median_out": float(m["mean_out"].median()),
        "mean_diff": float(m["diff"].mean()),
        "mean_diff_ci95": [
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)),
        ],
        "wilcoxon_W": float(w.statistic) if w else float("nan"),
        "wilcoxon_p": float(w.pvalue) if w else float("nan"),
        # matched-pairs rank-biserial correlation
        "rank_biserial": float(_rank_biserial(m["diff"].to_numpy())),
    }
    return m.reset_index(), summary


def _rank_biserial(d: np.ndarray) -> float:
    d = d[d != 0]
    if len(d) == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(d))
    return (ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum()


def multitoken_table(con: pd.DataFrame, rnc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in ["anchor", "filler", "other"]:
        s = con[con.type == t]
        rows.append(
            {"type": t, "n": len(s), "share_multitoken": s["multitoken"].mean()}
        )
    if len(rnc):
        rows.append(
            {
                "type": "rnc_target",
                "n": len(rnc),
                "share_multitoken": rnc["multitoken"].mean(),
            }
        )
    return pd.DataFrame(rows)


def qualitative(con: pd.DataFrame, col: str = "p_chain", k: int = 15) -> str:
    out = []
    a = con[con.type == "anchor"].nsmallest(k, col)
    f = con[con.type == "filler"].nlargest(k, col)
    for title, df in [
        (f"{k} якорей с наименьшей affinity ({col})", a),
        (f"{k} заполнителей с наибольшей affinity ({col})", f),
    ]:
        out.append(f"## {title}\n")
        for _, r in df.iterrows():
            top = ", ".join(f"{t.strip()} ({p:.3f})" for t, p in r["top5"])
            out.append(
                f"- **{r['text']}** ({r['pos']}, k={r['n_tokens']}) = {r[col]:.4f} — конструкция {r['record']} «{r['name']}»\n"
                f"  - предложение: {r['sentence']}\n"
                f"  - топ-5: {top}\n"
            )
        out.append("")
    return "\n".join(out)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    con, rnc = load()

    summary: dict = {
        "counts": {
            "n_constructions": int(con.record.nunique()),
            "n_examples": int(con.groupby(["record", "example_idx"]).ngroups),
            "n_anchor": int((con.type == "anchor").sum()),
            "n_filler": int((con.type == "filler").sum()),
            "n_other": int((con.type == "other").sum()),
            "n_rnc_targets": int(len(rnc)),
            "n_rnc_forms": int(rnc.form.nunique()) if len(rnc) else 0,
        }
    }

    constr_with_both = (
        con[con.type.isin(["anchor", "filler"])].groupby("record").type.nunique()
    )
    summary["counts"]["n_constructions_without_anchor_or_filler"] = int(
        con.record.nunique() - (constr_with_both == 2).sum()
    )
    summary["counts"]["constructions_without_anchor_or_filler"] = sorted(
        set(con.record.unique()) - set(constr_with_both[constr_with_both == 2].index)
    )

    main_rows = []
    for col in ["p_single", "p_chain"]:
        pooled = pooled_test(con, col)
        constr = construction_level(con, col)
        boot = bootstrap_by_construction(con, col)
        summary[f"anchor_vs_filler_{col}"] = {**pooled, **constr, **boot}
        main_rows.append({"variant": col, **pooled, **constr, **boot})
        construction_diffs(con, col).to_csv(TABLES / f"construction_diffs_{col}.csv")
        for by in ["pos", "pos_class", "kind", "syn"]:
            slice_table(con, col, by).to_csv(
                TABLES / f"slice_{by}_{col}.csv", index=False
            )
    pd.DataFrame(main_rows).to_csv(TABLES / "anchor_vs_filler.csv", index=False)

    # single-token-only subset under chain: isolates the effect of adding multitoken words
    single_only = con[con.n_tokens == 1]
    summary["anchor_vs_filler_p_chain_single_token_subset"] = pooled_test(
        single_only, "p_chain"
    )

    # lemma variant for anchors
    anc = con[(con.type == "anchor") & con.p_lemma.notna()]
    summary["lemma_variant"] = {
        "n": int(len(anc)),
        "median_form": float(anc.p_single.median()),
        "median_lemma": float(anc.p_lemma.median()),
        "mean_gain": float((anc.p_lemma - anc.p_single).mean()),
        "share_gain_gt_0.01": float(((anc.p_lemma - anc.p_single) > 0.01).mean()),
    }
    anc[["record", "text", "lemma", "p_single", "p_lemma"]].assign(
        gain=anc.p_lemma - anc.p_single
    ).sort_values("gain", ascending=False).to_csv(
        TABLES / "lemma_variant.csv", index=False
    )

    multitoken_table(con, rnc).to_csv(TABLES / "multitoken.csv", index=False)
    summary["multitoken"] = (
        multitoken_table(con, rnc).set_index("type")["share_multitoken"].to_dict()
    )

    if len(rnc):
        for col in ["p_single", "p_chain"]:
            m, s = in_vs_out(con, rnc, col)
            m.to_csv(TABLES / f"in_vs_out_{col}.csv", index=False)
            summary[f"in_vs_out_{col}"] = s
        # restrict to forms where the out-of-construction filter was applicable
        applicable = set(rnc.loc[rnc.filter_applicable, "form"])
        m, s = in_vs_out(
            con[con.form.isin(applicable)], rnc[rnc.form.isin(applicable)], "p_chain"
        )
        summary["in_vs_out_p_chain_filter_applicable_only"] = s
    else:
        summary["in_vs_out"] = "NOT RUN: RNC baseline unavailable"

    (TABLES / "qualitative.md").write_text(
        "# Качественный разбор\n\n" + qualitative(con, "p_chain"), encoding="utf-8"
    )
    (RES / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
