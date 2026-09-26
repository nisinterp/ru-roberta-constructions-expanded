"""Figures for the paper. Writes PNGs to results/figures/ and paper-typst/figures/."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "figures"
PAPER_FIG = ROOT / "paper-typst" / "figures"

ANCHOR = "#2a78d6"
FILLER = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "savefig.dpi": 300,
    }
)

KIND_RU = {
    "func": "служебные якоря",
    "content": "знаменательные якоря",
    "mixed": "смешанные якоря",
}
VARIANT_RU = {
    "p_single": "single (только однотокенные)",
    "p_chain": "chain (все слова)",
}


def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(x)
    return x, np.arange(1, len(x) + 1) / len(x)


def save(fig, name: str) -> None:
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path)
    shutil.copy(path, PAPER_FIG / name)
    plt.close(fig)
    print("wrote", path)


def fig_distributions(con: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, col in zip(axes, ["p_single", "p_chain"], strict=True):
        for t, color, label in [
            ("anchor", ANCHOR, "якоря"),
            ("filler", FILLER, "заполнители"),
        ]:
            x = con.loc[(con.type == t) & con[col].notna(), col].to_numpy()
            xs, ys = ecdf(x)
            ax.step(
                xs, ys, where="post", color=color, lw=2, label=f"{label} (n={len(x)})"
            )
            med = np.median(x)
            ax.plot(
                [med], [0.5], "o", ms=6, color=color, mec=SURFACE, mew=1.5, zorder=3
            )
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.01, 0.1, 0.5, 1])
        ax.set_xticklabels(["0", "0.01", "0.1", "0.5", "1"])
        ax.set_title(VARIANT_RU[col], loc="left")
        ax.set_xlabel("global affinity")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("доля слов с affinity ≤ x")
    save(fig, "fig_distributions.png")


def fig_construction_diffs(con: pd.DataFrame) -> None:
    col = "p_chain"
    sub = con[con.type.isin(["anchor", "filler"])]
    g = sub.groupby(["record", "type"])[col].mean().unstack().dropna()
    meta = con.drop_duplicates("record").set_index("record")[["name", "kind"]]
    g = g.join(meta)
    g["diff"] = g["anchor"] - g["filler"]
    order = ["func", "mixed", "content"]
    g["kind_order"] = g["kind"].map({k: i for i, k in enumerate(order)})
    g = g.sort_values(["kind_order", "diff"], ascending=[False, True])

    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    y = np.arange(len(g))
    for yi, (_, r) in zip(y, g.iterrows(), strict=True):
        ax.plot([r["filler"], r["anchor"]], [yi, yi], color=GRID, lw=2, zorder=1)
    ax.scatter(
        g["filler"],
        y,
        s=36,
        color=FILLER,
        edgecolor=SURFACE,
        lw=1.2,
        zorder=3,
        label="средняя affinity заполнителей",
    )
    ax.scatter(
        g["anchor"],
        y,
        s=36,
        color=ANCHOR,
        edgecolor=SURFACE,
        lw=1.2,
        zorder=3,
        label="средняя affinity якорей",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}" for n in g["name"]], fontsize=7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("средняя global affinity (chain)")
    # group separators and labels
    prev = None
    for yi, k in zip(y, g["kind"], strict=True):
        if k != prev:
            if prev is not None:
                ax.axhline(yi - 0.5, color=INK2, lw=0.6)
            prev = k
    for k in order:
        ys = y[g["kind"].to_numpy() == k]
        if len(ys):
            ax.text(
                1.03,
                ys.mean(),
                KIND_RU[k],
                transform=ax.get_yaxis_transform(),
                rotation=270,
                va="center",
                ha="left",
                color=INK2,
                fontsize=8,
            )
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_construction_diffs.png")


def fig_in_vs_out(con: pd.DataFrame, rnc: pd.DataFrame) -> None:
    col = "p_chain"
    con = con.assign(form=con["text"].str.lower().str.replace("ё", "е"))
    a = con[con.type == "anchor"].groupby("form")[col].mean()
    r = rnc.groupby("form")[col].mean()
    m = pd.concat([a.rename("in"), r.rename("out")], axis=1, join="inner")
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--", zorder=1)
    ax.scatter(
        m["out"], m["in"], s=40, color=ANCHOR, edgecolor=SURFACE, lw=1.2, zorder=3
    )
    m["diff"] = m["in"] - m["out"]
    for form, row in pd.concat(
        [m.nlargest(6, "diff"), m.nsmallest(1, "diff")]
    ).iterrows():
        ax.annotate(
            form,
            (row["out"], row["in"]),
            xytext=(4, -3),
            textcoords="offset points",
            fontsize=7,
            color=INK,
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("вне конструкции (НКРЯ), средняя affinity")
    ax.set_ylabel("в конструкции (конструктикон), средняя affinity")
    ax.text(
        0.03,
        0.97,
        "выше в конструкции",
        transform=ax.transAxes,
        va="top",
        color=INK2,
        fontsize=8,
    )
    ax.text(
        0.97,
        0.03,
        "выше вне конструкции",
        transform=ax.transAxes,
        ha="right",
        color=INK2,
        fontsize=8,
    )
    ax.set_title(f"Словоформы-якоря (n={len(m)}), chain", loc="left")
    save(fig, "fig_in_vs_out.png")


def fig_slices(con: pd.DataFrame) -> None:
    col = "p_chain"
    sub = con[con.type.isin(["anchor", "filler"])]
    panels = [
        (
            "pos_class",
            {"func": "служебные", "content": "знаменательные"},
            "Класс части речи слова",
        ),
        (
            "kind",
            {"func": "служебные", "mixed": "смешанные", "content": "знаменательные"},
            "Тип якорей конструкции",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, (by, labels, title) in zip(axes, panels, strict=True):
        keys = list(labels)
        y = np.arange(len(keys))
        for yi, k in zip(y, keys, strict=True):
            g = sub[sub[by] == k]
            ma = g.loc[g.type == "anchor", col].median()
            mf = g.loc[g.type == "filler", col].median()
            ya, yf = yi + 0.12, yi - 0.12
            ax.plot([mf, ma], [yf, ya], color=GRID, lw=2, zorder=1)
            ax.scatter(
                [ma], [ya], s=40, color=ANCHOR, edgecolor=SURFACE, lw=1.2, zorder=3
            )
            ax.scatter(
                [mf], [yf], s=40, color=FILLER, edgecolor=SURFACE, lw=1.2, zorder=3
            )
            for v, yy in [(ma, ya), (mf, yf)]:
                ha, dx = ("right", -6) if v > 0.8 else ("left", 6)
                ax.annotate(
                    f"{v:.2f}",
                    (v, yy),
                    xytext=(dx, 0),
                    textcoords="offset points",
                    ha=ha,
                    va="center",
                    fontsize=7,
                    color=INK,
                )
            na, nf = (g.type == "anchor").sum(), (g.type == "filler").sum()
            ax.text(
                0.5,
                yi - 0.38,
                f"якорей {na} / заполнителей {nf}",
                ha="center",
                fontsize=6.5,
                color=INK2,
            )
        ax.set_yticks(y)
        ax.set_yticklabels([labels[k] for k in keys])
        ax.set_ylim(-0.6, len(keys) - 0.4)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("медиана global affinity (chain)")
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="y", visible=False)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=ANCHOR, label="якоря"),
        plt.Line2D([], [], marker="o", ls="", color=FILLER, label="заполнители"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_slices.png")
    shutil.copy(OUT / "fig_slices.png", PAPER_FIG / "fig_slices.png")
    plt.close(fig)
    print("wrote", OUT / "fig_slices.png")


def main() -> None:
    con = pd.read_json(RES / "affinity_constructicon.jsonl", lines=True)
    con["pos_class"] = np.where(
        con["pos"].isin({"PREP", "CONJ", "PRCL", "NPRO", "INTJ", "PRED", "ADVB_PRO"}),
        "func",
        "content",
    )
    fig_distributions(con)
    fig_construction_diffs(con)
    fig_slices(con)
    rnc_path = RES / "affinity_rnc.jsonl"
    if rnc_path.exists():
        fig_in_vs_out(con, pd.read_json(rnc_path, lines=True))


if __name__ == "__main__":
    main()
