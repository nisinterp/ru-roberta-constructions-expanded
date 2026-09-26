"""Build figures and a self-contained PDF/Markdown report from completed results."""

import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
REPORT = ROOT / "report"
POS_NAMES = {
    "PREP": "Preposition",
    "CONJ": "Conjunction",
    "PRCL": "Particle",
    "NPRO": "Pronoun",
    "NOUN": "Noun",
    "VERB": "Finite verb",
    "INFN": "Infinitive",
    "ADVB": "Adverb",
    "ADJF": "Full adjective",
    "ADJS": "Short adjective",
    "PRED": "Predicative",
    "INTJ": "Interjection",
    "NUMR": "Numeral",
    "COMP": "Comparative",
    "PRTF": "Participle",
}


def pct(x):
    return "NA" if x is None or pd.isna(x) else f"{100 * x:.1f}%"


def ci(x, scale=100):
    return "NA" if x[0] is None else f"[{scale * x[0]:.1f}, {scale * x[1]:.1f}]"


def pval(x):
    return "NA" if x is None else f"{x:.3g}"


def pp(x):
    return "NA" if x is None else f"{100 * x:+.1f}"


def load(name):
    return json.loads((RES / name).read_text())


def figures(summary, outside):
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 170,
            "axes.titleweight": "bold",
        }
    )
    pos = pd.read_csv(RES / "tables/pos_accuracy.csv")
    pos = pos[(pos.role == "anchor") & (pos.n >= 30)].sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.errorbar(
        pos.accuracy * 100,
        np.arange(len(pos)),
        xerr=np.array([pos.accuracy - pos.ci_low, pos.ci_high - pos.accuracy]) * 100,
        fmt="o",
        color="#16697a",
        capsize=3,
    )
    ax.set_yticks(np.arange(len(pos)), [f"{POS_NAMES.get(r.pos, r.pos)} (n={r.n})" for r in pos.itertuples()])
    ax.set(xlabel="Exact anchor recovery (%) with 95% cluster bootstrap CI", xlim=(-2, 102))
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "anchor_pos.png")
    plt.close(fig)
    con = pd.read_csv(RES / "construction_scores.csv.gz")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for role, color in [("anchor", "#16697a"), ("filler", "#dd6e42")]:
        x = np.sort(con.loc[con.type == role, "p_chain"])
        axes[0].plot(x, np.arange(1, len(x) + 1) / len(x), label=role.capitalize(), color=color)
    axes[0].set(xlabel="Chain affinity", ylabel="Cumulative proportion")
    axes[0].legend()
    for i, role in enumerate(["anchor", "filler"]):
        rate = summary["correct"][role]
        lo, hi = rate["ci95"]
        axes[1].bar(i, 100 * rate["mean"], color=["#16697a", "#dd6e42"][i])
        axes[1].errorbar(
            i,
            100 * rate["mean"],
            yerr=[[100 * (rate["mean"] - lo)], [100 * (hi - rate["mean"])]],
            color="black",
            capsize=4,
        )
    axes[1].set(xticks=[0, 1], xticklabels=["Anchors", "Fillers"], ylabel="Exact recovery (%)", ylim=(0, 100))
    fig.tight_layout()
    fig.savefig(FIG / "main_results.png")
    plt.close(fig)
    matched = pd.read_csv(RES / "tables/inout_all_cached_controls_correct.csv")
    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    ax.scatter(matched.mean_out * 100, matched.mean_in * 100, s=35, color="#16697a")
    ax.plot([0, 100], [0, 100], "--", color="grey", lw=1)
    for r in matched.itertuples():
        ax.annotate(r.form, (100 * r.mean_out, 100 * r.mean_in), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(
        xlabel="RNC control exact recovery (%)",
        ylabel="In-construction anchor recovery (%)",
        xlim=(-5, 105),
        ylim=(-5, 110),
    )
    fig.tight_layout()
    fig.savefig(FIG / "matched_rnc.png")
    plt.close(fig)
    comp = [r for r in outside["comparisons"] if r["metric"] == "correct"]
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    for i, r in enumerate(comp):
        d, (lo, hi) = r["mean_diff"], r["ci95"]
        ax.errorbar(100 * d, i, xerr=[[100 * (d - lo)], [100 * (hi - d)]], fmt="o", color="#16697a", capsize=3)
    ax.set_yticks(
        range(len(comp)), [r["role"].replace("_", " ") + " / " + r["matching"].replace("_", " ") for r in comp]
    )
    ax.axvline(0, color="grey", ls="--")
    ax.set_xlabel("Inside minus outside exact recovery (percentage points)")
    fig.tight_layout()
    fig.savefig(FIG / "outside_comparisons.png")
    plt.close(fig)


def main():
    summary, outside = load("summary.json"), load("outside_summary.json")
    audit, spans = load("data_audit.json"), load("span_audit.json")
    legacy = load("original_metrics.json")
    base = json.loads((ROOT / "upstream/original_summary.json").read_text())
    manifest = json.loads((ROOT / "sources/manifest.json").read_text())
    pos = pd.read_csv(RES / "tables/pos_accuracy.csv")
    con = pd.read_csv(RES / "construction_scores.csv.gz")
    figures(summary, outside)
    REPORT.mkdir(exist_ok=True)
    font = Path("/usr/share/fonts/truetype/dejavu")
    pdfmetrics.registerFont(TTFont("DejaVu", str(font / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(font / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily(
        "DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu", boldItalic="DejaVu-Bold"
    )
    styles = getSampleStyleSheet()
    for name in ["Normal", "BodyText", "Heading1", "Heading2", "Title"]:
        styles[name].fontName = "DejaVu"
    styles["BodyText"].fontSize = 9.3
    styles["BodyText"].leading = 13.4
    styles["BodyText"].spaceAfter = 8
    styles["Heading1"].fontSize = 17
    styles["Heading1"].leading = 21
    styles["Heading1"].spaceAfter = 14
    styles["Heading2"].fontSize = 11.5
    styles["Heading2"].leading = 15
    styles["Title"].fontSize = 23
    styles["Title"].leading = 29
    styles.add(ParagraphStyle("SmallTable", fontName="DejaVu", fontSize=8, leading=10.5))
    styles.add(
        ParagraphStyle(
            "CaptionSmall", parent=styles["BodyText"], fontSize=8, leading=11, textColor=colors.HexColor("#52606d")
        )
    )
    story, md = [], []

    def h(text, level=1):
        story.append(Paragraph(escape(text), styles[f"Heading{level}"]))
        md.append("#" * level + " " + text + "\n")

    def para(text):
        story.append(Paragraph(escape(text), styles["BodyText"]))
        md.append(text + "\n")

    def table(headers, rows, widths=None):
        cells = [[Paragraph(escape(str(v)), styles["SmallTable"]) for v in row] for row in [headers] + rows]
        widths = widths or [499 / len(headers)] * len(headers)
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6f0f2")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#16697a")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fa")]),
                ]
            )
        )
        story.extend([t, Spacer(1, 10)])
        md.append(
            "| "
            + " | ".join(headers)
            + " |\n| "
            + " | ".join(["---"] * len(headers))
            + " |\n"
            + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
            + "\n"
        )

    def fig(name, caption, width=490):
        from PIL import Image as PILImage

        with PILImage.open(FIG / name) as im:
            w, hh = im.size
        story.append(Image(str(FIG / name), width=width, height=width * hh / w))
        story.append(Paragraph(escape(caption), styles["CaptionSmall"]))
        md.append(f"![{caption}](../results/figures/{name})\n")

    def page():
        story.append(PageBreak())
        md.append("\n---\n")

    a, f = summary["correct"]["anchor"], summary["correct"]["filler"]
    d = summary["correct"]["paired_construction_difference"]
    r = summary["in_vs_out"]["all_cached_controls"]["correct"]
    joint = summary["pos_association"]["joint_pos_wald_p"]
    adjusted = summary["adjusted_anchor_advantage"]
    story.append(Paragraph("Russian constructions:<br/>masked-word recovery<br/>on a larger sample", styles["Title"]))
    story.append(Spacer(1, 14))
    para("Expanded reproduction of nisinterp/ru-constructions-revealed • 26 September 2026")
    h("Findings", 2)
    para(
        f"On {summary['counts']['constructions']} constructions, anchors were recovered exactly in {pct(a['mean'])} of trials, compared with {pct(f['mean'])} for slot fillers. The average within-construction advantage was {pp(d['mean_diff'])} percentage points (95% CI {ci(d['ci95'])}); {d['positive']} of {d['n_pairs']} paired constructions favored anchors, with {d['ties']} ties."
    )
    para(
        f"Anchor recovery varies with part of speech. A joint test of POS terms after adjustment for token length, sentence length, word length and sample frequency gave p={pval(joint)}. The adjusted anchor-versus-filler association was {pp(adjusted['coefficient'])} percentage points (95% CI {ci([adjusted['ci_low'], adjusted['ci_high']])}). These are observational associations, not proof that POS causes recovery differences."
    )
    para(
        f"For {r['n_pairs']} matched word forms in cached RNC controls, the mean inside-minus-outside accuracy difference was {pp(r['mean_diff'])} percentage points (95% CI {ci(r['ci95'])}). The stricter co-anchor subset has only five forms and an interval spanning zero. Within-sentence POS matching supports an anchor advantage, but not an advantage for all inside words pooled together."
    )
    h("Which data sources should be used?", 2)
    para(
        "Use the sources in conjunction: the official Constructicon YAML supplies construction descriptions and slot annotations; the Hugging Face derivative supplies whole-construction boundaries; RNC sentences supply same-word controls. Do not concatenate Constructicon and Hugging Face examples as independent observations: their content overlaps heavily."
    )
    table(
        ["Dataset size", "Original repository", "Expanded experiment"],
        [
            ["Constructions", 30, audit["selected_constructions"]],
            ["Example sentences", 150, audit["selected_examples"]],
            [
                "Scored anchors / fillers (before scope correction)",
                "261 / 536",
                f"{legacy['counts']['n_anchor']} / {legacy['counts']['n_filler']}",
            ],
            ["Additional outside-span controls", 0, outside["counts"]["outside_words"]],
        ],
    )
    para(
        "All reported model evaluations were executed locally with the original ruRoberta-large checkpoint. No model fine-tuning or synthetic sentences were used."
    )
    page()
    h("Data choice, sampling and exclusions")
    para(
        f"The pinned official Constructicon snapshot contains {audit['source_constructions']:,} records. Parsing retained {audit['eligible_examples']:,} examples across {audit['eligible_constructions']:,} constructions. A uniform random sample of 300 eligible construction IDs (seed {audit['seed']}) was selected before model scoring, retaining all usable examples per selected construction. Identical normalized sentences were retained under only one construction/example owner."
    )
    para(
        "This is a tenfold expansion in construction count and an 8.2-fold expansion in examples, not an evaluation of all 4,001 entries. The original 30-item sample was balanced by syntactic type and anchor class; the new random sample has a different composition. Between-study changes therefore cannot be attributed to sample size alone."
    )
    hf = audit["hf"]
    table(
        ["Source", "Audit", "Role in this experiment"],
        [
            [
                "Official Constructicon YAML",
                f"{audit['source_constructions']:,} records",
                "Slot offsets; lexical anchor candidates; construction metadata",
            ],
            [
                "Hugging Face derivative",
                f"{hf['rows']:,} rows / {hf['constructions']:,} IDs",
                "Brace-delimited construction spans joined by ID + normalized text",
            ],
            [
                "Overlap",
                f"{hf['exact_normalized_source_matches']:,}/{hf['rows']:,} rows ({100 * hf['exact_normalized_source_matches'] / hf['rows']:.1f}%)",
                "HF rows matching official examples or illustrations; not new independent data",
            ],
            [
                "Cached RNC",
                f"{audit['rnc_targets']} targets / {audit['rnc_forms']} forms",
                "Same-word controls after expanded-inventory co-anchor filtering",
            ],
        ],
        [105, 140, 254],
    )
    para(
        f"All three HF splits were joined only for annotation lookup: train={hf['split_sizes']['train']:,}, validation={hf['split_sizes']['validation']:,}, test={hf['split_sizes']['test']:,}. There was no fitted task model or held-out predictive training evaluation. The HF table has {hf['unique_record_sentence_pairs']:,} unique ID/sentence pairs and includes examples plus illustrations; illustrations were not added to the scored sample because equivalent slot annotation was unavailable."
    )
    para(
        "Official-source exclusions: "
        + "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in audit["excluded_examples"].items())
        + ". Exclusion reasons follow a fixed priority and are counted per example."
    )
    para(
        f"HF boundaries matched {spans['example_counts']['matched']} of {audit['selected_examples']} selected examples; {spans['example_counts'].get('unmatched', 0)} remained unmatched. The boundary audit identified 94 anchor labels and 708 filler labels outside the annotated span. The primary analysis removes scored targets in that category; unmatched targets remain flagged, and a matched-only sensitivity analysis is reported. The source-rule analysis is preserved separately."
    )
    para(
        "The RNC controls are reused from the original repository, not newly crawled from the corpus. The original fetcher requires an RNC API token; this reproduction uses the publicly supplied cache and requires no credentials. The cached control vocabulary has not expanded with the construction inventory."
    )
    page()
    h("Pipeline and meaning of the measurements")
    table(
        ["Stage", "Original method retained", "Expansion / correction"],
        [
            [
                "Parse",
                "Strip [text]Role markup and retain character offsets",
                "Apply to full inventory, then sample 300 constructions",
            ],
            [
                "Assign word roles",
                "Slot overlap → filler; lexical or eligible lemma match → anchor",
                "Extract Cyrillic anchors without grammatical tags; audit HF boundaries",
            ],
            [
                "Mask and score",
                "ai-forever/ruRoberta-large; single / chain / lemma affinity",
                "Batch masked-position states; add exact top-1 and single-token top-5 recovery",
            ],
            [
                "Analyze",
                "Anchor/filler; POS/class/type slices; matched RNC forms; bootstrap",
                "Cluster-aware accuracy, adjusted POS models, outside-span controls and sensitivity analyses",
            ],
        ],
        [70, 208, 221],
    )
    para(
        "An anchor is fixed lexical material in a construction. A filler is a word overlapping an annotated variable slot. Slot labels take precedence over lexical matches. Both are roles relative to one annotated construction; a filler can itself be a function word or belong to another construction. Anchor candidates are extracted from the construction name, using the original exact-form, citation-lemma and compound matching rules."
    )
    para(
        "The model revision is "
        + manifest["model_revision"]
        + ". Inference uses evaluation mode, float32 and the original tokenizer. Each target word is masked separately; all other words remain visible. Inputs over 512 tokens and token boundaries crossing a target word are excluded rather than truncated. One filler target was excluded for a token crossing its word boundary; no overlength exclusion occurred. There were 7,101 construction, 695 RNC and 1,888 outside-span scored targets. The run logs and exclusion files record coverage."
    )
    para(
        "Single affinity is the softmax probability of the original token, for one-token words only. Chain affinity masks all target subtokens and multiplies their conditional probabilities while restoring the gold prefix from left to right. Lemma affinity sums eligible one-token vocabulary forms with the same pymorphy3 lemma and leading-space class; it is not full lemma-generation accuracy."
    )
    para(
        "Exact recovery is 1 only if every target subtoken is the top-ranked prediction at its chain step. The all-correct event equals fixed-length left-to-right greedy recovery: if any earlier prediction is wrong, exact word recovery is already impossible. Target subtoken count is known, so this is not unrestricted text generation. Single-token top-5 is reported separately. A probability above 0.5, affinity, and exact recovery are different measures."
    )
    para(
        "Confidence intervals use 10,000 bootstrap resamples. Pooled accuracy resamples construction clusters, retaining word weighting; paired effects average anchor-minus-filler means equally across constructions with both roles. RNC comparisons weight matched forms equally and bootstrap forms. Sign tests omit ties in the extended analysis; the original analysis is retained unchanged. OLS linear-probability regressions use construction-cluster robust standard errors; their coefficients express adjusted percentage-point associations."
    )
    para(
        "POS is the first context-free pymorphy3 analysis, as in the original pipeline. Ambiguous Russian words may be tagged incorrectly. The within-form-and-POS check therefore adds little independent disambiguation. Neither POS tagging nor automatic anchor alignment is independent human gold annotation."
    )
    page()
    h("Replication of affinity and word recovery")
    rows = []
    for metric in ["p_single", "p_chain"]:
        b = base["anchor_vs_filler_" + metric]
        expanded = legacy["anchor_vs_filler_" + metric]
        rows.append(
            [
                metric,
                f"{b['median_anchor']:.3f} / {b['median_filler']:.3f}",
                f"{expanded['median_anchor']:.3f} / {expanded['median_filler']:.3f}",
                f"{expanded['cliffs_delta']:.3f}",
                f"{expanded['mean_constr_diff']:.3f}",
            ]
        )
    table(
        ["Metric", "Original medians A/F", "Expanded source-rule medians A/F", "Expanded Cliff δ", "Paired mean Δ"],
        rows,
        [65, 100, 135, 90, 109],
    )
    para(
        "The table reproduces the original probability analysis using its uncorrected span rule, so its expanded counts differ from the corrected primary analysis below. Original pooled Mann–Whitney tests assume independent words and are descriptive; construction bootstrap intervals and paired effects are the main uncertainty summaries. Full syntactic-type, anchor-class, tokenization, lemma and qualitative tables are preserved in results/original_tables/."
    )
    fig(
        "main_results.png",
        "Primary analysis after construction-scope correction. Accuracy intervals resample constructions; chain affinity includes multi-token words.",
    )
    table(
        ["Primary exact recovery", "Targets", "Accuracy", "95% CI"],
        [["Anchors", a["n"], pct(a["mean"]), ci(a["ci95"])], ["Fillers", f["n"], pct(f["mean"]), ci(f["ci95"])]],
    )
    para(
        f"Equal-construction accuracy difference: {pp(d['mean_diff'])} points, CI {ci(d['ci95'])}; sign-test p={pval(d['sign_p'])}. Single-token-only accuracy is {pct(summary['single_token_accuracy']['anchor']['mean'])} for anchors and {pct(summary['single_token_accuracy']['filler']['mean'])} for fillers. Restricting to single tokens addresses part of the word-length confound, but does not match lexical identity or word frequency."
    )
    page()
    h("Does recovery depend on part of speech?")
    fig(
        "anchor_pos.png",
        "Anchor categories with at least 30 observations. These are word-level POS tags, distinct from the declared POS of an entire construction anchor.",
        width=480,
    )
    ranks = pos[(pos.role == "anchor") & (pos.n >= 30)].sort_values("accuracy")
    low, high = ranks.iloc[0], ranks.iloc[-1]
    para(
        f"Among categories with at least 30 anchor tokens, observed accuracy ranges from {pct(low.accuracy)} for {POS_NAMES.get(low.pos, low.pos).lower()} (n={low.n}) to {pct(high.accuracy)} for {POS_NAMES.get(high.pos, high.pos).lower()} (n={high.n}). The differences are substantial descriptively; overlapping intervals should not be read as a complete pairwise significance test."
    )
    para(
        f"The anchor-only model groups POS categories with fewer than 30 observations into OTHER_RARE. With log(1+subtoken count), log(1+sentence tokens), character count and log(1+observed sample form count) included, the joint POS Wald statistic is {summary['pos_association']['joint_pos_statistic']:.2f}, p={pval(joint)} (n={summary['pos_association']['n']}, R²={summary['pos_association']['r_squared']:.3f}). Sample frequency is a weak proxy and is not an independent corpus-frequency estimate."
    )
    para(
        f"Across anchors and fillers, the adjusted anchor coefficient is {pp(adjusted['coefficient'])} points, CI {ci([adjusted['ci_low'], adjusted['ci_high']])}, p={pval(adjusted['p'])}. Thus POS-related differences and the anchor-role association must be considered separately. Construction-specific lexical content, semantic predictability and training exposure remain potential confounders."
    )
    rows = []
    for k, g in summary["pos_class_accuracy"].items():
        rows.append([k, pct(g["anchor"]["mean"]), pct(g["filler"]["mean"]), f"{g['anchor']['n']} / {g['filler']['n']}"])
    table(["Original POS grouping", "Anchor accuracy", "Filler accuracy", "n A/F"], rows)
    para(
        "The inherited function-word group includes prepositions, conjunctions, particles, pronouns, interjections and predicatives; adverbs generally fall into the other group. This operational grouping is not a universal linguistic taxonomy. Within-POS, within-construction paired tests and Holm-adjusted sign-test p-values are supplied in tables/within_pos_construction_differences.csv."
    )
    page()
    h("Same word forms inside versus RNC controls")
    fig(
        "matched_rnc.png",
        "Each point is one normalized word form; above the diagonal means higher exact recovery in construction examples.",
        width=415,
    )
    rows = []
    for key, label in [
        ("all_cached_controls", "All matched forms"),
        ("filter_applicable", "Co-anchor filter applicable"),
        ("same_form_and_pos", "Same form + POS"),
        ("at_least_five_each", "At least 5 examples each"),
    ]:
        x = summary["in_vs_out"][key]["correct"]
        rows.append([label, x["n_pairs"], pct(x["mean_in"]), pct(x["mean_out"]), pp(x["mean_diff"]), ci(x["ci95"])])
    table(["Comparison", "Forms", "In", "Out", "Δ points", "95% CI"], rows, [148, 45, 57, 57, 68, 124])
    para(
        f"The all-form comparison gives a Wilcoxon p-value of {pval(r['wilcoxon_p'])}; {r['positive']} form means are higher inside, with {r['ties']} ties. The corresponding sign-test p-value is {pval(r['sign_p'])}, so the evidence is not uniform across tests. The strict five-form subset is inconclusive. Means in this table weight forms equally, not individual RNC sentences. Chain-affinity and single-token-affinity versions are in the machine-readable summary and matched-form CSV files."
    )
    para(
        f"The expanded co-anchor filter retains {audit['rnc_targets']} cached targets, of which {audit['rnc_filter_applicable_targets']} meet the filter-applicable criterion. It rejects a sentence if all observed co-anchor keys of any relevant construction are present. Single-anchor constructions cannot be excluded by this rule. Even the applicable subset is a heuristic control, not proof that every target is outside every construction."
    )
    para(
        "This baseline is limited by the original cache's vocabulary, few in-construction observations for some forms, differing genres and sentence lengths, and incomplete construction detection. The larger construction sample does not create a correspondingly larger independent RNC control sample. An authenticated, expanded RNC collection with manually verified negatives would be needed for a broad matched-lexeme claim."
    )
    page()
    h("Inside versus outside the same annotated span")
    para(
        f"To broaden the contextual comparison, {outside['counts']['outside_words']:,} words were sampled from outside HF-marked construction spans in {outside['counts']['outside_sentences']} sentences spanning {outside['counts']['outside_constructions']} constructions. Up to two outside words were sampled uniformly per sentence with a fixed seed, before scoring. All such sentences retain the same surrounding context as the inside targets."
    )
    fig(
        "outside_comparisons.png",
        "Construction-cluster bootstrap intervals. A positive difference favors words inside the annotated construction.",
        width=490,
    )
    rows = []
    for x in outside["comparisons"]:
        if x["metric"] == "correct":
            rows.append(
                [
                    x["role"].replace("_", " "),
                    x["matching"].replace("_", " "),
                    x["n_pairs"],
                    pp(x["mean_diff"]),
                    ci(x["ci95"]),
                ]
            )
    table(["Inside role", "Matching", "Constructions", "Δ points", "95% CI"], rows, [90, 139, 80, 70, 120])
    inside_pos = {
        x["role"]: x for x in outside["comparisons"] if x["metric"] == "correct" and x["matching"] == "sentence_and_pos"
    }
    para(
        f"After sentence-and-POS matching, the anchor advantage is {pp(inside_pos['anchor']['mean_diff'])} points, CI {ci(inside_pos['anchor']['ci95'])}. The pooled inside-word estimate is {pp(inside_pos['all_inside']['mean_diff'])} points, CI {ci(inside_pos['all_inside']['ci95'])}, and the filler estimate is {pp(inside_pos['filler']['mean_diff'])} points, CI {ci(inside_pos['filler']['ci95'])}. Thus these data support better recovery of anchors, not a general claim that every word inside a construction is easier to recover."
    )
    para(
        "For sentence matching, word accuracies are averaged by role within each sentence, then inside-minus-outside differences are averaged within each construction and across constructions. For sentence-and-POS matching, only POS categories observed on both sides are paired before the same construction-level aggregation. The two estimates target different supported subsets. “All inside” refers to classified anchor and filler targets; in-span words with neither role are excluded, as in the original pipeline."
    )
    para(
        "This comparison controls sentence context and, in the second version, observed POS, but not word identity. It addresses recovery inside versus outside the annotated target construction; an outside word may still instantiate some other construction. It cannot replace the same-word RNC comparison. Fillers and anchors should not be pooled without also examining their separate results."
    )
    page()
    h("Robustness, errors and limits")
    rows = []
    for key, label in [("original_rule", "Original span rule"), ("hf_matched_only", "Only HF-matched examples")]:
        x = summary["scope_sensitivity"][key]
        rows.append([label, x["n_pairs"], pp(x["mean_diff"]), ci(x["ci95"])])
    rows.append(["Primary corrected rule", d["n_pairs"], pp(d["mean_diff"]), ci(d["ci95"])])
    table(["Anchor–filler sensitivity", "Constructions", "Δ accuracy points", "95% CI"], rows, [190, 80, 100, 129])
    rows = []
    for kind, g in summary["kind_accuracy"].items():
        rows.append(
            [kind, pct(g["anchor"]["mean"]), pct(g["filler"]["mean"]), f"{g['anchor']['n']} / {g['filler']['n']}"]
        )
    table(["Construction anchor kind", "Anchor accuracy", "Filler accuracy", "n A/F"], rows)
    para(
        "Content-only constructions have pooled anchor accuracy of 44.2% versus 47.0% for fillers; superiority is not universal across construction kinds. Anchor kinds are automatically classified from name-derived lexical candidates. They are not identical to the original manually selected balanced strata. A small AI-assisted spot-check of 25 randomly selected in-span anchors (seed 991) found plausible alignments, but is not an independent annotation study and does not establish an error rate. Incidental occurrences of common anchor words inside broad spans and POS ambiguity can still bias results."
    )
    raw = pd.read_json(RES / "affinity_constructicon.jsonl", lines=True)
    keys = set(zip(con.record, con.example_idx, con.char_start, con.char_end))
    raw["form"] = raw.text.str.lower().str.replace("ё", "е")
    raw = raw[[k in keys for k in zip(raw.record, raw.example_idx, raw.char_start, raw.char_end)]]
    examples = raw[raw.type == "anchor"].nsmallest(3, "p_chain")
    rows = []
    for x in examples.itertuples():
        rows.append([x.text, x.name, str(x.top5[0][0]).strip() or "<space>", f"{x.p_chain:.3g}"])
    table(["Low-affinity anchor", "Construction", "Top first subtoken", "Chain affinity"], rows, [100, 229, 100, 70])
    para(
        "The prediction column shows only the first masked subtoken, not a complete generated replacement. A failure of exact lexical recovery need not mean failure to recognize a construction: alternatives or synonyms are counted as wrong. Conversely, high filler recovery can reflect a constrained slot or repeated material."
    )
    para(
        "Scope: one pretrained model, one fixed random construction sample, curated examples rather than naturally sampled construction frequencies, and observational comparisons. Unknown pretraining overlap is possible. Source snapshots and deterministic sampling support reproducibility, but tiny numerical differences across hardware or package builds are possible. The original study's local-affinity/JSD and other tasks were outside its implemented pipeline and were not added here."
    )
    page()
    h("Reproducibility and source references")
    para(
        "The local Git repository contains analysis code, pinned source copies, selected construction metadata, numeric word-level results, original and extended tables, figures, tests, and this report. Downloaded corpus sentence files, model weights, virtual environments and logs are excluded from Git. The full sentence-level scoring outputs remain available locally."
    )
    for command in [
        "python3.12 -m venv .venv",
        ".venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock.txt",
        ".venv/bin/python scripts/download_data.py",
        ".venv/bin/python scripts/prepare.py --constructions 300",
        ".venv/bin/python scripts/audit_spans.py",
        ".venv/bin/python scripts/prepare_outside.py",
        ".venv/bin/python scripts/score_expanded.py",
        ".venv/bin/python scripts/score_expanded.py --dataset rnc",
        ".venv/bin/python scripts/score_expanded.py --dataset outside",
        ".venv/bin/python scripts/analyze_expanded.py",
        ".venv/bin/python scripts/analyze_outside.py",
        ".venv/bin/python scripts/build_report.py",
    ]:
        story.append(Paragraph(escape(command), styles["CaptionSmall"]))
        md.append("`" + command + "`  ")
    para(
        "Scoring resumes from complete JSONL rows. Use a fresh output directory when changing the sample, model, masking or annotation rules. Analysis refuses incomplete or duplicated scoring coverage. Eight tests passed, including numerical equivalence with the original model path; lint checks passed. The full SHA-256 manifest is sources/manifest.json. README.md documents validation and output paths."
    )
    refs = [
        (
            "Original study and code",
            "https://github.com/nisinterp/ru-constructions-revealed/tree/" + manifest["upstream_revision"],
        ),
        ("Official Constructicon", "https://constructicon.ruscorpora.ru/"),
        (
            "Official YAML snapshot (CC BY 4.0)",
            "https://github.com/constructicon/russian-data/tree/" + manifest["constructicon_revision"],
        ),
        (
            "Hugging Face derivative",
            "https://huggingface.co/datasets/Futyn-Maker/russian-constructicon/tree/" + manifest["hf_revision"],
        ),
        ("Russian National Corpus", "https://ruscorpora.ru/"),
        (
            "ruRoberta-large checkpoint",
            "https://huggingface.co/ai-forever/ruRoberta-large/tree/" + manifest["model_revision"],
        ),
    ]
    for title, url in refs:
        # Display short labels to prevent long revision URLs from overflowing.
        story.append(
            Paragraph(f'<link href="{escape(url)}" color="#16697a">{escape(title)}</link>', styles["BodyText"])
        )
        md.append(f"[{title}]({url})\n")
    para(
        "Attribution: Russian Constructicon / constructicon/russian-data; nisinterp/ru-constructions-revealed; Futyn-Maker/russian-constructicon; Russian National Corpus; ai-forever/ruRoberta-large. Sources are complementary, not independent corpora. Consult the preserved source metadata and license files for attribution and reuse terms."
    )

    def footer(canvas, doc):
        canvas.setFont("DejaVu", 8)
        canvas.setFillColor(colors.HexColor("#52606d"))
        canvas.drawString(48, 25, "Russian constructions • expanded reproduction • 2026-09-26")
        canvas.drawRightString(A4[0] - 48, 25, str(doc.page))

    document = SimpleDocTemplate(
        str(REPORT / "findings.pdf"),
        pagesize=A4,
        rightMargin=48,
        leftMargin=48,
        topMargin=42,
        bottomMargin=42,
        title="Russian constructions: masked-word recovery on a larger sample",
        author="Project Interpretation",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    (REPORT / "findings.md").write_text("\n".join(md), encoding="utf-8")
    print(REPORT / "findings.pdf")


if __name__ == "__main__":
    main()
