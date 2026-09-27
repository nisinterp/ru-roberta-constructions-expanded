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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

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
    axes[0].set(xlabel="Probability of original word", ylabel="Cumulative proportion")
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
    for r in (matched.itertuples() if len(matched) <= 40 else []):
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
    """Write results from completed scores; never reuse the old sample's prose."""
    from analyze_expanded import require_complete
    for dataset, source in [("constructicon", "items.jsonl"), ("rnc", "rnc_items.jsonl"),
                            ("outside", "outside_items.jsonl")]:
        require_complete(dataset, source)
    summary, outside = load("summary.json"), load("outside_summary.json")
    audit, spans, collection = load("data_audit.json"), load("span_audit.json"), load("rnc_collection.json")
    manifest = json.loads((ROOT / "sources/manifest.json").read_text())
    if audit["selected_constructions"] != audit["eligible_constructions"]:
        raise ValueError("The full-inventory preprint requires every eligible construction")
    figures(summary, outside)
    REPORT.mkdir(exist_ok=True)
    font = Path("/usr/share/fonts/truetype/dejavu")
    for name, filename in [("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(font / filename)))
    styles = getSampleStyleSheet()
    for name in ["Normal", "BodyText", "Heading1", "Heading2", "Title"]:
        styles[name].fontName = "DejaVu"
    styles["BodyText"].fontSize, styles["BodyText"].leading = 10, 14
    styles["BodyText"].spaceAfter = 9
    styles.add(ParagraphStyle("Cell", fontName="DejaVu", fontSize=8, leading=11))
    story, markdown = [], []

    def heading(text, level=1):
        story.append(Paragraph(escape(text), styles[f"Heading{level}"]))
        markdown.append("#" * level + " " + text + "\n")

    def paragraph(text):
        story.append(Paragraph(escape(text), styles["BodyText"]))
        markdown.append(text + "\n")

    def table(headers, rows):
        cells = [[Paragraph(escape(str(v)), styles["Cell"]) for v in row] for row in [headers] + rows]
        item = Table(cells, colWidths=[499 / len(headers)] * len(headers), repeatRows=1)
        item.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f1f3")),
                                 ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                                 ("LINEBELOW", (0, 0), (-1, 0), .5, colors.grey)]))
        story.extend([item, Spacer(1, 12)])
        markdown.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"])
        markdown.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
        markdown.append("")

    def figure(filename, caption, height=230):
        story.append(KeepTogether([
            Image(str(FIG / filename), width=460, height=height, kind="proportional"),
            Paragraph(escape(caption), styles["BodyText"]),
        ]))
        markdown.append(f"![{caption}](../results/figures/{filename})\n")

    def page():
        story.append(Spacer(1, 10))

    def evidence(effect):
        low, high = effect["ci95"]
        if low is None:
            return "There are too few matched observations to estimate this comparison."
        if low > 0:
            return "The interval supports higher recovery in the first condition."
        if high < 0:
            return "The interval supports lower recovery in the first condition."
        return "The interval includes zero, so this comparison does not establish a clear difference."

    counts = summary["counts"]
    paired = summary["correct"]["paired_construction_difference"]
    matched = summary["in_vs_out"]["all_cached_controls"]["correct"]
    adjusted = summary["adjusted_anchor_advantage"]
    title = "Predicting fixed and variable words in Russian constructions"
    story.append(Paragraph(title, styles["Title"]))
    markdown.append("# " + title + "\n")
    paragraph("A full-inventory evaluation of ruRoberta-large with matched Russian National Corpus contexts. "
              "Preprint: observational model evaluation; not peer reviewed.")
    heading("Abstract")
    paragraph(f"We apply a masked-word prediction pipeline to all {audit['source_constructions']:,} entries in a pinned "
              f"Russian Constructicon snapshot. The original annotation rules allow {audit['eligible_constructions']:,} "
              f"constructions to enter preparation. After deduplication, the dataset contains {audit['selected_examples']:,} "
              f"sentences. Anchors are fixed lexical material; fillers occupy variable slots. "
              f"Exact recovery is {pct(summary['correct']['anchor']['mean'])} for anchors and "
              f"{pct(summary['correct']['filler']['mean'])} for fillers. The equally weighted within-construction "
              f"difference is {pp(paired['mean_diff'])} percentage points (95% confidence interval {ci(paired['ci95'])}). "
              f"The same-form comparison with newly collected corpus contexts covers {matched['n_pairs']:,} forms and "
              f"gives a difference of {pp(matched['mean_diff'])} points ({ci(matched['ci95'])}). "
              "These measurements describe lexical predictability. They do not by themselves establish abstract "
              "construction knowledge or a causal effect of construction membership.")
    heading("1. Research questions")
    paragraph("Does the model recover fixed anchors more accurately than variable fillers? Does this pattern remain "
              "after accounting for measured word properties? Is the same word easier to recover in a construction "
              "example than in an ordinary corpus context? These questions distinguish lexical-role differences "
              "from differences between surrounding contexts.")
    paragraph("This study extends the public ru-constructions-revealed pipeline. The earlier 300-construction "
              "experiment is preserved separately in archive/300-constructions. All numbers below refer to "
              "the full-inventory run, not to that earlier sample.")
    page()
    heading("2. Data and coverage")
    table(["Preparation stage", "Count"], [
        ["Constructicon entries examined", audit["source_constructions"]],
        ["Entries with usable annotated examples", audit["eligible_constructions"]],
        ["Entries represented after sentence deduplication", audit["represented_constructions"]],
        ["Unique example sentences", audit["selected_examples"]],
        ["Anchors in the primary scored analysis", counts["anchors"]],
        ["Fillers in the primary scored analysis", counts["fillers"]],
        ["Forms requested from RNC", collection["requested_forms"]],
        ["Forms with retained RNC contexts", collection["covered_forms"]],
        ["Scored RNC targets", counts["rnc_targets"]],
        ["Scored outside-span controls", outside["counts"]["outside_words"]]])
    paragraph("The official Constructicon supplies descriptions, examples, and variable-slot offsets. The Hugging "
              "Face derivative supplies construction boundaries. These sources overlap and are joined for annotation; "
              "they are not counted as independent collections. No synthetic sentences or model fine-tuning are used.")
    paragraph("The original parser excludes examples for these reasons: "
              + "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in audit["excluded_examples"].items())
              + ". Entries without any usable example remain in the coverage audit. Thus, full-inventory coverage "
              "means that every source entry is examined, not that every entry can receive a valid prediction score.")
    paragraph(f"Construction boundaries were matched for {spans['example_counts']['matched']:,} sentences. "
              "Known outside-span labels are removed from the primary analysis. Unmatched or ambiguous examples "
              "remain flagged; a matched-only sensitivity analysis tests their influence.")
    heading("RNC baseline", 2)
    paragraph(f"We queried the RNC main corpus for every observed in-scope anchor form, requesting up to "
              f"{collection['per_form']} retained contexts per form and searching at most {collection['max_pages']} "
              "pages. The retrieval cache supports resumption. Wrong-form hits, duplicate sentences, overlaps with "
              "Constructicon examples, and detectable combinations of construction anchors are excluded. "
              "Every queried form, including forms with zero retained contexts, is listed in the collection audit.")
    paragraph("These are ordinary corpus contexts selected by word form, not a random sample of all Russian sentences. "
              "They can still contain constructions. Single-anchor constructions cannot be ruled out by looking for "
              "other anchors. A stricter subset includes only forms for which this co-anchor check applies to every "
              "relevant construction. Corpus genres and sentence lengths can differ from Constructicon examples.")
    page()
    heading("3. Model and measurements")
    paragraph("We use the pinned ai-forever/ruRoberta-large checkpoint in evaluation mode and float32 precision. "
              "Each target word is masked separately while the other words remain visible. For words split into "
              "several model tokens, all target tokens are initially masked and the original prefix is restored "
              "from left to right. Exact recovery requires every target token to be ranked first. The target token "
              "count is given, so this is a constrained recovery task rather than unrestricted generation.")
    paragraph("Single affinity is the probability of an original one-token word. Chain affinity multiplies conditional "
              "token probabilities. Lemma affinity sums eligible one-token vocabulary forms sharing the target lemma. "
              "These probability measures are kept separate from exact recovery. Overlength sentences and word-token "
              "boundary conflicts are excluded and logged, rather than silently truncated.")
    paragraph("Pooled accuracies weight word occurrences. Paired anchor–filler differences weight constructions equally. "
              "The RNC comparison first averages accuracy within each word form and context, then weights forms equally. "
              "Confidence intervals use 10,000 bootstrap resamples of constructions or matched forms. These intervals "
              "describe heterogeneity under resampling; evaluating the full eligible inventory does not make it a "
              "random sample of natural Russian use. They do not measure variation across independently trained models.")
    paragraph("The adjusted linear probability model includes anchor role, part of speech, token count, sentence length, "
              "character count, and observed sample frequency. Standard errors are clustered by construction. "
              "Part-of-speech tags are automatic, context-free pymorphy3 analyses. Sample frequency is not an "
              "independent estimate of frequency in the model's training data. Associations are not causal effects.")
    heading("4. Anchor and filler recovery")
    table(["Role", "Exact recovery", "95% interval"], [
        [role.capitalize(), pct(summary["correct"][role]["mean"]), ci(summary["correct"][role]["ci95"])]
        for role in ["anchor", "filler"]])
    paragraph(f"The paired difference across {paired['n_pairs']:,} constructions is {pp(paired['mean_diff'])} "
              f"percentage points (95% interval {ci(paired['ci95'])}). {paired['positive']:,} paired constructions "
              f"have higher anchor recovery, with {paired['ties']:,} ties. " + evidence(paired))
    paragraph(f"The adjusted anchor coefficient is {pp(adjusted['coefficient'])} percentage points "
              f"(95% interval {ci([adjusted['ci_low'], adjusted['ci_high']])}). This estimate accounts for the "
              "measured covariates but leaves semantic predictability, training exposure, and annotation error uncontrolled.")
    figure("main_results.png", "Figure 1. The left panel shows the distribution of probabilities assigned to original "
           "words. The right panel shows exact recovery. Error bars are 95% construction-bootstrap intervals.")
    page()
    heading("5. Grammatical category and robustness")
    paragraph(f"The joint adjusted test for anchor part-of-speech terms gives p = "
              f"{pval(summary['pos_association']['joint_pos_wald_p'])}. Figure 2 describes categories with at least "
              "30 anchor occurrences. Category differences should not be interpreted as causal effects or as "
              "independent tests of every pair of categories.")
    figure("anchor_pos.png", "Figure 2. Exact recovery by grammatical category. Points farther right indicate better "
           "recovery. Error bars reflect clustering by construction.", 265)
    table(["Boundary rule", "Paired difference (points)", "95% interval"], [
        [name.replace("_", " "), pp(value["mean_diff"]), ci(value["ci95"])]
        for name, value in summary["scope_sensitivity"].items()])
    table(["Construction anchor kind", "Anchor recovery", "Filler recovery"], [
        [kind, pct(value.get("anchor", {}).get("mean")), pct(value.get("filler", {}).get("mean"))]
        for kind, value in summary["kind_accuracy"].items()])
    paragraph("Construction kinds are automatically assigned from name-derived anchor candidates. These descriptive "
              "subgroups differ in vocabulary and size. They show whether the pooled result conceals variation, but "
              "do not establish differences between otherwise comparable constructions.")
    page()
    heading("6. Same forms in ordinary corpus contexts")
    labels = {"all_cached_controls": "All live corpus controls", "filter_applicable": "Co-anchor check applicable",
              "same_form_and_pos": "Same form and automatic POS", "at_least_five_each": "At least five examples per context"}
    table(["Comparison", "Matched forms", "Difference (points)", "95% interval"], [
        [labels[key], value["correct"]["n_pairs"], pp(value["correct"]["mean_diff"]), ci(value["correct"]["ci95"])]
        for key, value in summary["in_vs_out"].items()])
    paragraph(f"For all matched forms, mean recovery is {pct(matched['mean_in'])} in construction examples and "
              f"{pct(matched['mean_out'])} in corpus controls. The Wilcoxon p-value is {pval(matched['wilcoxon_p'])}; "
              f"the sign-test p-value is {pval(matched['sign_p'])}. " + evidence(matched))
    paragraph("The same-form comparison controls lexical identity, but it does not match meaning, genre, or every "
              "contextual property. The same-form-and-POS analysis uses the same context-free tagger and therefore "
              "does not independently resolve ambiguous word uses. The stricter subset and minimum-count comparison "
              "are sensitivity checks, not separate replications.")
    figure("matched_rnc.png", "Figure 3. Each point represents one word form. Points above the diagonal indicate "
           "higher recovery in construction examples; points below it indicate higher recovery in corpus controls.", 290)
    page()
    heading("7. Inside and outside the same sentence")
    paragraph("As a complementary control, up to two words were sampled uniformly outside each annotated construction "
              "span before scoring. This keeps the surrounding sentence fixed, but does not match word identity. "
              "For POS matching, category differences are averaged within sentences, then within constructions, "
              "and finally across constructions with equal weights.")
    figure("outside_comparisons.png", "Figure 4. Positive differences indicate better recovery inside the annotated "
           "construction. An interval crossing zero does not establish a clear difference. Anchors, fillers, "
           "and their pooled result are shown separately.", 240)
    for role in ["anchor", "filler", "all_inside"]:
        result = next(x for x in outside["comparisons"] if x["role"] == role
                      and x["matching"] == "sentence_and_pos" and x["metric"] == "correct")
        paragraph(f"For {role.replace('_', ' ')}, the sentence-and-POS-matched difference is "
                  f"{pp(result['mean_diff'])} points (95% interval {ci(result['ci95'])}). " + evidence(result))
    heading("8. Limitations")
    paragraph("This study evaluates one pretrained model on curated examples from one pinned resource snapshot. "
              "Entries without usable slot annotations or detectable anchors cannot contribute to the recovery analysis. "
              "Automatic anchor matching and POS tags can be wrong. Unknown overlap with model pretraining is possible. "
              "Exact recovery counts synonyms and other acceptable replacements as errors. Corpus controls remain "
              "selected by anchor vocabulary and are not verified construction-free negatives. Multiple exploratory "
              "comparisons should be interpreted together; isolated small p-values are not evidence of broad linguistic competence.")
    heading("9. Conclusions")
    paragraph(f"Across the eligible inventory, the mean within-construction anchor–filler difference is "
              f"{pp(paired['mean_diff'])} points. " + evidence(paired))
    paragraph(f"The matched-form corpus difference is {pp(matched['mean_diff'])} points. " + evidence(matched)
              + " Together with the within-sentence comparison, this separates fixed-word predictability from "
              "a general claim about every word inside a construction. The results concern lexical recovery and "
              "should not be treated as proof that the model represents constructions in a human-like way.")
    page()
    heading("Data, code, and references")
    paragraph("The repository contains pinned source metadata, scripts, coverage audits, numeric tables, figures, "
              "and this report. Full corpus sentences, model weights, credentials, and local environments are excluded "
              "from Git. See README.md for commands and the source manifest for file hashes. The live RNC collection "
              f"completed at {collection['completed_utc']}. API results can change; its cached responses define this run.")
    refs = [
        ("Code for this study", "https://github.com/nisinterp/ru-roberta-constructions-expanded"),
        ("Original pipeline", "https://github.com/nisinterp/ru-constructions-revealed/tree/" + manifest["upstream_revision"]),
        ("Russian Constructicon source", "https://github.com/constructicon/russian-data/tree/" + manifest["constructicon_revision"]),
        ("Construction boundary annotations", "https://huggingface.co/datasets/Futyn-Maker/russian-constructicon/tree/" + manifest["hf_revision"]),
        ("Russian National Corpus", "https://ruscorpora.ru/"),
        ("RNC public API documentation", "https://ruscorpora.github.io/public-api/"),
        ("Model checkpoint", "https://huggingface.co/ai-forever/ruRoberta-large/tree/" + manifest["model_revision"])]
    for label, url in refs:
        story.append(Paragraph(f'<link href="{escape(url)}" color="#16697a">{escape(label)}</link>', styles["BodyText"]))
        markdown.append(f"[{label}]({url})\n")
    paragraph("Attribution: Russian Constructicon / constructicon/russian-data; nisinterp/ru-constructions-revealed; "
              "Futyn-Maker/russian-constructicon; Russian National Corpus; ai-forever/ruRoberta-large. "
              "Consult the preserved license and source files for reuse conditions.")

    def footer(canvas, document):
        canvas.setFont("DejaVu", 8)
        canvas.drawString(48, 24, "Russian constructions | Full-inventory preprint")
        canvas.drawRightString(A4[0] - 48, 24, str(document.page))

    document = SimpleDocTemplate(str(REPORT / "findings.pdf"), pagesize=A4, rightMargin=48, leftMargin=48,
                                 topMargin=42, bottomMargin=42, title=title)
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    (REPORT / "findings.md").write_text("\n".join(markdown), encoding="utf-8")
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text().replace(
            "**The full experiment is running. Its findings are not yet available.**",
            "**The full-inventory report has been generated.** Read [the PDF preprint](report/findings.pdf) "
            "or [the Markdown report](report/findings.md). Check `results/run_status.json` and "
            "`results/verification.json` for final validation status.",
        )
        readme.write_text(text)
    print(REPORT / "findings.pdf")


if __name__ == "__main__":
    main()
