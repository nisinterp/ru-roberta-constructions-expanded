"""Write the final cutoff preprint, with complete and partial evidence clearly labeled.

The ordinary complete-run report remains available through build_report.py.
This publication draft describes the frozen September 2026 experiment.
"""

import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

from build_report import figures, pct, ci, pval, pp


def load(name):
    return json.loads((RES / name).read_text())


def main(argv=None):
    """Build the final cutoff preprint from verified scores and saved statistics."""
    import argparse
    from study_coverage import check_coverage, verify_snapshot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cutoff", action="store_true", required=True, help="Use the frozen, explicitly requested final cutoff"
    )
    parser.parse_args(argv)
    snapshot = verify_snapshot(ROOT)
    for dataset, source in [
        ("constructicon", "items.jsonl"),
        ("outside", "outside_items.jsonl"),
        ("rnc", "rnc_items.jsonl"),
    ]:
        check_coverage(dataset, source, allow_cutoff=dataset == "rnc", root=ROOT)
    summary, outside, cutoff = load("summary.json"), load("outside_summary.json"), load("cutoff_analysis.json")
    audit = load("data_audit.json")
    if summary["analysis_mode"] != "user_requested_cutoff":
        raise ValueError("Recompute the primary analysis with --cutoff first")
    if audit["selected_constructions"] != audit["eligible_constructions"]:
        raise ValueError("The preprint requires the full eligible inventory")
    figures(summary, outside)
    cutoff_figures()
    REPORT.mkdir(exist_ok=True)
    font = Path("/usr/share/fonts/truetype/dejavu")
    for name, filename in [("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(font / filename)))
    styles = getSampleStyleSheet()
    for name in ["Normal", "BodyText", "Heading1", "Heading2", "Title"]:
        styles[name].fontName = "DejaVu"
    styles["BodyText"].fontSize, styles["BodyText"].leading = 9.5, 13
    styles["BodyText"].spaceAfter = 8
    styles["BodyText"].allowWidows = 0
    styles["BodyText"].allowOrphans = 0
    styles["Heading1"].fontSize = 13
    styles["Heading2"].fontSize = 11
    styles.add(ParagraphStyle("Cell", fontName="DejaVu", fontSize=8, leading=10.5))
    story, markdown = [], []

    def heading(text, level=1):
        story.append(Paragraph(escape(text), styles[f"Heading{level}"]))
        markdown.append("#" * (level + 1) + " " + text + "\n")

    def paragraph(text):
        story.append(Paragraph(escape(text), styles["BodyText"]))
        markdown.append(text + "\n")

    def table(headers, rows):
        cells = [[Paragraph(escape(str(v)), styles["Cell"]) for v in row] for row in [headers] + rows]
        item = Table(cells, colWidths=[499 / len(headers)] * len(headers), repeatRows=1)
        item.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f1f3")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.grey),
                ]
            )
        )
        story.extend([item, Spacer(1, 9)])
        markdown.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"])
        markdown.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
        markdown.append("")

    def figure(filename, caption, height=210):
        story.append(
            KeepTogether(
                [
                    Image(str(FIG / filename), width=460, height=height, kind="proportional"),
                    Paragraph(escape(caption), styles["BodyText"]),
                ]
            )
        )
        markdown.append(f"![{caption}](../results/figures/{filename})\n")

    paired = summary["correct"]["paired_construction_difference"]
    matched = summary["in_vs_out"]["all_cached_controls"]["correct"]
    adjusted = summary["adjusted_anchor_advantage"]
    counts = summary["counts"]
    coverage = cutoff["coverage"]
    title = "Fixed words are easier to recover than variable words in Russian constructions"
    story.append(Paragraph(title, styles["Title"]))
    markdown.append("# " + title + "\n")
    paragraph(
        "A full-inventory ruRoberta-large study with a partially scored corpus baseline. "
        f"Preprint, 29 September 2026. Final analysis of the saved cutoff at {snapshot['cutoff_utc']}. "
        "Not peer reviewed."
    )

    heading("1. Abstract")
    paragraph(
        f"We test whether a Russian masked language model predicts fixed construction words (anchors) "
        f"more accurately than variable words (fillers). We examine all {audit['source_constructions']:,} "
        f"Russian Constructicon entries; {audit['represented_constructions']:,} are represented after preparation. "
        f"The primary analysis contains {counts['anchors']:,} anchors and {counts['fillers']:,} fillers. "
        f"Exact recovery is {pct(summary['correct']['anchor']['mean'])} for anchors and "
        f"{pct(summary['correct']['filler']['mean'])} for fillers. The mean within-construction difference is "
        f"{pp(paired['mean_diff'])} percentage points (95% confidence interval {ci(paired['ci95'])}). "
        "The difference remains positive after adjustment for measured word properties. However, fillers "
        "have no clear advantage over outside-span words matched by sentence and part of speech. "
        f"A same-form corpus comparison also favors anchors, but only {pct(coverage['scored_fraction'])} "
        "of eligible corpus targets were scored, with short sentences processed first. "
        "The study supports a broad Russian lexical predictability pattern. It does not establish accurate "
        "prediction of complete argument combinations or universal construction knowledge."
    )

    heading("2. Related literature")
    paragraph(
        "Rozner et al. [1] use pretrained language-model distributions to investigate constructions. "
        "Their global affinity concerns the probability of a word given its sentence context. Local affinity "
        "compares prediction distributions before and after masking additional context. Their English "
        "experiments include fixed lexical material, idioms, and schematic constructions. Their section 5.1 "
        "uses six English construction types with about 50 examples each and single-token targets. We adapt the global "
        "measurement to Russian and add lexical-role and corpus-control comparisons."
    )
    paragraph(
        "RoBERTa [2] is a bidirectional Transformer encoder trained to recover masked tokens. It uses "
        "context on both sides of a missing word. The Russian model family described by Zmitrovich et al. [3] "
        "provides the ruRoberta-large checkpoint used here. The Russian Constructicon [4] is a curated "
        "resource of constructions and examples; its entries are not a frequency-weighted corpus sample."
    )
    paragraph(
        "Salazar et al. [5] describe masked-model pseudo-log-likelihood scoring. Kauf and Ivanova [6] "
        "show why visible pieces of the same word can inflate scores, and propose within-word left-to-right "
        "masking. Our multi-token measure follows this masking principle: later pieces remain hidden while "
        "earlier gold pieces are restored. We score individual words, not whole-sentence pseudo-log-likelihood. "
        "Exact recovery and top-five recovery complement probability scores because high probability and "
        "correct first choice are different outcomes."
    )

    heading("3. Methodology and key hypotheses")
    paragraph(
        "The purpose is to test whether constructional context is associated with lexical predictability. "
        "Anchors are fixed lexical material identified by the existing parser. Fillers are words aligned "
        "to variable slots; a filler word is not necessarily an entire semantic argument. "
        "H1: anchors are easier to recover than fillers. H2: an anchor advantage remains after adjustment "
        "for grammatical category and word properties. H3: the same anchor forms are easier to recover "
        "in construction examples than in corpus controls. H4: fillers are easier to recover than "
        "outside-span words of the same part of speech in the same sentence. These are working hypotheses, "
        "not preregistered tests; the earlier 300-construction results were already known."
    )
    paragraph(
        "We parse the inventory, remove duplicate sentences, align word roles with construction boundaries, "
        "prepare controls, mask each target separately, and analyze the saved predictions. "
        "Known outside-span targets are removed from the primary construction analysis. "
        "Unmatched boundaries are flagged and checked in a matched-only sensitivity analysis. "
        "No new predictions were generated for this final analysis."
    )
    paragraph(
        "For a one-token word, global affinity is P(original token | all other sentence tokens). "
        "For a word split into k tokens, chain affinity is the product, over positions j = 1 to k, of "
        "P(original token j | sentence context, original prefix, masked token j and masked later word tokens). "
        "Log-chain affinity is the sum of these log probabilities. Exact recovery requires the original "
        "token to rank first at every step. The target length is supplied. Lemma affinity sums eligible "
        "one-token vocabulary alternatives with the same lemma; it is available for anchors and RNC "
        "controls, not fillers. Local affinity was not computed."
    )
    paragraph(
        "Pooled accuracy weights word occurrences. The primary paired difference first averages within each "
        "construction and role, then gives constructions equal weight. RNC comparisons average each form "
        "in each context and give matched forms equal weight. Outside-span comparisons average matched "
        "sentence or sentence–POS cells within constructions, then weight constructions equally. "
        "We use 10,000 percentile bootstrap resamples of constructions or forms (seed 20260926), "
        "two-sided Wilcoxon and sign tests, and construction-clustered standard errors for regression. "
        "The adjusted linear probability model includes role, POS, log(1 + token count), "
        "log(1 + sentence token count), character count, and log(1 + sample form frequency). "
        "POS is controlled for both anchors and fillers; automatic pymorphy3 tags use context-free analyses."
    )
    paragraph(
        "Confidence intervals describe resampling uncertainty within this dataset, not uncertainty across "
        "independently trained models or correction for selection bias. POS-specific sign tests receive "
        "Holm adjustment across tested categories. Other intervals and tests are nominal exploratory "
        "comparisons; effect sizes and robustness matter more than isolated small p-values. "
        "A confidence interval containing zero is inconclusive, not proof of equivalence."
    )

    heading("4. Data, model architecture, and experiment")
    table(
        ["Stage or dataset", "Count"],
        [
            [
                "Source entries examined / eligible",
                f"{audit['source_constructions']:,} / {audit['eligible_constructions']:,}",
            ],
            [
                "Represented constructions / unique sentences",
                f"{audit['represented_constructions']:,} / {audit['selected_examples']:,}",
            ],
            ["Completed construction targets / alignment exclusions", "92,846 / 5"],
            ["Primary anchors / fillers", f"{counts['anchors']:,} / {counts['fillers']:,}"],
            ["Completed outside-span controls", f"{outside['counts']['outside_words']:,}"],
            ["RNC queried forms / forms with retained contexts", "3,754 / 3,682"],
            ["RNC retained targets / scoring exclusions", f"{coverage['expected']:,} / {coverage['excluded']:,}"],
            ["RNC scored / eligible but unscored", f"{coverage['scored']:,} / {coverage['unscored']:,}"],
            ["Forms represented in scored RNC data", f"{counts['rnc_forms']:,}"],
        ],
    )
    paragraph(
        "The official Constructicon snapshot [9] supplies entries and slot annotations. The Hugging Face "
        "derivative [10] supplies construction boundaries. They overlap and are joined, not counted as "
        "independent evidence. One eligible entry loses its only example during sentence deduplication. "
        f"The scope audit removes {summary['scope_filter']['excluded_scored_targets']:,} scored words "
        "outside the annotated span, leaving 84,397 primary targets in 15,895 sentences. "
        "All eligible entries were selected before scoring. No synthetic sentences were used."
    )
    paragraph(
        "The unchanged ai-forever/ruRoberta-large checkpoint has 24 Transformer layers, hidden size 1,024, "
        "16 attention heads, feed-forward size 4,096, and approximately 355 million parameters. "
        "Its pinned configuration has 50,265 vocabulary entries and 514 position embeddings. "
        "The scoring limit is 512 input tokens, including special tokens. It uses byte-level subword "
        "tokenization, evaluation mode, and float32 CPU computation. There was no fine-tuning. "
        "No Colab predictions were imported. Model revision: 5192d064ca6ac67c14c40e017ce41612e010f05f [7]."
    )
    paragraph(
        "RNC collection [8] finished all requested forms. It kept up to 50 contexts per form from at most "
        "three result pages, excluding wrong forms, duplicates, overlaps with Constructicon examples, "
        "incomplete snippets, and detectable co-anchor combinations. Controls can still contain constructions. "
        "Up to two words outside each eligible annotated span provide a separate sentence-matched baseline. "
        "These controls preserve sentence context but do not match word identity."
    )
    length = cutoff["sentence_lengths"]
    paragraph(
        f"At the user-requested stop, RNC scoring covered {coverage['scored']:,} of "
        f"{coverage['eligible']:,} eligible targets. The scorer sorted sentences by token length. "
        f"Scored RNC contexts have median {length['scored_rnc_tokens']['median']:.0f} model tokens "
        f"(range {length['scored_rnc_tokens']['minimum']:.0f}–{length['scored_rnc_tokens']['maximum']:.0f}). "
        f"The median whitespace length is {length['scored_rnc_whitespace_words']['median']:.0f} words "
        f"for scored contexts and {length['unscored_rnc_whitespace_words']['median']:.0f} for unscored contexts. "
        "This is a nonrandom cutoff. The corpus comparison cannot represent the full collected baseline."
    )

    heading("5. Key results")
    heading("5.1. Anchors, fillers, and probability scores", 2)
    paragraph(
        f"Anchor accuracy is {pct(summary['correct']['anchor']['mean'])} "
        f"(95% interval {ci(summary['correct']['anchor']['ci95'])}); filler accuracy is "
        f"{pct(summary['correct']['filler']['mean'])} ({ci(summary['correct']['filler']['ci95'])}). "
        f"The paired difference is {pp(paired['mean_diff'])} points ({ci(paired['ci95'])}), across "
        f"{paired['n_pairs']:,} constructions; {paired['positive']:,} favor anchors and "
        f"{paired['ties']:,} tie. Wilcoxon p = {pval(paired['wilcoxon_p'])}. "
        f"The adjusted difference is {pp(adjusted['coefficient'])} points "
        f"({ci([adjusted['ci_low'], adjusted['ci_high']])}; p = {pval(adjusted['p'])}."
    )
    table(
        ["Metric", "Anchors", "Fillers"],
        [
            [
                "Mean chain probability",
                f"{summary['p_chain']['anchor']['mean']:.3f}",
                f"{summary['p_chain']['filler']['mean']:.3f}",
            ],
            [
                "Median chain probability",
                f"{summary['probability_descriptives']['p_chain']['anchor']['median']:.3f}",
                f"{summary['probability_descriptives']['p_chain']['filler']['median']:.3f}",
            ],
            [
                "Mean single-token probability",
                f"{summary['p_single']['anchor']['mean']:.3f}",
                f"{summary['p_single']['filler']['mean']:.3f}",
            ],
            [
                "Single-token exact recovery (n)",
                f"{pct(summary['tokenization']['anchor']['single']['mean'])} (31,962)",
                f"{pct(summary['tokenization']['filler']['single']['mean'])} (38,341)",
            ],
            [
                "Multi-token exact recovery (n)",
                f"{pct(summary['tokenization']['anchor']['multi']['mean'])} (3,780)",
                f"{pct(summary['tokenization']['filler']['multi']['mean'])} (10,314)",
            ],
            [
                "Single-token top-five recovery",
                pct(summary["top5_single"]["anchor"]["mean"]),
                pct(summary["top5_single"]["filler"]["mean"]),
            ],
        ],
    )
    paragraph(
        "The paired chain-probability difference is +0.277 (95% interval [0.267, 0.287]); "
        "the single-token difference is +0.279 [0.269, 0.290]. Mean log-chain scores are −1.130 "
        "for anchors and −2.573 for fillers. Mean anchor lemma affinity is 0.769 across 31,962 "
        "one-token targets. Multi-token recovery is substantially lower in both roles. "
        "Probability products also depend on token count, so single- and multi-token results are separated."
    )
    figure(
        "main_results.png",
        "Figure 1. Anchors receive higher original-word probabilities and higher exact "
        "recovery. Left: empirical cumulative distributions of chain probability; a lower curve indicates "
        "more high-probability words. Right: exact recovery with 95% construction-bootstrap intervals.",
    )
    heading("5.2. POS and construction variation", 2)
    paragraph(
        f"Part of speech is associated with anchor recovery after adjustment (joint Wald "
        f"p = {pval(summary['pos_association']['joint_pos_wald_p'])}). Prepositions and conjunctions "
        "are recovered more often than nouns and finite verbs. This pattern also occurs among fillers: "
        "pooled preposition recovery is 94.1% for anchors and 95.1% for fillers, compared with "
        "65.4% and 39.1% for nouns. Thus, grammatical composition explains part of the pooled role gap."
    )
    paragraph(
        "Automatically classified content-anchor constructions have very similar pooled anchor and filler "
        "accuracy (49.6% and 49.7%). Function-anchor constructions show 77.6% versus 52.7%, and "
        "mixed-anchor constructions show 79.8% versus 50.4%. These are descriptive groups, not "
        "validated semantic construction types. Boundary sensitivity changes the paired gap little: "
        "+24.0 points under the original rule and +24.2 points with matched boundaries only."
    )
    figure(
        "anchor_pos.png",
        "Figure 2. Anchor recovery differs by grammatical category. Only categories with "
        "at least 30 anchor occurrences are shown. Error bars are 95% construction-bootstrap intervals.",
        235,
    )
    heading("5.3. Comparisons with ordinary words", 2)
    comparisons = [
        r for r in outside["comparisons"] if r["metric"] == "correct" and r["matching"] == "sentence_and_pos"
    ]
    table(
        ["Inside role vs outside", "Difference, points (95% CI)", "Constructions / Wilcoxon p"],
        [
            [
                r["role"].replace("_", " "),
                f"{pp(r['mean_diff'])} {ci(r['ci95'])}",
                f"{r['n_pairs']:,} / {pval(r['wilcoxon_p'])}",
            ]
            for r in comparisons
        ],
    )
    paragraph(
        "The strongest practical distinction is between anchors and fillers. Anchors show a clear "
        "sentence–POS matched advantage. Fillers show only +0.6 points, with an interval spanning "
        "−1.3 to +2.6 points. Their chain-probability difference is likewise inconclusive: "
        "+0.003 [−0.010, +0.016], Wilcoxon p = 0.368. Without POS matching, filler accuracy "
        "is lower by 8.2 points [−9.3, −7.1]. Matching changes the compared population as well as "
        "controlling category, so the difference between these estimates is not itself a causal POS effect."
    )
    figure(
        "outside_comparisons.png",
        "Figure 3. Inside-minus-outside recovery differences. Intervals crossing "
        "zero do not establish an advantage. Matching by sentence and POS removes the clear filler deficit "
        "seen with sentence matching alone. Anchors retain a positive difference.",
        220,
    )
    paragraph(
        f"The partial RNC comparison covers {matched['n_pairs']:,} forms. Equally weighted accuracy is "
        f"{pct(matched['mean_in'])} in construction contexts and {pct(matched['mean_out'])} in scored "
        f"RNC controls: {pp(matched['mean_diff'])} points ({ci(matched['ci95'])}), "
        f"Wilcoxon p = {pval(matched['wilcoxon_p'])}. These rates differ from the pooled role rates "
        "because every matched form receives equal weight. The matched chain-, single-token-, and "
        "lemma-probability differences are +0.231 [0.221, 0.241], +0.257 [0.244, 0.270], "
        "and +0.268 [0.255, 0.281], respectively."
    )
    table(
        ["RNC sensitivity comparison", "Forms", "Accuracy difference, points (95% CI)"],
        [
            [
                label.replace("_", " "),
                value["correct"]["n_pairs"],
                f"{pp(value['correct']['mean_diff'])} {ci(value['correct']['ci95'])}",
            ]
            for label, value in cutoff["sensitivity"].items()
        ],
    )
    paragraph(
        "Requiring at least five occurrences in each context gives +25.4 points [23.6, 27.2] "
        "across 1,230 forms. Restricting to forms for which the co-anchor filter applies gives "
        "+28.0 [26.6, 29.4] across 2,946 forms. Only 16 forms have all eligible contexts scored; their sign test is inconclusive (p = 0.070), despite the positive bootstrap interval. Length matching is an exploratory check on "
        "observed overlap. It cannot recover missing long-context outcomes or remove differences "
        "in genre, sense, position, or annotation quality."
    )
    figure(
        "cutoff_lengths.png",
        "Figure 4. Scored and unscored eligible RNC targets have different sentence "
        "length distributions. Length is measured in whitespace-separated words so both groups can be "
        "audited without new model inference. Values above 80 words are pooled at 80.",
        200,
    )
    heading("5.4. Changes from the earlier Russian sample", 2)
    paragraph(
        "Relative to the archived 300-construction experiment, pooled anchor accuracy rises from "
        "74.2% to 76.2%, and filler accuracy from 49.9% to 50.7%. The paired role gap changes "
        "from +20.1 [16.0, 24.2] to +24.2 [23.1, 25.3] points. The adjusted gap changes from "
        "+9.7 [6.3, 13.1] to +11.0 [10.0, 11.9]. The main role conclusion is stable and more precise."
    )
    paragraph(
        "The sentence–POS matched all-inside comparison now has a clearly positive interval: "
        "+8.2 [6.8, 9.7] points, compared with +2.7 [−2.8, 8.2] previously. The filler-specific "
        "comparison remains inconclusive, changing from −2.7 [−9.9, 4.4] to +0.6 [−1.3, 2.6]. "
        "The RNC gap changes from +11.5 [1.1, 21.5] across 27 forms to +24.7 [23.5, 26.0] "
        "across 3,608 forms, but the new corpus sample is incomplete and length-biased. "
        "These are descriptive changes in estimates, not tests that the estimates differ: the "
        "construction samples overlap, sentence–POS aggregation was refined to weight sentences equally within "
        "constructions, and the corpus controls have different selection rules. "
        "No result establishes that the unchanged model itself improved."
    )

    heading("6. Main conclusions, comparison, and limitations")
    paragraph(
        "The Russian results support a broad association between fixed lexical material and higher "
        "contextual predictability. H1 and H2 are supported within the analyzed data. H3 is supported "
        "only for the scored RNC subset. H4 is not supported: after sentence–POS matching, "
        "fillers have no clear accuracy or probability advantage over outside-span words. "
        "The positive all-inside average must therefore not be presented as evidence that variable "
        "arguments are generally easier to predict."
    )
    paragraph(
        "Relative to Rozner et al. [1], this is a partial methodological replication and Russian extension. "
        "It preserves global masked-word probability, but changes the language, checkpoint, inventory, "
        "and controls, and includes multi-token words. It does not reproduce local-affinity interventions, "
        "the full set of English contrasts, or the schematic-slot analyses. Agreement on lexical "
        "predictability therefore does not establish generalization of every original result across "
        "languages or construction types."
    )
    paragraph(
        "Practical use is currently limited to scoring or ranking candidate words in supplied contexts. "
        "The pipeline recovers about half of filler word occurrences exactly; top-five recovery is "
        "higher for single-token fillers, but does not validate every alternative. Whole slots, "
        "multiple arguments, semantic appropriateness, and free generation were not evaluated. "
        "The visible gold sentence and known target length make the task easier than predicting "
        "an entire argument combination. High-accuracy argument generation is an open research goal."
    )
    paragraph(
        "The main limitations are nonrandom RNC stopping; automatic role and context-free POS labels; "
        "curated examples and unequal numbers of examples per construction; imperfect negative controls; "
        "one model and language; and unknown training-data overlap. Sample frequency is not training "
        "frequency. Bootstrap clusters do not fully model shared documents or vocabulary across "
        "constructions. No local-affinity ablation, calibration study, human acceptability evaluation, "
        "or independent full annotation audit was performed. Very small p-values do not remove these limits."
    )
    paragraph(
        "Future work should score a balanced or complete corpus baseline, match genre and word sense, "
        "manually validate a stratified sample, and use context-aware morphology. Local-affinity "
        "interventions could test which context words affect predictions. A practical argument-prediction "
        "system would need whole-slot and multi-slot candidate generation, morphology and agreement "
        "constraints, held-out constructions and documents, and human judgments of acceptable alternatives. "
        "Comparisons across languages and checkpoints are needed before broader generalization."
    )

    heading("7. References")
    references = [
        (
            "Rozner, J., Weissweiler, L., Mahowald, K., and Shain, C. (2025). Constructions are Revealed in Word Distributions. EMNLP, 2116–2138.",
            "https://aclanthology.org/2025.emnlp-main.108/",
        ),
        (
            "Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., and Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.",
            "https://arxiv.org/abs/1907.11692",
        ),
        (
            "Zmitrovich, D., et al. (2023). A Family of Pretrained Transformer Language Models for Russian. arXiv:2309.10931.",
            "https://arxiv.org/abs/2309.10931",
        ),
        (
            "Janda, L. A., Lyashevskaya, O., Nesset, T., Rakhilina, E., and Tyers, F. M. (2018). A Constructicon for Russian: Filling in the Gaps. In Constructicography: Constructicon Development Across Languages, 165–181. John Benjamins.",
            "https://doi.org/10.1075/cal.22.06jan",
        ),
        (
            "Salazar, J., Liang, D., Nguyen, T. Q., and Kirchhoff, K. (2020). Masked Language Model Scoring. ACL, 2699–2712.",
            "https://aclanthology.org/2020.acl-main.240/",
        ),
        (
            "Kauf, C., and Ivanova, A. A. (2023). A Better Way to Do Masked Language Model Scoring. ACL Short Papers, 925–935.",
            "https://aclanthology.org/2023.acl-short.80/",
        ),
        (
            "AI Forever. ruRoberta-large model card and pinned model files. Accessed September 2026.",
            "https://huggingface.co/ai-forever/ruRoberta-large/tree/5192d064ca6ac67c14c40e017ce41612e010f05f",
        ),
        (
            "Russian National Corpus. Main corpus and public API documentation. Collection completed 28 September 2026.",
            "https://ruscorpora.github.io/public-api/",
        ),
        (
            "Russian Constructicon. Official data repository; revision 7806b7d74c56192b97150b06f755be4d37ad5b3c.",
            "https://github.com/constructicon/russian-data/tree/7806b7d74c56192b97150b06f755be4d37ad5b3c",
        ),
        (
            "Futyn-Maker. Russian Constructicon dataset with span annotations; revision 92d19006cb041c81b4bc21a34ebf3e4e082c5b4d.",
            "https://huggingface.co/datasets/Futyn-Maker/russian-constructicon/tree/92d19006cb041c81b4bc21a34ebf3e4e082c5b4d",
        ),
    ]
    for number, (citation, url) in enumerate(references, 1):
        story.append(
            Paragraph(
                escape(f"[{number}] {citation}") + f' <link href="{escape(url)}" color="#16697a">Source</link>.',
                styles["BodyText"],
            )
        )
        markdown.append(f"[{number}] {citation} [Source]({url}).\n")

    heading("8. Supplement")
    heading("8.1. AI declaration", 2)
    paragraph(
        "OpenAI Codex assisted with research-design discussions, hypothesis wording, literature discovery, "
        "code writing and debugging, run orchestration, statistical-analysis scripts, figures, and "
        "drafting and editing this English preprint. The user defined the research scope, provided "
        "RNC access, and requested the scoring stop. AI assistance was not an independent human "
        "annotation audit. Reported statistics were calculated by scripts from saved model outputs; "
        "the corpus sentences were obtained from the cited resources, not generated by AI. "
        "ruRoberta-large produced the predictions being evaluated and is distinct from the assistant "
        "used to prepare this report. Human authorship, attribution, and final scientific review "
        "must be settled by the researchers before external submission; this draft does not claim "
        "that such review has already occurred."
    )
    heading("8.2. Reproducibility and supplementary results", 2)
    paragraph(
        "The repository stores compact word-level numeric tables, bootstrap summaries, adjusted-model "
        "coefficients, POS comparisons, coverage audits, source revisions, and this report. "
        "Full corpus sentences, model weights, and credentials are excluded from Git. "
        "results/cutoff_snapshot.json fixes input and output hashes; results/cutoff_analysis.json "
        "records missing-target and length diagnostics. The strict default rejects incomplete "
        "scoring; the final analysis requires an explicit cutoff flag and matching hashes. "
        "Collection and scoring remain stopped. Source attribution is provided in [7–10]."
    )
    paragraph(
        "Supplementary files: results/summary.json (primary effects and probability summaries); "
        "results/outside_summary.json (all sentence and sentence–POS comparisons); "
        "results/tables/within_pos_construction_differences.csv (Holm-adjusted sign tests); "
        "results/tables/rnc_cutoff_form_coverage.csv (per-form scoring coverage); "
        "results/tables/cutoff_*.csv (length and complete-form sensitivities). "
        "Numerically underflowed test probabilities are stored as zero by the statistical library "
        "and must be interpreted as below numerical resolution, never as literally zero probability."
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("DejaVu", 8)
        canvas.drawString(48, 27, "Russian construction predictability • final cutoff analysis • preprint")
        canvas.drawRightString(A4[0] - 48, 27, str(doc.page))
        canvas.restoreState()

    SimpleDocTemplate(
        str(REPORT / "findings.pdf"),
        pagesize=A4,
        rightMargin=48,
        leftMargin=48,
        topMargin=42,
        bottomMargin=43,
        title=title,
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    (REPORT / "findings.md").write_text("\n".join(markdown), encoding="utf-8")
    print(REPORT / "findings.pdf")


def cutoff_figures():
    """Show the selection mechanism directly, rather than hiding incomplete coverage."""
    counts = pd.read_csv(RES / "tables/rnc_cutoff_length_counts.csv")
    fig, ax = plt.subplots(figsize=(7, 3))
    for scored, label, color in [(True, "Scored", "#16697a"), (False, "Not scored", "#dd6e42")]:
        sub = counts[counts.scored == scored].copy()
        sub["length"] = sub.whitespace_words.clip(upper=80)
        distribution = sub.groupby("length").targets.sum()
        ax.plot(distribution.index, distribution / distribution.sum() * 100, label=label, color=color)
    ax.set(
        xlabel="Sentence length in whitespace-separated words (80 = 80 or more)",
        ylabel="Share of targets (%)",
        xlim=(0, 80),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "cutoff_lengths.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
