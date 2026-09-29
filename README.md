# Predicting words in Russian constructions

This study asks whether **ruRoberta-large** predicts fixed words in constructions (**anchors**) more accurately than words in variable slots (**fillers**). It also compares construction words with ordinary corpus words and words outside the annotated construction in the same sentence.

**The experiment is stopped. The final analysis uses all locally saved predictions at the requested cutoff on 28 September 2026.** Construction and outside-span scoring finished. RNC collection finished, but RNC scoring did not. No Colab results were imported.

Read the [PDF preprint](report/findings.pdf) or its [editable text](report/findings.md). The preprint includes related literature, hypotheses, methods, results, four figures, limitations, references, and an AI declaration.

## Main findings

- Anchors were recovered exactly in **76.2%** of cases; fillers in **50.7%**. The equally weighted within-construction difference was **24.2 percentage points**, with a 95% confidence interval of **23.1–25.3**.
- The adjusted anchor advantage was **11.0 points [10.0, 11.9]**, accounting for POS, token count, sentence length, word length, and sample frequency.
- Against outside-span words matched by sentence and POS, anchors had an advantage of **16.3 points [14.3, 18.2]**. Fillers had **no clear advantage: 0.6 points [−1.3, 2.6]**.
- The partial same-form RNC comparison favored anchors by **24.7 points [23.5, 26.0]**. Exact sentence-token-length matching reduced this to **19.4 points [17.6, 21.3]**. Both estimates apply only to the scored subset.
- Single-token fillers had **58.9%** exact recovery and **81.9%** top-five recovery. Multi-token fillers had **20.3%** exact recovery. These results do not demonstrate reliable prediction of whole argument combinations.

The earlier [300-construction study](archive/300-constructions/report/findings.md) is preserved separately. Its results are not the full-inventory results. Changes in estimates are described in section 5.4 of the preprint; overlapping samples do not support a simple independent before–after significance test.

## Data and completion

| Stage | Final coverage |
| --- | --- |
| Constructicon entries examined | 4,001 |
| Eligible entries / represented after deduplication | 3,844 / 3,843 |
| Unique prepared example sentences | 15,897 |
| Completed construction targets / alignment exclusions | 92,846 / 5 |
| Primary in-scope anchors / fillers | 35,742 / 48,655 |
| Completed outside-span controls | 23,755 |
| RNC forms queried / forms with retained controls | 3,754 / 3,682 |
| Retained RNC targets / scoring exclusions | 165,896 / 124 |
| Scored RNC targets / eligible but unscored | 50,992 / 114,780 |
| Forms represented in scored RNC data | 3,608 |

The scorer processed short sentences first. Only **30.8%** of eligible RNC targets were scored, with **16 forms** fully scored. Scored contexts have a median of 7 whitespace-separated words; unscored contexts have a median of 21. The RNC results are therefore **not estimates for the complete collected baseline**. Length matching cannot recover missing predictions.

The [official Russian Constructicon](https://github.com/constructicon/russian-data) provides entries, examples, and variable-slot annotations. The [Hugging Face derivative](https://huggingface.co/datasets/Futyn-Maker/russian-constructicon) provides construction boundaries. These sources overlap and are joined, not counted as independent datasets. Every source entry is examined; entries without usable annotations cannot be scored. The scope audit removes 8,449 already-scored outside-span targets from the primary construction analysis.

The [Russian National Corpus](https://ruscorpora.ru/) supplies same-form controls, up to 50 retained contexts per form and three result pages. Wrong forms, duplicates, incomplete snippets, overlaps with construction examples, and detectable co-anchor combinations are filtered. Controls are not a random sample of Russian, and may still contain constructions. Up to two outside-span words per eligible construction sentence provide a separate control without matching word identity.

No synthetic sentences or fine-tuning were used. Source revisions and hashes are in `sources/manifest.json`. The model revision is `5192d064ca6ac67c14c40e017ce41612e010f05f`.

## What the pipeline measures

Each target word is masked separately; the rest of the sentence remains visible.

- **Exact recovery:** every original target token ranks first.
- **Single-token affinity:** probability of the original one-token word.
- **Chain affinity:** product of conditional token probabilities within a word. For a multi-token word, later pieces stay masked while original earlier pieces are restored from left to right.
- **Log-chain affinity:** sum of the same log probabilities.
- **Lemma affinity:** probability mass of eligible one-token vocabulary forms sharing a lemma; computed for anchors and RNC controls, not fillers.
- **Top-five recovery:** whether a one-token target occurs among the five highest-ranked predictions.

The multi-token method uses the within-word left-to-right masking principle; it is not a separate PLL model or a whole-sentence PLL calculation. Word length is supplied, and complete multiword arguments are not generated. **Local affinity was not computed.**

Confidence intervals use 10,000 bootstrap resamples, clustered by construction or matched form. POS adjustment includes both anchors and fillers. Automatic context-free POS labels, imperfect controls, training-data overlap, and corpus selection limit interpretation. See the preprint for weighting rules and hypothesis tests.

## Reproduce the final analysis without restarting scoring

Use Python 3.12 and the local source data and prediction JSONL files. Full sentence files are excluded from Git, so a fresh clone alone can inspect the published numeric results but cannot reproduce the raw-data coverage checks.

```bash
python3.12 -m venv .venv
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock.txt
.venv/bin/python scripts/analyze_expanded.py --cutoff --skip-legacy
.venv/bin/python scripts/analyze_outside.py
.venv/bin/python scripts/analyze_cutoff.py
.venv/bin/python scripts/build_preprint.py --cutoff
.venv/bin/python scripts/verify_results.py --cutoff
```

`--cutoff` permits the explicitly frozen RNC sample only when the saved input and output hashes match `results/cutoff_snapshot.json`. Construction and outside-span coverage must remain complete. `--skip-legacy` skips the duplicate upstream analysis, not any primary or cutoff statistics. PDF creation requires the DejaVu Sans system fonts.

The study runner refuses to restart a frozen study. For a new full experiment, use a separate checkout and output directory, preserving this snapshot. The original complete-run report builder remains `scripts/build_report.py`; the current publication draft is built by `scripts/build_preprint.py`.

## New collection and scoring

These instructions are for a **separate new experiment**, not the stopped study. The model download is approximately 1.4 GB; CPU scoring is lengthy.

Obtain an RNC credential through your account and configure `RNC_API_TOKEN` in the environment or a local ignored `.env` file. See the [RNC API documentation](https://ruscorpora.github.io/public-api/). Never commit the credential.

```text
RNC_API_TOKEN=your_token_here
```

Restrict `.env` permissions with `chmod 600 .env`. In a separate clean checkout, `scripts/run_study.py` prepares and runs all stages; `--resume` reuses completed batches after fingerprint checks. Never run two scorers on the same output file. Changing model files, inputs, or scoring code invalidates resume compatibility.

The [Colab notebook](notebooks/score_rnc_colab.ipynb), `scripts/export_colab.py`, and `scripts/import_colab.py` support an optional GPU transfer. The transfer ZIP contains corpus text and stays outside Git. The notebook checks model hashes, float32 CPU/GPU agreement, and performance before scoring. No GPU outputs form part of this preprint, and importing new predictions would invalidate the published cutoff hashes.

## Files and code

| File | Purpose |
| --- | --- |
| `scripts/prepare.py` | Parse all entries, assign roles, deduplicate examples |
| `scripts/audit_spans.py` | Check word roles against construction boundaries |
| `scripts/collect_rnc.py` | Collect resumable same-form controls |
| `scripts/score_expanded.py` | Score words with the unchanged pretrained model |
| `scripts/analyze_expanded.py` | Primary role, POS, probability, and matched-form analyses |
| `scripts/analyze_outside.py` | Sentence and sentence–POS comparisons |
| `scripts/analyze_cutoff.py` | Missing-data audit and length sensitivities; no model inference |
| `scripts/study_coverage.py` | Strict target coverage and cutoff hash checks |
| `scripts/build_preprint.py` | Final PDF and Markdown preprint |
| `scripts/verify_results.py` | Coverage, numeric-range, provenance, and report checks |
| `results/summary.json` | Primary statistics |
| `results/cutoff_analysis.json` | Partial-baseline diagnostics and sensitivities |
| `results/*_scores.csv.gz` | Word-level numeric results without full sentences |
| `results/tables/`, `results/figures/` | Detailed analyses and standalone figures |
| `sources/`, `upstream/` | Provenance, licenses, unchanged original code |
| `archive/300-constructions/` | Historical sample, kept separate |

The old table label `all_cached_controls` now means all **scored** controls in the new RNC collection; it does not mean all collected targets were scored. Null lemma probabilities for fillers mean “not computed,” not zero. Extremely small test probabilities may underflow to zero in machine-readable tables.

## Checks

```bash
.venv/bin/pytest -q
.venv/bin/ruff check scripts tests
.venv/bin/python scripts/verify_results.py --cutoff
```

Tests cover annotation offsets, duplicate removal, filtering, scoring agreement, report rendering, and the cutoff guard. Tests needing local model or corpus assets may skip in a fresh clone. Full verification requires local source and prediction files.

The project extends [ru-constructions-revealed](https://github.com/nisinterp/ru-constructions-revealed/tree/80234c447326551cc891c7c11174159af17c5158). Full source sentences, credentials, model weights, and generated environments are excluded from Git. Source licenses and RNC terms still apply. AI assistance is disclosed in section 8.1 of the preprint.
