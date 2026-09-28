# Russian constructions and word prediction

This project tests how well **ruRoberta-large** predicts missing words in Russian grammatical constructions. It compares fixed words (**anchors**) with words in variable positions (**fillers**), and compares the same anchor forms with ordinary sentences from the **Russian National Corpus (RNC)**.

The model stays unchanged. We do not train it on the examples or generate synthetic sentences.

## Current study

The new experiment examines **all 4,001 entries** in the pinned Russian Constructicon snapshot. The original parsing rules find usable examples for **3,844 constructions**. After duplicate sentences are removed, **15,897 sentences represent 3,843 constructions**. One construction loses its only example during deduplication. The coverage audit lists all excluded entries; “all constructions” does not mean that entries without usable annotations can be scored.

**The full experiment is running. Its findings are not yet available.** The previous results are preserved in [the 300-construction archive](archive/300-constructions/report/findings.md). They must not be cited as results of the full-inventory study.

When every scoring and validation step finishes, the pipeline writes:

- `report/findings.pdf`: the preprint in simple academic English, with explanatory figures.
- `report/findings.md`: the same report in editable text.
- `results/verification.json`: coverage and numerical checks.
- `results/run_status.json`: the current stage, or the reason to inspect a failed stage's log.

A report is built only from complete target coverage. A successful run is marked `complete` in `results/run_status.json`.

## Where the sentences come from

| Source | What we use it for |
| --- | --- |
| [Russian Constructicon](https://github.com/constructicon/russian-data) | Construction descriptions, example sentences, and variable-slot annotations |
| [Constructicon on Hugging Face](https://huggingface.co/datasets/Futyn-Maker/russian-constructicon) | Boundaries showing which words belong to the annotated construction |
| [Russian National Corpus](https://ruscorpora.ru/) | New corpus sentences containing the same anchor word forms |

The two Constructicon sources mostly contain the same examples. We join their annotations instead of counting them as separate data. All eligible examples are selected before model predictions are inspected. Source revisions and file hashes are recorded in `sources/manifest.json`.

The prepared inventory contains **3,754 anchor forms** eligible for RNC queries. The collector searches each form and keeps up to **50 usable contexts**, using at most three result pages. Some forms may have fewer or no controls. `results/rnc_collection.json` records requested forms, retained counts, and exclusions, including zero-coverage forms. Incomplete API snippets with missing text are rejected in full; per-form parser exclusion counts are saved with the cache and in the collection audit. The old 27-form baseline is not used for the new conclusions.

RNC controls are ordinary corpus contexts selected by word identity. They are **not a random sample of all Russian sentences**. A filter removes recognizable combinations of construction anchors, but cannot guarantee that every retained sentence is free of constructions. This matters especially for constructions with only one anchor.

We also sample up to two words outside the annotated construction in each eligible example sentence. This keeps the surrounding sentence fixed, but does not match word identity. The prepared sample has **23,755 outside-span targets** before scoring exclusions.

## What is measured

For each target, the model sees the sentence with that word masked. All other words remain visible.

- **Exact recovery:** the original word is the model's highest-ranked prediction at every token step.
- **Single-token probability:** the probability of an original word that occupies one model token.
- **Chain probability:** the product of conditional probabilities when a word occupies several tokens.
- **Lemma probability:** the summed probability of eligible one-token vocabulary forms sharing the lemma.

For multi-token words, the original prefix is restored from left to right. The target length is known. This is a constrained prediction experiment, not free text generation.

The main comparison averages anchor-minus-filler accuracy within each construction and then weights constructions equally. The RNC comparison weights matched word forms equally. Confidence intervals use 10,000 bootstrap resamples. Adjusted analyses account for grammatical category, token count, word length, sentence length, and sample frequency. These are observational associations, not causal effects.

## Setup

Use Python 3.12. The model download is about 1.4 GB. Allow several GB of disk space and memory. Full scoring is a long CPU job; collecting and scoring thousands of RNC forms can take a day or longer, depending on the computer and API response times.

```bash
python3.12 -m venv .venv
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock.txt
```

PDF creation uses the DejaVu Sans fonts, normally provided by `fonts-dejavu-core` on Linux. The pipeline runner uses Linux file locking.

### Configure RNC access

Obtain a token through your RNC account. See the [official API documentation](https://ruscorpora.github.io/public-api/).

Create `.env` in this repository, containing this line with your own token:

```text
RNC_API_TOKEN=your_token_here
```

Set file permissions with `chmod 600 .env`. The file is ignored by Git. Alternatively, set the `RNC_API_TOKEN` environment variable. Never add the token to scripts, reports, or commits.

### Run the full experiment

```bash
.venv/bin/python scripts/run_study.py
```

For a long local run, keep it alive after closing the terminal:

```bash
nohup .venv/bin/python scripts/run_study.py > study.log 2>&1 &
```

Check progress:

```bash
cat results/run_status.json
tail -n 10 results/score_expanded.log
tail -n 10 results/collect_rnc.log
```

After an interruption, resume with:

```bash
.venv/bin/python scripts/run_study.py --resume
```

Collection reuses completed per-form caches. Scoring reuses completed batches and checks input and scoring-code fingerprints before doing so. Do not change preparation, scoring settings, model files, or input data during a run. Use a fresh checkout or archive the previous outputs when changing the experiment. Do not start a second scorer on the same output file.

If a process was killed while writing its final JSONL row, back up the output and remove only that incomplete final row before resuming. Never remove complete rows or an exclusion file to bypass validation.

## Individual steps

These commands are also useful for inspecting or debugging one stage:

```bash
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/prepare.py --constructions 0
.venv/bin/python scripts/audit_spans.py
.venv/bin/python scripts/prepare_outside.py
.venv/bin/python scripts/collect_rnc.py
.venv/bin/python scripts/score_expanded.py
.venv/bin/python scripts/score_expanded.py --dataset outside
.venv/bin/python scripts/score_expanded.py --dataset rnc
.venv/bin/python scripts/analyze_expanded.py
.venv/bin/python scripts/analyze_outside.py
.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/verify_results.py
```

`prepare.py` defaults to the entire eligible inventory; `--constructions 0` makes that choice explicit. Positive values are for smaller development runs. `score_expanded.py --limit N` is for debugging; incomplete runs cannot produce the final preprint.

## Files and scripts

| File or directory | Purpose |
| --- | --- |
| `scripts/prepare.py` | Parse every entry, assign roles, remove duplicate sentences, and record coverage |
| `scripts/audit_spans.py` | Check word roles against construction boundaries |
| `scripts/collect_rnc.py` | Collect same-form corpus controls with resumable caches |
| `scripts/score_expanded.py` | Run the unchanged pretrained model and save predictions |
| `scripts/analyze_expanded.py` | Compare anchors, fillers, grammatical categories, and matched corpus forms |
| `scripts/analyze_outside.py` | Compare words inside and outside the same annotated span |
| `scripts/build_report.py` | Create figures and the PDF/Markdown preprint from completed results |
| `scripts/run_study.py` | Run all stages, save progress, and stop on failures |
| `results/*audit.json` | Coverage, annotation checks, and exclusions |
| `results/tables/` | Numeric results and sensitivity analyses |
| `results/figures/` | Standalone illustrations |
| `sources/` | Source revisions, attribution, and licenses |
| `upstream/` | Original scripts, preserved without edits |
| `archive/300-constructions/` | Historical results; not the current experiment |

Full source sentences, model weights, local environments, logs, and credentials stay outside Git. Numeric score tables and report artifacts can be shared after verification. Source licenses and RNC terms still apply.

For compatibility with the original analysis, some table names contain `all_cached_controls`. In the new run this label refers to the newly collected, locally cached RNC controls; provenance is recorded in `rnc_collection.json`.

## Checks

```bash
.venv/bin/pytest -q
.venv/bin/ruff check scripts tests
.venv/bin/python scripts/verify_results.py
```

Tests cover word offsets, construction boundaries, duplicate removal, RNC filtering, and agreement between optimized scoring and the original model calculation. The RNC data-integrity test is skipped until live collection is complete. Final verification requires all three scored datasets and the generated PDF.

## Interpretation and limits

Good recovery may reflect fixed phrases, common words, semantic constraints, or training exposure. It does not by itself prove human-like grammatical understanding. Automatic role alignment and POS tags can be wrong. The curated examples do not reflect natural construction frequencies, and corpus contexts are not manually verified negatives. The report presents these limits alongside the findings.

This project extends [nisinterp/ru-constructions-revealed](https://github.com/nisinterp/ru-constructions-revealed/tree/80234c447326551cc891c7c11174159af17c5158), using the same pinned `ai-forever/ruRoberta-large` checkpoint and original probability definitions.
