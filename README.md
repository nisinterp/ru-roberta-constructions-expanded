# Russian constructions: expanded masked-word recovery study

An executed expansion of [nisinterp/ru-constructions-revealed](https://github.com/nisinterp/ru-constructions-revealed/tree/80234c447326551cc891c7c11174159af17c5158), using the same `ai-forever/ruRoberta-large` checkpoint and single-token, multi-token chain and lemma affinity definitions.

**Read the [PDF findings](report/findings.pdf) or [Markdown report](report/findings.md).**

The experiment selects **300 constructions / 1,231 examples**, compared with 30 / 150 in the original repository. It adds exact word recovery, construction-cluster confidence intervals, POS comparisons and adjusted associations, matched-form RNC comparisons, and **1,888 sampled words outside annotated construction spans**.

## Data decision

Use the sources together, not as independent datasets to concatenate:

- The official [Constructicon YAML](https://github.com/constructicon/russian-data) provides slot annotations and construction descriptions. We audit all 4,001 entries and sample 300 of 3,844 eligible constructions with seed `20260926`.
- [Futyn-Maker/russian-constructicon](https://huggingface.co/datasets/Futyn-Maker/russian-constructicon) provides explicit construction boundaries. Of its 20,993 rows, 20,637 match normalized official examples or illustrations. Joining the boundaries corrects scope leakage in the original slot/anchor rule.
- The [Russian National Corpus](https://ruscorpora.ru/) supplies same-word controls. This run reuses the original repository's cached sentences and re-filters them against the enlarged construction inventory: 695 candidate targets covering 27 forms remain. **The RNC control vocabulary was not expanded.** Words outside the annotated span in the same sentences provide a broader, complementary comparison, but are not same-word matches or guaranteed to be free of all constructions.

This is frozen-model evaluation. HF train/validation/test are combined for annotation lookup, not model training. No synthetic examples or fine-tuning are used. POS tags and lexical role matching are automatic; the report describes their limitations.

## Reproduce

Tested with Python 3.12.14 on CPU. The model download is about 1.4 GB; allow several GB of storage and RAM. Scoring is the dominant runtime and depends on available CPU resources.

```bash
python3.12 -m venv .venv
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock.txt
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/prepare.py --constructions 300 --seed 20260926
.venv/bin/python scripts/audit_spans.py
.venv/bin/python scripts/prepare_outside.py
.venv/bin/python scripts/score_expanded.py
.venv/bin/python scripts/score_expanded.py --dataset rnc
.venv/bin/python scripts/score_expanded.py --dataset outside
.venv/bin/python scripts/analyze_expanded.py
.venv/bin/python scripts/analyze_outside.py
.venv/bin/python scripts/build_report.py
```

PDF generation uses the DejaVu Sans fonts normally available as `fonts-dejavu-core` on Linux. The report script reads `/usr/share/fonts/truetype/dejavu/`; adapt that font lookup for another OS.

Source downloads are pinned by revision in `scripts/download_data.py`; `sources/manifest.json` records SHA-256 hashes. Original scripts under `upstream/` are preserved without edits and attributed to the source repository. No API token is needed for the pinned-cache reproduction.

Scoring writes and flushes JSONL batches and skips already completed targets when resumed. After an abrupt interruption, remove an incomplete final JSONL line before restarting. **Use fresh result files if changing the sample, checkpoint, tokenizer, or role definitions.** Analysis rejects missing or duplicate target coverage. `--limit` is for development only; incomplete runs cannot produce the main summary.

For a larger follow-up sample, `prepare.py --constructions 0` selects all eligible constructions, but its scores must be generated in a fresh output directory. The report's narrative and sampling descriptions are written for the executed 300-construction experiment and must also be updated.

## Outputs

- `report/findings.pdf`, `report/findings.md`: findings, methods, figures, uncertainty and limitations.
- `results/summary.json`: primary boundary-corrected results, POS models and RNC comparisons.
- `results/outside_summary.json`: within-sentence inside/outside comparisons.
- `results/original_metrics.json`, `results/original_tables/`: the original analysis applied to expanded scores before scope correction.
- `results/tables/`: paired comparisons, POS effects, adjusted coefficients, and sensitivity results.
- `results/*_scores.csv.gz`: compact numeric word-level results without full source sentences.
- `results/figures/`: standalone PNG figures used in the PDF.
- `data/selected_constructions.json`, `data/scope_index.json`: frozen selection and annotation lookup.
- `results/data_audit.json`, `results/span_audit.json`, `results/outside_audit.json`: data selection and exclusion accounting.
- `sources/manifest.json`, `sources/*`: pinned provenance, source cards and Constructicon attribution/license.
- `results/annotation_spotcheck.json`: exploratory AI-assisted review of 25 anchors, **not** independent gold validation.

Large raw downloads, virtual environments, model weights, full-sentence score JSONL files and runtime logs are ignored by Git. They are present locally after reproduction. Numeric outputs and the PDF can be inspected without loading the model.

## Validation

```bash
.venv/bin/pytest -q
.venv/bin/ruff check scripts tests
.venv/bin/python scripts/verify_results.py
```

Tests check optimized fp32 probabilities against the original full model, padded versus unpadded scoring, exact greedy recovery, annotation offsets, malformed scope markup, sentence deduplication and outside-sample membership. Completion checks account for every target and exclusion before analysis. See `results/verification.json` for the final audit.

No claim of causal POS effects or construction-free negative contexts is made. RNC vocabulary coverage, automatic POS ambiguity, one model and one sample constrain generalization.
