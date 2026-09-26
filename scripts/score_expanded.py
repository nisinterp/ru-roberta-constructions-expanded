"""Original fp32 affinity metrics plus exact recovery; checkpointed CPU batches.

Only masked-position hidden states enter the vocabulary head. This avoids
calculating unused logits and is numerically checked against the full model.
"""

import argparse
from collections import defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
import sys
import time

import pymorphy3
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream"))
from affinity import build_lemma_table, token_span_for_word


class Scorer:
    def __init__(self, threads=4):
        torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(ROOT / "data/raw/model", local_files_only=True)
        self.model = AutoModelForMaskedLM.from_pretrained(ROOT / "data/raw/model", local_files_only=True).eval()
        self.morph = pymorphy3.MorphAnalyzer()
        self.lemma_ids = defaultdict(list)
        for tid, lem in build_lemma_table(self.tok, self.morph).items():
            space = self.tok.convert_ids_to_tokens(tid).startswith("Ġ")
            self.lemma_ids[(lem.replace("ё", "е"), space)].append(tid)

    @lru_cache(maxsize=None)
    def morph_word(self, text):
        parsed = self.morph.parse(text.lower())[0]
        return parsed.normal_form.replace("ё", "е"), str(parsed.tag.POS or "NONE")

    @torch.inference_mode()
    def masked_logits(self, ids, attention, positions):
        hidden = self.model.roberta(input_ids=ids, attention_mask=attention).last_hidden_state
        return self.model.lm_head(hidden[torch.arange(len(ids)), positions])

    @torch.inference_mode()
    def score_batch(self, targets):
        configs = []
        for ti, (_, ids, span) in enumerate(targets):
            for step, pos in enumerate(span):
                masked = ids.copy()
                for i in span[step:]:
                    masked[i] = self.tok.mask_token_id
                configs.append((ti, step, pos, masked))
        results = [{} for _ in targets]
        # Bound intermediate hidden states and attention memory.
        maxlen = max(len(x[3]) for x in configs)
        chunk_size = min(32, max(1, 2048 // maxlen))
        for start in range(0, len(configs), chunk_size):
            chunk = configs[start : start + chunk_size]
            length = max(len(c[3]) for c in chunk)
            ids = torch.full((len(chunk), length), self.tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for bi, (_, _, _, seq) in enumerate(chunk):
                ids[bi, : len(seq)] = torch.tensor(seq)
                mask[bi, : len(seq)] = 1
            logits = self.masked_logits(ids, mask, torch.tensor([c[2] for c in chunk]))
            for bi, (ti, step, _, _) in enumerate(chunk):
                row, orig_ids, span = targets[ti]
                orig = orig_ids[span[step]]
                lp = torch.log_softmax(logits[bi], dim=-1)
                topids = torch.topk(logits[bi], 5).indices.tolist()
                step_result = dict(logp=lp[orig].item(), correct=topids[0] == orig, top5=orig in topids)
                if step == 0:
                    step_result["predictions"] = [[self.tok.decode([t]), float(lp[t].exp())] for t in topids]
                    if len(span) == 1 and row["type"] in ("anchor", "rnc"):
                        lem, _ = self.morph_word(row["text"])
                        space = self.tok.convert_ids_to_tokens(orig).startswith("Ġ")
                        lem_ids = sorted(set(self.lemma_ids.get((lem, space), [])) | {orig})
                        step_result["p_lemma"] = float(lp[lem_ids].exp().sum())
                results[ti][step] = step_result
        scored = []
        for (row, ids, span), steps in zip(targets, results, strict=True):
            lem, pos = self.morph_word(row["text"])
            logp = sum(s["logp"] for s in steps.values())
            scored.append(
                {
                    **row,
                    "n_tokens": len(span),
                    "multitoken": len(span) > 1,
                    "p_chain": math.exp(logp),
                    "log_p_chain": logp,
                    "p_single": math.exp(logp) if len(span) == 1 else None,
                    "p_lemma": steps[0].get("p_lemma"),
                    "correct": int(all(s["correct"] for s in steps.values())),
                    "top5_single": int(steps[0]["top5"]) if len(span) == 1 else None,
                    "top5": steps[0]["predictions"],
                    "lemma": lem,
                    "pos": pos,
                    "sentence_tokens": len(ids),
                    "word_chars": len(row["text"]),
                }
            )
        return scored


def item_key(r):
    return (r["record"], r["example_idx"], r["char_start"], r["char_end"], r["type"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["constructicon", "rnc", "outside"], default="constructicon")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    path = (
        ROOT
        / "data"
        / {"constructicon": "items.jsonl", "rnc": "rnc_items.jsonl", "outside": "outside_items.jsonl"}[args.dataset]
    )
    output = ROOT / "results" / f"affinity_{args.dataset}.jsonl"
    done = {item_key(json.loads(s)) for s in output.read_text().splitlines()} if output.exists() else set()
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    rows = [r for r in rows if r["type"] != "other" and item_key(r) not in done]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("No unscored targets")
        return
    t0 = time.time()
    scorer = Scorer(args.threads)
    encs, work, excluded = {}, [], []
    for r in rows:
        sent = r["sentence"]
        if sent not in encs:
            encs[sent] = scorer.tok(sent, return_offsets_mapping=True, truncation=False)
        enc = encs[sent]
        span = token_span_for_word(enc["offset_mapping"], r["char_start"], r["char_end"])
        reason = None
        if len(enc["input_ids"]) > 512:
            reason = "sentence_over_512"
        elif not span:
            reason = "empty_alignment"
        elif any(
            enc["offset_mapping"][i][0] < r["char_start"] or enc["offset_mapping"][i][1] > r["char_end"] for i in span
        ):
            reason = "token_crosses_word_boundary"
        if reason:
            excluded.append({"key": item_key(r), "reason": reason})
        else:
            work.append((r, enc["input_ids"], span))
    work.sort(key=lambda x: len(x[1]))
    print(f"Scoring {len(work)} targets; {len(excluded)} exclusions; {len(done)} already done", flush=True)
    with output.open("a") as f:
        for start in range(0, len(work), 16):
            scores = scorer.score_batch(work[start : start + 16])
            for r in scores:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if start % 160 == 0 or start + 16 >= len(work):
                n = min(start + 16, len(work))
                print(f"{n}/{len(work)} targets; {time.time() - t0:.1f}s", flush=True)
    ep = ROOT / "results" / f"exclusions_{args.dataset}.jsonl"
    with ep.open("a") as f:
        for r in excluded:
            f.write(json.dumps(r) + "\n")
    print(f"Completed {args.dataset} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
