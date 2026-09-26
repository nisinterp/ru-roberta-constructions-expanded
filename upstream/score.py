"""Score global affinity for every anchor/filler word (constructicon) and every
target word (RNC baseline) with ai-forever/ruRoberta-large.

Usage:
  uv run scripts/score.py constructicon
  uv run scripts/score.py rnc
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pymorphy3
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from affinity import build_lemma_table, token_span_for_word  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "ai-forever/ruRoberta-large"
REVISION_FILE = ROOT / "results" / "model_revision.txt"
MAX_LEN = 512
BATCH = 32
TOPK = 5

torch.set_num_threads(4)


def resolve_revision() -> str:
    if REVISION_FILE.exists():
        return REVISION_FILE.read_text().strip()
    from huggingface_hub import model_info

    sha = model_info(MODEL_NAME).sha
    REVISION_FILE.parent.mkdir(parents=True, exist_ok=True)
    REVISION_FILE.write_text(sha + "\n")
    return sha


class Scorer:
    def __init__(self) -> None:
        rev = resolve_revision()
        print(f"model {MODEL_NAME} @ {rev}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=rev)
        if not self.tok.is_fast:
            sys.exit("fast tokenizer required (offset_mapping)")
        self.model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, revision=rev)
        self.model.eval()
        self.morph = pymorphy3.MorphAnalyzer()
        table = build_lemma_table(self.tok, self.morph)
        self.lemma_ids: dict[tuple[str, bool], list[int]] = defaultdict(list)
        for tid, lem in table.items():
            has_space = self.tok.convert_ids_to_tokens(tid).startswith("Ġ")
            self.lemma_ids[(lem.replace("ё", "е"), has_space)].append(tid)

    def lemma_of(self, word: str) -> str:
        return self.morph.parse(word.lower())[0].normal_form.replace("ё", "е")

    def pos_of(self, word: str) -> str:
        return str(self.morph.parse(word.lower())[0].tag.POS or "NONE")

    @torch.no_grad()
    def score_sentence(
        self, text: str, targets: list[dict], want_lemma: list[bool]
    ) -> list[dict | None]:
        """targets: dicts with char_start/char_end. Returns one result per target
        (None if excluded, with reason in result['_excluded'])."""
        enc = self.tok(
            text, return_offsets_mapping=True, return_tensors="pt", truncation=False
        )
        ids = enc["input_ids"][0]
        attn = enc["attention_mask"][0]
        if ids.shape[0] > MAX_LEN:
            return [
                {"_excluded": f"sentence longer than {MAX_LEN} tokens ({ids.shape[0]})"}
                for _ in targets
            ]
        offsets = enc["offset_mapping"][0].tolist()

        # (target_index, step, masked_positions, score_position)
        configs = []
        spans = []
        for ti, t in enumerate(targets):
            idxs = token_span_for_word(offsets, t["char_start"], t["char_end"])
            spans.append(idxs)
            for step, pos in enumerate(idxs):
                configs.append((ti, step, idxs[step:], pos))

        probs_at: dict[tuple[int, int], torch.Tensor] = {}
        for b in range(0, len(configs), BATCH):
            chunk = configs[b : b + BATCH]
            batch_ids = ids.unsqueeze(0).repeat(len(chunk), 1)
            for row, (_, _, masked, _) in enumerate(chunk):
                batch_ids[row, masked] = self.tok.mask_token_id
            logits = self.model(
                input_ids=batch_ids,
                attention_mask=attn.unsqueeze(0).repeat(len(chunk), 1),
            ).logits
            for row, (ti, step, _, pos) in enumerate(chunk):
                probs_at[(ti, step)] = torch.softmax(logits[row, pos], dim=-1)

        results: list[dict | None] = []
        for ti, t in enumerate(targets):
            idxs = spans[ti]
            if not idxs:
                results.append({"_excluded": "no tokens aligned to word span"})
                continue
            orig = [ids[i].item() for i in idxs]
            chain = 1.0
            for step in range(len(idxs)):
                chain *= probs_at[(ti, step)][orig[step]].item()
            first = probs_at[(ti, 0)]
            top = torch.topk(first, TOPK)
            res = {
                "n_tokens": len(idxs),
                "multitoken": len(idxs) > 1,
                "p_single": first[orig[0]].item() if len(idxs) == 1 else None,
                "p_chain": chain,
                "p_lemma": None,
                "top5": [
                    [
                        self.tok.convert_tokens_to_string(
                            [self.tok.convert_ids_to_tokens(i.item())]
                        ),
                        round(v.item(), 5),
                    ]
                    for i, v in zip(top.indices, top.values, strict=True)
                ],
                "lemma": self.lemma_of(t["text"]),
                "pos": self.pos_of(t["text"]),
            }
            if want_lemma[ti] and len(idxs) == 1:
                has_space = self.tok.convert_ids_to_tokens(orig[0]).startswith("Ġ")
                lemma_ids = set(self.lemma_ids.get((res["lemma"], has_space), [])) | {
                    orig[0]
                }
                res["p_lemma"] = first[sorted(lemma_ids)].sum().item()
            results.append(res)
        return results


def run_constructicon(scorer: Scorer) -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / "data" / "items.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    by_sent: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_sent[(r["record"], r["example_idx"])].append(r)
    out_path = ROOT / "results" / "affinity_constructicon.jsonl"
    excl_path = ROOT / "results" / "score_exclusions_constructicon.log"
    excl = []
    n = 0
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as out:
        for si, (key, words) in enumerate(by_sent.items(), 1):
            text = words[0]["sentence"]
            targets = [w for w in words if w["type"] in ("anchor", "filler")]
            others = [w for w in words if w["type"] == "other"]
            res = scorer.score_sentence(
                text, targets, [w["type"] == "anchor" for w in targets]
            )
            for w, r in zip(targets, res, strict=True):
                if "_excluded" in r:
                    excl.append(
                        f"record {key[0]} ex {key[1]} word {w['text']!r}: {r['_excluded']}"
                    )
                    continue
                out.write(json.dumps({**w, **r}, ensure_ascii=False) + "\n")
                n += 1
            # other words: token counts only (not scored)
            enc = scorer.tok(text, return_offsets_mapping=True)
            for w in others:
                k = len(
                    token_span_for_word(
                        enc["offset_mapping"], w["char_start"], w["char_end"]
                    )
                )
                out.write(
                    json.dumps(
                        {
                            **w,
                            "n_tokens": k,
                            "multitoken": k > 1,
                            "pos": scorer.pos_of(w["text"]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if si % 10 == 0:
                print(
                    f"  {si}/{len(by_sent)} sentences, {time.time() - t0:.0f}s",
                    flush=True,
                )
    excl_path.write_text("\n".join(excl) + "\n", encoding="utf-8")
    print(f"scored {n} words -> {out_path}; exclusions: {len(excl)}")


def run_rnc(scorer: Scorer) -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / "data" / "rnc_items.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    out_path = ROOT / "results" / "affinity_rnc.jsonl"
    excl_path = ROOT / "results" / "score_exclusions_rnc.log"
    excl = []
    n = 0
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as out:
        for i, r in enumerate(rows, 1):
            target = {
                "char_start": r["char_start"],
                "char_end": r["char_end"],
                "text": r["text"][r["char_start"] : r["char_end"]],
            }
            res = scorer.score_sentence(r["text"], [target], [True])[0]
            if "_excluded" in res:
                excl.append(f"{r['form']} {r['doc_id']}: {res['_excluded']}")
                continue
            out.write(
                json.dumps({**r, **res, "type": "rnc"}, ensure_ascii=False) + "\n"
            )
            n += 1
            if i % 200 == 0:
                print(f"  {i}/{len(rows)}, {time.time() - t0:.0f}s", flush=True)
    excl_path.write_text("\n".join(excl) + "\n", encoding="utf-8")
    print(f"scored {n} RNC targets -> {out_path}; exclusions: {len(excl)}")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "constructicon"
    scorer = Scorer()
    if which == "constructicon":
        run_constructicon(scorer)
    elif which == "rnc":
        run_rnc(scorer)
    else:
        sys.exit(f"unknown target {which}")


if __name__ == "__main__":
    main()
