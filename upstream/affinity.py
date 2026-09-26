"""Global affinity computation for masked language models (ruRoberta-large).

Implements the "global affinity" metric from Rozner, Weissweiler, Mahowald &
Shain (EMNLP 2025): the probability the model assigns to the original word
when only that word is masked.

Two variants:
- single: word must be exactly one token; P = softmax(logits)[orig_id].
- chain: word may span multiple tokens; mask all of them, then reveal true
  tokens left-to-right one at a time, re-running the model each step
  (PLL-word-l2r style). At k=1 (single-token word) this must equal `single`.

`build_lemma_table` supports the lemma variant computed in score.py.
Batched scoring lives in score.py and is tested against these reference
implementations.
"""

from __future__ import annotations

import torch


def _softmax_row(logits_row: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits_row, dim=-1)


def token_span_for_word(
    offsets: list[tuple[int, int]], word_start: int, word_end: int
) -> list[int]:
    """Indices of tokens whose char span overlaps [word_start, word_end)."""
    idxs = []
    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue  # special tokens (offset (0,0))
        if s < word_end and e > word_start:
            idxs.append(i)
    return idxs


@torch.no_grad()
def global_affinity_single(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_idx: int,
) -> float:
    """P(orig token | rest of sentence), masking exactly one token."""
    orig_id = input_ids[0, token_idx].item()
    masked = input_ids.clone()
    masked[0, token_idx] = tokenizer.mask_token_id
    out = model(input_ids=masked, attention_mask=attention_mask)
    probs = _softmax_row(out.logits[0, token_idx])
    return probs[orig_id].item()


@torch.no_grad()
def global_affinity_chain(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_idxs: list[int],
) -> float:
    """PLL-word-l2r: reveal word's subtokens left-to-right, multiply probs.

    At each step i, tokens token_idxs[:i] are true, token_idxs[i] is
    masked and scored, token_idxs[i+1:] remain masked.
    """
    orig_ids = [input_ids[0, i].item() for i in token_idxs]
    prob = 1.0
    current = input_ids.clone()
    for i in token_idxs:
        current[0, i] = tokenizer.mask_token_id
    for step, idx in enumerate(token_idxs):
        out = model(input_ids=current, attention_mask=attention_mask)
        probs = _softmax_row(out.logits[0, idx])
        prob *= probs[orig_ids[step]].item()
        current[0, idx] = orig_ids[step]  # reveal true token before next step
    return prob


def build_lemma_table(tokenizer, morph) -> dict[int, str]:
    """token_id -> lemma, for single-token vocabulary entries decodable to a
    single alphabetic Cyrillic word (with or without leading space)."""
    table: dict[int, str] = {}
    vocab_size = len(tokenizer)
    for tid in range(vocab_size):
        tok_str = tokenizer.convert_tokens_to_string(
            [tokenizer.convert_ids_to_tokens(tid)]
        )
        word = tok_str.strip()
        if not word or not all(
            ("а" <= ch.lower() <= "я") or ch.lower() == "ё" for ch in word
        ):
            continue
        lemma = morph.parse(word)[0].normal_form
        table[tid] = lemma
    return table
