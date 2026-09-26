import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "upstream"))
from affinity import global_affinity_chain, token_span_for_word
from prepare import anchors, original
from score_expanded import Scorer


def test_anchor_tags_and_hyphenation():
    assert anchors("Сделать-Imp милость NP-Dat") == ["милость", "сделать"]
    assert anchors("худо-бедно VP") == ["худо-бедно"]


def test_slot_offsets_and_roles():
    text, slots = original.strip_brackets("У [Петра]Agent получилось [уйти]Result.")
    assert text == "У Петра получилось уйти."
    assert [text[s:e] for s, e, _ in slots] == ["Петра", "уйти"]
    words = original.classify_words(original.split_words(text), slots, {"у", "получилось"}, set())
    assert [w["type"] for w in words] == ["anchor", "filler", "anchor", "filler"]


@pytest.fixture(scope="module")
def scorer():
    return Scorer()


def test_optimized_scoring_matches_original(scorer):
    sentence = "Бог его знает, где достопримечательности."
    enc = scorer.tok(sentence, return_offsets_mapping=True)
    targets = []
    for word in ["знает", "достопримечательности"]:
        start = sentence.index(word)
        row = dict(sentence=sentence, text=word, char_start=start, char_end=start + len(word), type="anchor")
        span = token_span_for_word(enc["offset_mapping"], start, start + len(word))
        targets.append((row, enc["input_ids"], span))
    scores = scorer.score_batch(targets)
    for (_, ids, span), result in zip(targets, scores, strict=True):
        ids = torch.tensor([ids])
        ref = global_affinity_chain(scorer.model, scorer.tok, ids, torch.ones_like(ids), span)
        assert result["p_chain"] == pytest.approx(ref, abs=2e-6, rel=2e-4)
        if len(span) == 1:
            assert result["p_single"] == result["p_chain"]
            assert result["p_lemma"] >= result["p_single"] - 2e-6
        # Verify gold-prefix all-correct equals actual autoregressive greedy exact recovery.
        current = ids.clone()
        current[0, span] = scorer.tok.mask_token_id
        predictions = []
        with torch.inference_mode():
            for pos in span:
                logits = scorer.masked_logits(current, torch.ones_like(current), torch.tensor([pos]))
                predicted = int(logits[0].argmax())
                predictions.append(predicted)
                current[0, pos] = predicted
        assert result["correct"] == int(predictions == [int(ids[0, pos]) for pos in span])


def test_padding_preserves_scores(scorer):
    targets = []
    for sentence, word in [("Это хорошо.", "хорошо"), ("Я не знаю, что он думает об этом сейчас.", "думает")]:
        enc = scorer.tok(sentence, return_offsets_mapping=True)
        start = sentence.index(word)
        span = token_span_for_word(enc["offset_mapping"], start, start + len(word))
        targets.append((dict(text=word, type="anchor"), enc["input_ids"], span))
    together = scorer.score_batch(targets)
    for target, combined in zip(targets, together, strict=True):
        alone = scorer.score_batch([target])[0]
        assert combined["p_chain"] == pytest.approx(alone["p_chain"], abs=2e-6, rel=2e-4)
        assert combined["correct"] == alone["correct"]
