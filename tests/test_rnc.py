"""Check the exclusions that define the corpus comparison."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_rnc import filter_rows
from prepare import sentence_key


def row(sentence, word):
    start = sentence.index(word)
    return dict(text=sentence, char_start=start, char_end=start + len(word), form=word)


def test_corpus_filter_checks_identity_overlap_and_coanchors():
    examples = [row("Он знает ответ.", "знает"),
                row("Бог знает ответ.", "знает"),
                row("Она знает ответ.", "знает"),
                row("Он знает ответ.", "знает"),
                row("Он читает ответ.", "читает")]
    kept, rejected = filter_rows(examples, "знает", [{"бог"}],
                                 {sentence_key("Она знает ответ.")}, set())
    assert [r["text"] for r in kept] == ["Он знает ответ."]
    assert rejected == {"coanchor_match": 1, "duplicate_or_overlap": 2, "wrong_form_or_offset": 1}


def test_single_anchor_is_not_automatically_rejected():
    examples = [row("Он знает ответ.", "знает")]
    kept, rejected = filter_rows(examples, "знает", [set()], set(), set())
    assert len(kept) == 1
    assert not rejected
