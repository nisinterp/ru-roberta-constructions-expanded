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


def test_incomplete_snippet_is_rejected_without_fabricating_text():
    from collect_rnc import parse_snippet, PARSER_EXCLUSIONS
    snippet = {"sequences": [{"words": [
        {"type": "WORD", "source": {"sentId": 1}, "displayParams": {"hit": True}}
    ]}]}
    PARSER_EXCLUSIONS.clear()
    assert parse_snippet(snippet) is None
    assert PARSER_EXCLUSIONS["snippet_missing_text"] == 1


def test_complete_snippet_keeps_exact_target_offsets():
    from collect_rnc import parse_snippet
    snippet = {"sequences": [{"words": [
        {"type": "WORD", "text": "Он", "source": {"sentId": 1}},
        {"type": "SPACE", "text": " "},
        {"type": "WORD", "text": "знает", "source": {"sentId": 1}, "displayParams": {"hit": True}},
        {"type": "PUNCT", "text": "."}
    ]}]}
    sentence, start, end, _ = parse_snippet(snippet)
    assert sentence == "Он знает."
    assert sentence[start:end] == "знает"
