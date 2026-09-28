"""Check snapshot boundaries and CPU/GPU result validation without requiring CUDA."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from colab_io import read_complete_rows, check_prediction
from score_colab import representative


def prediction():
    return dict(record=-1, example_idx=1, char_start=0, char_end=1, type='rnc',
                correct=1, top5_single=1, n_tokens=1, pos='NOUN', lemma='дом',
                log_p_chain=-0.1, p_chain=0.904837, p_single=0.904837, p_lemma=0.99)


def test_snapshot_ignores_only_unfinished_last_row(tmp_path):
    path = tmp_path / 'scores.jsonl'
    path.write_text(json.dumps(prediction())+'\n'+ '{"unfinished":')
    assert list(read_complete_rows(path)) == [prediction()]
    path.write_text('{"malformed":\n')
    with pytest.raises(json.JSONDecodeError):
        list(read_complete_rows(path))


def test_validation_rejects_changed_predictions_and_probabilities():
    reference = prediction()
    check_prediction(reference, reference | {'log_p_chain': -0.100001})
    for candidate in [reference | {'correct': 0}, reference | {'log_p_chain': -0.5},
                      reference | {'p_lemma': None}]:
        with pytest.raises(ValueError):
            check_prediction(reference, candidate)


def test_benchmark_covers_long_and_short_remaining_contexts():
    work = [({}, list(range(i)), [0]) for i in range(2, 202)]
    sample = representative(work, 10)
    assert len(sample) == 10
    assert len(sample[0][1]) == 2
    assert len(sample[-1][1]) == 201


def test_notebook_code_cells_compile():
    notebook = json.loads((Path(__file__).resolve().parents[1] / 'notebooks/score_rnc_colab.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), 'colab-cell', 'exec')
