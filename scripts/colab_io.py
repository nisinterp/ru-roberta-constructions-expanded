"""Small file helpers shared by the Colab exporter, scorer, and importer."""

import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def key(row):
    return tuple(row[name] for name in ('record', 'example_idx', 'char_start', 'char_end', 'type'))


def read_complete_rows(path):
    """Snapshot complete records while the CPU scorer may still be appending."""
    with Path(path).open('rb') as source:
        for line in source:
            if not line.endswith(b'\n'):
                return
            yield json.loads(line)


def check_prediction(reference, candidate):
    """Require matching recovery decisions and close fp32 log probabilities."""
    import math
    for name in ('correct', 'top5_single', 'n_tokens', 'pos', 'lemma'):
        if reference[name] != candidate[name]:
            raise ValueError(f'CPU/GPU mismatch in {name} for target {key(reference)}')
    if not math.isclose(reference['log_p_chain'], candidate['log_p_chain'], abs_tol=5e-4, rel_tol=2e-5):
        raise ValueError(f'CPU/GPU log-probability mismatch for target {key(reference)}')
    for name in ('p_chain', 'p_single', 'p_lemma'):
        a, b = reference.get(name), candidate.get(name)
        if (a is None) != (b is None) or (a is not None and not math.isclose(a, b, abs_tol=2e-5, rel_tol=5e-4)):
            raise ValueError(f'CPU/GPU probability mismatch in {name} for target {key(reference)}')


def model_artifacts(entries):
    """Select checkpoint/tokenizer files, excluding downloaded Hub API metadata."""
    required = {'config.json', 'merges.txt', 'vocab.json', 'pytorch_model.bin'}
    files = [entry for entry in entries
             if entry['path'] in {f'data/raw/model/{name}' for name in required}]
    if len(files) != len(required) or {Path(entry['path']).name for entry in files} != required:
        raise ValueError('Manifest must contain each required model artifact exactly once')
    return files
