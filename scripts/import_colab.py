"""Validate a returned GPU shard and merge it after local scoring has stopped."""

import argparse
import json
from pathlib import Path
import shutil

from colab_io import check_prediction, digest, key, read_complete_rows

ROOT = Path(__file__).resolve().parents[1]


def assert_idle():
    # Existing CPU scripts predate the GPU merge lock. Check this checkout's workers.
    for process in Path('/proc').glob('[0-9]*'):
        try:
            command = (process / 'cmdline').read_bytes().split(b'\0')
            if any(Path(arg.decode()).name in ('score_expanded.py', 'run_study.py') for arg in command if arg):
                if (process / 'cwd').resolve() == ROOT:
                    raise RuntimeError('Stop this checkout\'s scorer and study runner before merging GPU results.')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path)
    args = parser.parse_args()
    assert_idle()
    folder = ROOT / 'data/colab_transfer'
    meta = json.loads((folder / 'transfer.json').read_text())
    completion = json.loads((args.result_dir / 'gpu_completion.json').read_text())
    validation = json.loads((args.result_dir / 'gpu_validation.json').read_text())
    if not validation['passed'] or not completion['complete']:
        raise ValueError('GPU run did not pass validation and completion checks')
    if completion['transfer_sha256'] != digest(folder / 'transfer.json'):
        raise ValueError('Returned results belong to another transfer')
    shard = args.result_dir / 'gpu_scores.jsonl'
    if digest(shard) != completion['output_sha256']:
        raise ValueError('Returned output checksum differs')
    if digest(ROOT / 'data/rnc_items.jsonl') != meta['input_sha256']:
        raise ValueError('Current RNC inputs differ from the transfer')
    output = ROOT / 'results/affinity_rnc.jsonl'
    expected = {key(row['row']) for row in read_complete_rows(folder / 'pending.jsonl')}
    returned = list(read_complete_rows(shard))
    if {key(row) for row in returned} != expected or len(returned) != len(expected):
        raise ValueError('GPU shard contains missing, extra, or duplicate targets')
    originals = {key(row): row for row in read_complete_rows(ROOT / 'data/rnc_items.jsonl')}
    for row in returned:
        if any(row.get(field) != value for field, value in originals[key(row)].items()):
            raise ValueError(f'GPU output changed source fields for {key(row)}')
    existing = list(read_complete_rows(output))
    by_key = {key(row): row for row in existing}
    if len(existing) != len(by_key):
        raise ValueError('Existing CPU results contain duplicates')
    additions, overlap = [], 0
    for row in returned:
        if key(row) in by_key:
            check_prediction(by_key[key(row)], row)
            overlap += 1
        else:
            additions.append(row)
    # Preserve the CPU file before replacing it, and never append while a worker runs.
    backup = output.with_name('affinity_rnc_before_gpu.jsonl')
    if backup.exists():
        raise ValueError('A previous merge backup exists; inspect it before another merge')
    shutil.copy2(output, backup)
    temporary = output.with_suffix('.partial')
    with temporary.open('w') as destination:
        for row in existing + additions:
            destination.write(json.dumps(row, ensure_ascii=False) + '\n')
    assert_idle()
    temporary.replace(output)
    provenance = dict(cpu_preserved=len(existing), gpu_added=len(additions), checked_overlap=overlap,
                      gpu_environment=completion, validation=validation, merged_sha256=digest(output))
    (ROOT / 'results/gpu_merge.json').write_text(json.dumps(provenance, indent=2))
    from analyze_expanded import require_complete
    require_complete('rnc', 'rnc_items.jsonl')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
