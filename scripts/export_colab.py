"""Package unfinished RNC targets without credentials or model weights."""

from collections import defaultdict
import json
import random
import shutil
import zipfile

from transformers import AutoTokenizer
from colab_io import digest, key, read_complete_rows
from score_expanded import ROOT, token_span_for_word


def main():
    folder = ROOT / 'data/colab_transfer'
    if folder.exists():
        raise SystemExit('A transfer bundle already exists. Preserve it before creating a new snapshot.')
    folder.mkdir(parents=True)
    groups = defaultdict(list)
    done = set()
    for row in read_complete_rows(ROOT / 'results/affinity_rnc.jsonl'):
        done.add(key(row))
        groups[min(row['n_tokens'], 4)].append(row)
    excluded = {tuple(json.loads(line)['key']) for line in (ROOT / 'results/exclusions_rnc.jsonl').read_text().splitlines()}
    rng = random.Random(20260928)
    references = [r for group in groups.values() for r in rng.sample(group, min(8, len(group)))]
    tok = AutoTokenizer.from_pretrained(ROOT / 'data/raw/model', local_files_only=True)

    def encode(row):
        encoded = tok(row['sentence'], return_offsets_mapping=True, truncation=False)
        span = token_span_for_word(encoded['offset_mapping'], row['char_start'], row['char_end'])
        if len(encoded['input_ids']) > 512 or not span:
            raise ValueError('Unexpected invalid target in export')
        if any(encoded['offset_mapping'][i][0] < row['char_start'] or encoded['offset_mapping'][i][1] > row['char_end'] for i in span):
            raise ValueError('Unexpected word/token boundary conflict')
        return dict(row=row, ids=encoded['input_ids'], span=span)

    pending = 0
    with (folder / 'pending.jsonl').open('w') as target:
        for row in read_complete_rows(ROOT / 'data/rnc_items.jsonl'):
            if key(row) not in done and key(row) not in excluded:
                target.write(json.dumps(encode(row), ensure_ascii=False) + '\n')
                pending += 1
    with (folder / 'reference.jsonl').open('w') as target:
        for row in references:
            target.write(json.dumps(encode(row), ensure_ascii=False) + '\n')
    manifest = json.loads((ROOT / 'sources/manifest.json').read_text())
    meta = dict(pending_targets=pending, completed_cpu_targets=len(done), reference_targets=len(references),
                input_sha256=digest(ROOT / 'data/rnc_items.jsonl'),
                scorer_sha256=digest(ROOT / 'scripts/score_expanded.py'), model_revision=manifest['model_revision'],
                pending_sha256=digest(folder / 'pending.jsonl'), reference_sha256=digest(folder / 'reference.jsonl'),
                model_files=[r for r in manifest['files'] if r['path'].startswith('data/raw/model/')])
    for relative in ('scripts/score_expanded.py', 'scripts/score_colab.py', 'scripts/colab_io.py', 'upstream/affinity.py'):
        destination = folder / relative
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    meta['code_hashes'] = {name: digest(folder / name) for name in
                           ('scripts/score_expanded.py', 'scripts/score_colab.py', 'scripts/colab_io.py', 'upstream/affinity.py')}
    (folder / 'transfer.json').write_text(json.dumps(meta, indent=2))
    archive = ROOT / 'data/colab_transfer.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                output.write(path, path.relative_to(folder))
    print(json.dumps({'pending_targets': pending, 'cpu_reference_targets':len(references),
                      'zip_bytes':archive.stat().st_size, 'bundle':str(archive)}, indent=2))


if __name__ == '__main__':
    main()
