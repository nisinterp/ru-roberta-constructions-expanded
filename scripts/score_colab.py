"""Score an exported RNC subset on CUDA, preserving the existing fp32 method.

The original scoring implementation stays unchanged. Only the forward pass moves
to the GPU; its masked-position logits return to the CPU for identical bookkeeping.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch
import transformers
from colab_io import check_prediction, digest, key, read_complete_rows
from score_expanded import Scorer, ROOT


class GPUScorer(Scorer):
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError('Select a GPU runtime in Colab before running this cell.')
        super().__init__(threads=2)
        self.device = torch.device('cuda')
        self.model.to(self.device)
        # Keep fp32 arithmetic; do not silently introduce mixed precision.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    @torch.inference_mode()
    def masked_logits(self, ids, attention, positions):
        ids, attention, positions = ids.to(self.device), attention.to(self.device), positions.to(self.device)
        hidden = self.model.roberta(input_ids=ids, attention_mask=attention).last_hidden_state
        logits = self.model.lm_head(hidden[torch.arange(len(ids), device=self.device), positions])
        return logits.cpu()


def targets(path):
    return [(entry['row'], entry['ids'], entry['span']) for entry in read_complete_rows(path)]


def representative(work, count):
    ordered = sorted(work, key=lambda item: len(item[1]))
    if len(ordered) <= count:
        return ordered
    return [ordered[round(i * (len(ordered) - 1) / (count - 1))] for i in range(count)]


def validate(scorer, work):
    refs = targets(ROOT / 'reference.jsonl')
    for start in range(0, len(refs), 16):
        batch = refs[start:start + 16]
        for reference, score in zip(batch, scorer.score_batch(batch), strict=True):
            check_prediction(reference[0], score)
    # Also test long unfinished contexts, which may not occur in the CPU snapshot.
    sample = representative(work, 8)
    scorer.device = torch.device('cpu')
    scorer.model.to(scorer.device)
    cpu = scorer.score_batch(sample)
    scorer.device = torch.device('cuda')
    scorer.model.to(scorer.device)
    gpu = scorer.score_batch(sample)
    for a, b in zip(cpu, gpu, strict=True):
        check_prediction(a, b)
    return dict(passed=True, saved_cpu_references=len(refs), remaining_context_checks=len(sample),
                tolerance_log_absolute=5e-4, tolerance_log_relative=2e-5)


def benchmark(scorer, work):
    # Spread the sample across the remaining length range, including its long tail.
    sample = representative(work, 160)
    scorer.score_batch(sample[:4])
    begin = time.perf_counter()
    for start in range(0, len(sample), 16):
        scorer.score_batch(sample[start:start + 16])
    seconds = time.perf_counter() - begin
    return dict(targets=len(sample), seconds=seconds, targets_per_second=len(sample)/seconds,
                estimated_scoring_hours=len(work)*seconds/len(sample)/3600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='A persistent Google Drive directory')
    parser.add_argument('--benchmark-only', action='store_true')
    args = parser.parse_args()
    meta = json.loads((ROOT / 'transfer.json').read_text())
    for name, expected in meta['code_hashes'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Transfer code changed: {name}')
    for name in ('pending', 'reference'):
        if digest(ROOT / f'{name}.jsonl') != meta[f'{name}_sha256']:
            raise ValueError(f'Transfer input changed: {name}')
    for entry in meta['model_files']:
        if digest(ROOT / entry['path']) != entry['sha256']:
            raise ValueError(f'Model file differs from the CPU run: {entry["path"]}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity = dict(transfer_sha256=digest(ROOT / 'transfer.json'), dtype='float32',
                    torch=torch.__version__, transformers=transformers.__version__,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    metadata = args.output_dir / 'gpu_metadata.json'
    output = args.output_dir / 'gpu_scores.jsonl'
    if output.exists() and (not metadata.exists() or json.loads(metadata.read_text()) != identity):
        raise ValueError('Output belongs to a different bundle or environment. Use a separate output folder.')
    metadata.write_text(json.dumps(identity, indent=2))
    work = targets(ROOT / 'pending.jsonl')
    scorer = GPUScorer()
    checks = validate(scorer, work)
    checks['benchmark'] = benchmark(scorer, work)
    checks['completed_utc'] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / 'gpu_validation.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2), flush=True)
    if args.benchmark_only:
        return
    # An interrupted last line must be removed explicitly, never silently ignored.
    if output.exists() and output.stat().st_size:
        with output.open('rb') as source:
            source.seek(-1, 2)
            if source.read(1) != b'\n':
                raise ValueError('Incomplete final output line: back up the file and remove only that line.')
    done = {key(row) for row in read_complete_rows(output)} if output.exists() else set()
    all_keys = {key(item[0]) for item in work}
    if not done <= all_keys:
        raise ValueError('Unexpected targets in existing output')
    work = [item for item in work if key(item[0]) not in done]
    work.sort(key=lambda item: len(item[1]))
    begin = time.perf_counter()
    with output.open('a') as destination:
        for start in range(0, len(work), 16):
            scores = scorer.score_batch(work[start:start + 16])
            destination.write(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in scores))
            destination.flush()
            if start % 160 == 0 or start + 16 >= len(work):
                print(f'{min(start+16,len(work))}/{len(work)} targets; {time.perf_counter()-begin:.1f}s', flush=True)
    result_keys = [key(row) for row in read_complete_rows(output)]
    if set(result_keys) != all_keys or len(result_keys) != len(all_keys):
        raise ValueError('Final GPU target coverage is incomplete or duplicated')
    (args.output_dir / 'gpu_completion.json').write_text(json.dumps(
        dict(complete=True, targets=len(result_keys), output_sha256=digest(output), **identity), indent=2))
    print('GPU scoring complete. Return the result archive for validation and merging.', flush=True)


if __name__ == '__main__':
    main()
