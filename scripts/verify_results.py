"""Verify completed target coverage, output ranges and source integrity."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_expanded import require_complete

ROOT = Path(__file__).resolve().parents[1]


def main():
    counts = {}
    for dataset, input_name in [
        ("constructicon", "items.jsonl"),
        ("rnc", "rnc_items.jsonl"),
        ("outside", "outside_items.jsonl"),
    ]:
        require_complete(dataset, input_name)
        frame = pd.read_json(ROOT / "results" / f"affinity_{dataset}.jsonl", lines=True)
        assert frame.p_chain.between(0, 1).all()
        assert frame.correct.isin([0, 1]).all()
        single = frame[frame.n_tokens == 1]
        assert np.allclose(single.p_chain, single.p_single, atol=1e-12)
        assert np.isfinite(frame.log_p_chain).all()
        counts[dataset] = len(frame)
    manifest = json.loads((ROOT / "sources/manifest.json").read_text())
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        with path.open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == entry["sha256"], path
    for path in [ROOT / "results/summary.json", ROOT / "results/outside_summary.json"]:
        json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    pdf = ROOT / "report/findings.pdf"
    assert pdf.read_bytes().startswith(b"%PDF-") and pdf.stat().st_size > 10000
    result = dict(
        coverage="complete, no duplicate targets",
        scored_targets=counts,
        sources_verified=len(manifest["files"]),
        numeric_ranges="passed",
        summaries="valid finite JSON",
        pdf_bytes=pdf.stat().st_size,
    )
    (ROOT / "results/verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
