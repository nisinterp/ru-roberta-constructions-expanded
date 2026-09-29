"""Verify completed target coverage, output ranges and source integrity."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from study_coverage import check_coverage

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", action="store_true", help="Verify the explicitly frozen RNC cutoff")
    args = parser.parse_args()
    counts, coverage = {}, {}
    for dataset, input_name in [
        ("constructicon", "items.jsonl"),
        ("rnc", "rnc_items.jsonl"),
        ("outside", "outside_items.jsonl"),
    ]:
        coverage[dataset] = check_coverage(
            dataset, input_name, allow_cutoff=args.cutoff and dataset == "rnc", root=ROOT
        )
        frame = pd.read_json(ROOT / "results" / f"affinity_{dataset}.jsonl", lines=True)
        assert frame.p_chain.between(0, 1).all()
        assert frame.correct.isin([0, 1]).all()
        single = frame[frame.n_tokens == 1]
        assert np.allclose(single.p_chain, single.p_single, atol=1e-12)
        assert np.isfinite(frame.log_p_chain).all()
        assert np.allclose(np.exp(frame.log_p_chain), frame.p_chain, atol=1e-12)
        for metric in ["p_single", "p_lemma"]:
            # Small float32 summation error can put lemma mass just above one.
            assert frame[metric].dropna().between(0, 1 + 1e-6).all()
        assert single.top5_single.isin([0, 1]).all()
        assert (single.correct <= single.top5_single).all()
        counts[dataset] = len(frame)
    manifest = json.loads((ROOT / "sources/manifest.json").read_text())
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        with path.open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == entry["sha256"], path
    for path in [ROOT / "results/summary.json", ROOT / "results/outside_summary.json"]:
        json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    summary = json.loads((ROOT / "results/summary.json").read_text())
    assert summary["counts"]["rnc_targets"] == counts["rnc"]
    if args.cutoff:
        cutoff = json.loads((ROOT / "results/cutoff_analysis.json").read_text())
        assert summary["analysis_mode"] == "user_requested_cutoff"
        assert cutoff["coverage"] == coverage["rnc"] == summary["rnc_scoring_coverage"]
        report = (ROOT / "report/findings.md").read_text()
        assert "partially scored corpus baseline" in report
        assert "8.1. AI declaration" in report
    pdf = ROOT / "report/findings.pdf"
    assert pdf.read_bytes().startswith(b"%PDF-") and pdf.stat().st_size > 10000
    result = dict(
        analysis_mode="user_requested_cutoff" if args.cutoff else "complete_run",
        coverage=coverage,
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
