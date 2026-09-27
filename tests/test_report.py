"""Exercise PDF layout with historical data in a temporary, clearly marked fixture."""

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_report_renders_from_complete_fixture(tmp_path, monkeypatch):
    import analyze_expanded
    import build_report

    # The fixture tests rendering only. It is never written to the study's report directory.
    shutil.copytree(ROOT / "archive/300-constructions/results", tmp_path / "results")
    shutil.copytree(ROOT / "sources", tmp_path / "sources")
    audit_path = tmp_path / "results/data_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["eligible_constructions"] = audit["selected_constructions"]
    audit["represented_constructions"] = audit["selected_constructions"]
    audit_path.write_text(json.dumps(audit))
    collection = dict(requested_forms=27, covered_forms=27, per_form=50, max_pages=3,
                      completed_utc="LAYOUT TEST ONLY")
    (tmp_path / "results/rnc_collection.json").write_text(json.dumps(collection))
    monkeypatch.setattr(analyze_expanded, "require_complete", lambda *args: None)
    for name, path in [("ROOT", tmp_path), ("RES", tmp_path / "results"),
                       ("FIG", tmp_path / "results/figures"), ("REPORT", tmp_path / "report")]:
        monkeypatch.setattr(build_report, name, path)
    build_report.main()
    pdf = tmp_path / "report/findings.pdf"
    assert pdf.read_bytes().startswith(b"%PDF-")
    assert pdf.stat().st_size > 10000
    text = (tmp_path / "report/findings.md").read_text()
    assert "LAYOUT TEST ONLY" in text
    assert "9. Conclusions" in text
