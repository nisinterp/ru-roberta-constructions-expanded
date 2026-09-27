"""Run the full study, keep progress on disk, and stop if any required step fails."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "results/run_status.json"


def status(stage, state, **details):
    temporary = STATUS.with_suffix(".partial")
    temporary.write_text(json.dumps(dict(stage=stage, state=state,
                                         updated_utc=datetime.now(timezone.utc).isoformat(), **details), indent=2))
    temporary.replace(STATUS)


def wait_for(pid, stage):
    """An optional handoff lets this runner adopt jobs already started locally."""
    if not pid:
        return
    status(stage, "waiting_for_existing_job")
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(10)


def run(script, *args):
    status(script, "running")
    print(f"Running {script}", flush=True)
    # Append logs so a resumed run preserves the earlier error and progress history.
    with (ROOT / "results" / f"{Path(script).stem}.log").open("a") as log:
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Keep prepared data and reuse completed score batches")
    parser.add_argument("--wait-scoring-pid", type=int)
    parser.add_argument("--wait-collection-pid", type=int)
    args = parser.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    # Only one runner may write reports and start scoring jobs at a time.
    import fcntl
    with (ROOT / "results/.run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if not args.resume:
                if list((ROOT / "results").glob("affinity_*.jsonl")):
                    raise RuntimeError("Existing scores found. Use --resume or archive the existing run first.")
                run("download_data.py")
                run("prepare.py", "--constructions", "0")
                run("audit_spans.py")
                run("prepare_outside.py")
            audit = json.loads((ROOT / "results/data_audit.json").read_text())
            if audit["selected_constructions"] != audit["eligible_constructions"]:
                raise RuntimeError("Prepared input does not cover all eligible constructions")
            wait_for(args.wait_scoring_pid, "constructicon_scoring")
            run("score_expanded.py")
            run("score_expanded.py", "--dataset", "outside")
            wait_for(args.wait_collection_pid, "live_rnc_collection")
            run("collect_rnc.py")
            run("score_expanded.py", "--dataset", "rnc")
            run("analyze_expanded.py")
            run("analyze_outside.py")
            run("build_report.py")
            run("verify_results.py")
            status("complete", "complete", report="report/findings.pdf")
        except Exception as error:
            previous = json.loads(STATUS.read_text()) if STATUS.exists() else {}
            status(previous.get("stage", "setup"), "failed", error_type=type(error).__name__,
                   message="See the stage log. Fix the error, then rerun with --resume.")
            raise


if __name__ == "__main__":
    main()
