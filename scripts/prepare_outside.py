"""Sample up to two words outside each HF-annotated construction span.

These are within-sentence controls, not assertions that no other construction
occurs there. Sampling is independent of model scores and word roles/POS.
"""

from collections import defaultdict
import json
import random

from audit_spans import ROOT, row_key
from prepare import write_jsonl


def main():
    scopes = json.loads((ROOT / "data/scope_index.json").read_text())
    groups = defaultdict(list)
    for line in (ROOT / "data/items.jsonl").read_text().splitlines():
        row = json.loads(line)
        scope = scopes[row_key(row)]
        if scope["status"] == "matched" and scope["within"] is False:
            groups[row["sentence_id"]].append(row)
    rng = random.Random(20260926)
    selected = []
    for sid in sorted(groups):
        for row in rng.sample(groups[sid], min(2, len(groups[sid]))):
            selected.append({**row, "original_type": row["type"], "type": "outside"})
    write_jsonl(ROOT / "data/outside_items.jsonl", selected)
    audit = dict(
        seed=20260926,
        max_per_sentence=2,
        sentences=len(groups),
        candidate_words=sum(map(len, groups.values())),
        selected_words=len(selected),
    )
    (ROOT / "results/outside_audit.json").write_text(json.dumps(audit, indent=2))
    print(audit)


if __name__ == "__main__":
    main()
