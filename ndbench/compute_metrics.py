"""
Apply structural + surface metrics to every successful response in
data/responses/cache.jsonl and write data/metrics.csv.

The judge-based harm metrics live in data/judgments/cache.jsonl and are merged
in by ndbench/analyze.py.

Usage:
  python -m ndbench.compute_metrics
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ndbench.metrics import compute_structural, compute_surface


REPO_ROOT = Path(__file__).resolve().parent.parent
RESPONSES_FILE = REPO_ROOT / "data" / "responses" / "cache.jsonl"
METRICS_FILE = REPO_ROOT / "data" / "metrics.csv"

ID_COLS = ["model", "condition_id", "condition_label",
           "profile_id", "profile_label", "query_id", "query_domain"]


def main() -> None:
    records = [json.loads(l) for l in RESPONSES_FILE.read_text().splitlines() if l.strip()]
    rows = []
    skipped = 0
    for r in records:
        if not r.get("response") or r.get("error"):
            skipped += 1
            continue
        row = {k: r[k] for k in ID_COLS}
        row["response_text_tokens"] = r["usage"]["output_tokens"] if r.get("usage") else None
        text = r["response"]
        row.update(compute_structural(text))
        row.update(compute_surface(text))
        rows.append(row)

    df = pd.DataFrame(rows)
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(METRICS_FILE, index=False)
    print(f"Wrote {len(df)} rows × {len(df.columns)} cols → {METRICS_FILE}")
    print(f"Skipped (errors / missing responses): {skipped}")
    print(f"Rows by model × condition:")
    print(df.groupby(["model", "condition_id"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
