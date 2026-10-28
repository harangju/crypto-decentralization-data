"""Summarize row counts, date ranges, and groups of the Figshare metrics files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "data"

FILES = [
    "consensus.csv",
    "developers.csv",
    "defi_gov.csv",
    "defi_tvl.csv",
    "nft_marketplace.csv",
    "consensus_producers.csv",
    "developers_commits.csv",
    "defi_gov_holders.csv",
    "nft_marketplace_sales.csv",
]


def summarize(path: Path) -> dict:
    df = pd.read_csv(path)
    summary = {
        "file": path.name,
        "rows": len(df),
        "columns": list(df.columns),
    }
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
        summary["start_date"] = str(dates.min().date())
        summary["end_date"] = str(dates.max().date())
    if "group" in df.columns:
        summary["groups"] = sorted(df["group"].dropna().astype(str).unique().tolist())
    return summary


def main(deposit_dir: Path = DEFAULT_DIR, out_dir: Path | None = None) -> None:
    summaries = []
    for name in FILES:
        path = deposit_dir / name
        if path.exists():
            summaries.append(summarize(path))
        else:
            summaries.append({"file": name, "rows": None, "status": "missing"})

    out = (out_dir or deposit_dir) / "deposit_summary.json"
    out.write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR, DEFAULT_DIR)
