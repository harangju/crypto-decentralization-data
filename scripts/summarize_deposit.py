"""Summarize row counts and date ranges for Figshare deposit CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "data"

FILES = [
    "consensus.csv",
    "developers.csv",
    "exchanges.csv",
    "defi_gov.csv",
    "defi_tvl.csv",
    "nft_marketplace.csv",
    "metrics.csv",
]


def summarize(path: Path) -> dict:
    df = pd.read_csv(path)
    summary = {
        "file": path.name,
        "rows": len(df),
        "columns": list(df.columns),
    }
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
        summary["start_date"] = str(dates.min())
        summary["end_date"] = str(dates.max())
    if "group" in df.columns:
        summary["groups"] = sorted(df["group"].dropna().astype(str).unique().tolist())
    return summary


def main(deposit_dir: Path = DEFAULT_DIR) -> None:
    summaries = []
    for name in FILES:
        path = deposit_dir / name
        if path.exists():
            summaries.append(summarize(path))
        else:
            summaries.append({"file": name, "rows": None, "status": "missing"})

    out = deposit_dir / "deposit_summary.json"
    out.write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
