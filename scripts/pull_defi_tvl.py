"""Pull DefiLlama protocol TVL and compute daily DeFi TVL decentralization metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import entropy, gini, hhi, nakamoto

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "defi_tvl_metrics.csv"
BULK_URL = "https://api.llama.fi/simpleChainDataset/All"
START = "2018-04-27"


def load_bulk(source: str = BULK_URL) -> pd.DataFrame:
    """Return a protocol x date frame of TVL in USD; missing values are 0 and ignored downstream."""
    wide = pd.read_csv(source, low_memory=False)
    wide = wide[wide["Protocol"] != "Total"].set_index("Protocol")
    wide.columns = pd.to_datetime(wide.columns, format="%d/%m/%Y")
    return wide.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def compute_daily_metrics(wide: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    rows = []
    for date in pd.date_range(start, end):
        if date not in wide.columns:
            raise ValueError(f"DefiLlama export has no column for {date.date()}")
        x = wide[date].to_numpy()
        x = x[x > 0]
        rows.append(
            {
                "Date": date.strftime("%Y-%m-%d"),
                "Subsystem": "defi tvl",
                "Group": "defi tvl",
                "Entropy": entropy(x),
                "Nakamoto": nakamoto(x),
                "Gini": gini(x),
                "HHI": hhi(x),
                "Count": x.size,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", required=True, help="Last date to include (YYYY-MM-DD)")
    parser.add_argument("--source", default=BULK_URL, help="URL or local path of the bulk export")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    wide = load_bulk(args.source)
    metrics = compute_daily_metrics(wide, args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output, index=False)
    print(f"Wrote {len(metrics):,} days ({len(wide):,} protocols) to {args.output}")


if __name__ == "__main__":
    main()
