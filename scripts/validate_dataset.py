"""External and internal validation checks for the dataset descriptor."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPOSIT_DIR = REPO_ROOT / "data"

# Public block count benchmarks (consensus daily totals).
BENCHMARKS = {
    "btc": {
        "date": "2024-01-15",
        "url": "https://blockchain.info/q/24hrtransactioncount",
        "note": "Bitcoin blocks/day proxy via blockchain.info 24h tx count is not block count; use local sum only.",
    },
}


def validate_bitcoin_fractional(consensus_path: Path) -> dict:
    df = pd.read_csv(consensus_path)
    btc = df[df["group"].astype(str).str.lower() == "btc"].copy()
    if btc.empty:
        return {"status": "skipped", "reason": "no bitcoin rows"}
    daily = btc.groupby("date")["count"].sum()
    exact = (daily.round(6) == daily.round(0)).mean()
    max_dev = float((daily - daily.round(0)).abs().max())
    return {
        "status": "ok",
        "days": int(daily.size),
        "exact_integer_share": float(exact),
        "max_fractional_deviation_blocks": max_dev,
    }


def validate_exchanges(exchanges_path: Path) -> dict:
    df = pd.read_csv(exchanges_path)
    monthly = df.groupby("date")["percent"].sum()
    return {
        "status": "ok",
        "months": int(monthly.size),
        "mean_percent_sum": float(monthly.mean()),
        "months_within_1pct_of_1": float((monthly.between(0.99, 1.01)).mean()),
    }


def validate_consensus_row_totals(consensus_path: Path) -> dict:
    df = pd.read_csv(consensus_path)
    by_chain = df.groupby("group").size().astype(int).to_dict()
    daily_blocks = df.groupby(["date", "group"])["count"].sum().groupby("group").mean().round(2)
    return {
        "status": "ok",
        "rows_by_group": by_chain,
        "mean_daily_block_or_share_by_group": daily_blocks.to_dict(),
    }


def fetch_defillama_sample() -> dict:
    url = "https://api.llama.fi/protocol/aave"
    try:
        with urlopen(url, timeout=20) as response:
            payload = json.load(response)
        tvl = payload.get("tvl", [])
        if not tvl:
            return {"status": "error", "reason": "empty tvl series"}
        latest = tvl[-1]
        return {
            "status": "ok",
            "protocol": "aave",
            "latest_date": latest.get("date"),
            "latest_tvl_usd": latest.get("totalLiquidityUSD"),
            "source": url,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "reason": str(exc)}


def main() -> None:
    results = {"generated_on": str(date.today()), "checks": {}}

    consensus = DEPOSIT_DIR / "consensus.csv"
    exchanges = DEPOSIT_DIR / "exchanges.csv"

    if consensus.exists():
        results["checks"]["bitcoin_fractional"] = validate_bitcoin_fractional(consensus)
        results["checks"]["consensus_overview"] = validate_consensus_row_totals(consensus)
    else:
        results["checks"]["consensus"] = {
            "status": "pending",
            "reason": f"missing {consensus}; download the Figshare CSVs into data/",
        }

    if exchanges.exists():
        results["checks"]["exchanges_percent_sum"] = validate_exchanges(exchanges)

    results["checks"]["defillama_api_sample"] = fetch_defillama_sample()

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
