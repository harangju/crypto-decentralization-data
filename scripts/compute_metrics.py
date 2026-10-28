"""Compute decentralization metrics from entity-level subsystem CSVs."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data"
DEFAULT_OUTPUT = DEFAULT_INPUT / "metrics.csv"

REQUIRED = ("date", "entity", "count", "group")


def _clean_counts(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    values = values[values > 0]
    return values


def shannon_entropy(values: np.ndarray) -> float:
    total = values.sum()
    if total <= 0:
        return 0.0
    shares = values / total
    return float(-(shares * np.log2(shares)).sum())


def gini(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = sorted_vals.size
    index = np.arange(1, n + 1)
    return float((2 * (index * sorted_vals).sum()) / (n * sorted_vals.sum()) - (n + 1) / n)


def hhi(values: np.ndarray) -> float:
    total = values.sum()
    if total <= 0:
        return 0.0
    shares = values / total
    return float((shares**2).sum())


def nakamoto(values: np.ndarray, threshold: float = 0.5) -> int:
    total = values.sum()
    if total <= 0:
        return 0
    ranked = np.sort(values)[::-1]
    cumulative = 0.0
    for idx, value in enumerate(ranked, start=1):
        cumulative += value
        if cumulative / total >= threshold:
            return idx
    return ranked.size


def compute_group_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    values = _clean_counts(frame["count"])
    return {
        "entropy": shannon_entropy(values),
        "nakamoto": nakamoto(values),
        "gini": gini(values),
        "hhi": hhi(values),
        "count": int(values.size),
    }


def compute_metrics_for_file(path: Path, subsystem: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED if col not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    rows: list[dict] = []
    for (date, group), group_df in df.groupby(["date", "group"], sort=True):
        metrics = compute_group_metrics(group_df)
        rows.append(
            {
                "date": date,
                "subsystem": subsystem,
                "group": group,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def main(input_dir: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> None:
    specs = {
        "consensus.csv": "consensus",
        "developers.csv": "developers",
        "defi_gov.csv": "defi gov",
        "defi_tvl.csv": "defi tvl",
        "nft_marketplace.csv": "nft marketplace",
        "exchanges.csv": "exchanges",
    }
    frames = []
    for filename, subsystem in specs.items():
        path = input_dir / filename
        if not path.exists():
            continue
        frames.append(compute_metrics_for_file(path, subsystem))

    if not frames:
        raise SystemExit(f"No subsystem CSVs found in {input_dir}")

    metrics = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    print(f"Wrote {len(metrics):,} metric rows to {output_path}")


if __name__ == "__main__":
    main()
