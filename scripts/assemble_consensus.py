"""Assemble the entity-level consensus panel from per-chain Dune exports.

Inputs are directories of Dune CSV exports (columns date, entity, count,
group, and mev for Ethereum). Exports for one chain may overlap in time
(cumulative re-pulls, shared boundary days); each date/entity key is kept once,
taken from the export whose date range starts latest, since later pulls cover
boundary days completely.

Ethereum rows from ``--eth-switch`` onward are replaced by the builder-attributed
export (``--eth-new``); see the Scientific Data descriptor, Methods. Ronin rows
after ``--ronin-end`` are dropped (migration to an Ethereum layer 2).

Usage:
    uv run python scripts/assemble_consensus.py \
        --raw-dir "<folder with btc/ eth/ bnb/ ron/ sol/ ton/ tron/ subfolders>" \
        --extra "<more export csvs, same layout>" \
        --eth-new raw_eth/daily_miner_count-_new_eth.csv \
        --output deposit/consensus.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CHAINS = {"btc": "btc", "eth": "eth", "bnb": "bnb", "ron": "ron", "sol": "sol", "ton": "ton", "tron": "tron"}
COLUMNS = ["date", "entity", "count", "group", "mev"]


def load_export(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[[c for c in COLUMNS if c in df.columns]].copy()
    df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True).dt.strftime("%Y-%m-%d")
    df["entity"] = df["entity"].astype("string").str.strip().str.lower().fillna("")
    df["count"] = pd.to_numeric(df["count"], errors="raise")
    df["group"] = df["group"].astype(str).str.strip().str.lower()
    if "mev" in df.columns:
        df["mev"] = pd.to_numeric(df["mev"], errors="coerce")
    df["_start"] = df["date"].min()
    return df


def collect(dirs: list[Path], chain: str) -> list[Path]:
    files: list[Path] = []
    for d in dirs:
        sub = d / chain
        if sub.is_dir():
            files.extend(sorted(sub.glob("*.csv")))
    return [f for f in files if ".ipynb_checkpoints" not in f.parts]


def assemble_chain(files: list[Path], chain: str) -> pd.DataFrame:
    parts = [load_export(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    bad = df["group"].ne(chain)
    if bad.any():
        raise ValueError(f"{chain}: {int(bad.sum())} rows with group {df.loc[bad, 'group'].unique().tolist()}")
    before = len(df)
    df = df.sort_values(["date", "entity", "_start"]).drop_duplicates(["date", "entity"], keep="last")
    print(f"{chain}: {len(files)} exports, {before:,} rows, {before - len(df):,} overlapping keys dropped, "
          f"{df['date'].min()} to {df['date'].max()}", flush=True)
    return df.drop(columns="_start")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, action="append", required=True,
                        help="Directory with one subfolder per chain code; repeatable")
    parser.add_argument("--eth-new", type=Path, required=True, help="Builder-attributed Ethereum export")
    parser.add_argument("--eth-switch", default="2022-05-26", help="First date taken from --eth-new")
    parser.add_argument("--ronin-end", default="2026-05-11", help="Last Ronin date retained")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames = []
    for chain in CHAINS:
        files = collect(args.raw_dir, chain)
        if not files:
            raise FileNotFoundError(f"no exports found for {chain}")
        df = assemble_chain(files, chain)
        if chain == "eth":
            new = load_export(args.eth_new).drop(columns="_start")
            new = new[new["date"] >= args.eth_switch]
            df = pd.concat([df[df["date"] < args.eth_switch], new], ignore_index=True)
            print(f"eth: {len(new):,} rows from builder-attributed export from {args.eth_switch}", flush=True)
        if chain == "ron":
            dropped = int((df["date"] > args.ronin_end).sum())
            df = df[df["date"] <= args.ronin_end]
            print(f"ron: dropped {dropped:,} rows after {args.ronin_end}", flush=True)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    for c in COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[COLUMNS].sort_values(["group", "date", "entity"]).reset_index(drop=True)
    out.loc[out["group"] != "eth", "mev"] = pd.NA
    if out.duplicated(["date", "entity", "group"]).any():
        raise ValueError("duplicate date/entity/group keys remain")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} rows to {args.output}")
    print(out.groupby("group")["date"].agg(["min", "max", "size"]).to_string())


if __name__ == "__main__":
    main()
