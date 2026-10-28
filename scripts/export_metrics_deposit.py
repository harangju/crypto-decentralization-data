"""Write the Figshare deposit: one metrics CSV per subsystem plus entity-level files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOS = REPO_ROOT / "config" / "developers_repos.csv"

FILES = {
    "consensus": "consensus.csv",
    "developers": "developers.csv",
    "defi gov": "defi_gov.csv",
    "defi tvl": "defi_tvl.csv",
    "nft marketplace": "nft_marketplace.csv",
}
GROUP_NAME = {
    "btc": "Bitcoin", "eth": "Ethereum", "bnb": "BNB", "ron": "Ronin", "sol": "Solana",
    "ton": "TON", "tron": "Tron",
    "bitcoin": "Bitcoin", "polygon": "Polygon", "ronin": "Ronin", "solana": "Solana",
    "ethereum-execution": "Ethereum (execution)", "ethereum-consensus": "Ethereum (consensus)",
    "aave": "Aave", "compound": "Compound", "curve": "Curve", "lido": "Lido",
    "maker": "MakerDAO", "uni": "Uniswap",
}
METRIC_COLUMNS = ["Date", "Subsystem", "Group", "Entropy", "Nakamoto", "Gini", "HHI", "Count",
                  "Total", "Top1Share", "Top5Share", "Top10Share"]
# Decimal places written to the deposit (keeps the TVL file compact).
ROUNDING = {"Entropy": 6, "Gini": 6, "HHI": 8, "Top1Share": 6, "Top5Share": 6, "Top10Share": 6,
            "MEVShare": 6, "Total": 2}


def metrics_wide(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.rename(columns={
        "date": "Date", "subsystem": "Subsystem", "group": "Group", "entropy": "Entropy",
        "nakamoto": "Nakamoto", "gini": "Gini", "hhi": "HHI", "count": "Count",
        "total": "Total", "top1_share": "Top1Share", "top5_share": "Top5Share",
        "top10_share": "Top10Share", "mev_share": "MEVShare",
    })
    out["Date"] = pd.to_datetime(out["Date"], utc=True).dt.strftime("%Y-%m-%d")
    out["Group"] = out["Group"].map(lambda g: GROUP_NAME.get(g, g))
    out["Entropy"] = out["Entropy"].abs()  # -0.0 -> 0.0 for single-entity periods
    out["Nakamoto"] = out["Nakamoto"].astype(int)
    out["Count"] = out["Count"].astype(int)
    if "MEVShare" in out.columns:  # defined for Ethereum consensus only
        out.loc[~((out["Subsystem"] == "consensus") & (out["Group"] == "Ethereum")), "MEVShare"] = pd.NA
    return out


def developer_rows(internal_dir: Path, repos: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pseudonymized author-client-month commit rows with repository URLs."""
    df = pd.read_csv(internal_dir / "developers.csv")
    urls = pd.read_csv(repos).rename(columns={"chain": "group"})
    df = df.merge(urls[["group", "client", "repository_url"]], on=["group", "client"], how="left")
    if df["repository_url"].isna().any():
        missing = df.loc[df["repository_url"].isna(), ["group", "client"]].drop_duplicates()
        raise ValueError(f"client without repository_url:\n{missing}")
    df["Date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    first = df.groupby("entity")["Date"].min()
    order = sorted(first.index, key=lambda e: (first[e], e))
    mapping = pd.DataFrame({"author": order, "author_id": [f"dev-{i + 1:05d}" for i in range(len(order))]})
    ids = dict(zip(mapping["author"], mapping["author_id"]))
    rows = pd.DataFrame({
        "date": df["Date"],
        "author_id": df["entity"].map(ids),
        "count": df["count"].astype(int),
        "client": df["client"],
        "repository_url": df["repository_url"],
        "group": df["group"],
    }).sort_values(["group", "client", "date", "author_id"]).reset_index(drop=True)
    return rows, mapping


# internal file -> (deposit file, columns kept)
ENTITY_FILES = {
    "consensus.csv": ("consensus_producers.csv", ["date", "entity", "count", "group", "mev"]),
    "defi_gov.csv": ("defi_gov_holders.csv", ["date", "entity", "count", "group"]),
    "nft_marketplace.csv": ("nft_marketplace_sales.csv", ["date", "entity", "count"]),
}


def entity_rows(internal_dir: Path, source: str, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(internal_dir / source, usecols=columns, low_memory=False)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    if "mev" in df.columns:
        df["mev"] = df["mev"].astype("Int64")
    sort = [c for c in ["group", "date", "entity"] if c in df.columns]
    return df[columns].sort_values(sort).reset_index(drop=True)


def export(metrics: pd.DataFrame, output_dir: Path, internal_dir: Path | None, repos: Path) -> dict[str, int]:
    wide = metrics_wide(metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, int] = {}
    for subsystem, filename in FILES.items():
        cols = METRIC_COLUMNS + (["MEVShare"] if subsystem == "consensus" and "MEVShare" in wide.columns else [])
        part = wide[wide["Subsystem"] == subsystem][cols].sort_values(["Group", "Date"]).round(ROUNDING)
        part.to_csv(output_dir / filename, index=False)
        rows[filename] = len(part)
    if internal_dir is not None:
        entities, mapping = developer_rows(internal_dir, repos)
        mapping.to_csv(internal_dir / "developers_author_mapping.csv", index=False)
        entities.to_csv(output_dir / "developers_commits.csv", index=False)
        rows["developers_commits.csv"] = len(entities)
        for source, (filename, columns) in ENTITY_FILES.items():
            ent = entity_rows(internal_dir, source, columns)
            ent.to_csv(output_dir / filename, index=False)
            rows[filename] = len(ent)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--internal-dir", type=Path, default=None,
                        help="Internal entity counts; also writes the entity-level files")
    parser.add_argument("--repos", type=Path, default=DEFAULT_REPOS)
    args = parser.parse_args()
    rows = export(pd.read_csv(args.metrics), args.output_dir, args.internal_dir, args.repos)
    for filename, n in rows.items():
        print(f"{filename}: {n:,} rows")


if __name__ == "__main__":
    main()
