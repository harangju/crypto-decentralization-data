"""Pull full commit histories for the developer subsystem and compute monthly metrics.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import entropy, gini, hhi, nakamoto

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "developers_repos.csv"
UPSTREAM = "https://github.com/ethereum/go-ethereum"
DASHBOARD_GROUP = {
    "bitcoin": "Bitcoin",
    "bnb": "BNB",
    "polygon": "Polygon",
    "ronin": "Ronin",
    "solana": "Solana",
    "ethereum-execution": "Ethereum (execution)",
    "ethereum-consensus": "Ethereum (consensus)",
}


def clone(url: str, cache: Path) -> Path:
    dest = cache / url.rstrip("/").split("github.com/")[-1].replace("/", "__")
    if (dest / "HEAD").exists():
        subprocess.run(["git", "-C", str(dest), "fetch", "-q", "--filter=tree:0", "origin"], check=True)
    else:
        subprocess.run(
            ["git", "clone", "-q", "--bare", "--filter=tree:0", url, str(dest)], check=True
        )
    return dest


def log_commits(repo: Path) -> pd.DataFrame:
    """Default-branch commits: sha, author date (ISO, with offset), author name."""
    out = subprocess.run(
        ["git", "-C", str(repo), "log", "HEAD", "--format=%H%x09%aI%x09%aN"],
        check=True, capture_output=True, text=True,
    ).stdout
    rows = [line.split("\t", 2) for line in out.splitlines() if line.count("\t") == 2]
    df = pd.DataFrame(rows, columns=["sha", "author_date", "entity"])
    df["date"] = pd.to_datetime(df["author_date"], utc=True, format="ISO8601", errors="coerce")
    return df.dropna(subset=["date"]).drop(columns="author_date")


def load_commits(
    config: pd.DataFrame, cache: Path, keep_inherited: bool, keep_bots: bool
) -> pd.DataFrame:
    frames = []
    for row in config.itertuples(index=False):
        repo = clone(row.repository_url, cache)
        df = log_commits(repo)
        df["client"], df["group"] = row.client, row.chain
        frames.append(df)
        print(f"{row.repository_url}: {len(df):,} commits", flush=True)
    commits = pd.concat(frames, ignore_index=True)

    # Within a chain, count each commit once (first client listed wins).
    commits = commits.drop_duplicates(["group", "sha"], keep="first")

    if not keep_inherited:
        geth = commits.loc[commits["client"] == "geth", "sha"]
        if geth.empty:
            geth = log_commits(clone(UPSTREAM, cache))["sha"]
        inherited = (commits["client"] != "geth") & commits["sha"].isin(set(geth))
        print(f"dropped {int(inherited.sum()):,} commits inherited from go-ethereum", flush=True)
        commits = commits[~inherited]

    if not keep_bots:
        bots = commits["entity"].str.endswith("[bot]")
        print(f"dropped {int(bots.sum()):,} commits by [bot] authors", flush=True)
        commits = commits[~bots]
    return commits


def build_entities(commits: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    month = commits["date"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    panel = (
        commits.assign(date=month.dt.strftime("%Y-%m-%d"))
        .groupby(["date", "entity", "client", "group"], sort=True)
        .size()
        .rename("count")
        .reset_index()
    )
    return panel[panel["date"] <= end.strftime("%Y-%m-%d")][["date", "entity", "count", "client", "group"]]


def build_metrics(entities: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by_author = entities.groupby(["date", "group", "entity"])["count"].sum()
    for (date, group), s in by_author.groupby(level=[0, 1]):
        x = s.to_numpy(dtype=float)
        rows.append(
            {
                "Date": date,
                "Subsystem": "developers",
                "Group": DASHBOARD_GROUP.get(group, group),
                "Entropy": entropy(x),
                "Nakamoto": nakamoto(x),
                "Gini": gini(x),
                "HHI": hhi(x),
                "Count": x.size,
            }
        )
    return pd.DataFrame(rows).sort_values(["Date", "Group"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--end", required=True, help="Last month to include (any date in it)")
    parser.add_argument("--cache", type=Path, required=True, help="Directory for bare clones")
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--keep-inherited", action="store_true")
    parser.add_argument("--keep-bots", action="store_true")
    args = parser.parse_args()

    end = pd.Timestamp(args.end).to_period("M").to_timestamp()
    config = pd.read_csv(args.config)
    args.cache.mkdir(parents=True, exist_ok=True)

    commits = load_commits(config, args.cache, args.keep_inherited, args.keep_bots)
    entities = build_entities(commits, end)
    metrics = build_metrics(entities)

    for path in (args.entities, args.metrics):
        path.parent.mkdir(parents=True, exist_ok=True)
    entities.to_csv(args.entities, index=False)
    metrics.to_csv(args.metrics, index=False)
    print(f"Wrote {len(entities):,} entity rows to {args.entities}")
    print(f"Wrote {len(metrics):,} metric rows to {args.metrics}")


if __name__ == "__main__":
    main()
