"""Developer figures for the Scientific Data descriptor from the entity-level ``developers.csv``.

Two figure families, ported from the authors' notebooks (graphs_dev.ipynb and
graphs_distribution.ipynb):

* ``dev_cont_<chain>.png``: monthly share of commits among the five largest
  contributors of each calendar year (stacked bars, percent of those five),
  for Ethereum consensus, Ethereum execution, Bitcoin, and Solana.
* ``dev_distro_<chain>.png``: degree distribution of the monthly developer
  co-contribution network, where two authors are linked when they commit to
  the same client repository in the same month; densities are pooled over all
  monthly snapshots with at least three developers. Bitcoin, BNB, and Solana.

Usage:
    uv run python scripts/plot_developers.py --entities PATH/developers.csv
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENTITIES = REPO_ROOT / "data" / "developers.csv"
FIGURES_DIR = REPO_ROOT / "outputs"

CONTRIBUTION_PANELS = [
    ("ethereum-consensus", "dev_cont_eth_consensus.png", "(A) Ethereum (consensus)"),
    ("ethereum-execution", "dev_cont_eth_execution.png", "(B) Ethereum (execution)"),
    ("bitcoin", "dev_cont_bitcoin.png", "(C) Bitcoin"),
    ("solana", "dev_cont_solana.png", "(D) Solana"),
]
DISTRIBUTION_PANELS = [
    ("bitcoin", "dev_distro_bitcoin.png", "Bitcoin"),
    ("bnb", "dev_distro_bnb.png", "BNB Chain"),
    ("solana", "dev_distro_solana.png", "Solana"),
]
TOP_K_PER_YEAR = 5
# Automated merge identity in bitcoin/bitcoin (not a person); excluded from the
# contribution panels as in the original notebook.
EXCLUDE_AUTHORS = {"merge-script"}
LEGEND_ENTRIES = 10
MIN_DEVS_PER_SNAPSHOT = 3


def load_entities(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0.0)
    df["entity"] = df["entity"].astype(str).str.strip()
    return df


def top_contributor_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly percent of commits among each year's top-k authors (summed across clients)."""
    df = df[~df["entity"].str.lower().isin(EXCLUDE_AUTHORS)]
    monthly = df.groupby(["date", "entity"], as_index=False)["count"].sum()
    monthly["year"] = monthly["date"].dt.year
    frames = []
    for _, ydf in monthly.groupby("year"):
        keep = ydf.groupby("entity")["count"].sum().nlargest(TOP_K_PER_YEAR).index
        ydf = ydf[ydf["entity"].isin(keep)].copy()
        totals = ydf.groupby("date")["count"].transform("sum")
        ydf["percent"] = (ydf["count"] / totals.replace(0, np.nan) * 100).fillna(0.0)
        frames.append(ydf)
    return pd.concat(frames, ignore_index=True)


def plot_contributions(shares: pd.DataFrame, title: str) -> go.Figure:
    totals = shares.groupby("entity")["percent"].sum().sort_values(ascending=False)
    legend = set(totals.head(LEGEND_ENTRIES).index)
    fig = go.Figure()
    for entity in totals.index:
        sub = shares[shares["entity"] == entity].sort_values("date")
        fig.add_trace(
            go.Bar(
                x=sub["date"].dt.strftime("%Y-%m-%d").tolist(),
                y=sub["percent"],
                name=entity,
                hoverinfo="skip",
                showlegend=entity in legend,
            )
        )
    fig.update_layout(
        barmode="stack",
        title=title,
        template="plotly_white",
        legend={
            "title": None, "font": {"size": 12}, "x": 0.01, "y": 0.99,
            "xanchor": "left", "yanchor": "top", "bgcolor": "rgba(240, 240, 240, 0.85)",
        },
        margin={"l": 50, "r": 10, "t": 40, "b": 40},
        height=420,
        width=700,
    )
    fig.update_yaxes(title_text="Percent", range=[0, 100], showgrid=True)
    fig.update_xaxes(
        title_text="Date", type="date",
        range=[shares["date"].min().strftime("%Y-%m-%d"), shares["date"].max().strftime("%Y-%m-%d")],
    )
    return fig


def degree_density(df: pd.DataFrame) -> tuple[pd.Series, int]:
    """Pooled degree distribution of monthly co-contribution networks."""
    counts: Counter[int] = Counter()
    snapshots = 0
    for _, month in df.groupby("date"):
        neighbors: dict[str, set[str]] = defaultdict(set)
        for _, client in month.groupby("client"):
            devs = sorted(client["entity"].unique())
            if len(devs) > 1:
                for d in devs:
                    neighbors[d].update(x for x in devs if x != d)
        if len(neighbors) < MIN_DEVS_PER_SNAPSHOT:
            continue
        snapshots += 1
        counts.update(len(n) for n in neighbors.values())
    if not counts:
        return pd.Series(dtype=float), 0
    s = pd.Series(counts).sort_index()
    return s / s.sum(), snapshots


def plot_distribution(density: pd.Series, title: str) -> go.Figure:
    fig = go.Figure(
        go.Scatter(x=density.index.astype(float), y=density.values, mode="lines",
                   line={"width": 2, "color": "navy"})
    )
    fig.update_layout(
        title={"text": title, "font": {"size": 30}},
        template="plotly_white",
        height=500,
        width=600,
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
    )
    axis = {"type": "log", "dtick": 1, "exponentformat": "power", "showexponent": "all",
            "title_font": {"size": 20}, "tickfont": {"size": 16}}
    fig.update_xaxes(title_text="Degree", **axis)
    fig.update_yaxes(title_text="Density", **axis)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    df = load_entities(args.entities)

    for group, filename, title in CONTRIBUTION_PANELS:
        shares = top_contributor_shares(df[df["group"] == group])
        fig = plot_contributions(shares, title)
        fig.write_image(str(args.figures_dir / filename), scale=2)
        print(f"wrote {filename}: {shares['entity'].nunique()} authors ever in a yearly top {TOP_K_PER_YEAR}")

    for group, filename, title in DISTRIBUTION_PANELS:
        density, snapshots = degree_density(df[df["group"] == group])
        if density.empty:
            print(f"skipped {filename}: no usable snapshots")
            continue
        fig = plot_distribution(density, title)
        fig.write_image(str(args.figures_dir / filename), scale=2)
        print(f"wrote {filename}: {snapshots} monthly snapshots, max degree {int(density.index.max())}")


if __name__ == "__main__":
    main()
