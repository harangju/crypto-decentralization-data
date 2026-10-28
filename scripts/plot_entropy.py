"""Plot Shannon entropy over time for each subsystem from the deposit ``metrics.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.nonparametric.smoothers_lowess import lowess

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = REPO_ROOT / "data" / "metrics.csv"
FIGURES_DIR = REPO_ROOT / "outputs"

PANELS = [
    ("consensus", "entropy_consensus.png", "(A) Consensus", True),
    ("developers", "entropy_developers.png", "(B) Developers", True),
    ("defi gov", "entropy_defi_gov.png", "(C) DeFi Governance", True),
    ("defi tvl", "entropy_defi_tvl.png", "(D) DeFi TVL", False),
    ("nft marketplace", "entropy_nft.png", "(E) NFT Marketplaces", False),
]

PRETTY_GROUP = {
    "btc": "Bitcoin",
    "eth": "Ethereum",
    "bnb": "BNB",
    "ron": "Ronin",
    "sol": "Solana",
    "ton": "TON",
    "tron": "Tron",
    "bitcoin": "Bitcoin",
    "polygon": "Polygon",
    "ronin": "Ronin",
    "solana": "Solana",
    "ethereum-execution": "Ethereum (execution)",
    "ethereum-consensus": "Ethereum (consensus)",
    "aave": "Aave",
    "compound": "Compound",
    "curve": "Curve",
    "lido": "Lido",
    "maker": "MakerDAO",
    "uni": "Uniswap",
}


def plot_panel(df: pd.DataFrame, title: str, legend: bool) -> go.Figure:
    fig = go.Figure()
    groups = sorted(df["group"].unique(), key=str.lower)
    palette = px.colors.qualitative.Plotly
    for i, group in enumerate(groups):
        g = df[df["group"] == group].sort_values("date")
        color = palette[i % len(palette)]
        # ISO strings serialize reliably as a date axis across plotly versions.
        xs = g["date"].dt.strftime("%Y-%m-%d").tolist()
        fig.add_trace(
            go.Scatter(
                x=xs, y=g["entropy"], mode="markers", name=group,
                marker={"size": 3, "opacity": 0.5, "color": color, "line": {"width": 0}},
                hoverinfo="skip", showlegend=False,
            )
        )
        if len(g) >= 3:
            x = g["date"].astype("int64").to_numpy(dtype=float)
            smooth = lowess(g["entropy"].to_numpy(dtype=float), x, frac=0.1, return_sorted=False)
            fig.add_trace(
                go.Scatter(
                    x=xs, y=smooth, mode="lines", name=f"{group} trend",
                    line={"width": 1.0, "color": color}, hoverinfo="skip", showlegend=False,
                )
            )
        if legend:
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker={"size": 10, "color": color, "opacity": 1},
                    name=group, showlegend=True,
                )
            )

    if legend:
        fig.update_layout(
            legend={
                "title": "Group", "x": 0.01, "y": 0.99, "xanchor": "left", "yanchor": "top",
                "itemsizing": "constant", "font": {"size": 14}, "title_font": {"size": 16},
                "bgcolor": "rgba(0,0,0,0)",
            }
        )
    else:
        fig.update_layout(showlegend=False)

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title=None,
        yaxis_title="entropy (bits)",
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
        height=450,
        width=650,
        font={"size": 16},
    )
    fig.update_xaxes(type="date", title_font={"size": 16}, tickfont={"size": 14})
    fig.update_yaxes(title_font={"size": 16}, tickfont={"size": 14})
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics)
    metrics["date"] = pd.to_datetime(metrics["date"], utc=True).dt.tz_localize(None)
    metrics["group"] = metrics["group"].map(lambda g: PRETTY_GROUP.get(g, g))
    metrics = metrics.dropna(subset=["entropy"]).sort_values(["subsystem", "group", "date"])
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    for subsystem, filename, title, legend in PANELS:
        df = metrics[metrics["subsystem"] == subsystem]
        if df.empty:
            print(f"skipped {subsystem}: no rows")
            continue
        fig = plot_panel(df, title, legend)
        out = args.figures_dir / filename
        fig.write_image(str(out), scale=2)
        print(f"wrote {out} ({len(df):,} points)")


if __name__ == "__main__":
    main()
