"""Generate data overview JSON, LaTeX table, and summary figure."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPOSIT_DIR = REPO_ROOT / "data"
TABLES_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = REPO_ROOT / "outputs"

SUBSYSTEMS = [
    ("Consensus", "consensus.csv", ["date", "entity", "group"]),
    ("Developers", "developers.csv", ["date", "entity", "group"]),
    ("Exchanges", "exchanges.csv", ["date", "cex"]),
    ("DeFi Governance", "defi_gov.csv", ["date", "entity", "group"]),
    ("DeFi TVL", "defi_tvl.csv", ["date", "entity"]),
    ("NFT Marketplaces", "nft_marketplace.csv", ["date", "entity"]),
]

# Known from manuscript when CSVs not staged.
FALLBACK_ROWS = {"Consensus": 5_869_789}


def _entity_col(key_cols: list[str]) -> str:
    return "cex" if "cex" in key_cols else "entity"


def summarize_subsystem(label: str, path: Path, key_cols: list[str]) -> dict:
    if not path.exists():
        return {
            "subsystem": label,
            "file": path.name,
            "status": "pending",
            "rows": FALLBACK_ROWS.get(label),
        }
    df = pd.read_csv(path)
    entity_col = _entity_col(key_cols)
    dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
    daily_entities = df.groupby(dates.dt.date)[entity_col].nunique()
    return {
        "subsystem": label,
        "file": path.name,
        "status": "ok",
        "rows": int(len(df)),
        "unique_entities": int(df[entity_col].nunique()),
        "start_date": str(dates.min().date()) if dates.notna().any() else None,
        "end_date": str(dates.max().date()) if dates.notna().any() else None,
        "mean_entities_per_day": float(daily_entities.mean()),
        "median_entities_per_day": float(daily_entities.median()),
    }


def summarize_metrics(path: Path) -> dict:
    if not path.exists():
        return {"status": "pending"}
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    latest = df.loc[df.groupby(["subsystem", "group"])["date"].idxmax()]
    by_sub = (
        latest.groupby("subsystem")
        .agg(nakamoto=("nakamoto", "median"), entity_count=("count", "median"))
        .reset_index()
    )
    return {
        "status": "ok",
        "rows": int(len(df)),
        "latest_nakamoto_by_subsystem": {
            row["subsystem"]: int(row["nakamoto"]) for _, row in by_sub.iterrows()
        },
    }


def render_overview_tex(summaries: list[dict], out_path: Path) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\caption{Data overview by subsystem (descriptive statistics).}",
        "    \\label{tab:overview}",
        "    \\small",
        "    \\begin{tabular}{@{}lrrllr@{}}",
        "        \\toprule",
        "        Subsystem & Rows & Unique entities & Start & End & Median entities/day \\\\",
        "        \\midrule",
    ]
    for s in summaries:
        rows = f"{s['rows']:,}" if s.get("rows") is not None else "---"
        entities = f"{s['unique_entities']:,}" if s.get("unique_entities") is not None else "---"
        start = s.get("start_date") or "---"
        end = s.get("end_date") or "---"
        median = (
            f"{s['median_entities_per_day']:.0f}"
            if s.get("median_entities_per_day") is not None
            else "---"
        )
        lines.append(
            f"        {s['subsystem']} & {rows} & {entities} & {start} & {end} & {median} \\\\"
        )
    lines.extend(["        \\bottomrule", "    \\end{tabular}", "\\end{table}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def write_overview_figure(deposit_dir: Path, summaries: list[dict], metrics_info: dict, out_path: Path) -> bool:
    ok_summaries = [s for s in summaries if s.get("status") == "ok"]
    if len(ok_summaries) < 2:
        return False

    labels = [s["subsystem"] for s in ok_summaries]
    unique_entities = [s["unique_entities"] for s in ok_summaries]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Median active entities per day",
            "Unique entities by subsystem",
            "Consensus rows by chain",
            "Latest median Nakamoto by subsystem",
        ),
    )

    # Panel (a): median entities/day time series for first available daily file.
    for label, filename, key_cols in SUBSYSTEMS:
        path = deposit_dir / filename
        if not path.exists() or label == "Exchanges":
            continue
        df = pd.read_csv(path)
        entity_col = _entity_col(key_cols)
        dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
        series = df.groupby(dates.dt.date)[entity_col].nunique().rolling(30, min_periods=1).median()
        fig.add_trace(
            go.Scatter(x=series.index, y=series.values, mode="lines", name=label, legendgroup=label),
            row=1,
            col=1,
        )

    # Panel (b): unique entities bar.
    fig.add_trace(
        go.Bar(x=labels, y=unique_entities, name="Unique entities", showlegend=False, marker_color="#1f77b4"),
        row=1,
        col=2,
    )

    # Panel (c): consensus rows by chain.
    consensus_path = deposit_dir / "consensus.csv"
    if consensus_path.exists():
        cdf = pd.read_csv(consensus_path)
        by_chain = cdf.groupby("group").size().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(
                x=by_chain.index.astype(str),
                y=by_chain.values,
                name="Consensus rows",
                showlegend=False,
                marker_color="#2ca02c",
            ),
            row=2,
            col=1,
        )

    # Panel (d): latest Nakamoto by subsystem.
    if metrics_info.get("status") == "ok":
        nak = metrics_info["latest_nakamoto_by_subsystem"]
        fig.add_trace(
            go.Bar(
                x=list(nak.keys()),
                y=list(nak.values()),
                name="Nakamoto",
                showlegend=False,
                marker_color="#ff7f0e",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        template="plotly_white",
        width=900,
        height=700,
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        title="Data overview",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), scale=3)
    return True


def main(deposit_dir: Path = DEPOSIT_DIR) -> None:
    summaries = [
        summarize_subsystem(label, deposit_dir / fname, keys)
        for label, fname, keys in SUBSYSTEMS
    ]
    metrics_info = summarize_metrics(deposit_dir / "metrics.csv")
    results = {
        "generated_on": str(date.today()),
        "subsystems": summaries,
        "metrics": metrics_info,
    }

    json_path = deposit_dir / "overview_summary.json"
    deposit_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2))

    tex_path = TABLES_DIR / "data_overview.tex"
    render_overview_tex(summaries, tex_path)

    fig_path = FIGURES_DIR / "data_overview.png"
    wrote_fig = write_overview_figure(deposit_dir, summaries, metrics_info, fig_path)

    print(json.dumps(results, indent=2))
    print(f"\nWrote {json_path}")
    print(f"Wrote {tex_path}")
    if wrote_fig:
        print(f"Wrote {fig_path}")
    else:
        print(
            f"Skipped figure (stage deposit CSVs in {deposit_dir} to generate {fig_path.name})",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
