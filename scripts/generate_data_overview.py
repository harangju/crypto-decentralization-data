"""Generate the data overview JSON, LaTeX table, and summary figure from the metrics deposit.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPOSIT_DIR = REPO_ROOT / "data"
TABLES_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = REPO_ROOT / "outputs"

# (label, file, frequency)
SUBSYSTEMS = [
    ("Consensus", "consensus.csv", "Daily"),
    ("Developers", "developers.csv", "Monthly"),
    ("DeFi Governance", "defi_gov.csv", "Daily"),
    ("DeFi TVL", "defi_tvl.csv", "Daily"),
    ("NFT Marketplaces", "nft_marketplace.csv", "Daily"),
]


METRIC_COLUMNS = ["Date", "Subsystem", "Group", "Entropy", "Nakamoto", "Gini", "HHI", "Count",
                  "Total", "Top1Share", "Top5Share", "Top10Share"]


def load(path: Path) -> pd.DataFrame:
    """Group-period metric rows; developers and TVL files repeat them across entity rows."""
    df = pd.read_csv(path, usecols=lambda c: c in METRIC_COLUMNS or c == "MEVShare")
    df = df.drop_duplicates(["Date", "Group"]).reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    return df


def summarize_subsystem(label: str, path: Path, frequency: str) -> dict:
    if not path.exists():
        return {"subsystem": label, "file": path.name, "frequency": frequency, "status": "pending"}
    df = load(path)
    latest = df.loc[df.groupby("Group")["Date"].idxmax()]
    return {
        "subsystem": label,
        "file": path.name,
        "frequency": frequency,
        "status": "ok",
        "rows": int(len(df)),
        "groups": int(df["Group"].nunique()),
        "group_names": sorted(df["Group"].unique().tolist()),
        "start_date": str(df["Date"].min().date()),
        "end_date": str(df["Date"].max().date()),
        "median_entities_per_period": float(df["Count"].median()),
        "latest_median_nakamoto": float(latest["Nakamoto"].median()),
        "latest_median_entropy": float(latest["Entropy"].median()),
    }


def render_overview_tex(summaries: list[dict], out_path: Path) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\caption{Data overview by subsystem. Observations are group-periods (a chain, protocol, or pooled group "
        "on one day or month); entities per period is the median \\texttt{Count}.}",
        "    \\label{tab:overview}",
        "    \\small",
        "    \\begin{tabular}{@{}llrrllr@{}}",
        "        \\toprule",
        "        Subsystem & Frequency & Observations & Groups & Start & End & Median entities/period \\\\",
        "        \\midrule",
    ]
    for s in summaries:
        if s.get("status") != "ok":
            lines.append(f"        {s['subsystem']} & {s['frequency']} & --- & --- & --- & --- & --- \\\\")
            continue
        lines.append(
            f"        {s['subsystem']} & {s['frequency']} & {s['rows']:,} & {s['groups']} & "
            f"{s['start_date']} & {s['end_date']} & {s['median_entities_per_period']:,.0f} \\\\"
        )
    lines.extend(["        \\bottomrule", "    \\end{tabular}", "\\end{table}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def write_overview_figure(deposit_dir: Path, out_path: Path) -> bool:
    frames = []
    for label, filename, frequency in SUBSYSTEMS:
        path = deposit_dir / filename
        if path.exists():
            df = load(path)
            df["label"], df["window"] = label, (30 if frequency == "Daily" else 3)
            frames.append(df)
    if len(frames) < 2:
        return False
    data = pd.concat(frames, ignore_index=True)

    panels = [
        ("Count", "(A) Active entities per period", "entities (log)", True),
        ("Nakamoto", "(B) Nakamoto coefficient", "Nakamoto (log)", True),
        ("Gini", "(C) Gini coefficient", "Gini", False),
        ("HHI", "(D) Herfindahl-Hirschman Index", "HHI (log)", True),
    ]
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=[t for _, t, _, _ in panels],
        horizontal_spacing=0.10, vertical_spacing=0.18,
    )
    palette = px.colors.qualitative.Plotly
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (metric, _, ytitle, log), (row, col) in zip(panels, positions):
        for i, (label, _, _) in enumerate(SUBSYSTEMS):
            sub = data[data["label"] == label]
            if sub.empty:
                continue
            # Median across groups per period, then a rolling median to smooth.
            series = sub.groupby("Date")[metric].median().sort_index()
            series = series.rolling(int(sub["window"].iloc[0]), min_periods=1).median()
            fig.add_trace(
                go.Scatter(
                    x=series.index.strftime("%Y-%m-%d").tolist(), y=series.values, mode="lines",
                    name=label, legendgroup=label, showlegend=(row == 1 and col == 1),
                    line={"width": 1.5, "color": palette[i]}, hoverinfo="skip",
                ),
                row=row, col=col,
            )
        axis = {"title_text": ytitle, "title_font": {"size": 16}, "tickfont": {"size": 14}}
        if log:
            axis.update({"type": "log", "dtick": 1, "exponentformat": "power", "showexponent": "all"})
        fig.update_yaxes(row=row, col=col, **axis)
        fig.update_xaxes(type="date", tickfont={"size": 14}, row=row, col=col)

    fig.update_layout(
        template="plotly_white", width=1200, height=820, font={"size": 16},
        margin={"l": 70, "r": 20, "t": 90, "b": 60},
        legend={"title": None, "orientation": "h", "x": 0.5, "y": 1.08, "xanchor": "center",
                "yanchor": "bottom", "font": {"size": 15}, "bgcolor": "rgba(0,0,0,0)"},
    )
    fig.update_annotations(font={"size": 18})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), scale=3)
    return True


def main(deposit_dir: Path = DEPOSIT_DIR) -> None:
    summaries = [summarize_subsystem(label, deposit_dir / f, freq) for label, f, freq in SUBSYSTEMS]
    results = {"generated_on": str(date.today()), "subsystems": summaries,
               "total_rows": int(sum(s.get("rows", 0) for s in summaries))}

    json_path = deposit_dir / "overview_summary.json"
    deposit_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2))

    tex_path = TABLES_DIR / "data_overview.tex"
    render_overview_tex(summaries, tex_path)

    fig_path = FIGURES_DIR / "data_overview.png"
    wrote_fig = write_overview_figure(deposit_dir, fig_path)

    print(json.dumps(results, indent=2))
    print(f"\nWrote {json_path}")
    print(f"Wrote {tex_path}")
    if wrote_fig:
        print(f"Wrote {fig_path}")
    else:
        print(f"Skipped figure (stage deposit CSVs in {deposit_dir})", file=sys.stderr)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEPOSIT_DIR)
