"""Generate validation summary JSON, LaTeX table, and DefiLlama spot-check figure.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPOSIT_DIR = REPO_ROOT / "data"
TABLES_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = REPO_ROOT / "outputs"

# deposit metrics files (dashboard schema) and, for the internal checks, the
# key columns of the entity-level counts they are computed from
SUBSYSTEM_FILES = {
    "consensus": ("consensus.csv", ["date", "entity", "group"]),
    "developers": ("developers.csv", ["date", "entity", "client", "group"]),
    "defi_gov": ("defi_gov.csv", ["date", "entity", "group"]),
    "defi_tvl": ("defi_tvl.csv", ["date", "entity"]),
    "nft_marketplace": ("nft_marketplace.csv", ["date", "entity"]),
}
METRIC_COLUMNS = ["Date", "Subsystem", "Group", "Entropy", "Nakamoto", "Gini", "HHI", "Count",
                  "Total", "Top1Share", "Top5Share", "Top10Share"]

# (label, DefiLlama API slug, entity names in defi_tvl.csv). The bulk chain
# export omits DefiLlama's double-counted categories (liquid staking, yield,
# bridges, CEX), so the spot checks use large protocols that it does include.
DEFILLAMA_PROTOCOLS: list[tuple[str, str, list[str]]] = [
    ("Aave V3", "aave-v3", ["aave v3"]),
    ("Compound V3", "compound-v3", ["compound v3"]),
    ("Sky Lending", "sky-lending", ["sky lending", "makerdao"]),
    ("Uniswap V3", "uniswap-v3", ["uniswap v3"]),
]

# group code -> (low, high, expected blocks/day). Ethereum is checked separately
# before and after the Merge (2022-09-15): proof-of-work averaged roughly
# 6,000 to 6,500 blocks/day; proof-of-stake has a hard ceiling of 7,200 slots/day.
CONSENSUS_BASELINES: dict[str, tuple[float, float, float]] = {
    "btc": (130.0, 160.0, 144.0),
    "eth_pre_merge": (5500.0, 7000.0, 6500.0),
    "eth_post_merge": (6800.0, 7200.0, 7200.0),
}
ETH_MERGE_DATE = "2022-09-15"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "---"
    return f"{100 * value:.1f}\\%"


def _fmt_num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "---"
    return f"{value:.{digits}f}"


def _fmt_sci(value: float | None) -> str:
    if value is None:
        return "---"
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"${mantissa} \\times 10^{{{int(exponent)}}}$"


def _fmt_usd_b(value: float | None) -> str:
    if value is None:
        return "---"
    return f"\\${value / 1e9:.1f}B"


def validate_row_integrity(path: Path, key_cols: list[str]) -> dict:
    df = pd.read_csv(path)
    counts = pd.to_numeric(df["count"], errors="coerce")
    negative = int((counts < 0).sum())
    dupes = int(df.duplicated(subset=key_cols, keep=False).sum())
    return {
        "status": "ok",
        "rows": int(len(df)),
        "negative_counts": negative,
        "duplicate_keys": dupes,
        "key_columns": key_cols,
    }


def validate_bitcoin_fractional(consensus_path: Path) -> dict:
    df = pd.read_csv(consensus_path)
    btc = df[df["group"].astype(str).str.lower() == "btc"].copy()
    if btc.empty:
        return {"status": "skipped", "reason": "no bitcoin rows"}
    daily = btc.groupby("date")["count"].sum()
    exact = float((daily.round(6) == daily.round(0)).mean())
    max_dev = float((daily - daily.round(0)).abs().max())
    return {
        "status": "ok",
        "days": int(daily.size),
        "exact_integer_share": exact,
        "max_fractional_deviation_blocks": max_dev,
    }


def validate_consensus_daily_blocks(consensus_path: Path) -> dict:
    df = pd.read_csv(consensus_path)
    daily = df.groupby(["date", "group"])["count"].sum().reset_index()
    daily["group"] = daily["group"].astype(str).str.lower()
    eth = daily[daily["group"] == "eth"]
    daily.loc[eth.index, "group"] = (eth["date"] < ETH_MERGE_DATE).map(
        {True: "eth_pre_merge", False: "eth_post_merge"}
    )
    medians = daily.groupby("group")["count"].median()
    groups: dict[str, dict] = {}
    for g, median in medians.items():
        entry: dict = {"median_daily_blocks": float(median)}
        if g in CONSENSUS_BASELINES:
            lo, hi, expected = CONSENSUS_BASELINES[g]
            entry["expected_approx"] = expected
            entry["within_range"] = bool(lo <= median <= hi)
        groups[g] = entry
    return {"status": "ok", "groups": groups}


def validate_developers(developers_path: Path) -> dict:
    """Rows are (month, author, client, group); authors active in several clients
    of one chain collapse to one entity when metrics are computed."""
    df = pd.read_csv(developers_path)
    dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
    month_start = (dates.dt.day == 1).mean()
    grouped = df.groupby(["date", "entity", "group"], as_index=False)["count"].sum()
    bots = int(df["entity"].astype(str).str.endswith("[bot]").sum())
    return {
        "status": "ok",
        "raw_rows": int(len(df)),
        "deduped_rows": int(len(grouped)),
        "unique_developers": int(df["entity"].nunique()),
        "share_rows_dated_month_start": float(month_start),
        "bot_rows": bots,
        "total_commits": int(df["count"].sum()),
    }


def fetch_defillama_series(slug: str) -> pd.DataFrame:
    url = f"https://api.llama.fi/protocol/{slug}"
    with urlopen(url, timeout=20) as response:
        payload = json.load(response)
    rows = []
    for point in payload.get("tvl", []):
        ts = point.get("date")
        tvl = point.get("totalLiquidityUSD")
        if ts is None or tvl is None:
            continue
        rows.append({"date": datetime.fromtimestamp(ts, tz=timezone.utc).date(), "tvl_usd": float(tvl)})
    return pd.DataFrame(rows)


def _match_protocol_rows(df: pd.DataFrame, entity_names: list[str]) -> pd.DataFrame:
    entities = df["entity"].astype(str).str.lower()
    mask = False
    for name in entity_names:
        mask = mask | (entities == name.lower())
    return df[mask].copy()


def validate_defillama_protocol(defi_tvl_path: Path, label: str, slug: str, entity_names: list[str]) -> dict:
    df = pd.read_csv(defi_tvl_path)
    subset = _match_protocol_rows(df, entity_names)
    if subset.empty:
        return {"status": "skipped", "reason": f"no {label} rows in defi_tvl.csv", "label": label}
    subset["date"] = pd.to_datetime(subset["date"], errors="coerce").dt.date
    subset = subset.dropna(subset=["date"]).sort_values("date")
    api = fetch_defillama_series(slug)
    merged = subset.merge(api, on="date", how="inner", suffixes=("_deposit", "_api"))
    if merged.empty:
        return {"status": "skipped", "reason": f"no overlapping {label} dates with API", "label": label}
    merged["pct_diff"] = (
        (merged["count"] - merged["tvl_usd"]).abs() / merged["tvl_usd"].clip(lower=1)
    )
    latest = merged.iloc[-1]
    return {
        "status": "ok",
        "label": label,
        "slug": slug,
        "overlap_days": int(len(merged)),
        "median_pct_diff": float(merged["pct_diff"].median()),
        "latest_date": str(latest["date"]),
        "deposit_tvl_usd": float(latest["count"]),
        "api_tvl_usd": float(latest["tvl_usd"]),
        "latest_pct_diff": float(latest["pct_diff"]),
    }


def validate_defillama_protocols(defi_tvl_path: Path) -> dict:
    results: dict[str, dict] = {}
    for label, slug, entity_names in DEFILLAMA_PROTOCOLS:
        try:
            results[label.lower()] = validate_defillama_protocol(defi_tvl_path, label, slug, entity_names)
        except Exception as exc:  # noqa: BLE001
            results[label.lower()] = {"status": "error", "label": label, "reason": str(exc)}
    return {"status": "ok", "protocols": results}


def write_validation_panel(defi_tvl_path: Path | None, out_path: Path) -> bool:
    if defi_tvl_path is None or not defi_tvl_path.exists():
        return False

    df = pd.read_csv(defi_tvl_path)
    letters = "ABCD"
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"({letters[i]}) {label}" for i, (label, _, _) in enumerate(DEFILLAMA_PROTOCOLS)],
        vertical_spacing=0.16,
        horizontal_spacing=0.10,
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    wrote_any = False
    for (row, col), (label, slug, entity_names) in zip(positions, DEFILLAMA_PROTOCOLS):
        subset = _match_protocol_rows(df, entity_names)
        if subset.empty:
            continue
        subset["date"] = pd.to_datetime(subset["date"], errors="coerce")
        subset = subset.dropna(subset=["date"]).sort_values("date")
        try:
            api = fetch_defillama_series(slug)
        except Exception:  # noqa: BLE001
            continue
        api["date"] = pd.to_datetime(api["date"])
        fig.add_trace(
            go.Scatter(
                x=subset["date"].dt.strftime("%Y-%m-%d").tolist(),
                y=subset["count"],
                mode="lines",
                name="deposit (defi_tvl.csv)",
                line={"color": "#636EFA", "width": 2},
                hoverinfo="skip",
                legendgroup="deposit",
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=api["date"].dt.strftime("%Y-%m-%d").tolist(),
                y=api["tvl_usd"],
                mode="lines",
                name="DefiLlama API",
                line={"color": "#EF553B", "dash": "dot", "width": 1.5},
                hoverinfo="skip",
                legendgroup="api",
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
        )
        wrote_any = True

    if not wrote_any:
        return False

    fig.update_layout(
        template="plotly_white",
        width=1200,
        height=780,
        font={"size": 16},
        margin={"l": 70, "r": 20, "t": 50, "b": 70},
        legend={"orientation": "h", "yanchor": "top", "y": -0.07, "x": 0, "font": {"size": 14},
                "bgcolor": "rgba(0,0,0,0)"},
    )
    fig.update_annotations(font={"size": 18})
    fig.update_yaxes(title_text="TVL (USD)", title_font={"size": 16}, tickfont={"size": 14})
    fig.update_xaxes(type="date", title_font={"size": 16}, tickfont={"size": 14})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), scale=3)
    return True


def validate_metrics_file(path: Path) -> dict:
    """Schema, key uniqueness, missing values, metric ranges, and single-entity consistency."""
    full = pd.read_csv(path)
    missing_cols = [c for c in METRIC_COLUMNS if c not in full.columns]
    if missing_cols:
        return {"status": "error", "reason": f"missing columns {missing_cols}"}
    # developers and TVL files repeat group-period metrics across entity rows
    df = full.drop_duplicates(["Date", "Group"]).reset_index(drop=True)
    n = df["Count"].astype(float)
    tol = 1e-6  # metrics are stored to six decimals
    violations = {
        "entropy_range": int(((df["Entropy"] < -tol) | (df["Entropy"] > np.log2(n.clip(lower=1)) + tol)).sum()),
        "gini_range": int(((df["Gini"] < -tol) | (df["Gini"] > 1 + tol)).sum()),
        "hhi_range": int(((df["HHI"] < 1 / n - tol) | (df["HHI"] > 1 + tol)).sum()),
        "nakamoto_range": int(((df["Nakamoto"] < 1) | (df["Nakamoto"] > n)).sum()),
        "count_positive": int((n < 1).sum()),
        "total_positive": int((df["Total"] <= 0).sum()),
        "share_range": int(sum(((df[c] < -tol) | (df[c] > 1 + tol)).sum() for c in ["Top1Share", "Top5Share", "Top10Share"])),
        "share_monotone": int(((df["Top1Share"] > df["Top5Share"] + tol) | (df["Top5Share"] > df["Top10Share"] + tol)).sum()),
        # Nakamoto (51%) must agree with the top-k shares
        "nakamoto_vs_shares": int((((df["Top1Share"] >= 0.51) != (df["Nakamoto"] == 1))
                                   | ((df["Top5Share"] >= 0.51) != (df["Nakamoto"] <= 5))
                                   | ((df["Top10Share"] >= 0.51) != (df["Nakamoto"] <= 10))).sum()),
    }
    if "MEVShare" in df.columns:
        eth = df["Group"] == "Ethereum"
        violations["mev_share"] = int(((df.loc[eth, "MEVShare"] < -tol) | (df.loc[eth, "MEVShare"] > 1 + tol)).sum()
                                      + df.loc[~eth, "MEVShare"].notna().sum() + df.loc[eth, "MEVShare"].isna().sum())
    single = df[df["Count"] == 1]
    single_ok = int(
        ((single["Entropy"].abs() > tol) | ((single["HHI"] - 1).abs() > tol) | (single["Nakamoto"] != 1)).sum()
    )
    return {
        "status": "ok",
        "rows": int(len(full)),
        "observations": int(len(df)),
        "groups": int(df["Group"].nunique()),
        "duplicate_keys": int(_duplicate_keys(full)),
        "missing_values": int(df[METRIC_COLUMNS].isna().sum().sum()),
        "columns": list(df.columns),
        "entity_rows": _entity_consistency(full),
        "range_violations": violations,
        "single_entity_rows": int(len(single)),
        "single_entity_inconsistent": single_ok,
    }


def _duplicate_keys(full: pd.DataFrame) -> int:
    if "AuthorId" in full.columns:
        key = ["Date", "Group", "AuthorId", "Client"]
    elif "Protocol" in full.columns:
        key = ["Date", "Protocol"]
    else:
        key = ["Date", "Group"]
    return int(full.duplicated(subset=key, keep=False).sum())


def _entity_consistency(full: pd.DataFrame) -> dict | None:
    """For files with entity rows: metrics constant within a group-period, and entity
    rows reproduce Total and Count."""
    if "AuthorId" in full.columns:
        value, entity = "Commits", "AuthorId"
    elif "Protocol" in full.columns:
        value, entity = "ProtocolTVL", "Protocol"
    else:
        return None
    g = full.groupby(["Date", "Group"])
    const = int((g[[c for c in METRIC_COLUMNS if c not in ("Date", "Subsystem", "Group")]].nunique() > 1).any(axis=1).sum())
    pos = full[full[value] > 0]
    agg = pos.groupby(["Date", "Group"]).agg(total=(value, "sum"), n=(entity, "nunique"))
    first = g[["Total", "Count"]].first()
    j = first.join(agg)
    return {
        "entities": int(full[entity].nunique()),
        "negative_values": int((full[value] < 0).sum()),
        "group_periods_with_varying_metrics": const,
        "total_mismatch": int(((j["Total"] - j["total"]).abs() > 1e-2 + 1e-6 * j["total"].abs()).sum()),
        "count_mismatch": int((j["Count"] != j["n"]).sum()),
    }


# entity file -> (metrics file, entity column, value column, key columns, group column or None)
ENTITY_FILES = {
    "consensus_producers.csv": ("consensus.csv", "entity", "count", ["date", "entity", "group"], "group"),
    "developers_commits.csv": ("developers.csv", "author_id", "count", ["date", "author_id", "client", "group"], "group"),
    "defi_gov_holders.csv": ("defi_gov.csv", "entity", "count", ["date", "entity", "group"], "group"),
    "nft_marketplace_sales.csv": ("nft_marketplace.csv", "entity", "count", ["date", "entity"], None),
}
GROUP_NAME = {"btc": "Bitcoin", "eth": "Ethereum", "bnb": "BNB", "ron": "Ronin", "sol": "Solana", "ton": "TON", "tron": "Tron",
              "bitcoin": "Bitcoin", "polygon": "Polygon", "ronin": "Ronin", "solana": "Solana",
              "ethereum-execution": "Ethereum (execution)", "ethereum-consensus": "Ethereum (consensus)",
              "aave": "Aave", "compound": "Compound", "curve": "Curve", "lido": "Lido", "maker": "MakerDAO", "uni": "Uniswap"}


def validate_entity_files(deposit_dir: Path) -> dict:
    """Each entity-level file must be non-negative, unique on its key, and reproduce
    Total and Count of its metrics file within every group-period."""
    out: dict = {}
    for filename, (metrics_file, entity, value, key, group) in ENTITY_FILES.items():
        path = deposit_dir / filename
        if not path.exists():
            continue
        ent = pd.read_csv(path, usecols=key + [value], low_memory=False)
        met = pd.read_csv(deposit_dir / metrics_file)
        pos = ent[ent[value] > 0]
        if group:
            agg = pos.groupby(["date", group]).agg(total=(value, "sum"), n=(entity, "nunique")).reset_index()
            agg["Group"] = agg[group].map(lambda g: GROUP_NAME.get(g, g))
        else:
            agg = pos.groupby("date").agg(total=(value, "sum"), n=(entity, "nunique")).reset_index()
            agg["Group"] = met["Group"].iloc[0]
        m = met.merge(agg, left_on=["Date", "Group"], right_on=["date", "Group"], how="outer", indicator=True)
        out[filename] = {
            "status": "ok", "rows": int(len(ent)), "entities": int(ent[entity].nunique()),
            "negative_values": int((ent[value] < 0).sum()),
            "duplicate_keys": int(ent.duplicated(key, keep=False).sum()),
            "unmatched_group_periods": int((m["_merge"] != "both").sum()),
            "total_mismatch": int(((m["Total"] - m["total"]).abs() > 1e-2 + 1e-6 * m["total"].abs()).sum()),
            "count_mismatch": int((m["Count"] != m["n"]).sum()),
        }
    return out


def validate_dashboard_agreement(deposit_path: Path, dashboard_path: Path) -> dict:
    """Row-for-row agreement between the deposit file and the companion dashboard export."""
    dep = pd.read_csv(deposit_path).drop_duplicates(["Date", "Group"])
    dash = pd.read_csv(dashboard_path)
    dash["Date"] = pd.to_datetime(dash["Date"], utc=True).dt.strftime("%Y-%m-%d")
    dep["Date"] = pd.to_datetime(dep["Date"], utc=True).dt.strftime("%Y-%m-%d")
    merged = dep.merge(dash, on=["Date", "Group"], how="outer", suffixes=("", "_dash"), indicator=True)
    both = merged[merged["_merge"] == "both"]
    diffs = {c: int(((both[c] - both[f"{c}_dash"]).abs() > 1e-5).sum()) for c in ["Entropy", "Nakamoto", "Gini", "HHI", "Count"]}
    return {
        "status": "ok",
        "deposit_rows": int(len(dep)),
        "dashboard_rows": int(len(dash)),
        "matched_rows": int(len(both)),
        "unmatched_rows": int((merged["_merge"] != "both").sum()),
        "differing_values": diffs,
    }


def build_checks(deposit_dir: Path, internal_dir: Path | None, dashboard_dir: Path | None) -> dict:
    checks: dict = {"deposit": {}, "internal": {}}

    for name, (filename, _) in SUBSYSTEM_FILES.items():
        path = deposit_dir / filename
        if not path.exists():
            checks["deposit"][name] = {"status": "pending", "reason": f"missing {filename}"}
            continue
        entry = validate_metrics_file(path)
        if dashboard_dir is not None and (dashboard_dir / filename).exists():
            entry["dashboard"] = validate_dashboard_agreement(path, dashboard_dir / filename)
        checks["deposit"][name] = entry

    checks["entity_files"] = validate_entity_files(deposit_dir)

    if internal_dir is None:
        checks["internal"] = {"status": "skipped", "reason": "no internal directory given"}
        return checks

    for name, (filename, key_cols) in SUBSYSTEM_FILES.items():
        path = internal_dir / filename
        if path.exists():
            checks["internal"][f"{name}_integrity"] = validate_row_integrity(path, key_cols)
    consensus = internal_dir / "consensus.csv"
    if consensus.exists():
        checks["internal"]["bitcoin_fractional"] = validate_bitcoin_fractional(consensus)
        checks["internal"]["consensus_daily_blocks"] = validate_consensus_daily_blocks(consensus)
    developers = internal_dir / "developers.csv"
    if developers.exists():
        checks["internal"]["developers_dedup"] = validate_developers(developers)
    defi_tvl = internal_dir / "defi_tvl.csv"
    if defi_tvl.exists():
        try:
            checks["internal"]["defillama_protocols"] = validate_defillama_protocols(defi_tvl)
        except Exception as exc:  # noqa: BLE001
            checks["internal"]["defillama_protocols"] = {"status": "error", "reason": str(exc)}
    return checks


def _consensus_sanity_result(consensus_check: dict) -> str:
    if consensus_check.get("status") != "ok":
        return "---"
    groups = consensus_check.get("groups", {})
    parts: list[str] = []
    for code, label in [("btc", "BTC"), ("eth_pre_merge", "ETH pre-Merge"), ("eth_post_merge", "ETH post-Merge")]:
        entry = groups.get(code, {})
        median = entry.get("median_daily_blocks")
        if median is None:
            parts.append(f"{label} ---")
            continue
        expected = entry.get("expected_approx")
        within = entry.get("within_range")
        if expected is not None and within is not None:
            mark = "ok" if within else "check"
            parts.append(f"{label} {median:,.0f}/day ({mark} vs {expected:,.0f})")
        else:
            parts.append(f"{label} {median:,.0f}/day")
    return "; ".join(parts)


def render_validation_tex(checks: dict, out_path: Path) -> None:
    dep = [v for v in checks["deposit"].values() if v.get("status") == "ok"]
    rows_total = sum(v["observations"] for v in dep)
    dupes = sum(v["duplicate_keys"] for v in dep)
    missing = sum(v["missing_values"] for v in dep)
    range_viol = sum(sum(v["range_violations"].values()) for v in dep)
    range_result = "all within range at six-decimal precision" if range_viol == 0 else f"{range_viol:,} outside range"
    single_rows = sum(v["single_entity_rows"] for v in dep)
    single_bad = sum(v["single_entity_inconsistent"] for v in dep)
    dash = [v["dashboard"] for v in dep if "dashboard" in v]
    dash_matched = sum(d["matched_rows"] for d in dash)
    dash_bad = sum(d["unmatched_rows"] + sum(d["differing_values"].values()) for d in dash)

    internal = checks.get("internal", {})
    btc = internal.get("bitcoin_fractional", {})
    dev = internal.get("developers_dedup", {})
    defi = internal.get("defillama_protocols", {})
    integ = [v for k, v in internal.items() if k.endswith("_integrity")]

    lines = [
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\caption{Validation summary. Deposit checks run on the released metrics files; "
        "internal checks run on the entity-level counts from which the metrics are computed, which are not redistributed.}",
        "    \\label{tab:validation}",
        "    \\small",
        "    \\begin{tabular}{@{}p{0.22\\linewidth}p{0.14\\linewidth}p{0.28\\linewidth}p{0.28\\linewidth}@{}}",
        "        \\toprule",
        "        Check & Layer & Criterion & Result \\\\",
        "        \\midrule",
        f"        Schema and keys & Metrics & twelve shared columns; unique keys; no missing values & {rows_total:,} group-periods; {dupes} duplicates; {missing} missing \\\\",
        f"        Metric ranges & Metrics & $0 \\le E \\le \\log_2 n$; $0 \\le \\mathrm{{Gini}} \\le 1$; $1/n \\le \\mathrm{{HHI}} \\le 1$; $1 \\le N \\le n$; $0 \\le S_1 \\le S_5 \\le S_{{10}} \\le 1$; $N$ consistent with $S_k$ & {range_result} \\\\",
        f"        Single-entity periods & Metrics & $n = 1$ implies $E = 0$, $\\mathrm{{HHI}} = 1$, $N = 1$ & {single_rows:,} rows; {single_bad} inconsistent \\\\",
    ]
    if dash:
        lines.append(f"        Dashboard agreement & Metrics & identical to companion dashboard series & {dash_matched:,} rows matched; {dash_bad} differences \\\\")
    ent = checks.get("entity_files", {})
    for filename, (metrics_file, *_rest) in ENTITY_FILES.items():
        e = ent.get(filename)
        if e:
            bad = e["negative_values"] + e["duplicate_keys"] + e["unmatched_group_periods"] + e["total_mismatch"] + e["count_mismatch"]
            label = filename.replace(".csv", "").replace("_", "\\_")
            lines.append(f"        \\texttt{{{label}}} & Entity files & non-negative; unique keys; per-period sums and entity counts equal \\texttt{{{metrics_file.replace('_', chr(92) + '_')}}} & {e['rows']:,} rows; {e['entities']:,} entities; {bad} discrepancies \\\\")
    if integ:
        neg = sum(v["negative_counts"] for v in integ)
        dup = sum(v["duplicate_keys"] for v in integ)
        lines.append(f"        Entity-count integrity & Entity counts & \\texttt{{count}} $\\geq 0$; unique date-entity keys & {neg} negative; {dup} duplicates \\\\")
    if btc.get("status") == "ok":
        lines.append(
            f"        Fractional block conservation & Entity counts (Bitcoin) & daily sums $\\approx$ integer blocks & "
            f"{_fmt_pct(btc['exact_integer_share'])} days exact to $10^{{-6}}$; max dev. {_fmt_sci(btc['max_fractional_deviation_blocks'])} blocks \\\\"
        )
    if internal.get("consensus_daily_blocks", {}).get("status") == "ok":
        lines.append(f"        Daily block sanity & Entity counts (consensus) & median blocks/day vs baseline & {_consensus_sanity_result(internal['consensus_daily_blocks'])} \\\\")
    if defi.get("status") == "ok":
        for label, _, _ in DEFILLAMA_PROTOCOLS:
            r = defi["protocols"].get(label.lower(), {})
            if r.get("status") == "ok":
                lines.append(
                    f"        DefiLlama {label} spot check & Entity counts (DeFi TVL) & ingested vs API TVL & "
                    f"{_fmt_usd_b(r['deposit_tvl_usd'])} vs {_fmt_usd_b(r['api_tvl_usd'])} ({_fmt_pct(r['latest_pct_diff'])} diff.) \\\\"
                )
    if dev.get("status") == "ok":
        lines.append(
            f"        Developer author aggregation & Entity counts (developers) & one entity per author-month-group & "
            f"{dev['raw_rows']:,} author-client rows $\\to$ {dev['deduped_rows']:,} entities; {dev['bot_rows']} bot rows \\\\"
        )
    lines.extend(["        \\bottomrule", "    \\end{tabular}", "\\end{table}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main(deposit_dir: Path = DEPOSIT_DIR, internal_dir: Path | None = None, dashboard_dir: Path | None = None) -> None:
    checks = build_checks(deposit_dir, internal_dir, dashboard_dir)
    results = {"generated_on": str(date.today()), "checks": checks}

    json_path = deposit_dir / "validation_summary.json"
    deposit_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2))

    tex_path = TABLES_DIR / "validation_summary.tex"
    render_validation_tex(checks, tex_path)

    fig_path = FIGURES_DIR / "validation_defillama_panel.png"
    defi_tvl = internal_dir / "defi_tvl.csv" if internal_dir else None
    wrote_fig = write_validation_panel(defi_tvl, fig_path)

    print(json.dumps(results, indent=2))
    print(f"\nWrote {json_path}")
    print(f"Wrote {tex_path}")
    print(f"Wrote {fig_path}" if wrote_fig else "Skipped DefiLlama figure (no internal defi_tvl.csv)")


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]]
    main(*(args or [DEPOSIT_DIR]))
