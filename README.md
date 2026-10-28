# crypto-decentralization-data

Code for *A Dataset of Decentralization Metrics Across Blockchain Ecosystems* (Ju, Zain & Aral, Scientific Data).

- Dataset: https://doi.org/10.6084/m9.figshare.33752596 (CC BY 4.0)
- Dashboard: https://bc-crypto-lab.web.app/

The pipeline builds entity-level activity counts for five subsystems (consensus, developers, DeFi governance, DeFi TVL, NFT marketplaces) and computes Shannon entropy, the Nakamoto coefficient (51% threshold), the Gini coefficient, HHI, the number of active entities, total activity, and top-1/5/10 shares for every group and period.

## Layout

| Path | Purpose |
|------|---------|
| `sql/` | Dune Analytics queries for consensus, DeFi governance, and NFT marketplaces |
| `scripts/pull_developers.py` | Clone core client repositories and count monthly commits by author |
| `scripts/pull_defi_tvl.py` | Download the DefiLlama bulk protocol export and reshape it to protocol-day TVL |
| `scripts/compute_metrics.py` | Concentration metrics from entity-level counts |
| `scripts/validate_dataset.py` | Schema, key, range, and reconciliation checks |
| `scripts/generate_validation_summary.py` | Validation table and DefiLlama comparison figure |
| `scripts/generate_data_overview.py` | Overview table and figure |
| `scripts/summarize_deposit.py` | Row counts and date ranges of the released files |
| `data/` | Released CSVs (download from Figshare; not tracked) |
| `outputs/` | Generated tables and figures (not tracked) |

## Usage

```zsh
uv sync
uv run python scripts/compute_metrics.py
uv run python scripts/validate_dataset.py
uv run python scripts/generate_validation_summary.py
uv run python scripts/generate_data_overview.py
```

## License

Code: MIT. Data: CC BY 4.0 (see Figshare).
