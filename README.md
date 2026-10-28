# crypto-decentralization-data

Code for *A Dataset of Decentralization Metrics Across Blockchain Ecosystems* (Ju, Zain & Aral, Scientific Data).

- Dataset: https://doi.org/10.6084/m9.figshare.33752596 (CC BY 4.0)
- Dashboard: https://bc-crypto-lab.web.app/

The pipeline builds entity-level activity counts for five subsystems (consensus, developers, DeFi governance, DeFi TVL, NFT marketplaces) and computes Shannon entropy, the Nakamoto coefficient (51% threshold), the Gini coefficient, HHI, the number of active entities, total activity, and top-1/5/10 shares for every group and period.

## Layout

| Path | Purpose |
|------|---------|
| `sql/` | Dune Analytics queries for consensus, DeFi governance, and NFT marketplaces |
| `config/developers_repos.csv` | Core client repositories tracked for the developer subsystem |
| `src/metrics.py` | Entropy, Gini, HHI, and Nakamoto (51%) helpers used by the pull scripts |
| `scripts/assemble_consensus.py` | Entity-level consensus panel from per-chain Dune exports (Ethereum builder attribution, Ronin cutoff) |
| `scripts/pull_developers.py` | Clone core client repositories and count monthly commits by author |
| `scripts/pull_defi_tvl.py` | DefiLlama bulk export to daily DeFi TVL metrics |
| `scripts/compute_metrics.py` | Concentration metrics from entity-level counts |
| `scripts/export_metrics_deposit.py` | Write the released metrics and entity-level files (pseudonymized authors) |
| `scripts/validate_dataset.py`, `scripts/generate_validation_summary.py` | Schema, key, range, and reconciliation checks; validation table and DefiLlama figure |
| `scripts/generate_data_overview.py`, `scripts/plot_entropy.py`, `scripts/plot_developers.py` | Overview table and figures |
| `scripts/summarize_deposit.py` | Row counts and date ranges of the released files |
| `data/` | Released CSVs (download from Figshare; not tracked) |
| `outputs/` | Generated tables and figures (not tracked) |

## Usage

Download the Figshare files into `data/`, then:

```zsh
uv sync
uv run plotly_get_chrome -y   # once, for PNG export
uv run python scripts/summarize_deposit.py data
uv run python scripts/generate_data_overview.py data
```

Rebuilding the entity-level counts needs the Dune exports and Git clones; each script's `--help` lists its inputs.

## License

Code: MIT. Data: CC BY 4.0 (see Figshare).
