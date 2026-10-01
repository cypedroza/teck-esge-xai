# Data

| File | Produced by | Contents |
|---|---|---|
| `raw/Teck_YYYY.pdf` | downloaded manually from Teck's website | 24 annual / sustainability reports (2001–2024). Not versioned (copyright, size). |
| `raw/prices_*.csv` | `05_financial_benchmarking_tsx.py` (first run) | Daily adjusted closes for `TECK-B.TO` and `^GSPTSE` from Yahoo Finance, cached so every later run uses the same series. Not versioned (Yahoo terms); delete them or run with `--refresh` to re-download. |
| `processed/esge_master.csv` | `notebooks/02_data_extraction.ipynb` | One row per year: `year`, `char_count` (report length), `esg_disclosure_score` (anchor-term count), `close_price`, `volume` (mean daily volume), `annual_return_%`. |
| `processed/esge_master_final.csv` | see note below | Same columns plus the modelling target `ESGE_score`. Required by scripts 02 and 03. |

## Variable definitions (proxies)

| Column | Article name | ESGE dimension (proxy) | Definition |
|---|---|---|---|
| `esg_disclosure_score` | ESG Disclosure Index | Environmental | Absolute count of anchor terms in the report text (not normalised by length). Measures volume of disclosure, not ESG performance. |
| `char_count` | Report Quality Score | Social | Number of characters extracted from the report (report length). |
| `annual_return_%` | Annual Return | Economic | Year-over-year change of the year-end adjusted close (%). |
| `volume` | Market Liquidity | Governance | Mean daily trading volume in the year. |

These are proxies with partial coverage of each dimension; see the article's limitations section.

## Note on `ESGE_score`

`ESGE_score` is the composite score used as the prediction target. Its exact
construction must be documented here before the modelling results are cited.
Script 02 checks whether it is (almost) a linear combination of the four
features — if it is, model R² values measure the reproduction of a formula,
not predictive ability.
