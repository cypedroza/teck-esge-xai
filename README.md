# XAI-AHP-Gaussian ESGE Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code for a longitudinal single-case study of **Teck Resources Limited (TSX: TECK-B), 2001–2024**, that combines
explainable machine learning (SHAP, LIME, DiCE), a Gaussian (Monte Carlo) extension of the Analytic Hierarchy
Process, and financial benchmarking against the S&P/TSX Composite. It accompanies the author's MBA thesis in Data
Science and Analytics at USP/Esalq (advisor: Prof. Dr. Arthur Damasceno Vicente).

> **Scope.** ESG is measured here through disclosure proxies built from public reports and market data (see
> [`data/README.md`](data/README.md)). They measure the *volume of disclosure*, not ESG performance. The sample has
> 23–24 annual observations of one company; all results should be read as exploratory.

---

## Repository layout

```
teck-esge-xai/
├── 01_comprehensive_statistical_analysis.py   # descriptives, normality, Chow test (2018), correlations
├── 02_ml_advanced_validation.py               # baselines, RF/XGBoost, TimeSeriesSplit CV, OOB bootstrap
├── 03_xai_triangulation_convergence.py        # SHAP, LIME (k-medoids instances), SHAP–LIME convergence, DiCE
├── 04_ahp_sensitivity_analysis.py             # Gaussian AHP, σ ∈ {0.05,…,0.20}, P(CR<0.10), rank reversal
├── 05_financial_benchmarking_tsx.py           # Teck vs ^GSPTSE metrics, Mount Polley event analysis
├── run_pipeline.py                            # runs the scripts in order
├── CLAUDE.md                                  # rules for AI assistants (data integrity, verification)
├── src/
│   ├── config.py      # paths (relative), seeds, parameters, AHP judgment matrix
│   ├── data_io.py     # loading, column names, target-circularity check
│   ├── finance.py     # returns, risk metrics, event studies (daily market model; annual mean-adjusted)
│   └── ahp.py         # eigenvector weights, CR, Monte Carlo perturbation
├── notebooks/
│   ├── 01_setup.ipynb
│   ├── 02_data_extraction.ipynb   # PDF text mining + Yahoo Finance → data/processed/esge_master.csv
│   ├── 06_powerbi_export.ipynb    # star schema for Power BI
│   └── legacy/                    # early exploratory notebooks, superseded (see legacy/README.md)
├── app/streamlit_app.py           # decision-support dashboard
├── app/results/                   # result snapshot used by the app when outputs/ is empty
├── tests/                         # unit tests for finance, AHP and data checks
├── data/README.md                 # data sources and variable definitions
├── requirements.txt / requirements-dev.txt
└── CITATION.cff
```

## Reproducing the analysis

```bash
git clone https://github.com/cypedroza/teck-esge-xai.git
cd teck-esge-xai
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# 1) data: data/processed/esge_master.csv is versioned; to rebuild it, place the
#    reports in data/raw/ and run notebooks/02_data_extraction.ipynb
# 2) analysis (runs 01 → 04 → 05 and stops at the first error)
python run_pipeline.py
#    01 → 02 → 03 → 04 → 05; script 05 downloads and caches ^GSPTSE and TECK-B.TO prices
# 3) dashboard
streamlit run app/streamlit_app.py

# tests
pytest -q
```

Paths are resolved relative to the repository (override with the environment variable `ESGE_BASE_DIR`). Every
random step is seeded with `RANDOM_STATE = 42` in `src/config.py`. Tables are written to `outputs/tables/` and
figures to `outputs/figures/`.

## Methods in brief

| Module | Implementation |
|---|---|
| Structural break | Chow test at 2018 on a linear trend `y = β₀ + β₁·year`, pre (2001–2017) vs post (2018–2024). |
| Explanatory model | Target: log(1 + ESG Disclosure Index), the article's central variable, not built from the features (checked by an OLS circularity test). Features: report length, market liquidity, annual return, year-end price. Mean and linear baselines, Random Forest and XGBoost tuned by leave-one-out. Leave-one-out R² is the headline out-of-sample metric; TimeSeriesSplit R² is also reported and is strongly negative because of the 2018 break, so the module is explanatory rather than predictive; out-of-bag bootstrap intervals. |
| XAI | SHAP `TreeExplainer` (global mean \|φ\|); LIME on 10 k-medoids years with 1,000 perturbations each; Pearson and Spearman agreement between normalised importances (4 features, df = 2: descriptive only); DiCE counterfactuals (random sampling — gradient-based DiCE does not apply to tree ensembles) with sparsity and proximity. |
| Gaussian AHP | 4×4 Saaty matrix (E > S > G > Ec) set a priori by the author; each upper-triangular judgment multiplied by N(1, σ²); diagonal fixed at 1, lower triangle reciprocal; 10,000 simulations per σ. |
| Finance | Annual returns from year-end closes (Yahoo Finance). ^GSPTSE is a price index (no dividends), so the main comparison is **price vs price**: TECK-B.TO closes adjusted for splits only vs ^GSPTSE, 2001–2024. A **total-return check** compares dividend-adjusted TECK-B.TO with XIC.TO (S&P/TSX Capped Composite ETF, distributions reinvested), 2002–2024. Sharpe, Sortino, VaR/CVaR 95%, max drawdown, annual and daily beta, Treynor (risk-free rate in `config.py`); paired and Welch t-tests. |
| Event analysis | Mount Polley tailings breach, 4 Aug 2014 — an **Imperial Metals** operation, not a Teck asset. (a) Daily market model, estimation window −260…−11 trading days, event window 0…+2, CAR t-test; market = ^GSPTSE, with XIC.TO as a robustness check; the TSX was closed on 4 Aug 2014 (Civic Holiday), so day 0 is 5 Aug; (b) annual mean-adjusted abnormal return as an exploratory check. Events are located by date/year label, never by array position. |

## Revision history

**October 2026 — reproducibility revision.** Hard-coded local paths removed; all randomness seeded; the market
benchmark is now downloaded from Yahoo Finance (`^GSPTSE`) and cached, and the pipeline stops instead of
simulating data if the download fails; the event analysis locates the event year by label and adds the daily
market-model design; bootstrap intervals are now out-of-bag; LIME and DiCE are implemented as described in the
methods; the Gaussian AHP keeps the matrix diagonal at 1; unit tests added; exploratory notebooks moved to
`notebooks/legacy/`. The benchmark compares like with like (price vs price, plus a total-return check), since the
earlier code set dividend-adjusted Teck returns against a price index. Numerical results produced by this version
may differ from earlier outputs.

## How to cite

See [`CITATION.cff`](CITATION.cff) (GitHub shows a "Cite this repository" button).

## Author

Cesar Yoshio Machado Pedroza — MBA in Data Science and Analytics, USP/Esalq ·
[GitHub @cypedroza](https://github.com/cypedroza) · [LinkedIn](https://www.linkedin.com/in/cesar-y-m-pedroza)

## License

MIT — see [LICENSE](LICENSE).
