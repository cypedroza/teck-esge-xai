# Legacy notebooks (exploratory, superseded)

These notebooks are an early exploratory version of the pipeline and are kept
only for transparency. **They do not produce the results reported in the
thesis or the article.** In particular:

* `03_ml_xai_pipeline.ipynb` predicts `close_price` with a random train/test
  split; the final analysis predicts `ESGE_score` with temporal validation
  (`02_ml_advanced_validation.py`, `03_xai_triangulation_convergence.py`).
* `04_financial_analysis.ipynb` uses the sample mean as the "expected return"
  in the event analysis; the final version is `05_financial_benchmarking_tsx.py`.
* `05_ahp_gaussian.ipynb` uses an earlier pairwise matrix (weights ≈ 0.33 /
  0.27 / 0.22 / 0.18); the final matrix and weights are in
  `04_ahp_sensitivity_analysis.py` and `src/config.py`.
