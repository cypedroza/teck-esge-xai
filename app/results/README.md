# Result snapshot for the Streamlit app

The dashboard (`app/streamlit_app.py`) reads its numbers from the result tables
written by the pipeline into `outputs/tables/`. That folder is not versioned, so
this folder keeps a copy of the tables from the run that produced the corrected
article (scripts 01–04 on 22 Oct 2026, script 05 on 26 Oct 2026 with real Yahoo
Finance prices). It lets the app run where the pipeline has not been executed,
such as Streamlit Community Cloud.

| File | Written by |
|---|---|
| `chow_test_structural_break.csv`, `descriptive_statistics.csv` | `01_comprehensive_statistical_analysis.py` |
| `model_performance.csv`, `permutation_importance_xgboost.csv` | `02_ml_advanced_validation.py` |
| `shap_importance.csv`, `shap_lime_comparison.csv`, `shap_lime_convergence.csv`, `dice_counterfactuals.csv` | `03_xai_triangulation_convergence.py` |
| `ahp_weights.csv`, `ahp_sensitivity_analysis.csv` | `04_ahp_sensitivity_analysis.py` |
| `financial_metrics*.csv`, `*mean_tests*.csv`, `annual_returns_*.csv`, `event_study_mount_polley*.csv` | `05_financial_benchmarking_tsx.py` |

The app always prefers `outputs/tables/` when a file exists there. After
re-running the pipeline, refresh the snapshot with:

    cp outputs/tables/*.csv app/results/
