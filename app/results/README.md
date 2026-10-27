# Result snapshot for the Streamlit app

`ahp_weights.csv` and `shap_importance.csv` are copies of the tables written by
`04_ahp_sensitivity_analysis.py` and `03_xai_triangulation_convergence.py`
(seed 42) into `outputs/tables/`. They let the app run where the pipeline has not
been executed, such as Streamlit Community Cloud.

The app always prefers `outputs/tables/` when it exists. After re-running the
pipeline, refresh the snapshot with:

    cp outputs/tables/ahp_weights.csv outputs/tables/shap_importance.csv app/results/
