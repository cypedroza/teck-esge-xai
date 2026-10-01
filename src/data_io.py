"""
Dataset loading, column naming and integrity checks shared by all scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Raw column names (produced by notebooks/02_data_extraction.ipynb)
# → descriptive names used in the analysis and in the article.
FEATURE_MAPPING = {
    "esg_disclosure_score": "ESG_Disclosure_Index",
    "char_count": "Report_Quality_Score",
    "annual_return_%": "Annual_Return_Pct",
    "volume": "Market_Liquidity",
}

FEATURES: List[str] = [
    "ESG_Disclosure_Index",
    "Report_Quality_Score",
    "Annual_Return_Pct",
    "Market_Liquidity",
]

TARGET = "ESGE_score"


def normalize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=FEATURE_MAPPING)


def load_master(path: Path) -> pd.DataFrame:
    """Load the annual dataset, rename columns and sort by year."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run notebooks/02_data_extraction.ipynb first, or place "
            "the processed dataset in data/processed/ (see data/README.md)."
        )
    df = normalize_feature_names(pd.read_csv(path))
    if "year" not in df.columns:
        raise ValueError(f"{path.name} must contain a 'year' column.")
    df["year"] = df["year"].astype(int)
    if df["year"].duplicated().any():
        raise ValueError(f"Duplicated years in {path.name}.")
    return df.sort_values("year").reset_index(drop=True)


def returns_by_year(df: pd.DataFrame, column: str = "Annual_Return_Pct") -> pd.Series:
    """Annual returns (decimal) indexed by year label, NaNs removed."""
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found.")
    s = df.set_index("year")[column].dropna() / 100.0
    s.name = "return"
    return s


def modelling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with complete features and target, still carrying the year."""
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Columns missing for modelling: {missing}. The modelling scripts need the "
            f"dataset that contains '{TARGET}' (data/processed/esge_master_final.csv)."
        )
    return df[["year"] + FEATURES + [TARGET]].dropna().reset_index(drop=True)


def target_circularity(X: np.ndarray, y: np.ndarray) -> float:
    """In-sample R² of an ordinary least-squares fit of y on X.

    If the target is (close to) a linear combination of the features — for
    example a weighted sum Σ wᵢ·xᵢ of the same variables — this value is ≈ 1,
    and any model's R² says nothing about predictive ability: the model is
    reproducing a formula, not learning a relationship.
    """
    X1 = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ coef
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
