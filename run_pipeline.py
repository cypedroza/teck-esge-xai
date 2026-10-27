"""
Run the analysis in the correct order and stop at the first error.

    python run_pipeline.py

Order: 01 statistics → 02 explanatory model → 03 XAI → 04 AHP → 05 finance.
The ML/XAI module uses the reformulated target (log ESG Disclosure Index);
the earlier ESGE_score target was invalid (see data/README.md).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["01_comprehensive_statistical_analysis.py",
         "02_ml_advanced_validation.py",
         "03_xai_triangulation_convergence.py",
         "04_ahp_sensitivity_analysis.py",
         "05_financial_benchmarking_tsx.py"]


def main() -> None:
    for script in STEPS:
        print(f"\n{'=' * 78}\n▶ {script}\n{'=' * 78}", flush=True)
        result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f"\n✗ {script} failed (exit {result.returncode}). Fix it before continuing.")
    print("\n✓ Pipeline finished. Tables: outputs/tables/  Figures: outputs/figures/")


if __name__ == "__main__":
    main()
