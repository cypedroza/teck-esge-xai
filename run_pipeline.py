"""
Run the analysis in the correct order and stop at the first error.

    python run_pipeline.py            # 01 → 04 → 05 (validated modules)
    python run_pipeline.py --with-ml  # also 02 → 03 (needs a validated ESGE_score target)

Modules 02 and 03 are excluded by default: the target `ESGE_score` used in
earlier runs had E, S and G components equal to zero (see data/README.md).
They will be re-enabled once a new target is defined and documented.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE = ["01_comprehensive_statistical_analysis.py",
        "04_ahp_sensitivity_analysis.py",
        "05_financial_benchmarking_tsx.py"]
ML = ["02_ml_advanced_validation.py",
      "03_xai_triangulation_convergence.py"]


def main() -> None:
    steps = CORE + (ML if "--with-ml" in sys.argv else [])
    for script in steps:
        print(f"\n{'=' * 78}\n▶ {script}\n{'=' * 78}", flush=True)
        result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f"\n✗ {script} failed (exit {result.returncode}). Fix it before continuing.")
    print("\n✓ Pipeline finished. Tables: outputs/tables/  Figures: outputs/figures/")


if __name__ == "__main__":
    main()
