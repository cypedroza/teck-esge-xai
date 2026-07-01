"""
═══════════════════════════════════════════════════════════════════════════════
CONFIG.PY - CENTRALIZED CONFIGURATION
XAI-AHP-Gaussian ESGE Framework
═══════════════════════════════════════════════════════════════════════════════

This module centralizes all configuration parameters for the project.
Modify paths, hyperparameters, and settings here.

Author: Cesar Yoshio Machado Pedroza
Institution: USP/Esalq - MBA Data Science and Analytics
Last Updated: 2026-04-16
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging


@dataclass
class ESGEConfig:
    """
    Centralized configuration class for the ESGE Framework.
    
    Attributes:
        BASE_DIR: Root directory of the project
        TICKER: Yahoo Finance ticker symbol
        START_YEAR: Initial year for data collection
        END_YEAR: Final year for data collection
        RANDOM_STATE: Seed for reproducibility
        TEST_SIZE: Train/test split ratio
        N_SIMULATIONS: Monte Carlo iterations for AHP-Gaussian
        NOISE_STD: Standard deviation for Gaussian noise in AHP
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # BASE_DIR: resolves automatically from this file's location (src/config.py)
    # Override via env var ESGE_BASE_DIR if needed (e.g. Colab, CI)
    # ═══════════════════════════════════════════════════════════════════════
    BASE_DIR: Path = Path(
        os.environ.get("ESGE_BASE_DIR", str(Path(__file__).parent.parent))
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # DIRECTORY STRUCTURE (Auto-generated)
    # ═══════════════════════════════════════════════════════════════════════
    DATA_DIR: Path = field(init=False)
    DATA_RAW: Path = field(init=False)
    DATA_PROCESSED: Path = field(init=False)
    
    OUTPUTS_DIR: Path = field(init=False)
    OUTPUTS_FIGURES: Path = field(init=False)
    OUTPUTS_TABLES: Path = field(init=False)
    OUTPUTS_MODELS: Path = field(init=False)
    OUTPUTS_LOGS: Path = field(init=False)
    
    POWERBI_DIR: Path = field(init=False)
    
    # ═══════════════════════════════════════════════════════════════════════
    # DATA COLLECTION PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════
    TICKER: str = "TECK-B.TO"  # Teck Resources ticker (Toronto Stock Exchange)
    START_YEAR: int = 2001      # First year of analysis
    END_YEAR: int = 2024        # Last year of analysis
    
    # ═══════════════════════════════════════════════════════════════════════
    # MACHINE LEARNING PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════
    RANDOM_STATE: int = 42      # Seed for reproducibility
    TEST_SIZE: float = 0.2      # 80% train / 20% test
    
    # Random Forest
    RF_N_ESTIMATORS: int = 100
    RF_MAX_DEPTH: int = 10
    RF_MIN_SAMPLES_SPLIT: int = 2
    
    # XGBoost
    XGB_N_ESTIMATORS: int = 100
    XGB_MAX_DEPTH: int = 6
    XGB_LEARNING_RATE: float = 0.1
    XGB_SUBSAMPLE: float = 0.8
    
    # ═══════════════════════════════════════════════════════════════════════
    # XAI PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════
    SHAP_N_SAMPLES: int = 100   # Number of background samples for SHAP
    LIME_N_SAMPLES: int = 5000  # Number of perturbed samples for LIME
    DICE_N_CF: int = 5          # Number of counterfactuals to generate
    
    # ═══════════════════════════════════════════════════════════════════════
    # AHP-GAUSSIAN PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════
    N_SIMULATIONS: int = 10000  # Monte Carlo iterations
    NOISE_STD: float = 0.1      # Gaussian noise standard deviation
    CR_THRESHOLD: float = 0.10  # Consistency Ratio threshold (Saaty, 1980)
    
    # Random Index values (Saaty, 1980)
    RANDOM_INDEX: Dict[int, float] = field(default_factory=lambda: {
        1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
    })
    
    # ═══════════════════════════════════════════════════════════════════════
    # FEATURE NAMES AND MAPPING
    # ═══════════════════════════════════════════════════════════════════════
    # ESGE Criteria (4 dimensions)
    ESGE_CRITERIA: List[str] = field(default_factory=lambda: [
        'esg_disclosure_score',  # Environmental (E) proxy
        'char_count',             # Social (S) proxy - disclosure quality
        'annual_return_%',        # Economic (Ec) - financial performance
        'volume'                  # Governance (G) proxy - market liquidity
    ])
    
    # Feature display names for visualizations
    FEATURE_LABELS: Dict[str, str] = field(default_factory=lambda: {
        'esg_disclosure_score': 'ESG Disclosure Score',
        'char_count': 'Report Disclosure Quality',
        'annual_return_%': 'Annual Return (%)',
        'volume': 'Trading Volume',
        'close_price': 'Close Price (CAD)'
    })
    
    # ═══════════════════════════════════════════════════════════════════════
    # LOGGING CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'
    
    # ═══════════════════════════════════════════════════════════════════════
    # VISUALIZATION PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════
    FIG_DPI: int = 300          # DPI for exported figures
    FIG_FORMAT: str = 'png'     # Figure format (png, pdf, svg)
    FIG_STYLE: str = 'seaborn-v0_8-darkgrid'  # Matplotlib style
    
    # ═══════════════════════════════════════════════════════════════════════
    # POST-INIT: AUTO-CREATE DIRECTORY STRUCTURE
    # ═══════════════════════════════════════════════════════════════════════
    def __post_init__(self):
        """
        Automatically executed after initialization.
        Creates directory structure and validates paths.
        """
        # Define subdirectories
        self.DATA_DIR = self.BASE_DIR / "data"
        self.DATA_RAW = self.DATA_DIR / "raw"
        self.DATA_PROCESSED = self.DATA_DIR / "processed"
        
        self.OUTPUTS_DIR = self.BASE_DIR / "outputs"
        self.OUTPUTS_FIGURES = self.OUTPUTS_DIR / "figures"
        self.OUTPUTS_TABLES = self.OUTPUTS_DIR / "tables"
        self.OUTPUTS_MODELS = self.OUTPUTS_DIR / "models"
        self.OUTPUTS_LOGS = self.OUTPUTS_DIR / "logs"
        
        self.POWERBI_DIR = self.BASE_DIR / "powerbi_data"
        
        # Create all directories
        for directory in [
            self.DATA_RAW,
            self.DATA_PROCESSED,
            self.OUTPUTS_FIGURES,
            self.OUTPUTS_TABLES,
            self.OUTPUTS_MODELS,
            self.OUTPUTS_LOGS,
            self.POWERBI_DIR
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for the entire project."""
        log_file = self.OUTPUTS_LOGS / f"esge_framework_{self.get_timestamp()}.log"
        
        logging.basicConfig(
            level=self.LOG_LEVEL,
            format=self.LOG_FORMAT,
            datefmt=self.LOG_DATE_FORMAT,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Suppress noisy third-party loggers
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
    
    @staticmethod
    def get_timestamp() -> str:
        """
        Returns current timestamp for file naming.
        
        Returns:
            str: Timestamp in format YYYYMMDD_HHMMSS
        """
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def validate_paths(self) -> bool:
        """
        Validates that all required paths exist.
        
        Returns:
            bool: True if all paths are valid, False otherwise
        """
        paths_to_check = [
            self.DATA_RAW,
            self.DATA_PROCESSED,
            self.OUTPUTS_DIR,
            self.POWERBI_DIR
        ]
        
        for path in paths_to_check:
            if not path.exists():
                logging.error(f"Path does not exist: {path}")
                return False
        
        logging.info("✅ All paths validated successfully")
        return True
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        return (
            f"ESGEConfig(\n"
            f"  BASE_DIR={self.BASE_DIR},\n"
            f"  TICKER={self.TICKER},\n"
            f"  PERIOD={self.START_YEAR}-{self.END_YEAR},\n"
            f"  N_SIMULATIONS={self.N_SIMULATIONS},\n"
            f"  RANDOM_STATE={self.RANDOM_STATE}\n"
            f")"
        )


# ═══════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE (Import this in other modules)
# ═══════════════════════════════════════════════════════════════════════════

config = ESGEConfig()


# ═══════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Demonstration
    print("═" * 70)
    print("ESGE FRAMEWORK CONFIGURATION")
    print("═" * 70)
    print(config)
    print("\n" + "═" * 70)
    print("DIRECTORY STRUCTURE")
    print("═" * 70)
    print(f"📁 Base Directory: {config.BASE_DIR}")
    print(f"📂 Raw Data: {config.DATA_RAW}")
    print(f"📂 Processed Data: {config.DATA_PROCESSED}")
    print(f"📂 Outputs: {config.OUTPUTS_DIR}")
    print(f"📂 Power BI: {config.POWERBI_DIR}")
    print("\n" + "═" * 70)
    print("VALIDATION")
    print("═" * 70)
    config.validate_paths()
