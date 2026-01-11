"""
Main Entry Point for CPI Inflation Forecasting Project
=======================================================

This script runs the complete forecasting pipeline comparing machine learning
methods (Random Forest, LASSO) against traditional econometric approaches
(SARIMAX) for CPI inflation forecasting.

Research Question:
    Which forecasting methodology performs better for CPI inflation:
    ML (Random Forest, LASSO) or traditional econometrics (SARIMAX)?

Execution Order:
    1. Data Preprocessing
    2. ARIMAX/SARIMAX Models
    3. Random Forest Models
    4. LASSO Models
    5. Final Comparison

Usage:
    python main.py

Requirements:
    - All dependencies in requirements.txt must be installed
    - Raw data must be present in data/raw/

Output:
    - All results saved to results/ directory
    - Final comparison summary printed to console

Author: Elena Onieva Henrich
Date: January 2026
Course: Advanced Programming 2025 - Forecasting Project
University: Université de Lausanne
"""

import os
import sys
import time
import subprocess
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Script directories
DATA_DIR = PROJECT_ROOT / 'src' / 'data'
ARIMAX_DIR = PROJECT_ROOT / 'src' / 'models' / 'arimax'
RF_DIR = PROJECT_ROOT / 'src' / 'models' / 'random_forest'
LASSO_DIR = PROJECT_ROOT / 'src' / 'models' / 'lasso'
EVAL_DIR = PROJECT_ROOT / 'src' / 'evaluate_models'

# Python executable (use the same Python that runs this script)
PYTHON = sys.executable


# =============================================================================
# SCRIPT DEFINITIONS
# =============================================================================

# Scripts to run in order
# Format: (script_path, description)

DATA_SCRIPTS = [
    (DATA_DIR / '1_load_cpi.py', 'Load CPI data'),
    (DATA_DIR / '2_yoy_stationarity.py', 'Compute YoY and test stationarity'),
    (DATA_DIR / '3_load_unemployment.py', 'Load unemployment data'),
    (DATA_DIR / '4_load_import_prices.py', 'Load import prices data'),
    (DATA_DIR / '5_prep_data_GER.py', 'Prepare Germany high-dimensional data'),
]

ARIMAX_SCRIPTS = [
    (ARIMAX_DIR / '1_arimax_data.py', 'Prepare ARIMAX data'),
    (ARIMAX_DIR / '2_select_x_lags.py', 'Select exogenous lags'),
    (ARIMAX_DIR / '3_1_arimax_grid_search.py', 'ARIMAX grid search'),
    (ARIMAX_DIR / '3_2_acf_pacf_plots.py', 'ACF/PACF diagnostic plots'),
    (ARIMAX_DIR / '4_forecast_arimax.py', 'ARIMAX forecasting'),
    (ARIMAX_DIR / '5_1_sarima_forecast.py', 'SARIMAX forecast'),
    (ARIMAX_DIR / '5_2_sarima_comparison.py', 'SARIMAX comparison plots'),
]

RF_SCRIPTS = [
    (RF_DIR / '1_rf_standard_forecast.py', 'Random Forest standard forecast'),
    (RF_DIR / '2_rf_forecast_GER_highdim.py', 'Random Forest Germany high-dim'),
]

LASSO_SCRIPTS = [
    (LASSO_DIR / '1_lasso_standard_forecast.py', 'LASSO forecast'),
]

EVALUATION_SCRIPTS = [
    (EVAL_DIR / 'comparison.py', 'Final comparison summary'),
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(text, char='=', width=70):
    """Print a formatted header."""
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)


def print_subheader(text, char='-', width=70):
    """Print a formatted subheader."""
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)


def run_script(script_path, description):
    """
    Run a Python script and handle errors.

    Parameters
    ----------
    script_path : Path
        Path to the script to run.
    description : str
        Description of what the script does.

    Returns
    -------
    bool
        True if script ran successfully, False otherwise.
    """
    script_name = script_path.name

    # Check if script exists
    if not script_path.exists():
        print(f"  ⚠ SKIPPED: {script_name} (file not found)")
        return True  # Continue with other scripts

    print(f"\n  Running: {script_name}")
    print(f"  Purpose: {description}")
    print(f"  {'─' * 50}")

    start_time = time.time()

    try:
        # Run the script
        result = subprocess.run(
            [PYTHON, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"  ✓ Completed: {script_name} ({elapsed:.1f}s)")
            return True
        else:
            print(f"  ✗ FAILED: {script_name} (exit code {result.returncode})")
            return False

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ✗ ERROR: {script_name} - {str(e)} ({elapsed:.1f}s)")
        return False


def run_script_group(scripts, group_name):
    """
    Run a group of scripts.

    Parameters
    ----------
    scripts : list
        List of (script_path, description) tuples.
    group_name : str
        Name of the script group.

    Returns
    -------
    tuple
        (n_success, n_failed, n_skipped)
    """
    print_subheader(group_name)

    n_success = 0
    n_failed = 0
    n_skipped = 0

    for script_path, description in scripts:
        if not script_path.exists():
            print(f"  ⚠ SKIPPED: {script_path.name} (not found)")
            n_skipped += 1
            continue

        success = run_script(script_path, description)
        if success:
            n_success += 1
        else:
            n_failed += 1
            # Ask user if they want to continue after failure
            print(f"\n  Script failed. Continue anyway? (y/n): ", end='')
            try:
                response = input().strip().lower()
                if response != 'y':
                    print("  Aborting pipeline.")
                    return n_success, n_failed, n_skipped
            except EOFError:
                # Non-interactive mode, continue
                pass

    return n_success, n_failed, n_skipped


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main entry point for the forecasting pipeline.

    Runs all scripts in the correct order and produces a final summary.
    """
    total_start = time.time()

    # Print project header
    print_header("CPI INFLATION FORECASTING PROJECT", '=', 70)
    print()
    print("  Research Question:")
    print("  Can ML (Random Forest, LASSO) beat traditional econometrics")
    print("  (SARIMAX) for CPI inflation forecasting?")
    print()
    print("  Author: Elena Onieva Henrich")
    print("  Course: Advanced Programming 2025")
    print("  University: Université de Lausanne")
    print()
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Python: {PYTHON}")

    # Track overall statistics
    total_success = 0
    total_failed = 0
    total_skipped = 0

    # =========================================================================
    # STEP 1: DATA PREPROCESSING
    # =========================================================================
    print_header("STEP 1: DATA PREPROCESSING", '=', 70)

    n_success, n_failed, n_skipped = run_script_group(
        DATA_SCRIPTS, "Loading and preparing data"
    )
    total_success += n_success
    total_failed += n_failed
    total_skipped += n_skipped

    # =========================================================================
    # STEP 2: ARIMAX/SARIMAX MODELS
    # =========================================================================
    print_header("STEP 2: ARIMAX/SARIMAX MODELS", '=', 70)

    n_success, n_failed, n_skipped = run_script_group(
        ARIMAX_SCRIPTS, "Econometric time series models"
    )
    total_success += n_success
    total_failed += n_failed
    total_skipped += n_skipped

    # =========================================================================
    # STEP 3: RANDOM FOREST MODELS
    # =========================================================================
    print_header("STEP 3: RANDOM FOREST MODELS", '=', 70)

    n_success, n_failed, n_skipped = run_script_group(
        RF_SCRIPTS, "Machine learning - Random Forest"
    )
    total_success += n_success
    total_failed += n_failed
    total_skipped += n_skipped

    # =========================================================================
    # STEP 4: LASSO MODELS
    # =========================================================================
    print_header("STEP 4: LASSO MODELS", '=', 70)

    n_success, n_failed, n_skipped = run_script_group(
        LASSO_SCRIPTS, "Machine learning - LASSO"
    )
    total_success += n_success
    total_failed += n_failed
    total_skipped += n_skipped

    # =========================================================================
    # STEP 5: FINAL COMPARISON
    # =========================================================================
    print_header("STEP 5: FINAL COMPARISON", '=', 70)

    n_success, n_failed, n_skipped = run_script_group(
        EVALUATION_SCRIPTS, "Cross-model comparison"
    )
    total_success += n_success
    total_failed += n_failed
    total_skipped += n_skipped

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_elapsed = time.time() - total_start

    print_header("PIPELINE COMPLETE", '=', 70)
    print()
    print(f"  Total scripts run:     {total_success + total_failed}")
    print(f"  Successful:            {total_success}")
    print(f"  Failed:                {total_failed}")
    print(f"  Skipped (not found):   {total_skipped}")
    print(f"  Total time:            {total_elapsed:.1f} seconds")
    print()

    if total_failed == 0:
        print("  ✓ All scripts completed successfully!")
        print()
        print("  Results saved to:")
        print(f"    {PROJECT_ROOT / 'results'}")
        print()
        print("  Next steps:")
        print("    1. Review results in results/ directory")
        print("    2. Check comparison summary in results/comparison/")
        print("    3. Update project_report.pdf with findings")
    else:
        print(f"  ⚠ {total_failed} script(s) failed. Check output above.")

    print()
    print("=" * 70)

    return 0 if total_failed == 0 else 1


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    sys.exit(main())
