"""
Select Optimal Lags for ARIMAX Exogenous Variables

This module tests different lag combinations for exogenous variables (Unemployment
and Import Prices) in ARIMAX models and selects optimal lags based on BIC.

We test contemporaneous and lagged effects of:
- x1_t: Unemployment Rate (lags 0-6)
- x2_t: Import Prices (lags 0-6)

For each country, fits ARIMA(1,0,0) with different lag combinations and selects
the combination that minimizes information criteria. The final ARIMA(p,d,q)
parameters will be determined by grid search in the next script.

Rationale: Exogenous variables may affect CPI with a delay. For example, changes
in unemployment or import costs may take several months to fully transmit to
consumer prices. Testing multiple lags allows us to identify the optimal timing
of these effects.

Note: Using fixed ARIMA(1,0,0) for lag selection is methodologically sound because:
1. The lag selection is not highly sensitive to ARIMA order
2. Final ARIMA parameters are determined by grid search in script 3
3. d=0 is correct since YoY transformation ensures stationarity

Author: Elena Onieva Henrich
Date: November 2025
Course: Advanced Programming 2025 - Forecasting Project
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import warnings
from itertools import product
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ============================================================================
# CONFIGURATION
# ============================================================================

# Fixed ARIMA order for lag selection
# Final parameters determined by grid search in script 3
ARIMA_ORDER_FOR_LAG_SELECTION = (1, 0, 0)

# Maximum lag to test for exogenous variables
MAX_LAG = 6


# ============================================================================
# LAG CREATION FUNCTIONS
# ============================================================================

def create_lagged_variables(df, max_lag=MAX_LAG):
    """
    Create lagged versions of exogenous variables.

    For each exogenous variable (x1, x2), creates lags from 0 to max_lag.
    Lag 0 = contemporaneous (no lag)
    Lag k = value from k periods ago

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns: period, y_t, x1_t, x2_t
    max_lag : int
        Maximum number of lags to create (default: 6).

    Returns
    -------
    pd.DataFrame
        Dataframe with original and lagged variables.
    """
    df_lagged = df.copy()

    # Create lags for x1_t (Unemployment)
    for lag in range(1, max_lag + 1):
        df_lagged[f'x1_lag{lag}'] = df_lagged['x1_t'].shift(lag)

    # Create lags for x2_t (Import Prices)
    for lag in range(1, max_lag + 1):
        df_lagged[f'x2_lag{lag}'] = df_lagged['x2_t'].shift(lag)

    return df_lagged


def get_exog_columns(x1_lag, x2_lag):
    """
    Get column names for specified lag combination.

    Parameters
    ----------
    x1_lag : int
        Lag for x1 (0 = contemporaneous).
    x2_lag : int
        Lag for x2 (0 = contemporaneous).

    Returns
    -------
    list
        List of column names to use as exogenous variables.
    """
    exog_cols = []

    # Add x1 with appropriate lag
    if x1_lag == 0:
        exog_cols.append('x1_t')
    else:
        exog_cols.append(f'x1_lag{x1_lag}')

    # Add x2 with appropriate lag
    if x2_lag == 0:
        exog_cols.append('x2_t')
    else:
        exog_cols.append(f'x2_lag{x2_lag}')

    return exog_cols


# ============================================================================
# MODEL FITTING FUNCTIONS
# ============================================================================

def fit_arimax_with_lags(df, x1_lag, x2_lag, arima_order):
    """
    Fit ARIMAX model with specified lag combination.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with lagged variables.
    x1_lag : int
        Lag for x1 (0-6).
    x2_lag : int
        Lag for x2 (0-6).
    arima_order : tuple
        ARIMA(p,d,q) order.

    Returns
    -------
    dict
        Results containing AIC, BIC, and convergence status.
    """
    try:
        # Get exogenous variable columns for this lag combination
        exog_cols = get_exog_columns(x1_lag, x2_lag)

        # Prepare data - drop rows with any missing values
        df_model = df[['y_t'] + exog_cols].dropna()

        # Check if enough data remains
        if len(df_model) < 30:
            return {
                'x1_lag': x1_lag,
                'x2_lag': x2_lag,
                'aic': np.nan,
                'bic': np.nan,
                'converged': False,
                'n_obs': len(df_model),
                'error': 'Insufficient data'
            }

        # Prepare endogenous and exogenous variables
        endog = df_model['y_t']
        exog = df_model[exog_cols]

        # Fit ARIMAX model
        model = ARIMA(endog=endog, exog=exog, order=arima_order)
        results = model.fit()

        return {
            'x1_lag': x1_lag,
            'x2_lag': x2_lag,
            'aic': results.aic,
            'bic': results.bic,
            'converged': True,
            'n_obs': len(df_model),
            'error': None
        }

    except Exception as e:
        return {
            'x1_lag': x1_lag,
            'x2_lag': x2_lag,
            'aic': np.nan,
            'bic': np.nan,
            'converged': False,
            'n_obs': 0,
            'error': str(e)
        }


def test_all_lag_combinations(df, max_lag=MAX_LAG, arima_order=ARIMA_ORDER_FOR_LAG_SELECTION):
    """
    Test all combinations of lags for x1 and x2.

    Tests all combinations from (0,0) to (max_lag, max_lag).
    Total combinations: (max_lag + 1)^2 = 49 for max_lag=6

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with y_t, x1_t, x2_t.
    max_lag : int
        Maximum lag to test (default: 6).
    arima_order : tuple
        ARIMA(p,d,q) order for testing.

    Returns
    -------
    pd.DataFrame
        Results for all lag combinations.
    """
    print(f"  Testing {(max_lag + 1) ** 2} lag combinations...")

    # Create lagged variables
    df_lagged = create_lagged_variables(df, max_lag=max_lag)

    # Test all combinations
    results = []

    for x1_lag, x2_lag in product(range(max_lag + 1), range(max_lag + 1)):
        result = fit_arimax_with_lags(df_lagged, x1_lag, x2_lag, arima_order)
        results.append(result)

    df_results = pd.DataFrame(results)

    # Sort by BIC (ascending) - BIC is preferred for model selection
    df_results = df_results.sort_values('bic').reset_index(drop=True)

    return df_results


# ============================================================================
# COUNTRY PROCESSING FUNCTIONS
# ============================================================================

def process_country(filepath):
    """
    Process one country: test all lag combinations and select optimal.

    Uses fixed ARIMA(1,0,0) order for lag selection. Final ARIMA parameters
    will be determined by grid search in the next script.

    Parameters
    ----------
    filepath : str
        Path to country's ARIMAX ready data.

    Returns
    -------
    tuple
        (country_name, best_result_aic, best_result_bic, all_results)
    """
    # Extract country name from filename
    filename = os.path.basename(filepath)
    country = filename.replace('_arimax.csv', '').replace('_', ' ').title()

    print(f"\n{'=' * 70}")
    print(f"PROCESSING: {country}")
    print(f"{'=' * 70}")

    # Use fixed ARIMA order for lag selection
    arima_order = ARIMA_ORDER_FOR_LAG_SELECTION
    print(f"  ARIMA order for lag selection: {arima_order}")
    print(f"  (Final ARIMA parameters determined by grid search in script 3)")

    # Load data
    df = pd.read_csv(filepath)
    df['period'] = pd.to_datetime(df['period'])

    print(f"  Loaded: {len(df)} observations")
    print(f"  Date range: {df['period'].min().strftime('%Y-%m')} to {df['period'].max().strftime('%Y-%m')}")

    # Test all lag combinations
    df_results = test_all_lag_combinations(df, max_lag=MAX_LAG, arima_order=arima_order)

    # Get best result by BIC (primary criterion)
    df_results_bic = df_results[df_results['converged']].sort_values('bic')
    best_bic = df_results_bic.iloc[0] if len(df_results_bic) > 0 else None

    # Get best result by AIC (secondary criterion)
    df_results_aic = df_results[df_results['converged']].sort_values('aic')
    best_aic = df_results_aic.iloc[0] if len(df_results_aic) > 0 else None

    if best_bic is not None:
        print(f"\n  BEST BY BIC:")
        print(f"    x1_lag = {int(best_bic['x1_lag'])}, x2_lag = {int(best_bic['x2_lag'])}")
        print(f"    AIC = {best_bic['aic']:.2f}, BIC = {best_bic['bic']:.2f}")
        print(f"    Observations: {int(best_bic['n_obs'])}")

    if best_aic is not None and (best_aic['x1_lag'] != best_bic['x1_lag'] or best_aic['x2_lag'] != best_bic['x2_lag']):
        print(f"\n  BEST BY AIC (different from BIC):")
        print(f"    x1_lag = {int(best_aic['x1_lag'])}, x2_lag = {int(best_aic['x2_lag'])}")
        print(f"    AIC = {best_aic['aic']:.2f}, BIC = {best_aic['bic']:.2f}")

    # Check convergence rate
    converged_count = df_results['converged'].sum()
    total_count = len(df_results)
    print(f"\n  Convergence: {converged_count}/{total_count} models ({100 * converged_count / total_count:.1f}%)")

    return country, best_aic, best_bic, df_results


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to select optimal lags for all countries.

    Steps:
    1. Load ARIMAX-ready data per country
    2. Test all lag combinations (0-6) for x1 and x2
    3. Select optimal lags based on BIC (primary) and AIC (secondary)
    4. Save results

    Returns
    -------
    pd.DataFrame
        Summary of optimal lags for all countries.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "ARIMAX LAG SELECTION")
    print("=" * 70)
    print(f"\nUsing fixed ARIMA{ARIMA_ORDER_FOR_LAG_SELECTION} for lag selection")
    print("(Final ARIMA parameters determined by grid search in script 3)")
    print(f"\nTesting lags 0-{MAX_LAG} for x1_t (Unemployment) and x2_t (Import Prices)")
    print(f"Total combinations per country: {(MAX_LAG + 1) ** 2}")
    print("Selection criterion: BIC (primary), AIC (secondary)")

    # Define paths
    DATA_DIR = os.path.join(PROJECT_ROOT, 'results', 'arimax_forecast', 'arimax')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'arimax_forecast', 'lag_selection')

    # Find all ARIMAX ready files
    assert os.path.exists(DATA_DIR), f"Directory not found: {DATA_DIR}"
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('_arimax.csv')]

    # Exclude summary file
    files = [f for f in files if 'summary' not in f.lower()]

    print(f"\n{len(files)} countries found:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    # Process each country
    best_lags_summary = []
    all_results_dict = {}

    for filepath in sorted(files):
        country, best_aic, best_bic, df_results = process_country(filepath)

        # Store summary (use BIC as primary criterion)
        if best_bic is not None:
            best_lags_summary.append({
                'Country': country,
                'Best_x1_lag_BIC': int(best_bic['x1_lag']),
                'Best_x2_lag_BIC': int(best_bic['x2_lag']),
                'BIC': best_bic['bic'],
                'Best_x1_lag_AIC': int(best_aic['x1_lag']) if best_aic is not None else None,
                'Best_x2_lag_AIC': int(best_aic['x2_lag']) if best_aic is not None else None,
                'AIC': best_aic['aic'] if best_aic is not None else None,
                'N_observations': int(best_bic['n_obs'])
            })

        # Store full results
        df_results['Country'] = country
        all_results_dict[country] = df_results

    # Create summary dataframe
    df_summary = pd.DataFrame(best_lags_summary)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, 'optimal_lags_summary.csv')
    df_summary.to_csv(summary_path, index=False)

    # Save detailed results for each country
    for country, df_results in all_results_dict.items():
        country_filename = country.replace(' ', '_').lower()
        results_path = os.path.join(OUTPUT_DIR, f'{country_filename}_all_lags.csv')
        df_results.to_csv(results_path, index=False)

    # Print final summary
    print("\n" + "=" * 70)
    print("OPTIMAL LAG SELECTION SUMMARY")
    print("=" * 70)
    print("\n" + df_summary.to_string(index=False))

    # Lag distribution analysis
    print("\n" + "-" * 70)
    print("LAG DISTRIBUTION (BIC-selected)")
    print("-" * 70)

    print("\nx1 (Unemployment) lag distribution:")
    x1_counts = df_summary['Best_x1_lag_BIC'].value_counts().sort_index()
    for lag, count in x1_counts.items():
        countries = df_summary[df_summary['Best_x1_lag_BIC'] == lag]['Country'].tolist()
        print(f"  Lag {lag}: {count} countries - {', '.join(countries)}")

    print("\nx2 (Import Prices) lag distribution:")
    x2_counts = df_summary['Best_x2_lag_BIC'].value_counts().sort_index()
    for lag, count in x2_counts.items():
        countries = df_summary[df_summary['Best_x2_lag_BIC'] == lag]['Country'].tolist()
        print(f"  Lag {lag}: {count} countries - {', '.join(countries)}")

    print(f"\n{'=' * 70}")
    print("LAG SELECTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"✓ Summary saved to: {summary_path}")
    print(f"✓ Detailed results saved to: {OUTPUT_DIR}")
    print(f"✓ {len(files)} countries processed")
    print("=" * 70)

    return df_summary


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    df_summary = main()
