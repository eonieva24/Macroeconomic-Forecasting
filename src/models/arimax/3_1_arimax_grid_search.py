"""
ARIMAX Parameter Selection via Grid Search

Performs systematic grid search over ARIMA(p,d,q) parameters for ARIMAX models.
Since YoY transformation already ensures stationarity, d is fixed at 0.

Grid:
- p ∈ {0, 1, 2, 3}  (AR order)
- d = 0             (fixed, data is stationary)
- q ∈ {0, 1, 2, 3}  (MA order)

Selection: BIC (primary), AIC (secondary)
Validation: Ljung-Box test for residual autocorrelation

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
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings('ignore')

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ============================================================================
# CONFIGURATION
# ============================================================================

# Grid search parameters
P_VALUES = [0, 1, 2, 3]  # AR order
D_VALUE = 0  # Fixed: data is stationary after YoY transformation
Q_VALUES = [0, 1, 2, 3]  # MA order

# Significance level for Ljung-Box test
LJUNG_BOX_ALPHA = 0.05
LJUNG_BOX_LAGS = 10


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_country_data(data_dir, country):
    """
    Load ARIMAX-ready data for a country.

    Parameters
    ----------
    data_dir : str
        Directory containing ARIMAX CSV files.
    country : str
        Country name.

    Returns
    -------
    pd.DataFrame
        Data with columns: period, y_t, x1_t, x2_t
    """
    filename = f"{country.replace(' ', '_').lower()}_arimax.csv"
    filepath = os.path.join(data_dir, filename)

    assert os.path.exists(filepath), f"File not found: {filepath}"

    df = pd.read_csv(filepath)
    df['period'] = pd.to_datetime(df['period'])

    return df


def prepare_lagged_data(df, x1_lag, x2_lag):
    """
    Prepare data with specified lags for exogenous variables.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with y_t, x1_t, x2_t.
    x1_lag : int
        Lag for unemployment.
    x2_lag : int
        Lag for import prices.

    Returns
    -------
    pd.DataFrame
        Data with lagged exogenous variables.
    """
    df_model = df.copy()

    # Apply lags
    df_model['x1_used'] = df_model['x1_t'].shift(x1_lag) if x1_lag > 0 else df_model['x1_t']
    df_model['x2_used'] = df_model['x2_t'].shift(x2_lag) if x2_lag > 0 else df_model['x2_t']

    # Drop missing values
    df_model = df_model.dropna()

    return df_model


# ============================================================================
# MODEL FITTING AND EVALUATION
# ============================================================================

def fit_arimax_model(endog, exog, order):
    """
    Fit ARIMAX model with given order.

    Parameters
    ----------
    endog : pd.Series
        Endogenous variable (y_t).
    exog : pd.DataFrame
        Exogenous variables (x1_used, x2_used).
    order : tuple
        (p, d, q) ARIMA order.

    Returns
    -------
    dict
        Fitting results including AIC, BIC, diagnostics.
    """
    p, d, q = order

    try:
        # Fit model
        model = ARIMA(endog=endog, exog=exog, order=order)
        results = model.fit()

        # Ljung-Box test on residuals
        lb_result = acorr_ljungbox(results.resid, lags=[LJUNG_BOX_LAGS], return_df=True)
        lb_pvalue = lb_result['lb_pvalue'].values[0]
        lb_pass = lb_pvalue > LJUNG_BOX_ALPHA

        # Check coefficient significance
        pvalues = results.pvalues
        n_significant = (pvalues < 0.05).sum()
        n_total = len(pvalues)

        # Log-likelihood
        log_likelihood = results.llf

        return {
            'p': p,
            'd': d,
            'q': q,
            'order': f"({p},{d},{q})",
            'aic': results.aic,
            'bic': results.bic,
            'log_likelihood': log_likelihood,
            'lb_pvalue': lb_pvalue,
            'lb_pass': lb_pass,
            'n_significant': n_significant,
            'n_total': n_total,
            'converged': True,
            'error': None
        }

    except Exception as e:
        return {
            'p': p,
            'd': d,
            'q': q,
            'order': f"({p},{d},{q})",
            'aic': np.inf,
            'bic': np.inf,
            'log_likelihood': np.nan,
            'lb_pvalue': np.nan,
            'lb_pass': False,
            'n_significant': 0,
            'n_total': 0,
            'converged': False,
            'error': str(e)
        }


def grid_search_arimax(df_model, p_values=P_VALUES, d_value=D_VALUE, q_values=Q_VALUES):
    """
    Perform grid search over ARIMA parameters.

    Parameters
    ----------
    df_model : pd.DataFrame
        Prepared data with y_t, x1_used, x2_used.
    p_values : list
        AR orders to test.
    d_value : int
        Differencing order (fixed).
    q_values : list
        MA orders to test.

    Returns
    -------
    pd.DataFrame
        Results for all parameter combinations.
    """
    endog = df_model['y_t']
    exog = df_model[['x1_used', 'x2_used']]

    results = []

    for p, q in product(p_values, q_values):
        order = (p, d_value, q)
        result = fit_arimax_model(endog, exog, order)
        results.append(result)

    df_results = pd.DataFrame(results)

    return df_results


# ============================================================================
# MODEL SELECTION
# ============================================================================

def select_best_model(df_results, criterion='bic'):
    """
    Select best model based on information criterion.

    Parameters
    ----------
    df_results : pd.DataFrame
        Grid search results.
    criterion : str
        'aic' or 'bic' (default: 'bic').

    Returns
    -------
    dict
        Best model details.
    """
    # Filter to converged models only
    df_converged = df_results[df_results['converged']].copy()

    if len(df_converged) == 0:
        return None

    # Sort by criterion
    df_sorted = df_converged.sort_values(criterion)

    # Best by criterion
    best = df_sorted.iloc[0].to_dict()

    # Check if best model passes Ljung-Box
    if not best['lb_pass']:
        # Find best model that DOES pass Ljung-Box
        df_passing = df_converged[df_converged['lb_pass']]
        if len(df_passing) > 0:
            best_passing = df_passing.sort_values(criterion).iloc[0]
            best['alternative'] = best_passing.to_dict()
            best['warning'] = 'Best model fails Ljung-Box; alternative available'

    return best


# ============================================================================
# COUNTRY PROCESSING
# ============================================================================

def process_country(country, data_dir, lags_df):
    """
    Run full parameter selection for one country.

    Parameters
    ----------
    country : str
        Country name.
    data_dir : str
        Directory with ARIMAX data files.
    lags_df : pd.DataFrame
        Optimal lags from lag selection.

    Returns
    -------
    dict
        Complete results for this country.
    """
    print(f"\n{'=' * 70}")
    print(f"GRID SEARCH: {country}")
    print(f"{'=' * 70}")

    # Load data
    df = load_country_data(data_dir, country)
    print(f"Observations: {len(df)}")

    # Get optimal lags for this country
    country_lags = lags_df[lags_df['Country'] == country]

    if len(country_lags) == 0:
        print(f"Warning: No lag information for {country}, using lag=0")
        x1_lag, x2_lag = 0, 0
    else:
        x1_lag = int(country_lags['Best_x1_lag_BIC'].values[0])
        x2_lag = int(country_lags['Best_x2_lag_BIC'].values[0])

    print(f"Using lags: x1_lag={x1_lag}, x2_lag={x2_lag}")

    # Prepare data with lags
    df_model = prepare_lagged_data(df, x1_lag, x2_lag)
    print(f"Observations after lagging: {len(df_model)}")

    # Grid search
    print(f"\nTesting {len(P_VALUES) * len(Q_VALUES)} parameter combinations...")
    print(f"Grid: p ∈ {P_VALUES}, d = {D_VALUE}, q ∈ {Q_VALUES}")

    df_results = grid_search_arimax(df_model)

    # Count converged models
    n_converged = df_results['converged'].sum()
    n_total = len(df_results)
    print(f"Converged: {n_converged}/{n_total}")

    # Select best models
    best_bic = select_best_model(df_results, criterion='bic')
    best_aic = select_best_model(df_results, criterion='aic')

    # Print results table
    print(f"\n{'─' * 70}")
    print("GRID SEARCH RESULTS (sorted by BIC)")
    print(f"{'─' * 70}")

    display_cols = ['order', 'aic', 'bic', 'lb_pvalue', 'lb_pass', 'converged']
    df_display = df_results[df_results['converged']][display_cols].sort_values('bic')
    df_display['aic'] = df_display['aic'].round(2)
    df_display['bic'] = df_display['bic'].round(2)
    df_display['lb_pvalue'] = df_display['lb_pvalue'].round(4)
    print(df_display.to_string(index=False))

    # Print best model
    print(f"\n{'─' * 70}")
    print("BEST MODEL SELECTION")
    print(f"{'─' * 70}")

    if best_bic:
        print(f"\nBest by BIC: ARIMA{best_bic['order']}")
        print(f"  BIC: {best_bic['bic']:.2f}")
        print(f"  AIC: {best_bic['aic']:.2f}")
        print(f"  Ljung-Box p-value: {best_bic['lb_pvalue']:.4f}")
        print(f"  Ljung-Box: {'✓ PASS' if best_bic['lb_pass'] else '✗ FAIL'}")

        if 'warning' in best_bic:
            print(f"\n  ⚠ {best_bic['warning']}")
            alt = best_bic['alternative']
            print(f"  Alternative: ARIMA{alt['order']} (BIC={alt['bic']:.2f})")

    if best_aic and best_aic['order'] != best_bic['order']:
        print(f"\nBest by AIC: ARIMA{best_aic['order']}")
        print(f"  AIC: {best_aic['aic']:.2f}")
        print(f"  BIC: {best_aic['bic']:.2f}")
        print(f"  Note: Different from BIC selection")

    return {
        'country': country,
        'x1_lag': x1_lag,
        'x2_lag': x2_lag,
        'n_obs': len(df_model),
        'best_bic': best_bic,
        'best_aic': best_aic,
        'all_results': df_results
    }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function: Grid search parameter selection for all countries.

    Returns
    -------
    tuple
        (summary_df, all_results)
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "ARIMAX PARAMETER GRID SEARCH")
    print("=" * 70)
    print(f"\nGrid: p ∈ {P_VALUES}, d = {D_VALUE} (fixed), q ∈ {Q_VALUES}")
    print(f"Total combinations per country: {len(P_VALUES) * len(Q_VALUES)}")
    print(f"\nSelection criterion: BIC (primary), AIC (secondary)")
    print(f"Diagnostic: Ljung-Box test (α = {LJUNG_BOX_ALPHA})")
    print(f"\nNote: d=0 because YoY transformation already ensures stationarity")

    # Paths
    DATA_DIR = os.path.join(PROJECT_ROOT, 'results', 'arimax_forecast', 'arimax')
    LAGS_PATH = os.path.join(PROJECT_ROOT, 'results', 'arimax_forecast', 'lag_selection', 'optimal_lags_summary.csv')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'arimax_forecast', 'parameters')

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load optimal lags
    print("\nLoading optimal lags...")
    assert os.path.exists(LAGS_PATH), f"File not found: {LAGS_PATH}"
    lags_df = pd.read_csv(LAGS_PATH)
    print(f"✓ Loaded lags for {len(lags_df)} countries")

    # Get list of countries
    countries = lags_df['Country'].tolist()

    # Process each country
    all_results = {}
    summary_data = []

    for country in countries:
        result = process_country(country, DATA_DIR, lags_df)
        all_results[country] = result

        # Extract summary
        best = result['best_bic']
        if best:
            summary_data.append({
                'Country': country,
                'p': best['p'],
                'd': best['d'],
                'q': best['q'],
                'Order': best['order'],
                'AIC': best['aic'],
                'BIC': best['bic'],
                'LB_pvalue': best['lb_pvalue'],
                'LB_Pass': best['lb_pass'],
                'x1_lag': result['x1_lag'],
                'x2_lag': result['x2_lag'],
                'N_obs': result['n_obs']
            })

    # Create summary dataframe
    df_summary = pd.DataFrame(summary_data)

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, 'arimax_grid_search_results.csv')
    df_summary.to_csv(summary_path, index=False)

    # Save detailed results per country
    for country, result in all_results.items():
        country_filename = country.replace(' ', '_').lower()
        detail_path = os.path.join(OUTPUT_DIR, f'{country_filename}_grid_search.csv')
        result['all_results'].to_csv(detail_path, index=False)

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: OPTIMAL ARIMA PARAMETERS")
    print("=" * 70)

    print("\n" + df_summary.to_string(index=False))

    # Parameter distribution
    print("\n" + "-" * 70)
    print("PARAMETER DISTRIBUTION")
    print("-" * 70)

    print("\nAR order (p):")
    for p_val in sorted(df_summary['p'].unique()):
        count = (df_summary['p'] == p_val).sum()
        countries_with_p = df_summary[df_summary['p'] == p_val]['Country'].tolist()
        print(f"  p={p_val}: {count} countries - {', '.join(countries_with_p)}")

    print("\nMA order (q):")
    for q_val in sorted(df_summary['q'].unique()):
        count = (df_summary['q'] == q_val).sum()
        countries_with_q = df_summary[df_summary['q'] == q_val]['Country'].tolist()
        print(f"  q={q_val}: {count} countries - {', '.join(countries_with_q)}")

    # Ljung-Box results
    print("\n" + "-" * 70)
    print("DIAGNOSTIC CHECK")
    print("-" * 70)
    lb_pass_count = df_summary['LB_Pass'].sum()
    print(f"Ljung-Box test: {lb_pass_count}/{len(df_summary)} countries pass")

    if lb_pass_count < len(df_summary):
        failing = df_summary[~df_summary['LB_Pass']]['Country'].tolist()
        print(f"⚠ Residual autocorrelation in: {', '.join(failing)}")
        print("  Consider: higher p/q, or check for structural breaks")

    print(f"\n{'=' * 70}")
    print("GRID SEARCH COMPLETE")
    print(f"{'=' * 70}")
    print(f"✓ Summary saved: {summary_path}")
    print(f"✓ Detailed results saved: {OUTPUT_DIR}")
    print("=" * 70)

    return df_summary, all_results


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    df_summary, all_results = main()