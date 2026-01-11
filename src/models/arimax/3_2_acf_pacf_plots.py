"""
ACF and PACF Visualization for ARIMA Parameter Verification

Creates ACF and PACF plots for each country to visually verify the
lag orders selected by grid search.

Author: Elena Onieva Henrich
Date: November 2025
Course: Advanced Programming 2025 - Forecasting Project
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_acf_pacf_for_country(series, country_name, selected_p, selected_q, save_dir):
    """
    Create ACF and PACF plots for a country with grid search results annotated.

    Parameters
    ----------
    series : pd.Series
        YoY transformed time series.
    country_name : str
        Country name.
    selected_p : int
        AR order selected by grid search.
    selected_q : int
        MA order selected by grid search.
    save_dir : str
        Directory to save plots.
    """
    # Remove NaN values
    series_clean = series.dropna()

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f'{country_name} - ACF and PACF Analysis', fontsize=14, fontweight='bold')

    # PACF plot (for AR order p)
    plot_pacf(series_clean, lags=12, ax=axes[0])
    axes[0].set_title(f'PACF - Grid search selected p={selected_p}')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # ACF plot (for MA order q)
    plot_acf(series_clean, lags=12, ax=axes[1])
    axes[1].set_title(f'ACF - Grid search selected q={selected_q}')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    # Save plot
    filename = f'{country_name.replace(" ", "_").lower()}_acf_pacf.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved plot for {country_name}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Generate ACF and PACF plots for all countries.

    Steps:
    1. Load YoY CPI data from country subsets
    2. Load grid search results for p, q parameters
    3. Generate ACF/PACF plots for each country
    4. Save plots to results directory
    """
    print("\n" + "=" * 70)
    print("ACF AND PACF VISUALIZATION")
    print("=" * 70)

    # Define paths
    COUNTRY_INPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'yoy_cpi_country_subset')
    PARAMS_PATH = os.path.join(PROJECT_ROOT, 'results', 'arimax_forecast', 'parameters', 'arimax_grid_search_results.csv')
    SAVE_DIR = os.path.join(PROJECT_ROOT, 'results', 'arimax_forecast', 'plots', 'acf_pacf')

    # Load data from individual country files
    print("\nLoading data from country subsets...")

    assert os.path.exists(COUNTRY_INPUT_DIR), f"Directory not found: {COUNTRY_INPUT_DIR}"
    country_files = [f for f in os.listdir(COUNTRY_INPUT_DIR) if f.endswith('_cpi_yoy.csv')]

    # Load each country file and merge into single dataframe
    df = None
    for country_file in sorted(country_files):
        country_path = os.path.join(COUNTRY_INPUT_DIR, country_file)
        country_df = pd.read_csv(country_path)
        country_df['period'] = pd.to_datetime(country_df['period'])

        # Extract country name from filename
        country_name = country_file.replace('_cpi_yoy.csv', '').replace('_', ' ').title()

        # Rename column to include country name
        country_df = country_df.rename(columns={'cpi_yoy': f'{country_name} – CPI'})

        if df is None:
            df = country_df
        else:
            df = df.merge(country_df, on='period', how='outer')

    df = df.sort_values('period').reset_index(drop=True)
    print(f"✓ Loaded {len(country_files)} countries")

    # Load grid search results
    assert os.path.exists(PARAMS_PATH), f"File not found: {PARAMS_PATH}"
    params = pd.read_csv(PARAMS_PATH)
    print(f"✓ Loaded parameters for {len(params)} countries")

    # Create output directory
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\nGenerating ACF/PACF plots for {len(params)} countries...")
    print(f"Saving to: {SAVE_DIR}\n")

    # Generate plots for each country
    for _, row in params.iterrows():
        country = row['Country']
        selected_p = row['p']
        selected_q = row['q']

        # Find matching column in dataframe
        matching_cols = [c for c in df.columns if country in c]
        if len(matching_cols) == 0:
            print(f"⚠ No data found for {country}, skipping")
            continue

        col = matching_cols[0]
        series = df[col]

        # Generate plot
        plot_acf_pacf_for_country(series, country, selected_p, selected_q, SAVE_DIR)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nPlots saved to: {SAVE_DIR}")
    print("\nVisual Interpretation Guide:")
    print("  PACF: Significant lags suggest AR order (p)")
    print("  ACF:  Significant lags suggest MA order (q)")
    print("  → Compare with grid search selections in arimax_grid_search_results.csv")
    print("=" * 70)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
