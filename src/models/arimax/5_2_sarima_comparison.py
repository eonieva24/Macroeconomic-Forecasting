"""
SARIMA vs ARIMAX Comparison.

This script compares SARIMA-X and ARIMAX forecasting results:
1. Load SARIMA results from 5_1_sarima_forecast.py
2. Load ARIMAX results from 4_forecast_arimax.py
3. Create side-by-side comparison plots (both on YoY scale)
4. Generate comparison tables and summary statistics
5. Determine which approach performs better per country

Comparison is done on YoY scale for fair comparison:
- ARIMAX operates directly on YoY-transformed CPI
- SARIMA forecasts are converted to YoY using: log(y_t) - log(y_{t-12})

Author: Elena Onieva Henrich
Date: January 2026
Course: Advanced Programming 2025 - Forecasting Project
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

PLOT_START_DATE = '2015-01-01'

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))

# SARIMA results
SARIMA_SUMMARY_PATH = os.path.join(
    PROJECT_ROOT, 'results', 'arimax_forecast',
    'forecasts_seasonal', 'forecast_accuracy_summary_seasonal.csv')
SARIMA_FORECASTS_DIR = os.path.join(
    PROJECT_ROOT, 'results', 'arimax_forecast', 'forecasts_seasonal')

# ARIMAX results
ARIMAX_SUMMARY_PATH = os.path.join(
    PROJECT_ROOT, 'results', 'arimax_forecast',
    'forecasts', 'forecast_accuracy_summary.csv')
ARIMAX_FORECASTS_DIR = os.path.join(
    PROJECT_ROOT, 'results', 'arimax_forecast', 'forecasts')

# Output
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, 'results', 'arimax_forecast', 'forecasts_seasonal')
PLOT_DIR = os.path.join(
    PROJECT_ROOT, 'results', 'arimax_forecast', 'plots', 'forecasts_seasonal')


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_sarima_summary():
    """
    Load SARIMA forecast summary.

    Returns
    -------
    pd.DataFrame or None
        SARIMA summary dataframe.
    """
    print("Loading SARIMA results...")

    if not os.path.exists(SARIMA_SUMMARY_PATH):
        print(f"  Error: SARIMA summary not found: {SARIMA_SUMMARY_PATH}")
        print("  Run 5_1_sarima_forecast.py first")
        return None

    df = pd.read_csv(SARIMA_SUMMARY_PATH)
    print(f"✓ Loaded SARIMA results for {len(df)} countries")

    return df


def load_arimax_summary():
    """
    Load ARIMAX forecast summary.

    Returns
    -------
    pd.DataFrame or None
        ARIMAX summary dataframe.
    """
    print("\nLoading ARIMAX results...")

    if not os.path.exists(ARIMAX_SUMMARY_PATH):
        print(f"  Warning: ARIMAX summary not found: {ARIMAX_SUMMARY_PATH}")
        return None

    df = pd.read_csv(ARIMAX_SUMMARY_PATH)

    # Scale RMSE and MAE by 100 for consistency with SARIMA
    if 'RMSE' in df.columns:
        df['RMSE'] = df['RMSE'] * 100
    if 'MAE' in df.columns:
        df['MAE'] = df['MAE'] * 100

    print(f"✓ Loaded ARIMAX results for {len(df)} countries")

    return df


def load_sarima_forecast(country):
    """
    Load SARIMA forecast results for a specific country.

    Parameters
    ----------
    country : str
        Country name.

    Returns
    -------
    pd.DataFrame or None
        SARIMA forecast dataframe.
    """
    country_filename = country.replace(' ', '_').lower()
    forecast_path = os.path.join(
        SARIMA_FORECASTS_DIR, f'{country_filename}_forecasts_seasonal.csv'
    )

    if os.path.exists(forecast_path):
        df = pd.read_csv(forecast_path)
        df['period'] = pd.to_datetime(df['period'])
        return df

    return None


def load_arimax_forecast(country):
    """
    Load ARIMAX forecast results for a specific country.

    Parameters
    ----------
    country : str
        Country name.

    Returns
    -------
    pd.DataFrame or None
        ARIMAX forecast dataframe.
    """
    country_filename = country.replace(' ', '_').lower()
    forecast_path = os.path.join(
        ARIMAX_FORECASTS_DIR, f'{country_filename}_forecasts.csv'
    )

    if os.path.exists(forecast_path):
        df = pd.read_csv(forecast_path)
        df['period'] = pd.to_datetime(df['period'])
        return df

    return None


def get_sarima_params(df_sarima, country):
    """
    Get SARIMA model parameters for a country.

    Parameters
    ----------
    df_sarima : pd.DataFrame
        SARIMA summary dataframe.
    country : str
        Country name.

    Returns
    -------
    dict or None
        Dictionary with model parameters.
    """
    row = df_sarima[df_sarima['Country'] == country]

    if len(row) == 0:
        return None

    row = row.iloc[0]

    return {
        'model': row.get('Model', 'N/A'),
        'rmse_yoy': row.get('RMSE_yoy', np.nan),
        'mae_yoy': row.get('MAE_yoy', np.nan),
        'coverage_yoy': row.get('CI_Coverage_yoy_%', np.nan)
    }


def get_arimax_params(df_arimax, country):
    """
    Get ARIMAX model parameters for a country.

    Parameters
    ----------
    df_arimax : pd.DataFrame
        ARIMAX summary dataframe.
    country : str
        Country name.

    Returns
    -------
    dict or None
        Dictionary with model parameters.
    """
    row = df_arimax[df_arimax['Country'] == country]

    if len(row) == 0:
        return None

    row = row.iloc[0]

    return {
        'model': row.get('Model', 'N/A'),
        'rmse': row.get('RMSE', np.nan),
        'mae': row.get('MAE', np.nan),
        'coverage': row.get('CI_Coverage_%', np.nan)
    }


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def create_comparison_table(df_sarima, df_arimax):
    """
    Create comparison table between SARIMA and ARIMAX results.

    Comparison uses YoY-scale metrics for fair comparison.

    Parameters
    ----------
    df_sarima : pd.DataFrame
        SARIMA results summary.
    df_arimax : pd.DataFrame
        ARIMAX results summary.

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    if df_arimax is None:
        print("\nNo ARIMAX results available for comparison")
        return None

    # Prepare SARIMA columns (YoY scale)
    df_compare = df_sarima[[
        'Country', 'Model', 'RMSE_yoy', 'MAE_yoy', 'CI_Coverage_yoy_%'
    ]].copy()
    df_compare = df_compare.rename(columns={
        'Model': 'SARIMA_Model',
        'RMSE_yoy': 'SARIMA_RMSE',
        'MAE_yoy': 'SARIMA_MAE',
        'CI_Coverage_yoy_%': 'SARIMA_Coverage'
    })

    # Prepare ARIMAX columns
    arimax_cols = ['Country']
    col_mapping = {}

    if 'Model' in df_arimax.columns:
        arimax_cols.append('Model')
        col_mapping['Model'] = 'ARIMAX_Model'
    if 'RMSE' in df_arimax.columns:
        arimax_cols.append('RMSE')
        col_mapping['RMSE'] = 'ARIMAX_RMSE'
    if 'MAE' in df_arimax.columns:
        arimax_cols.append('MAE')
        col_mapping['MAE'] = 'ARIMAX_MAE'
    if 'CI_Coverage_%' in df_arimax.columns:
        arimax_cols.append('CI_Coverage_%')
        col_mapping['CI_Coverage_%'] = 'ARIMAX_Coverage'

    df_arimax_subset = df_arimax[arimax_cols].copy()
    df_arimax_subset = df_arimax_subset.rename(columns=col_mapping)

    # Merge
    df_compare = df_compare.merge(df_arimax_subset, on='Country', how='outer')

    # Calculate difference and winner
    if ('ARIMAX_RMSE' in df_compare.columns and
            'SARIMA_RMSE' in df_compare.columns):
        df_compare['RMSE_Diff'] = (df_compare['SARIMA_RMSE'] -
                                   df_compare['ARIMAX_RMSE'])
        df_compare['Better_Model'] = df_compare['RMSE_Diff'].apply(
            lambda x: 'ARIMAX' if x > 0 else ('SARIMA' if x < 0 else 'Tie')
        )

    return df_compare


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_comparison(country, df_sarima_summary, df_arimax_summary, output_dir):
    """
    Plot side-by-side comparison of SARIMA vs ARIMAX forecasts.

    Both panels on YoY scale for fair comparison.
    Unified format with training data starting from 2015.

    Parameters
    ----------
    country : str
        Country name.
    df_sarima_summary : pd.DataFrame
        SARIMA summary for model parameters.
    df_arimax_summary : pd.DataFrame
        ARIMAX summary for model parameters.
    output_dir : str
        Directory to save plot.

    Returns
    -------
    bool
        True if plot created successfully.
    """
    # Load forecasts
    df_sarima = load_sarima_forecast(country)
    df_arimax = load_arimax_forecast(country)

    if df_sarima is None:
        print(f"  No SARIMA forecast data for {country}")
        return False

    if df_arimax is None:
        print(f"  No ARIMAX forecast data for {country}")
        return False

    # Get model parameters
    sarima_params = get_sarima_params(df_sarima_summary, country)
    arimax_params = get_arimax_params(df_arimax_summary, country)

    if sarima_params is None or arimax_params is None:
        print(f"  Missing parameters for {country}")
        return False

    # Filter SARIMA to valid YoY data
    df_sarima_yoy = df_sarima.dropna(subset=['actual_yoy', 'forecast_yoy'])

    if len(df_sarima_yoy) == 0:
        print(f"  No YoY-converted SARIMA data for {country}")
        return False

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    fig.suptitle(
        f'{country} - SARIMA vs ARIMAX Comparison',
        fontsize=16, fontweight='bold', y=0.98
    )

    # =========================================================================
    # TOP PANEL: SARIMA (YoY scale)
    # =========================================================================
    ax1 = axes[0]

    # Actual YoY values
    ax1.plot(
        df_sarima_yoy['period'], df_sarima_yoy['actual_yoy'],
        label='Actual', color='green', linewidth=2,
        marker='o', markersize=5
    )

    # SARIMA Forecasts
    ax1.plot(
        df_sarima_yoy['period'], df_sarima_yoy['forecast_yoy'],
        label='Forecast', color='red', linewidth=2,
        linestyle='--', marker='x', markersize=5
    )

    # Confidence interval
    ax1.fill_between(
        df_sarima_yoy['period'],
        df_sarima_yoy['lower_ci_yoy'],
        df_sarima_yoy['upper_ci_yoy'],
        color='red', alpha=0.2, label='95% CI'
    )

    # Zero line
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    # Formatting
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('YoY Log Difference', fontsize=11)
    ax1.set_title(
        f'(a) SARIMA-X: {sarima_params["model"]}',
        fontsize=12, loc='left', fontweight='bold'
    )
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Metrics box (top-left)
    sarima_textstr = (
        f"Model: SARIMA{sarima_params['model']}\n"
        f"RMSE: {sarima_params['rmse_yoy']:.4f}\n"
        f"MAE:  {sarima_params['mae_yoy']:.4f}\n"
        f"95% CI Coverage: {sarima_params['coverage_yoy']:.1f}%"
    )
    props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.3)
    ax1.text(
        0.02, 0.98, sarima_textstr, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, family='monospace'
    )

    # =========================================================================
    # BOTTOM PANEL: ARIMAX (YoY scale)
    # =========================================================================
    ax2 = axes[1]

    # Actual values
    ax2.plot(
        df_arimax['period'], df_arimax['actual'],
        label='Actual', color='green', linewidth=2,
        marker='o', markersize=5
    )

    # ARIMAX Forecasts
    ax2.plot(
        df_arimax['period'], df_arimax['forecast'],
        label='Forecast', color='blue', linewidth=2,
        linestyle='--', marker='s', markersize=5
    )

    # Confidence interval
    if 'lower_ci' in df_arimax.columns and 'upper_ci' in df_arimax.columns:
        ax2.fill_between(
            df_arimax['period'],
            df_arimax['lower_ci'],
            df_arimax['upper_ci'],
            color='blue', alpha=0.2, label='95% CI'
        )

    # Zero line
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    # Formatting
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('YoY Log Difference', fontsize=11)
    ax2.set_title(
        f'(b) ARIMAX: {arimax_params["model"]}',
        fontsize=12, loc='left', fontweight='bold'
    )
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Metrics box (top-left)
    arimax_textstr = (
        f"Model: ARIMAX{arimax_params['model']}\n"
        f"RMSE: {arimax_params['rmse']:.4f}\n"
        f"MAE:  {arimax_params['mae']:.4f}\n"
        f"95% CI Coverage: {arimax_params['coverage']:.1f}%"
    )
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.3)
    ax2.text(
        0.02, 0.98, arimax_textstr, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, family='monospace'
    )

    # =========================================================================
    # COMPARISON SUMMARY BOX (between panels)
    # =========================================================================
    sarima_rmse = sarima_params['rmse_yoy']
    arimax_rmse = arimax_params['rmse']

    if not np.isnan(arimax_rmse) and not np.isnan(sarima_rmse):
        if sarima_rmse < arimax_rmse:
            winner = "SARIMA"
            winner_color = 'lightcoral'
            diff = arimax_rmse - sarima_rmse
        elif arimax_rmse < sarima_rmse:
            winner = "ARIMAX"
            winner_color = 'lightblue'
            diff = sarima_rmse - arimax_rmse
        else:
            winner = "TIE"
            winner_color = 'lightgray'
            diff = 0

        comparison_text = (
            f"COMPARISON SUMMARY\n"
            f"{'─' * 25}\n"
            f"SARIMA RMSE: {sarima_rmse:.4f}\n"
            f"ARIMAX RMSE: {arimax_rmse:.4f}\n"
            f"{'─' * 25}\n"
            f"Winner: {winner}\n"
            f"Difference: {diff:.4f}"
        )

        props = dict(
            boxstyle='round', facecolor=winner_color,
            alpha=0.4, edgecolor='black'
        )
        fig.text(
            0.5, 0.48, comparison_text, fontsize=11, ha='center',
            va='center', bbox=props, family='monospace'
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.35)

    # Save
    country_filename = country.replace(' ', '_').lower()
    plot_path = os.path.join(
        output_dir, f'{country_filename}_sarima_vs_arimax_comparison.png'
    )
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Comparison plot saved: {plot_path}")
    return True


def plot_summary_comparison(df_comparison, output_dir):
    """
    Create summary bar chart comparing SARIMA vs ARIMAX across all countries.

    Parameters
    ----------
    df_comparison : pd.DataFrame
        Comparison dataframe with SARIMA_RMSE and ARIMAX_RMSE columns.
    output_dir : str
        Directory to save plot.
    """
    if df_comparison is None:
        return

    required_cols = ['SARIMA_RMSE', 'ARIMAX_RMSE']
    if not all(col in df_comparison.columns for col in required_cols):
        print("  Missing RMSE columns for summary comparison plot")
        return

    # Filter to countries with both values
    df_plot = df_comparison.dropna(
        subset=['SARIMA_RMSE', 'ARIMAX_RMSE']).copy()

    if len(df_plot) == 0:
        return

    # Sort by ARIMAX RMSE
    df_plot = df_plot.sort_values('ARIMAX_RMSE')

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Bar positions
    x = np.arange(len(df_plot))
    width = 0.35

    # Create bars
    bars1 = ax.bar(
        x - width/2, df_plot['ARIMAX_RMSE'], width,
        label='ARIMAX', color='steelblue', alpha=0.8
    )
    bars2 = ax.bar(
        x + width/2, df_plot['SARIMA_RMSE'], width,
        label='SARIMA', color='indianred', alpha=0.8
    )

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f'{height:.2f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points",
            ha='center', va='bottom', fontsize=8
        )

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f'{height:.2f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points",
            ha='center', va='bottom', fontsize=8
        )

    # Formatting
    ax.set_xlabel('Country', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSE (YoY Scale, x100)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Forecast Accuracy Comparison: ARIMAX vs SARIMA\n'
        '(Lower RMSE = Better)',
        fontsize=14, fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['Country'], rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Summary statistics (top-left)
    arimax_avg = df_plot['ARIMAX_RMSE'].mean()
    sarima_avg = df_plot['SARIMA_RMSE'].mean()
    arimax_wins = (df_plot['ARIMAX_RMSE'] < df_plot['SARIMA_RMSE']).sum()
    sarima_wins = (df_plot['SARIMA_RMSE'] < df_plot['ARIMAX_RMSE']).sum()

    summary_text = (
        f"Average RMSE:\n"
        f"  ARIMAX: {arimax_avg:.4f}\n"
        f"  SARIMA: {sarima_avg:.4f}\n\n"
        f"Wins (lower RMSE):\n"
        f"  ARIMAX: {arimax_wins}\n"
        f"  SARIMA: {sarima_wins}"
    )

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(
        0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, family='monospace'
    )

    plt.tight_layout()

    # Save
    plot_path = os.path.join(
        output_dir, 'all_countries_sarima_vs_arimax_comparison.png'
    )
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Summary comparison plot saved: {plot_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function: Compare SARIMA and ARIMAX forecasting results.

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "SARIMA vs ARIMAX COMPARISON")
    print("=" * 70)

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Load results
    print("\n" + "-" * 70)
    print("LOADING RESULTS")
    print("-" * 70)

    df_sarima = load_sarima_summary()
    df_arimax = load_arimax_summary()

    if df_sarima is None:
        print("\nCannot proceed without SARIMA results")
        print("Run 5_1_sarima_forecast.py first")
        return None

    # Create comparison table
    print("\n" + "-" * 70)
    print("CREATING COMPARISON TABLE")
    print("-" * 70)

    df_comparison = create_comparison_table(df_sarima, df_arimax)

    if df_comparison is not None:
        # Save comparison table
        comparison_path = os.path.join(
            OUTPUT_DIR, 'sarima_vs_arimax_comparison.csv')
        df_comparison.to_csv(comparison_path, index=False)
        print(f"✓ Comparison table saved: {comparison_path}")

        # Print comparison
        print("\n" + df_comparison.to_string(index=False))

        # Summary statistics
        if 'Better_Model' in df_comparison.columns:
            print("\n" + "-" * 70)
            print("COMPARISON SUMMARY")
            print("-" * 70)

            arimax_wins = (df_comparison['Better_Model'] == 'ARIMAX').sum()
            sarima_wins = (df_comparison['Better_Model'] == 'SARIMA').sum()
            ties = (df_comparison['Better_Model'] == 'Tie').sum()

            print(f"ARIMAX wins (lower RMSE): {arimax_wins} countries")
            print(f"SARIMA wins (lower RMSE): {sarima_wins} countries")
            print(f"Ties: {ties} countries")

            if 'ARIMAX_RMSE' in df_comparison.columns:
                df_valid = df_comparison.dropna(
                    subset=['ARIMAX_RMSE', 'SARIMA_RMSE']
                )
                arimax_avg = df_valid['ARIMAX_RMSE'].mean()
                sarima_avg = df_valid['SARIMA_RMSE'].mean()

                print(f"\nAverage RMSE (x100):")
                print(f"  ARIMAX: {arimax_avg:.4f}")
                print(f"  SARIMA: {sarima_avg:.4f}")

                if arimax_avg < sarima_avg:
                    print(f"\n→ ARIMAX performs better on average")
                else:
                    print(f"\n→ SARIMA performs better on average")

    # Generate comparison plots
    print("\n" + "-" * 70)
    print("GENERATING COMPARISON PLOTS")
    print("-" * 70)

    if df_arimax is not None:
        countries = df_sarima['Country'].unique()

        for country in countries:
            plot_comparison(country, df_sarima, df_arimax, PLOT_DIR)

        # Summary bar chart
        plot_summary_comparison(df_comparison, PLOT_DIR)

    print(f"\n{'=' * 70}")
    print("COMPARISON COMPLETE")
    print(f"{'=' * 70}")
    print(f"✓ Comparison table saved: {OUTPUT_DIR}")
    print(f"✓ Comparison plots saved: {PLOT_DIR}")
    print("=" * 70)

    return df_comparison


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    df_comparison = main()
