"""
ARIMA Data Preparation Module - YoY Transformation and Stationarity Testing

This module prepares CPI data for ARIMA modeling by:
1. Calculating Year-over-Year (YoY) log differences to deseasonalize the data
2. Performing comprehensive stationarity tests (ADF, PP) on each country
3. Generating diagnostic plots and reports to guide ARIMA model specification

The YoY transformation captures annual growth rates while neutralizing recurring
seasonal effects, which is essential for accurate time series forecasting.

Formula: YoY_t = log(y_t) - log(y_{t-12})

Reference:
- https://www.numberanalytics.com/blog/mastering-stationarity-tests
- https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/

Author: Elena Onieva Henrich
Date: November 2025
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
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import PhillipsPerron

warnings.filterwarnings('ignore')

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# DESEASONALIZATION FUNCTIONS
# ============================================================================

def calculate_yoy_log_difference(df):
    """
    Calculate Year-over-Year (YoY) log differences to deseasonalize data.

    The YoY transformation compares each observation to its value 12 months
    earlier, capturing annual growth while neutralizing recurring seasonal
    effects. Using log differences provides better statistical properties:
    - Stabilizes variance
    - Interpretable as approximate percentage changes
    - Symmetric treatment of increases and decreases

    Formula: YoY_t = log(y_t) - log(y_{t-12})

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'period' column and country columns containing CPI values.

    Returns
    -------
    pd.DataFrame
        Transformed dataframe with YoY log differences for each country.

    Raises
    ------
    AssertionError
        If dataframe structure is invalid or transformation fails.

    Notes
    -----
    First 12 months will have NaN values (no year-ago comparison available).
    """
    # Validate input dataframe
    assert 'period' in df.columns, "Dataframe must contain 'period' column"
    assert df.shape[1] > 1, "Dataframe must contain at least one country column"

    print("\n" + "=" * 70)
    print("CALCULATING YEAR-OVER-YEAR LOG DIFFERENCES")
    print("=" * 70)
    print("Formula: YoY_t = log(y_t) - log(y_{t-12})")

    # Create copy to avoid modifying original data
    df_transformed = df.copy()

    # Store period column separately
    period_col = df_transformed['period']

    # Get list of country columns
    country_cols = [col for col in df_transformed.columns if col != 'period']

    print(f"\nProcessing {len(country_cols)} countries...")

    # Calculate YoY log difference for each country
    for col in country_cols:
        # Extract country name
        country = col.split(' – ')[0]

        # Get original values
        original_values = df_transformed[col].values

        # Check that all values are positive (required for log transformation)
        valid_values = original_values[~np.isnan(original_values)]
        assert (valid_values > 0).all(), \
            f"{country}: Cannot take log of non-positive values"

        # Calculate log values
        log_values = np.log(original_values)

        # Calculate YoY difference: log(y_t) - log(y_{t-12})
        yoy_diff = log_values - np.roll(log_values, 12)

        # Set first 12 values to NaN (no year-ago comparison available)
        yoy_diff[:12] = np.nan

        # Store transformed values
        df_transformed[col] = yoy_diff

    # Validate transformation
    assert df_transformed.shape == df.shape, "Transformation changed dataframe shape"
    assert (df_transformed['period'] == period_col).all(), "Period column was modified"

    print(f"\n✓ YoY log differences calculated successfully")
    print(f"✓ Valid observations per country: {len(df_transformed) - 12}")
    print(f"✓ Missing observations (first 12 months): 12 per country")

    return df_transformed


# ============================================================================
# STATIONARITY TEST FUNCTIONS
# ============================================================================

def perform_adf_test(series, country_name, significance_level=0.05):
    """
    Perform Augmented Dickey-Fuller (ADF) test for stationarity.

    The ADF test checks the null hypothesis that a unit root is present
    (i.e., the series is non-stationary). A p-value < significance level
    provides evidence to reject the null and conclude stationarity.

    Parameters
    ----------
    series : pd.Series
        Time series data to test.
    country_name : str
        Name of country (for reporting).
    significance_level : float, optional
        Significance level for hypothesis test (default: 0.05).

    Returns
    -------
    dict
        Dictionary containing test results.
    """
    # Remove NaN values
    series_clean = series.dropna()

    # Ensure sufficient observations
    assert len(series_clean) >= 12, \
        f"{country_name}: Insufficient observations for ADF test"

    # Perform ADF test
    adf_result = adfuller(series_clean, autolag='AIC', regression='c')

    # Extract results
    test_statistic = adf_result[0]
    p_value = adf_result[1]
    n_lags = adf_result[2]
    n_obs = adf_result[3]
    critical_values = adf_result[4]

    # Determine stationarity
    is_stationary = p_value < significance_level

    if is_stationary:
        conclusion = f"STATIONARY (p={p_value:.4f} < {significance_level})"
    else:
        conclusion = f"NON-STATIONARY (p={p_value:.4f} >= {significance_level})"

    return {
        'country': country_name,
        'test_statistic': test_statistic,
        'p_value': p_value,
        'n_lags': n_lags,
        'n_obs': n_obs,
        'critical_values': critical_values,
        'is_stationary': is_stationary,
        'conclusion': conclusion
    }


def perform_pp_test(series, country_name, significance_level=0.05):
    """
    Perform Phillips-Perron (PP) test for stationarity.

    The PP test is similar to ADF but uses non-parametric corrections for
    serial correlation and heteroskedasticity.

    Parameters
    ----------
    series : pd.Series
        Time series data to test.
    country_name : str
        Name of country (for reporting).
    significance_level : float, optional
        Significance level for hypothesis test (default: 0.05).

    Returns
    -------
    dict
        Dictionary containing test results.
    """
    # Remove NaN values
    series_clean = series.dropna().values

    # Ensure sufficient observations
    assert len(series_clean) >= 12, \
        f"{country_name}: Insufficient observations for PP test"

    # Perform PP test
    pp = PhillipsPerron(series_clean, lags=12, trend='c')

    # Extract results
    test_statistic = pp.stat
    p_value = pp.pvalue
    critical_values = pp.critical_values

    # Determine stationarity
    is_stationary = p_value < significance_level

    if is_stationary:
        conclusion = f"STATIONARY (p={p_value:.4f} < {significance_level})"
    else:
        conclusion = f"NON-STATIONARY (p={p_value:.4f} >= {significance_level})"

    return {
        'country': country_name,
        'test_statistic': test_statistic,
        'p_value': p_value,
        'critical_values': critical_values,
        'is_stationary': is_stationary,
        'conclusion': conclusion
    }


def test_country_stationarity(series, country_name, significance_level=0.05):
    """
    Perform both stationarity tests for a single country.

    Parameters
    ----------
    series : pd.Series
        Time series data to test.
    country_name : str
        Name of country for identification.
    significance_level : float, optional
        Significance level for all tests (default: 0.05).

    Returns
    -------
    dict
        Dictionary containing results from both tests plus consensus.
    """
    print(f"\n{'=' * 70}")
    print(f"STATIONARITY TESTS: {country_name}")
    print(f"{'=' * 70}")

    # Perform both tests
    adf_results = perform_adf_test(series, country_name, significance_level)
    pp_results = perform_pp_test(series, country_name, significance_level)

    # Print results
    print(f"\n1. ADF Test (Augmented Dickey-Fuller):")
    print(f"   Test Statistic: {adf_results['test_statistic']:.6f}")
    print(f"   P-Value: {adf_results['p_value']:.6f}")
    print(f"   Lags Used: {adf_results['n_lags']}")
    print(f"   -> {adf_results['conclusion']}")

    print(f"\n2. PP Test (Phillips-Perron):")
    print(f"   Test Statistic: {pp_results['test_statistic']:.6f}")
    print(f"   P-Value: {pp_results['p_value']:.6f}")
    print(f"   -> {pp_results['conclusion']}")

    # Determine consensus
    stationary_count = sum([
        adf_results['is_stationary'],
        pp_results['is_stationary']
    ])

    if stationary_count == 2:
        consensus = "Both tests agree: STATIONARY"
        consensus_category = "strong_stationary"
    elif stationary_count == 0:
        consensus = "Both tests agree: NON-STATIONARY"
        consensus_category = "strong_nonstationary"
    else:
        consensus = "Tests disagree: mixed result"
        consensus_category = "mixed"

    print(f"\n{'=' * 70}")
    print(f"CONSENSUS: {consensus}")
    print(f"{'=' * 70}")

    return {
        'country': country_name,
        'adf': adf_results,
        'pp': pp_results,
        'consensus': consensus,
        'consensus_category': consensus_category,
        'stationary_count': stationary_count
    }


def test_all_countries_stationarity(df):
    """
    Perform stationarity tests on all countries in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'period' column and country columns (YoY transformed).

    Returns
    -------
    pd.DataFrame
        Summary dataframe with test results for all countries.
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE STATIONARITY TESTING - ALL COUNTRIES")
    print("=" * 70)

    # Get country columns
    country_cols = [col for col in df.columns if col != 'period']

    print(f"\nTesting {len(country_cols)} countries...")
    print(f"Significance level: 0.05 (5%)")
    print(f"Tests performed: ADF, PP")

    # Initialize results storage
    all_results = []

    # Test each country
    for col in country_cols:
        country = col.split(' – ')[0]
        series = df[col]

        results = test_country_stationarity(series, country)

        all_results.append({
            'Country': country,
            'ADF_Statistic': results['adf']['test_statistic'],
            'ADF_PValue': results['adf']['p_value'],
            'ADF_Stationary': results['adf']['is_stationary'],
            'PP_Statistic': results['pp']['test_statistic'],
            'PP_PValue': results['pp']['p_value'],
            'PP_Stationary': results['pp']['is_stationary'],
            'Stationary_Count': results['stationary_count'],
            'Consensus': results['consensus'],
            'Consensus_Category': results['consensus_category']
        })

    # Create summary dataframe
    results_df = pd.DataFrame(all_results)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS ALL COUNTRIES")
    print("=" * 70)
    print(f"\nTotal countries tested: {len(results_df)}")
    print(f"\nStationarity by test:")
    print(f"  ADF: {results_df['ADF_Stationary'].sum()}/{len(results_df)} stationary")
    print(f"  PP:  {results_df['PP_Stationary'].sum()}/{len(results_df)} stationary")

    return results_df


def create_stationarity_summary_table(results_df):
    """
    Create a formatted summary table of stationarity test results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe from test_all_countries_stationarity().

    Returns
    -------
    pd.DataFrame
        Formatted summary table with Pass/Fail indicators.
    """
    print("\n" + "=" * 70)
    print("STATIONARITY TEST SUMMARY TABLE")
    print("=" * 70)

    # Create summary table
    summary = pd.DataFrame({
        'Country': results_df['Country'],
        'ADF': results_df['ADF_Stationary'].apply(lambda x: 'Pass' if x else 'Fail'),
        'PP': results_df['PP_Stationary'].apply(lambda x: 'Pass' if x else 'Fail'),
        'Tests_Passed': results_df['Stationary_Count'].apply(lambda x: f"{x}/2"),
        'Status': results_df['Consensus_Category'].apply(lambda x:
            'Both Stationary' if x == 'strong_stationary' else
            'Both Non-Stationary' if x == 'strong_nonstationary' else
            'Mixed results'
        )
    })

    # Print table
    print("\n" + summary.to_string(index=False))

    # Print summary statistics
    print("\n" + "-" * 70)
    adf_pass = results_df['ADF_Stationary'].sum()
    pp_pass = results_df['PP_Stationary'].sum()
    total = len(results_df)

    print(f"\nTest Pass Rates:")
    print(f"  ADF: {adf_pass}/{total} countries ({100 * adf_pass / total:.1f}%)")
    print(f"  PP:  {pp_pass}/{total} countries ({100 * pp_pass / total:.1f}%)")

    return summary


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_raw_vs_deseasonalized(df_raw, df_yoy, save_dir):
    """
    Plot raw CPI series vs deseasonalized (YoY) series for each country.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Original CPI dataframe (before transformation).
    df_yoy : pd.DataFrame
        YoY transformed dataframe.
    save_dir : str
        Directory to save plots.
    """
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("PLOTTING RAW vs DESEASONALIZED SERIES")
    print("=" * 70)

    country_cols = [col for col in df_raw.columns if col != 'period']

    print(f"Creating plots for {len(country_cols)} countries...")

    for col in country_cols:
        country = col.split(' – ')[0]
        print(f"  Plotting {country}...", end=" ")

        # Create figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle(f'{country} - Raw CPI vs Deseasonalized (YoY)',
                     fontsize=16, fontweight='bold')

        # Top panel: Raw CPI
        ax1 = axes[0]
        ax1.plot(df_raw['period'], df_raw[col],
                 color='steelblue', linewidth=1.5, label='Raw CPI')
        ax1.set_ylabel('CPI Index', fontsize=12, fontweight='bold')
        ax1.set_title('(a) Raw CPI - Level and Trend', fontsize=12, loc='left')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper left', fontsize=10)

        # Bottom panel: YoY
        ax2 = axes[1]
        ax2.plot(df_yoy['period'], df_yoy[col],
                 color='darkred', linewidth=1.5, label='YoY Log Difference')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel('YoY Log Difference', fontsize=12, fontweight='bold')
        ax2.set_title('(b) Deseasonalized - Year-over-Year Growth Rate',
                      fontsize=12, loc='left')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='upper left', fontsize=10)

        plt.tight_layout()

        # Save plot
        country_filename = country.replace(' ', '_').lower()
        plot_path = os.path.join(save_dir, f'{country_filename}_raw_vs_yoy.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print("Done")

    print(f"\n✓ All plots saved to: {save_dir}")


def plot_all_countries_comparison(df_yoy, save_path):
    """
    Create a multi-panel plot comparing all countries' YoY transformations.

    Parameters
    ----------
    df_yoy : pd.DataFrame
        YoY transformed dataframe.
    save_path : str
        Path to save the combined plot.
    """
    country_cols = [col for col in df_yoy.columns if col != 'period']
    n_countries = len(country_cols)

    print("\n" + "=" * 70)
    print("CREATING COMBINED COMPARISON PLOT")
    print("=" * 70)

    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_countries + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    fig.suptitle('All Countries - Deseasonalized (YoY) Series Comparison',
                 fontsize=16, fontweight='bold')

    axes_flat = axes.flatten() if n_countries > 1 else [axes]

    for idx, col in enumerate(country_cols):
        ax = axes_flat[idx]
        country = col.split(' – ')[0]

        ax.plot(df_yoy['period'], df_yoy[col],
                color='darkred', linewidth=1, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_title(country, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.tick_params(axis='x', rotation=45)

    # Hide empty subplots
    for idx in range(n_countries, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()

    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Combined plot saved to: {save_path}")


# ============================================================================
# DATA EXPORT FUNCTIONS
# ============================================================================

def save_country_subsets(df_yoy, results_df, output_dir):
    """
    Save individual country YoY files for countries passing stationarity.

    Criteria: PP test must pass OR country is Switzerland.

    Parameters
    ----------
    df_yoy : pd.DataFrame
        YoY transformed dataframe.
    results_df : pd.DataFrame
        Stationarity test results.
    output_dir : str
        Directory to save country files.

    Returns
    -------
    list
        List of countries retained.
    """
    print("\n" + "=" * 70)
    print("FILTERING COUNTRIES BY STATIONARITY CRITERIA")
    print("=" * 70)
    print("Criteria: PP test must pass OR country is Switzerland")

    # Filter countries
    countries_to_keep = results_df[
        (results_df['PP_Stationary']) |
        (results_df['Country'] == 'Switzerland')
    ]

    keep_countries = countries_to_keep['Country'].tolist()

    print(f"\nCountries retained: {len(keep_countries)}/{len(results_df)}")
    print(f"Countries kept: {', '.join(keep_countries)}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save individual country files
    for col in df_yoy.columns[1:]:
        country = col.split(' – ')[0]
        if country in keep_countries:
            country_df = df_yoy[['period', col]].copy()
            country_df.columns = ['period', 'cpi_yoy']

            country_filename = country.lower().replace(' ', '_')
            country_path = os.path.join(output_dir, f'{country_filename}_cpi_yoy.csv')
            country_df.to_csv(country_path, index=False)

    print(f"\n✓ Individual country files saved to: {output_dir}")

    return keep_countries


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def main():
    """
    Main execution function for ARIMA data preparation pipeline.

    Steps:
    1. Load processed CPI data
    2. Calculate YoY log differences
    3. Create visualization plots
    4. Perform stationarity tests
    5. Save results and country subsets

    Returns
    -------
    tuple
        (df_yoy, results_df, summary_table)
    """
    print("\n" + "=" * 70)
    print(" " * 10 + "ARIMA DATA PREPARATION PIPELINE")
    print("=" * 70)

    # Define file paths
    INPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'CPI_1996_2019.csv')
    COUNTRY_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'yoy_cpi_country_subset')
    STATIONARITY_DIR = os.path.join(PROJECT_ROOT, 'results', 'arimax_forecast', 'stationarity_test')
    PLOTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'arimax_forecast', 'plots', 'deseason')

    # STEP 1: Load processed CPI data
    print("\n[STEP 1/6] Loading processed CPI data...")
    assert os.path.exists(INPUT_PATH), f"Input file not found: {INPUT_PATH}"
    df = pd.read_csv(INPUT_PATH)
    df['period'] = pd.to_datetime(df['period'])
    print(f"✓ Loaded {df.shape[0]} observations for {df.shape[1] - 1} countries")

    # STEP 2: Calculate YoY log differences
    print("\n[STEP 2/6] Calculating YoY log differences...")
    df_yoy = calculate_yoy_log_difference(df)

    # STEP 3: Create visualization plots
    print("\n[STEP 3/6] Creating visualization plots...")
    plot_raw_vs_deseasonalized(df, df_yoy, save_dir=PLOTS_DIR)
    comparison_plot_path = os.path.join(PLOTS_DIR, 'all_countries_comparison.png')
    plot_all_countries_comparison(df_yoy, save_path=comparison_plot_path)

    # STEP 4: Perform stationarity tests
    print("\n[STEP 4/6] Performing stationarity tests...")
    results_df = test_all_countries_stationarity(df_yoy)

    # STEP 5: Create and save summary table
    print("\n[STEP 5/6] Saving stationarity results...")
    summary_table = create_stationarity_summary_table(results_df)

    # Create output directory
    os.makedirs(STATIONARITY_DIR, exist_ok=True)

    # Save results
    results_path = os.path.join(STATIONARITY_DIR, 'stationarity_test_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"✓ Results saved to: {results_path}")

    summary_path = os.path.join(STATIONARITY_DIR, 'stationarity_summary_table.csv')
    summary_table.to_csv(summary_path, index=False)
    print(f"✓ Summary saved to: {summary_path}")

    # STEP 6: Save country subsets
    print("\n[STEP 6/6] Saving country subsets...")
    keep_countries = save_country_subsets(df_yoy, results_df, COUNTRY_OUTPUT_DIR)

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE - READY FOR ARIMA MODELING")
    print("=" * 70)
    print("\nOutputs generated:")
    print(f"  1. Stationarity results: {results_path}")
    print(f"  2. Summary table: {summary_path}")
    print(f"  3. Country subsets: {COUNTRY_OUTPUT_DIR}/")
    print(f"  4. Plots: {PLOTS_DIR}/")
    print(f"\nCountries ready for modeling: {len(keep_countries)}")
    print("=" * 70)

    return df_yoy, results_df, summary_table


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    df_yoy, results_df, summary_table = main()
