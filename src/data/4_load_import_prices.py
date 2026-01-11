"""
Import Prices Data Loading and Processing Module

This script loads import price data for each country, rebases all indices to
2015=100, automatically detects and handles quarterly data on a year-by-year
basis, applies YoY transformation, and tests stationarity.

Automatic frequency detection:
- For each country and each year, counts observations
- 3-4 observations/year -> quarterly -> expand to monthly
- 10-12 observations/year -> monthly -> keep as-is
- Handles mixed frequency within same country

Processing steps:
1. Load raw import price files for each country
2. Automatically detect quarterly vs monthly frequency per year
3. Expand quarterly data to monthly where detected
4. Rebase all indices to 2015=100
5. Filter to 1996-2019 period
6. Apply log Year-over-Year transformation
7. Test stationarity (ADF, PP)

Rebasing Formula: new_index = (old_index / index_2015) * 100

Author: Elena Onieva Henrich
Date: November 2025
Course: Advanced Programming 2025 - Forecasting Project
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import warnings

import numpy as np
import pandas as pd
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# ============================================================================
# COUNTRY CONFIGURATION
# ============================================================================

# Mapping: output column name -> filename component
COUNTRY_FILE_MAPPING = {
    'Belgium': 'Belgium',
    'Germany': 'Germany',
    'Israel': 'Israel',
    'Korea': 'South_Korea',
    'Latvia': 'Latvia',
    'Lithuania': 'Lithuania',
    'Norway': 'Norway',
    'Switzerland': 'Switzerland'
}


# ============================================================================
# FREQUENCY DETECTION FUNCTIONS
# ============================================================================

def detect_year_frequency(df, year):
    """
    Detect whether a specific year has quarterly or monthly data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'period' and 'value' columns.
    year : int
        Year to analyze.

    Returns
    -------
    str
        'quarterly' if 3-4 observations, 'monthly' if 10-12 observations,
        'unknown' otherwise.
    """
    year_data = df[df['period'].dt.year == year]
    obs_count = len(year_data)

    if obs_count >= 3 and obs_count <= 4:
        return 'quarterly'
    elif obs_count >= 10 and obs_count <= 12:
        return 'monthly'
    else:
        return 'unknown'


def analyze_frequency_by_year(df):
    """
    Analyze frequency for each year in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'period' and 'value' columns.

    Returns
    -------
    dict
        Dictionary mapping year -> frequency ('quarterly', 'monthly', 'unknown').
    """
    years = df['period'].dt.year.unique()
    frequency_map = {}

    for year in sorted(years):
        frequency_map[year] = detect_year_frequency(df, year)

    return frequency_map


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_single_country_import_prices(filepath, country_name):
    """
    Load import price data for a single country from CSV file.

    Parameters
    ----------
    filepath : str
        Path to the country's import price CSV file.
    country_name : str
        Name of the country (used for output column naming).

    Returns
    -------
    pd.DataFrame
        Dataframe with 'period' and 'value' columns.

    Raises
    ------
    AssertionError
        If file does not exist or required columns are missing.
    """
    assert os.path.exists(filepath), f"File not found: {filepath}"

    df = pd.read_csv(filepath)

    # Determine value column (different file formats)
    if 'Value' in df.columns:
        value_col = 'Value'
    elif 'Close' in df.columns:
        value_col = 'Close'
    else:
        raise ValueError(f"Cannot find value column in {filepath}")

    assert 'DateTime' in df.columns, f"Missing 'DateTime' column in {filepath}"

    df_clean = df[['DateTime', value_col]].copy()
    df_clean.columns = ['period', 'value']

    # Convert to datetime and normalize to month-start
    df_clean['period'] = pd.to_datetime(df_clean['period'])
    df_clean['period'] = df_clean['period'].apply(lambda x: x.replace(day=1))

    return df_clean


def expand_quarterly_year_to_monthly(year_data, year):
    """
    Expand quarterly observations for a single year to monthly.

    Parameters
    ----------
    year_data : pd.DataFrame
        Dataframe containing only observations for the specified year.
    year : int
        The year being processed.

    Returns
    -------
    pd.DataFrame
        Dataframe with monthly frequency for this year (up to 12 rows).
    """
    expanded_rows = []

    for _, row in year_data.iterrows():
        quarterly_date = row['period']
        value = row['value']

        quarter_end_month = quarterly_date.month

        # Map quarter end month to first month of that quarter
        if quarter_end_month in [1, 2, 3]:
            first_month = 1
        elif quarter_end_month in [4, 5, 6]:
            first_month = 4
        elif quarter_end_month in [7, 8, 9]:
            first_month = 7
        else:
            first_month = 10

        # Generate 3 monthly dates for this quarter
        for month_offset in range(3):
            month = first_month + month_offset
            monthly_date = pd.Timestamp(year=year, month=month, day=1)
            expanded_rows.append({
                'period': monthly_date,
                'value': value
            })

    df_expanded = pd.DataFrame(expanded_rows)
    return df_expanded


def expand_quarterly_to_monthly_auto(df, country_name):
    """
    Automatically detect and expand quarterly data to monthly, year by year.

    This function analyzes each year's frequency independently:
    - Years with 3-4 observations are detected as quarterly and expanded
    - Years with 10-12 observations are kept as monthly
    - Handles mixed frequency within the same country

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'period' and 'value' columns.
    country_name : str
        Name of the country (for logging).

    Returns
    -------
    pd.DataFrame
        Dataframe with all years converted to monthly frequency.
    """
    assert not df.empty, "Cannot expand empty dataframe"

    # Analyze frequency by year
    frequency_map = analyze_frequency_by_year(df)

    # Count quarterly vs monthly years
    quarterly_years = [y for y, f in frequency_map.items() if f == 'quarterly']
    monthly_years = [y for y, f in frequency_map.items() if f == 'monthly']

    # Report detection results
    if quarterly_years:
        print(f"  Detected quarterly years: {min(quarterly_years)}-"
              f"{max(quarterly_years)} ({len(quarterly_years)} years)")
    if monthly_years:
        print(f"  Detected monthly years: {min(monthly_years)}-"
              f"{max(monthly_years)} ({len(monthly_years)} years)")

    # If all monthly, no transformation needed
    if not quarterly_years:
        print(f"  All data is monthly, no expansion needed")
        return df.copy()

    # Process each year
    result_frames = []

    for year in sorted(frequency_map.keys()):
        year_data = df[df['period'].dt.year == year].copy()
        freq = frequency_map[year]

        if freq == 'quarterly':
            expanded = expand_quarterly_year_to_monthly(year_data, year)
            result_frames.append(expanded)
        else:
            result_frames.append(year_data)

    # Combine all years
    df_result = pd.concat(result_frames, ignore_index=True)

    # Sort and remove any duplicates
    df_result = df_result.sort_values('period').drop_duplicates(
        subset='period'
    ).reset_index(drop=True)

    # Report transformation
    print(f"  Expanded {len(quarterly_years)} quarterly years to monthly")
    print(f"  Final observations: {len(df_result)}")

    return df_result


# ============================================================================
# REBASING FUNCTIONS
# ============================================================================

def get_2015_value(df):
    """
    Extract the index value at 2015 for rebasing.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'period' and 'value' columns.

    Returns
    -------
    float
        Index value in 2015.

    Raises
    ------
    AssertionError
        If no 2015 data found.
    """
    df_2015 = df[df['period'].dt.year == 2015]
    assert len(df_2015) > 0, "No data found for year 2015"

    return df_2015['value'].iloc[0]


def rebase_to_2015(df, country_name):
    """
    Rebase index series to 2015 = 100.

    Formula: new_index = (old_index / index_2015) * 100

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'period' and 'value' columns.
    country_name : str
        Name of country for reporting.

    Returns
    -------
    pd.DataFrame
        Rebased dataframe with 2015 = 100.
    """
    value_2015 = get_2015_value(df)

    print(f"  Index value at 2015: {value_2015:.2f}")

    # Check if already rebased
    if abs(value_2015 - 100.0) < 0.1:
        print(f"  -> Already at 2015=100, no rebasing needed")
        return df.copy()

    # Apply rebasing formula
    df_rebased = df.copy()
    df_rebased['value'] = (df_rebased['value'] / value_2015) * 100

    # Verify rebasing
    new_2015_value = get_2015_value(df_rebased)
    assert abs(new_2015_value - 100.0) < 0.01, "Rebasing failed"

    print(f"  -> Rebased to 2015=100")

    return df_rebased


# ============================================================================
# DATA FILTERING FUNCTIONS
# ============================================================================

def filter_time_period(df, start_date='1996-01-01', end_date='2020-01-01'):
    """
    Filter dataframe to 1996-2019 period.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'period' column.
    start_date : str
        Start date (inclusive).
    end_date : str
        End date (exclusive).

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    df_filtered = df[
        (df['period'] >= start_date) &
        (df['period'] < end_date)
    ].copy()

    return df_filtered


# ============================================================================
# MAIN LOADING FUNCTION
# ============================================================================

def load_all_import_prices(raw_data_dir):
    """
    Load import price data for all countries and merge into single dataframe.

    This function loads individual country files, automatically detects
    and handles quarterly data on a year-by-year basis for ALL countries,
    and merges all countries into one dataframe.

    Parameters
    ----------
    raw_data_dir : str
        Path to directory containing raw import price CSV files.

    Returns
    -------
    pd.DataFrame
        Merged dataframe with 'period' column and one column per country.
    """
    assert os.path.exists(raw_data_dir), f"Directory not found: {raw_data_dir}"

    print("\n" + "=" * 70)
    print("LOADING AND REBASING IMPORT PRICE DATA")
    print("=" * 70)

    # Create complete date range
    date_range = pd.date_range(start='1996-01-01', end='2019-12-31', freq='MS')
    df_merged = pd.DataFrame({'period': date_range})

    # Process each country
    for country_name, file_component in COUNTRY_FILE_MAPPING.items():
        filename = (
            f"historical_country_{file_component}_indicator_Import_Prices.csv"
        )
        filepath = os.path.join(raw_data_dir, filename)

        print(f"\nProcessing {country_name}...")

        # Load data
        df_country = load_single_country_import_prices(filepath, country_name)

        original_obs = len(df_country)
        print(f"  Raw observations: {original_obs}")

        # Automatic frequency detection and expansion for ALL countries
        df_country = expand_quarterly_to_monthly_auto(df_country, country_name)

        # Rebase to 2015=100
        df_country = rebase_to_2015(df_country, country_name)

        # Filter to 1996-2019
        df_country = filter_time_period(df_country)

        # Rename value column
        df_country = df_country.rename(columns={'value': country_name})

        print(f"  Final observations (1996-2019): {len(df_country)}")
        print(f"  Date range: {df_country['period'].min().strftime('%Y-%m')} "
              f"to {df_country['period'].max().strftime('%Y-%m')}")

        # Merge with main dataframe
        df_merged = pd.merge(df_merged, df_country, on='period', how='left')

    print("\n" + "=" * 70)
    print("MERGE COMPLETE")
    print("=" * 70)
    print(f"Total observations: {len(df_merged)}")
    print(f"Total countries: {len(df_merged.columns) - 1}")

    return df_merged


# ============================================================================
# YOY TRANSFORMATION FUNCTIONS
# ============================================================================

def apply_yoy_transformation(df):
    """
    Apply log Year-over-Year transformation to import prices.

    Formula: yoy = 100 * (log(value_t) - log(value_t-12))

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'period' and country columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with YoY transformed values.
    """
    print("\n" + "=" * 70)
    print("APPLYING LOG YEAR-OVER-YEAR TRANSFORMATION")
    print("=" * 70)

    df_yoy = df[['period']].copy()

    for col in df.columns[1:]:
        # Apply log YoY: 100 * (log(value_t) - log(value_t-12))
        log_values = np.log(df[col])
        log_yoy = 100 * (log_values - log_values.shift(12))

        df_yoy[col] = log_yoy

        print(f"✓ Transformed {col}")

    print(f"\nNote: First 12 months are NaN (1996-01 to 1996-12)")
    print(f"Usable data: 1997-01 to 2019-12 (276 months)")

    return df_yoy


# ============================================================================
# STATIONARITY TEST FUNCTIONS
# ============================================================================

def test_stationarity(series, country_name, significance_level=0.05):
    """
    Test stationarity using ADF and Phillips-Perron tests.

    Parameters
    ----------
    series : pd.Series
        Time series to test.
    country_name : str
        Name of country for display.
    significance_level : float
        Significance level (default: 0.05).

    Returns
    -------
    dict
        Test results.
    """
    series_clean = series.dropna()

    if len(series_clean) < 13:
        return {
            'Country': country_name,
            'ADF_Statistic': None,
            'ADF_PValue': None,
            'ADF_Stationary': False,
            'PP_Statistic': None,
            'PP_PValue': None,
            'PP_Stationary': False,
            'Both_Pass': False
        }

    # ADF Test
    adf_result = adfuller(series_clean, autolag='AIC')
    adf_stationary = adf_result[1] < significance_level

    # Phillips-Perron Test
    pp_result = PhillipsPerron(series_clean.values)
    pp_stationary = pp_result.pvalue < significance_level

    both_pass = adf_stationary and pp_stationary

    return {
        'Country': country_name,
        'ADF_Statistic': adf_result[0],
        'ADF_PValue': adf_result[1],
        'ADF_Stationary': adf_stationary,
        'PP_Statistic': pp_result.stat,
        'PP_PValue': pp_result.pvalue,
        'PP_Stationary': pp_stationary,
        'Both_Pass': both_pass
    }


def test_all_stationarity(df_yoy):
    """
    Test stationarity for all countries.

    Parameters
    ----------
    df_yoy : pd.DataFrame
        YoY transformed dataframe.

    Returns
    -------
    pd.DataFrame
        Stationarity test results.
    """
    print("\n" + "=" * 70)
    print("STATIONARITY TESTS (ADF, PHILLIPS-PERRON)")
    print("=" * 70)

    results = []

    for col in df_yoy.columns[1:]:
        print(f"\nTesting {col}...")
        result = test_stationarity(df_yoy[col], col)
        results.append(result)

        adf_status = "✓ Stationary" if result['ADF_Stationary'] else "✗ Non-stat"
        pp_status = "✓ Stationary" if result['PP_Stationary'] else "✗ Non-stat"

        print(f"  ADF: p-value={result['ADF_PValue']:.4f} {adf_status}")
        print(f"  PP:  p-value={result['PP_PValue']:.4f} {pp_status}")
        print(f"  Result: {'✓ BOTH PASS' if result['Both_Pass'] else '✗ FAIL'}")

    results_df = pd.DataFrame(results)

    # Summary
    print("\n" + "=" * 70)
    print("STATIONARITY TEST SUMMARY")
    print("=" * 70)
    n_pass = results_df['Both_Pass'].sum()
    n_total = len(results_df)
    print(f"Countries passing both tests: {n_pass}/{n_total}")

    return results_df


# ============================================================================
# DATA COMPLETENESS ANALYSIS
# ============================================================================

def analyze_data_completeness(df):
    """
    Analyze and report data completeness by country.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'period' column and country columns.

    Returns
    -------
    None
    """
    total_months = len(df)

    print("\n" + "=" * 70)
    print("DATA COMPLETENESS BY COUNTRY")
    print("=" * 70)
    print(f"\nTotal possible observations: {total_months} months")
    print("-" * 70)

    for col in df.columns[1:]:
        count = df[col].notna().sum()
        missing = df[col].isna().sum()
        pct = (count / total_months) * 100

        print(f"{col:15s}: {count:3d} obs ({pct:5.1f}%) | Missing: {missing}")


# ============================================================================
# DATA EXPORT FUNCTIONS
# ============================================================================

def save_processed_data(df, output_path):
    """
    Save processed dataframe to CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataframe to save.
    output_path : str
        Full path for output CSV file.
    """
    assert not df.empty, "Cannot save empty dataframe"

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(output_path, index=False)

    assert os.path.exists(output_path), f"Failed to create file: {output_path}"

    print(f"✓ Saved to: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def main():
    """
    Main execution function for import price data pipeline.

    Steps:
    1. Load raw import price data for all countries
    2. Automatically detect quarterly vs monthly frequency per year
    3. Expand quarterly data to monthly where detected
    4. Rebase to 2015=100
    5. Filter to 1996-2019
    6. Save rebased data
    7. Apply log YoY transformation
    8. Save YoY data
    9. Test stationarity
    10. Save stationarity results

    Returns
    -------
    tuple
        (df_rebased, df_yoy, results_df)
    """
    print("\n" + "=" * 70)
    print(" " * 12 + "IMPORT PRICES DATA PIPELINE")
    print("=" * 70)

    # Define paths
    RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'import_prices')
    REBASED_OUTPUT_PATH = os.path.join(
        PROJECT_ROOT, 'data', 'processed', 'import_prices_1996_2019.csv'
    )
    YOY_OUTPUT_PATH = os.path.join(
        PROJECT_ROOT, 'data', 'processed', 'import_prices_yoy.csv'
    )
    STATIONARITY_OUTPUT_PATH = os.path.join(
        PROJECT_ROOT, 'results', 'arimax_forecast', 'stationarity_test',
        'import_prices_stationarity_results.csv'
    )

    # STEP 1: Load and process all countries with automatic frequency detection
    print("\n[STEP 1/5] Loading import price data for all countries...")
    print("  (Automatic quarterly/monthly detection enabled)")
    df_rebased = load_all_import_prices(RAW_DATA_DIR)

    # STEP 2: Analyze completeness
    print("\n[STEP 2/5] Analyzing data completeness...")
    analyze_data_completeness(df_rebased)

    # STEP 3: Save rebased data
    print("\n[STEP 3/5] Saving rebased data...")
    save_processed_data(df_rebased, REBASED_OUTPUT_PATH)

    # STEP 4: Apply YoY transformation
    print("\n[STEP 4/5] Applying YoY transformation...")
    df_yoy = apply_yoy_transformation(df_rebased)
    save_processed_data(df_yoy, YOY_OUTPUT_PATH)

    # STEP 5: Test stationarity
    print("\n[STEP 5/5] Testing stationarity...")
    results_df = test_all_stationarity(df_yoy)

    # Create output directory for stationarity results
    stationarity_dir = os.path.dirname(STATIONARITY_OUTPUT_PATH)
    os.makedirs(stationarity_dir, exist_ok=True)

    results_df.to_csv(STATIONARITY_OUTPUT_PATH, index=False)
    print(f"\n✓ Stationarity results saved to: {STATIONARITY_OUTPUT_PATH}")

    # Final summary
    print("\n" + "=" * 70)
    print("IMPORT PRICES PROCESSING COMPLETE")
    print("=" * 70)
    print("\nOutputs generated:")
    print(f"  1. Rebased data: {REBASED_OUTPUT_PATH}")
    print(f"  2. YoY data: {YOY_OUTPUT_PATH}")
    print(f"  3. Stationarity results: {STATIONARITY_OUTPUT_PATH}")
    print(f"\nData summary:")
    print(f"  - Raw data: 1996-01 to 2019-12 (288 months)")
    print(f"  - YoY data: 1997-01 to 2019-12 (276 usable months)")
    print(f"  - Countries: {len(df_rebased.columns) - 1}")
    print(f"  - All indices rebased to 2015=100")
    print(f"  - Quarterly data automatically detected and expanded")
    print("=" * 70)

    return df_rebased, df_yoy, results_df


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    df_rebased, df_yoy, results_df = main()
