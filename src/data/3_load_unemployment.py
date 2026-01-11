"""
Unemployment Rate Data Loading and Cleaning Module

This script loads unemployment rate data from individual country CSV files,
automatically detects quarterly vs monthly frequency on a year-by-year basis,
and expands quarterly data to monthly frequency where needed.

Automatic frequency detection:
- For each year, counts observations
- 3-4 observations/year -> quarterly -> expand to monthly
- 10-12 observations/year -> monthly -> keep as-is
- Handles mixed frequency within same country (e.g., Israel: quarterly
  1992-2011, monthly 2012+)

Note: Some countries have data availability limitations:
- Latvia: Data starts 2002
- Korea: Data starts ~1999
- Israel: Mixed frequency (quarterly until 2011, monthly after)

Author: Elena Onieva Henrich
Date: November 2025
Course: Advanced Programming 2025 - Forecasting Project
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os

import pandas as pd

# Display configuration for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

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

def detect_year_frequency(df, country_name, year):
    """
    Detect whether a specific year has quarterly or monthly data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'period' and country column.
    country_name : str
        Name of the country column.
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


def analyze_frequency_by_year(df, country_name):
    """
    Analyze frequency for each year in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'period' and country column.
    country_name : str
        Name of the country column.

    Returns
    -------
    dict
        Dictionary mapping year -> frequency ('quarterly', 'monthly', 'unknown').
    """
    years = df['period'].dt.year.unique()
    frequency_map = {}

    for year in sorted(years):
        frequency_map[year] = detect_year_frequency(df, country_name, year)

    return frequency_map


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_single_country_unemployment(filepath, country_name):
    """
    Load unemployment data for a single country from CSV file.

    Parameters
    ----------
    filepath : str
        Path to the country's unemployment CSV file.
    country_name : str
        Name of the country (used for output column naming).

    Returns
    -------
    pd.DataFrame
        Dataframe with 'period' and country_name columns.

    Raises
    ------
    AssertionError
        If file does not exist or required columns are missing.
    """
    assert os.path.exists(filepath), f"File not found: {filepath}"

    df = pd.read_csv(filepath)

    assert 'DateTime' in df.columns, f"Missing 'DateTime' column in {filepath}"
    assert 'Value' in df.columns, f"Missing 'Value' column in {filepath}"

    df_clean = df[['DateTime', 'Value']].copy()
    df_clean['period'] = pd.to_datetime(
        df_clean['DateTime']
    ).dt.to_period('M').dt.to_timestamp()

    df_clean = df_clean[['period', 'Value']].copy()
    df_clean.columns = ['period', country_name]

    return df_clean


def expand_quarterly_year_to_monthly(year_data, country_name, year):
    """
    Expand quarterly observations for a single year to monthly.

    For a year with quarterly data, this function creates monthly
    observations by repeating the quarterly value for each month
    in the quarter.

    Parameters
    ----------
    year_data : pd.DataFrame
        Dataframe containing only observations for the specified year.
    country_name : str
        Name of the country column.
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
        value = row[country_name]

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
                country_name: value
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
        Dataframe with 'period' and country column.
    country_name : str
        Name of the country column.

    Returns
    -------
    pd.DataFrame
        Dataframe with all years converted to monthly frequency.
    """
    assert not df.empty, "Cannot expand empty dataframe"

    # Analyze frequency by year
    frequency_map = analyze_frequency_by_year(df, country_name)

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

    # Process each year
    result_frames = []

    for year in sorted(frequency_map.keys()):
        year_data = df[df['period'].dt.year == year].copy()
        freq = frequency_map[year]

        if freq == 'quarterly':
            # Expand quarterly to monthly
            expanded = expand_quarterly_year_to_monthly(
                year_data, country_name, year
            )
            result_frames.append(expanded)
        else:
            # Keep monthly data as-is
            result_frames.append(year_data)

    # Combine all years
    df_result = pd.concat(result_frames, ignore_index=True)

    # Sort and remove any duplicates
    df_result = df_result.sort_values('period').drop_duplicates(
        subset='period'
    ).reset_index(drop=True)

    # Report transformation
    if quarterly_years:
        print(f"  Expanded {len(quarterly_years)} quarterly years to monthly")
        print(f"  Final observations: {len(df_result)}")

    return df_result


def load_all_unemployment_data(raw_data_dir):
    """
    Load unemployment data for all countries and merge into single dataframe.

    This function loads individual country files, automatically detects
    and handles quarterly data on a year-by-year basis, and merges
    all countries into one dataframe.

    Parameters
    ----------
    raw_data_dir : str
        Path to directory containing raw unemployment CSV files.

    Returns
    -------
    pd.DataFrame
        Merged dataframe with 'period' column and one column per country.

    Raises
    ------
    AssertionError
        If raw data directory does not exist.
    """
    assert os.path.exists(raw_data_dir), f"Directory not found: {raw_data_dir}"

    print("\n" + "=" * 60)
    print("LOADING UNEMPLOYMENT DATA BY COUNTRY")
    print("=" * 60)

    df_merged = None

    for country_name, file_component in COUNTRY_FILE_MAPPING.items():
        filename = (
            f"historical_country_{file_component}_indicator_Unemployment_Rate.csv"
        )
        filepath = os.path.join(raw_data_dir, filename)

        print(f"\nLoading {country_name}...")

        # Load country data
        df_country = load_single_country_unemployment(filepath, country_name)

        original_obs = len(df_country)
        print(f"  Raw observations: {original_obs}")

        # Automatic frequency detection and expansion
        df_country = expand_quarterly_to_monthly_auto(df_country, country_name)

        print(f"  Date range: {df_country['period'].min().strftime('%Y-%m')} "
              f"to {df_country['period'].max().strftime('%Y-%m')}")

        # Merge with main dataframe
        if df_merged is None:
            df_merged = df_country
        else:
            df_merged = pd.merge(df_merged, df_country, on='period', how='outer')

    df_merged = df_merged.sort_values('period').reset_index(drop=True)

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"Total observations: {len(df_merged)}")
    print(f"Total countries: {len(df_merged.columns) - 1}")

    return df_merged


# ============================================================================
# DATA FILTERING FUNCTIONS
# ============================================================================

def filter_time_period(df, start_date='1996-01-01', end_date='2020-01-01'):
    """
    Filter dataframe for specified time period.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with datetime 'period' column.
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format (default: '1996-01-01').
    end_date : str, optional
        End date in 'YYYY-MM-DD' format, exclusive (default: '2020-01-01').

    Returns
    -------
    pd.DataFrame
        Filtered dataframe for specified date range.

    Raises
    ------
    AssertionError
        If filtering results in empty dataframe.
    """
    original_size = len(df)

    df_filtered = df[
        (df['period'] >= start_date) & (df['period'] < end_date)
    ].copy()

    assert not df_filtered.empty, "Filtering resulted in empty dataframe"

    print("\n" + "=" * 60)
    print("TIME PERIOD FILTERING")
    print("=" * 60)
    print(f"Original data: {original_size} rows")
    print(f"Filtered data: {len(df_filtered)} rows")
    print(f"Date range: {df_filtered['period'].min().strftime('%Y-%m')} "
          f"to {df_filtered['period'].max().strftime('%Y-%m')}")
    print("\n✓ Data filtered to 1996-2019")

    return df_filtered


# ============================================================================
# DATA VALIDATION FUNCTIONS
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
    dict
        Dictionary mapping country names to observation counts.
    """
    total_months = len(df)

    print("\n" + "=" * 60)
    print("DATA COMPLETENESS BY COUNTRY")
    print("=" * 60)
    print(f"\nTotal possible observations: {total_months} months")
    print("-" * 60)

    obs_count = {}

    for col in df.columns[1:]:
        count = df[col].notna().sum()
        missing = df[col].isna().sum()
        pct = (count / total_months) * 100
        obs_count[col] = count

        print(f"{col:15s}: {count:3d} obs ({pct:5.1f}%) | Missing: {missing}")

    return obs_count


# ============================================================================
# DATA EXPORT FUNCTIONS
# ============================================================================

def save_processed_data(df, output_path, create_dir=True):
    """
    Save processed dataframe to CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataframe to save.
    output_path : str
        Full path for output CSV file.
    create_dir : bool, optional
        If True, create output directory if it doesn't exist (default: True).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If dataframe is empty or save operation fails.
    """
    assert not df.empty, "Cannot save empty dataframe"

    if create_dir:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✓ Created directory: {output_dir}")

    df.to_csv(output_path, index=False)

    assert os.path.exists(output_path), f"Failed to create file: {output_path}"

    print("\n" + "=" * 60)
    print("DATA EXPORT")
    print("=" * 60)
    print(f"Saved to: {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\n✓ Processed data saved successfully")


def print_final_summary(df):
    """
    Print comprehensive summary of processed dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Final processed dataframe.

    Returns
    -------
    None
    """
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)

    print(f"Total observations: {df.shape[0]} months")
    print(f"Total countries: {df.shape[1] - 1}")

    print(f"\nTime period:")
    print(f"  Start: {df['period'].min()}")
    print(f"  End:   {df['period'].max()}")

    print(f"\nCountries included:")
    print(", ".join(df.columns[1:]))

    total_cells = df.shape[0] * (df.shape[1] - 1)
    missing_cells = df.iloc[:, 1:].isnull().sum().sum()
    completeness = (1 - missing_cells / total_cells) * 100

    print(f"\nData quality:")
    print(f"  Total data points: {total_cells:,}")
    print(f"  Missing values: {missing_cells:,}")
    print(f"  Completeness: {completeness:.2f}%")

    print("\nData availability notes:")
    for col in df.columns[1:]:
        valid_data = df[df[col].notna()]
        if len(valid_data) > 0:
            start = valid_data['period'].min().strftime('%Y-%m')
            end = valid_data['period'].max().strftime('%Y-%m')
            count = len(valid_data)
            if count < df.shape[0]:
                print(f"  {col}: {start} to {end} ({count} obs)")

    print("\n" + "=" * 60)
    print("UNEMPLOYMENT DATA LOADING COMPLETE")
    print("=" * 60)


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def main():
    """
    Main execution function for unemployment data loading pipeline.

    This function orchestrates the complete data preparation workflow:
    1. Load raw unemployment data for all countries
    2. Automatically detect quarterly vs monthly frequency per year
    3. Expand quarterly data to monthly where detected
    4. Merge all countries into single dataframe
    5. Filter to 1996-2019 period
    6. Analyze data completeness
    7. Save processed data
    8. Print final summary

    Returns
    -------
    pd.DataFrame
        Cleaned and filtered dataframe ready for merging with CPI data.
    """
    print("\n" + "=" * 70)
    print(" " * 12 + "UNEMPLOYMENT DATA LOADING PIPELINE")
    print("=" * 70)

    # Define file paths
    RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'unemployment')
    PROCESSED_DATA_PATH = os.path.join(
        PROJECT_ROOT, 'data', 'processed', 'unemployment_1996_2019.csv'
    )

    # STEP 1: Load all country data with automatic frequency detection
    print("\n[STEP 1/5] Loading unemployment data for all countries...")
    print("  (Automatic quarterly/monthly detection enabled)")
    df = load_all_unemployment_data(RAW_DATA_DIR)

    # STEP 2: Filter time period
    print("\n[STEP 2/5] Filtering time period...")
    df = filter_time_period(df, start_date='1996-01-01', end_date='2020-01-01')

    # STEP 3: Analyze completeness
    print("\n[STEP 3/5] Analyzing data completeness...")
    analyze_data_completeness(df)

    # STEP 4: Save processed data
    print("\n[STEP 4/5] Saving processed data...")
    save_processed_data(df, PROCESSED_DATA_PATH)

    # STEP 5: Print final summary
    print("\n[STEP 5/5] Generating final summary...")
    print_final_summary(df)

    return df


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    df_unemployment = main()
