"""
OECD CPI Data Loading and Cleaning Module

This script loads OECD Consumer Price Index (CPI) data excluding food and energy,
filters for the pre-COVID period (1996-2019) to avoid structural breaks introduced
by the pandemic, and selects countries with complete data coverage for time series
forecasting analysis.

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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_oecd_cpi_data(filepath):
    """
    Load OECD CPI data from CSV file.

    This function reads the raw OECD monthly CPI data excluding food and energy
    components. It performs basic validation to ensure the file exists and
    contains expected structure.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV data file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with period column and country columns.

    Raises
    ------
    AssertionError
        If the file does not exist or loaded data does not meet validation.
    """
    # Check if file exists
    assert os.path.exists(filepath), f"Data file not found at: {filepath}"

    print("Loading OECD CPI data (excluding food and energy)...")

    # Load CSV file into pandas dataframe
    df = pd.read_csv(filepath)

    # Validate loaded data structure
    assert not df.empty, "Loaded dataframe is empty"
    assert 'period' in df.columns, "Data must contain 'period' column"
    assert df.shape[1] > 1, "Data must contain at least one country column"

    # Print dataset overview
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB")
    print(f"Date range: {df['period'].iloc[0]} to {df['period'].iloc[-1]}")

    return df


def convert_period_to_datetime(df):
    """
    Convert period column from string to datetime format.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'period' column as string.

    Returns
    -------
    pd.DataFrame
        Dataframe with 'period' column converted to datetime.

    Raises
    ------
    AssertionError
        If period column is not present or conversion fails.
    """
    # Defensive check
    assert 'period' in df.columns, "Dataframe must have 'period' column"

    # Convert period column to datetime
    df['period'] = pd.to_datetime(df['period'])

    # Validate conversion succeeded
    assert pd.api.types.is_datetime64_any_dtype(df['period']), \
        "Period conversion to datetime failed"

    print("\n✓ Period column converted to datetime format")

    return df


# ============================================================================
# DATA FILTERING FUNCTIONS
# ============================================================================

def filter_time_period(df, start_date='1996-01-01', end_date='2020-01-01'):
    """
    Filter dataframe for specified time period.

    Excludes the COVID-19 period which introduced structural breaks in economic
    time series data. Using data from 1996-2019 ensures consistent economic
    regime for forecasting model training.

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
        If period column is not datetime or filtering results in empty dataframe.
    """
    # Validate inputs
    assert pd.api.types.is_datetime64_any_dtype(df['period']), \
        "Period column must be datetime type"

    # Store original size for comparison
    original_size = len(df)

    # Filter dataframe for specified date range
    df_filtered = df[(df['period'] >= start_date) & (df['period'] < end_date)].copy()

    # Ensure filtering produced valid result
    assert not df_filtered.empty, "Filtering resulted in empty dataframe"

    # Report filtering results
    print("\n" + "=" * 60)
    print("TIME PERIOD FILTERING")
    print("=" * 60)
    print(f"Original data: {original_size} rows")
    print(f"  From: {df['period'].min()}")
    print(f"  To:   {df['period'].max()}")
    print(f"\nFiltered data: {len(df_filtered)} rows")
    print(f"  From: {df_filtered['period'].min()}")
    print(f"  To:   {df_filtered['period'].max()}")
    print(f"\nRows removed: {original_size - len(df_filtered)} " +
          f"({(original_size - len(df_filtered)) / original_size * 100:.1f}%)")
    print("\n✓ Data filtered to pre-COVID period (1996-2019)")

    return df_filtered


def analyze_data_completeness(df, verbose=True):
    """
    Analyze and report data completeness by country.

    This function counts non-missing observations for each country and calculates
    the percentage of complete data. Essential for identifying countries with
    sufficient data quality for time series modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'period' column and country columns.
    verbose : bool, optional
        If True, print detailed completeness report (default: True).

    Returns
    -------
    dict
        Dictionary mapping country names to observation counts.
    """
    # Calculate total possible observations
    total_months = len(df)

    # Initialize dictionary to store observation counts
    obs_count = {}

    # Count non-missing values for each country
    for col in df.columns[1:]:  # Skip 'period' column
        # Extract country name from column header
        country = col.split(' – ')[0]

        # Count non-null observations
        count = df[col].notna().sum()

        # Store in dictionary
        obs_count[country] = count

    # Print completeness report if verbose
    if verbose:
        print("\n" + "=" * 60)
        print("DATA COMPLETENESS BY COUNTRY")
        print("=" * 60)

        # Sort countries by observation count (descending)
        sorted_obs = sorted(obs_count.items(), key=lambda x: x[1], reverse=True)

        print(f"\nTotal possible observations: {total_months} months")
        print("\nCountries ranked by data completeness:")
        print("-" * 60)

        # Print each country with count and percentage
        for country, count in sorted_obs:
            pct = (count / total_months) * 100
            print(f"{country:30s}: {count:3d} obs ({pct:5.1f}%)")

    return obs_count


def select_complete_countries(df, countries_to_keep=None):
    """
    Select countries with complete or near-complete data.

    Filters dataframe to retain only specified countries. This ensures robust
    time series models by avoiding gaps in historical data. Excludes aggregate
    measures (G7, OECD Total, etc.).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'period' column and country columns.
    countries_to_keep : list of str, optional
        Explicit list of country names to retain. If None, uses default list
        of countries with complete data (default: None).

    Returns
    -------
    pd.DataFrame
        Filtered dataframe containing only selected countries.

    Raises
    ------
    AssertionError
        If no countries meet the selection criteria.
    """
    # Use default country list if none provided
    if countries_to_keep is None:
        countries_to_keep = [
            'Belgium', 'Germany', 'Israel', 'Korea',
            'Latvia', 'Lithuania', 'Norway', 'Switzerland'
        ]

    print("\n" + "=" * 60)
    print("SELECTING COUNTRIES WITH COMPLETE DATA")
    print("=" * 60)

    # Build list of columns to keep
    cols_to_keep = ['period']

    # Iterate through country columns
    for col in df.columns[1:]:
        # Extract country name
        country = col.split(' – ')[0]

        # Check if country is in keep list
        if country in countries_to_keep:
            cols_to_keep.append(col)

    # Ensure we found at least one country to keep
    assert len(cols_to_keep) > 1, "No countries found matching the keep list"

    # Create filtered dataframe
    df_clean = df[cols_to_keep].copy()

    # Report filtering results
    print(f"Original: {len(df.columns) - 1} country columns")
    print(f"Selected: {len(df_clean.columns) - 1} country columns")
    print(f"\nCountries retained ({len(countries_to_keep)}):")
    print(", ".join(sorted(countries_to_keep)))
    print("\n✓ Data filtered to complete countries only")

    # Validate final dataframe
    assert df_clean.shape[0] > 0, "Filtered dataframe has no rows"
    assert df_clean.shape[1] > 1, "Filtered dataframe has no country columns"

    return df_clean


# ============================================================================
# DATA EXPORT FUNCTIONS
# ============================================================================

def save_processed_data(df, output_path, create_dir=True):
    """
    Save processed dataframe to CSV file.

    Exports cleaned and filtered dataframe to the processed data directory.
    Creates directory if it doesn't exist.

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
    # Validate dataframe is not empty
    assert not df.empty, "Cannot save empty dataframe"

    # Create output directory if needed
    if create_dir:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✓ Created directory: {output_dir}")

    # Save dataframe to CSV
    df.to_csv(output_path, index=False)

    # Verify file was created
    assert os.path.exists(output_path), f"Failed to create file: {output_path}"

    # Report save results
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

    Provides final overview of cleaned data including dimensions, date range,
    countries included, and data quality metrics.

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

    # Dimensions
    print(f"Total observations: {df.shape[0]} months")
    print(f"Total countries: {df.shape[1] - 1}")

    # Date range
    print(f"\nTime period:")
    print(f"  Start: {df['period'].min().strftime('%Y-%m')}")
    print(f"  End:   {df['period'].max().strftime('%Y-%m')}")
    print(f"  Duration: {(df['period'].max() - df['period'].min()).days / 365.25:.1f} years")

    # Data quality
    total_cells = df.shape[0] * (df.shape[1] - 1)
    missing_cells = df.iloc[:, 1:].isnull().sum().sum()
    completeness = (1 - missing_cells / total_cells) * 100

    print(f"\nData quality:")
    print(f"  Total data points: {total_cells:,}")
    print(f"  Missing values: {missing_cells:,}")
    print(f"  Completeness: {completeness:.2f}%")

    print("\n" + "=" * 60)
    print("DATA LOADING COMPLETE - READY FOR FEATURE ENGINEERING")
    print("=" * 60)


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def main():
    """
    Main execution function for OECD CPI data loading and cleaning pipeline.

    This function orchestrates the complete data preparation workflow:
    1. Load raw OECD CPI data
    2. Convert period column to datetime
    3. Filter to pre-COVID period (1996-2019)
    4. Analyze data completeness
    5. Select countries with complete data
    6. Save processed data
    7. Print final summary

    Returns
    -------
    pd.DataFrame
        Cleaned and filtered dataframe ready for feature engineering.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "OECD CPI DATA LOADING PIPELINE")
    print("=" * 70)

    # Define file paths
    RAW_DATA_PATH = os.path.join(
        PROJECT_ROOT, 'data', 'raw', 'cpi', 'OECD-MEI time series_CPGRLE01.csv'
    )
    PROCESSED_DATA_PATH = os.path.join(
        PROJECT_ROOT, 'data', 'processed', 'CPI_1996_2019.csv'
    )

    # STEP 1: Load raw data
    print("\n[STEP 1/7] Loading raw data...")
    df = load_oecd_cpi_data(RAW_DATA_PATH)

    # STEP 2: Convert period to datetime
    print("\n[STEP 2/7] Converting period column...")
    df = convert_period_to_datetime(df)

    # STEP 3: Filter time period
    print("\n[STEP 3/7] Filtering time period...")
    df = filter_time_period(df, start_date='1996-01-01', end_date='2020-01-01')

    # STEP 4: Analyze completeness
    print("\n[STEP 4/7] Analyzing data completeness...")
    obs_count = analyze_data_completeness(df, verbose=True)

    # STEP 5: Select complete countries
    print("\n[STEP 5/7] Selecting complete countries...")
    df = select_complete_countries(df)

    # STEP 6: Save processed data
    print("\n[STEP 6/7] Saving processed data...")
    save_processed_data(df, PROCESSED_DATA_PATH)

    # STEP 7: Print final summary
    print("\n[STEP 7/7] Generating final summary...")
    print_final_summary(df)

    return df


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    df_clean = main()
