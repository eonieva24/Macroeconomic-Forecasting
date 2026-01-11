"""
Prepare Clean Datasets for ARIMA-X Modeling

This module loads YoY transformed CPI, Unemployment Rate, and Import Prices data,
merges them by country, and creates clean datasets with no missing values
for ARIMA-X forecasting.

Each country gets one DataFrame with:
- y_t: YoY CPI inflation (dependent variable)
- x1_t: Unemployment Rate (exogenous variable 1) - raw percentage
- x2_t: Import Prices YoY (exogenous variable 2)

Author: Elena Onieva Henrich
Date: November 2025
Course: Advanced Programming 2025 - Forecasting Project
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import pandas as pd

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_cpi_yoy_data(input_dir):
    """
    Load YoY CPI data from country subset files.

    Parameters
    ----------
    input_dir : str
        Directory containing country YoY CSV files.

    Returns
    -------
    pd.DataFrame
        Merged dataframe with all countries.
    """
    assert os.path.exists(input_dir), f"Directory not found: {input_dir}"

    country_files = [f for f in os.listdir(input_dir) if f.endswith('_cpi_yoy.csv')]
    assert len(country_files) > 0, f"No country files found in {input_dir}"

    df = None

    for country_file in sorted(country_files):
        country_path = os.path.join(input_dir, country_file)
        country_df = pd.read_csv(country_path)
        country_df['period'] = pd.to_datetime(country_df['period'])

        # Extract country name from filename
        country_name = country_file.replace('_cpi_yoy.csv', '').replace('_', ' ').title()

        # Rename column
        country_df = country_df.rename(columns={'cpi_yoy': f'{country_name} – CPI'})

        if df is None:
            df = country_df
        else:
            df = df.merge(country_df, on='period', how='outer')

    df = df.sort_values('period').reset_index(drop=True)

    return df


def load_yoy_datasets():
    """
    Load all three datasets needed for ARIMA-X.

    Returns
    -------
    tuple
        (df_cpi_yoy, df_unemployment, df_import_yoy)
    """
    print("Loading datasets...")

    # Load dependent variable: CPI YoY from individual country files
    cpi_input_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'yoy_cpi_country_subset')
    df_cpi = load_cpi_yoy_data(cpi_input_dir)
    print(f"✓ CPI YoY: {df_cpi.shape[0]} rows, {df_cpi.shape[1] - 1} countries")

    # Load exogenous variable 1: Unemployment Rate (raw percentage)
    unemployment_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'unemployment_1996_2019.csv')
    assert os.path.exists(unemployment_path), f"File not found: {unemployment_path}"
    df_unemployment = pd.read_csv(unemployment_path)
    df_unemployment['period'] = pd.to_datetime(df_unemployment['period'])
    print(f"✓ Unemployment Rate: {df_unemployment.shape[0]} rows, {df_unemployment.shape[1] - 1} countries")

    # Load exogenous variable 2: Import Prices YoY
    import_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'import_prices_yoy.csv')
    assert os.path.exists(import_path), f"File not found: {import_path}"
    df_import = pd.read_csv(import_path)
    df_import['period'] = pd.to_datetime(df_import['period'])
    print(f"✓ Import Prices YoY: {df_import.shape[0]} rows, {df_import.shape[1] - 1} countries")

    return df_cpi, df_unemployment, df_import


def identify_common_countries(df_cpi, df_unemployment, df_import):
    """
    Identify countries present in all three datasets.

    Parameters
    ----------
    df_cpi : pd.DataFrame
        CPI YoY data.
    df_unemployment : pd.DataFrame
        Unemployment rate data.
    df_import : pd.DataFrame
        Import Prices YoY data.

    Returns
    -------
    list
        List of countries present in all datasets.
    """
    # Extract country names from column names
    cpi_countries = set([col.split(' – ')[0] for col in df_cpi.columns if col != 'period'])

    # Unemployment columns don't have ' – ' separator, they're just country names
    unemployment_countries = set([col for col in df_unemployment.columns if col != 'period'])

    # Import prices - check format
    import_cols = [col for col in df_import.columns if col != 'period']
    if ' – ' in import_cols[0] if import_cols else False:
        import_countries = set([col.split(' – ')[0] for col in import_cols])
    else:
        import_countries = set(import_cols)

    # Find intersection
    common_countries = cpi_countries & unemployment_countries & import_countries

    print(f"\n{'=' * 70}")
    print("COUNTRY AVAILABILITY")
    print(f"{'=' * 70}")
    print(f"CPI data: {len(cpi_countries)} countries")
    print(f"  {', '.join(sorted(cpi_countries))}")
    print(f"Unemployment data: {len(unemployment_countries)} countries")
    print(f"  {', '.join(sorted(unemployment_countries))}")
    print(f"Import Prices data: {len(import_countries)} countries")
    print(f"  {', '.join(sorted(import_countries))}")
    print(f"\nCommon countries (in all 3 datasets): {len(common_countries)}")
    print(f"  {', '.join(sorted(common_countries))}")

    # Countries missing
    missing_unemployment = cpi_countries - unemployment_countries
    if missing_unemployment:
        print(f"\n⚠ Excluded (no unemployment): {', '.join(sorted(missing_unemployment))}")

    missing_import = cpi_countries - import_countries
    if missing_import:
        print(f"⚠ Excluded (no import prices): {', '.join(sorted(missing_import))}")

    return sorted(common_countries)


# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

def find_column_for_country(df, country):
    """
    Find the column name in a dataframe that matches a country.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to search.
    country : str
        Country name to find.

    Returns
    -------
    str
        Column name matching the country.
    """
    for col in df.columns:
        if col == 'period':
            continue
        # Check if country name is in column (handles both 'Belgium' and 'Belgium – CPI')
        col_country = col.split(' – ')[0] if ' – ' in col else col
        if col_country == country:
            return col

    raise ValueError(f"No column found for country: {country}")


def prepare_country_data(country, df_cpi, df_unemployment, df_import):
    """
    Prepare clean ARIMA-X dataset for one country.

    Merges y_t (CPI), x1_t (Unemployment), x2_t (Import Prices) and drops rows
    with any missing values.

    Parameters
    ----------
    country : str
        Country name.
    df_cpi : pd.DataFrame
        CPI YoY data.
    df_unemployment : pd.DataFrame
        Unemployment rate data.
    df_import : pd.DataFrame
        Import Prices YoY data.

    Returns
    -------
    pd.DataFrame
        Clean dataset with columns: period, y_t, x1_t, x2_t
    """
    print(f"\nPreparing data for {country}...")

    # Find matching columns in each dataset
    cpi_col = find_column_for_country(df_cpi, country)
    unemployment_col = find_column_for_country(df_unemployment, country)
    import_col = find_column_for_country(df_import, country)

    # Extract relevant columns
    df_y = df_cpi[['period', cpi_col]].copy()
    df_x1 = df_unemployment[['period', unemployment_col]].copy()
    df_x2 = df_import[['period', import_col]].copy()

    # Rename to standard names
    df_y = df_y.rename(columns={cpi_col: 'y_t'})
    df_x1 = df_x1.rename(columns={unemployment_col: 'x1_t'})
    df_x2 = df_x2.rename(columns={import_col: 'x2_t'})

    # Merge all three on period
    df_merged = df_y.merge(df_x1, on='period', how='outer')
    df_merged = df_merged.merge(df_x2, on='period', how='outer')

    # Sort by date
    df_merged = df_merged.sort_values('period').reset_index(drop=True)

    # Count missing before dropping
    initial_rows = len(df_merged)
    missing_y = df_merged['y_t'].isna().sum()
    missing_x1 = df_merged['x1_t'].isna().sum()
    missing_x2 = df_merged['x2_t'].isna().sum()

    print(f"  Initial rows: {initial_rows}")
    print(f"  Missing y_t (CPI): {missing_y}")
    print(f"  Missing x1_t (Unemployment): {missing_x1}")
    print(f"  Missing x2_t (Import): {missing_x2}")

    # Drop rows with ANY missing values
    df_clean = df_merged.dropna().copy()

    # Report final dataset
    final_rows = len(df_clean)
    rows_dropped = initial_rows - final_rows

    if final_rows > 0:
        date_start = df_clean['period'].min().strftime('%Y-%m-%d')
        date_end = df_clean['period'].max().strftime('%Y-%m-%d')

        print(f"  Dropped {rows_dropped} rows with missing values")
        print(f"  ✓ Clean data: {final_rows} observations")
        print(f"  ✓ Date range: {date_start} to {date_end}")
    else:
        print(f"  ✗ No complete observations after dropping missing values")

    return df_clean


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to prepare ARIMA-X datasets for all countries.

    Steps:
    1. Load YoY CPI, Unemployment, and Import Prices data
    2. Identify common countries
    3. Merge variables and drop missing values
    4. Save clean datasets per country

    Returns
    -------
    pd.DataFrame
        Summary of prepared datasets.
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "ARIMA-X DATA PREPARATION")
    print("=" * 70)

    # Define output directory
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'arimax_forecast', 'arimax')

    # STEP 1: Load datasets
    print("\n[STEP 1/3] Loading datasets...")
    df_cpi, df_unemployment, df_import = load_yoy_datasets()

    # STEP 2: Identify common countries
    print("\n[STEP 2/3] Identifying common countries...")
    common_countries = identify_common_countries(df_cpi, df_unemployment, df_import)

    if len(common_countries) == 0:
        print("\n✗ No common countries found in all datasets")
        return None

    # STEP 3: Prepare data for each country
    print("\n[STEP 3/3] Preparing clean datasets...")
    print("=" * 70)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_data = []

    for country in common_countries:
        df_clean = prepare_country_data(country, df_cpi, df_unemployment, df_import)

        if len(df_clean) > 0:
            # Save to CSV
            filename = f"{country.replace(' ', '_').lower()}_arimax.csv"
            filepath = os.path.join(OUTPUT_DIR, filename)
            df_clean.to_csv(filepath, index=False)
            print(f"  ✓ Saved to: {filepath}")

            # Store summary info
            summary_data.append({
                'Country': country,
                'Observations': len(df_clean),
                'Start_Date': df_clean['period'].min().strftime('%Y-%m-%d'),
                'End_Date': df_clean['period'].max().strftime('%Y-%m-%d'),
                'File': filename
            })

    # Create summary table
    print("\n" + "=" * 70)
    print("SUMMARY: CLEAN ARIMAX DATASETS")
    print("=" * 70)

    summary_df = pd.DataFrame(summary_data)

    if len(summary_df) > 0:
        print(f"\n{summary_df.to_string(index=False)}")

        # Save summary
        summary_path = os.path.join(OUTPUT_DIR, 'arimax_data_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Summary saved to: {summary_path}")

        print(f"\n✓ Successfully prepared {len(summary_df)} country datasets")
        print(f"✓ All files saved to: {OUTPUT_DIR}")
    else:
        print("\n✗ No datasets were successfully prepared")

    print("\n" + "=" * 70)
    print("ARIMA-X DATA PREPARATION COMPLETE")
    print("=" * 70)
    print("\nVariables in each dataset:")
    print("  - y_t: CPI YoY (dependent variable)")
    print("  - x1_t: Unemployment Rate (exogenous)")
    print("  - x2_t: Import Prices YoY (exogenous)")

    return summary_df


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    summary = main()
