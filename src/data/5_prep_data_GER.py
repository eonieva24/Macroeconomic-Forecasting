#!/usr/bin/env python3
"""
5_prep_data_GER.py

Prepare a monthly Germany (GER) feature dataset for high-dimensional
Random Forest inflation forecasting.

This script loads raw indicator data, applies indicator-specific
transformations, creates lagged features, and outputs a model-ready dataset.

Design Principles
-----------------
- Sample restriction: 1996-01 to 2019-12 (applied BEFORE lagging)
- Strictly lagged predictors (no contemporaneous X_t)
- Indicator-specific transformations + standardized lags [1, 2, 3, 4]
  following Beck & Wolf (2025) methodology
- Interest rate daily -> monthly average (if daily)
- Drop wages (yearly USD) by not loading it
- Final dropna after all transformations/lags

Inputs
------
1. Target (CPI YoY): data/processed/yoy_cpi_country_subset/germany_cpi_yoy.csv
2. Exogenous indicators: data/raw/GER_indicators/

Output
------
data/processed/GER_indicators/GER_features_1996_2019.csv

Reference
---------
Beck, E. and Wolf, M. (2025). "Forecasting Inflation with the Hedged
Random Forest". SNB Working Papers 07/2025.

Author: Elena Onieva Henrich
Date: January 2026
Course: Advanced Programming 2025 - Forecasting Project
"""

from __future__ import annotations

# =============================================================================
# IMPORTS
# =============================================================================

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RAW_DIR = PROJECT_ROOT / 'data' / 'raw' / 'GER_indicators'
CPI_YOY_PATH = (
    PROJECT_ROOT / 'data' / 'processed' /
    'yoy_cpi_country_subset' / 'germany_cpi_yoy.csv'
)

OUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'GER_indicators'


# =============================================================================
# SAMPLE CONFIGURATION
# =============================================================================

SAMPLE_START = '1996-01-01'
SAMPLE_END = '2019-12-31'


# =============================================================================
# LAG CONFIGURATION (Beck & Wolf 2025)
# =============================================================================

# Standardized lag structure following Beck & Wolf (2025), Section 3:
# "The feature set includes four autoregressive lags of the target variable...
# and four lags of each selected variable."
STANDARD_LAGS: List[int] = [1, 2, 3, 4]


# =============================================================================
# INDICATOR SPECIFICATIONS
# =============================================================================

@dataclass(frozen=True)
class IndicatorSpec:
    """
    Specification for an exogenous indicator.

    Attributes
    ----------
    key : str
        Internal name for the indicator.
    filename_contains : str
        Substring to match in filename for auto-discovery.
    transform : str
        Transformation: 'level', 'yoy_log', 'yoy_diff', 'monthly_avg_level'.
    lags : List[int]
        List of lag periods to create as features.
    freq_hint : Optional[str]
        Frequency hint ('daily' or None) for parsing.
    """

    key: str
    filename_contains: str
    transform: str
    lags: List[int]
    freq_hint: Optional[str] = None


# Indicator specifications with standardized lags [1, 2, 3, 4]
# per Beck & Wolf (2025) methodology
SPECS: List[IndicatorSpec] = [
    # Prices & cost pressures
    IndicatorSpec(
        'import_prices', 'Import_Prices', 'yoy_log', STANDARD_LAGS
    ),
    IndicatorSpec(
        'producer_prices', 'Producer_Prices', 'yoy_log', STANDARD_LAGS
    ),
    IndicatorSpec(
        'energy_inflation', 'Energy_Inflation', 'level', STANDARD_LAGS
    ),

    # Real activity & labor demand
    IndicatorSpec(
        'manufacturing_production', 'Manufacturing_Production', 'level',
        STANDARD_LAGS
    ),
    IndicatorSpec(
        'employed_persons', 'Employed_Persons', 'yoy_log', STANDARD_LAGS
    ),
    IndicatorSpec(
        'unemployment_rate', 'Unemployment_Rate', 'level', STANDARD_LAGS
    ),

    # Financial conditions & monetary policy
    IndicatorSpec(
        'private_sector_credit', 'Private_Sector_Credit', 'yoy_log',
        STANDARD_LAGS
    ),
    IndicatorSpec(
        'banks_balance_sheet', 'Banks_Balance_Sheet', 'yoy_log', STANDARD_LAGS
    ),
    IndicatorSpec(
        'interest_rate', 'Interest_Rate', 'monthly_avg_level', STANDARD_LAGS,
        freq_hint='daily'
    ),

    # External sector flows
    IndicatorSpec(
        'current_account', 'Current_Account', 'yoy_diff', STANDARD_LAGS
    ),
    IndicatorSpec(
        'foreign_direct_investment', 'Foreign_Direct_Investment', 'yoy_diff',
        STANDARD_LAGS
    ),
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_directory(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def parse_dates_auto(series: pd.Series) -> pd.Series:
    """
    Parse date strings to datetime with automatic format detection.

    Tries multiple formats to handle various date representations.

    Parameters
    ----------
    series : pd.Series
        Series of date strings.

    Returns
    -------
    pd.Series
        Parsed datetime series.
    """
    # First try: automatic parsing without dayfirst (ISO format)
    result = pd.to_datetime(series, errors='coerce')

    # If many NaT, try with dayfirst=True
    if result.isna().sum() > len(result) * 0.5:
        result = pd.to_datetime(series, errors='coerce', dayfirst=True)

    return result


def to_month_start(dt_series: pd.Series) -> pd.Series:
    """
    Convert datetime series to month-start timestamps.

    Parameters
    ----------
    dt_series : pd.Series
        Datetime series.

    Returns
    -------
    pd.Series
        Month-start datetime series.
    """
    # Normalize to first day of month
    return dt_series.dt.to_period('M').dt.to_timestamp()


def safe_read_csv(path: Path) -> pd.DataFrame:
    """
    Read CSV file, trying comma separator first, then semicolon.

    Parameters
    ----------
    path : Path
        Path to CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=';')


def infer_date_column(df: pd.DataFrame) -> str:
    """
    Infer the date column from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    str
        Name of the inferred date column.
    """
    candidates = [
        c for c in df.columns
        if re.search(r'(date|time|period)', c, flags=re.I)
    ]
    if candidates:
        return candidates[0]
    return df.columns[0]


def infer_value_column(df: pd.DataFrame, date_col: str) -> str:
    """
    Infer the numeric value column from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    date_col : str
        Name of the date column to exclude.

    Returns
    -------
    str
        Name of the inferred value column.

    Raises
    ------
    ValueError
        If no suitable numeric column is found.
    """
    numeric_cols = []
    for col in df.columns:
        if col == date_col:
            continue
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        valid_count = numeric_series.notna().sum()
        min_required = max(5, int(0.1 * len(df)))  # Reduced threshold
        if valid_count >= min_required:
            numeric_cols.append((col, valid_count))

    if not numeric_cols:
        raise ValueError(
            f"Could not infer numeric value column (date_col={date_col}). "
            f"Available columns: {list(df.columns)}"
        )

    numeric_cols.sort(key=lambda x: x[1], reverse=True)
    return numeric_cols[0][0]


def restrict_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict dataframe to the configured sample period.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'period' column.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    mask = (
        (df['period'] >= pd.Timestamp(SAMPLE_START)) &
        (df['period'] <= pd.Timestamp(SAMPLE_END))
    )
    return df[mask].copy()


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_monthly_series(path: Path, freq_hint: Optional[str] = None) -> pd.DataFrame:
    """
    Load a CSV file and convert to monthly frequency.

    Parameters
    ----------
    path : Path
        Path to CSV file.
    freq_hint : Optional[str]
        Frequency hint ('daily' or None).

    Returns
    -------
    pd.DataFrame
        Monthly dataframe with columns: period, value.
    """
    df = safe_read_csv(path)

    date_col = infer_date_column(df)
    val_col = infer_value_column(df, date_col)

    out = df[[date_col, val_col]].copy()
    out.columns = ['date', 'value']
    out['date'] = parse_dates_auto(out['date'])
    out['value'] = pd.to_numeric(out['value'], errors='coerce')

    out = out.dropna(subset=['date', 'value']).sort_values('date')

    # Infer frequency from median day spacing
    if len(out) >= 3:
        diffs = out['date'].diff().dt.total_seconds().dropna() / (24 * 3600)
        median_days = float(diffs.median())
    else:
        median_days = 30.0

    is_daily = median_days <= 2.0
    if freq_hint and freq_hint.lower() == 'daily':
        is_daily = True

    # Aggregate to monthly
    if is_daily:
        out['period'] = to_month_start(out['date'])
        out = out.groupby('period', as_index=False)['value'].mean()
    else:
        out['period'] = to_month_start(out['date'])
        out = out.groupby('period', as_index=False)['value'].last()

    return out[['period', 'value']]


def find_indicator_file(raw_dir: Path, contains: str) -> Optional[Path]:
    """
    Find a CSV file in raw_dir containing a specific substring.

    Parameters
    ----------
    raw_dir : Path
        Directory to search.
    contains : str
        Substring to match in filename (case-insensitive).

    Returns
    -------
    Optional[Path]
        Path to matching file, or None if not found.
    """
    candidates = list(raw_dir.glob('*.csv'))
    matches = [p for p in candidates if contains.lower() in p.name.lower()]

    if not matches:
        return None

    matches.sort(key=lambda p: len(p.name), reverse=True)
    return matches[0]


def load_cpi_yoy_target(path: Path) -> pd.DataFrame:
    """
    Load Germany CPI YoY series as target variable.

    Parameters
    ----------
    path : Path
        Path to CPI YoY CSV file.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns: period, inflation_yoy.
    """
    print(f"  Loading target from: {path.name}")

    df = safe_read_csv(path)
    print(f"  Raw file rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    date_col = infer_date_column(df)
    val_col = infer_value_column(df, date_col)
    print(f"  Detected date column: '{date_col}', value column: '{val_col}'")

    out = df[[date_col, val_col]].copy()
    out.columns = ['date', 'inflation_yoy']

    # Parse dates - use automatic detection for ISO format
    out['date'] = parse_dates_auto(out['date'])
    out['inflation_yoy'] = pd.to_numeric(out['inflation_yoy'], errors='coerce')

    # Check parsing results
    valid_dates = out['date'].notna().sum()
    valid_values = out['inflation_yoy'].notna().sum()
    print(f"  Valid dates after parsing: {valid_dates}/{len(out)}")
    print(f"  Valid values after parsing: {valid_values}/{len(out)}")

    # Drop rows with NaN (first 12 months have no YoY value)
    out = out.dropna(subset=['date', 'inflation_yoy']).sort_values('date')
    print(f"  Rows after dropna: {len(out)}")

    # Convert to month-start period
    out['period'] = to_month_start(out['date'])

    # Keep last value per month (shouldn't matter for monthly data)
    out = out.groupby('period', as_index=False)['inflation_yoy'].last()
    print(f"  Rows after groupby: {len(out)}")

    # Restrict to sample period
    out = restrict_sample(out)
    print(f"  Rows after restrict_sample ({SAMPLE_START} to {SAMPLE_END}): {len(out)}")

    # Validate we have enough data
    if len(out) < 200:
        print(f"  WARNING: Only {len(out)} observations loaded!")
        print(f"  Date range: {out['period'].min()} to {out['period'].max()}")

    return out[['period', 'inflation_yoy']]


# =============================================================================
# TRANSFORMATION FUNCTIONS
# =============================================================================

def yoy_log_inflation(series: pd.Series, lag: int = 12) -> pd.Series:
    """
    Compute year-over-year log inflation rate.

    Formula: 100 * (log(x_t) - log(x_{t-12}))

    Parameters
    ----------
    series : pd.Series
        Price index series.
    lag : int
        Number of periods for YoY calculation (default 12).

    Returns
    -------
    pd.Series
        YoY log inflation rate.
    """
    return 100.0 * (np.log(series) - np.log(series.shift(lag)))


def yoy_level_diff(series: pd.Series, lag: int = 12) -> pd.Series:
    """
    Compute year-over-year level difference.

    Formula: x_t - x_{t-12}

    Parameters
    ----------
    series : pd.Series
        Value series.
    lag : int
        Number of periods for YoY calculation (default 12).

    Returns
    -------
    pd.Series
        YoY level difference.
    """
    return series - series.shift(lag)


def add_lags(
    df: pd.DataFrame,
    col: str,
    lags: List[int],
    prefix: Optional[str] = None
) -> None:
    """
    Add lagged versions of a column to the dataframe (in-place).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to modify.
    col : str
        Column name to lag.
    lags : List[int]
        List of lag periods.
    prefix : Optional[str]
        Prefix for new column names (defaults to col name).
    """
    p = prefix if prefix else col
    for lag in lags:
        df[f'{p}_lag{lag}'] = df[col].shift(lag)


# =============================================================================
# MAIN DATA BUILDING FUNCTION
# =============================================================================

def build_germany_dataset() -> Tuple[pd.DataFrame, Dict]:
    """
    Build the complete Germany feature dataset.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        Model-ready dataframe and metadata dictionary.
    """
    # Step 1: Load target
    if not CPI_YOY_PATH.exists():
        raise FileNotFoundError(f"Target CPI YoY file not found: {CPI_YOY_PATH}")

    target = load_cpi_yoy_target(CPI_YOY_PATH)
    print(f"✓ Loaded target: {len(target)} observations")

    if len(target) == 0:
        raise ValueError("Target loaded with 0 observations!")

    # Step 2: Load exogenous indicators
    series_map: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []

    for spec in SPECS:
        path = find_indicator_file(RAW_DIR, spec.filename_contains)
        if path is None:
            missing.append(spec.filename_contains)
            print(f"  ⚠ Missing: {spec.filename_contains}")
            continue

        series = load_monthly_series(path, freq_hint=spec.freq_hint)
        series = restrict_sample(series)
        series = series.rename(columns={'value': spec.key})
        series_map[spec.key] = series
        print(f"  ✓ Loaded: {spec.key} ({len(series)} obs)")

    # Step 3: Merge onto target calendar
    merged = target.copy()
    for key, series in series_map.items():
        merged = merged.merge(series[['period', key]], on='period', how='left')

    merged = merged.sort_values('period').reset_index(drop=True)
    print(f"\n✓ Merged dataset: {len(merged)} observations")

    # Step 4: Add seasonality feature
    merged['month'] = merged['period'].dt.month

    # Step 5: Apply indicator-specific transformations
    transformed_base: Dict[str, str] = {}

    for spec in SPECS:
        if spec.key not in merged.columns:
            continue

        if spec.transform in ('level', 'monthly_avg_level'):
            base_col = spec.key
        elif spec.transform == 'yoy_log':
            base_col = f'{spec.key}_yoy'
            merged[base_col] = yoy_log_inflation(merged[spec.key], lag=12)
        elif spec.transform == 'yoy_diff':
            base_col = f'{spec.key}_yoy_diff'
            merged[base_col] = yoy_level_diff(merged[spec.key], lag=12)
        else:
            raise ValueError(
                f"Unknown transform '{spec.transform}' for {spec.key}"
            )

        transformed_base[spec.key] = base_col

    # Step 6: Add target lags (Beck & Wolf 2025: 4 autoregressive lags)
    add_lags(merged, 'inflation_yoy', STANDARD_LAGS, prefix='inflation_yoy')

    # Step 7: Add predictor lags
    for spec in SPECS:
        if spec.key not in transformed_base:
            continue
        base = transformed_base[spec.key]
        add_lags(merged, base, spec.lags, prefix=base)

    # Step 8: Keep only model columns
    keep = ['period', 'inflation_yoy', 'month']
    keep += [c for c in merged.columns if c.startswith('inflation_yoy_lag')]

    lag_cols = [c for c in merged.columns if re.search(r'_lag\d+$', c)]
    lag_cols = [c for c in lag_cols if c not in keep]
    keep += sorted(lag_cols)

    df_model = merged[keep].copy()

    # Step 9: Diagnostic before dropna
    print(f"\n  Before dropna: {len(df_model)} rows")
    for col in df_model.columns:
        na_count = df_model[col].isna().sum()
        if na_count > 0:
            print(f"    {col}: {na_count} NaN")

    # Step 10: Drop NA
    n_before = len(df_model)
    df_model = df_model.dropna().reset_index(drop=True)
    n_after = len(df_model)
    print(f"\n✓ After dropna: {n_after} observations (dropped {n_before - n_after})")

    # Validate
    if n_after == 0:
        print("\n  ERROR: All rows dropped by dropna!")
        print("  This usually means one or more columns are entirely NaN")
        print("  Check that all indicator data covers the sample period")

    # Build metadata
    metadata = {
        'raw_dir': str(RAW_DIR),
        'cpi_yoy_path': str(CPI_YOY_PATH),
        'out_dir': str(OUT_DIR),
        'sample_start': SAMPLE_START,
        'sample_end': SAMPLE_END,
        'n_rows': int(df_model.shape[0]),
        'n_cols': int(df_model.shape[1]),
        'columns': list(df_model.columns),
        'missing_indicator_files': missing,
        'lag_structure': STANDARD_LAGS,
        'notes': [
            'Wages excluded (yearly USD wages).',
            'Food_Inflation excluded (file not found).',
            'Target loaded from germany_cpi_yoy.csv as inflation_yoy.',
            'All predictors are lagged; no contemporaneous X_t exported.',
            'Interest rate aggregated daily->monthly mean if daily detected.',
            'Lag structure [1, 2, 3, 4] per Beck & Wolf (2025) methodology.',
        ],
    }

    return df_model, metadata


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main() -> None:
    """Main entry point for Germany data preparation."""
    print("=" * 70)
    print(" " * 10 + "GERMANY DATA PREPARATION")
    print("=" * 70)
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Raw data dir: {RAW_DIR}")
    print(f"Output dir:   {OUT_DIR}")
    print(f"Sample:       {SAMPLE_START} to {SAMPLE_END}")
    print(f"Lag structure: {STANDARD_LAGS} (Beck & Wolf 2025)")

    ensure_directory(OUT_DIR)

    print("\n" + "-" * 70)
    print("LOADING AND PROCESSING DATA")
    print("-" * 70)

    df_model, metadata = build_germany_dataset()

    print("\n" + "-" * 70)
    print("SAVING OUTPUTS")
    print("-" * 70)

    out_csv = OUT_DIR / 'GER_features_1996_2019.csv'
    out_json = OUT_DIR / 'GER_features_1996_2019_metadata.json'

    df_model.to_csv(out_csv, index=False)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Data saved:     {out_csv}")
    print(f"  Rows: {df_model.shape[0]}, Cols: {df_model.shape[1]}")
    print(f"✓ Metadata saved: {out_json}")

    if df_model.shape[0] > 0:
        print("\n" + "-" * 70)
        print("SANITY CHECKS")
        print("-" * 70)
        print(f"Date range: {df_model['period'].min()} to "
              f"{df_model['period'].max()}")
        print(f"\nFirst 5 columns: {list(df_model.columns[:5])}")
        print(f"Last 5 columns: {list(df_model.columns[-5:])}")
    else:
        print("\n  WARNING: Output file has 0 rows!")

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    main()
