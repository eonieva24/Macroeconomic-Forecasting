"""
SARIMA-X Forecasting Pipeline with AutoARIMA Seasonal Detection.

This script forecasts CPI inflation using SARIMA-X models on raw CPI data,
letting AutoARIMA handle parameter selection including seasonality detection.

Pipeline:
1. Load raw CPI data (index levels, not YoY transformed)
2. Load exogenous variables (Unemployment, Import Prices)
3. Load optimal lags from previous lag selection
4. Merge and align data
5. Run AutoARIMA with seasonal=True to select SARIMA parameters
6. Rolling window 1-step-ahead forecasting
7. Convert forecasts to YoY for comparable metrics
8. Save results and generate plots

Approach:
- SARIMA operates on raw CPI levels (can capture seasonality directly)
- AutoARIMA selects (p,d,q)(P,D,Q,m) using BIC criterion
- Forecasts are converted to YoY scale for comparison with other models

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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

MIN_TRAINING_OBS = 36
SEASONAL_PERIOD = 12
PLOT_START_DATE = '2015-01-01'

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))

# Input paths
CPI_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'CPI_1996_2019.csv')
UNEMPLOYMENT_PATH = os.path.join(
    PROJECT_ROOT, 'data', 'processed', 'unemployment_1996_2019.csv')
IMPORT_PRICES_PATH = os.path.join(
    PROJECT_ROOT, 'data', 'processed', 'import_prices_1996_2019.csv')
OPTIMAL_LAGS_PATH = os.path.join(
    PROJECT_ROOT, 'results', 'arimax_forecast',
    'lag_selection', 'optimal_lags_summary.csv')

# Output paths
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, 'results', 'arimax_forecast', 'forecasts_seasonal')
PLOT_DIR = os.path.join(
    PROJECT_ROOT, 'results', 'arimax_forecast', 'plots', 'forecasts_seasonal')


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_cpi_data():
    """
    Load raw CPI data (index levels) and convert to long format.

    Returns
    -------
    pd.DataFrame
        CPI data with columns: Country, period, cpi_value
    """
    print("Loading CPI data...")
    assert os.path.exists(CPI_PATH), f"File not found: {CPI_PATH}"

    df_wide = pd.read_csv(CPI_PATH)
    df_wide['period'] = pd.to_datetime(df_wide['period'])

    # Melt to long format
    value_cols = [col for col in df_wide.columns if col != 'period']
    df_long = df_wide.melt(
        id_vars=['period'],
        value_vars=value_cols,
        var_name='country_col',
        value_name='cpi_value'
    )

    # Extract country name from column (before ' – ')
    df_long['Country'] = df_long['country_col'].str.split(' – ').str[0]
    df_long = df_long[['Country', 'period', 'cpi_value']].copy()
    df_long = df_long.dropna(subset=['cpi_value'])

    print(f"✓ Loaded CPI data: {len(df_long)} observations")
    print(f"  Countries: {df_long['Country'].nunique()}")
    print(f"  Date range: {df_long['period'].min().strftime('%Y-%m')} to "
          f"{df_long['period'].max().strftime('%Y-%m')}")

    return df_long


def load_unemployment_data():
    """
    Load unemployment data and convert to long format.

    Returns
    -------
    pd.DataFrame
        Unemployment data with columns: Country, period, unemployment_value
    """
    print("\nLoading Unemployment data...")
    assert os.path.exists(UNEMPLOYMENT_PATH), \
        f"File not found: {UNEMPLOYMENT_PATH}"

    df_wide = pd.read_csv(UNEMPLOYMENT_PATH)
    df_wide['period'] = pd.to_datetime(df_wide['period'])

    # Melt to long format
    value_cols = [col for col in df_wide.columns if col != 'period']
    df_long = df_wide.melt(
        id_vars=['period'],
        value_vars=value_cols,
        var_name='Country',
        value_name='unemployment_value'
    )
    df_long = df_long.dropna(subset=['unemployment_value'])

    print(f"✓ Loaded Unemployment data: {len(df_long)} observations")
    print(f"  Countries: {df_long['Country'].nunique()}")

    return df_long


def load_import_prices_data():
    """
    Load import prices data and convert to long format.

    Returns
    -------
    pd.DataFrame
        Import prices data with columns: Country, period, import_prices_value
    """
    print("\nLoading Import Prices data...")
    assert os.path.exists(IMPORT_PRICES_PATH), \
        f"File not found: {IMPORT_PRICES_PATH}"

    df_wide = pd.read_csv(IMPORT_PRICES_PATH)
    df_wide['period'] = pd.to_datetime(df_wide['period'])

    # Melt to long format
    value_cols = [col for col in df_wide.columns if col != 'period']
    df_long = df_wide.melt(
        id_vars=['period'],
        value_vars=value_cols,
        var_name='Country',
        value_name='import_prices_value'
    )
    df_long = df_long.dropna(subset=['import_prices_value'])

    print(f"✓ Loaded Import Prices data: {len(df_long)} observations")
    print(f"  Countries: {df_long['Country'].nunique()}")

    return df_long


def load_optimal_lags():
    """
    Load optimal lags from lag selection results.

    Returns
    -------
    dict
        Dictionary mapping country names to optimal lags.
    """
    print("\nLoading optimal lags...")

    if not os.path.exists(OPTIMAL_LAGS_PATH):
        print(f"  Warning: Optimal lags file not found: {OPTIMAL_LAGS_PATH}")
        print("  Using default lags (0, 0) for all countries")
        return {}

    df_lags = pd.read_csv(OPTIMAL_LAGS_PATH)

    lags_dict = {}
    for _, row in df_lags.iterrows():
        country = row['Country']
        lags_dict[country] = {
            'x1_lag': int(row['Best_x1_lag_BIC']),
            'x2_lag': int(row['Best_x2_lag_BIC'])
        }

    print(f"✓ Loaded optimal lags for {len(lags_dict)} countries")

    return lags_dict


# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

def prepare_country_data(cpi_df, unemp_df, import_df, country):
    """
    Prepare merged dataset for a single country.

    Parameters
    ----------
    cpi_df : pd.DataFrame
        CPI data (long format).
    unemp_df : pd.DataFrame
        Unemployment data (long format).
    import_df : pd.DataFrame
        Import prices data (long format).
    country : str
        Country name.

    Returns
    -------
    pd.DataFrame
        Merged data with columns: period, y_t, x1_t, x2_t
    """
    # Filter by country
    cpi_country = cpi_df[cpi_df['Country'] == country][
        ['period', 'cpi_value']].copy()
    unemp_country = unemp_df[unemp_df['Country'] == country][
        ['period', 'unemployment_value']].copy()
    import_country = import_df[import_df['Country'] == country][
        ['period', 'import_prices_value']].copy()

    # Check if data exists
    if len(cpi_country) == 0:
        print(f"  No CPI data for {country}")
        return pd.DataFrame()
    if len(unemp_country) == 0:
        print(f"  No Unemployment data for {country}")
        return pd.DataFrame()
    if len(import_country) == 0:
        print(f"  No Import Prices data for {country}")
        return pd.DataFrame()

    # Merge datasets
    df = cpi_country.merge(unemp_country, on='period', how='inner')
    df = df.merge(import_country, on='period', how='inner')

    # Rename columns
    df = df.rename(columns={
        'cpi_value': 'y_t',
        'unemployment_value': 'x1_t',
        'import_prices_value': 'x2_t'
    })

    # Sort by date
    df = df.sort_values('period').reset_index(drop=True)
    df = df.dropna()

    return df


def apply_exog_lags(df, x1_lag=0, x2_lag=0):
    """
    Apply lags to exogenous variables.

    Parameters
    ----------
    df : pd.DataFrame
        Data with y_t, x1_t, x2_t.
    x1_lag : int
        Lag for x1 (Unemployment).
    x2_lag : int
        Lag for x2 (Import Prices).

    Returns
    -------
    pd.DataFrame
        Data with lagged exogenous variables.
    """
    df_model = df.copy()

    if x1_lag == 0:
        df_model['x1_used'] = df_model['x1_t']
    else:
        df_model['x1_used'] = df_model['x1_t'].shift(x1_lag)

    if x2_lag == 0:
        df_model['x2_used'] = df_model['x2_t']
    else:
        df_model['x2_used'] = df_model['x2_t'].shift(x2_lag)

    df_model = df_model.dropna()

    return df_model


def compute_yoy_for_series(df, value_col='y_t'):
    """
    Compute YoY log difference for a series.

    Formula: YoY_t = log(y_t) - log(y_{t-12})

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with period and value column.
    value_col : str
        Name of column with values.

    Returns
    -------
    pd.DataFrame
        DataFrame with yoy column added.
    """
    df_yoy = df.copy()
    df_yoy = df_yoy.sort_values('period').reset_index(drop=True)

    log_values = np.log(df_yoy[value_col])
    df_yoy['yoy'] = log_values - log_values.shift(12)

    return df_yoy


def find_common_countries(cpi_df, unemp_df, import_df):
    """
    Find countries that exist in all three datasets.

    Parameters
    ----------
    cpi_df : pd.DataFrame
        CPI data.
    unemp_df : pd.DataFrame
        Unemployment data.
    import_df : pd.DataFrame
        Import prices data.

    Returns
    -------
    list
        List of common country names.
    """
    cpi_countries = set(cpi_df['Country'].unique())
    unemp_countries = set(unemp_df['Country'].unique())
    import_countries = set(import_df['Country'].unique())

    common = cpi_countries & unemp_countries & import_countries

    print(f"\nCountries in CPI data: {len(cpi_countries)}")
    print(f"Countries in Unemployment data: {len(unemp_countries)}")
    print(f"Countries in Import Prices data: {len(import_countries)}")
    print(f"Common countries: {len(common)}")
    print(f"  {sorted(common)}")

    return sorted(list(common))


# ============================================================================
# AUTO ARIMA PARAMETER SELECTION
# ============================================================================

def select_sarima_params(df_train):
    """
    Use auto_arima to select optimal SARIMA parameters.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data with y_t, x1_used, x2_used.

    Returns
    -------
    tuple
        ((p, d, q), (P, D, Q, m)) - order and seasonal_order
    """
    try:
        auto_model = auto_arima(
            y=df_train['y_t'],
            exogenous=df_train[['x1_used', 'x2_used']],
            start_p=0, max_p=3,
            start_q=0, max_q=3,
            d=None,
            max_d=2,
            seasonal=True,
            m=SEASONAL_PERIOD,
            start_P=0, max_P=2,
            start_Q=0, max_Q=2,
            D=None,
            max_D=1,
            stepwise=True,
            information_criterion='bic',
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            n_fits=50
        )

        order = auto_model.order
        seasonal_order = auto_model.seasonal_order

        return order, seasonal_order

    except Exception as e:
        print(f"    Warning: auto_arima failed: {str(e)}")
        return (1, 1, 0), (0, 0, 0, SEASONAL_PERIOD)


# ============================================================================
# ROLLING WINDOW FORECASTING
# ============================================================================

def rolling_window_forecast_sarima(df, order, seasonal_order,
                                   train_end_date='2018-12-01',
                                   test_end_date='2019-12-01'):
    """
    Perform rolling window 1-step-ahead forecasting with SARIMA.

    Computes forecasts on raw CPI scale and converts to YoY for comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with y_t, x1_used, x2_used.
    order : tuple
        (p, d, q) ARIMA order.
    seasonal_order : tuple
        (P, D, Q, m) seasonal order.
    train_end_date : str
        Initial training end date.
    test_end_date : str
        Last date to forecast.

    Returns
    -------
    pd.DataFrame
        Forecast results with both raw and YoY-converted values.
    """
    train_end = pd.to_datetime(train_end_date)
    test_end = pd.to_datetime(test_end_date)

    test_periods = df[
        (df['period'] > train_end) & (df['period'] <= test_end)
    ]['period'].values

    # Create lookup for historical values (needed for YoY conversion)
    df_sorted = df.sort_values('period').copy()
    historical_values = dict(zip(df_sorted['period'], df_sorted['y_t']))

    results = []

    for forecast_date in test_periods:
        df_train = df[df['period'] < forecast_date].copy()
        df_test_row = df[df['period'] == forecast_date].copy()

        if len(df_test_row) == 0:
            continue

        try:
            model = SARIMAX(
                endog=df_train['y_t'],
                exog=df_train[['x1_used', 'x2_used']],
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted = model.fit(disp=False)

            exog_forecast = df_test_row[['x1_used', 'x2_used']].values

            forecast_obj = fitted.get_forecast(steps=1, exog=exog_forecast)
            forecast_value = forecast_obj.predicted_mean.values[0]

            conf_int = forecast_obj.conf_int(alpha=0.05)
            lower_ci = conf_int.iloc[0, 0]
            upper_ci = conf_int.iloc[0, 1]

            actual_value = df_test_row['y_t'].values[0]
            error = actual_value - forecast_value
            pct_error = (forecast_value - actual_value) / actual_value * 100
            in_ci = (actual_value >= lower_ci) and (actual_value <= upper_ci)

            # YoY conversion
            forecast_date_ts = pd.Timestamp(forecast_date)
            date_12m_ago = forecast_date_ts - pd.DateOffset(months=12)
            actual_12m_ago = historical_values.get(date_12m_ago, np.nan)

            if not np.isnan(actual_12m_ago) and actual_12m_ago > 0:
                actual_yoy = np.log(actual_value) - np.log(actual_12m_ago)
                forecast_yoy = np.log(forecast_value) - np.log(actual_12m_ago)
                lower_ci_yoy = (np.log(max(lower_ci, 0.01)) -
                               np.log(actual_12m_ago))
                upper_ci_yoy = np.log(upper_ci) - np.log(actual_12m_ago)
                error_yoy = actual_yoy - forecast_yoy
                in_ci_yoy = ((actual_yoy >= lower_ci_yoy) and
                            (actual_yoy <= upper_ci_yoy))
            else:
                actual_yoy = np.nan
                forecast_yoy = np.nan
                lower_ci_yoy = np.nan
                upper_ci_yoy = np.nan
                error_yoy = np.nan
                in_ci_yoy = False

            results.append({
                'period': forecast_date_ts,
                'actual': actual_value,
                'forecast': forecast_value,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'error': error,
                'abs_error': abs(error),
                'pct_error': pct_error,
                'abs_pct_error': abs(pct_error),
                'in_ci': in_ci,
                'actual_yoy': actual_yoy,
                'forecast_yoy': forecast_yoy,
                'lower_ci_yoy': lower_ci_yoy,
                'upper_ci_yoy': upper_ci_yoy,
                'error_yoy': error_yoy,
                'abs_error_yoy': (abs(error_yoy) if not np.isnan(error_yoy)
                                 else np.nan),
                'in_ci_yoy': in_ci_yoy,
                'n_train': len(df_train)
            })

        except Exception as e:
            print(f"    Warning: Failed to forecast "
                  f"{pd.Timestamp(forecast_date).strftime('%Y-%m')}: {str(e)}")
            continue

    return pd.DataFrame(results)


# ============================================================================
# COUNTRY PROCESSING
# ============================================================================

def process_country(cpi_df, unemp_df, import_df, country, x1_lag=0, x2_lag=0):
    """
    Run full SARIMA pipeline for a single country.

    Parameters
    ----------
    cpi_df : pd.DataFrame
        CPI data.
    unemp_df : pd.DataFrame
        Unemployment data.
    import_df : pd.DataFrame
        Import prices data.
    country : str
        Country name.
    x1_lag : int
        Lag for unemployment.
    x2_lag : int
        Lag for import prices.

    Returns
    -------
    dict
        Results dictionary with forecasts and metrics.
    """
    print(f"\n{'=' * 70}")
    print(f"PROCESSING: {country}")
    print(f"{'=' * 70}")

    # Prepare data
    df = prepare_country_data(cpi_df, unemp_df, import_df, country)

    if len(df) == 0:
        print(f"  No data available for {country}")
        return None

    print(f"Merged observations: {len(df)}")
    print(f"Date range: {df['period'].min().strftime('%Y-%m')} to "
          f"{df['period'].max().strftime('%Y-%m')}")
    print(f"Using lags: x1_lag={x1_lag}, x2_lag={x2_lag}")

    # Apply lags
    df_model = apply_exog_lags(df, x1_lag, x2_lag)
    print(f"After lagging: {len(df_model)} observations")

    # Compute YoY for training data (needed for plotting)
    df_model = compute_yoy_for_series(df_model, 'y_t')

    # Get initial training data
    train_end = pd.to_datetime('2018-12-01')
    df_initial_train = df_model[df_model['period'] <= train_end]

    if len(df_initial_train) < MIN_TRAINING_OBS:
        print(f"  Insufficient training data ({len(df_initial_train)} obs), "
              f"need at least {MIN_TRAINING_OBS}, skipping...")
        return None

    print(f"Initial training: {len(df_initial_train)} observations")

    # Run auto_arima with seasonality
    print("\nRunning auto_arima with seasonal=True (BIC criterion)...")
    order, seasonal_order = select_sarima_params(df_initial_train)
    p, d, q = order
    P, D, Q, m = seasonal_order

    print(f"Selected: SARIMA({p},{d},{q})({P},{D},{Q})[{m}]")

    # Rolling window forecast
    print("\nRunning rolling window forecasts...")
    df_forecast = rolling_window_forecast_sarima(df_model, order, seasonal_order)

    if len(df_forecast) == 0:
        print("  No forecasts generated!")
        return None

    # Calculate metrics on RAW scale (scaled by 100)
    rmse_raw = np.sqrt(np.mean(df_forecast['error'] ** 2)) * 100
    mae_raw = np.mean(df_forecast['abs_error']) * 100
    mape_raw = np.mean(df_forecast['abs_pct_error'])
    coverage_raw = (df_forecast['in_ci'].sum() / len(df_forecast)) * 100

    # Calculate metrics on YOY scale (scaled by 100)
    df_yoy_valid = df_forecast.dropna(subset=['error_yoy'])
    if len(df_yoy_valid) > 0:
        rmse_yoy = np.sqrt(np.mean(df_yoy_valid['error_yoy'] ** 2)) * 100
        mae_yoy = np.mean(df_yoy_valid['abs_error_yoy']) * 100
        coverage_yoy = (df_yoy_valid['in_ci_yoy'].sum() /
                       len(df_yoy_valid)) * 100
        mape_yoy = np.mean(
            np.abs(df_yoy_valid['error_yoy'] /
                   df_yoy_valid['actual_yoy']) * 100
        )
    else:
        rmse_yoy = np.nan
        mae_yoy = np.nan
        mape_yoy = np.nan
        coverage_yoy = np.nan

    print(f"\nForecasted {len(df_forecast)} periods "
          f"({len(df_yoy_valid)} with YoY conversion)")

    print("\n" + "-" * 70)
    print("FORECAST ACCURACY METRICS")
    print("-" * 70)
    print(f"\nRaw Scale (CPI Index Levels, x100):")
    print(f"  RMSE: {rmse_raw:.4f}")
    print(f"  MAE:  {mae_raw:.4f}")
    print(f"  MAPE: {mape_raw:.4f}%")
    print(f"  95% CI Coverage: {coverage_raw:.1f}%")
    print(f"\nYoY Scale (x100):")
    print(f"  RMSE: {rmse_yoy:.4f}")
    print(f"  MAE:  {mae_yoy:.4f}")
    print(f"  95% CI Coverage: {coverage_yoy:.1f}%")

    return {
        'country': country,
        'order': order,
        'seasonal_order': seasonal_order,
        'x1_lag': x1_lag,
        'x2_lag': x2_lag,
        'n_forecasts': len(df_forecast),
        'rmse_raw': rmse_raw,
        'mae_raw': mae_raw,
        'mape_raw': mape_raw,
        'coverage_raw': coverage_raw,
        'rmse_yoy': rmse_yoy,
        'mae_yoy': mae_yoy,
        'mape_yoy': mape_yoy,
        'coverage_yoy': coverage_yoy,
        'df_forecast': df_forecast,
        'df_model': df_model
    }


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_forecast(result, output_dir):
    """
    Plot SARIMA forecast on YoY scale with training data.

    Unified format matching RF and LASSO plots:
    - Training data from 2015 (blue line)
    - Vertical dashed line at forecast start
    - Actual values (green)
    - Forecast values (red dashed)
    - 95% CI shaded area
    - Metrics box (RMSE, MAE, CI Coverage)

    Parameters
    ----------
    result : dict
        Forecast results.
    output_dir : str
        Directory to save plot.
    """
    country = result['country']
    df_model = result['df_model']
    df_forecast = result['df_forecast']
    p, d, q = result['order']
    P, D, Q, m = result['seasonal_order']
    model_str = f"SARIMA({p},{d},{q})({P},{D},{Q})[{m}]"

    # Filter forecast to valid YoY observations
    df_yoy = df_forecast.dropna(subset=['actual_yoy', 'forecast_yoy'])

    if len(df_yoy) == 0:
        print(f"  No YoY data available for {country} plot")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    # First forecast date
    first_forecast = df_yoy['period'].min()
    plot_start = pd.to_datetime(PLOT_START_DATE)

    # Training data: from 2015 to first forecast, with valid YoY
    df_train = df_model[
        (df_model['period'] >= plot_start) &
        (df_model['period'] < first_forecast)
    ].dropna(subset=['yoy'])

    # Plot training data
    ax.plot(
        df_train['period'], df_train['yoy'],
        label='Training Data', color='blue', linewidth=1.5, alpha=0.7
    )

    # Plot actual values in test period
    ax.plot(
        df_yoy['period'], df_yoy['actual_yoy'],
        label='Actual', color='green', linewidth=2, marker='o', markersize=6
    )

    # Plot forecasts
    ax.plot(
        df_yoy['period'], df_yoy['forecast_yoy'],
        label=f'Forecast {model_str}', color='red', linewidth=2,
        linestyle='--', marker='x', markersize=6
    )

    # Confidence interval
    ax.fill_between(
        df_yoy['period'],
        df_yoy['lower_ci_yoy'],
        df_yoy['upper_ci_yoy'],
        color='red', alpha=0.2, label='95% CI'
    )

    # Vertical line at forecast start
    ax.axvline(
        x=first_forecast, color='black', linestyle=':',
        linewidth=1.5, label='Forecast Start'
    )

    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('YoY Log Difference', fontsize=12)
    ax.set_title(f'{country} - SARIMA-X Forecast', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Metrics box (top-left to avoid overlap with legend)
    textstr = (
        f"Model: {model_str}\n"
        f"RMSE: {result['rmse_yoy']:.4f}\n"
        f"MAE:  {result['mae_yoy']:.4f}\n"
        f"95% CI Coverage: {result['coverage_yoy']:.1f}%"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(
        0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, family='monospace'
    )

    plt.tight_layout()

    # Save
    country_filename = country.replace(' ', '_').lower()
    plot_path = os.path.join(output_dir, f'{country_filename}_sarima_forecast.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Plot saved: {plot_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function: SARIMA-X forecasting pipeline.

    Returns
    -------
    tuple
        (forecast_results, df_summary)
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "SARIMA-X FORECASTING PIPELINE")
    print("=" * 70)
    print("\nPipeline:")
    print("  1. Load raw CPI data (index levels)")
    print("  2. Load exogenous variables")
    print("  3. Load optimal lags from previous selection")
    print("  4. AutoARIMA with seasonal=True (BIC criterion)")
    print("  5. Rolling window 1-step-ahead forecasting")
    print("  6. Evaluate metrics (raw and YoY scale)")
    print("  7. Generate plots")

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Load all data
    print("\n" + "-" * 70)
    print("LOADING DATA")
    print("-" * 70)

    cpi_df = load_cpi_data()
    unemp_df = load_unemployment_data()
    import_df = load_import_prices_data()
    optimal_lags = load_optimal_lags()

    # Find common countries
    common_countries = find_common_countries(cpi_df, unemp_df, import_df)

    # Store results
    forecast_results = {}
    summary_stats = []

    # Process each country
    print("\n" + "-" * 70)
    print("PROCESSING COUNTRIES")
    print("-" * 70)

    for country in common_countries:
        try:
            lags = optimal_lags.get(country, {'x1_lag': 0, 'x2_lag': 0})

            result = process_country(
                cpi_df, unemp_df, import_df, country,
                x1_lag=lags['x1_lag'],
                x2_lag=lags['x2_lag']
            )

            if result is None:
                continue

            forecast_results[country] = result

            # Save individual forecasts
            country_filename = country.replace(' ', '_').lower()
            forecast_path = os.path.join(
                OUTPUT_DIR, f'{country_filename}_forecasts_seasonal.csv'
            )
            result['df_forecast'].to_csv(forecast_path, index=False)
            print(f"✓ Forecasts saved: {forecast_path}")

            # Plot forecast
            plot_forecast(result, PLOT_DIR)

            # Summary stats
            p, d, q = result['order']
            P, D, Q, m = result['seasonal_order']
            summary_stats.append({
                'Country': country,
                'p': p, 'd': d, 'q': q,
                'P': P, 'D': D, 'Q': Q, 'm': m,
                'Model': f"({p},{d},{q})({P},{D},{Q})[{m}]",
                'x1_lag': result['x1_lag'],
                'x2_lag': result['x2_lag'],
                'N_forecasts': result['n_forecasts'],
                'RMSE_raw': result['rmse_raw'],
                'MAE_raw': result['mae_raw'],
                'MAPE_raw': result['mape_raw'],
                'CI_Coverage_raw_%': result['coverage_raw'],
                'RMSE_yoy': result['rmse_yoy'],
                'MAE_yoy': result['mae_yoy'],
                'MAPE_yoy': result['mape_yoy'],
                'CI_Coverage_yoy_%': result['coverage_yoy']
            })

        except Exception as e:
            print(f"\n✗ Error processing {country}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Create summary dataframe
    df_summary = pd.DataFrame(summary_stats)

    # Save summary
    summary_path = os.path.join(
        OUTPUT_DIR, 'forecast_accuracy_summary_seasonal.csv')
    df_summary.to_csv(summary_path, index=False)

    # Print final summary
    print("\n" + "=" * 70)
    print("FORECAST ACCURACY SUMMARY (SARIMA-X)")
    print("=" * 70)

    display_cols = ['Country', 'Model', 'RMSE_yoy', 'MAE_yoy',
                    'CI_Coverage_yoy_%']
    print("\n" + df_summary[display_cols].to_string(index=False))

    # Overall statistics
    print("\n" + "-" * 70)
    print("OVERALL STATISTICS")
    print("-" * 70)
    print(f"\nYoY Scale (x100):")
    print(f"  Average RMSE: {df_summary['RMSE_yoy'].mean():.4f}")
    print(f"  Average MAE:  {df_summary['MAE_yoy'].mean():.4f}")
    print(f"  Average CI Coverage: {df_summary['CI_Coverage_yoy_%'].mean():.1f}%")

    # Performance ranking
    print("\n" + "-" * 70)
    print("PERFORMANCE RANKING (by RMSE on YoY Scale)")
    print("-" * 70)
    df_sorted = df_summary.sort_values('RMSE_yoy')
    print(f"\nBest:  {df_sorted.iloc[0]['Country']} "
          f"(RMSE: {df_sorted.iloc[0]['RMSE_yoy']:.4f})")
    print(f"Worst: {df_sorted.iloc[-1]['Country']} "
          f"(RMSE: {df_sorted.iloc[-1]['RMSE_yoy']:.4f})")

    print(f"\n{'=' * 70}")
    print("SARIMA-X FORECASTING COMPLETE")
    print(f"{'=' * 70}")
    print(f"✓ Countries processed: {len(forecast_results)}")
    print(f"✓ Summary saved: {summary_path}")
    print(f"✓ Forecasts saved: {OUTPUT_DIR}")
    print(f"✓ Plots saved: {PLOT_DIR}")
    print("=" * 70)

    return forecast_results, df_summary


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    forecast_results, df_summary = main()
