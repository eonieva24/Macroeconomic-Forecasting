"""
Standard Random Forest for CPI Inflation Forecasting

Implementation following Beck & Wolf (2025) "Forecasting Inflation with the
Hedged Random Forest" methodology, adapted for our 8-country dataset.

Methodology:
1. Target: Log YoY inflation π_t = 100 × (log(CPI_t) - log(CPI_{t-12}))
2. Explicit lag structure (economically meaningful):
   - Inflation: t-1, t-3, t-6, t-12
   - Import prices: t-1, t-3, t-6
   - Unemployment: t-1, t-3, t-6
3. Trend/Momentum feature: 3-month change in inflation (helps RF capture
   directional movements since RF cannot extrapolate beyond training range)
4. Seasonality: Month-of-year feature
5. Macro-appropriate regularization
6. Rolling h=1 forecast (matches SARIMAX)

RF Hyperparameters:
- n_estimators = 500
- max_features = n_features / 3
- min_samples_leaf = 15 (increased from 10 to reduce overfitting)
- max_depth = 6
- random_state = 42

Reference:
Beck, E. and Wolf, M. (2025). "Forecasting Inflation with the Hedged
Random Forest". SNB Working Papers 07/2025.

Author: Elena Onieva Henrich
Date: January 2026
Course: Advanced Programming 2025 - Forecasting Project
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import warnings

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))

# Input paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
CPI_PATH = os.path.join(DATA_DIR, 'CPI_1996_2019.csv')
UNEMPLOYMENT_PATH = os.path.join(DATA_DIR, 'unemployment_1996_2019.csv')
IMPORT_PRICES_PATH = os.path.join(DATA_DIR, 'import_prices_1996_2019.csv')

# Output paths
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'rf_forecast')
FORECAST_DIR = os.path.join(OUTPUT_DIR, 'forecasts')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Countries to process
COUNTRIES = [
    'Belgium', 'Germany', 'Israel', 'Korea',
    'Latvia', 'Lithuania', 'Norway', 'Switzerland'
]

# Random Forest hyperparameters
RF_N_ESTIMATORS = 500
RF_MAX_FEATURES_DIVISOR = 3
RF_MIN_SAMPLES_LEAF = 15  # Increased from 10 to reduce overfitting
RF_MAX_DEPTH = 6
RF_RANDOM_STATE = 42

# Lag configuration
INFLATION_LAGS = [1, 3, 6, 12]
IMPORT_LAGS = [1, 3, 6]
UNEMPLOYMENT_LAGS = [1, 3, 6]

# Trend/Momentum configuration
TREND_WINDOW = 3  # 3-month momentum indicator

# Date configuration
TRAIN_END = '2018-12-01'
TEST_END = '2019-12-01'

# Plot configuration
PLOT_START_DATE = '2015-01-01'

# Confidence interval level
CI_LEVEL = 0.95


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_cpi_data():
    """
    Load CPI data and convert to long format.

    Returns
    -------
    pd.DataFrame
        CPI data with columns: Country, period, cpi_value
    """
    print("Loading CPI data...")
    assert os.path.exists(CPI_PATH), f"File not found: {CPI_PATH}"

    df_wide = pd.read_csv(CPI_PATH)
    df_wide['period'] = pd.to_datetime(df_wide['period'])

    value_cols = [col for col in df_wide.columns if col != 'period']
    df_long = df_wide.melt(
        id_vars=['period'],
        value_vars=value_cols,
        var_name='country_col',
        value_name='cpi_value'
    )

    df_long['Country'] = df_long['country_col'].str.split(' – ').str[0]
    df_long = df_long[['Country', 'period', 'cpi_value']].copy()
    df_long = df_long.dropna(subset=['cpi_value'])

    print(f"  ✓ Loaded CPI data: {len(df_long)} observations")
    print(f"    Countries: {sorted(df_long['Country'].unique())}")

    return df_long


def load_unemployment_data():
    """
    Load unemployment data and convert to long format.

    Returns
    -------
    pd.DataFrame
        Unemployment data with columns: Country, period, unemployment_value
    """
    print("Loading Unemployment data...")
    assert os.path.exists(UNEMPLOYMENT_PATH), \
        f"File not found: {UNEMPLOYMENT_PATH}"

    df_wide = pd.read_csv(UNEMPLOYMENT_PATH)
    df_wide['period'] = pd.to_datetime(df_wide['period'])

    value_cols = [col for col in df_wide.columns if col != 'period']
    df_long = df_wide.melt(
        id_vars=['period'],
        value_vars=value_cols,
        var_name='Country',
        value_name='unemployment_value'
    )
    df_long = df_long.dropna(subset=['unemployment_value'])

    print(f"  ✓ Loaded Unemployment data: {len(df_long)} observations")

    return df_long


def load_import_prices_data():
    """
    Load import prices data and convert to long format.

    Returns
    -------
    pd.DataFrame
        Import prices data with columns: Country, period, import_prices_value
    """
    print("Loading Import Prices data...")
    assert os.path.exists(IMPORT_PRICES_PATH), \
        f"File not found: {IMPORT_PRICES_PATH}"

    df_wide = pd.read_csv(IMPORT_PRICES_PATH)
    df_wide['period'] = pd.to_datetime(df_wide['period'])

    value_cols = [col for col in df_wide.columns if col != 'period']
    df_long = df_wide.melt(
        id_vars=['period'],
        value_vars=value_cols,
        var_name='Country',
        value_name='import_prices_value'
    )
    df_long = df_long.dropna(subset=['import_prices_value'])

    print(f"  ✓ Loaded Import Prices data: {len(df_long)} observations")

    return df_long


# ============================================================================
# DATA TRANSFORMATION FUNCTIONS
# ============================================================================

def compute_log_yoy_inflation(df, value_col, new_col_name):
    """
    Compute Log Year-over-Year inflation.

    Formula: π_t = 100 × (log(CPI_t) - log(CPI_{t-12}))

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with period and value column, sorted by period.
    value_col : str
        Name of the column containing price levels.
    new_col_name : str
        Name for the new inflation column.

    Returns
    -------
    pd.DataFrame
        DataFrame with added log YoY inflation column.
    """
    df = df.sort_values('period').copy()
    df[new_col_name] = 100 * (
        np.log(df[value_col]) - np.log(df[value_col].shift(12))
    )
    return df


def prepare_country_data(cpi_df, unemp_df, import_df, country):
    """
    Prepare dataset for a single country with all transformations.

    Parameters
    ----------
    cpi_df : pd.DataFrame
        CPI data (long format, levels).
    unemp_df : pd.DataFrame
        Unemployment data (long format, percentage).
    import_df : pd.DataFrame
        Import prices data (long format, levels).
    country : str
        Country name.

    Returns
    -------
    pd.DataFrame
        Prepared dataset with target and exogenous variables.
    """
    cpi_country = cpi_df[cpi_df['Country'] == country][
        ['period', 'cpi_value']].copy()
    unemp_country = unemp_df[unemp_df['Country'] == country][
        ['period', 'unemployment_value']].copy()
    import_country = import_df[import_df['Country'] == country][
        ['period', 'import_prices_value']].copy()

    if len(cpi_country) == 0:
        print(f"    Warning: No CPI data for {country}")
        return None
    if len(unemp_country) == 0:
        print(f"    Warning: No Unemployment data for {country}")
        return None
    if len(import_country) == 0:
        print(f"    Warning: No Import Prices data for {country}")
        return None

    cpi_country = cpi_country.sort_values('period')
    unemp_country = unemp_country.sort_values('period')
    import_country = import_country.sort_values('period')

    cpi_country = compute_log_yoy_inflation(
        cpi_country, 'cpi_value', 'inflation_yoy')
    import_country = compute_log_yoy_inflation(
        import_country, 'import_prices_value', 'import_yoy')

    cpi_country = cpi_country[['period', 'inflation_yoy']]
    import_country = import_country[['period', 'import_yoy']]

    df = cpi_country.merge(unemp_country, on='period', how='inner')
    df = df.merge(import_country, on='period', how='inner')

    df = df.rename(columns={
        'inflation_yoy': 'y_t',
        'unemployment_value': 'x1_t',
        'import_yoy': 'x2_t'
    })

    df = df.sort_values('period').reset_index(drop=True)
    df = df.dropna()

    return df


def create_lag_features(df):
    """
    Create explicit lag features, trend indicator, and month-of-year.

    The trend feature (3-month momentum) helps RF capture directional
    movements, since RF cannot extrapolate beyond the training data range.

    Parameters
    ----------
    df : pd.DataFrame
        Data with columns: period, y_t, x1_t, x2_t

    Returns
    -------
    pd.DataFrame
        Data with lag features, trend feature, and month feature added.
    """
    df_lagged = df.copy()

    # Seasonality: month-of-year
    df_lagged['month'] = df_lagged['period'].dt.month

    # Trend/Momentum feature: 3-month change in inflation
    # This helps RF capture directional movements
    df_lagged['trend_3m'] = df_lagged['y_t'].diff(TREND_WINDOW)

    # Inflation lags
    for lag in INFLATION_LAGS:
        df_lagged[f'inflation_lag{lag}'] = df_lagged['y_t'].shift(lag)

    # Import price lags
    for lag in IMPORT_LAGS:
        df_lagged[f'import_lag{lag}'] = df_lagged['x2_t'].shift(lag)

    # Unemployment lags
    for lag in UNEMPLOYMENT_LAGS:
        df_lagged[f'unemp_lag{lag}'] = df_lagged['x1_t'].shift(lag)

    df_lagged = df_lagged.dropna()

    return df_lagged


def get_feature_columns():
    """
    Get list of feature column names.

    Returns
    -------
    list
        List of feature column names.
    """
    features = ['month', 'trend_3m']

    for lag in INFLATION_LAGS:
        features.append(f'inflation_lag{lag}')

    for lag in IMPORT_LAGS:
        features.append(f'import_lag{lag}')

    for lag in UNEMPLOYMENT_LAGS:
        features.append(f'unemp_lag{lag}')

    return features


# ============================================================================
# RANDOM FOREST MODEL
# ============================================================================

def create_rf_model(n_features):
    """
    Create Random Forest model with macro-appropriate hyperparameters.

    Parameters
    ----------
    n_features : int
        Number of features.

    Returns
    -------
    RandomForestRegressor
        Configured RF model.
    """
    max_features = max(1, n_features // RF_MAX_FEATURES_DIVISOR)

    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_features=max_features,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1
    )

    return model


def predict_with_ci(model, X_test, ci_level=CI_LEVEL):
    """
    Get prediction with confidence interval from Random Forest.

    Uses individual tree predictions to compute prediction intervals.

    Parameters
    ----------
    model : RandomForestRegressor
        Fitted RF model.
    X_test : np.ndarray
        Test features (single observation).
    ci_level : float
        Confidence level (default 0.95).

    Returns
    -------
    tuple
        (prediction, lower_ci, upper_ci)
    """
    # Get predictions from all trees
    tree_predictions = np.array([
        tree.predict(X_test) for tree in model.estimators_
    ])

    # Mean prediction
    prediction = np.mean(tree_predictions)

    # Compute quantiles for CI
    alpha = 1 - ci_level
    lower_ci = np.percentile(tree_predictions, 100 * alpha / 2)
    upper_ci = np.percentile(tree_predictions, 100 * (1 - alpha / 2))

    return prediction, lower_ci, upper_ci


# ============================================================================
# ROLLING WINDOW FORECASTING
# ============================================================================

def rolling_window_forecast(df, feature_cols, train_end_date=TRAIN_END,
                            test_end_date=TEST_END):
    """
    Expanding-window rolling forecast with confidence intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with features and target.
    feature_cols : list
        List of feature column names.
    train_end_date : str
        End date of initial training period.
    test_end_date : str
        End date of test period.

    Returns
    -------
    pd.DataFrame
        Forecast results with columns: period, actual, forecast, lower_ci,
        upper_ci
    """
    # Convert strings to datetime : make sure we on the same "metric"
    train_end = pd.to_datetime(train_end_date)
    test_end = pd.to_datetime(test_end_date)
    # Define test period : forecast 2019 JAN-DEC
    test_periods = df[
        (df['period'] > train_end) & (df['period'] <= test_end) # Starts after train
    ]['period'].tolist()
    # Safety in case is empty
    if len(test_periods) == 0:
        print("    Warning: No test periods found")
        return None
    # Based on number of predictors
    results = []
    n_features = len(feature_cols)
    # leak safe : forecast at time t
    for test_date in test_periods:
        test_date = pd.to_datetime(test_date)
        # Training set only uses observations before forecast date
        # No look ahead
        train_df = df[df['period'] < test_date]
        # The test set is the forecasted month
        test_df = df[df['period'] == test_date]
        # Safety : skip if not enough training data
        if len(train_df) < 50:
            print(f"    Warning: Insufficient training data for {test_date}")
            continue
        # Safety : skip if test month is missing
        if len(test_df) == 0:
            continue
        # Matrix and target for training
        X_train = train_df[feature_cols].values
        y_train = train_df['y_t'].values
        # Extract features for each forecast
        X_test = test_df[feature_cols].values
        y_actual = test_df['y_t'].values[0]
        # Pre declared parameters only, fixed
        model = create_rf_model(n_features)
        model.fit(X_train, y_train)

        # Get prediction with CI
        y_pred, lower_ci, upper_ci = predict_with_ci(model, X_test)

        # Check if actual is within CI
        in_ci = (y_actual >= lower_ci) and (y_actual <= upper_ci)
        # Store results
        results.append({
            'period': test_date,
            'actual': y_actual,
            'forecast': y_pred,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'in_ci': in_ci,
            'n_train': len(train_df)
        })
    # Safety , if not enough training data
    if len(results) == 0:
        return None

    return pd.DataFrame(results)


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_metrics(results_df):
    """
    Compute forecast accuracy metrics.

    Parameters
    ----------
    results_df : pd.DataFrame
        Forecast results with 'actual' and 'forecast' columns.

    Returns
    -------
    dict
        Dictionary with RMSE, MAE, and CI coverage.
    """
    actual = results_df['actual'].values
    forecast = results_df['forecast'].values

    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mae = mean_absolute_error(actual, forecast)
    coverage = (results_df['in_ci'].sum() / len(results_df)) * 100

    return {
        'RMSE': rmse,
        'MAE': mae,
        'CI_Coverage': coverage,
        'N_forecasts': len(actual)
    }


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_forecast(result, output_dir):
    """
    Plot forecast results with unified format.

    Shows training data from 2015, actual values, forecasts, 95% CI,
    and metrics box.

    Parameters
    ----------
    result : dict
        Results dictionary.
    output_dir : str
        Directory to save plot.
    """
    country = result['country']
    df_model = result['df_model']
    df_forecast = result['df_forecast']

    fig, ax = plt.subplots(figsize=(14, 7))

    first_forecast = df_forecast['period'].min()
    plot_start = pd.to_datetime(PLOT_START_DATE)

    # Training data from 2015
    df_train = df_model[
        (df_model['period'] >= plot_start) &
        (df_model['period'] < first_forecast)
    ]

    # Plot training data
    ax.plot(
        df_train['period'], df_train['y_t'],
        label='Training Data', color='blue', linewidth=1.5, alpha=0.7
    )

    # Actual values in test period
    ax.plot(
        df_forecast['period'], df_forecast['actual'],
        label='Actual', color='green', linewidth=2, marker='o', markersize=6
    )

    # RF Forecasts
    ax.plot(
        df_forecast['period'], df_forecast['forecast'],
        label='RF Forecast', color='red', linewidth=2,
        linestyle='--', marker='x', markersize=6
    )

    # 95% Confidence Interval
    ax.fill_between(
        df_forecast['period'],
        df_forecast['lower_ci'],
        df_forecast['upper_ci'],
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
    ax.set_title(
        f'{country} - Random Forest Forecast',
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Metrics box (top-left)
    n_features = len(get_feature_columns())
    max_features = max(1, n_features // RF_MAX_FEATURES_DIVISOR)

    textstr = (
        f"Model: Random Forest\n"
        f"n_estimators: {RF_N_ESTIMATORS}\n"
        f"max_features: {max_features}\n"
        f"max_depth: {RF_MAX_DEPTH}\n"
        f"min_samples_leaf: {RF_MIN_SAMPLES_LEAF}\n"
        f"trend_window: {TREND_WINDOW}m\n"
        f"{'─' * 22}\n"
        f"RMSE: {result['rmse']:.6f}\n"
        f"MAE:  {result['mae']:.6f}\n"
        f"95% CI Coverage: {result['coverage']:.1f}%"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(
        0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, family='monospace'
    )

    plt.tight_layout()

    # Save
    country_filename = country.replace(' ', '_').lower()
    plot_path = os.path.join(output_dir, f'{country_filename}_rf_forecast.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Plot saved: {plot_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_rf_forecast():
    """
    Main function to run RF forecasting for all countries.

    Returns
    -------
    pd.DataFrame
        Summary of forecast accuracy for all countries.
    """
    print("\n" + "=" * 70)
    print(" " * 10 + "STANDARD RANDOM FOREST - CPI INFLATION FORECASTING")
    print("=" * 70)
    print("\nMethodology:")
    print("  1. Target: Log YoY inflation")
    print("  2. Explicit lags:")
    print(f"     - Inflation: {INFLATION_LAGS}")
    print(f"     - Import prices: {IMPORT_LAGS}")
    print(f"     - Unemployment: {UNEMPLOYMENT_LAGS}")
    print(f"  3. Trend feature: {TREND_WINDOW}-month momentum")
    print("  4. Seasonality: Month-of-year feature")
    print("  5. Rolling h=1 forecast (matches SARIMAX)")
    print(f"\nRF Hyperparameters:")
    print(f"  - n_estimators: {RF_N_ESTIMATORS}")
    print(f"  - max_features: n_features / {RF_MAX_FEATURES_DIVISOR}")
    print(f"  - max_depth: {RF_MAX_DEPTH}")
    print(f"  - min_samples_leaf: {RF_MIN_SAMPLES_LEAF}")
    print(f"\nEvaluation Window:")
    print(f"  - Training: up to {TRAIN_END}")
    print(f"  - Test window: ({TRAIN_END}, {TEST_END}]")

    # Create output directories
    os.makedirs(FORECAST_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Load data
    print("\n" + "-" * 70)
    print("LOADING DATA")
    print("-" * 70)
    cpi_df = load_cpi_data()
    unemp_df = load_unemployment_data()
    import_df = load_import_prices_data()

    # Get feature columns
    feature_cols = get_feature_columns()
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")

    # Process each country
    print("\n" + "-" * 70)
    print("RUNNING FORECASTS")
    print("-" * 70)

    all_results = []
    all_forecasts = {}

    for country in COUNTRIES:
        print(f"\n{'─' * 50}")
        print(f"Processing: {country}")
        print(f"{'─' * 50}")

        df_country = prepare_country_data(cpi_df, unemp_df, import_df, country)

        if df_country is None or len(df_country) == 0:
            print(f"  Skipping {country}: No data available")
            continue

        print(f"  Data range: {df_country['period'].min().strftime('%Y-%m')} "
              f"to {df_country['period'].max().strftime('%Y-%m')}")
        print(f"  Observations (after YoY): {len(df_country)}")

        df_features = create_lag_features(df_country)
        print(f"  After lagging: {len(df_features)} observations")

        train_end = pd.to_datetime(TRAIN_END)
        test_end = pd.to_datetime(TEST_END)
        test_obs = df_features[
            (df_features['period'] > train_end) &
            (df_features['period'] <= test_end)
        ]

        if len(test_obs) == 0:
            print(f"  Skipping {country}: No test period data")
            continue

        print(f"  Test observations available: {len(test_obs)}")

        results_df = rolling_window_forecast(df_features, feature_cols)

        if results_df is None or len(results_df) == 0:
            print(f"  Skipping {country}: Forecast failed")
            continue

        metrics = compute_metrics(results_df)
        print(f"  Results:")
        print(f"    RMSE:  {metrics['RMSE']:.6f}")
        print(f"    MAE:   {metrics['MAE']:.6f}")
        print(f"    95% CI Coverage: {metrics['CI_Coverage']:.1f}%")
        print(f"    N forecasts: {metrics['N_forecasts']}")

        all_results.append({
            'Country': country,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'CI_Coverage_%': metrics['CI_Coverage'],
            'N_forecasts': metrics['N_forecasts']
        })

        result = {
            'country': country,
            'df_model': df_features,
            'df_forecast': results_df,
            'rmse': metrics['RMSE'],
            'mae': metrics['MAE'],
            'coverage': metrics['CI_Coverage'],
            'n_forecasts': metrics['N_forecasts']
        }

        results_df['Country'] = country
        all_forecasts[country] = result

        # Save country forecasts
        forecast_file = os.path.join(
            FORECAST_DIR,
            f'rf_forecast_{country.lower().replace(" ", "_")}.csv'
        )
        results_df.to_csv(forecast_file, index=False)
        print(f"✓ Forecasts saved: {forecast_file}")

        # Plot
        plot_forecast(result, PLOT_DIR)

    # Create summary
    print("\n" + "-" * 70)
    print("FORECAST SUMMARY")
    print("-" * 70)

    if len(all_results) == 0:
        print("No forecasts completed successfully.")
        return None

    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values('RMSE')

    print("\n" + summary_df.to_string(index=False))

    # Save summary
    summary_file = os.path.join(OUTPUT_DIR, 'rf_forecast_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved: {summary_file}")

    # Save all forecasts combined
    if all_forecasts:
        all_forecast_dfs = [
            result['df_forecast'] for result in all_forecasts.values()
        ]
        all_forecasts_df = pd.concat(all_forecast_dfs, ignore_index=True)
        all_forecasts_file = os.path.join(FORECAST_DIR, 'rf_all_forecasts.csv')
        all_forecasts_df.to_csv(all_forecasts_file, index=False)
        print(f"All forecasts saved: {all_forecasts_file}")

    # Print average metrics
    print("\n" + "-" * 70)
    print("AVERAGE METRICS ACROSS COUNTRIES")
    print("-" * 70)
    print(f"  Mean RMSE:  {summary_df['RMSE'].mean():.6f}")
    print(f"  Mean MAE:   {summary_df['MAE'].mean():.6f}")
    print(f"  Mean 95% CI Coverage: {summary_df['CI_Coverage_%'].mean():.1f}%")

    print("\n" + "=" * 70)
    print("RANDOM FOREST FORECASTING COMPLETE")
    print("=" * 70)

    return summary_df


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    summary = run_rf_forecast()
