"""
High-Dimensional Random Forest for Germany CPI Inflation Forecasting

This script implements Random Forest forecasting for Germany using a rich
information set with many macroeconomic indicators, following the methodology
of Beck & Wolf (2025) "Forecasting Inflation with the Hedged Random Forest".

Design Principles:
- Rich information set: Many predictors (prices, labor, financial, external)
- All predictors are strictly lagged (no contemporaneous variables)
- Trend/Momentum feature: 3-month change in inflation (helps RF capture
  directional movements since RF cannot extrapolate beyond training range)
- Rolling h=1 expanding window evaluation (matches SARIMAX exactly)
- RF regularization appropriate for macroeconomic data

Input:
Pre-processed Germany feature dataset:
    data/processed/GER_indicators/GER_features_1996_2019.csv

Output:
results/rf_forecast/GER/
├── forecasts/rf_ger_highdim_forecasts.csv
├── summary/rf_ger_highdim_summary.csv
└── plots/rf_ger_highdim_forecast.png

Reference:
Beck, E. and Wolf, M. (2025). "Forecasting Inflation with the Hedged
Random Forest". SNB Working Papers 07/2025.

Author: Elena Onieva Henrich
Date: January 2026
Course: Advanced Programming 2025 - Forecasting Project
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Input path
GER_DATA_PATH = (PROJECT_ROOT / 'data' / 'processed' /
                 'GER_indicators' / 'GER_features_1996_2019.csv')

# Output paths
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'rf_forecast' / 'GER'
FORECAST_DIR = OUTPUT_DIR / 'forecasts'
SUMMARY_DIR = OUTPUT_DIR / 'summary'
PLOT_DIR = OUTPUT_DIR / 'plots'


# =============================================================================
# CONFIGURATION
# =============================================================================

# Target column
TARGET_COL = 'inflation_yoy'

# Date configuration
TRAIN_END = '2018-12-01'
TEST_END = '2019-12-01'

# Plot configuration
PLOT_START_DATE = '2015-01-01'

# Random Forest hyperparameters
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = 6
RF_MIN_SAMPLES_LEAF = 15  # Increased from 10 to reduce overfitting
RF_MAX_FEATURES_DIVISOR = 3
RF_RANDOM_STATE = 42

# Trend/Momentum configuration
TREND_WINDOW = 3  # 3-month momentum indicator

# Confidence interval level
CI_LEVEL = 0.95


# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================

def load_germany_data():
    """
    Load pre-processed Germany feature dataset.

    Returns
    -------
    pd.DataFrame
        Germany dataset ready for modeling.
    """
    print("Loading Germany feature dataset...")

    assert GER_DATA_PATH.exists(), f"Data file not found: {GER_DATA_PATH}"

    df = pd.read_csv(GER_DATA_PATH)
    df['period'] = pd.to_datetime(df['period'])
    df = df.sort_values('period').reset_index(drop=True)

    print(f"  ✓ Loaded: {len(df)} observations")
    print(f"  ✓ Columns: {len(df.columns)}")
    print(f"  ✓ Date range: {df['period'].min().strftime('%Y-%m')} to "
          f"{df['period'].max().strftime('%Y-%m')}")

    return df


def add_trend_feature(df):
    """
    Add trend/momentum feature to help RF capture directional movements.

    The trend feature (3-month momentum) helps RF capture directional
    movements, since RF cannot extrapolate beyond the training data range.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with TARGET_COL column.

    Returns
    -------
    pd.DataFrame
        Dataset with trend_3m feature added.
    """
    print(f"\nAdding trend feature (trend_3m = {TREND_WINDOW}-month momentum)...")

    df = df.copy()
    df['trend_3m'] = df[TARGET_COL].diff(TREND_WINDOW)

    # Drop rows with NaN from diff operation
    n_before = len(df)
    df = df.dropna(subset=['trend_3m']).reset_index(drop=True)
    n_after = len(df)

    print(f"  ✓ Added trend_3m feature")
    print(f"  ✓ Dropped {n_before - n_after} rows due to differencing")

    return df


def get_feature_columns(df):
    """
    Extract feature column names dynamically.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with all columns.

    Returns
    -------
    list
        List of feature column names.
    """
    exclude = ['period', TARGET_COL]
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols


def validate_no_leakage(feature_cols):
    """
    Validate that all features are properly lagged.

    Parameters
    ----------
    feature_cols : list
        List of feature column names.
    """
    print("\nValidating feature columns (no leakage check)...")

    # Allowed columns without '_lag' suffix
    allowed_without_lag = ['month', 'trend_3m']

    suspicious = []
    for col in feature_cols:
        if col in allowed_without_lag:
            continue
        if '_lag' in col:
            continue
        suspicious.append(col)

    if suspicious:
        print(f"  ⚠ WARNING: Suspicious columns found:")
        for col in suspicious:
            print(f"      - {col}")
        raise AssertionError(
            f"Potential leakage: {len(suspicious)} columns without '_lag' suffix."
        )

    print(f"  ✓ All {len(feature_cols)} features are properly lagged or allowed")


def validate_no_missing(df, feature_cols):
    """
    Validate that there are no missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    feature_cols : list
        List of feature column names.
    """
    print("\nValidating no missing values...")

    cols_to_check = feature_cols + [TARGET_COL]
    missing_count = df[cols_to_check].isna().sum().sum()

    assert missing_count == 0, f"Found {missing_count} missing values"

    print(f"  ✓ No missing values in {len(cols_to_check)} columns")


def validate_test_window(df):
    """
    Validate that the test window exists.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'period' column.

    Returns
    -------
    list
        List of test period timestamps.
    """
    print("\nValidating test window...")

    train_end = pd.to_datetime(TRAIN_END)
    test_end = pd.to_datetime(TEST_END)

    test_mask = (df['period'] > train_end) & (df['period'] <= test_end)
    test_periods = df[test_mask]['period'].tolist()

    assert len(test_periods) > 0, (
        f"No test periods found in ({TRAIN_END}, {TEST_END}]."
    )

    print(f"  ✓ Test window: {min(test_periods).strftime('%Y-%m')} to "
          f"{max(test_periods).strftime('%Y-%m')}")
    print(f"  ✓ Test observations: {len(test_periods)}")

    return test_periods


# =============================================================================
# RANDOM FOREST MODEL
# =============================================================================

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
    tree_predictions = np.array([
        tree.predict(X_test) for tree in model.estimators_
    ])

    prediction = np.mean(tree_predictions)

    alpha = 1 - ci_level
    lower_ci = np.percentile(tree_predictions, 100 * alpha / 2)
    upper_ci = np.percentile(tree_predictions, 100 * (1 - alpha / 2))

    return prediction, lower_ci, upper_ci


# =============================================================================
# ROLLING WINDOW FORECASTING
# =============================================================================

def rolling_window_forecast(df, feature_cols):
    """
    Rolling h=1 expanding window forecast with confidence intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with features and target.
    feature_cols : list
        List of feature column names.

    Returns
    -------
    pd.DataFrame
        Forecast results.
    """
    train_end = pd.to_datetime(TRAIN_END)
    test_end = pd.to_datetime(TEST_END)

    test_periods = df[
        (df['period'] > train_end) & (df['period'] <= test_end)
    ]['period'].tolist()

    print(f"\nRolling forecast: {len(test_periods)} test periods")
    print(f"  Test window: ({TRAIN_END}, {TEST_END}]")

    n_features = len(feature_cols)
    results = []

    for i, test_date in enumerate(test_periods):
        test_date = pd.to_datetime(test_date)

        train_df = df[df['period'] < test_date]
        test_df = df[df['period'] == test_date]

        if len(train_df) < 50:
            print(f"  Warning: Insufficient training data for "
                  f"{test_date.strftime('%Y-%m')}")
            continue

        if len(test_df) == 0:
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df[TARGET_COL].values
        X_test = test_df[feature_cols].values
        y_actual = test_df[TARGET_COL].values[0]

        model = create_rf_model(n_features)
        model.fit(X_train, y_train)

        y_pred, lower_ci, upper_ci = predict_with_ci(model, X_test)

        in_ci = (y_actual >= lower_ci) and (y_actual <= upper_ci)

        results.append({
            'period': test_date,
            'actual': y_actual,
            'forecast': y_pred,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'in_ci': in_ci,
            'error': y_actual - y_pred,
            'n_train': len(train_df)
        })

        if (i + 1) % 4 == 0 or (i + 1) == len(test_periods):
            print(f"  Completed {i + 1}/{len(test_periods)} forecasts")

    return pd.DataFrame(results)


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def compute_metrics(results_df):
    """
    Compute forecast accuracy metrics.

    RMSE and MAE are scaled by 100 for consistency with other models.

    Parameters
    ----------
    results_df : pd.DataFrame
        Forecast results.

    Returns
    -------
    dict
        Dictionary with metrics.
    """
    actual = results_df['actual'].values
    forecast = results_df['forecast'].values

    rmse = np.sqrt(mean_squared_error(actual, forecast)) * 100
    mae = mean_absolute_error(actual, forecast) * 100
    coverage = (results_df['in_ci'].sum() / len(results_df)) * 100

    return {
        'RMSE': rmse,
        'MAE': mae,
        'CI_Coverage': coverage,
        'N_forecasts': len(actual)
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_forecast(df, results_df, feature_cols, metrics, output_path):
    """
    Plot forecast results with unified format.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    results_df : pd.DataFrame
        Forecast results.
    feature_cols : list
        Feature column names.
    metrics : dict
        Computed metrics.
    output_path : Path
        Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    first_forecast = results_df['period'].min()
    plot_start = pd.to_datetime(PLOT_START_DATE)

    # Training data from 2015
    df_train = df[
        (df['period'] >= plot_start) &
        (df['period'] < first_forecast)
    ]

    # Plot training data
    ax.plot(
        df_train['period'], df_train[TARGET_COL],
        label='Training Data', color='blue', linewidth=1.5, alpha=0.7
    )

    # Actual values
    ax.plot(
        results_df['period'], results_df['actual'],
        label='Actual', color='green', linewidth=2, marker='o', markersize=6
    )

    # RF Forecasts
    ax.plot(
        results_df['period'], results_df['forecast'],
        label='RF Forecast', color='red', linewidth=2,
        linestyle='--', marker='x', markersize=6
    )

    # 95% Confidence Interval
    ax.fill_between(
        results_df['period'],
        results_df['lower_ci'],
        results_df['upper_ci'],
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
    ax.set_title('Germany - Random Forest Forecast', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Metrics box (top-left)
    n_features = len(feature_cols)
    max_features = max(1, n_features // RF_MAX_FEATURES_DIVISOR)

    textstr = (
        f"Model: Random Forest (High-Dim)\n"
        f"n_features: {n_features}\n"
        f"n_estimators: {RF_N_ESTIMATORS}\n"
        f"max_features: {max_features}\n"
        f"max_depth: {RF_MAX_DEPTH}\n"
        f"min_samples_leaf: {RF_MIN_SAMPLES_LEAF}\n"
        f"trend_window: {TREND_WINDOW}m\n"
        f"{'─' * 22}\n"
        f"RMSE: {metrics['RMSE']:.4f}\n"
        f"MAE:  {metrics['MAE']:.4f}\n"
        f"95% CI Coverage: {metrics['CI_Coverage']:.1f}%"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(
        0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, family='monospace'
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Plot saved: {output_path}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function: Germany high-dimensional RF forecasting pipeline.

    Returns
    -------
    tuple
        (results_df, metrics)
    """
    print("\n" + "=" * 70)
    print(" " * 5 + "GERMANY HIGH-DIMENSIONAL RANDOM FOREST FORECASTING")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Input file:   {GER_DATA_PATH}")
    print(f"  Output dir:   {OUTPUT_DIR}")
    print(f"  Train end:    {TRAIN_END}")
    print(f"  Test end:     {TEST_END}")

    print(f"\nRF Hyperparameters:")
    print(f"  n_estimators:     {RF_N_ESTIMATORS}")
    print(f"  max_depth:        {RF_MAX_DEPTH}")
    print(f"  min_samples_leaf: {RF_MIN_SAMPLES_LEAF}")
    print(f"  max_features:     n_features / {RF_MAX_FEATURES_DIVISOR}")
    print(f"  trend_window:     {TREND_WINDOW}m")

    # Create output directories
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    print("\n" + "-" * 70)
    print("STEP 1: LOADING DATA")
    print("-" * 70)

    df = load_germany_data()

    # Step 2: Add trend feature
    print("\n" + "-" * 70)
    print("STEP 2: ADDING TREND FEATURE")
    print("-" * 70)

    df = add_trend_feature(df)

    # Step 3: Extract and validate features
    print("\n" + "-" * 70)
    print("STEP 3: FEATURE EXTRACTION AND VALIDATION")
    print("-" * 70)

    feature_cols = get_feature_columns(df)
    print(f"\nExtracted {len(feature_cols)} features:")
    for i, col in enumerate(feature_cols):
        print(f"  {i+1:2d}. {col}")

    validate_no_leakage(feature_cols)
    validate_no_missing(df, feature_cols)
    validate_test_window(df)

    # Step 4: Rolling forecast
    print("\n" + "-" * 70)
    print("STEP 4: ROLLING WINDOW FORECAST")
    print("-" * 70)

    results_df = rolling_window_forecast(df, feature_cols)

    if results_df is None or len(results_df) == 0:
        print("ERROR: No forecasts generated!")
        return None, None

    # Step 5: Compute metrics
    print("\n" + "-" * 70)
    print("STEP 5: EVALUATION METRICS")
    print("-" * 70)

    metrics = compute_metrics(results_df)

    print(f"\nForecast Accuracy (n={metrics['N_forecasts']}, x100 scale):")
    print(f"  RMSE:  {metrics['RMSE']:.4f}")
    print(f"  MAE:   {metrics['MAE']:.4f}")
    print(f"  95% CI Coverage: {metrics['CI_Coverage']:.1f}%")

    # Step 6: Save outputs
    print("\n" + "-" * 70)
    print("STEP 6: SAVING OUTPUTS")
    print("-" * 70)

    # Save forecasts
    forecast_path = FORECAST_DIR / 'rf_ger_highdim_forecasts.csv'
    results_df.to_csv(forecast_path, index=False)
    print(f"  ✓ Forecasts saved: {forecast_path}")

    # Save summary
    summary_data = {
        'Country': ['Germany'],
        'Model': ['RF_HighDim'],
        'N_features': [len(feature_cols)],
        'N_forecasts': [metrics['N_forecasts']],
        'RMSE': [metrics['RMSE']],
        'MAE': [metrics['MAE']],
        'CI_Coverage_%': [metrics['CI_Coverage']],
        'Train_end': [TRAIN_END],
        'Test_end': [TEST_END],
        'RF_n_estimators': [RF_N_ESTIMATORS],
        'RF_max_depth': [RF_MAX_DEPTH],
        'RF_min_samples_leaf': [RF_MIN_SAMPLES_LEAF],
        'Trend_window': [TREND_WINDOW]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_path = SUMMARY_DIR / 'rf_ger_highdim_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✓ Summary saved: {summary_path}")

    # Save plot
    plot_path = PLOT_DIR / 'rf_ger_highdim_forecast.png'
    plot_forecast(df, results_df, feature_cols, metrics, plot_path)

    # Final summary
    print("\n" + "=" * 70)
    print("GERMANY HIGH-DIMENSIONAL RF FORECAST COMPLETE")
    print("=" * 70)
    print(f"\nResults Summary (x100 scale):")
    print(f"  Features used:    {len(feature_cols)}")
    print(f"  Test window:      ({TRAIN_END}, {TEST_END}]")
    print(f"  Forecasts:        {metrics['N_forecasts']}")
    print(f"  RMSE:             {metrics['RMSE']:.4f}")
    print(f"  MAE:              {metrics['MAE']:.4f}")
    print(f"  95% CI Coverage:  {metrics['CI_Coverage']:.1f}%")
    print(f"\nOutputs:")
    print(f"  {forecast_path}")
    print(f"  {summary_path}")
    print(f"  {plot_path}")
    print("=" * 70)

    return results_df, metrics


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    results, metrics = main()
