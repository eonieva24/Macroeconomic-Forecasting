"""
Rolling Window ARIMA-X Forecasting

Implements rolling window forecasting methodology for ARIMA-X models:
1. Start with training data up to Dec 2018
2. Forecast Jan 2019 (1-step ahead)
3. Roll window forward: add Jan 2019 to training, forecast Feb 2019
4. Continue until Dec 2019

This approach:
- Re-estimates the model at each step with updated data
- Produces 1-step-ahead forecasts for each month in 2019
- Evaluates forecast accuracy using RMSE, MAE, and MAPE

Note: Only countries passing the Ljung-Box diagnostic test are included.
      This ensures residuals are white noise and the model is well-specified.

Reference: MathWorks - Rolling-Window Analysis of Time-Series Models
           Zivot, E., and J. Wang. Modeling Financial Time Series with S_PLUS.

Author: Elena Onieva Henrich
Date: December 2025
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
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

def prepare_lagged_data(df, x1_lag, x2_lag):
    """
    Prepare data with specified lags for exogenous variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns: period, y_t, x1_t, x2_t
    x1_lag : int
        Lag for x1 (Unemployment).
    x2_lag : int
        Lag for x2 (Import Prices).

    Returns
    -------
    pd.DataFrame
        Dataframe with lagged exogenous variables.
    """
    df_model = df.copy()

    # Create lagged x1
    if x1_lag == 0:
        df_model['x1_used'] = df_model['x1_t']
    else:
        df_model['x1_used'] = df_model['x1_t'].shift(x1_lag)

    # Create lagged x2
    if x2_lag == 0:
        df_model['x2_used'] = df_model['x2_t']
    else:
        df_model['x2_used'] = df_model['x2_t'].shift(x2_lag)

    # Drop rows with missing values
    df_model = df_model.dropna()

    return df_model


# ============================================================================
# ROLLING WINDOW FORECASTING
# ============================================================================

def rolling_window_forecast(df, p, d, q, train_end_date='2018-12-01', test_end_date='2019-12-01'):
    """
    Perform rolling window 1-step-ahead forecasting.

    For each period in the test set:
    1. Estimate model on all available data up to that point
    2. Forecast 1 step ahead
    3. Roll window forward by adding actual observation to training

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with y_t, x1_used, x2_used.
    p : int
        AR order.
    d : int
        Integration order.
    q : int
        MA order.
    train_end_date : str
        Initial training end date (default: '2018-12-01').
    test_end_date : str
        Last date to forecast (default: '2019-12-01').

    Returns
    -------
    pd.DataFrame
        Forecast results with actual, forecast, and errors.
    """
    train_end = pd.to_datetime(train_end_date)
    test_end = pd.to_datetime(test_end_date)

    # Get test periods (dates to forecast)
    test_periods = df[(df['period'] > train_end) & (df['period'] <= test_end)]['period'].values

    # Store results
    results = []

    for i, forecast_date in enumerate(test_periods):
        # Training data: all observations up to (but not including) forecast_date
        df_train = df[df['period'] < forecast_date].copy()

        # Get the actual observation to forecast
        df_test_row = df[df['period'] == forecast_date].copy()

        if len(df_test_row) == 0:
            continue

        try:
            # Fit model on training data
            model = ARIMA(
                endog=df_train['y_t'],
                exog=df_train[['x1_used', 'x2_used']],
                order=(p, d, q)
            )
            fitted = model.fit()

            # Get exogenous values for forecast period
            exog_forecast = df_test_row[['x1_used', 'x2_used']].values

            # 1-step ahead forecast with confidence interval
            forecast_obj = fitted.get_forecast(steps=1, exog=exog_forecast)
            forecast_value = forecast_obj.predicted_mean.values[0]

            # 95% confidence interval
            conf_int = forecast_obj.conf_int(alpha=0.05)
            lower_ci = conf_int.iloc[0, 0]
            upper_ci = conf_int.iloc[0, 1]

            # Actual value
            actual_value = df_test_row['y_t'].values[0]

            # Forecast error
            error = actual_value - forecast_value

            # Check if actual is within CI
            in_ci = (actual_value >= lower_ci) and (actual_value <= upper_ci)

            results.append({
                'period': pd.Timestamp(forecast_date),
                'actual': actual_value,
                'forecast': forecast_value,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'error': error,
                'abs_error': abs(error),
                'pct_error': abs(error / actual_value) * 100 if actual_value != 0 else np.nan,
                'in_ci': in_ci,
                'n_train': len(df_train)
            })

        except Exception as e:
            print(f"    Warning: Failed to forecast {pd.Timestamp(forecast_date).strftime('%Y-%m')}: {str(e)}")
            continue

    return pd.DataFrame(results)


def forecast_country(df, p, d, q, x1_lag, x2_lag, country_name):
    """
    Run rolling window forecast for a single country.

    Parameters
    ----------
    df : pd.DataFrame
        Raw country data with y_t, x1_t, x2_t.
    p : int
        AR order.
    d : int
        Integration order.
    q : int
        MA order.
    x1_lag : int
        Lag for x1 (Unemployment).
    x2_lag : int
        Lag for x2 (Import Prices).
    country_name : str
        Country name.

    Returns
    -------
    dict
        Forecast results and metrics.
    """
    print(f"\n{'=' * 70}")
    print(f"ROLLING WINDOW FORECAST: {country_name}")
    print(f"{'=' * 70}")
    print(f"Model: ARIMA({p},{d},{q})")
    print(f"Exogenous lags: x1_lag={x1_lag}, x2_lag={x2_lag}")

    # Prepare data with lags
    df_model = prepare_lagged_data(df, x1_lag, x2_lag)

    print(f"Total observations: {len(df_model)}")
    print(f"Date range: {df_model['period'].min().strftime('%Y-%m')} to {df_model['period'].max().strftime('%Y-%m')}")

    # Run rolling window forecast
    print("\nRunning rolling window forecasts (1-step ahead)...")
    df_forecast = rolling_window_forecast(df_model, p, d, q)

    if len(df_forecast) == 0:
        print("No forecasts generated!")
        return None

    # Calculate metrics
    rmse = np.sqrt(np.mean(df_forecast['error'] ** 2))
    mae = np.mean(df_forecast['abs_error'])
    mape = np.mean(df_forecast['pct_error'])
    coverage = (df_forecast['in_ci'].sum() / len(df_forecast)) * 100

    print(f"\nForecasted {len(df_forecast)} periods")
    print(
        f"Test period: {df_forecast['period'].min().strftime('%Y-%m')} to {df_forecast['period'].max().strftime('%Y-%m')}")

    print("\n" + "-" * 70)
    print("FORECAST ACCURACY METRICS")
    print("-" * 70)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"95% CI Coverage: {coverage:.1f}% ({df_forecast['in_ci'].sum()}/{len(df_forecast)} observations)")

    return {
        'country': country_name,
        'order': (p, d, q),
        'x1_lag': x1_lag,
        'x2_lag': x2_lag,
        'n_forecasts': len(df_forecast),
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'coverage': coverage,
        'df_forecast': df_forecast,
        'df_model': df_model
    }


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_rolling_forecast(result, output_dir):
    """
    Plot rolling window forecast results with confidence intervals.

    Parameters
    ----------
    result : dict
        Forecast results from forecast_country().
    output_dir : str
        Directory to save plot.
    """
    country = result['country']
    df_model = result['df_model']
    df_forecast = result['df_forecast']
    p, d, q = result['order']

    fig, ax = plt.subplots(figsize=(14, 7))

    # Get training data (before first forecast date)
    first_forecast = df_forecast['period'].min()
    df_train = df_model[df_model['period'] < first_forecast]

    # Plot training data (last 36 months for clarity)
    train_display = df_train.tail(36)
    ax.plot(train_display['period'], train_display['y_t'],
            label='Training Data', color='blue', linewidth=1.5, alpha=0.7)

    # Plot actual test values
    ax.plot(df_forecast['period'], df_forecast['actual'],
            label='Actual', color='green', linewidth=2, marker='o', markersize=6)

    # Plot rolling forecasts
    ax.plot(df_forecast['period'], df_forecast['forecast'],
            label=f'ARIMAX({p},{d},{q}) Forecast', color='red', linewidth=2,
            linestyle='--', marker='x', markersize=6)

    # Plot 95% confidence interval
    ax.fill_between(df_forecast['period'],
                    df_forecast['lower_ci'],
                    df_forecast['upper_ci'],
                    color='red', alpha=0.2, label='95% Confidence Interval')

    # Add vertical line at train-test split
    ax.axvline(x=first_forecast, color='black', linestyle=':',
               linewidth=1.5, label='Forecast Start')

    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('YoY CPI Inflation', fontsize=12)
    ax.set_title(f'{country} - Rolling Window ARIMAX({p},{d},{q}) Forecast\n(Grid Search Parameters)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add metrics text box
    textstr = (f"RMSE: {result['rmse']:.4f}\n"
               f"MAE:  {result['mae']:.4f}\n"
               f"MAPE: {result['mape']:.2f}%\n"
               f"CI Coverage: {result['coverage']:.1f}%")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save plot
    country_filename = country.replace(' ', '_').lower()
    plot_path = os.path.join(output_dir, f'{country_filename}_forecast.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Plot saved to: {plot_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to run rolling window forecasts for all countries.

    Only countries passing the Ljung-Box diagnostic test are included.
    This ensures the ARIMA model residuals are white noise (no autocorrelation).

    Steps:
    1. Load optimal model parameters from grid search
    2. Filter to countries passing Ljung-Box test
    3. For each country, run rolling window forecast
    4. Calculate accuracy metrics
    5. Save results and plots

    Returns
    -------
    tuple
        (forecast_results, df_summary)
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "ROLLING WINDOW ARIMAX FORECASTING")
    print("=" * 70)
    print("\nMethodology: Rolling window with 1-step-ahead forecasts")
    print("Initial training: Up to December 2018")
    print("Test period: January 2019 to December 2019 (12 forecasts)")
    print("Window rolls forward by 1 month after each forecast")
    print("Metrics: RMSE, MAE, MAPE, CI Coverage")
    print("\nNote: Only countries passing Ljung-Box test are included")

    # Define paths
    MODELS_PATH = os.path.join(PROJECT_ROOT, 'results', 'arimax_forecast', 'parameters',
                               'arimax_grid_search_results.csv')
    DATA_DIR = os.path.join(PROJECT_ROOT, 'results', 'arimax_forecast', 'arimax')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'arimax_forecast', 'forecasts')
    PLOT_DIR = os.path.join(PROJECT_ROOT, 'results', 'arimax_forecast', 'plots', 'forecasts')

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Load optimal models from grid search
    print("\nLoading optimal model parameters from grid search...")
    assert os.path.exists(MODELS_PATH), f"File not found: {MODELS_PATH}"
    models_df = pd.read_csv(MODELS_PATH)
    print(f"✓ Loaded parameters for {len(models_df)} countries")

    # Filter to countries passing Ljung-Box test
    models_df_filtered = models_df[models_df['LB_Pass']].copy()

    excluded_countries = models_df[~models_df['LB_Pass']]['Country'].tolist()

    print(f"\n{'─' * 70}")
    print("LJUNG-BOX FILTER")
    print(f"{'─' * 70}")
    print(f"Countries passing Ljung-Box test: {len(models_df_filtered)}/{len(models_df)}")
    print(f"Included: {', '.join(models_df_filtered['Country'].tolist())}")

    if len(excluded_countries) > 0:
        print(f"Excluded (residual autocorrelation): {', '.join(excluded_countries)}")

    # Store results
    forecast_results = {}
    summary_stats = []

    # Forecast each country that passes diagnostic
    for idx, row in models_df_filtered.iterrows():
        country = row['Country']
        p = int(row['p'])
        d = int(row['d'])
        q = int(row['q'])
        x1_lag = int(row['x1_lag'])
        x2_lag = int(row['x2_lag'])

        # Load country data
        country_filename = country.replace(' ', '_').lower()
        data_path = os.path.join(DATA_DIR, f'{country_filename}_arimax.csv')

        assert os.path.exists(data_path), f"File not found: {data_path}"
        df = pd.read_csv(data_path)
        df['period'] = pd.to_datetime(df['period'])

        try:
            # Run rolling window forecast
            result = forecast_country(df, p, d, q, x1_lag, x2_lag, country)

            if result is None:
                continue

            # Store results
            forecast_results[country] = result

            # Save individual forecast results
            forecast_path = os.path.join(OUTPUT_DIR, f'{country_filename}_forecasts.csv')
            result['df_forecast'].to_csv(forecast_path, index=False)
            print(f"✓ Forecasts saved to: {forecast_path}")

            # Plot
            plot_rolling_forecast(result, PLOT_DIR)

            # Store summary
            summary_stats.append({
                'Country': country,
                'Model': f"ARIMA({p},{d},{q})",
                'p': p,
                'd': d,
                'q': q,
                'x1_lag': x1_lag,
                'x2_lag': x2_lag,
                'N_forecasts': result['n_forecasts'],
                'RMSE': result['rmse'],
                'MAE': result['mae'],
                'MAPE': result['mape'],
                'CI_Coverage_%': result['coverage']
            })

        except Exception as e:
            print(f"\n✗ Error forecasting {country}: {str(e)}")
            continue

    # Create summary dataframe
    df_summary = pd.DataFrame(summary_stats)

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, 'forecast_accuracy_summary.csv')
    df_summary.to_csv(summary_path, index=False)

    # Print final summary
    print("\n" + "=" * 70)
    print("FORECAST ACCURACY SUMMARY (ARIMAX - Grid Search)")
    print("=" * 70)

    display_cols = ['Country', 'Model', 'RMSE', 'MAE', 'MAPE', 'CI_Coverage_%']
    print("\n" + df_summary[display_cols].to_string(index=False))

    # Overall statistics
    print("\n" + "-" * 70)
    print("OVERALL STATISTICS")
    print("-" * 70)
    print(f"Countries forecasted: {len(df_summary)}")
    print(f"Average RMSE: {df_summary['RMSE'].mean():.4f}")
    print(f"Average MAE:  {df_summary['MAE'].mean():.4f}")
    print(f"Average MAPE: {df_summary['MAPE'].mean():.2f}%")
    print(f"Average CI Coverage: {df_summary['CI_Coverage_%'].mean():.1f}%")

    # Best and worst performers
    print("\n" + "-" * 70)
    print("PERFORMANCE RANKING (by MAPE)")
    print("-" * 70)
    df_sorted = df_summary.sort_values('MAPE')
    print(f"\nBest:  {df_sorted.iloc[0]['Country']} (MAPE: {df_sorted.iloc[0]['MAPE']:.2f}%)")
    print(f"Worst: {df_sorted.iloc[-1]['Country']} (MAPE: {df_sorted.iloc[-1]['MAPE']:.2f}%)")

    # Model distribution
    print("\n" + "-" * 70)
    print("MODEL DISTRIBUTION")
    print("-" * 70)
    model_counts = df_summary['Model'].value_counts()
    for model, count in model_counts.items():
        countries_with_model = df_summary[df_summary['Model'] == model]['Country'].tolist()
        print(f"{model}: {count} - {', '.join(countries_with_model)}")

    print(f"\n{'=' * 70}")
    print("ROLLING WINDOW FORECASTING COMPLETE")
    print(f"{'=' * 70}")
    print(f"✓ Countries forecasted: {len(forecast_results)}")
    print(f"✓ Countries excluded (Ljung-Box fail): {len(excluded_countries)}")
    print(f"✓ Summary saved to: {summary_path}")
    print(f"✓ Individual forecasts saved to: {OUTPUT_DIR}")
    print(f"✓ Plots saved to: {PLOT_DIR}")
    print("=" * 70)

    return forecast_results, df_summary


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    forecast_results, df_summary = main()
