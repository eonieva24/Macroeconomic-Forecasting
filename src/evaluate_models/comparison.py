"""
Final Comparison of Forecasting Methods
=======================================

This script produces three comparison tables:

    TABLE A: Cross-Country Comparison (8 countries)
        - SARIMAX vs Random Forest vs LASSO
        - Best model per country

    TABLE B: Germany Model Comparison (Germany only)
        - SARIMAX vs LASSO vs RF (2 features) vs RF (high-dimensional)
        - Which model performs best for Germany?

    TABLE C: Germany RF Comparison (Germany only)
        - RF (2 features) vs RF (high-dimensional)
        - Does adding indicators improve RF for Germany?

Methodology:
    - All models evaluated on year-on-year (YoY) inflation
    - All RMSE/MAE

Author: Elena Onieva Henrich
Course: Advanced Programming 2025
University: Université de Lausanne
"""

import warnings
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# =============================================================================
# PATH CONFIGURATION (relative paths only)
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Input paths - model summary files
SARIMAX_SUMMARY = (
    PROJECT_ROOT / 'results' / 'arimax_forecast' / 'forecasts_seasonal' /
    'forecast_accuracy_summary_seasonal.csv'
)
RF_SUMMARY = PROJECT_ROOT / 'results' / 'rf_forecast' / 'rf_forecast_summary.csv'
LASSO_SUMMARY = (
    PROJECT_ROOT / 'results' / 'lasso_forecast' / 'summary' / 'lasso_summary.csv'
)
RF_GER_HIGHDIM_SUMMARY = (
    PROJECT_ROOT / 'results' / 'rf_forecast' / 'GER' / 'summary' /
    'rf_ger_highdim_summary.csv'
)

# Output directory
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'comparison'


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_sarimax_results(filepath):
    """
    Load SARIMAX forecast results.

    Parameters
    ----------
    filepath : Path
        Path to SARIMAX summary CSV.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns: Country, RMSE, MAE.
    """
    if not filepath.exists():
        print(f"  WARNING: SARIMAX results not found at {filepath}")
        return None

    df = pd.read_csv(filepath)

    # Verify expected columns exist
    if 'RMSE_yoy' not in df.columns:
        print(f"  ERROR: SARIMAX file missing 'RMSE_yoy' column")
        print(f"  Available columns: {df.columns.tolist()}")
        return None

    # Select and rename columns
    df_out = df[['Country', 'RMSE_yoy', 'MAE_yoy']].copy()
    df_out = df_out.rename(columns={'RMSE_yoy': 'RMSE', 'MAE_yoy': 'MAE'})

    print(f"  Loaded SARIMAX: {len(df_out)} countries")
    print(f"    RMSE range: {df_out['RMSE'].min():.4f} - {df_out['RMSE'].max():.4f}")
    return df_out


def load_rf_results(filepath):
    """
    Load Random Forest (2-feature) forecast results.

    Parameters
    ----------
    filepath : Path
        Path to RF summary CSV.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns: Country, RMSE, MAE.
    """
    if not filepath.exists():
        print(f"  WARNING: RF results not found at {filepath}")
        return None

    df = pd.read_csv(filepath)

    # Verify expected columns
    if 'RMSE' not in df.columns or 'Country' not in df.columns:
        print(f"  ERROR: RF file missing expected columns")
        print(f"  Available columns: {df.columns.tolist()}")
        return None

    df_out = df[['Country', 'RMSE', 'MAE']].copy()

    print(f"  Loaded RF (2-feature): {len(df_out)} countries")
    print(f"    RMSE range: {df_out['RMSE'].min():.4f} - {df_out['RMSE'].max():.4f}")
    return df_out


def load_lasso_results(filepath):
    """
    Load LASSO forecast results.

    Parameters
    ----------
    filepath : Path
        Path to LASSO summary CSV.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns: Country, RMSE, MAE.
    """
    if not filepath.exists():
        print(f"  WARNING: LASSO results not found at {filepath}")
        return None

    df = pd.read_csv(filepath)

    # Verify expected columns
    if 'RMSE' not in df.columns or 'Country' not in df.columns:
        print(f"  ERROR: LASSO file missing expected columns")
        print(f"  Available columns: {df.columns.tolist()}")
        return None

    df_out = df[['Country', 'RMSE', 'MAE']].copy()

    print(f"  Loaded LASSO: {len(df_out)} countries")
    print(f"    RMSE range: {df_out['RMSE'].min():.4f} - {df_out['RMSE'].max():.4f}")
    return df_out


def load_rf_germany_highdim(filepath):
    """
    Load Random Forest Germany High-Dimensional results.

    Parameters
    ----------
    filepath : Path
        Path to RF Germany high-dim summary CSV.

    Returns
    -------
    float or None
        RMSE value for Germany high-dim model.
    """
    if not filepath.exists():
        print(f"  WARNING: RF Germany High-Dim not found at {filepath}")
        return None

    df = pd.read_csv(filepath)

    # Verify it contains Germany
    if 'Country' in df.columns:
        ger_rows = df[df['Country'] == 'Germany']
        if len(ger_rows) == 0:
            print(f"  ERROR: RF High-Dim file has no Germany row")
            return None
        rmse = ger_rows['RMSE'].values[0]
    else:
        # Assume single row is Germany
        rmse = df['RMSE'].values[0]

    print(f"  Loaded RF Germany High-Dim: RMSE = {rmse:.4f}")
    return rmse


# =============================================================================
# TABLE A: CROSS-COUNTRY COMPARISON
# =============================================================================

def build_cross_country_table(sarimax_df, rf_df, lasso_df):
    """
    Build cross-country comparison table (8 countries, 3 models).

    Structure:
        Country | rmse_sarimax | rmse_rf | rmse_lasso | best_model

    Parameters
    ----------
    sarimax_df : pd.DataFrame
        SARIMAX results.
    rf_df : pd.DataFrame
        RF results.
    lasso_df : pd.DataFrame
        LASSO results.

    Returns
    -------
    pd.DataFrame
        Cross-country comparison table.
    """
    # Start with SARIMAX as base
    df = sarimax_df[['Country', 'RMSE']].copy()
    df = df.rename(columns={'RMSE': 'rmse_sarimax'})

    # Merge RF
    if rf_df is not None:
        rf_temp = rf_df[['Country', 'RMSE']].rename(columns={'RMSE': 'rmse_rf'})
        df = df.merge(rf_temp, on='Country', how='outer')

    # Merge LASSO
    if lasso_df is not None:
        lasso_temp = lasso_df[['Country', 'RMSE']].rename(
            columns={'RMSE': 'rmse_lasso'}
        )
        df = df.merge(lasso_temp, on='Country', how='outer')

    # Determine best model (explicit logic, no idxmin)
    def select_best_model(row):
        """Select model with lowest RMSE."""
        candidates = {
            'SARIMAX': row.get('rmse_sarimax'),
            'Random Forest': row.get('rmse_rf'),
            'LASSO': row.get('rmse_lasso'),
        }

        # Filter out None/NaN
        valid = {k: v for k, v in candidates.items() if pd.notna(v)}

        if not valid:
            return None

        return min(valid, key=valid.get)

    df['best_model'] = df.apply(select_best_model, axis=1)

    # Sort by country
    df = df.sort_values('Country').reset_index(drop=True)

    return df


# =============================================================================
# TABLE B: GERMANY MODEL COMPARISON
# =============================================================================

def build_germany_model_table(sarimax_df, rf_df, lasso_df, rf_ger_highdim_rmse):
    """
    Build Germany-only model comparison table (1 row, 4 models).

    Structure:
        Country | rmse_sarimax | rmse_lasso | rmse_rf_2feat | rmse_rf_highdim | best_model

    Parameters
    ----------
    sarimax_df : pd.DataFrame
        SARIMAX results.
    rf_df : pd.DataFrame
        RF results.
    lasso_df : pd.DataFrame
        LASSO results.
    rf_ger_highdim_rmse : float
        RF Germany high-dim RMSE.

    Returns
    -------
    pd.DataFrame
        Germany model comparison table (single row).
    """
    # Extract Germany values from each model
    ger_sarimax = sarimax_df[sarimax_df['Country'] == 'Germany']['RMSE'].values
    ger_rf = rf_df[rf_df['Country'] == 'Germany']['RMSE'].values if rf_df is not None else []
    ger_lasso = lasso_df[lasso_df['Country'] == 'Germany']['RMSE'].values if lasso_df is not None else []

    # Build single-row DataFrame
    data = {
        'Country': 'Germany',
        'rmse_sarimax': ger_sarimax[0] if len(ger_sarimax) > 0 else None,
        'rmse_lasso': ger_lasso[0] if len(ger_lasso) > 0 else None,
        'rmse_rf_2feat': ger_rf[0] if len(ger_rf) > 0 else None,
        'rmse_rf_highdim': rf_ger_highdim_rmse,
    }

    df = pd.DataFrame([data])

    # Determine best model (explicit logic)
    def select_best_model_germany(row):
        """Select model with lowest RMSE for Germany."""
        candidates = {
            'SARIMAX': row['rmse_sarimax'],
            'LASSO': row['rmse_lasso'],
            'RF (2 features)': row['rmse_rf_2feat'],
            'RF (high-dim)': row['rmse_rf_highdim'],
        }

        # Filter out None/NaN
        valid = {k: v for k, v in candidates.items() if pd.notna(v)}

        if not valid:
            return None

        return min(valid, key=valid.get)

    df['best_model'] = df.apply(select_best_model_germany, axis=1)

    return df


# =============================================================================
# TABLE C: GERMANY RF COMPARISON
# =============================================================================

def build_germany_rf_table(rf_df, rf_ger_highdim_rmse):
    """
    Build Germany RF-only comparison table (1 row, 2 RF variants).

    Structure:
        Country | rmse_rf_2feat | rmse_rf_highdim | better_rf

    Parameters
    ----------
    rf_df : pd.DataFrame
        RF (2-feature) results.
    rf_ger_highdim_rmse : float
        RF Germany high-dim RMSE.

    Returns
    -------
    pd.DataFrame
        Germany RF comparison table (single row).
    """
    # Extract Germany RF 2-feature
    ger_rf = rf_df[rf_df['Country'] == 'Germany']['RMSE'].values if rf_df is not None else []

    # Build single-row DataFrame
    data = {
        'Country': 'Germany',
        'rmse_rf_2feat': ger_rf[0] if len(ger_rf) > 0 else None,
        'rmse_rf_highdim': rf_ger_highdim_rmse,
    }

    df = pd.DataFrame([data])

    # Determine which RF is better
    def select_better_rf(row):
        """Select better RF variant."""
        rf_2feat = row['rmse_rf_2feat']
        rf_highdim = row['rmse_rf_highdim']

        if pd.isna(rf_2feat) or pd.isna(rf_highdim):
            return None

        if rf_highdim < rf_2feat:
            return 'High-dimensional'
        else:
            return '2-feature'

    df['better_rf'] = df.apply(select_better_rf, axis=1)

    # Compute improvement
    if df['rmse_rf_2feat'].notna().all() and df['rmse_rf_highdim'].notna().all():
        improvement = (
            (df['rmse_rf_2feat'].values[0] - df['rmse_rf_highdim'].values[0]) /
            df['rmse_rf_2feat'].values[0] * 100
        )
        df['improvement_%'] = improvement

    return df


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def compute_cross_country_summary(df_cross):
    """
    Compute summary statistics for cross-country comparison.

    Parameters
    ----------
    df_cross : pd.DataFrame
        Cross-country comparison table.

    Returns
    -------
    pd.DataFrame
        Summary with mean RMSE and win count per model.
    """
    summary_data = []

    models = [
        ('SARIMAX', 'rmse_sarimax'),
        ('Random Forest', 'rmse_rf'),
        ('LASSO', 'rmse_lasso'),
    ]

    for model_name, col_name in models:
        if col_name not in df_cross.columns:
            continue

        rmse_values = df_cross[col_name].dropna()
        wins = (df_cross['best_model'] == model_name).sum()

        summary_data.append({
            'Model': model_name,
            'Mean_RMSE': rmse_values.mean(),
            'Std_RMSE': rmse_values.std(),
            'Min_RMSE': rmse_values.min(),
            'Max_RMSE': rmse_values.max(),
            'N_countries': len(rmse_values),
            'Wins': wins,
        })

    return pd.DataFrame(summary_data)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_cross_country_comparison(df_cross, output_path):
    """
    Create bar chart comparing RMSE across countries and models.

    Parameters
    ----------
    df_cross : pd.DataFrame
        Cross-country comparison table.
    output_path : Path
        Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    countries = df_cross['Country'].tolist()
    x = range(len(countries))
    width = 0.25

    ax.bar(
        [i - width for i in x],
        df_cross['rmse_sarimax'],
        width,
        label='SARIMAX',
        color='#3498db',
        edgecolor='black',
        linewidth=0.5
    )
    ax.bar(
        x,
        df_cross['rmse_rf'],
        width,
        label='Random Forest',
        color='#2ecc71',
        edgecolor='black',
        linewidth=0.5
    )
    ax.bar(
        [i + width for i in x],
        df_cross['rmse_lasso'],
        width,
        label='LASSO',
        color='#e74c3c',
        edgecolor='black',
        linewidth=0.5
    )

    ax.set_xlabel('Country', fontsize=12)
    ax.set_ylabel('RMSE (YoY Inflation, x100)', fontsize=12)
    ax.set_title(
        'Cross-Country Forecast Accuracy: SARIMAX vs RF vs LASSO',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=45, ha='right')
    ax.legend(title='Model')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


def plot_germany_model_comparison(df_germany, output_path):
    """
    Create bar chart comparing all models for Germany.

    Parameters
    ----------
    df_germany : pd.DataFrame
        Germany model comparison table.
    output_path : Path
        Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ['SARIMAX', 'LASSO', 'RF (2-feat)', 'RF (high-dim)']
    rmse_cols = ['rmse_sarimax', 'rmse_lasso', 'rmse_rf_2feat', 'rmse_rf_highdim']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    values = [df_germany[col].values[0] for col in rmse_cols]

    bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, values):
        if pd.notna(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.4f}',
                ha='center',
                fontsize=10
            )

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('RMSE (YoY Inflation, x100)', fontsize=12)
    ax.set_title(
        'Germany: Model Comparison (4 models)',
        fontsize=14,
        fontweight='bold'
    )
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


def plot_heatmap(df_cross, output_path):
    """
    Create heatmap of RMSE values across countries and models.

    Parameters
    ----------
    df_cross : pd.DataFrame
        Cross-country comparison table.
    output_path : Path
        Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    heatmap_data = df_cross.set_index('Country')[
        ['rmse_sarimax', 'rmse_rf', 'rmse_lasso']
    ]
    heatmap_data.columns = ['SARIMAX', 'Random Forest', 'LASSO']

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.4f',
        cmap='RdYlGn_r',
        ax=ax,
        cbar_kws={'label': 'RMSE (x100)'}
    )

    ax.set_title(
        'RMSE Heatmap: Model Performance by Country',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function to run the comparison analysis.

    Produces three tables:
        A. Cross-country comparison (8 countries, 3 models)
        B. Germany model comparison (1 row, 4 models)
        C. Germany RF comparison (1 row, 2 RF variants)
    """
    print("\n" + "=" * 70)
    print(" FINAL COMPARISON: ML vs ECONOMETRICS")
    print("=" * 70)

    print(f"\nProject root: {PROJECT_ROOT}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STEP 1: Load all model results
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 1: Loading Model Results")
    print("-" * 70)

    sarimax_df = load_sarimax_results(SARIMAX_SUMMARY)
    rf_df = load_rf_results(RF_SUMMARY)
    lasso_df = load_lasso_results(LASSO_SUMMARY)
    rf_ger_highdim_rmse = load_rf_germany_highdim(RF_GER_HIGHDIM_SUMMARY)

    # Validate minimum requirements
    if sarimax_df is None:
        print("\n  ERROR: SARIMAX results required. Aborting.")
        return

    # =========================================================================
    # STEP 2: Verify scale alignment
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 2: Verifying Scale Alignment")
    print("-" * 70)

    print("\n  RMSE ranges after loading (should be similar scale):")
    print(f"    SARIMAX:       {sarimax_df['RMSE'].min():.4f} - {sarimax_df['RMSE'].max():.4f}")
    if rf_df is not None:
        print(f"    RF:            {rf_df['RMSE'].min():.4f} - {rf_df['RMSE'].max():.4f}")
    if lasso_df is not None:
        print(f"    LASSO:         {lasso_df['RMSE'].min():.4f} - {lasso_df['RMSE'].max():.4f}")
    if rf_ger_highdim_rmse is not None:
        print(f"    RF High-Dim:   {rf_ger_highdim_rmse:.4f}")

    # =========================================================================
    # STEP 3: Build TABLE A - Cross-Country Comparison
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 3: TABLE A - Cross-Country Comparison (8 countries, 3 models)")
    print("-" * 70)

    df_cross = build_cross_country_table(sarimax_df, rf_df, lasso_df)

    print("\n  CROSS-COUNTRY COMPARISON:")
    print("  " + "-" * 66)
    print(df_cross.to_string(index=False))

    # Compute summary
    summary = compute_cross_country_summary(df_cross)
    print("\n  SUMMARY (Cross-Country):")
    print("  " + "-" * 66)
    print(summary.to_string(index=False))

    # =========================================================================
    # STEP 4: Build TABLE B - Germany Model Comparison
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 4: TABLE B - Germany Model Comparison (4 models)")
    print("-" * 70)

    df_germany = build_germany_model_table(
        sarimax_df, rf_df, lasso_df, rf_ger_highdim_rmse
    )

    print("\n  GERMANY MODEL COMPARISON:")
    print("  " + "-" * 66)
    print(df_germany.to_string(index=False))

    # =========================================================================
    # STEP 5: Build TABLE C - Germany RF Comparison
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 5: TABLE C - Germany RF Comparison (2-feat vs high-dim)")
    print("-" * 70)

    df_rf = build_germany_rf_table(rf_df, rf_ger_highdim_rmse)

    print("\n  GERMANY RF COMPARISON:")
    print("  " + "-" * 66)
    print(df_rf.to_string(index=False))

    # =========================================================================
    # STEP 6: Save outputs
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 6: Saving Outputs")
    print("-" * 70)

    # Table A
    path_a = OUTPUT_DIR / 'table_a_cross_country_comparison.csv'
    df_cross.to_csv(path_a, index=False)
    print(f"  Saved: {path_a.name}")

    # Summary
    path_summary = OUTPUT_DIR / 'table_a_summary.csv'
    summary.to_csv(path_summary, index=False)
    print(f"  Saved: {path_summary.name}")

    # Table B
    path_b = OUTPUT_DIR / 'table_b_germany_model_comparison.csv'
    df_germany.to_csv(path_b, index=False)
    print(f"  Saved: {path_b.name}")

    # Table C
    path_c = OUTPUT_DIR / 'table_c_germany_rf_comparison.csv'
    df_rf.to_csv(path_c, index=False)
    print(f"  Saved: {path_c.name}")

    # =========================================================================
    # STEP 7: Generate plots
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 7: Generating Plots")
    print("-" * 70)

    plot_cross_country_comparison(df_cross, OUTPUT_DIR / 'plot_cross_country.png')
    plot_germany_model_comparison(df_germany, OUTPUT_DIR / 'plot_germany_models.png')
    plot_heatmap(df_cross, OUTPUT_DIR / 'plot_heatmap.png')

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print(" COMPARISON COMPLETE")
    print("=" * 70)

    # Cross-country winner
    best_mean_model = summary.loc[summary['Mean_RMSE'].idxmin(), 'Model']

    print(f"\n  CROSS-COUNTRY RESULTS (8 countries, 3 models):")
    print(f"  {'─' * 50}")
    print(f"    Best mean RMSE: {best_mean_model}")

    # Germany winner
    ger_best = df_germany['best_model'].values[0]
    print(f"\n  GERMANY RESULTS (4 models):")
    print(f"  {'─' * 50}")
    print(f"    Best model: {ger_best}")

    # Germany RF comparison
    rf_better = df_rf['better_rf'].values[0]
    if 'improvement_%' in df_rf.columns:
        improvement = df_rf['improvement_%'].values[0]
        print(f"\n  GERMANY RF COMPARISON:")
        print(f"  {'─' * 50}")
        print(f"    Better RF variant: {rf_better}")
        print(f"    Improvement: {improvement:.2f}%")

    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 70 + "\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    main()
