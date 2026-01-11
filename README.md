# Can ML Beat Traditional Macroeconomic Forecasts?

**Author:** Elena Onieva Henrich  
**Course:** Data Science and Advanced Programming 2025-2026  
**University:** Université de Lausanne

---

## Research Question

Which forecasting methodology performs better for CPI inflation:
- **SARIMAX** (econometric time series with exogenous variables)
- **Random Forest** (machine learning)
- **LASSO** (sparse linear regression)

---

## Project Overview

This project tests whether machine learning models can outperform standard econometric benchmarks in short-term macroeconomic forecasting. I forecast CPI inflation for 8 OECD economies using monthly panel data (1996–2019, pre-COVID).

**Countries:** Belgium, Germany, Israel, Korea, Latvia, Lithuania, Norway, Switzerland

**Exogenous Variables:** Unemployment rate, Import prices

---

## Setup

```bash
# Clone the repository
git clone <repository-url>
cd DSAP_Foreacast_ML_vs_STAT

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

-----

## Usage

```bash
python main.py
```

This runs the complete pipeline:

1. Data preprocessing (load CPI, unemployment, import prices)
1. ARIMAX/SARIMAX models (grid search, forecasting)
1. Random Forest models (standard + Germany high-dimensional)
1. LASSO models
1. Final comparison across all methods

-----

## Expected Output

- All results saved to `results/` directory
- Forecast plots per country and model
- Summary CSV files with RMSE, MAE metrics
- Final comparison table in `results/comparison/`

-----

## Project Structure

```
DSAP_Foreacast_ML_vs_STAT/
├── main.py                 # Entry point - runs entire pipeline
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                # Original data files
│   └── processed/          # Processed datasets
├── src/
│   ├── data/               # Data loading scripts
│   │   ├── 1_load_cpi.py
│   │   ├── 2_yoy_stationarity.py
│   │   ├── 3_load_unemployment.py
│   │   ├── 4_load_import_prices.py
│   │   └── 5_prep_data_GER.py
│   ├── models/
│   │   ├── arimax/         # ARIMAX/SARIMAX models
│   │   ├── random_forest/  # Random Forest models
│   │   └── lasso/          # LASSO models
│   └── evaluate_models/    # Final comparison
├── results/                # Output plots and metrics
└── notebooks/              # Exploration (optional)
```

-----

## Requirements

- Python 3.11+
- pandas, numpy, scipy
- statsmodels, pmdarima, arch
- scikit-learn
- matplotlib, seaborn

See `requirements.txt` for full list.

-----

## License

This project is submitted as coursework for Advanced Programming 2025.

