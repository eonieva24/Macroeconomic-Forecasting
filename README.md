#Can ML Beat Traditional Macroeconomic Forecasts ?

**Author:** Elena Onieva Henrich
**Course:** Datascience and Advanced programming 2025-2026
**University:** Université de Lausanne

##Project proposal
For my project i would like to test if machine learning model can outperform a stan-
dard econometric benchmark in short-term macroeconomic forecasting. I will forecast
quarter-ahead CPI inflation for OECD economies using a quarterly panel (1990–2024),
or I could also just focus on one country like the US. The feature set will start with core
indicators (inflation lags, GDP growth, unemployment, interest rates, industrial produc-
tion) and may be modestly expanded after exploratory analysis.
I will compare two approaches: (i) a univariate ARIMA/ARIMAX benchmark and
(ii) a regularized linear model (Lasso) that can handle many lagged predictors while re-
maining interpretable. I choose Lasso regression because it provides a transparent and
interpretable framework for assessing the relative importance of macroeconomic indica-
tors, which is essential for drawing policy-relevant insights.
To incorporate a nonlinear ML dimension, I will implement a tree-based ensemble
method, XGBoost. I reviewed recent studies using ML in macroeconomics, and the lit-
erature suggests that XGBoost tends to deliver stronger predictive accuracy while main-
taining reasonable interpretability. If time permits, I will also estimate a Random Forest
model to compare the two machine-learning techniques; however, Random Forests are
known to be more prone to overfitting in smaller macroeconomic datasets, which is why
XGBoost will serve as my primary nonlinear approach.
Evaluation will use a rolling-window scheme with out-of-sample RMSE/MAE and
forecast-bias checks.
Interpretation will rely on Lasso coefficient paths and stability across windows. At the
end of the project I’d like to see if ML adds predictive value over a classical time-series
baseline.
