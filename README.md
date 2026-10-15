# Apple Stock Time-Series Analysis and Forecasting

A time-series portfolio project exploring historical Apple stock prices and comparing statistical and machine-learning forecasting approaches.

## Dataset
The current project documentation identifies `P614_DATASET.csv` with daily market observations from 2012 to 2019. Verify the actual file, date range, adjusted-price definition, and data source before reproducing the analysis.

## Methods Documented
- Missing-value handling and business-day resampling
- Augmented Dickey-Fuller stationarity testing
- Trend/seasonality decomposition and ACF/PACF analysis
- Simple and Holt-Winters exponential smoothing
- ARIMA and SARIMA variants
- Feature-engineered regression
- A stacked ensemble using XGBoost as a meta-learner

## Project Goals
Compare model approaches, examine time-series patterns, and evaluate forecast errors using a chronological validation strategy.

## Tools
Python, pandas, statsmodels/time-series methods, scikit-learn, XGBoost, and visualization libraries as used in the notebook.

## Evaluation Checklist
- Use chronological train/validation/test splits.
- Fit transformations only on historical training data.
- Compare against naive and seasonal-naive baselines.
- Report MAE, RMSE, and MAPE only where appropriate.
- Ensure ensemble meta-features are generated without leakage.
- State the forecast horizon and retraining procedure.

## Limitations
Historical stock prices are noisy and do not reliably predict future returns. This project is for learning and model-comparison purposes, not investment advice.

## Recommended Improvements
Document exact model metrics and evaluation windows, add forecast-versus-actual plots, provide reproducible environment instructions, and explain how look-ahead bias is avoided.