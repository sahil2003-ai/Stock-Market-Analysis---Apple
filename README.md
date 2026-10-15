# Apple Stock Time-Series Analysis and Forecasting

A time-series project comparing statistical and machine-learning methods on historical Apple stock-price data.

## Step-by-step workflow

1. Inspect `P614 DATASET.csv` and confirm its source, date range, price column, and missing values.
2. Open `SMA_apple.ipynb` and review the chronological ordering and preprocessing steps.
3. Explore price trends, returns, stationarity, decomposition, and ACF/PACF as implemented.
4. Create chronological training, validation, and test windows; do not randomly shuffle time-series observations.
5. Fit statistical models and machine-learning approaches using only historical data available at each prediction point.
6. Compare against naive or seasonal-naive baselines.
7. Evaluate using a stated forecast horizon and appropriate error metrics such as MAE and RMSE.
8. Check ensemble construction for leakage and compare notebook results with saved model files.
9. Review `app.py` and `APP_LINK.md` for the intended app usage.
10. Document verified results and limitations.

## Tools
Python, pandas, time-series/statistical libraries, scikit-learn, XGBoost, Streamlit.

## Screenshots
No screenshot files were found in the repository root. Add an app or forecast plot to `images/` and embed it here.

## Limitations
Stock prices are noisy and historical performance does not reliably predict future returns. This project is educational, not investment advice.

## Next improvements
Document dataset provenance, exact forecast horizon, baseline comparison, verified metrics, and a reproducible environment.