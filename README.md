# Apple Stock Time-Series Analysis and Forecasting

A time-series portfolio project exploring historical Apple price data and a forecasting application with multiple statistical and machine-learning models.

## Project files

| File | Purpose |
|---|---|
| `P614 DATASET.csv` | Historical daily price data |
| `SMA_apple.ipynb` | Analysis notebook |
| `app.py` | Streamlit forecasting application |
| `requirements.txt` | Python dependencies |
| `*.joblib` | Saved forecasting model artifacts |
| `APP_LINK.md` | App URL and usage instructions |

## Verified dataset profile

Recalculated from the repository's current `P614 DATASET.csv`:

| Metric | Result |
|---|---:|
| Rows | 2,011 |
| Columns | 7 |
| Missing cells | 0 |
| Date range | 2012-01-03 to 2019-12-30 |
| Close price minimum in the supplied file | 55.790001 |
| Close price maximum in the supplied file | 291.519989 |
| Daily volume minimum | 11,362,000 |
| Daily volume maximum | 376,530,000 |

Columns: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, and `Volume`. Price units/currency and the exact adjustment methodology should be confirmed against the dataset source.

## Forecasting workflow

The repository documents an ensemble with multiple forecasting approaches (including ARIMA/SARIMA and exponential smoothing variants) and an XGBoost meta-learner. The Streamlit app accepts a CSV containing date and closing-price columns and supports a selectable forecast horizon of 1–60 business days.

1. Verify and sort the historical date series.
2. Inspect stationarity and temporal patterns.
3. Split data chronologically before fitting models or transformations.
4. Compare each model to a naive baseline on the same forecast horizon.
5. Evaluate holdout predictions using MAE/RMSE and other appropriate metrics.
6. Check that the app applies the same preprocessing expected by each saved model.

## Results and model validation

The historical CSV profile above was recalculated from the current repository file. The notebook could not be fully extracted within the connector's response limit in this review, and no forecast-versus-actual metric table was independently confirmed. Therefore, this README does not claim a verified forecasting accuracy, MAE, or RMSE.

A valid model comparison should document the forecast horizon, chronological split dates, baseline score, test period, model configuration, and error metric. Historical performance is not a guarantee of future price direction.

## Run the app

1. Create a Python environment and install `requirements.txt`.
2. Launch the app from the repository root:

   ```bash
   streamlit run app.py
   ```

3. Upload a CSV with date and closing-price fields and select the correct columns.
4. Select a forecast horizon and generate predictions.

Live app link, as documented in `APP_LINK.md`: [Stock Price Forecast App](https://stock-market-analysis---apple-qj4jebpk8mthmdmysuy5vy.streamlit.app/).

## Limitations

- The dataset spans historical observations only; forecasts are uncertain.
- Price adjustment, source, and licensing details need explicit documentation.
- Saved forecasts need chronological holdout evaluation and a naive baseline before performance claims are made.
- This is an educational modelling project, not investment advice.

## Skills demonstrated

Python, pandas, time-series preparation, statistical forecasting, model comparison, validation design, model serialization, and Streamlit.

## Author

Sahil Gaikwad · [GitHub](https://github.com/sahil2003-ai) · [LinkedIn](https://www.linkedin.com/in/sahil-gaikwad-a76b7224a/)
