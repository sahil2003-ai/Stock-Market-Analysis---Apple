# Apple Stock Market Analysis & Forecasting

An end-to-end time-series analysis project built around historical Apple (AAPL) market data. It combines dataset inspection, statistical forecasting approaches, a stacked forecasting workflow and an interactive Streamlit application that lets users upload a stock-price CSV and generate forecasts for a configurable business-day horizon.

<p align="left">
  <img src="https://img.shields.io/badge/Python-Time%20Series-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Forecasting-ARIMA%20%7C%20SARIMA-6A5ACD" alt="Time series forecasting" />
  <img src="https://img.shields.io/badge/ML-XGBoost-EC6B23" alt="XGBoost" />
  <img src="https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
</p>

## Project overview

This project explores daily Apple stock OHLCV data and provides an interactive workflow for forecasting future closing prices. The repository includes the original dataset, analysis notebook, Streamlit app, saved model artifacts and deployment notes.

The project demonstrates time-series data preparation, model integration, feature generation and an interactive forecasting interface. Forecast values are experimental estimates and must not be interpreted as guaranteed prices or investment advice.

## Live demo

**[Open the Stock Price Forecast App](https://stock-market-analysis---apple-qj4jebpk8mthmdmysuy5vy.streamlit.app/)**

The app accepts a CSV with a date column and a closing-price column, lets the user choose the relevant columns, and generates a forecast of **1–60 business days**.

## Dataset

**File:** [`P614 DATASET.csv`](P614%20DATASET.csv)

The stored dataset contains daily historical market records with the fields `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close` and `Volume`.

| Dataset metric | Result |
|---|---:|
| Observations | 2,011 |
| Columns | 7 |
| Missing cells in the reviewed profile | 0 |
| First date | 2012-01-03 |
| Last date | 2019-12-30 |
| Minimum recorded Close | 55.790001 |
| Maximum recorded Close | 291.519989 |
| Minimum recorded Volume | 11,362,000 |
| Maximum recorded Volume | 376,530,000 |

**Data interpretation:** prices are presented as stored in the CSV. Verify the dataset's provenance, currency, and price-adjustment methodology before reusing or comparing values with other market data. The supplied history ends on 2019-12-30; it is not current market data.

### Column guide

| Column | Meaning |
|---|---|
| `Date` | Trading date |
| `Open` | Opening price |
| `High` | Intraday high |
| `Low` | Intraday low |
| `Close` | Closing price used by the app when selected |
| `Adj Close` | Adjusted closing price as provided in the source file |
| `Volume` | Reported trading volume |

## Forecasting methodology

The repository contains forecasting artifacts for a set of statistical models and a stacked prediction workflow. The Streamlit app loads these saved model files:

| Model/component | Saved artifact |
|---|---|
| Simple Exponential Smoothing | `ses_model.joblib` |
| Exponential Smoothing | `es_model.joblib` |
| ARIMA | `arima_model.joblib` |
| SARIMA | `sarima_model.joblib` |
| Log-transformed SARIMA | `log_sarima_model.joblib` |
| Linear Regression meta-feature model | `linear_regression_meta.joblib` |
| XGBoost meta-model | `best_xgb_model.joblib` |

At forecast time, the application resamples the chosen price series to business-day frequency, forward-fills gaps, generates forecasts from statistical model families, builds additional features for the linear-regression component, and passes the combined forecast signals into the XGBoost meta-model. It converts the predicted percentage changes to a predicted price series.

### Workflow

1. Load historical prices from the supplied CSV or a user-uploaded file.
2. Parse the date field, sort chronologically and remove duplicate dates.
3. Convert the selected closing-price series to business-day frequency.
4. Prepare statistical-model forecasts for the requested horizon.
5. Construct features for the linear-regression component.
6. Combine base-model outputs and generate the final forecast using the XGBoost component.
7. Display a table of predicted prices and a chart comparing recent history with the forecast.

## Results and evaluation

### Dataset-level results

The dataset profile confirms **2,011 observations** across 2012-01-03 to 2019-12-30. The supplied closing prices range from **55.790001 to 291.519989**.

### Model-performance metrics

This README does **not** publish MAE, RMSE, MAPE or directional-accuracy values because I could not verify a reproducible chronological holdout comparison from the repository files reviewed. The existence of saved model artifacts is not, by itself, evidence of forecast accuracy.

For a rigorous performance report, evaluate all models on the same chronological test period and horizon, compare them against a naive baseline (such as carrying the last observed close forward), and record the test dates, target column, and metric definitions.

## Streamlit app features

The app in [`app.py`](app.py) supports:

- CSV upload
- Interactive selection of date and closing-price columns
- Date parsing, sorting and duplicate-date handling
- Business-day resampling and forward-fill
- Configurable forecast horizon from 1 to 60 business days
- Predicted-price table
- Historical-versus-forecast plot
- Loading of saved statistical and machine-learning model artifacts

Live app and usage notes: [`APP_LINK.md`](APP_LINK.md).

## Technology stack

- **Python** — analysis and app logic
- **pandas / NumPy** — time-series preparation and feature calculations
- **statsmodels** — exponential smoothing, ARIMA and SARIMAX
- **scikit-learn** — model components and forecasting workflow support
- **XGBoost** — stacked forecast component
- **joblib** — serialized model artifacts
- **Matplotlib / Seaborn** — visualization tools
- **Streamlit** — interactive user interface

Dependency list: [`requirements.txt`](requirements.txt).

## Repository structure

```text
Stock-Market-Analysis---Apple/
├── P614 DATASET.csv
├── SMA_apple.ipynb
├── app.py
├── APP_LINK.md
├── requirements.txt
├── ses_model.joblib
├── es_model.joblib
├── arima_model.joblib
├── sarima_model.joblib
├── log_sarima_model.joblib
├── linear_regression_meta.joblib
├── best_xgb_model.joblib
└── README.md
```

The repository currently has no separate image or screenshot files. GitHub does not render the Streamlit UI from the source code as an inline screenshot, so this README avoids embedding invented images. To add real visuals, capture screenshots from the running app and commit them under an `images/` directory, then embed those files here.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/sahil2003-ai/Stock-Market-Analysis---Apple.git
cd Stock-Market-Analysis---Apple
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the application

Run from the repository root so the app can locate the saved `.joblib` files:

```bash
streamlit run app.py
```

Upload a CSV with a date column and a closing-price column. Select those columns in the UI, choose the forecast horizon, and select **Generate Forecast**.

### 5. Review the notebook

Open [`SMA_apple.ipynb`](SMA_apple.ipynb) to review the model-development and analysis workflow. Re-run the notebook only after checking the data path, environment compatibility and evaluation strategy.

## Limitations and next steps

- **Historical coverage:** the included dataset ends on 2019-12-30 and should not be mistaken for live market data.
- **Forecast evaluation:** add a chronological holdout and report MAE, RMSE, MAPE where appropriate, and comparison with a naive baseline.
- **Potential leakage:** verify how the saved models and meta-model were trained, and ensure no future information enters training features or stacking.
- **Inference consistency:** confirm training and app use the same target definition, resampling rules, feature order and preprocessing.
- **Business-day filling:** forward-filling weekend and holiday gaps creates a regularized series; document this assumption and evaluate its impact.
- **Model compatibility:** serialized `.joblib` artifacts can depend on package versions and training-time code. Pin dependencies and validate a clean install.
- **Data provenance:** document where the CSV came from and its usage license.
- **Visual documentation:** add actual screenshots of the app and forecast chart after a successful run.
- **Financial safety:** forecasts can be wrong and are not personalized financial advice.

## Skills demonstrated

Time-series data preparation · exploratory data analysis · forecasting workflow design · ARIMA/SARIMA · exponential smoothing · feature engineering · stacked model integration · XGBoost · model serialization · Streamlit · visualization · evaluation design

## Author

**Sahil Gaikwad**  
[GitHub](https://github.com/sahil2003-ai) · [LinkedIn](https://www.linkedin.com/in/sahil-gaikwad-a76b7224a/)
