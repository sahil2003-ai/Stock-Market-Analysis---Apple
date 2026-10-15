# Apple Stock Time-Series Analysis and Forecasting

A time-series project exploring historical Apple price data and a forecasting application with statistical and machine-learning approaches.

![Python](https://img.shields.io/badge/Python-Time%20Series-blue) ![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B)

## Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Forecasting Workflow](#forecasting-workflow)
- [Results and Metrics](#results-and-metrics)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Run the App](#run-the-app)
- [Limitations and Next Steps](#limitations-and-next-steps)

## Overview
The project examines historical daily Apple stock prices and explores multi-step forecasting through a Streamlit application. Forecasting results should be interpreted as experimental estimates, not investment advice.

## Dataset
Source file: `P614 DATASET.csv`.

| Metric | Result |
|---|---:|
| Rows | 2,011 |
| Columns | 7 |
| Missing cells | 0 |
| Date range | 2012-01-03 to 2019-12-30 |
| Close minimum in supplied file | 55.790001 |
| Close maximum in supplied file | 291.519989 |
| Daily volume minimum | 11,362,000 |
| Daily volume maximum | 376,530,000 |

Columns: Date, Open, High, Low, Close, Adj Close, and Volume. Confirm currency/price units, adjustment methodology, and original data source before redistributing the dataset.

## Forecasting Workflow
The repository describes an ensemble approach with ARIMA/SARIMA and exponential-smoothing variants plus an XGBoost meta-learner. The Streamlit app accepts a CSV with date and closing-price fields and supports a forecast horizon of 1–60 business days.

1. Validate and sort historical dates.
2. Inspect trends, stationarity, and temporal patterns.
3. Use chronological splits to prevent future information entering training.
4. Compare model forecasts with a naive baseline on the same horizon.
5. Evaluate with MAE/RMSE and clearly state the forecast horizon.
6. Ensure inference preprocessing matches training.

## Results and Metrics
The verified data profile contains 2,011 daily observations over the listed date range, with no missing cells. No model error metric is presented here without a reproducible holdout result. When documenting model results, include each model's MAE/RMSE, the evaluation dates, forecast horizon, and baseline comparison.

## Technology Stack
- Python
- pandas and numerical/data-science libraries used in the notebook
- Time-series/statistical and machine-learning models documented in the project
- Streamlit for the forecasting interface

## Repository Structure
```text
Stock-Market-Analysis---Apple/
├── P614 DATASET.csv
├── SMA_apple.ipynb
├── app.py
├── requirements.txt
├── APP_LINK.md
└── README.md
```
The repository may also contain saved model artifacts; check the current file tree for exact filenames.

## Run the App
```bash
git clone https://github.com/sahil2003-ai/Stock-Market-Analysis---Apple.git
cd Stock-Market-Analysis---Apple
python -m venv .venv
```
Activate the environment, install `requirements.txt`, then run `streamlit run app.py`. To reproduce the analysis, run `SMA_apple.ipynb` and verify all data paths.

## Limitations and Next Steps
- Document source and licensing.
- Compare all models on the same chronological holdout.
- Include a naive baseline and avoid random shuffling for time-series evaluation.
- State whether prices are adjusted and define the target field.
- Add actual app screenshots and verified model metrics.
- Do not treat historical backtests as guaranteed future returns.

## Skills Demonstrated
Time-series data preparation, forecasting workflow design, model comparison, error-metric interpretation, and interactive app development.

## Author
**Sahil Gaikwad** · [GitHub](https://github.com/sahil2003-ai) · [LinkedIn](https://www.linkedin.com/in/sahil-gaikwad-a76b7224a/)
