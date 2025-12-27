# Stock Market Analysis - Apple

A comprehensive time series forecasting project that predicts Apple stock prices using an ensemble of machine learning and statistical models. This project demonstrates advanced techniques in time series analysis, feature engineering, and model stacking.

## Project Overview

This project analyzes historical Apple stock price data (2012-2019) and builds a sophisticated forecasting system using multiple time series models combined through ensemble learning. The final prediction model is a stacked ensemble that uses XGBoost as a meta-learner to combine predictions from 6 different base models.

## Dataset

- **File**: `P614_DATASET.csv`
- **Data Points**: 2,011 trading days
- **Time Period**: January 3, 2012 - November 18, 2019
- **Features**: Date, Open, High, Low, Close, Adjusted Close, Volume
- **Target Variable**: Close Price (Adjusted Close)

## Key Features

### Data Analysis & Preprocessing
- Missing value imputation using forward fill method
- Business day frequency resampling (excluding weekends)
- Stationarity testing using Augmented Dickey-Fuller (ADF) test
- Data decomposition (STL) to analyze trend, seasonality, and residuals
- ACF/PACF analysis for identifying AR and MA components

### Base Models

1. **Simple Exponential Smoothing (SES)**
   - Optimized alpha parameter
   - Best for data with no trend or seasonality

2. **Exponential Smoothing (ES)**
   - Tested with additive and multiplicative trend/seasonal components
   - Best configuration: multiplicative trend and seasonality with period 5

3. **ARIMA (AutoRegressive Integrated Moving Average)**
   - Auto ARIMA determined optimal order: (3, 1, 5)
   - Uses differencing (d=1) to achieve stationarity

4. **SARIMA (Seasonal ARIMA)**
   - Order: (3, 1, 5) with constant trend
   - Includes seasonal components for capturing periodic patterns

5. **Log-SARIMA**
   - Applies log transformation to handle multiplicative effects
   - Order: (3, 1, 5) on log-transformed data
   - Predictions exponentially transformed back to price scale

6. **Linear Regression with Feature Engineering**
   - Features include: 1-day and 5-day returns, rolling std, 5/20-day moving averages, RSI-14, day of week
   - Predicts percentage change in stock price

### Ensemble & Meta-Learning

- **Stacking Architecture**: Base model predictions used as meta-features
- **Meta-Learner**: XGBoost Regressor with hyperparameter tuning
- **Hyperparameter Optimization**: RandomizedSearchCV with 3-fold cross-validation
- **Best XGBoost Params**: 
  - n_estimators: 100
  - learning_rate: 0.01
  - max_depth: 4
  - subsample: 0.8
  - colsample_bytree: 0.8

### Model Performance

**Test Set Results (30-day forecast)**:
- XGBoost Meta-Model RMSE: 0.0088
- XGBoost Meta-Model MAE: 0.0054

**Feature Importance (XGBoost Meta-Model)**:
1. Linear Regression: 36.3%
2. Log-SARIMA: 24.1%
3. ARIMA: 15.8%
4. ES: 14.0%
5. SARIMA: 9.7%
6. SES: 0.0%

## Files in Repository

- `P614_DATASET.csv` - Historical Apple stock data (2012-2019)
- `SMA_apple.ipynb` - Jupyter notebook with complete analysis and model building
- `app.py` - Streamlit web application for interactive forecasting
- Model files (`.joblib`):
  - `ses_model.joblib`
  - `es_model.joblib`
  - `arima_model.joblib`
  - `sarima_model.joblib`
  - `log_sarima_model.joblib`
  - `linear_regression_meta.joblib`
  - `best_xgb_model.joblib`

## Installation & Usage

### Requirements
```
pandas
numpy
matplotlib
seaborn
statsmodels>=0.14.5
scikit-learn
xgboost
pmdarima
joblib
streamlit (for web app)
```

### Running the Analysis
```bash
# Install dependencies
pip install -r requirements.txt

# Run the Jupyter notebook
jupyter notebook SMA_apple.ipynb
```

### Running the Web Application
```bash
streamlit run app.py
```

Then:
1. Upload your CSV file with Date and Close Price columns
2. Select the appropriate columns from your data
3. Choose forecast horizon (1-60 business days)
4. Click "Generate Forecast" to see predictions

## Methodology

### Data Exploration
- No missing values or duplicates
- 2,919 business days after resampling
- Close price range: $55.54 - $291.12
- Daily returns mean: 0.0009, std: 0.0158
- Strong positive trend over the period

### Stationarity Analysis
- Original series: Non-stationary (ADF p-value: 0.994)
- After 1st differencing: Stationary (ADF p-value: 1.85e-11)
- Differencing order (d=1) applied in ARIMA/SARIMA models

### Time Series Decomposition
- **Trend**: Consistent upward trend over 2012-2019
- **Seasonality**: Minimal fixed-period seasonality detected
- **Residuals**: Variance increases post-2018, indicating higher volatility

## Technical Highlights

1. **Comprehensive EDA**: ACF/PACF analysis, lag plots, distribution analysis
2. **Multiple Modeling Approaches**: Statistical (ARIMA/SARIMA) and ML (XGBoost)
3. **Feature Engineering**: Technical indicators like RSI, moving averages
4. **Hyperparameter Optimization**: Grid and randomized search for best parameters
5. **Ensemble Learning**: Stacking architecture with meta-learner
6. **Production Ready**: Streamlit app for real-time forecasting

## Model Evaluation Metrics

Metrics used:
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors
- **MAE (Mean Absolute Error)**: Absolute prediction error
- **R² Score**: Variance explained by the model

## Future Improvements

- Include external features (market indices, sentiment analysis)
- Implement LSTM/GRU neural networks for sequence modeling
- Add confidence intervals and prediction uncertainty quantification
- Cross-validation on multiple forecast horizons
- Real-time data integration for live predictions

## Author

Sahil Kumar

## License

This project is open source and available under the MIT License.

## Live Demo

**Try the app online:** [Stock Price Forecast App](https://stock-market-analysis---apple-qj4jebpk8mthmdmysuy5vy.streamlit.app/)

The Streamlit application is deployed on Streamlit Cloud and is ready to use. Simply visit the link above to upload your stock data and get instant forecasts!

This project is open source and available under the MIT License.
