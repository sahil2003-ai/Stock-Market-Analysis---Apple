import streamlit as st
import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# --- Helper functions (adapted from notebook) ---

# This function will now only process the dataframe, not interact with widgets
@st.cache_data
def process_uploaded_data(df, date_col_name, close_col_name):
    try:
        df[date_col_name] = pd.to_datetime(df[date_col_name])
    except Exception as e:
        st.error(f"Error converting Date column to datetime: {e}. Please ensure the date column is in a valid format.")
        return None

    df = df.set_index(date_col_name).sort_index()
    df = df[~df.index.duplicated(keep='first')]

    fq_bd = df[close_col_name].asfreq('B')
    fq_bd = fq_bd.ffill()
    fq_bd.name = 'stock_price'

    if fq_bd.isnull().any():
        st.warning("Missing values found after resampling and forward-fill. Please check your data.")

    return fq_bd


@st.cache_resource
def load_models():
    try:
        ses_model = joblib.load("ses_model.joblib")
        es_model = joblib.load("es_model.joblib")
        arima_model = joblib.load("arima_model.joblib")
        sarima_model = joblib.load("sarima_model.joblib")
        log_sarima_model = joblib.load("log_sarima_model.joblib")
        lr_model = joblib.load("linear_regression_meta.joblib")
        xgb_model = joblib.load("best_xgb_model.joblib")
        return ses_model, es_model, arima_model, sarima_model, log_sarima_model, lr_model, xgb_model
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Please ensure all .joblib model files are in the same directory as the app.")
        return None, None, None, None, None, None, None

# Function to create features for the Linear Regression model's input for future dates
def create_features_for_lr(price_series_full):
    pct = price_series_full.pct_change()
    features = pd.DataFrame({
        'r_1' : pct.shift(1),
        'r_5' : pct.rolling(5).mean().shift(1),
        'std' : pct.rolling(10).std().shift(1),
        'mavg_5' : price_series_full.rolling(5).mean().shift(1),
        'mavg_20' : price_series_full.rolling(20).mean().shift(1),
        'rsi_14' : (pct.clip(lower = 0).rolling(14).mean()/
                    (pct.abs().rolling(14).mean() + 1e-9)).shift(1),
        'dow' : price_series_full.index.dayofweek
    })
    return features

# --- Streamlit App ---
st.title("Stock Price Forecast App")
st.write("Forecast stock prices using a stacked ensemble model.")

# --- Widget interaction moved outside cached function ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

df = None
fq_bd_full = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Head:")
    st.dataframe(df.head())

    col_names = df.columns.tolist()
    date_col_name = st.selectbox("Select Date Column", options=col_names, index=col_names.index('Date') if 'Date' in col_names else 0)
    close_col_name = st.selectbox("Select Close Price Column", options=col_names, index=col_names.index('Close') if 'Close' in col_names else 0)

    if date_col_name not in df.columns or close_col_name not in df.columns:
        st.error("Selected Date or Close Price column not found in the uploaded file. Please check column names.")
    else:
        fq_bd_full = process_uploaded_data(df.copy(), date_col_name, close_col_name)


if fq_bd_full is not None:
    if fq_bd_full.empty:
        st.warning("The processed dataset is empty. Please check your uploaded file and column selections.")
    else:
        log_fq_bd_full = np.log(fq_bd_full)

        ses, es, arima, sarima, log_sarima, lr, best_xgb = load_models()

        if all(model is not None for model in [ses, es, arima, sarima, log_sarima, lr, best_xgb]):
            st.write(f"Data loaded successfully. Last date: {fq_bd_full.index[-1].strftime('%Y-%m-%d')}")
            st.write(f"Number of historical data points: {len(fq_bd_full)}")

            h = st.sidebar.slider("Forecast Horizon (business days)", min_value=1, max_value=60, value=30)

            if st.sidebar.button("Generate Forecast"):
                st.subheader(f"Forecasting for the next {h} business days")

                # Generate future dates for forecasting
                last_date = fq_bd_full.index[-1]
                future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=h, freq='B')

                # --- Re-fitting base models on full data for fresh forecasts ---
                # Simple Exponential Smoothing (SES)
                ses_re_fit = SimpleExpSmoothing(fq_bd_full).fit(optimized=True)
                ses_forecast = pd.Series(ses_re_fit.forecast(h), index=future_dates)

                # Exponential Smoothing (ES) - Using best params from notebook ('mul', 'mul', 5) or fallback
                try:
                    # Use the previously found best parameters for ES if available, otherwise default
                    # Removed seasonal='mul' seasonal_periods=5 as it was causing issues
                    es_re_fit = ExponentialSmoothing(fq_bd_full, trend='mul', seasonal=None).fit()
                except Exception:
                    st.warning("Could not re-fit ExponentialSmoothing with 'mul' trend. Falling back to 'add' trend and no seasonality.")
                    es_re_fit = ExponentialSmoothing(fq_bd_full, trend='add', seasonal=None).fit()
                es_forecast = pd.Series(es_re_fit.forecast(h), index=future_dates)

                # ARIMA (Order (3,1,5) from auto_arima)
                arima_re_fit = ARIMA(fq_bd_full, order=(3,1,5)).fit()
                arima_forecast = arima_re_fit.forecast(h)
                arima_forecast.index = future_dates

                # SARIMA (Order (3,1,5), trend='c')
                sarima_re_fit = SARIMAX(fq_bd_full, order=(3,1,5), trend='c').fit()
                sarima_forecast = sarima_re_fit.forecast(steps=h)
                sarima_forecast.index = future_dates

                # Log-SARIMA (Order (3,1,5), trend='c')
                log_sarima_re_fit = SARIMAX(log_fq_bd_full, order=(3,1,5), trend='c').fit()
                log_sarima_forecast = pd.Series(np.exp(log_sarima_re_fit.get_forecast(steps=h).predicted_mean), index=future_dates)

                # Linear Regression (meta-feature) - predicts pct_change
                # We need to generate future features for LR based on some proxy of future prices.
                # Using SARIMA forecast as a proxy for future prices to generate features for LR.
                combined_price_for_lr_features = pd.concat([fq_bd_full, sarima_forecast])
                lr_features_full = create_features_for_lr(combined_price_for_lr_features)
                # Extract features for the forecast period (last h rows)
                lr_forecast_X = lr_features_full.iloc[-h:]

                if not lr_forecast_X.empty:
                    lr_forecast_pct_change = lr.predict(lr_forecast_X)
                    lr_forecast_pct_change_series = pd.Series(lr_forecast_pct_change, index=future_dates)
                else:
                    st.warning("Not enough data to generate features for Linear Regression forecast. Filling with zeros.")
                    lr_forecast_pct_change_series = pd.Series(0.0, index=future_dates)


                # --- Prepare X_meta for XGBoost meta-model ---
                # Ensure all base model predictions are aligned by index and converted to numpy arrays
                X_meta_forecast = pd.DataFrame({
                    "SES": ses_forecast.values,
                    "ES": es_forecast.values,
                    "ARIMA": arima_forecast.values,
                    "SARIMA": sarima_forecast.values,
                    "LOG_SARIMA": log_sarima_forecast.values,
                    "Linear_Regression": lr_forecast_pct_change_series.values # This is the predicted pct_change
                }, index=future_dates)

                # --- XGBoost Meta-Model Prediction ---
                # The XGBoost model predicts the future percentage change
                final_xgb_pred_pct_change = best_xgb.predict(X_meta_forecast)

                # Convert predicted pct_change back to actual stock prices
                predicted_prices = [fq_bd_full.iloc[-1]] # Start with the last known actual price
                for pct_change_val in final_xgb_pred_pct_change:
                    next_price = predicted_prices[-1] * (1 + pct_change_val)
                    predicted_prices.append(next_price)

                # Remove the initial actual price, leaving only the 'h' forecasted prices
                final_forecast_prices = pd.Series(predicted_prices[1:], index=future_dates)


                # --- Display Results ---
                st.write("### Predicted Stock Prices")
                st.dataframe(final_forecast_prices.to_frame(name='Predicted Price'))

                st.write("### Forecast Visualization")
                fig, ax = plt.subplots(figsize=(12, 6))
                # Plot recent historical data
                recent_history = fq_bd_full.iloc[-max(100, h):] # Show enough history to context
                ax.plot(recent_history.index, recent_history, label="Historical Data", color='blue')
                # Plot forecasted data
                ax.plot(final_forecast_prices.index, final_forecast_prices, label="Forecasted Prices", color='red', linestyle='--')
                ax.set_title("Stock Price Forecast (Historical and Predicted)")
                ax.set_xlabel("Date")
                ax.set_ylabel("Stock Price")
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                st.pyplot(fig)
        else:
            st.error("Cannot proceed without all models loaded. Please check for missing .joblib files.")
else:
    st.info("Please upload a CSV file to get started with forecasting.")
