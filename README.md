# Stock Price Prediction

## Overview

This project explores forecasting the closing prices of AMGEN stock using a variety of models: traditional time series (ARIMA), deep learning (LSTM, GRU, and their ensemble), and machine learning (XGBoost). By comparing these approaches, the goal is to understand which model best captures the nuances of stock price movements and delivers the lowest prediction error.

## Motivation

Stock price prediction is a challenging endeavor due to the noisy, non-stationary nature of financial data. This project was initiated to:
- Evaluate traditional forecasting methods (ARIMA) versus modern deep learning and ensemble techniques.
- Explore the effectiveness of recurrent neural network architectures (LSTM and GRU) on time series data.
- Investigate the impact of feature engineering and validation strategies on model performance, particularly with XGBoost using lag features and compounded (log) returns.
- Determine the most robust approach for predicting AMGEN’s closing prices, providing insights into potential applications in financial forecasting.

## Models & Approaches

### ARIMA
- **Method:** A classical time-series forecasting model that relies solely on historical closing prices.
- **Findings:** ARIMA's performance was comparable to a naive random model in terms of RMSE, indicating its limitations for this dataset.

### Deep Learning Models: LSTM, GRU, and Ensemble
- **LSTM (Long Short-Term Memory):** Designed to capture long-term dependencies in sequential data.
- **GRU (Gated Recurrent Unit):** A streamlined alternative to LSTM that, in this project, outperformed LSTM in forecasting accuracy.
- **Ensemble:** An ensemble approach combining GRU and LSTM predictions was also explored to leverage the strengths of both models.
- **Findings:** Both LSTM and GRU outperformed ARIMA, with GRU achieving the best results among the deep learning models.

### XGBoost
- **Method:** A gradient boosting framework that was applied using lagged features (lag of close 1 and 2) and compounded (log) returns as the target variable.
- **Validation:** Rolling window validation was employed to ensure robust model evaluation.
- **Findings:** Initial RMSE scores were similar to those from LSTM; however, the rolling window validation strategy significantly reduced RMSE, making XGBoost the best-performing model in this study.

## Data

- **Stock:** AMGEN
- **Feature:** Only the closing prices were used for analysis and prediction (further features will be uploaded later). 
