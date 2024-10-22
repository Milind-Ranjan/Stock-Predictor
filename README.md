# Stock Price Predictor

## Overview
This project implements a stock price prediction model using Long Short-Term Memory (LSTM) networks. The model is designed to predict future stock prices based on historical data retrieved from Yahoo Finance.

## Features
- Fetch historical stock data using the Yahoo Finance API.
- Preprocess data by scaling and transforming it for LSTM input.
- Train an LSTM model to predict stock prices.
- Evaluate model performance using metrics such as Mean Absolute Error (MAE) and R² score.
- Visualize predicted stock prices against actual stock prices.

## Technologies Used
- Python
- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib
- Yahoo Finance API

## Requirements
To run this project, you will need the following Python packages:
- `pandas`
- `numpy`
- `matplotlib`
- `tensorflow`
- `yfinance`
- `sklearn`

You can install the required packages using pip:
```bash
pip install pandas numpy matplotlib tensorflow yfinance scikit-learn
```
## Getting Started

You can install the required packages using pip:
```bash
pip install pandas numpy matplotlib tensorflow yfinance scikit-learn
```
Clone the Repository:
   ```bash
git clone https://github.com/milindranjan/stock-predictor.git
cd stock-price-predictor
```
Run the Script:
```bash
python stock_predictor.py
```
View Results:
After running the script, you will see the predicted stock prices alongside the actual stock prices in a visual plot.

## Model Architecture

### The LSTM model consists of:

•	4 LSTM layers with 50 units each and a dropout layer to prevent overfitting.
•	An output layer predicting the next stock price.

### Performance Metrics

•	Mean Absolute Error (MAE): Measures the average magnitude of errors between predicted and actual prices.
•	R² Score: Indicates the proportion of variance in the dependent variable that can be explained by the independent variable(s).
