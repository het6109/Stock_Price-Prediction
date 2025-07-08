# Stock Price Prediction Using LSTM with Manual Feature Engineering

## Project Overview

This project implements a deep learning model using Long Short-Term Memory (LSTM) networks to predict stock closing prices based on historical market data and technical indicators. Unlike relying on external libraries like TA-Lib, this project manually computes key financial indicators such as Moving Averages (MA), Exponential Moving Averages (EMA), Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD) using pandas and numpy.

The model is trained on historical stock data downloaded dynamically via the `yfinance` library, making it adaptable to any publicly traded company by simply changing the ticker symbol.

## Features

- Historical stock data retrieval using `yfinance`
- Manual calculation of technical indicators:
  - Moving Averages (MA10, MA50)
  - Exponential Moving Averages (EMA10, EMA50)
  - Relative Strength Index (RSI)
  - MACD and MACD Signal Line
- Data preprocessing and normalization using MinMaxScaler
- Sequence creation for time-series prediction with LSTM
- Deep LSTM architecture with dropout layers to reduce overfitting
- Early stopping during training to optimize performance
- Model evaluation using RMSE, MAE, MAPE, R-squared, and approximate accuracy percentage
- Visualization of training loss and prediction results

## Technologies Used

- Python 3.x
- TensorFlow / Keras
- yfinance
- pandas, numpy
- scikit-learn
- matplotlib

## Getting Started

### Prerequisites

Make sure you have Python 3.x installed. Install the required Python packages using:

pip install -r requirements.txt


### Running the Project

1. Clone the repository.
   
2. Open the Jupyter Notebook in Jupyter Lab or Jupyter Notebook.
  
3. Run the notebook cells sequentially.  
   You can change the stock ticker symbol in the notebook to predict prices for different companies by modifying the `ticker` variable.

### Changing the Stock Ticker

In the notebook, locate the line:

ticker = 'AAPL' # Example ticker symbol


Replace `'AAPL'` with any valid ticker symbol such as `'MSFT'`, `'TSLA'`, `'GOOG'`, etc.

## Results

The model achieves competitive prediction accuracy with metrics including:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- R-squared (Coefficient of Determination)
- Approximate prediction accuracy percentage derived from MAPE

Visualizations of actual vs predicted closing prices and training loss are provided to assess performance.

## Future Improvements

- Hyperparameter tuning for model optimization  
- Incorporation of additional features like sentiment analysis or macroeconomic indicators  
- Deployment as a web API or dashboard for real-time predictions  
- Experimentation with other deep learning architectures (e.g., GRU, Transformer)




