# SPY LSTM Forecasting with TensorFlow

This project forecasts SPY (S&P 500 ETF) using a Bidirectional LSTM neural network built in TensorFlow/Keras. The model integrates historical OHLCV data, MAG7 equity prices, Commodities prices, Crypto prices, Forex prices, Indices prices, Macro prices, Sector prices , Macroeconomic indicators, and technical indicators like RSI, MACD, and Bollinger Bands.

---

## 📊 Project Highlights

- Forecasting 30-day SPY movement using 252-day rolling window
- Bidirectional LSTM with Layer Normalization and Dropout
- Features include:
  - MAG7 stock prices (Meta, Apple, Google, Amazon, Microsoft, Nvidia, Tesla)
  - Commodities prices (CLF, GCF, NGF, SIF, ZCF, ZWF)
  - Crypto prices ( BTCUSD, DOGEUSD, ETHUSD, SOLUSD, XRPUSD)
  - Forex Prices ( AUDUSD, EURUSD, GBPUSD, USDCAD, USDJPY)
  - Indices Prices ( DAX, FTSE, GSPC, HSI, N225, QQQ)
  - Macro Prices ( MOVE, TIP, TLT, TNX, VIX)
  - Sector Prices (XLB, XLC, XLE, XLF, XLI, XLK, XLRE, XLU, XLV, XLY)
  - Technical Indicators: RSI, MACD, Bollinger Bands
  - Granger causality + Pearson correlation filtering
- Trained using Kaggle GPU environment (Tesla T4, CUDA 11)

---

## 🧠 Technologies Used

- TensorFlow 2.x
- Keras (Sequential API)
- NumPy & pandas
- scikit-learn (StandardScaler, train_test_split)
- statsmodels (Granger causality)
- ta (Technical Analysis indicators)
- Plotly (for interactive charts)
- Matplotlib

---

## 🗂️ Project Structure

