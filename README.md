📈 KLSE Maybank – LSTM Backtest & Multi-day Forecast (Pre-trained Model)
Disclaimer: This project is for educational and portfolio purposes only. It is not intended as financial advice.

📜 Project Overview
This project demonstrates how to use a Long Short-Term Memory (LSTM) neural network to forecast Maybank (KLSE: 1155.KL) stock prices.
The model uses historical daily prices and technical indicators to make multi-day predictions.

The app:

Fetches live stock data from Yahoo Finance

Computes technical indicators (SMA, MACD, RSI)

Performs backtesting from 2021 onwards

Forecasts the next N business days

Provides interactive visualization and metrics

🧠 Pre-trained Model
The repository contains a pre-trained LSTM model and scaler stored in the models/ directory.
A pre-trained model means the neural network has already been trained on historical data and can make predictions without retraining.

⚙️ Features
📊 Interactive Stock Chart – Historical, Backtest, and Forecast

📈 Technical Indicators – SMA (10, 30), MACD, RSI

🔍 Backtesting – Compare predicted vs actual prices

📅 Multi-day Forecast – Predicts next N business days

📥 Downloadable Forecast – Export to CSV

📊 Model Performance
The LSTM model was evaluated on backtest data from January 2021 onwards.

Metric	Value	Interpretation
RMSE	0.3555	Predictions deviate by ~RM 0.36 on average, penalizing large errors.
MAE	0.2841	Average absolute difference is ~RM 0.28 per day.
MAPE	3.17%	Model achieves ~96.8% accuracy on average.

✅ Conclusion: With a MAPE under 5%, the model demonstrates strong predictive capability for Maybank’s daily closing prices.
⚠ Disclaimer: Stock prices are influenced by unpredictable market factors.

🚀 Installation & Usage
1️⃣ Clone Repository
git clone https://github.com/<your-username>/KLSE-Maybank-LSTM.git
cd KLSE-Maybank-LSTM

2️⃣ Create Virtual Environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

3️⃣ Install Requirements
pip install -r requirements.txt

4️⃣ Run the App
streamlit run app.py


🗂 Folder Structure
graphql
Copy
Edit
├── app.py                  # Main Streamlit app
├── models/                 # Pre-trained LSTM model & scaler
│   ├── 1155_KL_lstm.h5
│   ├── 1155_KL_scaler.pkl
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
📌 Notes
Data source: Yahoo Finance

Forecast horizon: Adjustable in app (1–14 business days)

The project is adaptable for other tickers with minimal changes

📜 License
This project is open-source under the MIT License.