
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')

class StockPredictor:
    def __init__(self):
        self.ticker = None
        self.df = None
        self.scaler = None
        self.model = None
        self.prediction_days = 1  # Fixed to 1-day prediction
        self.future_predictions = None
        self.future_dates = None

    def get_user_input(self):
        print("\n" + "="*60)
        print("AI STOCK PRICE PREDICTION SYSTEM".center(60))
        print("="*60 + "\n")

        self.ticker = input("Enter stock ticker (e.g., AAPL, MSFT, TCS.NS): ").strip().upper()

        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=365)

        print(f"\nUsing fixed 1-year historical range: {start_date.date()} to {end_date.date()}")
        print(f"Forecasting for next {self.prediction_days} day only (forced setting).")

        return start_date, end_date

    def fetch_live_data(self, start_date, end_date):
        self.df = yf.download(self.ticker, start_date, end_date)
        if self.df.empty:
            raise ValueError("No data found for the given ticker and date range.")
        self.df['Pct_Change'] = self.df['Close'].pct_change() * 100
        return self.df

    def add_technical_indicators(self):
        self.df['MA_50'] = self.df['Close'].rolling(window=50).mean()
        self.df['MA_200'] = self.df['Close'].rolling(window=200).mean()
        self.df.ffill(inplace=True)
        return self.df

    def prepare_data(self, look_back=60):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(self.df[['Close']].values)
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        split = int(0.9 * len(X))
        return X[:split], X[split:], y[:split], y[split:]

    def build_model(self, input_shape):
        model = Sequential()
        model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, X_train, y_train):
        self.model = self.build_model((X_train.shape[1], 1))
        self.model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_actual = self.scaler.inverse_transform(y_pred)
        return y_test_actual, y_pred_actual

    def predict_future(self, X_last):
        future_predictions = []
        current_sequence = X_last.copy()
        last_date = self.df.index[-1]
        self.future_dates = pd.date_range(start=last_date + dt.timedelta(days=1), periods=self.prediction_days)
        future_df = pd.DataFrame(index=self.future_dates, columns=['Open', 'High', 'Low', 'Close'])
        last_close = self.df['Close'].iloc[-1]

        for i in range(self.prediction_days):
            next_close = self.model.predict(current_sequence.reshape(1, -1, 1))[0, 0]
            next_close = self.scaler.inverse_transform(np.array([[next_close]]))[0, 0]
            open_price = last_close * (1 + np.random.uniform(-0.005, 0.005)) if i == 0 else future_df['Close'].iloc[i-1] * (1 + np.random.uniform(-0.005, 0.005))
            high = next_close * (1 + np.random.uniform(0, 0.01))
            low = next_close * (1 - np.random.uniform(0, 0.01))
            future_df.iloc[i] = [open_price, high, low, next_close]
            next_scaled = self.scaler.transform(np.array([[next_close]]))[0, 0]
            current_sequence = np.append(current_sequence[1:], next_scaled)

        self.future_predictions = future_df
        return future_df

    def visualize_results(self, y_test_actual, y_pred_actual):
        last_date = self.df.index[-1]
        test_dates = pd.date_range(start=self.df.index[-len(y_test_actual)], end=last_date)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.1, subplot_titles=('Historical Data', 'Predictions'))

        fig.add_trace(go.Candlestick(x=self.df.index[-100:], open=self.df['Open'][-100:], high=self.df['High'][-100:], low=self.df['Low'][-100:], close=self.df['Close'][-100:], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index[-100:], y=self.df['MA_50'][-100:], line=dict(color='blue', width=1), name='50-day MA'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index[-100:], y=self.df['MA_200'][-100:], line=dict(color='red', width=1), name='200-day MA'), row=1, col=1)
        fig.add_trace(go.Bar(x=self.df.index[-100:], y=self.df['Volume'][-100:], marker_color='rgba(100, 100, 100, 0.5)', name='Volume', yaxis='y2'), row=1, col=1)

        fig.add_trace(go.Scatter(x=test_dates, y=y_test_actual.flatten(), line=dict(color='blue', width=2), name='Actual Price'), row=2, col=1)
        fig.add_trace(go.Scatter(x=test_dates, y=y_pred_actual.flatten(), line=dict(color='red', width=2, dash='dot'), name='Predicted Price'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.future_dates[:1], y=self.future_predictions['Close'][:1], line=dict(color='green', width=2, dash='dot'), name='1-day Forecast'), row=2, col=1)

        fig.update_layout(title=f'{self.ticker} Stock Analysis & 1-Day Prediction', height=850, xaxis_rangeslider_visible=False, showlegend=True, yaxis2=dict(overlaying='y', side='right', showgrid=False))
        # fig.show()
        
        # Export the plot to an HTML file
        fig.write_html("stock_analysis.html")
        print("Visualization saved as 'stock_analysis.html'. Open this file in a browser to view the plot.")

    def run(self):
        try:
            start_date, end_date = self.get_user_input()
            self.fetch_live_data(start_date, end_date)
            self.add_technical_indicators()
            X_train, X_test, y_train, y_test = self.prepare_data()
            self.train_model(X_train, y_train)
            y_test_actual, y_pred_actual = self.evaluate_model(X_test, y_test)
            self.predict_future(X_test[-1])
            self.visualize_results(y_test_actual, y_pred_actual)
        except Exception as e:
            print(f"\nError: {str(e)}\nPlease check your inputs and try again.")

if __name__ == "__main__":
    predictor = StockPredictor()
    predictor.run()
