# quant_engine/models/gbm.py
import pandas as pd
import numpy as np
from ..model_interface import QuantitativeModel
from .. import stats_lib, stoch_proc_lib
import logging
logger = logging.getLogger("NSE-API.GBMModel")

class GbmModel(QuantitativeModel):
    def __init__(self, num_simulations: int = 1000):
         super().__init__()
         self.num_simulations = num_simulations # Number of paths for Monte Carlo if needed for confidence

    def prepare_data(self, raw_data: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(raw_data)
        df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True) # Adjust unit if needed
        df.sort_index(inplace=True)
        return df[['close']].rename(columns={'close': 'price'})

    def calibrate(self, data: pd.DataFrame):
        # Calculate drift (mu) and volatility (sigma) from log returns
        log_returns = stats_lib.calculate_log_returns(data['price'])

        # Assuming daily data for drift/vol calculation
        trading_days_per_year = 252
        dt = 1.0 / trading_days_per_year

        mu = stats_lib.calculate_mean(log_returns) / dt # Annualized drift
        sigma = stats_lib.calculate_std_dev(log_returns) / np.sqrt(dt) # Annualized volatility

        self.parameters = {
            "mu": mu,
            "sigma": sigma,
            "dt": dt
        }
        self.is_calibrated = True
        logger.info(f"GBM Model calibrated: mu={mu:.6f}, sigma={sigma:.6f}")

    def predict(self, data: pd.DataFrame, horizon: int) -> list[dict]:
        if not self.is_calibrated:
            raise RuntimeError("Model must be calibrated before prediction.")

        mu = self.parameters["mu"]
        sigma = self.parameters["sigma"]
        dt = self.parameters["dt"]

        S0 = data['price'].iloc[-1] # Last known price
        T = horizon / 252.0 # Horizon in years (assuming daily steps)
        N = horizon # Number of steps

        # Generate one prediction path (could generate multiple for confidence bands)
        # Use the discrete solution: S_{t+dt} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        predictions = []
        current_price = S0
        start_time = data.index[-1] # Last known timestamp

        for i in range(1, N + 1): # Predict for i=1, 2, ..., N steps ahead
            Z = np.random.normal(0, 1) # Standard normal random variable
            exponent = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            next_price = current_price * np.exp(exponent)

            future_time = start_time + pd.Timedelta(days=i) # Adjust based on your time unit
            predictions.append({
                "timestamp": future_time.isoformat(),
                "predicted_value": float(next_price)
            })
            current_price = next_price # Update for next step

        return predictions

    def evaluate(self, actual: pd.DataFrame, predicted: pd.DataFrame) -> dict:
         actual_values = actual['price'].values
         predicted_values = predicted['predicted_value'].values
         if len(actual_values) != len(predicted_values):
              raise ValueError("Actual and predicted series must have the same length for evaluation.")

         rmse = np.sqrt(np.mean((actual_values - predicted_values) ** 2))
         mae = np.mean(np.abs(actual_values - predicted_values))
         mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
         return {"rmse": rmse, "mae": mae, "mape": mape}

    def backtest(self, data: pd.DataFrame, window_size: int, horizon: int) -> list[dict]:
         # Implement backtesting logic (e.g., rolling window)
         results = []
         for i in range(window_size, len(data) - horizon + 1):
             train_data = data.iloc[i-window_size:i]
             actual_future = data.iloc[i:i+horizon]['price']

             # Calibrate on train_data slice
             temp_model = GbmModel(num_simulations=self.num_simulations) # Temporary instance
             temp_model.prepare_data(train_data.reset_index().to_dict('records'))
             temp_model.calibrate(train_data)

             # Predict using the temporary model (single path for simplicity in backtest)
             pred_result = temp_model.predict(train_data, horizon)
             pred_df = pd.DataFrame(pred_result)

             # Evaluate this specific prediction window
             eval_metrics = self.evaluate(actual_future.to_frame('price'), pred_df)
             results.append({
                 "start_index": i-window_size,
                 "end_index": i-1,
                 "prediction_start_index": i,
                 "prediction_end_index": i+horizon-1,
                 "metrics": eval_metrics
             })
         return results

