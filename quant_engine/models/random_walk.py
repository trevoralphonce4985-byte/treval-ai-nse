# quant_engine/models/random_walk.py
import pandas as pd
from ..model_interface import QuantitativeModel
from .. import stoch_proc_lib, stats_lib
import numpy as np

class RandomWalkModel(QuantitativeModel):
    def prepare_data(self, raw_data: list[dict]) -> pd.DataFrame:
        # Assuming raw_data is a list of dicts like [{"timestamp": ..., "close": ...}]
        df = pd.DataFrame(raw_data)
        df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True) # Adjust unit if needed (e.g., 's' for seconds)
        df.sort_index(inplace=True)
        return df[['close']].rename(columns={'close': 'price'})

    def calibrate(self, data: pd.DataFrame):
        # Random Walk assumes no drift, volatility estimated from recent changes
        log_returns = stats_lib.calculate_log_returns(data['price'])
        self.sigma = stats_lib.calculate_std_dev(log_returns)
        self.parameters = {"sigma": self.sigma}
        self.is_calibrated = True
        print(f"RandomWalkModel calibrated: sigma={self.sigma}")

    def predict(self, data: pd.DataFrame, horizon: int) -> list[dict]:
        if not self.is_calibrated:
            raise RuntimeError("Model must be calibrated before prediction.")

        last_price = data['price'].iloc[-1]
        T = horizon # Assuming steps equal to horizon for simplicity
        dt = 1.0 # Time step, often 1 day in discrete models
        N = horizon

        # Generate a single path for prediction (could generate multiple and average/std dev)
        path = stoch_proc_lib.random_walk(last_price, T, self.sigma)
        predicted_prices = path[1:] # Exclude the starting price (last known)

        # Create result list
        predictions = []
        start_time = data.index[-1] # Last known timestamp
        for i, pred_price in enumerate(predicted_prices):
            future_time = start_time + pd.Timedelta(days=i+1) # Adjust based on your time unit
            predictions.append({
                "timestamp": future_time.isoformat(), # Convert back to string for JSON
                "predicted_value": float(pred_price)
            })

        return predictions

    def evaluate(self, actual: pd.DataFrame, predicted: pd.DataFrame) -> dict:
         # Implement evaluation metrics (RMSE, MAE, etc.)
         # This is a simplified example
         actual_values = actual['price'].values
         predicted_values = predicted['predicted_value'].values
         if len(actual_values) != len(predicted_values):
              raise ValueError("Actual and predicted series must have the same length for evaluation.")

         rmse = np.sqrt(np.mean((actual_values - predicted_values) ** 2))
         mae = np.mean(np.abs(actual_values - predicted_values))
         # MAPE calculation
         mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
         return {"rmse": rmse, "mae": mae, "mape": mape}

    def backtest(self, data: pd.DataFrame, window_size: int, horizon: int) -> list[dict]:
         # Implement backtesting logic (e.g., rolling window)
         # This is a simplified example
         results = []
         for i in range(window_size, len(data) - horizon + 1):
             train_data = data.iloc[i-window_size:i]
             actual_future = data.iloc[i:i+horizon]['price']
             # Calibrate on train_data slice
             temp_model = RandomWalkModel() # Create a temporary instance for this window
             temp_model.prepare_data(train_data.reset_index().to_dict('records'))
             temp_model.calibrate(train_data)
             # Predict using the temporary model
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
