# quant_engine/models/moving_average.py
import pandas as pd
from ..model_interface import QuantitativeModel
from .. import stats_lib
import numpy as np

class SimpleMovingAverageModel(QuantitativeModel):
    def __init__(self, window_size: int = 10):
        super().__init__()
        self.window_size = window_size # Default window size

    def prepare_data(self, raw_data: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(raw_data)
        df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True) # Adjust unit if needed
        df.sort_index(inplace=True)
        return df[['close']].rename(columns={'close': 'price'})

    def calibrate(self, data: pd.DataFrame):
        # SMA doesn't really calibrate, just sets parameters
        self.parameters = {"window_size": self.window_size}
        self.is_calibrated = True
        print(f"SMA Model calibrated: window_size={self.window_size}")

    def predict(self, data: pd.DataFrame, horizon: int) -> list[dict]:
        if not self.is_calibrated:
            raise RuntimeError("Model must be calibrated before prediction.")

        # Use the last calculated SMA value as the prediction for all future points
        # This is a very simple prediction rule for SMA
        sma_series = stats_lib.calculate_moving_average(data['price'], self.window_size)
        if sma_series.empty:
             raise RuntimeError("Not enough data to calculate SMA for prediction.")
        last_sma_value = sma_series.iloc[-1] # Last calculated SMA

        predictions = []
        start_time = data.index[-1] # Last known timestamp
        for i in range(horizon):
            future_time = start_time + pd.Timedelta(days=i+1) # Adjust based on your time unit
            predictions.append({
                "timestamp": future_time.isoformat(),
                "predicted_value": float(last_sma_value)
            })

        return predictions

    def evaluate(self, actual: pd.DataFrame, predicted: pd.DataFrame) -> dict:
         # Implement evaluation metrics (RMSE, MAE, etc.)
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
         # This is a simplified example, recalculating SMA at each step
         results = []
         for i in range(window_size, len(data) - horizon + 1):
             train_data = data.iloc[i-window_size:i]
             actual_future = data.iloc[i:i+horizon]['price']

             # Predict using current SMA
             sma_val = stats_lib.calculate_moving_average(train_data['price'], self.window_size).iloc[-1]
             pred_series = pd.Series([sma_val] * horizon, index=actual_future.index) # Predict constant SMA value

             # Evaluate this specific prediction window
             eval_metrics = self.evaluate(actual_future.to_frame('price'), pred_series.to_frame('predicted_value'))
             results.append({
                 "start_index": i-window_size,
                 "end_index": i-1,
                 "prediction_start_index": i,
                 "prediction_end_index": i+horizon-1,
                 "metrics": eval_metrics
             })
         return results


# Example for EMA could be similar
class ExponentialMovingAverageModel(QuantitativeModel):
    def __init__(self, span: int = 10):
        super().__init__()
        self.span = span # Default span for EMA

    def prepare_data(self, raw_data: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(raw_data)
        df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True) # Adjust unit if needed
        df.sort_index(inplace=True)
        return df[['close']].rename(columns={'close': 'price'})

    def calibrate(self, data: pd.DataFrame):
        # EMA doesn't really calibrate, just sets parameters
        self.parameters = {"span": self.span}
        self.is_calibrated = True
        print(f"EMA Model calibrated: span={self.span}")

    def predict(self, data: pd.DataFrame, horizon: int) -> list[dict]:
        if not self.is_calibrated:
            raise RuntimeError("Model must be calibrated before prediction.")

        ema_series = stats_lib.calculate_exponential_moving_average(data['price'], self.span)
        if ema_series.empty:
             raise RuntimeError("Not enough data to calculate EMA for prediction.")
        last_ema_value = ema_series.iloc[-1]

        predictions = []
        start_time = data.index[-1]
        for i in range(horizon):
            future_time = start_time + pd.Timedelta(days=i+1) # Adjust based on your time unit
            predictions.append({
                "timestamp": future_time.isoformat(),
                "predicted_value": float(last_ema_value)
            })

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
         # Similar to SMA backtest, using EMA
         results = []
         for i in range(window_size, len(data) - horizon + 1):
             train_data = data.iloc[i-window_size:i]
             actual_future = data.iloc[i:i+horizon]['price']

             ema_val = stats_lib.calculate_exponential_moving_average(train_data['price'], self.span).iloc[-1]
             pred_series = pd.Series([ema_val] * horizon, index=actual_future.index)

             eval_metrics = self.evaluate(actual_future.to_frame('price'), pred_series.to_frame('predicted_value'))
             results.append({
                 "start_index": i-window_size,
                 "end_index": i-1,
                 "prediction_start_index": i,
                 "prediction_end_index": i+horizon-1,
                 "metrics": eval_metrics
             })
         return results

