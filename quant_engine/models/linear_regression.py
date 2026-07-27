# quant_engine/models/linear_regression.py
import pandas as pd
import numpy as np
from scipy import stats # For linear regression
from ..model_interface import QuantitativeModel
from .. import stats_lib
import logging
logger = logging.getLogger("NSE-API.LinearRegressionModel")

class LinearRegressionModel(QuantitativeModel):
    def __init__(self, lookback_window: int = 20):
        super().__init__()
        self.lookback_window = lookback_window # How much history to use for fitting

    def prepare_data(self, raw_data: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(raw_data)
        df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True) # Adjust unit if needed
        df.sort_index(inplace=True)
        # Add a time index column (e.g., days since start) for the X variable
        df['time_index'] = range(len(df))
        return df[['time_index', 'close']].rename(columns={'close': 'price'})

    def calibrate(self, data: pd.DataFrame):
        if len(data) < 2:
             raise ValueError("Need at least 2 data points for linear regression.")

        # Use the last 'lookback_window' points, or all if less available
        fit_data = data.tail(self.lookback_window) if len(data) >= self.lookback_window else data

        x_vals = fit_data['time_index'].values
        y_vals = fit_data['price'].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)

        self.parameters = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_err": std_err
        }
        self.is_calibrated = True
        logger.info(f"LinearRegressionModel calibrated: slope={slope:.4f}, intercept={intercept:.4f}, R²={r_value**2:.4f}")

    def predict(self, data: pd.DataFrame, horizon: int) -> list[dict]:
        if not self.is_calibrated:
            raise RuntimeError("Model must be calibrated before prediction.")

        slope = self.parameters["slope"]
        intercept = self.parameters["intercept"]

        # Get the last known time index
        last_time_index = data['time_index'].iloc[-1]

        predictions = []
        start_time = data.index[-1] # Last known timestamp
        for i in range(1, horizon + 1): # Predict for i=1, 2, ..., horizon steps ahead
            future_time_index = last_time_index + i
            predicted_price = slope * future_time_index + intercept

            future_time = start_time + pd.Timedelta(days=i) # Adjust based on your time unit
            predictions.append({
                "timestamp": future_time.isoformat(),
                "predicted_value": float(predicted_price)
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
         # Implement backtesting logic (e.g., rolling window)
         # Use the specified window_size for calibration
         results = []
         for i in range(window_size, len(data) - horizon + 1):
             train_data = data.iloc[i-window_size:i].copy() # Need to recalculate time_index for slice
             train_data['time_index'] = range(len(train_data))

             actual_future = data.iloc[i:i+horizon]['price']

             # Calibrate on train_data slice
             temp_model = LinearRegressionModel(lookback_window=self.lookback_window) # Temporary instance
             temp_model.prepare_data(train_data.reset_index().to_dict('records')) # Re-prepares with new time_index
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

