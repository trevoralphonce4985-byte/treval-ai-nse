# quant_engine/model_interface.py
from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd # Or your chosen data structure

class QuantitativeModel(ABC):
    """
    Abstract base class for all quantitative models.
    Defines the standard lifecycle and interface.
    """
    def __init__(self):
        self.is_calibrated = False
        self.parameters = {}

    @abstractmethod
    def prepare_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        """Prepare raw data from API into a format suitable for the model."""
        pass

    @abstractmethod
    def calibrate(self, data: pd.DataFrame) -> Dict:
        """Calibrate the model parameters using historical data."""
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame, horizon: int) -> List[Dict]: # Or return a DataFrame
        """Generate predictions based on calibrated parameters."""
        pass

    @abstractmethod
    def evaluate(self, actual: pd.DataFrame, predicted: pd.DataFrame) -> Dict:
        """Evaluate model performance using standard metrics."""
        pass

    @abstractmethod
    def backtest(self, data: pd.DataFrame, window_size: int, horizon: int) -> List[Dict]:
        """Perform backtesting."""
        pass

    def get_model_info(self) -> Dict:
        """Return basic information about the model."""
        return {
            "name": self.__class__.__name__,
            "is_calibrated": self.is_calibrated,
            "parameters": self.parameters
        }
