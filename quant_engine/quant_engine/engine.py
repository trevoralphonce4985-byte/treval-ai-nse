# quant_engine/engine.py
import importlib
from typing import Type, Dict, List
from .model_interface import QuantitativeModel
import logging
logger = logging.getLogger("NSE-API.QuantEngine")

class QuantitativeEngine:
    def __init__(self):
        self.models = {} # Store instances
        self.model_classes: Dict[str, Type[QuantitativeModel]] = {} # Store classes

    def register_model(self, name: str, model_class: Type[QuantitativeModel]):
        """Register a model class."""
        self.model_classes[name] = model_class
        logger.info(f"Registered model: {name}")

    def get_model_instance(self, name: str) -> QuantitativeModel:
        """Get or create an instance of a registered model."""
        if name not in self.models:
            if name in self.model_classes:
                self.models[name] = self.model_classes[name]()
                logger.info(f"Created instance for model: {name}")
            else:
                raise ValueError(f"Model {name} not registered.")
        return self.models[name]

    def run_prediction(self, model_name: str, data: List[Dict], horizon: int) -> Dict:
        """Run a prediction using the specified model."""
        try:
            logger.info(f"Running prediction for {model_name} with horizon {horizon}")
            model = self.get_model_instance(model_name)
            prepared_data = model.prepare_data(data)
            if not model.is_calibrated:
                logger.info(f"Calibrating model: {model_name}")
                model.calibrate(prepared_data)
            prediction_result = model.predict(prepared_data, horizon)
            return {
                "model_used": model_name,
                "predictions": prediction_result,
                "info": model.get_model_info()
            }
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {e}")
            return {"error": f"Prediction failed: {str(e)}"}

# Global instance (or inject via DI if using Hilt for backend - less common)
quant_engine = QuantitativeEngine()
