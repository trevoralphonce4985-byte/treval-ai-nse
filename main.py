# main.py
import os
import logging
import re
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import platform
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import threading

# --- NEW: Import Quantitative Engine ---
from quant_engine.engine import quant_engine
from quant_engine.models.random_walk import RandomWalkModel
from quant_engine.models.moving_average import SimpleMovingAverageModel, ExponentialMovingAverageModel
from quant_engine.models.linear_regression import LinearRegressionModel
from quant_engine.models.gbm import GbmModel

# ----------------------------------------------------
# Logging Configuration
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NSE-API")

# ----------------------------------------------------
# FastAPI App Configuration
# ----------------------------------------------------
app = FastAPI(
    title="Treval AI NSE Live Data API",
    version="2.0.0",  # Major version bump for the new data source
    description="Live NSE stock data powered by Yahoo Finance (RapidAPI) — deployed on Render"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Data Models (Pydantic V2 Compatible)
# ----------------------------------------------------
class Stock(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., SCOM.NSE)")
    company: str = Field(..., description="Full company name")
    price: float = Field(..., gt=0, description="Current stock price")
    change: float = Field(0.0, description="Absolute price change")
    change_percent: float = Field(0.0, description="Percentage price change")
    volume: int = Field(0, ge=0, description="Trading volume")
    dividend_yield: Optional[float] = Field(None, ge=0, le=100, description="Annual dividend yield (%)")
    pe_ratio: Optional[float] = Field(None, gt=0, description="Price-to-Earnings ratio")
    market_cap: Optional[float] = Field(None, gt=0, description="Market capitalization")
    recommendation: str = Field("HOLD", description="Basic recommendation")

class PredictionPoint(BaseModel):
    timestamp: str = Field(..., description="ISO 8601 Timestamp")
    predicted_value: float = Field(..., description="Predicted value")

class PredictionResult(BaseModel):
    model_used: str = Field(..., description="Name of the model used")
    predictions: List[PredictionPoint] = Field(..., description="List of predictions")
    info: Dict[str, Any] = Field(..., description="Model run information")

# ----------------------------------------------------
# Utility Functions
# ----------------------------------------------------
def safe_float(s: str, default: float = 0.0) -> float:
    if s is None:
        return default
    try:
        cleaned = re.sub(r'[^\d.-]', '', str(s))
        if cleaned in ['', '-', '.', '-.']:
            return default
        return float(cleaned)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert '{s}' to float, returning {default}")
        return default

def safe_int(s: str, default: int = 0) -> int:
    if s is None:
        return default
    try:
        cleaned = re.sub(r'[^\d-]', '', str(s))
        if cleaned in ['', '-']:
            return default
        return int(cleaned)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert '{s}' to int, returning {default}")
        return default

# ----------------------------------------------------
# Caching Mechanism (Critical for Rate Limiting)
# ----------------------------------------------------
_cached_stocks = []
_cache_timestamp = datetime.min
_CACHE_DURATION = timedelta(minutes=5)  # Refresh every 5 minutes (well within free tier limits)
_cache_lock = threading.Lock()

def get_cached_stocks():
    global _cached_stocks, _cache_timestamp
    now = datetime.now()
    with _cache_lock:
        if now - _cache_timestamp < _CACHE_DURATION:
            return _cached_stocks.copy()
        else:
            return []

def update_cache(new_stocks: List[Stock]):
    global _cached_stocks, _cache_timestamp
    with _cache_lock:
        _cached_stocks = new_stocks
        _cache_timestamp = datetime.now()

# ----------------------------------------------------
# Data Fetching from Yahoo Finance (RapidAPI) - PRIMARY SOURCE
# ----------------------------------------------------
def fetch_nse_stocks_from_yfinance() -> List[Stock]:
    """
    Fetches live NSE stock data from the Yahoo Finance proxy on RapidAPI.
    This is the new primary data source.
    """
    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    rapidapi_host = "finance-api.p.rapidapi.com"

    if not rapidapi_key:
        logger.error("❌ RAPIDAPI_KEY environment variable not set.")
        return []

    # Use the Yahoo Finance endpoint for NSE stocks
    url = "https://finance-api.p.rapidapi.com/stock/v2/get-quote"
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": rapidapi_host,
        "Content-Type": "application/json"
    }

    # We'll fetch a list of major NSE tickers. For a production app, this would be dynamic.
    nse_tickers = ["SCOM.NSE", "KCB.NSE", "EQTY.NSE", "EABL.NSE", "KPLC.NSE", "COOP.NSE", "NCBA.NSE", "BAT.NSE", "UNGA.NSE", "SASIN.NSE"]
    all_stocks = []

    for ticker in nse_tickers:
        params = {"symbol": ticker}
        try:
            logger.info(f"📡 Fetching data for {ticker} from Yahoo Finance...")
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Parse the response structure (adjust based on actual Yahoo Finance API response)
            # Example structure: data['quoteSummary']['result'][0]
            quote = data.get('quoteSummary', {}).get('result', [{}])[0]

            # Map fields to our Stock model
            ticker_clean = ticker.replace(".NSE", "")
            company = quote.get('shortName') or quote.get('longName') or f"{ticker_clean} PLC"
            price_raw = quote.get('currentPrice') or quote.get('regularMarketPrice')
            price = safe_float(price_raw)
            if price <= 0:
                logger.warning(f"Skipping {ticker} due to invalid price: {price_raw}")
                continue

            change_raw = quote.get('change')
            change = safe_float(change_raw) if change_raw is not None else 0.0

            change_pct_raw = quote.get('changePercent')
            change_percent = safe_float(change_pct_raw) if change_pct_raw is not None else 0.0

            volume_raw = quote.get('volume') or quote.get('regularMarketVolume')
            volume = safe_int(volume_raw) if volume_raw is not None else 0

            dividend_yield = safe_float(quote.get('dividendYield')) if quote.get('dividendYield') else None
            pe_ratio = safe_float(quote.get('trailingPE')) if quote.get('trailingPE') else None
            market_cap_raw = quote.get('marketCap') or quote.get('enterpriseValue')
            market_cap = safe_float(market_cap_raw) if market_cap_raw else None

            stock_obj = Stock(
                ticker=ticker_clean.upper(),
                company=company,
                price=price,
                change=change,
                change_percent=change_percent,
                volume=volume,
                dividend_yield=dividend_yield,
                pe_ratio=pe_ratio,
                market_cap=market_cap,
                recommendation="HOLD"
            )
            all_stocks.append(stock_obj)

        except requests.exceptions.RequestException as e:
            logger.error(f"📡 Network error for {ticker}: {e}")
            # Continue to next ticker, don't fail the whole batch
            continue
        except Exception as e:
            logger.error(f"💥 Parsing error for {ticker}: {e}")
            continue

    logger.info(f"✅ Successfully fetched and processed {len(all_stocks)} stocks from Yahoo Finance.")
    return all_stocks

def fetch_live_stocks() -> List[Stock]:
    """Main entry point for fetching stocks. Uses cache first, then Yahoo Finance."""
    cached = get_cached_stocks()
    if cached:
        logger.info("✅ Returning cached stock data.")
        return cached

    logger.info("🔄 Cache miss. Fetching fresh data from Yahoo Finance...")
    stocks = fetch_nse_stocks_from_yfinance()
    update_cache(stocks)
    return stocks

# ----------------------------------------------------
# NEW: Unified Data Fetching Function (For Extended Data & Predictions)
# ----------------------------------------------------
def fetch_stock_data_for_ticker(ticker: str) -> Optional[Stock]:
    """
    Fetches detailed data for a single ticker using the same Yahoo Finance source.
    Used by prediction and extended data endpoints.
    """
    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    rapidapi_host = "finance-api.p.rapidapi.com"

    if not rapidapi_key:
        return None

    url = "https://finance-api.p.rapidapi.com/stock/v2/get-quote"
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": rapidapi_host,
        "Content-Type": "application/json"
    }

    # Ensure the ticker is in the correct format for Yahoo Finance (e.g., SCOM.NSE)
    if not ticker.endswith(".NSE"):
        ticker = f"{ticker}.NSE"

    params = {"symbol": ticker}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        quote = data.get('quoteSummary', {}).get('result', [{}])[0]

        # Parse the data (same logic as above)
        company = quote.get('shortName') or quote.get('longName') or f"{ticker.replace('.NSE', '')} PLC"
        price = safe_float(quote.get('currentPrice'))
        change = safe_float(quote.get('change')) or 0.0
        change_percent = safe_float(quote.get('changePercent')) or 0.0
        volume = safe_int(quote.get('volume')) or 0
        dividend_yield = safe_float(quote.get('dividendYield')) if quote.get('dividendYield') else None
        pe_ratio = safe_float(quote.get('trailingPE')) if quote.get('trailingPE') else None
        market_cap = safe_float(quote.get('marketCap')) if quote.get('marketCap') else None

        return Stock(
            ticker=ticker.replace(".NSE", "").upper(),
            company=company,
            price=price,
            change=change,
            change_percent=change_percent,
            volume=volume,
            dividend_yield=dividend_yield,
            pe_ratio=pe_ratio,
            market_cap=market_cap,
            recommendation="HOLD"
        )

    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {e}")
        return None

# ----------------------------------------------------
# API Endpoints
# ----------------------------------------------------
@app.on_event('startup')
async def startup_event():
    logger.info("🚀 Starting up and registering quantitative models...")
    try:
        quant_engine.register_model("RandomWalk", RandomWalkModel)
        quant_engine.register_model("SMA", SimpleMovingAverageModel)
        quant_engine.register_model("EMA", ExponentialMovingAverageModel)
        quant_engine.register_model("LinearRegression", LinearRegressionModel)
        quant_engine.register_model("GBM", GbmModel)
        logger.info("✅ All quantitative models registered successfully.")
    except Exception as e:
        logger.error(f"💥 Failed to register quantitative models: {e}")

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Treval AI NSE Live Data API v2.0",
        "source": "Yahoo Finance (RapidAPI)",
        "cloud_hosted": True,
        "python_version": platform.python_version(),
        "timestamp": datetime.now().isoformat(),
        "note": "Deployed on Render. Uses a high-rate-limit free-tier API."
    }

@app.get("/api/v1/stocks")
async def get_all_stocks():
    """Fetch all currently tracked stocks from Yahoo Finance."""
    stocks = fetch_live_stocks()
    if not stocks:
        logger.info("No stocks returned from primary fetch function.")
    return stocks

@app.get("/api/v1/stock/{ticker}")
async def get_stock_details(ticker: str):
    """Get details for a specific stock by ticker symbol."""
    ticker = ticker.upper()
    stock = fetch_stock_data_for_ticker(ticker)
    if stock is None:
        raise HTTPException(status_code=404, detail=f"Stock with ticker '{ticker}' not found in Yahoo Finance data.")
    return stock

@app.get("/api/v1/market-summary")
async def market_summary():
    """Provide a high-level summary of the market based on Yahoo Finance data."""
    stocks = fetch_live_stocks()
    if not stocks:
        return {"message": "No live stock data available."}

    total_stocks = len(stocks)
    gainers = [s for s in stocks if s.change_percent > 0]
    losers = [s for s in stocks if s.change_percent < 0]
    unchanged = [s for s in stocks if s.change_percent == 0]

    avg_change = sum(s.change_percent for s in stocks) / total_stocks if total_stocks > 0 else 0
    total_volume = sum(s.volume for s in stocks)

    return {
        "timestamp": datetime.now().isoformat(),
        "data_source": "Yahoo Finance (RapidAPI)",
        "total_stocks_analyzed": total_stocks,
        "gainers_count": len(gainers),
        "losers_count": len(losers),
        "unchanged_count": len(unchanged),
        "average_change_percent": round(avg_change, 2),
        "total_volume": total_volume,
        "top_gainer": max(gainers, key=lambda x: x.change_percent).ticker if gainers else None,
        "top_loser": min(losers, key=lambda x: x.change_percent).ticker if losers else None
    }

# --- NEW: Prediction Endpoint ---
@app.get("/api/v1/predict/{ticker}", response_model=PredictionResult)
async def get_prediction(ticker: str, model_name: str, horizon: int = 5):
    """Get a prediction for a specific ticker using a specified quantitative model."""
    ticker = ticker.upper()
    logger.info(f"🔮 Requesting prediction for {ticker} using {model_name}, horizon={horizon}")

    # Fetch the stock's current data for the model
    stock_data = fetch_stock_data_for_ticker(ticker)
    if not stock_data:
        raise HTTPException(status_code=404, detail=f"Stock {ticker} not found.")

    # Prepare data for the model (simple: use current price as the last point)
    # In a real implementation, you would fetch historical data.
    raw_data_for_model = [
        {"timestamp": int(time.time() * 1000), "close": stock_data.price}
    ]

    result = quant_engine.run_prediction(model_name, raw_data_for_model, horizon)
    if "error" in result:
        logger.error(f"Prediction failed: {result['error']}")
        raise HTTPException(status_code=500, detail=result["error"])

    # Convert the result to the Pydantic model
    prediction_points = [
        PredictionPoint(timestamp="2026-07-28T10:00:00", predicted_value=stock_data.price * 1.01) # Placeholder
    ]
    return PredictionResult(
        model_used=model_name,
        predictions=prediction_points,
        info={"ticker": ticker, "horizon": horizon}
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "time": datetime.now().isoformat()}
