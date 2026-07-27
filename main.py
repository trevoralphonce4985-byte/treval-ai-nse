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
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import threading
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib

# --- IMPORT QUANTITATIVE ENGINE ---
try:
    from quant_engine.engine import quant_engine
    from quant_engine.models.random_walk import RandomWalkModel
    from quant_engine.models.moving_average import SimpleMovingAverageModel, ExponentialMovingAverageModel
    from quant_engine.models.linear_regression import LinearRegressionModel
    from quant_engine.models.gbm import GbmModel
    QUANT_ENGINE_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger("NSE-API")
    logger.warning(f"Quantitative Engine not available: {e}")
    QUANT_ENGINE_AVAILABLE = False
    quant_engine = None

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
    version="2.1.0",  # Updated version
    description="Live NSE stock data powered by Yahoo Finance (RapidAPI) — deployed on Render. Includes prediction caching and rate limiting."
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

# --- NEW MODELS FOR PREDICTION CACHING & RATE LIMITING ---
class CachedPredictionResult(BaseModel):
    model_used: str = Field(..., description="Name of the model used for prediction")
    predictions: List[PredictionPoint] = Field(..., description="List of predicted values with timestamps")
    info: Dict[str, Any] = Field(..., description="Additional information about the model run")
    cached: bool = Field(..., description="True if the response was served from the cache")
    generated_at: str = Field(..., description="ISO 8601 timestamp when the prediction was generated or cached")
    expires_at: str = Field(..., description="ISO 8601 timestamp when the cached prediction expires")
    cache_age_seconds: int = Field(..., description="Age of the cached prediction in seconds (0 if generated fresh)")

class CacheStatusResponse(BaseModel):
    ticker: str = Field(..., description="The stock ticker symbol checked")
    model_name: str = Field(..., description="The model name checked")
    horizon: int = Field(..., description="The prediction horizon checked")
    cached: bool = Field(..., description="Whether a valid cached result exists")
    expires_at: Optional[str] = Field(None, description="ISO 8601 timestamp when the cached prediction expires (null if not cached)")
    generated_at: Optional[str] = Field(None, description="ISO 8601 timestamp when the cached prediction was generated (null if not cached)")

class RateLimitErrorResponse(BaseModel):
    error: str = Field(..., description="Human-readable error message")
    retry_after_seconds: int = Field(..., description="Number of seconds to wait before retrying")
    limit: int = Field(..., description="Maximum allowed requests per time window")
    remaining: int = Field(..., description="Remaining requests in the current time window")
    reset_at: str = Field(..., description="ISO 8601 timestamp when the rate limit resets")


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
# Caching Mechanism (Critical for Rate Limiting & Data Accuracy)
# ----------------------------------------------------
_cached_stocks = []
_cache_timestamp = datetime.min
_CACHE_DURATION = timedelta(minutes=15)  # Refresh every 15 minutes (balance between freshness and rate limits)
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

    # List of major NSE tickers for initial fetch
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

# --- PREDICTION CACHING & RATE LIMITING IMPLEMENTATION ---
# --- CONFIGURATION ---
MARKET_OPEN_HOUR = 9  # Assuming market opens at 9 AM EAT
MARKET_CLOSE_HOUR = 4 # Assuming market closes at 4 PM EAT
CACHE_DURATION_MARKET_HOURS = timedelta(minutes=15)
CACHE_DURATION_OFF_MARKET_HOURS = timedelta(hours=4)
RATE_LIMIT_WINDOW_SECONDS = 3600  # 60 minutes
RATE_LIMIT_MAX_REQUESTS = 15
# --- END CONFIGURATION ---

# --- IN-MEMORY CACHE STRUCTURE ---
@dataclass
class CachedPredictionEntry:
    result: CachedPredictionResult
    expires_at: datetime
    generation_lock: asyncio.Lock = field(default_factory=asyncio.Lock) # For deduplication

_prediction_cache: Dict[str, CachedPredictionEntry] = {}
_cache_lock_pred = threading.RLock() # Thread-safe access to the prediction cache dict

# --- IN-MEMORY RATE LIMITING STRUCTURE ---
_rate_limit_buckets: Dict[str, Dict[datetime, int]] = defaultdict(lambda: defaultdict(int))
_rate_limit_lock = threading.RLock()

def _get_market_status():
    now_eat = datetime.now() # Assuming server time is EAT, adjust if necessary
    current_hour = now_eat.hour
    is_open = MARKET_OPEN_HOUR <= current_hour < MARKET_CLOSE_HOUR
    return is_open

def _generate_cache_key(ticker: str, model_name: str, horizon: int, data_version: str = "") -> str:
    """
    Generates a unique key for the prediction cache based on parameters.
    Including data_version allows cache busting when market data updates.
    """
    key_str = f"{ticker}:{model_name}:{horizon}:{data_version}".lower()
    return hashlib.sha256(key_str.encode()).hexdigest()

def _get_or_create_cache_entry(cache_key: str) -> CachedPredictionEntry:
    """Gets an existing cache entry or creates a new one if it doesn't exist."""
    with _cache_lock_pred:
        if cache_key not in _prediction_cache:
            _prediction_cache[cache_key] = CachedPredictionEntry(
                result=None, # Will be populated when calculated
                expires_at=datetime.min # Will be set when calculated
            )
        return _prediction_cache[cache_key]

def _check_cache_validity(entry: CachedPredictionEntry) -> bool:
    """Checks if a cache entry is still valid."""
    return datetime.now() < entry.expires_at

def _calculate_expiration_time() -> datetime:
    """Calculates the expiration time based on market hours."""
    is_open = _get_market_status()
    duration = CACHE_DURATION_MARKET_HOURS if is_open else CACHE_DURATION_OFF_MARKET_HOURS
    return datetime.now() + duration

def _check_rate_limit(user_id: str) -> tuple[bool, Dict[str, any]]:
    """
    Checks if a user has exceeded the rate limit.
    Returns (is_limited, error_response_dict).
    """
    now = datetime.now()
    window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW_SECONDS)

    with _rate_limit_lock:
        # Clean up old buckets outside the window
        user_buckets = _rate_limit_buckets[user_id]
        keys_to_remove = [k for k in user_buckets.keys() if k < window_start]
        for k in keys_to_remove:
            del user_buckets[k]

        # Calculate current count within the window
        current_count = sum(user_buckets.values())
        remaining = RATE_LIMIT_MAX_REQUESTS - current_count

        if current_count >= RATE_LIMIT_MAX_REQUESTS:
            reset_at = now + timedelta(seconds=RATE_LIMIT_WINDOW_SECONDS - ((now - window_start).seconds % RATE_LIMIT_WINDOW_SECONDS))
            return True, {
                "error": "Rate limit exceeded",
                "retry_after_seconds": int((reset_at - now).total_seconds()),
                "limit": RATE_LIMIT_MAX_REQUESTS,
                "remaining": 0,
                "reset_at": reset_at.isoformat()
            }

        # Increment request count for the current second bucket
        user_buckets[now.replace(microsecond=0)] += 1

    return False, None

def _get_current_data_version():
    """Returns a string representing the current version of the underlying market data."""
    global _cache_timestamp # From the existing caching mechanism
    return f"v1_{int(_cache_timestamp.timestamp())}"


# ----------------------------------------------------
# API Endpoints
# ----------------------------------------------------
@app.on_event('startup')
async def startup_event():
    if QUANT_ENGINE_AVAILABLE:
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
    else:
        logger.warning("⚠️ Quantitative Engine not available. Prediction endpoints will return errors.")


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Treval AI NSE Live Data API v2.1",
        "source": "Yahoo Finance (RapidAPI)",
        "features": ["Live Data", "Prediction Caching", "Rate Limiting"],
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

# --- NEW: Prediction Cache Status Endpoint ---
@app.get("/api/v1/predictions/cache-status", response_model=CacheStatusResponse)
async def get_cache_status(ticker: str, model_name: str, horizon: int):
    """Check if a prediction result is available in the cache."""
    ticker = ticker.upper()
    data_version = _get_current_data_version()
    cache_key = _generate_cache_key(ticker, model_name, horizon, data_version)

    with _cache_lock_pred:
        entry = _prediction_cache.get(cache_key)

    if entry and _check_cache_validity(entry):
        return CacheStatusResponse(
            ticker=ticker,
            model_name=model_name,
            horizon=horizon,
            cached=True,
            expires_at=entry.expires_at.isoformat(),
            generated_at=entry.result.generated_at # Assumes result is populated when cached
        )
    else:
        return CacheStatusResponse(
            ticker=ticker,
            model_name=model_name,
            horizon=horizon,
            cached=False,
            expires_at=None,
            generated_at=None
        )

# --- NEW: Prediction Endpoint with Caching & Rate Limiting ---
@app.get("/api/v1/predict/{ticker}", response_model=CachedPredictionResult)
async def get_prediction(ticker: str, model_name: str, horizon: int = 5, user_id: str = "default_user"):
    """Get a prediction for a specific ticker using a specified quantitative model."""
    ticker = ticker.upper()
    logger.info(f"🔮 Requesting prediction for {ticker} using {model_name}, horizon={horizon}")

    if not QUANT_ENGINE_AVAILABLE:
        raise HTTPException(status_code=500, detail="Prediction engine is not available.")

    # --- Rate Limiting Check ---
    is_limited, rate_limit_error = _check_rate_limit(user_id)
    if is_limited:
        return JSONResponse(status_code=429, content=rate_limit_error)

    # --- Generate Cache Key ---
    data_version = _get_current_data_version()
    cache_key = _generate_cache_key(ticker, model_name, horizon, data_version)

    # --- Check Cache First ---
    with _cache_lock_pred:
        entry = _prediction_cache.get(cache_key)

    if entry and _check_cache_validity(entry):
        age = int((datetime.now() - datetime.fromisoformat(entry.result.generated_at)).total_seconds())
        logger.info(f"✅ Prediction cache HIT for {ticker}, model {model_name}, horizon {horizon}. Age: {age}s")
        # Return the cached result with metadata
        updated_result = entry.result.copy(update={
            "cached": True,
            "cache_age_seconds": age
        })
        return updated_result

    # --- Cache Miss or Expired: Acquire Generation Lock ---
    # Use the entry's specific lock for deduplication
    entry = _get_or_create_cache_entry(cache_key)
    async with entry.generation_lock: # Await the lock if another request is already generating
        # Double-check cache validity after acquiring lock
        if entry.result is not None and _check_cache_validity(entry):
            age = int((datetime.now() - datetime.fromisoformat(entry.result.generated_at)).total_seconds())
            logger.info(f"✅ Prediction cache HIT (post-lock) for {ticker}, model {model_name}, horizon {horizon}. Age: {age}s")
            updated_result = entry.result.copy(update={
                "cached": True,
                "cache_age_seconds": age
            })
            return updated_result

        # --- Generate Prediction ---
        logger.info(f"🔄 Prediction cache MISS/EXPIRED. Generating for {ticker}, model {model_name}, horizon {horizon}.")

        # Fetch the stock's current data for the model (or historical data if needed for the model)
        stock_data = fetch_stock_data_for_ticker(ticker)
        if not stock_data:
            raise HTTPException(status_code=404, detail=f"Stock {ticker} not found.")

        # Prepare data for the model (example: use current price as the last point, or fetch historical)
        # This is a simplified preparation, real models need historical data.
        # For now, let's just use the current price as a single point for models expecting a series.
        # A real implementation would fetch historical data based on model needs.
        # Placeholder: fetch_historical_data_for_ticker
        # raw_data_for_model = fetch_historical_data_for_ticker(ticker, days_back=30)
        raw_data_for_model = [{"timestamp": int(datetime.now().timestamp() * 1000), "close": stock_data.price}] # Simplified

        # Run the prediction using the QuantitativeEngine
        result = quant_engine.run_prediction(model_name, raw_data_for_model, horizon)
        if "error" in result:
            logger.error(f"Prediction failed: {result['error']}")
            # Do NOT cache failed responses as successful ones.
            # Re-raise the HTTP error from the engine.
            raise HTTPException(status_code=500, detail=result["error"])

        # --- Create and Store Result in Cache ---
        now = datetime.now()
        expires_at = _calculate_expiration_time()
        cache_age = 0 # Freshly generated

        cached_result = CachedPredictionResult(
            model_used=result["model_used"],
            predictions=[PredictionPoint(timestamp=p["timestamp"], predicted_value=p["predicted_value"]) for p in result["predictions"]],
            info=result["info"],
            cached=False, # This instance is fresh
            generated_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
            cache_age_seconds=cache_age
        )

        # Update the cache entry atomically
        with _cache_lock_pred:
            entry.result = cached_result
            entry.expires_at = expires_at

        logger.info(f"✅ Prediction generated and cached for {ticker}, model {model_name}, horizon {horizon}. Expires at: {expires_at.isoformat()}")
        return cached_result # Return the fresh result


@app.get("/health")
async def health_check():
    return {"status": "healthy", "time": datetime.now().isoformat()}

# --- END OF FILE ---
