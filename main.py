# main.py (v2.1.0 - MODIFIED to use NSE API)
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
    version="2.1.1",  # Updated version to reflect NSE API change
    description="Live NSE stock data powered by NSE Scraper API (RapidAPI) — deployed on Render. Includes prediction caching and rate limiting."
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
    ticker: str = Field(..., description="Stock ticker symbol (e.g., SCOM)")
    company: str = Field(..., description="Full company name")
    price: float = Field(..., gt=0, description="Current stock price")
    change: float = Field(0.0, description="Absolute price change")
    change_percent: float = Field(0.0, description="Percentage price change")
    volume: int = Field(0, ge=0, description="Trading volume")
    # Dividend yield, P/E, Market Cap might not be available from the NSE Scraper API directly
    # dividend_yield: Optional[float] = Field(None, ge=0, le=100, description="Annual dividend yield (%)")
    # pe_ratio: Optional[float] = Field(None, gt=0, description="Price-to-Earnings ratio")
    # market_cap: Optional[float] = Field(None, gt=0, description="Market capitalization")
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

def parse_change_string(change_str: str) -> tuple[float, float]:
    """
    Parses a change string like "+2.50 (+5.82%)" or "-0.50 (-1.71%)".
    Returns (change_amount, change_percentage).
    """
    if not change_str:
        return 0.0, 0.0

    import re
    # Match the first number (change amount) and the number inside parentheses (change percentage)
    match = re.search(r'([+-]?\d+\.?\d*)\s*\(\s*([+-]?\d+\.?\d*)%\s*\)', change_str)
    if match:
        change_amount = float(match.group(1))
        change_percent = float(match.group(2))
        return change_amount, change_percent
    else:
        logger.warning(f"Could not parse change string: {change_str}")
        # If parsing fails, try to extract percentage only if format is like "+1.28%"
        pct_match = re.search(r'([+-]?\d+\.?\d*)%', change_str)
        if pct_match:
            return 0.0, float(pct_match.group(1)) # Assume 0 change amount if only % is found
        return 0.0, 0.0

# ----------------------------------------------------
# Caching Mechanism (Critical for Rate Limiting & Data Accuracy)
# ----------------------------------------------------
_cached_stocks = []
_cache_timestamp = datetime.min
_CACHE_DURATION = timedelta(minutes=5)  # Refresh every 5 minutes to stay within NSE API free tier limits and get fresher data
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
# Data Fetching from NSE Scraper API (RapidAPI) - PRIMARY SOURCE
# ----------------------------------------------------
def fetch_nse_stocks_from_nse_scraper() -> List[Stock]:
    """
    Fetches live NSE stock data from the NSE Scraper API on RapidAPI.
    This is the new primary data source.
    """
    rapidapi_key = os.getenv("RAPIDAPI_KEY") # Use the same key for the NSE API
    rapidapi_host = "nairobi-stock-exchange-nse.p.rapidapi.com" # New host

    if not rapidapi_key:
        logger.error("❌ RAPIDAPI_KEY environment variable not set.")
        return []

    # Use the NSE Scraper API endpoint
    url = f"https://{rapidapi_host}/stocks" # Base endpoint for all stocks
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": rapidapi_host,
        "Content-Type": "application/json"
    }
    # Optional params can be added here if needed (e.g., limit, sort)
    params = {}

    try:
        logger.info(f"📡 Fetching data from NSE Scraper API: {url.split('/')[2]}")
        response = requests.get(url, headers=headers, params=params, timeout=15)
        # Check for rate limit
        if response.status_code == 429:
             logger.warning("429 Too Many Requests received from NSE Scraper API.")
             # Return cached data if available, otherwise empty list
             cached = get_cached_stocks()
             if cached:
                 logger.info("Returning cached data due to API rate limit.")
                 return cached
             else:
                 logger.error("No cached data available and API returned 429.")
                 return []
        response.raise_for_status()
        data = response.json()

        # Check the structure: {"success": true, "data": [...], "meta": {...}}
        if not data.get('success'):
            logger.error(f"NSE Scraper API returned success=false: {data}")
            return []

        stock_list_raw = data.get('data', [])
        if not isinstance(stock_list_raw, list):
            logger.error(f"Expected list of stocks in 'data', got {type(stock_list_raw)}. Response: {data}")
            return []

        all_stocks = []
        for item in stock_list_raw:
            # Map fields from the NSE Scraper API response to our Stock model
            # Example item: {"ticker": "EQTY", "name": "Equity Group Holdings Plc", "volume": "1,234,567", "price": "45.50", "change": "+2.50 (+5.82%)"}
            ticker = item.get('ticker')
            if not ticker:
                logger.warning(f"Skipping item due to missing ticker: {item}")
                continue

            name = item.get('name', f"{ticker} PLC")
            price_raw = item.get('price')
            price = safe_float(price_raw)
            if price <= 0:
                logger.warning(f"Skipping {ticker} due to invalid/missing price: {price_raw}")
                continue

            change_str = item.get('change')
            change_amount, change_percent = parse_change_string(change_str)

            volume_raw = item.get('volume')
            # Remove commas from volume string like "1,234,567"
            volume_str_cleaned = volume_raw.replace(',', '') if isinstance(volume_raw, str) else str(volume_raw)
            volume = safe_int(volume_str_cleaned) if volume_str_cleaned else 0

            stock_obj = Stock(
                ticker=ticker.upper(), # Ensure uppercase
                company=name,
                price=price,
                change=change_amount,
                change_percent=change_percent,
                volume=volume,
                # Dividend yield, P/E, Market Cap might not be available, leave as None/default
                # dividend_yield=None,
                # pe_ratio=None,
                # market_cap=None,
                recommendation="HOLD"
            )
            all_stocks.append(stock_obj)

        logger.info(f"✅ Successfully fetched and processed {len(all_stocks)} stocks from NSE Scraper API.")
        return all_stocks

    except requests.exceptions.RequestException as e:
        logger.error(f"📡 Network error for NSE Scraper API: {e}")
        # Log response details if available
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            logger.error(f"Response text (first 200 chars): {e.response.text[:200]}...")
        # Return cached data if available, otherwise empty list
        cached = get_cached_stocks()
        if cached:
             logger.info("Returning cached data due to network error.")
             return cached
        else:
             logger.error("No cached data available and API call failed.")
             return []
    except Exception as e:
        logger.error(f"💥 Parsing error for NSE Scraper API: {e}")
        logger.exception("Full traceback:")
        # Return cached data if available, otherwise empty list
        cached = get_cached_stocks()
        if cached:
             logger.info("Returning cached data due to parsing error.")
             return cached
        else:
             logger.error("No cached data available and API call failed.")
             return []

def fetch_live_stocks() -> List[Stock]:
    """Main entry point for fetching stocks. Uses cache first, then NSE Scraper API."""
    cached = get_cached_stocks()
    if cached:
        logger.info("✅ Returning cached stock data.")
        return cached

    logger.info("🔄 Cache miss. Fetching fresh data from NSE Scraper API...")
    stocks = fetch_nse_stocks_from_nse_scraper()
    update_cache(stocks)
    return stocks

def fetch_stock_data_for_ticker(ticker: str) -> Optional[Stock]:
    """
    Fetches detailed data for a single ticker using the NSE Scraper API.
    This implementation fetches all stocks and filters, which is inefficient for single lookups.
    A better approach would be an API endpoint like `/stocks?search={ticker}` if available.
    For now, this works with the existing structure.
    """
    # Fetch all stocks (uses cache internally)
    all_stocks = fetch_live_stocks()
    # Find the specific stock
    for stock in all_stocks:
        if stock.ticker == ticker.upper():
            return stock
    return None # Not found


# --- PREDICTION CACHING & RATE LIMITING IMPLEMENTATION ---
# --- CONFIGURATION ---
MARKET_OPEN_HOUR = 9  # Assuming market opens at 9 AM EAT
MARKET_CLOSE_HOUR = 4 # Assuming market closes at 4 PM EAT
CACHE_DURATION_MARKET_HOURS = timedelta(minutes=15)
CACHE_DURATION_OFF_MARKET_HOURS = timedelta(hours=4)
RATE_LIMIT_WINDOW_SECONDS = 3600  # 60 minutes
RATE_LIMIT_MAX_REQUESTS = 15 # Adjust based on NSE API limits if needed
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
        "service": "Treval AI NSE Live Data API v2.1.1",
        "source": "NSE Scraper API (RapidAPI)", # Updated source
        "features": ["Live Data", "Prediction Caching", "Rate Limiting"],
        "cloud_hosted": True,
        "python_version": platform.python_version(),
        "timestamp": datetime.now().isoformat(),
        "note": "Deployed on Render. Uses the verified NSE Scraper API."
    }

@app.get("/api/v1/stocks")
async def get_all_stocks():
    """Fetch all currently tracked stocks from NSE Scraper API."""
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
        raise HTTPException(status_code=404, detail=f"Stock with ticker '{ticker}' not found in NSE Scraper data.")
    return stock

@app.get("/api/v1/market-summary")
async def market_summary():
    """Provide a high-level summary of the market based on NSE Scraper data."""
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
        "data_source": "NSE Scraper API (RapidAPI)", # Updated source
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
