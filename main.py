# main.py
import os
import logging
import re
from datetime import datetime
from typing import List, Optional, Dict, Any
import platform  # Import platform to get the actual Python version
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# Import for Pydantic v2
from pydantic import BaseModel, Field

# --- NEW IMPORTS FOR QUANTITATIVE ENGINE ---
from quant_engine.engine import quant_engine # Import the global engine instance
# Import the specific model classes to register them
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
    title="NSE Kenya Live Data API",
    version="1.2.0",  # Incremented version to reflect changes
    description="Live NSE stock data via RapidAPI — deployed on Render. Includes extended data from subscribed APIs and quantitative predictions."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider restricting this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Data Models (Pydantic V2 Compatible)
# ----------------------------------------------------
class Stock(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., SCOM, EQTY)")
    company: str = Field(..., description="Full company name")
    price: float = Field(..., gt=0, description="Current stock price in KES")  # Pydantic v2 allows gt, le, etc. in Field
    change: float = Field(0.0, description="Absolute price change")
    change_percent: float = Field(0.0, description="Percentage price change")
    volume: int = Field(0, ge=0, description="Trading volume")
    dividend_yield: Optional[float] = Field(None, ge=0, le=100, description="Annual dividend yield (%)")
    pe_ratio: Optional[float] = Field(None, gt=0, description="Price-to-Earnings ratio")
    market_cap: Optional[float] = Field(None, gt=0, description="Market capitalization in millions/billions KES")
    recommendation: str = Field("HOLD", description="Basic recommendation (e.g., HOLD)")

# --- NEW MODEL FOR PREDICTION RESULT ---
class PredictionPoint(BaseModel):
    timestamp: str = Field(..., description="Predicted timestamp (ISO 8601 format)")
    predicted_value: float = Field(..., description="Predicted value (e.g., price)")

class PredictionResult(BaseModel):
    model_used: str = Field(..., description="Name of the model used for prediction")
    predictions: List[PredictionPoint] = Field(..., description="List of predicted values with timestamps")
    info: Dict[str, Any] = Field(..., description="Additional information about the model run")

# --- NEW MODEL FOR EXTENDED DATA (Keeping existing) ---
class ExtendedDataPoint(BaseModel):
    """
    Generic model for extended data points (e.g., historical prices, fundamentals, news).
    Customize fields based on the specific subscribed API response structure.
    """
    ticker: str = Field(..., description="Stock ticker symbol")
    date: Optional[str] = Field(None, description="Date for historical data points (e.g., YYYY-MM-DD)")
    # Example fields - add/remove based on the subscribed API's data structure
    open_price: Optional[float] = Field(None, description="Opening price (for historical data)")
    high_price: Optional[float] = Field(None, description="Highest price (for historical data)")
    low_price: Optional[float] = Field(None, description="Lowest price (for historical data)")
    close_price: Optional[float] = Field(None, description="Closing price (for historical data)")
    volume: Optional[int] = Field(None, description="Trading volume (for historical data)")
    pe_ratio: Optional[float] = Field(None, description="P/E Ratio (for fundamentals)")
    dividend_yield: Optional[float] = Field(None, description="Dividend Yield (for fundamentals)")
    market_cap: Optional[float] = Field(None, description="Market Cap (for fundamentals)")
    eps: Optional[float] = Field(None, description="Earnings Per Share (for fundamentals)")
    book_value: Optional[float] = Field(None, description="Book Value (for fundamentals)")
    news_title: Optional[str] = Field(None, description="News headline (for news data)")
    news_link: Optional[str] = Field(None, description="Link to the news article")
    news_publisher: Optional[str] = Field(None, description="Publisher of the news article")
    news_published_date: Optional[str] = Field(None, description="Publication date of news")
    # Add more fields as needed by your specific API

class ExtendedDataResponse(BaseModel):
    """
    Model for the response from the extended data endpoint.
    """
    ticker: str = Field(..., description="Stock ticker symbol queried")
    source: str = Field(..., description="Name of the data source/API used")
    data_type: str = Field(..., description="Type of data returned (e.g., historical, fundamentals, news)")
    data: List[ExtendedDataPoint] = Field(..., description="List of data points")


# ----------------------------------------------------
# Utility Functions
# ----------------------------------------------------
def safe_float(s: str, default: float = 0.0) -> float:
    """Safely convert a string to float."""
    if s is None:
        return default
    try:
        # Remove common non-numeric characters except minus and decimal point
        cleaned = re.sub(r'[^\d.-]', '', str(s))
        if cleaned in ['', '-', '.', '-.']:
            return default
        return float(cleaned)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert '{s}' to float, returning {default}")
        return default

def safe_int(s: str, default: int = 0) -> int:
    """Safely convert a string to int."""
    if s is None:
        return default
    try:
        # Remove common non-numeric characters except minus and digits
        cleaned = re.sub(r'[^\d-]', '', str(s))
        if cleaned in ['', '-']:
            return default
        return int(cleaned)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert '{s}' to int, returning {default}")
        return default

# ----------------------------------------------------
# Data Fetching from RapidAPI (Original NSE Source)
# ----------------------------------------------------
def fetch_nse_stocks_from_rapidapi() -> List[Stock]:
    """
    Fetches live stock data from the RapidAPI NSE endpoint.
    Uses Pydantic V2 compatible logic.
    """
    # Retrieve the API key from environment variables
    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    rapidapi_host = "nairobi-stock-exchange-nse.p.rapidapi.com"

    if not rapidapi_key:
        logger.error("❌ RAPIDAPI_KEY environment variable not set. Cannot fetch live data.")
        return []

    url = f"https://{rapidapi_host}/stocks"
    headers = {
        "Content-Type": "application/json",
        "X-RapidAPI-Key": rapidapi_key,  # Use the key from environment
        "X-RapidAPI-Host": rapidapi_host  # Use the host from environment
    }

    try:
        logger.info(f"📡 Fetching live data from RapidAPI: {url.split('/')[2]}")
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        raw_data = response.json()
        logger.debug(f"Raw API response type: {type(raw_data)}, length/data: {len(raw_data) if isinstance(raw_data, (list, str)) else 'N/A'}")

        stocks = []
        # Assuming the API returns a list of stock objects under a key like 'data' or directly as a list
        # Adjust this based on the *actual* structure of your RapidAPI response
        stock_list = raw_data if isinstance(raw_data, list) else raw_data.get('data', []) or raw_data.get('stocks', [])

        for item in stock_list:
            # Map fields from the API response to your Stock model
            # Adjust 'symbol', 'current_price', etc. to match the actual keys in your API response
            ticker = item.get('symbol') or item.get('ticker') or item.get('code')
            if not ticker:
                logger.warning(f"Skipping item due to missing ticker/symbol/code: {item}")
                continue  # Skip items without a primary identifier

            price_raw = item.get('price') or item.get('current_price') or item.get('close_price')
            price = safe_float(price_raw)
            if price <= 0:
                logger.warning(f"Skipping {ticker} due to invalid/missing price: {price_raw}")
                continue  # Skip items with no valid price

            # Try to get other fields, defaulting gracefully
            change_raw = item.get('change')  # Absolute change
            change = safe_float(change_raw) if change_raw is not None else 0.0

            change_pct_raw = item.get('change_percent') or item.get('chg_pct') or item.get('percent_change')
            change_percent = safe_float(change_pct_raw) if change_pct_raw is not None else 0.0

            volume_raw = item.get('volume') or item.get('vol') or item.get('traded_volume')
            volume = safe_int(volume_raw) if volume_raw is not None else 0

            # Fields like Dividend Yield, P/E, Market Cap might not be available in the free/basic API tier
            dividend_yield = safe_float(item.get('dividend_yield')) if item.get('dividend_yield') else None
            pe_ratio = safe_float(item.get('pe_ratio') or item.get('pe')) if (item.get('pe_ratio') or item.get('pe')) else None
            market_cap_raw = item.get('market_cap') or item.get('mkt_cap')
            market_cap = safe_float(market_cap_raw) if market_cap_raw else None

            # Derive company name if not provided explicitly
            company = item.get('company') or item.get('name') or item.get('issuer_name') or f"{ticker} PLC"

            try:
                # Create Stock object - Pydantic V2 will validate based on Field constraints
                stock_obj = Stock(
                    ticker=ticker.upper(),  # Ensure uppercase for consistency
                    company=company,
                    price=price,
                    change=change,
                    change_percent=change_percent,
                    volume=volume,
                    dividend_yield=dividend_yield,
                    pe_ratio=pe_ratio,
                    market_cap=market_cap,
                    recommendation="HOLD"  # Default, can be calculated later by business logic if needed
                )
                stocks.append(stock_obj)
            except Exception as ve:  # Catch validation errors from Pydantic V2
                logger.error(f"Validation error for item {item}: {ve}")
                continue  # Skip this item if it fails validation

        logger.info(f"✅ Successfully fetched and processed {len(stocks)} stocks from RapidAPI NSE source.")
        return stocks

    except requests.exceptions.RequestException as e:
        logger.error(f"📡 Network error during RapidAPI NSE call: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            logger.error(f"Response text (first 200 chars): {e.response.text[:200]}...")
    except Exception as e:
        logger.error(f"💥 Unexpected error during RapidAPI NSE fetch: {e}")
        logger.exception("Full traceback:")  # Log the full stack trace for debugging

    logger.warning("📡 RapidAPI NSE call failed or returned no valid data matching the model. Returning empty list.")
    return []  # Return empty list on failure


def fetch_live_stocks() -> List[Stock]:
    """
    Wrapper function to fetch live data. Currently uses the RapidAPI source.
    Can be extended to include caching or fallbacks later.
    """
    return fetch_nse_stocks_from_rapidapi()


# ----------------------------------------------------
# NEW: Data Fetching from Finance API (Yahoo Finance via RapidAPI) - Keeping existing code
# ----------------------------------------------------
def fetch_extended_data_from_finance_api(ticker: str, data_type: str = "historical") -> List[ExtendedDataPoint]:
    """
    Fetches extended data from Finance API (Yahoo Finance powered) on RapidAPI.
    
    Supports:
    - historical: Historical stock price data (OHLCV)
    - fundamentals: Company fundamentals (P/E, dividend yield, market cap, EPS, book value)
    - news: Latest news articles related to the stock
    
    Args:
        ticker: Stock ticker symbol (e.g., 'SCOM.NSE' for NSE stocks)
        data_type: Type of data ('historical', 'fundamentals', 'news')
    
    Returns:
        List of ExtendedDataPoint objects
    """
    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    
    if not rapidapi_key:
        logger.error("❌ RAPIDAPI_KEY environment variable not set.")
        return []
    
    # Finance API host on RapidAPI
    rapidapi_host = "finance-api.p.rapidapi.com"
    
    try:
        if data_type.lower() == "historical":
            return _fetch_historical_data(ticker, rapidapi_key, rapidapi_host)
        elif data_type.lower() == "fundamentals":
            return _fetch_fundamentals_data(ticker, rapidapi_key, rapidapi_host)
        elif data_type.lower() == "news":
            return _fetch_news_data(ticker, rapidapi_key, rapidapi_host)
        else:
            logger.warning(f"Unknown data_type: {data_type}. Supported: historical, fundamentals, news")
            return []
    
    except Exception as e:
        logger.error(f"💥 Error fetching {data_type} data for {ticker}: {e}")
        logger.exception("Full traceback:")
        return []


def _fetch_historical_data(ticker: str, rapidapi_key: str, rapidapi_host: str) -> List[ExtendedDataPoint]:
    """Fetch historical OHLCV data from Finance API."""
    # Ensure ticker has .NSE suffix for NSE stocks
    if not ticker.endswith(".NSE"):
        ticker = f"{ticker}.NSE"
    
    url = "https://finance-api.p.rapidapi.com/stock/v2/get-historical-data"
    params = {
        "symbol": ticker,
        "period": "3mo"  # Last 3 months
    }
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": rapidapi_host
    }
    
    try:
        logger.info(f"📊 Fetching historical data for {ticker}")
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        logger.debug(f"Historical data response keys: {data.keys() if isinstance(data, dict) else 'list'}")
        
        extended_points = []
        
        # Parse the response structure - adjust based on actual Finance API response
        prices = data.get('prices', []) or data.get('data', [])
        
        for item in prices[:30]:  # Limit to last 30 records
            try:
                point = ExtendedDataPoint(
                    ticker=ticker.replace(".NSE", ""),
                    date=item.get('date') or item.get('timestamp'),
                    open_price=safe_float(item.get('open')),
                    high_price=safe_float(item.get('high')),
                    low_price=safe_float(item.get('low')),
                    close_price=safe_float(item.get('close') or item.get('adjclose')),
                    volume=safe_int(item.get('volume'))
                )
                extended_points.append(point)
            except Exception as e:
                logger.warning(f"Skipping historical data point: {e}")
                continue
        
        logger.info(f"✅ Fetched {len(extended_points)} historical data points for {ticker}")
        return extended_points
    
    except requests.exceptions.RequestException as e:
        logger.error(f"📡 Network error fetching historical data: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response: {e.response.text[:300]}")
        return []


def _fetch_fundamentals_data(ticker: str, rapidapi_key: str, rapidapi_host: str) -> List[ExtendedDataPoint]:
    """Fetch company fundamentals from Finance API."""
    # Ensure ticker has .NSE suffix for NSE stocks
    if not ticker.endswith(".NSE"):
        ticker = f"{ticker}.NSE"
    
    url = "https://finance-api.p.rapidapi.com/stock/v2/get-profile"
    params = {
        "symbol": ticker
    }
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": rapidapi_host
    }
    
    try:
        logger.info(f"📈 Fetching fundamentals for {ticker}")
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        logger.debug(f"Fundamentals response keys: {data.keys() if isinstance(data, dict) else 'list'}")
        
        # Parse fundamentals - adjust based on actual Finance API response
        profile = data.get('profile', data)  # Could be nested under 'profile'
        
        fundamentals = ExtendedDataPoint(
            ticker=ticker.replace(".NSE", ""),
            pe_ratio=safe_float(profile.get('trailingPE') or profile.get('pe')),
            dividend_yield=safe_float(profile.get('dividendYield')),
            market_cap=safe_float(profile.get('marketCap')),
            eps=safe_float(profile.get('trailingEps') or profile.get('eps')),
            book_value=safe_float(profile.get('bookValue'))
        )
        
        logger.info(f"✅ Fetched fundamentals for {ticker}")
        return [fundamentals]
    
    except requests.exceptions.RequestException as e:
        logger.error(f"📡 Network error fetching fundamentals: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response: {e.response.text[:300]}")
        return []


def _fetch_news_data(ticker: str, rapidapi_key: str, rapidapi_host: str) -> List[ExtendedDataPoint]:
    """Fetch latest news articles from Finance API."""
    # Remove .NSE suffix for news search
    ticker_clean = ticker.replace(".NSE", "")
    
    url = "https://finance-api.p.rapidapi.com/news/v2/list"
    params = {
        "symbols": ticker_clean,
        "limit": 10
    }
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": rapidapi_host
    }
    
    try:
        logger.info(f"📰 Fetching news for {ticker_clean}")
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        logger.debug(f"News response keys: {data.keys() if isinstance(data, dict) else 'list'}")
        
        news_points = []
        
        # Parse news items - adjust based on actual Finance API response
        items = data.get('items', data.get('news', []))
        
        for item in items:
            try:
                point = ExtendedDataPoint(
                    ticker=ticker_clean,
                    news_title=item.get('title') or item.get('headline'),
                    news_link=item.get('link') or item.get('url'),
                    news_publisher=item.get('publisher') or item.get('source'),
                    news_published_date=item.get('published') or item.get('date')
                )
                news_points.append(point)
            except Exception as e:
                logger.warning(f"Skipping news item: {e}")
                continue
        
        logger.info(f"✅ Fetched {len(news_points)} news articles for {ticker_clean}")
        return news_points
    
    except requests.exceptions.RequestException as e:
        logger.error(f"📡 Network error fetching news: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response: {e.response.text[:300]}")
        return []


# Backwards compatibility - keep old function name
def fetch_extended_data_from_subscribed_api(ticker: str, data_type: str = "historical") -> List[ExtendedDataPoint]:
    """Wrapper for backwards compatibility. Calls fetch_extended_data_from_finance_api."""
    return fetch_extended_data_from_finance_api(ticker, data_type)


# ----------------------------------------------------
# API Endpoints
# ----------------------------------------------------

# --- NEW: Register Models on Startup ---
@app.on_event('startup')
async def startup_event():
    logger.info("🚀 Starting up and registering quantitative models...")
    try:
        quant_engine.register_model("RandomWalk", RandomWalkModel)
        quant_engine.register_model("SMA", SimpleMovingAverageModel)
        quant_engine.register_model("EMA", ExponentialMovingAverageModel) # Added EMA
        quant_engine.register_model("LinearRegression", LinearRegressionModel)
        quant_engine.register_model("GBM", GbmModel)
        logger.info("✅ All quantitative models registered successfully.")
    except Exception as e:
        logger.error(f"💥 Failed to register quantitative models: {e}")
        logger.exception("Full traceback:")


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "NSE Kenya Live Data API v1.2",
        "source": "RapidAPI (nairobi-stock-exchange-nse) + Finance API (Yahoo Finance) + Quantitative Engine",
        "cloud_hosted": True,
        "python_version": platform.python_version(),
        "pydantic_version": "V2 (Compatible)",
        "timestamp": datetime.now().isoformat(),
        "note": "Deployed on Render. Requires RAPIDAPI_KEY environment variable.",
        "endpoints": {
            "live_stocks": "/api/v1/stocks",
            "stock_detail": "/api/v1/stock/{ticker}",
            "market_summary": "/api/v1/market-summary",
            "extended_data_historical": "/api/v1/extended-data/SCOM?type=historical",
            "extended_data_fundamentals": "/api/v1/extended-data/SCOM?type=fundamentals",
            "extended_data_news": "/api/v1/extended-data/SCOM?type=news",
            "prediction": "/api/v1/predict/SCOM?model_name=RandomWalk&horizon=5" # Example usage
        }
    }

@app.get("/api/v1/stocks")
async def get_all_stocks():
    """Fetch all currently tracked stocks from the primary NSE RapidAPI source."""
    stocks = fetch_live_stocks()
    if not stocks:
        logger.info("No stocks returned from primary fetch function.")
    return stocks

@app.get("/api/v1/stock/{ticker}")
async def get_stock_details(ticker: str):
    """Get details for a specific stock by ticker symbol from the primary NSE data source."""
    ticker = ticker.upper()
    stocks = fetch_live_stocks()
    if not stocks:
        raise HTTPException(status_code=503, detail="Live stock data temporarily unavailable (API down/error).")

    for stock in stocks:
        if stock.ticker == ticker:
            return stock
    raise HTTPException(status_code=404, detail=f"Stock with ticker '{ticker}' not found in current data.")

# --- NEW ENDPOINT FOR EXTENDED DATA ---
@app.get("/api/v1/extended-data/{ticker}", response_model=ExtendedDataResponse)
async def get_extended_data(ticker: str, data_type: str = "historical"):
    """
    Fetch extended data (historical prices, fundamentals, news) for a specific ticker
    from Finance API (Yahoo Finance powered).
    
    Query Parameters:
    - ticker: Stock ticker symbol (e.g., SCOM, EQTY) - .NSE suffix added automatically
    - type: Data type to fetch (historical, fundamentals, news). Default: historical
    
    Examples:
    - /api/v1/extended-data/SCOM?type=historical
    - /api/v1/extended-data/EQTY?type=fundamentals
    - /api/v1/extended-data/SCOM?type=news
    """
    ticker = ticker.upper()
    logger.info(f"🔍 Requesting extended data for ticker: {ticker}, type: {data_type}")
    
    extended_data_points = fetch_extended_data_from_finance_api(ticker, data_type.lower())

    if not extended_data_points:
        logger.warning(f"No extended data found for ticker: {ticker}, type: {data_type}")

    return ExtendedDataResponse(
        ticker=ticker,
        source="Finance API (Yahoo Finance)",
        data_type=data_type.lower(),
        data=extended_data_points
    )

# --- NEW ENDPOINT FOR QUANTITATIVE PREDICTIONS ---
@app.get("/api/v1/predict/{ticker}", response_model=PredictionResult)
async def get_prediction(ticker: str, model_name: str, horizon: int = 5):
    """
    Get a prediction for a specific ticker using a specified quantitative model.

    Args:
        ticker: The stock ticker symbol (e.g., 'SCOM'). Case-insensitive, will be uppercased.
        model_name: The name of the model to use (e.g., 'RandomWalk', 'SMA', 'GBM').
        horizon: The number of future time steps to predict (default 5).

    Returns:
        PredictionResult object containing the model used, the predictions, and model info.
    """
    ticker = ticker.upper()
    logger.info(f"🔮 Requesting prediction for ticker: {ticker}, model: {model_name}, horizon: {horizon}")

    # Fetch data for the ticker (reuse existing logic or create a helper)
    stocks = fetch_live_stocks() # Or fetch specific ticker data
    if not stocks:
        raise HTTPException(status_code=503, detail="Live stock data temporarily unavailable for prediction.")

    # Filter data for the specific ticker
    ticker_data = [s for s in stocks if s.ticker == ticker]
    if not ticker_data:
        raise HTTPException(status_code=404, detail=f"Stock with ticker '{ticker}' not found for prediction.")

    # Prepare data in the format expected by the model (list of dicts with timestamp and close price)
    # The model's prepare_data expects a list of dicts like [{"timestamp": ..., "close": ...}]
    raw_data_for_model = [{"timestamp": int(s.model_dump()['price']), "close": s.price} for s in ticker_data] # This is a simplification for now, using price as timestamp placeholder if no real timestamp exists in Stock model
    # A better approach would be to add a timestamp field to the Stock model or fetch historical data with timestamps specifically for prediction.
    # For now, let's assume the Stock model might have a timestamp, or we derive one from the fetch time or use an index.
    # Let's modify the raw_data_for_model creation to include a proper timestamp if available in the source API, or use a mock one.
    # Assuming fetch_nse_stocks_from_rapidapi could potentially provide a timestamp, let's add a placeholder field to Stock if needed, or use a fixed timestamp for now.
    # For this example, let's add a current timestamp placeholder. A real implementation needs historical data with timestamps.
    import time
    timestamp_placeholder = int(time.time() * 1000) - (len(ticker_data) * 86400 * 1000) # Placeholder: approx start of data
    raw_data_for_model = [{"timestamp": timestamp_placeholder + (i * 86400 * 1000), "close": s.price} for i, s in enumerate(ticker_data)] # Add 1 day per data point

    result = quant_engine.run_prediction(model_name, raw_data_for_model, horizon)
    if "error" in result:
        logger.error(f"Prediction failed: {result['error']}")
        raise HTTPException(status_code=500, detail=result["error"])

    logger.info(f"✅ Prediction completed for {ticker} using {model_name}. Horizon: {horizon}. Points: {len(result['predictions'])}")
    return PredictionResult(**result) # Unpack the result dict into the Pydantic model


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "time": datetime.now().isoformat()}

# --- Optional: Example endpoint for market summary ---
@app.get("/api/v1/market-summary")
async def market_summary():
    """Provide a high-level summary of the market based on fetched stocks."""
    stocks = fetch_live_stocks()
    if not stocks:
        return {"message": "No live stock data available for summary (API down/error)."}

    total_stocks = len(stocks)
    gainers = [s for s in stocks if s.change_percent > 0]
    losers = [s for s in stocks if s.change_percent < 0]
    unchanged = [s for s in stocks if s.change_percent == 0]

    avg_change = sum(s.change_percent for s in stocks) / total_stocks if total_stocks > 0 else 0
    total_volume = sum(s.volume for s in stocks)

    return {
        "timestamp": datetime.now().isoformat(),
        "data_source": "RapidAPI Nairobi-Stock-Exchange",
        "total_stocks_analyzed": total_stocks,
        "gainers_count": len(gainers),
        "losers_count": len(losers),
        "unchanged_count": len(unchanged),
        "average_change_percent": round(avg_change, 2),
        "total_volume": total_volume,
        "top_gainer": max(gainers, key=lambda x: x.change_percent).ticker if gainers else None,
        "top_loser": min(losers, key=lambda x: x.change_percent).ticker if losers else None
    }
