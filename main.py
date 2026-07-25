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
    version="1.1.0",  # Incremented version to reflect changes
    description="Live NSE stock data via RapidAPI — deployed on Render. Includes extended data from subscribed APIs."
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

# --- NEW MODEL FOR EXTENDED DATA ---
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
    news_title: Optional[str] = Field(None, description="News headline (for news data)")
    news_link: Optional[str] = Field(None, description="Link to the news article")
    news_publisher: Optional[str] = Field(None, description="Publisher of the news article")
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
# NEW: Data Fetching from Subscribed API (Template Placeholder)
# ----------------------------------------------------
def fetch_extended_data_from_subscribed_api(ticker: str, data_type: str = "historical") -> List[ExtendedDataPoint]:
    """
    Fetches extended data (e.g., historical prices, fundamentals, news) for a given ticker
    from a subscribed API on RapidAPI (e.g., Yahoo Finance, Alpha Vantage).

    This is a template function. You need to replace the placeholder logic with actual API calls
    based on the specific API you subscribed to and its endpoints.

    Args:
        ticker: The stock ticker symbol (e.g., 'SCOM').
        data_type: The type of data to fetch (e.g., 'historical', 'fundamentals', 'news').

    Returns:
        A list of ExtendedDataPoint objects containing the fetched data.
        Returns an empty list if the call fails or no data is found.
    """
    # Retrieve the API key from environment variables (same key used for NSE API, assuming it's linked to your account)
    rapidapi_key = os.getenv("RAPIDAPI_KEY")

    if not rapidapi_key:
        logger.error("❌ RAPIDAPI_KEY environment variable not set. Cannot fetch extended data.")
        return []

    # --- PLACEHOLDER LOGIC: Replace this section with actual API call logic ---
    # Example: If you subscribed to Yahoo Finance API
    # rapidapi_host = "yahoo-finance160.p.rapidapi.com" # Example host
    # url = f"https://{rapidapi_host}/history" # Example endpoint
    # payload = {"stock": f"{ticker}.KE", "period": "1mo"} # Example payload
    # headers = {
    #     "content-type": "application/json",
    #     "X-RapidAPI-Key": rapidapi_key,
    #     "X-RapidAPI-Host": rapidapi_host
    # }
    # try:
    #     response = requests.post(url, json=payload, headers=headers, timeout=15) # Or GET depending on API
    #     response.raise_for_status()
    #     raw_response = response.json()
    #     # Process raw_response according to the specific API's structure
    #     # e.g., extract 'historical' data from raw_response['chart']['result'][0]['indicators']['quote'][0]
    #     processed_data = []
    #     # ... parsing logic ...
    #     for item in raw_response.get('data', []): # Adjust key based on API
    #          # Map API fields to ExtendedDataPoint fields
    #          point = ExtendedDataPoint(
    #              ticker=ticker,
    #              date=item.get('date'),
    #              close_price=item.get('close'),
    #              volume=item.get('volume'),
    #              # ... map other fields ...
    #          )
    #          processed_data.append(point)
    #     return processed_data
    # except requests.exceptions.RequestException as e:
    #     logger.error(f"📡 Network error during subscribed API call for {ticker} ({data_type}): {e}")
    #     if hasattr(e, 'response') and e.response is not None:
    #         logger.error(f"Response status code: {e.response.status_code}")
    #         logger.error(f"Response text: {e.response.text[:200]}...")
    # except Exception as e:
    #     logger.error(f"💥 Unexpected error during subscribed API fetch for {ticker} ({data_type}): {e}")
    #     logger.exception("Full traceback:")
    # return []

    # --- END PLACEHOLDER LOGIC ---

    # --- CURRENT PLACEHOLDER RETURN (Remove this when implementing actual logic) ---
    logger.warning(f"🔍 Fetching extended data for {ticker} ({data_type}) - Placeholder logic. Replace with actual API call.")
    # Simulate some dummy data for demonstration - REMOVE THIS
    if data_type == "historical":
        return [
            ExtendedDataPoint(ticker=ticker, date="2026-07-25", close_price=35.60, volume=707630),
            ExtendedDataPoint(ticker=ticker, date="2026-07-24", close_price=35.55, volume=650000),
            ExtendedDataPoint(ticker=ticker, date="2026-07-23", close_price=35.70, volume=800000),
        ]
    elif data_type == "fundamentals":
        return [
            ExtendedDataPoint(ticker=ticker, pe_ratio=14.5, dividend_yield=5.2),
            ExtendedDataPoint(ticker=ticker, pe_ratio=14.2, dividend_yield=5.1),
        ]
    elif data_type == "news":
        return [
            ExtendedDataPoint(ticker=ticker, news_title="Safaricom Reports Strong Q2 Results", news_publisher="Business Daily", news_link="https://example.com/news/safaricom-q2"),
            ExtendedDataPoint(ticker=ticker, news_title="Telecom Sector Outlook Positive", news_publisher="Capital FM", news_link="https://example.com/news/telecom-outlook"),
        ]
    else:
        logger.info(f"No placeholder data for data_type: {data_type}")
        return []
    # --- END PLACEHOLDER RETURN ---


# ----------------------------------------------------
# API Endpoints
# ----------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "NSE Kenya Live Data API v1.1",
        "source": "RapidAPI (nairobi-stock-exchange-nse) & Subscribed APIs",
        "cloud_hosted": True,
        "python_version": platform.python_version(),  # Report the actual Python version being used
        "pydantic_version": "V2 (Compatible)",
        "timestamp": datetime.now().isoformat(),
        "note": "Deployed on Render. Requires RAPIDAPI_KEY environment variable. Includes extended data endpoint.",
        "endpoints": {
            "live_stocks": "/api/v1/stocks",
            "stock_detail": "/api/v1/stock/{ticker}",
            "market_summary": "/api/v1/market-summary",
            "extended_data": "/api/v1/extended-data/{ticker}?type=historical"  # Example usage
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
        # If the overall fetch failed, it's a system issue
        raise HTTPException(status_code=503, detail="Live stock data temporarily unavailable (API down/error).")

    for stock in stocks:
        if stock.ticker == ticker:
            return stock
    # If the fetch worked but the specific ticker wasn't found
    raise HTTPException(status_code=404, detail=f"Stock with ticker '{ticker}' not found in current data.")

# --- NEW ENDPOINT FOR EXTENDED DATA ---
@app.get("/api/v1/extended-data/{ticker}", response_model=ExtendedDataResponse)
async def get_extended_data(ticker: str, data_type: str = "historical"):
    """
    Fetch extended data (e.g., historical prices, fundamentals, news) for a specific ticker
    from a subscribed RapidAPI source.

    Args:
        ticker: The stock ticker symbol (e.g., 'SCOM').
        data_type: The type of data to fetch (e.g., 'historical', 'fundamentals', 'news'). Defaults to 'historical'.

    Returns:
        ExtendedDataResponse object containing the requested data.
    """
    ticker = ticker.upper()
    logger.info(f"🔍 Requesting extended data for ticker: {ticker}, type: {data_type}")
    extended_data_points = fetch_extended_data_from_subscribed_api(ticker, data_type.lower())

    if not extended_data_points:
        logger.warning(f"No extended data found for ticker: {ticker}, type: {data_type}")
        # Consider returning an empty list within the response model or raising a 404
        # For now, returning an empty list as per the model
        # raise HTTPException(status_code=404, detail=f"Extended data ({data_type}) not found for ticker '{ticker}'.")

    return ExtendedDataResponse(
        ticker=ticker,
        source="Subscribed API Placeholder",  # Replace with actual source name when implemented
        data_type=data_type.lower(),
        data=extended_data_points
    )


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
