# main.py
import os
import logging
import re
from datetime import datetime
from typing import List, Optional
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

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
    version="1.0.0",
    description="Live NSE stock data via RapidAPI — deployed on Render"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Consider restricting this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Data Models (Pydantic V1)
# ----------------------------------------------------
class Stock(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., SCOM, EQTY)")
    company: str = Field(..., description="Full company name")
    price: float = Field(..., description="Current stock price in KES")
    change: float = Field(0.0, description="Absolute price change")
    change_percent: float = Field(0.0, description="Percentage price change")
    volume: int = Field(0, description="Trading volume")
    dividend_yield: Optional[float] = Field(None, ge=0, le=100, description="Annual dividend yield (%)")
    pe_ratio: Optional[float] = Field(None, gt=0, description="Price-to-Earnings ratio")
    market_cap: Optional[float] = Field(None, gt=0, description="Market capitalization in millions/billions KES")
    recommendation: str = Field("HOLD", description="Basic recommendation (e.g., HOLD)")

    # Pydantic V1 validators for fields that need positive values
    @validator('price', 'volume')
    def value_must_be_positive(cls, v, values, field):
        if field.name in ['price', 'volume'] and v <= 0:
            raise ValueError(f'{field.name} must be greater than 0')
        return v

# ----------------------------------------------------
# Utility Functions (Kept for potential future use or data cleaning)
# ----------------------------------------------------
def safe_float(s, default=0.0):
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

def safe_int(s, default=0):
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
# Data Fetching from RapidAPI
# ----------------------------------------------------
def fetch_nse_stocks_from_rapidapi() -> List[Stock]:
    """
    Fetches live stock data from the RapidAPI NSE endpoint.
    Uses Pydantic V1 compatible logic.
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
        "X-RapidAPI-Host": rapidapi_host # Use the host from environment
    }

    try:
        logger.info(f"📡 Fetching live data from RapidAPI: {url.split('/')[2]}")
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

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
                continue # Skip items without a primary identifier

            price_raw = item.get('price') or item.get('current_price') or item.get('close_price')
            price = safe_float(price_raw)
            if price <= 0:
                logger.warning(f"Skipping {ticker} due to invalid/missing price: {price_raw}")
                continue # Skip items with no valid price

            # Try to get other fields, defaulting gracefully
            change_raw = item.get('change') # Absolute change
            change = safe_float(change_raw) if change_raw is not None else 0.0

            change_pct_raw = item.get('change_percent') or item.get('chg_pct') or item.get('percent_change')
            change_percent = safe_float(change_pct_raw) if change_pct_raw is not None else 0.0

            volume_raw = item.get('volume') or item.get('vol') or item.get('traded_volume')
            volume = safe_int(volume_raw) if volume_raw is not None else 0

            # Fields like Dividend Yield, P/E, Market Cap might not be available in the free/basic API tier
            dividend_yield = safe_float(item.get('dividend_yield')) if item.get('dividend_yield') else None
            pe_ratio = safe_float(item.get('pe_ratio') or item.get('pe')) if item.get('pe_ratio') or item.get('pe') else None
            market_cap_raw = item.get('market_cap') or item.get('mkt_cap')
            market_cap = safe_float(market_cap_raw) if market_cap_raw else None

            # Derive company name if not provided explicitly
            company = item.get('company') or item.get('name') or item.get('issuer_name') or f"{ticker} PLC"

            try:
                # Create Stock object - Pydantic V1 will run the validators defined above
                stock_obj = Stock(
                    ticker=ticker.upper(), # Ensure uppercase for consistency
                    company=company,
                    price=price,
                    change=change,
                    change_percent=change_percent,
                    volume=volume,
                    dividend_yield=dividend_yield,
                    pe_ratio=pe_ratio,
                    market_cap=market_cap,
                    recommendation="HOLD" # Default, can be calculated later by business logic if needed
                )
                stocks.append(stock_obj)
            except ValueError as ve: # Catch validation errors from Pydantic V1
                 logger.error(f"Validation error for item {item}: {ve}")
                 continue # Skip this item if it fails validation

        logger.info(f"✅ Successfully fetched and processed {len(stocks)} stocks from RapidAPI.")
        return stocks

    except requests.exceptions.RequestException as e:
        logger.error(f"📡 Network error during RapidAPI call: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            logger.error(f"Response text (first 200 chars): {e.response.text[:200]}...")
    except Exception as e:
        logger.error(f"💥 Unexpected error during RapidAPI fetch: {e}")
        logger.exception("Full traceback:") # Log the full stack trace for debugging

    logger.warning("📡 RapidAPI call failed or returned no valid data matching the model. Returning empty list.")
    return [] # Return empty list on failure


def fetch_live_stocks() -> List[Stock]:
    """
    Wrapper function to fetch live data. Currently uses the RapidAPI source.
    Can be extended to include caching or fallbacks later.
    """
    return fetch_nse_stocks_from_rapidapi()

# ----------------------------------------------------
# API Endpoints
# ----------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "NSE Kenya Live Data API v1.0",
        "source": "RapidAPI (nairobi-stock-exchange-nse)",
        "cloud_hosted": True,
        "python_version": "3.14.3 (Render default)",
        "pydantic_version": "V1 (Compatible)",
        "timestamp": datetime.now().isoformat(),
        "note": "Deployed on Render. Requires RAPIDAPI_KEY environment variable to be set."
    }

@app.get("/api/v1/stocks")
async def get_all_stocks():
    """Fetch all currently tracked stocks from RapidAPI."""
    stocks = fetch_live_stocks()
    if not stocks:
        logger.warning("No stocks returned from fetch function. Check logs/API status.")
        # Returning an empty list is often acceptable if no data is available right now.
        # Alternatively, raise an HTTPException like below if preferred.
        # raise HTTPException(status_code=503, detail="Live stock data temporarily unavailable (API down/error).")
    return stocks

@app.get("/api/v1/stock/{ticker}")
async def get_stock_details(ticker: str):
    """Get details for a specific stock by ticker symbol from RapidAPI data."""
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
