# main.py
import os
import logging
import re
from datetime import datetime
from typing import List, Optional
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator # Import validator

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
# Data Models (Strictly Pydantic V1 Compatible)
# ----------------------------------------------------
class Stock(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., SCOM, EQTY)")
    company: str = Field(..., description="Full company name")
    price: float = Field(..., description="Current stock price in KES") # Must be provided and > 0
    change: float = Field(0.0, description="Absolute price change")
    change_percent: float = Field(0.0, description="Percentage price change")
    volume: int = Field(0, description="Trading volume") # Default 0
    # Removed constraints like ge, le, gt from Field(...) itself for v1 compatibility
    dividend_yield: Optional[float] = Field(None, description="Annual dividend yield (%)")
    pe_ratio: Optional[float] = Field(None, description="Price-to-Earnings ratio")
    market_cap: Optional[float] = Field(None, description="Market capitalization in millions/billions KES")
    recommendation: str = Field("HOLD", description="Basic recommendation (e.g., HOLD)")

    # Pydantic V1 validators
    @validator('price')
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('price must be greater than 0')
        return v

    @validator('volume')
    def volume_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError('volume must be 0 or greater')
        return v

    # Optional validators for optional fields, only check if value is present and numeric
    @validator('pe_ratio', 'market_cap', 'dividend_yield')
    def value_must_be_positive_if_present(cls, v):
        if v is not None and isinstance(v, (int, float)):
            if v <= 0:
                raise ValueError('This value must be greater than 0 if provided and numeric')
        return v # Return the value as is, allowing None or non-numeric values to pass


# ----------------------------------------------------
# Utility Functions
# ----------------------------------------------------
def safe_float(s, default=0.0):
    """Safely convert a string to float."""
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

def safe_int(s, default=0):
    """Safely convert a string to int."""
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
# Data Fetching from RapidAPI
# ----------------------------------------------------
def fetch_nse_stocks_from_rapidapi() -> List[Stock]:
    """
    Fetches live stock data from the RapidAPI NSE endpoint.
    Uses Pydantic V1 compatible logic.
    """
    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    rapidapi_host = "nairobi-stock-exchange-nse.p.rapidapi.com"

    if not rapidapi_key:
        logger.error("❌ RAPIDAPI_KEY environment variable not set. Cannot fetch live data.")
        return []

    url = f"https://{rapidapi_host}/stocks"
    headers = {
        "Content-Type": "application/json",
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": rapidapi_host
    }

    try:
        logger.info(f"📡 Fetching live data from RapidAPI: {url.split('/')[2]}")
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        raw_data = response.json()
        logger.debug(f"Raw API response type: {type(raw_data)}, length: {len(raw_data) if isinstance(raw_data, (list, str)) else 'N/A'}")

        stocks = []
        stock_list = raw_data if isinstance(raw_data, list) else raw_data.get('data', []) or raw_data.get('stocks', [])

        for item in stock_list:
            ticker = item.get('symbol') or item.get('ticker') or item.get('code')
            if not ticker:
                logger.warning(f"Skipping item due to missing ticker/symbol/code: {item}")
                continue

            price_raw = item.get('price') or item.get('current_price') or item.get('close_price')
            price = safe_float(price_raw)
            if price <= 0:
                logger.warning(f"Skipping {ticker} due to invalid/missing price: {price_raw}")
                continue

            change_raw = item.get('change', 0.0)
            change = safe_float(change_raw)

            change_pct_raw = item.get('change_percent') or item.get('chg_pct') or item.get('percent_change')
            change_percent = safe_float(change_pct_raw)

            volume_raw = item.get('volume') or item.get('vol') or item.get('traded_volume')
            volume = safe_int(volume_raw)

            dividend_yield = safe_float(item.get('dividend_yield'))
            pe_ratio = safe_float(item.get('pe_ratio') or item.get('pe'))
            market_cap_raw = item.get('market_cap') or item.get('mkt_cap')
            market_cap = safe_float(market_cap_raw)

            company = item.get('company') or item.get('name') or item.get('issuer_name') or f"{ticker} PLC"

            try:
                stock_obj = Stock(
                    ticker=ticker.upper(),
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
                stocks.append(stock_obj)
            except ValueError as ve:
                 logger.error(f"Validation error for item {item}: {ve}")
                 continue

        logger.info(f"✅ Successfully fetched and processed {len(stocks)} stocks from RapidAPI.")
        return stocks

    except requests.exceptions.RequestException as e:
        logger.error(f"📡 Network error during RapidAPI call: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            logger.error(f"Response text (first 200 chars): {e.response.text[:200]}...")
    except Exception as e:
        logger.error(f"💥 Unexpected error during RapidAPI fetch: {e}")
        logger.exception("Full traceback:")

    logger.warning("📡 RapidAPI call failed or returned no valid data matching the model. Returning empty list.")
    return []


def fetch_live_stocks() -> List[Stock]:
    """Wrapper function to fetch live data."""
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
        logger.info("No stocks returned from fetch function.")
    return stocks

@app.get("/api/v1/stock/{ticker}")
async def get_stock_details(ticker: str):
    """Get details for a specific stock by ticker symbol."""
    ticker = ticker.upper()
    stocks = fetch_live_stocks()
    if not stocks:
         raise HTTPException(status_code=503, detail="Live stock data temporarily unavailable (API down/error).")

    for stock in stocks:
        if stock.ticker == ticker:
            return stock
    raise HTTPException(status_code=404, detail=f"Stock with ticker '{ticker}' not found in current data.")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "time": datetime.now().isoformat()}

@app.get("/api/v1/market-summary")
async def market_summary():
    """Provide a high-level summary of the market."""
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
