# main.py
import os
import logging
import re
from datetime import datetime
from typing import List, Optional
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ----------------------------------------------------
# Logging
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NSE-API")

# ----------------------------------------------------
# FastAPI App
# ----------------------------------------------------
app = FastAPI(
    title="NSE Kenya Live Data API",
    version="1.0.0",
    description="Live NSE stock data via RapidAPI — deployed on Render"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Models
# ----------------------------------------------------
class Stock(BaseModel):
    ticker: str = Field(..., description="Stock symbol (e.g., SCOM)")
    company: str = Field(..., description="Company name")
    price: float = Field(..., gt=0, description="Current price (KES)")
    change: float = Field(0.0, description="Absolute change")
    change_percent: float = Field(0.0, description="Change %")
    volume: int = Field(0, ge=0, description="Trading volume")
    dividend_yield: Optional[float] = Field(None, ge=0, le=100)
    pe_ratio: Optional[float] = Field(None, gt=0)
    market_cap: Optional[float] = Field(None, gt=0)
    recommendation: str = Field("HOLD", description="Recommendation")

# ----------------------------------------------------
# RapidAPI Config
# The key is read from the environment variable set in Render.
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = "nairobi-stock-exchange-nse.p.rapidapi.com"

if not RAPIDAPI_KEY:
    logger.warning("⚠️ RAPIDAPI_KEY environment variable not set. API will fail unless set in Render environment.")

# ----------------------------------------------------
# Helper Functions: Safe conversion
# ----------------------------------------------------
def safe_float(s, default=0.0):
    try:
        if s is None:
            return default
        # Remove common non-numeric characters except minus and decimal point
        cleaned = re.sub(r'[^\d.-]', '', str(s))
        if cleaned in ['', '-', '.', '-.']:
             return default
        return float(cleaned)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert '{s}' to float, returning {default}")
        return default

def safe_int(s, default=0):
    try:
        if s is None:
            return default
        # Remove common non-numeric characters except minus and digits
        cleaned = re.sub(r'[^\d-]', '', str(s))
        if cleaned in ['', '-']:
             return default
        return int(cleaned)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert '{s}' to int, returning {default}")
        return default

# ----------------------------------------------------
# Fetch Function: Get data from RapidAPI
# ----------------------------------------------------
def fetch_nse_stocks() -> List[Stock]:
    if not RAPIDAPI_KEY:
        logger.error("❌ RAPIDAPI_KEY is missing in environment variables.")
        return []

    url = f"https://{RAPIDAPI_HOST}/stocks"
    headers = {
        "Content-Type": "application/json",
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }

    try:
        logger.info("📡 Calling RapidAPI NSE endpoint...")
        res = requests.get(url, headers=headers, timeout=15) # Increased timeout slightly
        res.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
        data = res.json()

        # Handle common response formats: list of stocks OR { "stocks": [...] } OR { "data": [...] }
        stocks_data = data if isinstance(data, list) else data.get("stocks", []) or data.get("data", [])

        stocks = []
        for item in stocks_data:
            # Attempt to map fields from the API response to your Stock model
            # Adjust keys ('symbol', 'price', etc.) based on the *actual* API response structure
            ticker = item.get("symbol") or item.get("ticker") or item.get("code") or item.get("stock_code")
            if not ticker:
                logger.debug(f"Skipping item due to missing ticker: {item}")
                continue # Skip items without a ticker symbol

            price_raw = item.get("price") or item.get("current_price") or item.get("last_price") or item.get("close_price")
            price = safe_float(price_raw)
            if price <= 0:
                logger.debug(f"Skipping {ticker} due to invalid price: {price_raw}")
                continue # Skip items with invalid price

            stock = Stock(
                ticker=ticker.upper(), # Standardize ticker to uppercase
                company=item.get("company") or item.get("name") or item.get("issuer_name") or f"{ticker} PLC", # Try different common names
                price=price,
                change=safe_float(item.get("change")), # Use safe conversion
                change_percent=safe_float(item.get("change_percent") or item.get("chg_pct") or item.get("percent_change")),
                volume=safe_int(item.get("volume") or item.get("vol") or item.get("traded_volume")),
                dividend_yield=safe_float(item.get("dividend_yield")),
                pe_ratio=safe_float(item.get("pe_ratio") or item.get("pe")),
                market_cap=safe_float(item.get("market_cap") or item.get("mkt_cap")),
                recommendation="HOLD" # Default, can be calculated later
            )
            stocks.append(stock)

        logger.info(f"✅ Successfully fetched and parsed {len(stocks)} stocks from RapidAPI.")
        return stocks

    except requests.exceptions.HTTPError as he:
        logger.error(f"HTTP error occurred: {he}")
        logger.error(f"Response status code: {he.response.status_code}")
        logger.error(f"Response text: {he.response.text}")
        return []
    except requests.exceptions.ConnectionError as ce:
        logger.error(f"Connection error occurred: {ce}")
        return []
    except requests.exceptions.Timeout as te:
        logger.error(f"Timeout error occurred: {te}")
        return []
    except requests.exceptions.RequestException as re:
        logger.error(f"An error occurred during the request: {re}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during fetch_nse_stocks: {e}")
        logger.exception("Full traceback:") # Log the full stack trace for debugging
        return []

# ----------------------------------------------------
# API Endpoints
# ----------------------------------------------------
@app.get("/")
async def home():
    return {
        "status": "online",
        "service": "NSE Kenya Live API v1.0",
        "source": "RapidAPI (nairobi-stock-exchange-nse)",
        "cloud_hosted": True,
        "timestamp": datetime.utcnow().isoformat(),
        "note": "Deployed on Render. Set RAPIDAPI_KEY in environment variables."
    }

@app.get("/stocks")
async def get_stocks():
    stocks = fetch_nse_stocks()
    if not stocks:
        logger.warning("No stocks returned from fetch function. Check logs/RapidAPI.")
        # Return empty list with a 200 OK, or raise 503 depending on preference
        # For now, returning empty list is common if API has no data momentarily
        # raise HTTPException(status_code=503, detail="No live data available from upstream API. Check logs or try again later.")
        return [] # Returning empty list if no data
    return stocks

@app.get("/health")
async def health():
    return {"status": "healthy", "time": datetime.utcnow().isoformat()}

# Example endpoint to get a specific stock by ticker
@app.get("/stock/{ticker}")
async def get_stock(ticker: str):
    ticker = ticker.upper()
    stocks = fetch_nse_stocks()
    if not stocks:
        raise HTTPException(status_code=503, detail="Live data temporarily unavailable.")
    for stock in stocks:
        if stock.ticker == ticker:
            return stock
    raise HTTPException(status_code=404, detail=f"Stock with ticker '{ticker}' not found.")
