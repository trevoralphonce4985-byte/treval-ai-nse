# main.py - Treval AI Financial Engine v2.2 (Python 3.14.3 + Pydantic V2 Compatible)

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field # Import Field for Pydantic V2

# ----------------------------------------------------
# Logging
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrevalAI")

# ----------------------------------------------------
# FastAPI App
# ----------------------------------------------------
app = FastAPI(
    title="Treval AI Financial Engine",
    version="2.2.0",
    description="Live NSE Stock Analysis API (Python 3.14.3 + Pydantic V2)"
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
    ticker: str
    company: str
    price: float
    change: float
    change_percent: float
    volume: int
    dividend_yield: float
    pe_ratio: Optional[float] = Field(default=None) # Use Field for optional with default
    market_cap: Optional[float] = Field(default=None)
    recommendation: Optional[str] = Field(default=None)

class WealthPick(BaseModel):
    ticker: str
    company: str
    score: float
    recommendation: str
    dividend_yield: float
    pe_ratio: Optional[float] = Field(default=None)
    momentum: float
    price: float

# ----------------------------------------------------
# Mock Data (fallback if scraping fails)
MOCK_DATA = [
    Stock(ticker="SCOM", company="Safaricom PLC", price=35.55, change=0.15, change_percent=0.42, volume=1850000, dividend_yield=5.2, pe_ratio=14.8, recommendation="BUY"),
    Stock(ticker="EQTY", company="Equity Group Holdings", price=86.50, change=-0.40, change_percent=-0.46, volume=930000, dividend_yield=4.8, pe_ratio=6.4, recommendation="BUY"),
    Stock(ticker="KCB", company="KCB Group PLC", price=80.75, change=0.35, change_percent=0.44, volume=1120000, dividend_yield=5.1, pe_ratio=5.9, recommendation="BUY"),
]

# ----------------------------------------------------
# Safe parsing helpers
def safe_float(s: str, default=0.0):
    try:
        # Remove non-numeric characters except for minus and decimal point
        cleaned = re.sub(r'[^\d.-]', '', s)
        return float(cleaned)
    except Exception:
        return default

def safe_int(s: str, default=0):
    try:
        # Remove non-numeric characters
        cleaned = re.sub(r'[^\d-]', '', s)
        return int(cleaned)
    except Exception:
        return default

# Import re here since it's used in the helpers above
import re


# ----------------------------------------------------
# Scrape from mystocks.co.ke (with fallback)
def fetch_live_stocks():
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        resp = requests.get("https://mystocks.co.ke/", headers=headers, timeout=10)
        resp.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(resp.text, 'html.parser')

        # Look for the main table containing stock quotes
        # The actual structure might vary, this is a general approach
        # Inspect the HTML of mystocks.co.ke to find the correct selector
        # Example: table id might be 'main-table', 'quotes-table', etc.
        # For now, let's try to find a table that looks like it contains stock data
        # This might need adjustment based on the actual HTML structure
        table = soup.find('table', {'id': 'main-table'}) # Adjust selector as needed
        if not table:
            # If no specific table found, try to find any table with rows that have multiple cells
            table = soup.find('table')
            if table:
                rows = table.find_all('tr')
                # Heuristic: if a table has rows with at least 3 cells, it might be the quotes table
                potential_table = None
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 3: # At least ticker, price, change
                         potential_table = table
                         break
                table = potential_table

        if not table:
            logger.warning("No table found on mystocks.co.ke → using mock data")
            return MOCK_DATA

        stocks = []
        rows = table.find_all('tr')[1:] # Skip header row if present
        for row in rows[:10]: # Limit to first 10 rows to avoid huge lists
            cols = row.find_all(['td', 'th'])
            if len(cols) < 2: # Need at least ticker and price
                continue

            ticker_elem = cols[0]
            ticker = ticker_elem.get_text(strip=True).upper()
            # Validate ticker format (e.g., 2-6 uppercase letters/digits)
            if not re.match(r'^[A-Z0-9]{2,6}$', ticker):
                 logger.debug(f"Skipping invalid ticker format: {ticker}")
                 continue

            price_elem = cols[1]
            price_str = price_elem.get_text(strip=True)
            price = safe_float(price_str)
            if price <= 0:
                logger.debug(f"Skipping stock {ticker} with invalid price: {price_str}")
                continue

            # Extract change percentage
            change_pct = 0.0
            change_elem_idx = 2 # Assume change is in the 3rd column
            if change_elem_idx < len(cols):
                change_elem = cols[change_elem_idx]
                change_text = change_elem.get_text(strip=True)
                # Look for percentage sign
                pct_match = re.search(r'([+-]?\d+\.?\d*)%', change_text)
                if pct_match:
                    change_pct = float(pct_match.group(1))

            # Extract volume
            volume = 100000 # Default volume
            vol_elem_idx = 3 # Assume volume is in the 4th column
            if vol_elem_idx < len(cols):
                vol_elem = cols[vol_elem_idx]
                vol_text = vol_elem.get_text(strip=True)
                volume = safe_int(vol_text)

            # Append stock object
            stock_obj = Stock(
                ticker=ticker,
                company=f"{ticker} PLC", # Derive company name, could be improved
                price=price,
                change=0.0, # Not scraped directly here
                change_percent=change_pct,
                volume=volume,
                dividend_yield=4.5, # Default, should ideally be scraped
                pe_ratio=None, # Default, should ideally be scraped
                recommendation="HOLD" # Default recommendation
            )
            stocks.append(stock_obj)

        if stocks:
            logger.info(f"Scraped {len(stocks)} stocks from mystocks.co.ke")
            return stocks
        else:
            logger.warning("Scraping from mystocks.co.ke returned no valid data → using mock data")
            return MOCK_DATA

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during scraping: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during scraping: {e}")
    # Always return mock data if anything goes wrong
    logger.info("Returning mock data due to scraping error.")
    return MOCK_DATA


# ----------------------------------------------------
# Scoring logic (Simplified Example)
def calculate_score(stock: Stock) -> float:
    score = 0
    # Example: Weighted scoring based on available fields
    # Dividend Yield: Higher is better, max 25 points
    score += min(stock.dividend_yield * 2, 25)
    # P/E Ratio: Lower (within reason) is better, max 20 points
    if stock.pe_ratio is not None and stock.pe_ratio > 0:
        # Inverse relationship, capped
        pe_score = max(0, 20 - (stock.pe_ratio * 0.5))
        score += min(pe_score, 20)
    else:
        # If P/E is unknown, give a moderate default
        score += 5 # Default assumption if P/E is missing
    # Price Level: Moderate prices might be preferred, max 10 points
    if 10 <= stock.price <= 100:
        score += 10
    elif 5 <= stock.price <= 200:
        score += 7
    else:
        score += 3 # Lower score for very high or very low prices
    # Volume: Higher volume implies liquidity, max 15 points
    if stock.volume >= 2000000:
        score += 15
    elif stock.volume >= 1000000:
        score += 12
    elif stock.volume >= 500000:
        score += 10
    elif stock.volume >= 100000:
        score += 7
    else:
        score += 3 # Lower score for low volume

    return round(score, 1)

def generate_recommendation(score: float) -> str:
    if score >= 70:
        return "STRONG BUY"
    elif score >= 55:
        return "BUY"
    elif score >= 40:
        return "HOLD"
    elif score >= 25:
        return "SELL"
    else:
        return "STRONG SELL"

# ----------------------------------------------------
# Endpoints
@app.get("/")
async def root():
    return {
        "status": "ONLINE",
        "service": "Treval AI Financial Engine v2.2",
        "cloud_hosted": True,
        "python_version": "3.14.3",
        "pydantic_version": "V2",
        "timestamp": datetime.now().isoformat(),
        "note": "API running on Python 3.14.3 with Pydantic V2."
    }

@app.get("/api/v1/wealth-picks")
async def wealth_picks():
    stocks = fetch_live_stocks()
    picks = []
    for stock in stocks:
        score = calculate_score(stock)
        picks.append(
            WealthPick(
                ticker=stock.ticker,
                company=stock.company,
                score=score,
                recommendation=generate_recommendation(score),
                dividend_yield=stock.dividend_yield,
                pe_ratio=stock.pe_ratio,
                momentum=round((score / 75) * 100, 1), # Example momentum calc
                price=stock.price
            )
        )
    # Sort by calculated score in descending order
    picks.sort(key=lambda x: x.score, reverse=True)
    return picks

@app.get("/health")
async def health():
    return {"status": "healthy", "time": datetime.now().isoformat()}

# Optional: Add a simple test endpoint
@app.get("/api/v1/test")
async def test():
    return {"message": "Test endpoint is working!", "pydantic_check": hasattr(Stock, 'model_fields')}
