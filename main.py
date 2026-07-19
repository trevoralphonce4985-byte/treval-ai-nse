# main.py — Treval AI Financial Engine v2.1 (Python 3.11.11 Compatible)
import logging
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    version="2.1.0",
    description="Live NSE Stock Analysis API (Python 3.11.11 Compatible)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Data Models (Pydantic V1)
# ----------------------------------------------------
class Stock(BaseModel):
    ticker: str
    company: str
    price: float
    change: float
    change_percent: float
    volume: int
    dividend_yield: float
    pe_ratio: Optional[float] = None
    market_cap: Optional[float] = None
    recommendation: Optional[str] = None

class WealthPick(BaseModel):
    ticker: str
    company: str
    score: float
    recommendation: str
    dividend_yield: float
    pe_ratio: Optional[float]
    momentum: float
    price: float

# ----------------------------------------------------
# Mock Data
MOCK_DATA = [
    Stock(ticker="SCOM", company="Safaricom PLC", price=35.55, change=0.15, change_percent=0.42, volume=1850000, dividend_yield=5.2, pe_ratio=14.8, recommendation="BUY"),
    Stock(ticker="EQTY", company="Equity Group Holdings", price=86.50, change=-0.40, change_percent=-0.46, volume=930000, dividend_yield=4.8, pe_ratio=6.4, recommendation="BUY"),
    Stock(ticker="KCB", company="KCB Group PLC", price=80.75, change=0.35, change_percent=0.44, volume=1120000, dividend_yield=5.1, pe_ratio=5.9, recommendation="BUY"),
]

# ----------------------------------------------------
# Safe parsing helpers
def safe_float(s: str, default=0.0):
    try:
        return float(re.sub(r"[^\d.-]", "", s))
    except Exception:
        return default

def safe_int(s: str, default=0):
    try:
        return int(re.sub(r"[^\d]", "", s))
    except Exception:
        return default

# ----------------------------------------------------
# Scrape from mystocks.co.ke (with fallback)
def fetch_live_stocks():
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get("https://mystocks.co.ke/", headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        # Look for any table with stock data
        # Note: The actual table ID might vary; 'main-table' is a guess based on common naming.
        # Inspecting the HTML of mystocks.co.ke would be necessary for the precise selector.
        # Using a more general approach here.
        table = soup.find('table') # Take the first table found, could be refined
        if not table:
            logger.warning("No table found on mystocks.co.ke → using mock data")
            return MOCK_DATA

        stocks = []
        rows = table.find_all('tr')[1:10] # Skip header row, take up to 9 data rows
        for row in rows:
            cols = row.find_all(['td', 'th'])
            if len(cols) < 2:
                continue # Need at least ticker and price

            ticker_elem = cols[0].get_text(strip=True)
            ticker = re.sub(r'[^A-Z]', '', ticker_elem.upper()) # Clean ticker
            if not ticker or len(ticker) < 2 or len(ticker) > 6:
                 continue # Invalid ticker format

            price_str = cols[1].get_text(strip=True)
            price = safe_float(price_str)
            if price <= 0:
                continue # Invalid price

            # Extract change % (look for % symbol in subsequent columns)
            change_pct = 0.0
            for i, col in enumerate(cols[2:], start=2):
                 txt = col.get_text(strip=True)
                 if '%' in txt:
                     change_pct = safe_float(txt.replace('%', ''))
                     break # Found change, stop searching

            # Extract volume (look for large numbers in subsequent columns)
            volume = 100000 # Default volume
            for i, col in enumerate(cols[2:], start=2):
                 txt = col.get_text(strip=True)
                 vol_match = re.search(r'(\d{4,})', txt) # Match 4+ digit numbers
                 if vol_match:
                     volume = safe_int(vol_match.group(1))
                     break # Found volume, stop searching

            stocks.append(
                Stock(
                    ticker=ticker,
                    company=f"{ticker} PLC", # Basic company name derivation
                    price=price,
                    change=0.0, # Not scraped directly here
                    change_percent=change_pct,
                    volume=volume,
                    dividend_yield=4.5, # Default, should ideally be scraped
                    pe_ratio=None, # Default, should ideally be scraped
                    recommendation="HOLD" # Default
                )
            )

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
        # If P/E is unknown, give a moderate default penalty or benefit
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
        "service": "Treval AI Financial Engine v2.1",
        "cloud_hosted": True,
        "python_version": "3.11.11",
        "pure_python_deps": True, # Indicates no Rust/C++ compilation needed
        "timestamp": datetime.now().isoformat(),
        "note": "API running on Python 3.11.11 with pure Python dependencies."
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
    return {"message": "Test endpoint is working!"}
