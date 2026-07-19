# main.py - Enhanced Treval AI Financial Engine v2.3 (Python 3.14.3 + Pydantic V2)
import logging
import re
from datetime import datetime
from typing import List, Optional, Dict, Any

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

# ----------------------------------------------------
# Logging Configuration
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrevalAI")

# ----------------------------------------------------
# FastAPI App Configuration
# ----------------------------------------------------
app = FastAPI(
    title="Enhanced Treval AI Financial Engine",
    version="2.3.0",
    description="Advanced NSE Stock Analysis API (Python 3.14.3 + Pydantic V2)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Consider restricting this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Data Models (Pydantic V2)
# ----------------------------------------------------
class Stock(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., SCOM, EQTY)")
    company: str = Field(..., description="Full company name")
    price: float = Field(..., gt=0, description="Current stock price in KES")
    change: float = Field(0.0, description="Absolute price change")
    change_percent: float = Field(0.0, description="Percentage price change")
    volume: int = Field(0, ge=0, description="Trading volume")
    dividend_yield: Optional[float] = Field(None, ge=0, le=100, description="Annual dividend yield (%)")
    pe_ratio: Optional[float] = Field(None, gt=0, description="Price-to-Earnings ratio")
    market_cap: Optional[float] = Field(None, gt=0, description="Market capitalization in millions/billions KES")
    recommendation: Optional[str] = Field(None, description="Basic recommendation (e.g., HOLD)")

class WealthPick(BaseModel):
    ticker: str
    company: str
    score: float = Field(..., ge=0, description="Calculated wealth score")
    recommendation: str = Field(..., description="AI-generated recommendation (e.g., STRONG BUY)")
    dividend_yield: Optional[float]
    pe_ratio: Optional[float]
    momentum_score: float = Field(..., ge=0, le=100, description="Momentum indicator score")
    value_score: float = Field(..., ge=0, le=100, description="Value indicator score")
    price: float
    # Add more fields as needed for deeper analysis

# ----------------------------------------------------
# Mock Data (Used if scraping fails or returns insufficient data)
# ----------------------------------------------------
MOCK_DATA = [
    Stock(
        ticker="SCOM",
        company="Safaricom PLC",
        price=35.55,
        change=0.15,
        change_percent=0.42,
        volume=1850000,
        dividend_yield=5.2,
        pe_ratio=14.8,
        market_cap=1400000.0, # in millions
        recommendation="BUY"
    ),
    Stock(
        ticker="EQTY",
        company="Equity Group Holdings",
        price=86.50,
        change=-0.40,
        change_percent=-0.46,
        volume=930000,
        dividend_yield=4.8,
        pe_ratio=6.4,
        market_cap=327000.0,
        recommendation="BUY"
    ),
    Stock(
        ticker="KCB",
        company="KCB Group PLC",
        price=80.75,
        change=0.35,
        change_percent=0.44,
        volume=1120000,
        dividend_yield=5.1,
        pe_ratio=5.9,
        market_cap=258000.0,
        recommendation="BUY"
    ),
    # Add more mock stocks as needed
]

# ----------------------------------------------------
# Utility Functions
# ----------------------------------------------------
def safe_float(s: str, default: float = 0.0) -> float:
    """Safely convert a string to float, handling common formatting issues."""
    if not s:
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
    if not s:
        return default
    try:
        cleaned = re.sub(r'[^\d-]', '', str(s))
        if cleaned in ['', '-']:
             return default
        return int(cleaned)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert '{s}' to int, returning {default}")
        return default

def calculate_dividend_score(dividend_yield: Optional[float]) -> float:
    """Calculate a score based on dividend yield."""
    if dividend_yield is None:
        return 0.0
    # Example scoring: 0 for 0%, 50 for 5%, 100 for 10%+, capped
    return min(dividend_yield * 10, 100.0)

def calculate_pe_score(pe_ratio: Optional[float]) -> float:
    """Calculate a score based on P/E ratio (lower P/E is often better for value)."""
    if pe_ratio is None or pe_ratio <= 0:
        return 0.0
    # Example scoring: higher score for lower P/E, capped around P/E 20
    if pe_ratio <= 5:
        return 100.0
    elif pe_ratio <= 10:
        return 85.0
    elif pe_ratio <= 15:
        return 70.0
    elif pe_ratio <= 20:
        return 50.0
    elif pe_ratio <= 25:
        return 30.0
    else:
        return 10.0 # Low score for high P/E

def calculate_momentum_score(change_percent: float) -> float:
    """Calculate a score based on recent price movement."""
    # Example: Score increases with positive momentum, decreases with negative
    # Clamp between 0 and 100
    score = 50 + change_percent * 5 # Adjust multiplier as needed
    return max(0.0, min(100.0, score))

def calculate_value_score(pe_ratio: Optional[float], dividend_yield: Optional[float]) -> float:
    """Combine P/E and dividend yield for a value score."""
    pe_s = calculate_pe_score(pe_ratio)
    div_s = calculate_dividend_score(dividend_yield)
    # Average the two scores, or use a weighted average
    return (pe_s + div_s) / 2

def generate_recommendation(score: float) -> str:
    """Generate a basic recommendation string based on the score."""
    if score >= 85:
        return "STRONG BUY"
    elif score >= 70:
        return "BUY"
    elif score >= 55:
        return "HOLD"
    elif score >= 40:
        return "SELL"
    else:
        return "STRONG SELL"

# ----------------------------------------------------
# Data Fetching and Processing
# ----------------------------------------------------
def fetch_live_stocks() -> List[Stock]:
    """
    Attempts to scrape live data from mystocks.co.ke.
    Falls back to MOCK_DATA if scraping fails or returns insufficient data.
    """
    try:
        logger.info("Attempting to fetch live data from mystocks.co.ke...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
        }
        response = requests.get("https://mystocks.co.ke/", headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        stocks = []
        # The structure of mystocks.co.ke might require inspecting the HTML
        # Let's try to find a table that looks like the main quotes table.
        # This is a heuristic and might need adjustment if the site changes.
        # Look for tables with common identifiers or classes related to quotes
        potential_tables = soup.find_all('table') # Start with all tables

        # Heuristic: Look for tables that have rows with many columns (e.g., ticker, price, change, volume)
        # This is fragile, but often tables with more than 3 columns are quote tables.
        for table in potential_tables:
            rows = table.find_all('tr')
            if len(rows) > 2: # At least header + 2 data rows
                for row in rows[1:]: # Skip header row
                    cols = row.find_all(['td', 'th'])
                    if len(cols) >= 3: # At least ticker, price, and one other metric
                        ticker_elem = cols[0]
                        price_elem = cols[1]

                        ticker = ticker_elem.get_text(strip=True).upper()
                        # Basic validation for ticker format (e.g., 2-6 uppercase alphanumeric)
                        if not re.match(r'^[A-Z0-9]{2,6}$', ticker):
                             continue

                        price_str = price_elem.get_text(strip=True)
                        price = safe_float(price_str)
                        if price <= 0:
                            continue

                        # Extract change percentage (often in col 2 or 3)
                        change_pct = 0.0
                        for i in range(2, min(len(cols), 5)): # Check next few columns
                            text = cols[i].get_text(strip=True)
                            if '%' in text:
                                 change_pct = safe_float(text.replace('%', '').replace('+', '').replace('-', ''))
                                 if '-' in cols[i].get_text(): # Determine sign based on text
                                     change_pct = -abs(change_pct)
                                 break

                        # Extract volume (often in col 3 or 4)
                        volume = 0
                        for i in range(3, min(len(cols), 6)): # Check next few columns
                            text = cols[i].get_text(strip=True)
                            # Look for large numbers, potentially with 'K' or 'M' suffixes
                            vol_match = re.search(r'(\d+(?:\.\d+)?)\s*([KMB]?)', text, re.IGNORECASE)
                            if vol_match:
                                num_part = float(vol_match.group(1))
                                suffix = vol_match.group(2).upper()
                                multiplier = 1
                                if suffix == 'K':
                                    multiplier = 1000
                                elif suffix == 'M':
                                    multiplier = 1000000
                                elif suffix == 'B':
                                    multiplier = 1000000000
                                volume = int(num_part * multiplier)
                                break
                        if volume == 0: # Default if not found
                            volume = 100000

                        # Create Stock object with defaults for non-scraped fields
                        stock_obj = Stock(
                            ticker=ticker,
                            company=f"{ticker} PLC", # Improve derivation if possible
                            price=price,
                            change=0.0, # Change amount often not directly available, derive from price change %
                            change_percent=change_pct,
                            volume=volume,
                            dividend_yield=None, # Scraping dividends is complex, set to None
                            pe_ratio=None,      # Scraping P/E is complex, set to None
                            market_cap=None,    # Scraping market cap is complex, set to None
                            recommendation="HOLD" # Default, will be overridden by AI
                        )
                        stocks.append(stock_obj)
                        if len(stocks) >= 15: # Limit to avoid huge lists if table is long
                            break
                if stocks:
                    break # Found a table with data, stop looking

        logger.info(f"Scraped {len(stocks)} stocks from mystocks.co.ke")
        if not stocks:
            logger.warning("Scraping returned no valid stocks, falling back to mock data.")
            return MOCK_DATA
        else:
            # Append mock data if scraping didn't get enough unique stocks
            scraped_tickers = {s.ticker for s in stocks}
            for mock_stock in MOCK_DATA:
                if mock_stock.ticker not in scraped_tickers and len(stocks) < 15:
                    stocks.append(mock_stock)
            return stocks

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during scraping: {e}")
    except requests.exceptions.Timeout:
        logger.error("Request timed out while fetching data from mystocks.co.ke")
    except Exception as e:
        logger.error(f"Unexpected error during scraping: {e}")

    logger.info("Scraping failed, returning mock data.")
    return MOCK_DATA


# ----------------------------------------------------
# API Endpoints
# ----------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "ONLINE",
        "service": "Enhanced Treval AI Financial Engine v2.3",
        "cloud_hosted": True,
        "python_version": "3.14.3",
        "pydantic_version": "V2",
        "timestamp": datetime.now().isoformat(),
        "note": "Enhanced API running on Python 3.14.3 with Pydantic V2. Provides more detailed analysis."
    }

@app.get("/api/v1/stocks")
async def get_all_stocks():
    """Fetch all currently tracked stocks (scraped + mock)."""
    stocks = fetch_live_stocks()
    return stocks

@app.get("/api/v1/wealth-picks")
async def get_wealth_picks():
    """
    Fetch stocks and apply enhanced AI analysis to generate wealth picks.
    """
    stocks = fetch_live_stocks()
    picks = []
    for stock in stocks:
        # Calculate scores using the utility functions
        momentum_s = calculate_momentum_score(stock.change_percent)
        value_s = calculate_value_score(stock.pe_ratio, stock.dividend_yield)
        # Combine scores (weights can be adjusted)
        combined_score = (momentum_s * 0.4) + (value_s * 0.6) # Example weighting

        pick = WealthPick(
            ticker=stock.ticker,
            company=stock.company,
            score=combined_score,
            recommendation=generate_recommendation(combined_score),
            dividend_yield=stock.dividend_yield,
            pe_ratio=stock.pe_ratio,
            momentum_score=momentum_s,
            value_score=value_s,
            price=stock.price
        )
        picks.append(pick)

    # Sort by the calculated score in descending order
    picks.sort(key=lambda x: x.score, reverse=True)
    logger.info(f"Generated {len(picks)} wealth picks.")
    return picks

@app.get("/api/v1/stock/{ticker}")
async def get_stock_details(ticker: str):
    """Get details for a specific stock by ticker symbol."""
    ticker = ticker.upper()
    stocks = fetch_live_stocks()
    for stock in stocks:
        if stock.ticker == ticker:
            # Calculate scores for the specific stock
            momentum_s = calculate_momentum_score(stock.change_percent)
            value_s = calculate_value_score(stock.pe_ratio, stock.dividend_yield)
            combined_score = (momentum_s * 0.4) + (value_s * 0.6)
            
            pick = WealthPick(
                ticker=stock.ticker,
                company=stock.company,
                score=combined_score,
                recommendation=generate_recommendation(combined_score),
                dividend_yield=stock.dividend_yield,
                pe_ratio=stock.pe_ratio,
                momentum_score=momentum_s,
                value_score=value_s,
                price=stock.price
            )
            return pick
    raise HTTPException(status_code=404, detail=f"Stock with ticker '{ticker}' not found.")

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
        return {"message": "No stock data available for summary."}

    total_stocks = len(stocks)
    gainers = [s for s in stocks if s.change_percent > 0]
    losers = [s for s in stocks if s.change_percent < 0]
    unchanged = [s for s in stocks if s.change_percent == 0]

    avg_change = sum(s.change_percent for s in stocks) / total_stocks if total_stocks > 0 else 0
    total_volume = sum(s.volume for s in stocks)

    return {
        "timestamp": datetime.now().isoformat(),
        "total_stocks_analyzed": total_stocks,
        "gainers_count": len(gainers),
        "losers_count": len(losers),
        "unchanged_count": len(unchanged),
        "average_change_percent": round(avg_change, 2),
        "total_volume": total_volume,
        "top_gainer": max(gainers, key=lambda x: x.change_percent).ticker if gainers else None,
        "top_loser": min(losers, key=lambda x: x.change_percent).ticker if losers else None
    }
