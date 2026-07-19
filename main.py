# main.py — Treval AI Financial Engine v2.1 (Final: Pure Python)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re
import logging

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
    description="Live NSE Stock Analysis API (Final: Pure Python)"
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
        return float(re.sub(r'[^\d.-]', '', s))
    except:
        return default

def safe_int(s: str, default=0):
    try:
        return int(re.sub(r'[^\d]', '', s))
    except:
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
        table = soup.find('table', {'id': 'main-table'}) or soup.find('table')
        if not table:
            logger.warning("No table found → using mock data")
            return MOCK_DATA

        stocks = []
        rows = table.find_all('tr')[1:10]
        for row in rows:
            cols = row.find_all(['td', 'th'])
            if len(cols) < 2:
                continue

            ticker = cols[0].get_text(strip=True).upper()[:6]
            if not ticker or len(ticker) < 2:
                continue

            price_str = cols[1].get_text(strip=True)
            price = safe_float(price_str)
            if price <= 0:
                continue

            # Extract change %
            change_pct = 0.0
            for i, col in enumerate(cols[2:], start=2):
                txt = col.get_text(strip=True)
                if '%' in txt:
                    change_pct = safe_float(txt.replace('%', ''))
                    break

            stocks.append(
                Stock(
                    ticker=ticker,
                    company=f"{ticker} PLC",
                    price=price,
                    change=0.0,
                    change_percent=change_pct,
                    volume=safe_int(cols[3].get_text(strip=True)) if len(cols) > 3 else 100000,
                    dividend_yield=4.5,
                    pe_ratio=None,
                    recommendation="HOLD"
                )
            )

        if stocks:
            logger.info(f"Scraped {len(stocks)} stocks")
            return stocks
        else:
            logger.warning("Scraping returned empty → using mock data")
            return MOCK_DATA

    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        return MOCK_DATA

# ----------------------------------------------------
# Scoring logic
def calculate_score(stock: Stock) -> float:
    score = 0
    score += min(stock.dividend_yield * 2, 25)  # Max 25
    score += max(20 - (stock.pe_ratio or 20), 0) if stock.pe_ratio else 5
    score += 10 if stock.price <= 100 else 5
    score += 5 if stock.volume >= 500000 else 2
    return round(score, 1)

def generate_recommendation(score: float) -> str:
    if score >= 70: return "STRONG BUY"
    if score >= 55: return "BUY"
    if score >= 40: return "HOLD"
    if score >= 25: return "SELL"
    return "STRONG SELL"

# ----------------------------------------------------
# Endpoints
@app.get("/")
async def root():
    return {
        "status": "ONLINE",
        "service": "Treval AI Financial Engine v2.1",
        "cloud_hosted": True,
        "pure_python": True,
        "timestamp": datetime.utcnow().isoformat(),
        "note": "This API runs 24/7 on cloud — no PC required! (Pure Python build)"
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
                momentum=round((score / 75) * 100, 1),
                price=stock.price
            )
        )
    picks.sort(key=lambda x: x.score, reverse=True)
    return picks

@app.get("/health")
async def health():
    return {"status": "healthy", "time": datetime.utcnow().isoformat()}
