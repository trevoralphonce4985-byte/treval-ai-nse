# main.py — Treval AI Financial Engine v2.1 (Free Tier Optimized)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re
import logging

# ----------------------------------------------------
# Logging (lightweight)
# ----------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("TrevalAI")

# ----------------------------------------------------
# FastAPI App
# ----------------------------------------------------
app = FastAPI(
    title="Treval AI NSE",
    version="2.1.0",
    description="Lightweight NSE Screener for Free Tier"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Data Models
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
# Mock Data (fallback — ensures uptime even if scraping fails)
MOCK_DATA = [
    Stock(ticker="SCOM", company="Safaricom PLC", price=35.55, change=0.15, change_percent=0.42, volume=1850000, dividend_yield=5.2, pe_ratio=14.8, recommendation="BUY"),
    Stock(ticker="EQTY", company="Equity Group Holdings", price=86.50, change=-0.40, change_percent=-0.46, volume=930000, dividend_yield=4.8, pe_ratio=6.4, recommendation="BUY"),
    Stock(ticker="KCB", company="KCB Group PLC", price=80.75, change=0.35, change_percent=0.44, volume=1120000, dividend_yield=5.1, pe_ratio=5.9, recommendation="BUY"),
]

# ----------------------------------------------------
# Safe parsing (no bare except)
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
# Scrape (lightweight — minimal HTML parsing)
def fetch_live_stocks():
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get("https://mystocks.co.ke/", headers=headers, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        # Try to find any table — fallback to mock if none found
        table = soup.find('table') or soup.find('tbody')
        if not table:
            logger.warning("No table → using mock")
            return MOCK_DATA

        stocks = []
        rows = table.find_all('tr')[1:6]  # Only 5 rows to save RAM/CPU
        for row in rows:
            cols = row.find_all(['td', 'th'])
            if len(cols) < 2:
                continue

            ticker = re.sub(r'[^A-Z]', '', cols[0].get_text(strip=True).upper()[:6])
            if not ticker or len(ticker) < 2:
                continue

            price = safe_float(cols[1].get_text(strip=True))
            if price <= 0:
                continue

            change_pct = safe_float(cols[2].get_text(strip=True).replace('%', '')) if len(cols) > 2 else 0.0
            volume = safe_int(cols[3].get_text(strip=True")) if len(cols) > 3 else 100000

            stocks.append(
                Stock(
                    ticker=ticker,
                    company=f"{ticker} PLC",
                    price=price,
                    change=0.0,
                    change_percent=change_pct,
                    volume=volume,
                    dividend_yield=4.5,
                    pe_ratio=None,
                    recommendation="HOLD"
                )
            )

        if stocks:
            return stocks
        logger.warning("No valid stocks → using mock")
        return MOCK_DATA

    except Exception as e:
        logger.error(f"Scrape failed: {e}")
        return MOCK_DATA

# ----------------------------------------------------
# Scoring (lightweight)
def calculate_score(stock: Stock) -> float:
    score = 0
    score += min(stock.dividend_yield * 2, 25)
    score += max(20 - (stock.pe_ratio or 20), 0) if stock.pe_ratio else 5
    score += 10 if stock.price <= 100 else 5
    score += 5 if stock.volume >= 500000 else 2
    return round(score, 1)

def generate_recommendation(score: float) -> str:
    return "STRONG BUY" if score >= 70 else \
           "BUY" if score >= 55 else \
           "HOLD" if score >= 40 else \
           "SELL" if score >= 25 else "STRONG SELL"

# ----------------------------------------------------
# Endpoints
@app.get("/")
async def root():
    return {
        "status": "ONLINE",
        "service": "Treval AI NSE",
        "free_tier": True,
        "timestamp": datetime.now().isoformat(),
        "note": "Running on Render Free Plan — no PC required"
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
    return {"status": "healthy"}
