# main.py (v2.1.2 - FIX: Volume Decimal Shift, Change Calculation & Multi-Key Parsing)
# PLUS: WhatsApp Notification Integration (v2.1.3)
import os
import logging
import re
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import platform
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import threading
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
# --- NEW: Import for WhatsApp Scheduling ---
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.cron import CronTrigger
import atexit # For scheduler cleanup

# --- IMPORT QUANTITATIVE ENGINE ---
try:
    from quant_engine.engine import quant_engine
    from quant_engine.models.random_walk import RandomWalkModel
    from quant_engine.models.moving_average import SimpleMovingAverageModel, ExponentialMovingAverageModel
    from quant_engine.models.linear_regression import LinearRegressionModel
    from quant_engine.models.gbm import GbmModel
    QUANT_ENGINE_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger("NSE-API")
    logger.warning(f"Quantitative Engine not available: {e}")
    QUANT_ENGINE_AVAILABLE = False
    quant_engine = None

# ----------------------------------------------------
# Logging Configuration
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NSE-API")

# ----------------------------------------------------
# FastAPI App Configuration
# ----------------------------------------------------
app = FastAPI(
    title="Treval AI NSE Live Data API",
    version="2.1.3", # Updated version
    description="Live NSE stock data powered by NSE Scraper API (RapidAPI) — deployed on Render. Features corrected volume scaling & absolute price change calculations. Includes WhatsApp notifications for admin."
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
    ticker: str = Field(..., description="Stock ticker symbol (e.g., SCOM)")
    company: str = Field(..., description="Full company name")
    price: float = Field(..., gt=0, description="Current stock price")
    change: float = Field(0.0, description="Absolute price change in KES")
    change_percent: float = Field(0.0, description="Percentage price change")
    volume: int = Field(0, ge=0, description="Trading volume")
    recommendation: str = Field("HOLD", description="Market sentiment recommendation")

class PredictionPoint(BaseModel):
    timestamp: str = Field(..., description="ISO 8601 Timestamp")
    predicted_value: float = Field(..., description="Predicted value")

class CachedPredictionResult(BaseModel):
    model_used: str = Field(..., description="Name of the model used for prediction")
    predictions: List[PredictionPoint] = Field(..., description="List of predicted values with timestamps")
    info: Dict[str, Any] = Field(..., description="Additional information about the model run")
    cached: bool = Field(..., description="True if the response was served from the cache")
    generated_at: str = Field(..., description="ISO 8601 timestamp when prediction was generated")
    expires_at: str = Field(..., description="ISO 8601 timestamp when cached prediction expires")
    cache_age_seconds: int = Field(..., description="Age of cached prediction in seconds")

class CacheStatusResponse(BaseModel):
    ticker: str = Field(..., description="The stock ticker symbol checked")
    model_name: str = Field(..., description="The model name checked")
    horizon: int = Field(..., description="The prediction horizon checked")
    cached: bool = Field(..., description="Whether a valid cached result exists")
    expires_at: Optional[str] = Field(None, description="ISO 8601 timestamp when cached prediction expires")
    generated_at: Optional[str] = Field(None, description="ISO 8601 timestamp when cached prediction was generated")

# ----------------------------------------------------
# Utility Functions (FIXED: Volume & Float Parsing)
# ----------------------------------------------------
def safe_float(s: Any, default: float = 0.0) -> float:
    if s is None:
        return default
    try:
        s_str = str(s).replace(',', '').strip()
        cleaned = re.sub(r'[^\d.-]', '', s_str)
        if cleaned in ['', '-', '.', '-.']:
            return default
        return float(cleaned)
    except (ValueError, TypeError):
        return default

def safe_int(s: Any, default: int = 0) -> int:
    """FIXED: Converts via float first so that '769.00' parses to 769 instead of 76900."""
    if s is None:
        return default
    try:
        val = safe_float(s, default=float(default))
        return int(val)
    except (ValueError, TypeError):
        return default

def parse_change_string(change_str: Any) -> tuple[float, float]:
    """
    Parses change strings like "+2.50 (+5.82%)" or "+5.82%".
    Returns (change_amount, change_percentage).
    """
    if not change_str:
        return 0.0, 0.0

    s = str(change_str).strip()
    match = re.search(r'([+-]?\d+\.?\d*)\s*\(\s*([+-]?\d+\.?\d*)%\s*\)', s)
    if match:
        return float(match.group(1)), float(match.group(2))

    pct_match = re.search(r'([+-]?\d+\.?\d*)%', s)
    if pct_match:
        return 0.0, float(pct_match.group(1))

    # Single numeric string fallback
    val = safe_float(s)
    return val, 0.0

def derive_recommendation(change_pct: float) -> str:
    """Generates market sentiment signals based on percentage movement."""
    if change_pct > 1.5:
        return "BUY"
    elif change_pct < -1.5:
        return "SELL"
    else:
        return "HOLD"

# ----------------------------------------------------
# Caching Mechanism
# ----------------------------------------------------
_cached_stocks = []
_cache_timestamp = datetime.min
_CACHE_DURATION = timedelta(minutes=5)
_cache_lock = threading.Lock()

def get_cached_stocks():
    global _cached_stocks, _cache_timestamp
    now = datetime.now()
    with _cache_lock:
        if now - _cache_timestamp < _CACHE_DURATION:
            return _cached_stocks.copy()
        else:
            return []

def update_cache(new_stocks: List[Stock]):
    global _cached_stocks, _cache_timestamp
    with _cache_lock:
        _cached_stocks = new_stocks
        _cache_timestamp = datetime.now()

# ----------------------------------------------------
# Data Fetching from NSE Scraper API (RapidAPI)
# ----------------------------------------------------
def fetch_nse_stocks_from_nse_scraper() -> List[Stock]:
    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    rapidapi_host = "nairobi-stock-exchange-nse.p.rapidapi.com"

    if not rapidapi_key:
        logger.error("❌ RAPIDAPI_KEY environment variable not set.")
        return []

    url = f"https://{rapidapi_host}/stocks"
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": rapidapi_host,
        "Content-Type": "application/json"
    }

    try:
        logger.info(f"📡 Fetching data from NSE Scraper API: {rapidapi_host}")
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 429:
            logger.warning("429 Rate limit encountered. Using cached fallback.")
            return get_cached_stocks()
            
        response.raise_for_status()
        data = response.json()

        if not data.get('success'):
            logger.error(f"NSE Scraper API returned success=false: {data}")
            return []

        stock_list_raw = data.get('data', [])
        if not isinstance(stock_list_raw, list):
            return []

        all_stocks = []
        for item in stock_list_raw:
            ticker = item.get('ticker')
            if not ticker:
                continue

            name = item.get('name') or item.get('company') or f"{ticker} PLC"
            
            # Multi-key price check
            price_raw = item.get('price') or item.get('last_price') or item.get('close')
            price = safe_float(price_raw)
            if price <= 0:
                continue

            # Parse change amount & percent
            change_str = item.get('change')
            change_amount, change_percent = parse_change_string(change_str)

            # Direct fallback for change_percent if provided as separate field
            if change_percent == 0.0 and 'change_percent' in item:
                change_percent = safe_float(item.get('change_percent'))

            # FIX: Calculate absolute KES change if missing/zero
            if change_amount == 0.0 and change_percent != 0.0 and price > 0:
                prev_price = price / (1.0 + (change_percent / 100.0))
                change_amount = round(price - prev_price, 2)

            # FIX: Parse volume safely via float conversion
            volume_raw = item.get('volume', 0)
            volume = safe_int(volume_raw)

            stock_obj = Stock(
                ticker=ticker.upper(),
                company=name,
                price=price,
                change=round(change_amount, 2),
                change_percent=round(change_percent, 2),
                volume=volume,
                recommendation=derive_recommendation(change_percent)
            )
            all_stocks.append(stock_obj)

        logger.info(f"✅ Successfully fetched & normalized {len(all_stocks)} stocks.")
        return all_stocks

    except Exception as e:
        logger.error(f"💥 Error fetching/parsing NSE data: {e}")
        return get_cached_stocks()

def fetch_live_stocks() -> List[Stock]:
    cached = get_cached_stocks()
    if cached:
        return cached

    stocks = fetch_nse_stocks_from_nse_scraper()
    update_cache(stocks)
    return stocks

def fetch_stock_data_for_ticker(ticker: str) -> Optional[Stock]:
    all_stocks = fetch_live_stocks()
    for stock in all_stocks:
        if stock.ticker == ticker.upper():
            return stock
    return None

# --- NEW: WhatsApp Test Function ---
async def send_whatsapp_test_message():
    """
    Sends a test message using the WhatsApp API from RapidAPI.
    This function will be scheduled by APScheduler.
    """
    rapidapi_key = os.getenv("WHATSAPP_RAPIDAPI_KEY") # Get the key from environment variables
    instance_unique_key = os.getenv("WHATSAPP_INSTANCE_UNIQUE_KEY") # Get the instance key
    admin_phone_number = "+254797780877" # Your admin number

    if not rapidapi_key:
        logger.error("❌ 'WHATSAPP_RAPIDAPI_KEY' environment variable not set. Cannot send message.")
        return

    if not instance_unique_key:
        logger.error("❌ 'WHATSAPP_INSTANCE_UNIQUE_KEY' environment variable not set. Cannot send message. Link your WhatsApp account first.")
        return

    url = "https://whatsapp-api98.p.rapidapi.com/send-whatsapp-message"
    payload = {
        "account": instance_unique_key,  # The unique identifier for your linked WhatsApp number
        "recipient": admin_phone_number,
        "message": f"Hello Admin, this is a test message from Treval AI NSE App. Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.",
        "type": "text" # Specify the message type
    }
    headers = {
        "x-rapidapi-key": rapidapi_key, # Use the key provided by RapidAPI
        "x-rapidapi-host": "whatsapp-api98.p.rapidapi.com", # Use the host from the API docs
        "Content-Type": "application/json"
    }

    try:
        logger.info(f"Attempting to send WhatsApp message to {admin_phone_number}...")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status() # Raises an exception for bad status codes (4xx or 5xx)

        result = response.json()
        logger.info(f"✅ WhatsApp API Response: {result}")
        # Check the response structure for success indication (adjust based on actual API response)
        # Example: if result.get('success'):
        #             logger.info("✅ Message sent successfully!")

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Failed to send WhatsApp message: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"   Status Code: {e.response.status_code}")
            logger.error(f"   Response Text: {e.response.text}")
    except Exception as e:
        logger.error(f"💥 Unexpected error sending WhatsApp message: {e}")

# --- NEW: Scheduler Setup ---
scheduler = AsyncIOScheduler()

# 1. Schedule a single test message 10 minutes from now
start_time_single = datetime.now() + timedelta(minutes=10)
scheduler.add_job(send_whatsapp_test_message, DateTrigger(run_date=start_time_single), id='test_msg_10_min')

# 2. Schedule recurring messages every hour starting tomorrow at 8 AM EAT (5 AM UTC)
#    Cron: At minute 0 past hour 5 (5 AM UTC = 8 AM EAT)
scheduler.add_job(send_whatsapp_test_message, CronTrigger(minute=0, hour=5, timezone='UTC'), id='daily_recurring_msgs', start_date=datetime.now().replace(hour=5, minute=0, second=0, microsecond=0) + timedelta(days=1))

scheduler.start()
logger.info("⏰ Background scheduler started for WhatsApp test messages.")

# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())

# ----------------------------------------------------
# API Endpoints
# ----------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Treval AI NSE Live Data API v2.1.3",
        "source": "NSE Scraper API (RapidAPI)",
        "features": ["Live Data", "Volume/Change Correction", "WhatsApp Notifications"],
        "cloud_hosted": True,
        "timestamp": datetime.now().isoformat(),
        "scheduled_whatsapp_test": f"A test message is scheduled for {datetime.now() + timedelta(minutes=10)}",
        "scheduled_whatsapp_daily": "Messages are scheduled daily at 8 AM EAT (5 AM UTC)"
    }

@app.get("/api/v1/stocks")
async def get_all_stocks():
    return fetch_live_stocks()

@app.get("/api/v1/stock/{ticker}")
async def get_stock_details(ticker: str):
    stock = fetch_stock_data_for_ticker(ticker)
    if stock is None:
        raise HTTPException(status_code=404, detail=f"Stock '{ticker}' not found.")
    return stock

@app.get("/api/v1/market-summary")
async def market_summary():
    stocks = fetch_live_stocks()
    if not stocks:
        return {"message": "No live stock data available."}

    total_stocks = len(stocks)
    gainers = [s for s in stocks if s.change_percent > 0]
    losers = [s for s in stocks if s.change_percent < 0]
    unchanged = [s for s in stocks if s.change_percent == 0]
    avg_change = sum(s.change_percent for s in stocks) / total_stocks if total_stocks > 0 else 0

    return {
        "timestamp": datetime.now().isoformat(),
        "total_stocks_analyzed": total_stocks,
        "gainers_count": len(gainers),
        "losers_count": len(losers),
        "unchanged_count": len(unchanged),
        "average_change_percent": round(avg_change, 2),
        "total_volume": sum(s.volume for s in stocks),
        "top_gainer": max(gainers, key=lambda x: x.change_percent).ticker if gainers else None,
        "top_loser": min(losers, key=lambda x: x.change_percent).ticker if losers else None
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "time": datetime.now().isoformat()}

# --- END OF FILE ---
