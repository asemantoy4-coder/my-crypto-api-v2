# api/data_collector.py - نسخه 7.4.0
"""
Data Collector - Lightweight version
"""

from datetime import datetime
import random
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# تابع اصلی که در main.py شما استفاده شده
def get_collected_data(symbols=None, timeframe="5m", limit=50, include_analysis=False):
    """
    دریافت داده جمع‌آوری شده
    """
    if not symbols:
        symbols = ["BTCUSDT", "ETHUSDT"]
    
    results = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "symbols_analyzed": symbols,
        "price_data": {},
        "technical_analysis": {} if include_analysis else None
    }
    
    for symbol in symbols:
        symbol_upper = symbol.upper()
        
        # داده mock ساده
        base_price = 88271.00 if "BTC" in symbol_upper else 3450.00 if "ETH" in symbol_upper else 100.00
        
        results["price_data"][symbol_upper] = {
            "price": round(base_price * random.uniform(0.99, 1.01), 2),
            "high": round(base_price * random.uniform(1.005, 1.015), 2),
            "low": round(base_price * random.uniform(0.985, 0.995), 2),
            "volume": round(random.uniform(1000, 5000), 2),
            "timeframe": timeframe
        }
    
    return results

# تابع دوم که در main.py استفاده شده
def get_market_overview(timeframe="5m"):
    """
    دریافت نمای کلی بازار
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "timeframe": timeframe,
        "market_status": "open",
        "top_gainers": [],
        "top_losers": []
    }

# توابع اضافی که export می‌شوند
__all__ = ['get_collected_data', 'get_market_overview']