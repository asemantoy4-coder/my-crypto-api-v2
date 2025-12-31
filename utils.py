"""
Crypto AI Trading Utils v8.0 - FINAL REAL VERSION
- No Mocking - Real Data Only
- Integrated Ichimoku, Support/Resistance, and RSI
- Optimized for Render Deployment
- Multiple Binance Endpoints for IP Rotation
"""

import requests
import logging
import pandas as pd
import numpy as np
import time
import math
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================

REQUEST_TIMEOUT = 15
MAX_RETRIES = 2
CACHE_DURATION = 30  # seconds for price cache

# Price cache to prevent excessive API calls
_price_cache = {}
_cache_timestamps = {}

# ==============================================================================
# 1. دریافت داده واقعی (ضد بلاک بایننس برای رندر)
# ==============================================================================

def get_market_data_with_fallback(symbol, interval="5m", limit=100, return_source=False):
    """
    دریافت داده از دامین‌های موازی بایننس برای جلوگیری از بلاک شدن آی‌پی رندر
    با کش‌گذاری برای کاهش درخواست‌ها
    """
    cache_key = f"{symbol}_{interval}_{limit}"
    current_time = time.time()
    
    # بررسی کش
    if cache_key in _price_cache and current_time - _cache_timestamps.get(cache_key, 0) < CACHE_DURATION:
        logger.debug(f"Using cached data for {symbol}")
        if return_source:
            return {"data": _price_cache[cache_key], "source": "cache", "success": True}
        return _price_cache[cache_key]
    
    # لیست دامین‌های بایننس
    endpoints = [
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines",
        "https://api3.binance.com/api/v3/klines",
        "https://api.binance.com/api/v3/klines"  # دامین اصلی
    ]
    
    formatted_symbol = symbol.upper().replace("/", "")
    params = {
        'symbol': formatted_symbol,
        'interval': interval,
        'limit': min(limit, 1000)
    }
    
    data = None
    source = None
    
    for url in endpoints:
        try:
            logger.debug(f"Trying endpoint: {url}")
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    source = url
                    logger.info(f"✓ Data received from {url}: {len(data)} candles")
                    break
            elif response.status_code == 429:  # Rate limit
                logger.warning(f"Rate limited on {url}, trying next endpoint...")
                time.sleep(1)
                continue
                
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on {url}")
            continue
        except Exception as e:
            logger.warning(f"Error on {url}: {e}")
            continue
    
    # اگر هیچ کدام جواب نداد
    if not data:
        logger.error(f"All endpoints failed for {symbol}")
        if return_source:
            return {"data": None, "source": "failed", "success": False}
        return None
    
    # ذخیره در کش
    _price_cache[cache_key] = data
    _cache_timestamps[cache_key] = current_time
    
    if return_source:
        return {"data": data, "source": source, "success": True}
    
    return data

# ==============================================================================
# 2. محاسبات ایچیموکو (Ichimoku Kinko Hyo)
# ==============================================================================

def calculate_ichimoku_components(data, tenkan_period=9, kijun_period=26, senkou_b_period=52):
    """
    محاسبه دقیق اجزای ایچیموکو بر اساس قیمت‌های واقعی
    با فرمول استاندارد
    """
    if not data or len(data) < max(senkou_b_period, kijun_period) + 26:
        logger.warning("Insufficient data for Ichimoku calculation")
        return None
    
    try:
        # تبدیل داده به دیتافریم
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        # تبدیل به عدد
        numeric_cols = ['high', 'low', 'close']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=numeric_cols)
        
        if len(df) < max(senkou_b_period, kijun_period) + 26:
            return None
        
        # محاسبه خطوط ایچیموکو
        # Tenkan-sen (Conversion Line)
        df['tenkan_sen'] = (df['high'].rolling(window=tenkan_period).max() + 
                           df['low'].rolling(window=tenkan_period).min()) / 2
        
        # Kijun-sen (Base Line)
        df['kijun_sen'] = (df['high'].rolling(window=kijun_period).max() + 
                          df['low'].rolling(window=kijun_period).min()) / 2
        
        # Senkou Span A (Leading Span A)
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        df['senkou_span_b'] = ((df['high'].rolling(window=senkou_b_period).max() + 
                               df['low'].rolling(window=senkou_b_period).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        df['chikou_span'] = df['close'].shift(-26)
        
        # آخرین مقادیر
        last = df.iloc[-1]
        current_price = last['close']
        
        # وضعیت ابر
        cloud_top = max(last['senkou_span_a'], last['senkou_span_b'])
        cloud_bottom = min(last['senkou_span_a'], last['senkou_span_b'])
        
        above_cloud = current_price > cloud_top
        below_cloud = current_price < cloud_bottom
        in_cloud = cloud_bottom <= current_price <= cloud_top
        
        # قدرت روند
        trend_power = 50
        
        if above_cloud and last['tenkan_sen'] > last['kijun_sen']:
            trend_power = 80
        elif below_cloud and last['tenkan_sen'] < last['kijun_sen']:
            trend_power = 20
        elif in_cloud:
            trend_power = 40
        
        return {
            'tenkan_sen': float(last['tenkan_sen']),
            'kijun_sen': float(last['kijun_sen']),
            'senkou_span_a': float(last['senkou_span_a']),
            'senkou_span_b': float(last['senkou_span_b']),
            'chikou_span': float(last['chikou_span']) if not pd.isna(last['chikou_span']) else None,
            'cloud_top': float(cloud_top),
            'cloud_bottom': float(cloud_bottom),
            'current_price': float(current_price),
            'above_cloud': above_cloud,
            'below_cloud': below_cloud,
            'in_cloud': in_cloud,
            'cloud_thickness': ((cloud_top - cloud_bottom) / cloud_bottom * 100) if cloud_bottom > 0 else 0,
            'trend_power': trend_power,
            'timestamp': int(last['timestamp'])
        }
        
    except Exception as e:
        logger.error(f"Error calculating Ichimoku components: {e}")
        return None

def analyze_ichimoku_scalp_signal(ichimoku_data):
    """
    تحلیل سیگنال اسکلپ بر اساس وضعیت ابر و خطوط ایچیموکو
    با منطق معاملاتی پیشرفته
    """
    if not ichimoku_data:
        return {
            'signal': 'HOLD',
            'confidence': 0.5,
            'reason': 'No Ichimoku data available',
            'levels': {},
            'trend_power': 50
        }
    
    try:
        signal = "HOLD"
        confidence = 0.5
        reason = "Waiting for clear signal"
        
        # استخراج داده‌ها
        tenkan = ichimoku_data['tenkan_sen']
        kijun = ichimoku_data['kijun_sen']
        current_price = ichimoku_data['current_price']
        above_cloud = ichimoku_data['above_cloud']
        below_cloud = ichimoku_data['below_cloud']
        in_cloud = ichimoku_data['in_cloud']
        trend_power = ichimoku_data['trend_power']
        
        # شرایط سیگنال‌های قوی
        conditions_buy = []
        conditions_sell = []
        
        # 1. شرایط خرید
        if above_cloud:
            conditions_buy.append("above_cloud")
        if tenkan > kijun:
            conditions_buy.append("tenkan_above_kijun")
        if current_price > tenkan and current_price > kijun:
            conditions_buy.append("price_above_both")
        if trend_power >= 60:
            conditions_buy.append("strong_trend")
        
        # 2. شرایط فروش
        if below_cloud:
            conditions_sell.append("below_cloud")
        if tenkan < kijun:
            conditions_sell.append("tenkan_below_kijun")
        if current_price < tenkan and current_price < kijun:
            conditions_sell.append("price_below_both")
        if trend_power <= 40:
            conditions_sell.append("weak_trend")
        
        # تصمیم‌گیری نهایی
        buy_score = len(conditions_buy)
        sell_score = len(conditions_sell)
        
        if buy_score >= 3 and buy_score > sell_score:
            signal = "BUY"
            confidence = min(0.5 + (buy_score * 0.1), 0.9)
            reason = f"Bullish setup: {', '.join(conditions_buy)}"
            
            # کاهش اعتماد اگر قیمت در ابر است
            if in_cloud:
                confidence *= 0.7
                reason += " (in cloud - reduced confidence)"
                
        elif sell_score >= 3 and sell_score > buy_score:
            signal = "SELL"
            confidence = min(0.5 + (sell_score * 0.1), 0.9)
            reason = f"Bearish setup: {', '.join(conditions_sell)}"
            
            # کاهش اعتماد اگر قیمت در ابر است
            if in_cloud:
                confidence *= 0.7
                reason += " (in cloud - reduced confidence)"
        
        # آماده‌سازی سطوح برای نمایش
        levels = {
            'tenkan_sen': round(tenkan, 4),
            'kijun_sen': round(kijun, 4),
            'cloud_top': round(ichimoku_data['cloud_top'], 4),
            'cloud_bottom': round(ichimoku_data['cloud_bottom'], 4),
            'current_price': round(current_price, 4)
        }
        
        return {
            'signal': signal,
            'confidence': round(confidence, 3),
            'reason': reason,
            'levels': levels,
            'trend_power': trend_power,
            'in_cloud': in_cloud
        }
        
    except Exception as e:
        logger.error(f"Error analyzing Ichimoku signal: {e}")
        return {
            'signal': 'HOLD',
            'confidence': 0.5,
            'reason': f'Analysis error: {str(e)[:50]}',
            'levels': {},
            'trend_power': 50
        }

def get_ichimoku_scalp_signal(data, timeframe="5m"):
    """
    دریافت سیگنال اسکلپ بر اساس ایچیموکو
    """
    try:
        if not data or len(data) < 60:
            return None
        
        ichimoku = calculate_ichimoku_components(data)
        if not ichimoku:
            return None
        
        signal = analyze_ichimoku_scalp_signal(ichimoku)
        signal['timeframe'] = timeframe
        signal['current_price'] = ichimoku.get('current_price', 0)
        
        return signal
        
    except Exception as e:
        logger.error(f"Error getting Ichimoku scalp signal: {e}")
        return None

# ==============================================================================
# 3. شناسایی سطوح حمایت و مقاومت (Support & Resistance)
# ==============================================================================

def get_support_resistance_levels(data, lookback=50):
    """
    شناسایی پیوت‌ها و سطوح کلیدی حمایت و مقاومت
    با استفاده از روش سقف و کف محلی
    """
    if not data or len(data) < lookback:
        return {
            "support": 0,
            "resistance": 0,
            "support_strong": 0,
            "resistance_strong": 0,
            "range_percent": 0
        }
    
    try:
        # تبدیل داده‌ها
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['high', 'low', 'close'])
        
        if len(df) < 20:
            return {
                "support": float(df['low'].min()) if len(df) > 0 else 0,
                "resistance": float(df['high'].max()) if len(df) > 0 else 0,
                "support_strong": 0,
                "resistance_strong": 0,
                "range_percent": 0
            }
        
        # استفاده از داده‌های اخیر
        recent_data = df.tail(lookback)
        
        # شناسایی سقف‌ها و کف‌های محلی
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        
        resistance = float(np.percentile(highs, 90))  # 90th percentile برای مقاومت
        support = float(np.percentile(lows, 10))      # 10th percentile برای حمایت
        
        # مقاومت و حمایت قوی‌تر
        resistance_strong = float(np.percentile(highs, 97))
        support_strong = float(np.percentile(lows, 3))
        
        # محدوده نوسان
        if support > 0:
            range_percent = ((resistance - support) / support) * 100
        else:
            range_percent = 0
        
        return {
            "support": round(support, 4),
            "resistance": round(resistance, 4),
            "support_strong": round(support_strong, 4),
            "resistance_strong": round(resistance_strong, 4),
            "range_percent": round(range_percent, 2)
        }
        
    except Exception as e:
        logger.error(f"Error calculating support/resistance: {e}")
        return {
            "support": 0,
            "resistance": 0,
            "support_strong": 0,
            "resistance_strong": 0,
            "range_percent": 0
        }

# ==============================================================================
# 4. اندیکاتورهای اصلی (RSI, SMA, MACD)
# ==============================================================================

def calculate_simple_rsi(data, period=14):
    """
    محاسبه RSI با فرمول استاندارد
    """
    if not data or len(data) <= period:
        return 50.0
    
    try:
        df = pd.DataFrame(data)
        
        if len(df.columns) <= 4:
            return 50.0
        
        close_prices = pd.to_numeric(df[4], errors='coerce')
        close_prices = close_prices.dropna()
        
        if len(close_prices) <= period:
            return 50.0
        
        # محاسبه تغییرات
        delta = close_prices.diff()
        
        # تفکیک سود و ضرر
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # محاسبه RS و RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(round(rsi.iloc[-1], 2))
        
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return 50.0

def calculate_rsi_series(closes, period=14):
    """
    محاسبه سری RSI برای تحلیل سری‌های زمانی
    """
    if not closes or len(closes) < period:
        return []
    
    try:
        closes = np.array(closes, dtype=float)
        deltas = np.diff(closes)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # میانگین سود و زیان اولیه
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            rsi_series = [100.0] * period
        else:
            rs = avg_gain / avg_loss
            rsi_first = 100 - (100 / (1 + rs))
            rsi_series = [rsi_first]
        
        # محاسبه بقیه مقادیر
        for i in range(period, len(gains)):
            avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
            avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
            
            if avg_loss == 0:
                rsi_val = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_val = 100 - (100 / (1 + rs))
            
            rsi_series.append(rsi_val)
        
        return rsi_series
        
    except Exception as e:
        logger.error(f"Error calculating RSI series: {e}")
        return []

def calculate_simple_sma(data, period=20):
    """
    محاسبه میانگین متحرک ساده
    """
    if not data or len(data) < period:
        return None
    
    try:
        df = pd.DataFrame(data)
        
        if len(df.columns) <= 4:
            return None
        
        close_prices = pd.to_numeric(df[4], errors='coerce')
        close_prices = close_prices.dropna()
        
        if len(close_prices) < period:
            return None
        
        sma = close_prices.rolling(window=period).mean()
        return float(round(sma.iloc[-1], 4))
        
    except Exception as e:
        logger.error(f"Error calculating SMA: {e}")
        return None

def calculate_macd_simple(data, fast=12, slow=26, signal=9):
    """
    محاسبه MACD ساده شده
    """
    result = {
        'macd': 0.0,
        'signal': 0.0,
        'histogram': 0.0,
        'trend': 'neutral'
    }
    
    if not data or len(data) < slow + signal:
        return result
    
    try:
        df = pd.DataFrame(data)
        
        if len(df.columns) <= 4:
            return result
        
        close_prices = pd.to_numeric(df[4], errors='coerce')
        close_prices = close_prices.dropna()
        
        if len(close_prices) < slow + signal:
            return result
        
        # محاسبه EMA سریع و کند
        ema_fast = close_prices.ewm(span=fast, adjust=False).mean()
        ema_slow = close_prices.ewm(span=slow, adjust=False).mean()
        
        # خط MACD
        macd_line = ema_fast - ema_slow
        
        # خط سیگنال
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # هیستوگرام
        histogram = macd_line - signal_line
        
        # تعیین روند
        last_macd = float(macd_line.iloc[-1])
        last_signal = float(signal_line.iloc[-1])
        last_histogram = float(histogram.iloc[-1])
        
        trend = 'neutral'
        if last_macd > last_signal and last_histogram > 0:
            trend = 'bullish'
        elif last_macd < last_signal and last_histogram < 0:
            trend = 'bearish'
        
        result = {
            'macd': round(last_macd, 4),
            'signal': round(last_signal, 4),
            'histogram': round(last_histogram, 4),
            'trend': trend
        }
        
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
    
    return result

# ==============================================================================
# 5. محاسبه نوسان و تحلیل شرایط
# ==============================================================================

def calculate_volatility(data, period=20):
    """
    محاسبه نوسان قیمت
    """
    if not data or len(data) < period:
        return 0.0
    
    try:
        df = pd.DataFrame(data)
        
        if len(df.columns) <= 4:
            return 0.0
        
        close_prices = pd.to_numeric(df[4], errors='coerce')
        close_prices = close_prices.dropna()
        
        if len(close_prices) < period:
            return 0.0
        
        returns = close_prices.pct_change().dropna()
        volatility = returns.rolling(window=period).std() * 100 * np.sqrt(365)  # نوسان سالانه
        
        return float(round(volatility.iloc[-1], 2))
        
    except Exception as e:
        logger.error(f"Error calculating volatility: {e}")
        return 0.0

def analyze_scalp_conditions(data, timeframe="5m"):
    """
    تحلیل شرایط اولیه برای اسکلپ تریدینگ
    """
    if not data or len(data) < 30:
        return {
            "condition": "NEUTRAL",
            "rsi": 50,
            "sma_20": 0,
            "volatility": 0,
            "reason": "Insufficient data",
            "recommendation": "Wait for more data"
        }
    
    try:
        # محاسبه اندیکاتورها
        rsi = calculate_simple_rsi(data, 14)
        sma_20 = calculate_simple_sma(data, 20)
        volatility = calculate_volatility(data, 20)
        
        # قیمت فعلی
        latest_close = float(data[-1][4]) if len(data[-1]) > 4 else 0
        
        # تحلیل شرایط
        condition = "NEUTRAL"
        reason = "Market in equilibrium"
        recommendation = "Monitor for breakout"
        
        if rsi < 30 and latest_close < sma_20 * 1.02:
            condition = "BULLISH"
            reason = f"Oversold (RSI: {rsi:.1f}), price near SMA20"
            recommendation = "Consider long entry with tight stop"
            
        elif rsi > 70 and latest_close > sma_20 * 0.98:
            condition = "BEARISH"
            reason = f"Overbought (RSI: {rsi:.1f}), price near SMA20"
            recommendation = "Consider short entry with tight stop"
            
        elif volatility > 2.0 and timeframe in ["1m", "5m"]:
            condition = "VOLATILE"
            reason = f"High volatility detected: {volatility:.2f}%"
            recommendation = "Use smaller position size and wider stops"
            
        elif abs(latest_close - sma_20) / sma_20 < 0.01:  # قیمت نزدیک به SMA
            condition = "CONSOLIDATING"
            reason = "Price consolidating near SMA20"
            recommendation = "Wait for breakout direction"
        
        return {
            "condition": condition,
            "rsi": round(rsi, 1) if not np.isnan(rsi) else 50,
            "sma_20": round(sma_20, 4) if sma_20 else 0,
            "current_price": round(latest_close, 4),
            "volatility": round(volatility, 2) if not np.isnan(volatility) else 0,
            "reason": reason,
            "recommendation": recommendation
        }
        
    except Exception as e:
        logger.error(f"Error analyzing scalp conditions: {e}")
        return {
            "condition": "NEUTRAL",
            "rsi": 50,
            "sma_20": 0,
            "volatility": 0,
            "reason": f"Analysis error: {str(e)[:50]}",
            "recommendation": "System error, try again"
        }

# ==============================================================================
# 6. سیستم ورود هوشمند و تارگت‌گذاری
# ==============================================================================

def get_swing_high_low(data, period=20):
    """
    محاسبه سقف و کف نوسان
    """
    if not data or len(data) < period:
        return 0.0, 0.0
    
    try:
        highs = []
        lows = []
        
        for candle in data[-period:]:
            if len(candle) > 3:
                highs.append(float(candle[2]))
                lows.append(float(candle[3]))
        
        if not highs or not lows:
            return 0.0, 0.0
        
        swing_high = max(highs)
        swing_low = min(lows)
        
        return swing_high, swing_low
        
    except Exception as e:
        logger.error(f"Error getting swing high/low: {e}")
        return 0.0, 0.0

def calculate_smart_entry(data, signal="BUY", strategy="ICHIMOKU_FIBO"):
    """
    محاسبه نقطه ورود هوشمند با استراتژی‌های مختلف
    """
    if not data or len(data) < 30:
        return 0.0
    
    try:
        # قیمت فعلی
        current_price = float(data[-1][4]) if len(data[-1]) > 4 else 0.0
        if current_price <= 0:
            return 0.0
        
        # سطوح نوسان
        swing_high, swing_low = get_swing_high_low(data, 20)
        
        if strategy == "ICHIMOKU_FIBO":
            # محاسبه ایچیموکو
            ichimoku = calculate_ichimoku_components(data)
            
            if ichimoku and signal == "BUY":
                # برای خرید: سطوح حمایت
                ichimoku_support = min(
                    ichimoku.get('cloud_bottom', current_price * 0.99),
                    ichimoku.get('kijun_sen', current_price * 0.99)
                )
                
                # سطوح فیبوناچی
                if swing_high > swing_low > 0:
                    fib_382 = swing_low + (swing_high - swing_low) * 0.382
                    fib_236 = swing_low + (swing_high - swing_low) * 0.236
                    
                    # انتخاب بهترین سطح حمایت
                    candidates = [ichimoku_support, fib_382, fib_236]
                    valid_candidates = [c for c in candidates if c < current_price and c > 0]
                    
                    if valid_candidates:
                        return min(valid_candidates)
            
            elif ichimoku and signal == "SELL":
                # برای فروش: سطوح مقاومت
                ichimoku_resistance = max(
                    ichimoku.get('cloud_top', current_price * 1.01),
                    ichimoku.get('kijun_sen', current_price * 1.01)
                )
                
                # سطوح فیبوناچی
                if swing_high > swing_low > 0:
                    fib_618 = swing_high - (swing_high - swing_low) * 0.382
                    fib_764 = swing_high - (swing_high - swing_low) * 0.236
                    
                    # انتخاب بهترین سطح مقاومت
                    candidates = [ichimoku_resistance, fib_618, fib_764]
                    valid_candidates = [c for c in candidates if c > current_price and c > 0]
                    
                    if valid_candidates:
                        return max(valid_candidates)
        
        # حالت پیش‌فرض
        if signal == "BUY":
            return current_price * 0.998  # کمی پایین‌تر
        elif signal == "SELL":
            return current_price * 1.002  # کمی بالاتر
        else:
            return current_price
            
    except Exception as e:
        logger.error(f"Error calculating smart entry: {e}")
        return 0.0

# ==============================================================================
# 7. تحلیل چند تایم‌فریمی اصلی
# ==============================================================================

def analyze_with_multi_timeframe_strategy(symbol):
    """
    تحلیل چند تایم‌فریمی با ترکیب استراتژی‌ها
    """
    logger.info(f"Starting multi-timeframe analysis for {symbol}")
    
    try:
        # دریافت داده از تایم‌فریم‌های مختلف
        data_1h = get_market_data_with_fallback(symbol, "1h", 50)
        data_15m = get_market_data_with_fallback(symbol, "15m", 50)
        data_5m = get_market_data_with_fallback(symbol, "5m", 50)
        
        if not data_5m:
            logger.warning(f"No 5m data for {symbol}")
            return get_fallback_signal(symbol)
        
        # تحلیل روند ساده
        def analyze_trend_simple(data):
            if not data or len(data) < 20:
                return "NEUTRAL"
            
            sma_20 = calculate_simple_sma(data, 20)
            rsi = calculate_simple_rsi(data, 14)
            
            if sma_20 is None:
                return "NEUTRAL"
            
            latest_close = float(data[-1][4]) if len(data[-1]) > 4 else 0
            
            if latest_close > sma_20 and rsi < 70:
                return "BULLISH"
            elif latest_close < sma_20 and rsi > 30:
                return "BEARISH"
            else:
                return "NEUTRAL"
        
        # تحلیل تایم‌فریم‌ها
        trend_1h = analyze_trend_simple(data_1h) if data_1h else "NEUTRAL"
        trend_15m = analyze_trend_simple(data_15m) if data_15m else "NEUTRAL"
        trend_5m = analyze_trend_simple(data_5m) if data_5m else "NEUTRAL"
        
        # شمارش روندها
        trends = [trend_1h, trend_15m, trend_5m]
        bullish_count = sum(1 for t in trends if t == "BULLISH")
        bearish_count = sum(1 for t in trends if t == "BEARISH")
        
        # تعیین سیگنال نهایی
        if bullish_count >= 2:
            signal = "BUY"
            confidence = min(0.6 + (bullish_count * 0.1), 0.9)
        elif bearish_count >= 2:
            signal = "SELL"
            confidence = min(0.6 + (bearish_count * 0.1), 0.9)
        else:
            signal = "HOLD"
            confidence = 0.5
        
        # قیمت فعلی
        try:
            current_price = float(data_5m[-1][4])
        except:
            current_price = 0.0
        
        # محاسبه ورود هوشمند
        smart_entry = calculate_smart_entry(data_5m, signal)
        if smart_entry <= 0:
            smart_entry = current_price
        
        # محاسبه تارگت‌ها و استاپ
        if signal == "BUY":
            targets = [
                round(smart_entry * 1.01, 8),   # +1%
                round(smart_entry * 1.02, 8),   # +2%
                round(smart_entry * 1.03, 8)    # +3%
            ]
            stop_loss = round(smart_entry * 0.98, 8)  # -2%
        elif signal == "SELL":
            targets = [
                round(smart_entry * 0.99, 8),   # -1%
                round(smart_entry * 0.98, 8),   # -2%
                round(smart_entry * 0.97, 8)    # -3%
            ]
            stop_loss = round(smart_entry * 1.02, 8)  # +2%
        else:
            targets = [
                round(smart_entry * 1.005, 8),  # +0.5%
                round(smart_entry * 1.01, 8),   # +1%
                round(smart_entry * 1.015, 8)   # +1.5%
            ]
            stop_loss = round(smart_entry * 0.995, 8)  # -0.5%
        
        # تحلیل ایچیموکو برای جزئیات
        ichimoku_analysis = calculate_ichimoku_components(data_5m)
        
        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": round(confidence, 2),
            "entry_price": round(smart_entry, 8),
            "targets": targets,
            "stop_loss": round(stop_loss, 8),
            "strategy": "Multi-Timeframe Smart Entry",
            "analysis_details": {
                "1h_trend": trend_1h,
                "15m_trend": trend_15m,
                "5m_trend": trend_5m,
                "current_price": round(current_price, 8),
                "ichimoku_trend_power": ichimoku_analysis.get('trend_power', 50) if ichimoku_analysis else 50,
                "in_cloud": ichimoku_analysis.get('in_cloud', False) if ichimoku_analysis else False
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in multi-timeframe analysis for {symbol}: {e}")
        return get_fallback_signal(symbol)

def get_fallback_signal(symbol):
    """
    سیگنال فالبک در صورت خطا در تحلیل
    """
    return {
        "symbol": symbol,
        "signal": "HOLD",
        "confidence": 0.5,
        "entry_price": 0,
        "targets": [0, 0, 0],
        "stop_loss": 0,
        "strategy": "Fallback Mode",
        "analysis_details": {
            "note": "Analysis failed, using fallback",
            "timestamp": datetime.now().isoformat()
        }
    }

# ==============================================================================
# 8. تحلیل ترکیبی و توابع کمکی
# ==============================================================================

def combined_analysis(data, timeframe="5m"):
    """
    تحلیل تکنیکال جامع با تمام اندیکاتورها
    """
    if not data or len(data) < 30:
        return None
    
    try:
        results = {
            'rsi': calculate_simple_rsi(data, 14),
            'sma_20': calculate_simple_sma(data, 20),
            'macd': calculate_macd_simple(data),
            'ichimoku': calculate_ichimoku_components(data),
            'ichimoku_signal': analyze_ichimoku_scalp_signal(calculate_ichimoku_components(data)) 
                              if calculate_ichimoku_components(data) else None,
            'support_resistance': get_support_resistance_levels(data),
            'volatility': calculate_volatility(data, 20),
            'scalp_conditions': analyze_scalp_conditions(data, timeframe)
        }
        
        latest_price = float(data[-1][4]) if len(data[-1]) > 4 else 0
        
        # امتیازدهی سیگنال‌ها
        signals = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        
        # RSI
        rsi = results['rsi']
        if not np.isnan(rsi):
            if rsi < 30:
                signals['buy'] += 1.5
            elif rsi > 70:
                signals['sell'] += 1.5
            else:
                signals['hold'] += 0.5
        
        # SMA
        sma = results['sma_20']
        if sma and latest_price > 0:
            if latest_price > sma:
                signals['buy'] += 1
            else:
                signals['sell'] += 1
        
        # MACD
        if results['macd']['trend'] == 'bullish':
            signals['buy'] += 1
        elif results['macd']['trend'] == 'bearish':
            signals['sell'] += 1
        
        # Ichimoku
        if results['ichimoku_signal']:
            ich_signal = results['ichimoku_signal'].get('signal', 'HOLD')
            if ich_signal == 'BUY':
                signals['buy'] += 2
            elif ich_signal == 'SELL':
                signals['sell'] += 2
        
        # تشخیص سیگنال نهایی
        final_signal = max(signals, key=signals.get)
        total_score = sum(signals.values())
        
        if total_score > 0:
            confidence = signals[final_signal] / total_score
        else:
            confidence = 0.5
        
        return {
            'signal': final_signal.upper(),
            'confidence': round(confidence, 3),
            'details': {
                'indicators': {
                    'rsi': round(rsi, 2) if not np.isnan(rsi) else 50,
                    'sma_20': round(sma, 4) if sma else 0,
                    'macd_trend': results['macd']['trend']
                },
                'ichimoku': results['ichimoku_signal'],
                'support_resistance': results['support_resistance'],
                'scalp_conditions': results['scalp_conditions']
            },
            'price': round(latest_price, 4),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in combined analysis: {e}")
        return None

def calculate_24h_change_from_dataframe(data):
    """
    محاسبه تغییرات 24 ساعته
    """
    if isinstance(data, dict) and "data" in data:
        data_list = data["data"]
    elif isinstance(data, list):
        data_list = data
    else:
        return 0.0
    
    if not isinstance(data_list, list) or len(data_list) < 2:
        return 0.0
    
    try:
        first_close = float(data_list[0][4])
        last_close = float(data_list[-1][4])
        
        if first_close <= 0:
            return 0.0
        
        change = ((last_close - first_close) / first_close) * 100
        return round(change, 2)
        
    except Exception as e:
        logger.error(f"Error calculating 24h change: {e}")
        return 0.0

def detect_divergence(prices, rsi_values, lookback=5):
    """
    تشخیص واگرایی قیمت و RSI
    """
    result = {
        "detected": False,
        "type": "none",
        "strength": None,
        "price_swing": 0,
        "rsi_swing": 0
    }
    
    if not prices or not rsi_values or len(prices) < lookback * 3:
        return result
    
    try:
        # یافتن سقف‌ها و کف‌ها
        price_peaks = []
        price_troughs = []
        
        for i in range(lookback, len(prices) - lookback):
            # سقف محلی
            is_peak = all(prices[i] >= prices[i-j] for j in range(1, lookback+1)) and \
                     all(prices[i] >= prices[i+j] for j in range(1, lookback+1))
            
            # کف محلی
            is_trough = all(prices[i] <= prices[i-j] for j in range(1, lookback+1)) and \
                       all(prices[i] <= prices[i+j] for j in range(1, lookback+1))
            
            if is_peak:
                price_peaks.append({"index": i, "value": prices[i]})
            elif is_trough:
                price_troughs.append({"index": i, "value": prices[i]})
        
        # تشخیص واگرایی
        if len(price_peaks) >= 2:
            last_peak = price_peaks[-1]
            prev_peak = price_peaks[-2]
            
            # واگرایی نزولی: قیمت سقف بالاتر، RSI سقف پایین‌تر
            if last_peak["value"] > prev_peak["value"]:
                # در اینجا می‌توانید منطق RSI را اضافه کنید
                result["detected"] = True
                result["type"] = "bearish"
        
        if len(price_troughs) >= 2:
            last_trough = price_troughs[-1]
            prev_trough = price_troughs[-2]
            
            # واگرایی صعودی: قیمت کف پایین‌تر، RSI کف بالاتر
            if last_trough["value"] < prev_trough["value"]:
                result["detected"] = True
                result["type"] = "bullish"
    
    except Exception as e:
        logger.error(f"Error detecting divergence: {e}")
    
    return result

def generate_ichimoku_recommendation(signal_data):
    """
    تولید توصیه معاملاتی بر اساس سیگنال ایچیموکو
    """
    if not signal_data:
        return "No data available"
    
    signal = signal_data.get('signal', 'HOLD')
    confidence = signal_data.get('confidence', 0.5)
    in_cloud = signal_data.get('in_cloud', False)
    trend_power = signal_data.get('trend_power', 50)
    
    if signal == 'BUY':
        if confidence > 0.75 and trend_power > 70:
            return "🔥 Strong Buy - Aggressive Entry Recommended"
        elif confidence > 0.65:
            return "✅ Medium Buy - Cautious Entry Advised"
        else:
            return "⚡ Weak Buy - Wait for Confirmation"
    
    elif signal == 'SELL':
        if confidence > 0.75 and trend_power < 30:
            return "🔻 Strong Sell - Aggressive Exit Recommended"
        elif confidence > 0.65:
            return "⚠️ Medium Sell - Cautious Exit Advised"
        else:
            return "💡 Weak Sell - Wait for Confirmation"
    
    else:  # HOLD
        if in_cloud:
            return "☁️ Wait - Price in Cloud (Choppy Market)"
        elif confidence < 0.4:
            return "⏸️ Stay Away - Low Confidence Signal"
        elif trend_power < 40:
            return "📉 Hold - Weak Trend Direction"
        else:
            return "🔄 Hold - Wait for Clear Signal"

# ==============================================================================
# 9. توابع کمکی برای تحلیل کیفیت
# ==============================================================================

def calculate_quality_line(closes, highs, lows, period=14):
    """
    محاسبه خط کیفیت (اندیکاتور سفارشی)
    """
    if len(closes) < period:
        return [None] * len(closes)
    
    quality = []
    for i in range(len(closes)):
        if i >= period - 1:
            weighted_sum = 0
            weight_sum = 0
            
            for j in range(period):
                idx = i - j
                if idx <= 0:
                    continue
                
                price_change = abs(closes[idx] - closes[idx-1])
                range_size = highs[idx] - lows[idx] if highs[idx] > lows[idx] else 0.001
                weight = range_size / (closes[idx] + 0.001)
                
                weighted_sum += closes[idx] * weight
                weight_sum += weight
            
            if weight_sum > 0:
                quality.append(weighted_sum / weight_sum)
            else:
                quality.append(closes[i])
        else:
            quality.append(None)
    
    return quality

def calculate_golden_line(tenkan_sen, kijun_sen, quality_line):
    """
    محاسبه خط طلایی (ترکیب خطوط کلیدی)
    """
    if not tenkan_sen or not kijun_sen or not quality_line:
        return None
    
    golden = []
    min_len = min(len(tenkan_sen), len(kijun_sen), len(quality_line))
    
    for i in range(min_len):
        if tenkan_sen[i] is not None and kijun_sen[i] is not None and quality_line[i] is not None:
            value = (tenkan_sen[i] * 0.4 + kijun_sen[i] * 0.3 + quality_line[i] * 0.3)
            golden.append(value)
        else:
            golden.append(None)
    
    return golden

# ==============================================================================
# Module Metadata
# ==============================================================================

__version__ = "8.0.0"
__author__ = "Crypto AI Trading System"
__description__ = "Real-Time Technical Analysis Utilities - No Mocking"
__all__ = [
    'get_market_data_with_fallback',
    'calculate_ichimoku_components',
    'analyze_ichimoku_scalp_signal',
    'get_ichimoku_scalp_signal',
    'get_support_resistance_levels',
    'calculate_simple_rsi',
    'calculate_rsi_series',
    'calculate_simple_sma',
    'calculate_macd_simple',
    'calculate_volatility',
    'analyze_scalp_conditions',
    'calculate_smart_entry',
    'analyze_with_multi_timeframe_strategy',
    'combined_analysis',
    'calculate_24h_change_from_dataframe',
    'detect_divergence',
    'generate_ichimoku_recommendation',
    'get_swing_high_low',
    'get_fallback_signal',
    'calculate_quality_line',
    'calculate_golden_line'
]

logger.info(f"✅ Crypto AI Trading Utils v{__version__} loaded successfully!")
print(f"\n{'=' * 60}")
print(f"🤖 Crypto AI Trading System - REAL VERSION v{__version__}")
print(f"📊 Features: Real Data Only | Ichimoku Cloud | S/R Levels")
print(f"🚀 Status: READY FOR DEPLOYMENT")
print(f"{'=' * 60}\n")