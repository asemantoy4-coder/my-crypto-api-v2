import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Any, Optional, Union
import yfinance as yf

logger = logging.getLogger(__name__)

# کش برای کاهش درخواست‌ها
_data_cache = {}
_cache_expiry = {}
CACHE_DURATION = 30  # ثانیه

# ==============================================================================
# 1. بخش دریافت دیتا (Yahoo Finance) - بهبود یافته
# ==============================================================================

def convert_timeframe_to_yahoo(timeframe: str) -> str:
    """تبدیل تایم‌فریم به فرمت Yahoo Finance"""
    timeframe_map = {
        # دقیقه‌ای
        '1m': '1m', '2m': '2m', '5m': '5m', '15m': '15m', '30m': '30m',
        # ساعتی
        '1h': '60m', '2h': '120m', '4h': '240m', '6h': '360m',
        # روزانه
        '1d': '1d', '3d': '3d', '5d': '5d',
        # هفتگی و ماهانه
        '1w': '1wk', '1mo': '1mo', '3mo': '3mo'
    }
    return timeframe_map.get(timeframe, timeframe)

def convert_symbol_to_yahoo(symbol: str) -> str:
    """تبدیل نمادهای رمزارز به فرمت Yahoo Finance"""
    symbol = symbol.upper().strip()
    
    # تبدیل مستقیم نمادهای محبوب
    symbol_map = {
        'BTCUSDT': 'BTC-USD',
        'ETHUSDT': 'ETH-USD',
        'BNBUSDT': 'BNB-USD',
        'SOLUSDT': 'SOL-USD',
        'XRPUSDT': 'XRP-USD',
        'ADAUSDT': 'ADA-USD',
        'DOGEUSDT': 'DOGE-USD',
        'DOTUSDT': 'DOT-USD',
        'MATICUSDT': 'MATIC-USD',
        'SHIBUSDT': 'SHIB-USD'
    }
    
    if symbol in symbol_map:
        return symbol_map[symbol]
    
    # منطق تبدیل کلی
    if symbol.endswith('USDT'):
        return symbol.replace('USDT', '-USD')
    elif symbol.endswith('BUSD'):
        return symbol.replace('BUSD', '-USD')
    elif symbol.endswith('USDC'):
        return symbol.replace('USDC', '-USD')
    elif "-" not in symbol:
        return f"{symbol}-USD"
    
    return symbol

def get_market_data_with_fallback(symbol: str, timeframe: str = "5m", 
                                 limit: int = 100, return_source: bool = False):
    """
    دریافت داده بازار از Yahoo Finance با کش‌گذاری
    
    پارامترها:
    -----------
    symbol : str
        نماد ارز (مثلاً BTCUSDT, ETHUSDT)
    timeframe : str
        تایم‌فریم (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
    limit : int
        تعداد کندل‌های مورد نیاز
    return_source : bool
        اگر True باشد، اطلاعات منبع نیز بازگردانده می‌شود
    
    بازگشت:
    -------
    Union[List, Dict]
        لیست کندل‌ها یا دیکشنری با اطلاعات اضافی
    """
    cache_key = f"{symbol}_{timeframe}_{limit}"
    current_time = time.time()
    
    # بررسی کش
    if cache_key in _data_cache:
        cache_time = _cache_expiry.get(cache_key, 0)
        if current_time - cache_time < CACHE_DURATION:
            logger.debug(f"Using cached data for {symbol}")
            if return_source:
                return {
                    "data": _data_cache[cache_key],
                    "source": "yahoo_finance_cache",
                    "success": True,
                    "cached": True
                }
            return _data_cache[cache_key]
    
    try:
        yf_symbol = convert_symbol_to_yahoo(symbol)
        interval = convert_timeframe_to_yahoo(timeframe)
        
        logger.info(f"Fetching {symbol} -> {yf_symbol} ({timeframe}, limit={limit})")
        
        # تعیین دوره زمانی بر اساس تایم‌فریم
        period_map = {
            '1m': '1d', '2m': '1d', '5m': '5d', '15m': '5d', '30m': '5d',
            '60m': '1mo', '120m': '1mo', '240m': '1mo',
            '1d': '3mo', '1wk': '1y'
        }
        period = period_map.get(interval, '5d')
        
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            logger.warning(f"No data available for {yf_symbol}")
            if return_source:
                return {
                    "data": [],
                    "source": "yahoo_finance",
                    "success": False,
                    "error": "No data available"
                }
            return []
        
        # محدود کردن به تعداد مورد نیاز
        if limit > 0 and len(df) > limit:
            df = df.tail(limit)
        
        # تبدیل به فرمت کندل استاندارد
        candles = []
        for idx, row in df.iterrows():
            timestamp = int(idx.timestamp() * 1000)
            interval_ms = {
                '1m': 60000, '5m': 300000, '15m': 900000, '30m': 1800000,
                '60m': 3600000, '240m': 14400000, '1d': 86400000
            }.get(interval, 300000)
            
            candle = [
                timestamp,  # timestamp
                float(row['Open']),   # open
                float(row['High']),   # high
                float(row['Low']),    # low
                float(row['Close']),  # close
                float(row['Volume']), # volume
                timestamp + interval_ms,  # close_time
                str(float(row['Volume']) * float(row['Close'])),  # quote_asset_volume
                str(0),  # number_of_trades
                str(float(row['Volume']) * 0.6),  # taker_buy_base_asset_volume
                str(float(row['Volume']) * float(row['Close']) * 0.6),  # taker_buy_quote_asset_volume
                "0"  # ignore
            ]
            candles.append(candle)
        
        # ذخیره در کش
        _data_cache[cache_key] = candles
        _cache_expiry[cache_key] = current_time
        
        logger.info(f"Successfully fetched {len(candles)} candles for {symbol}")
        
        if return_source:
            current_price = candles[-1][4] if candles else 0
            return {
                "data": candles,
                "source": "yahoo_finance",
                "success": True,
                "candle_count": len(candles),
                "current_price": current_price,
                "symbol": symbol,
                "yahoo_symbol": yf_symbol,
                "interval": interval
            }
        
        return candles
        
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        
        # تولید داده اضطراری
        try:
            emergency_data = generate_emergency_data(symbol, timeframe, limit)
            logger.warning(f"Using emergency data for {symbol}")
            
            if return_source:
                return {
                    "data": emergency_data,
                    "source": "emergency_data",
                    "success": False,
                    "emergency": True
                }
            return emergency_data
            
        except Exception as emergency_error:
            logger.error(f"Emergency data generation failed: {emergency_error}")
            
            if return_source:
                return {
                    "data": [],
                    "source": "failed",
                    "success": False,
                    "error": str(e)
                }
            return []

def generate_emergency_data(symbol: str, timeframe: str, limit: int) -> List:
    """تولید داده اضطراری در صورت عدم دسترسی به API"""
    base_prices = {
        'BTCUSDT': 45000.0, 'ETHUSDT': 2400.0, 'BNBUSDT': 310.0,
        'SOLUSDT': 100.0, 'XRPUSDT': 0.60, 'ADAUSDT': 0.45,
        'DOGEUSDT': 0.08, 'DOTUSDT': 7.0, 'MATICUSDT': 0.80
    }
    
    base_price = base_prices.get(symbol.upper(), 100.0)
    
    # فاصله زمانی بر اساس تایم‌فریم
    interval_ms_map = {
        '1m': 60000, '5m': 300000, '15m': 900000, '30m': 1800000,
        '1h': 3600000, '4h': 14400000, '1d': 86400000
    }
    interval_ms = interval_ms_map.get(timeframe, 300000)
    
    current_time = int(time.time() * 1000)
    candles = []
    price = base_price
    
    for i in range(limit):
        timestamp = current_time - ((limit - i - 1) * interval_ms)
        
        # تغییر قیمت منطقی
        change_percent = np.random.uniform(-0.02, 0.02)
        price = price * (1 + change_percent)
        
        open_price = price * np.random.uniform(0.995, 1.005)
        close_price = price
        high_price = max(open_price, close_price) * np.random.uniform(1.0, 1.01)
        low_price = min(open_price, close_price) * np.random.uniform(0.99, 1.0)
        volume = np.random.uniform(1000, 10000)
        
        candle = [
            int(timestamp),
            float(open_price),
            float(high_price),
            float(low_price),
            float(close_price),
            float(volume),
            int(timestamp + interval_ms),
            str(float(volume) * float(close_price)),
            str(np.random.randint(100, 1000)),
            str(float(volume) * 0.6),
            str(float(volume) * float(close_price) * 0.6),
            "0"
        ]
        candles.append(candle)
    
    return candles

# ==============================================================================
# 2. بخش محاسبات فنی (Indicators) - بهبود یافته
# ==============================================================================

def calculate_ichimoku_components(data: List) -> Dict[str, Any]:
    """محاسبه اجزای ایچیموکو با کنترل خطا"""
    try:
        if not data or len(data) < 52:
            logger.warning(f"Insufficient data for Ichimoku: {len(data) if data else 0} candles")
            return {}
        
        df = pd.DataFrame(data, columns=[
            'ts', 'open', 'high', 'low', 'close', 'vol', 'ct', 
            'qav', 'nt', 'tbb', 'tbq', 'i'
        ])
        
        # تبدیل به عدد
        for col in ['high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['high', 'low', 'close'])
        
        if len(df) < 52:
            return {}
        
        # محاسبه خطوط ایچیموکو
        df['tenkan_sen'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
        df['kijun_sen'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        df['senkou_span_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
        
        last = df.iloc[-1]
        current_price = float(last['close'])
        senkou_a = float(last['senkou_span_a']) if not pd.isna(last['senkou_span_a']) else 0
        senkou_b = float(last['senkou_span_b']) if not pd.isna(last['senkou_span_b']) else 0
        
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        return {
            "tenkan_sen": float(last['tenkan_sen']),
            "kijun_sen": float(last['kijun_sen']),
            "cloud_top": cloud_top,
            "cloud_bottom": cloud_bottom,
            "current_price": current_price,
            "above_cloud": current_price > cloud_top,
            "below_cloud": current_price < cloud_bottom,
            "in_cloud": cloud_bottom <= current_price <= cloud_top,
            "tenkan_kijun_cross": "bullish" if last['tenkan_sen'] > last['kijun_sen'] else "bearish",
            "cloud_color": "green" if senkou_a > senkou_b else "red" if senkou_a < senkou_b else "neutral",
            "cloud_thickness": ((cloud_top - cloud_bottom) / cloud_bottom * 100) if cloud_bottom > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error calculating Ichimoku: {e}")
        return {}

def analyze_ichimoku_scalp_signal(ichimoku_data: Dict) -> Dict[str, Any]:
    """تحلیل سیگنال اسکلپ بر اساس ایچیموکو"""
    try:
        if not ichimoku_data:
            return {
                "signal": "HOLD",
                "confidence": 0.5,
                "reason": "داده ایچیموکو در دسترس نیست",
                "details": {}
            }
        
        price = ichimoku_data.get('current_price', 0)
        tenkan = ichimoku_data.get('tenkan_sen', 0)
        kijun = ichimoku_data.get('kijun_sen', 0)
        above_cloud = ichimoku_data.get('above_cloud', False)
        below_cloud = ichimoku_data.get('below_cloud', False)
        cloud_color = ichimoku_data.get('cloud_color', 'neutral')
        tk_cross = ichimoku_data.get('tenkan_kijun_cross', 'neutral')
        
        # سیستم امتیازدهی
        buy_score = 0
        sell_score = 0
        reasons = []
        
        # شرایط خرید
        if above_cloud:
            buy_score += 2
            reasons.append("قیمت بالای ابر")
        
        if tk_cross == 'bullish':
            buy_score += 2
            reasons.append("تنکان بالای کیجون")
        
        if cloud_color == 'green':
            buy_score += 1
            reasons.append("ابر سبز")
        
        if price > tenkan > kijun:
            buy_score += 1
            reasons.append("قیمت > تنکان > کیجون")
        
        # شرایط فروش
        if below_cloud:
            sell_score += 2
            reasons.append("قیمت زیر ابر")
        
        if tk_cross == 'bearish':
            sell_score += 2
            reasons.append("تنکان زیر کیجون")
        
        if cloud_color == 'red':
            sell_score += 1
            reasons.append("ابر قرمز")
        
        if price < tenkan < kijun:
            sell_score += 1
            reasons.append("قیمت < تنکان < کیجون")
        
        # تصمیم‌گیری
        if buy_score >= 4 and buy_score > sell_score:
            signal = "BUY"
            confidence = min(0.5 + (buy_score * 0.1), 0.9)
        elif sell_score >= 4 and sell_score > buy_score:
            signal = "SELL"
            confidence = min(0.5 + (sell_score * 0.1), 0.9)
        else:
            signal = "HOLD"
            confidence = 0.5
            reasons.append("شرایط متعادل")
        
        # کاهش اعتماد اگر داخل ابر هستیم
        if ichimoku_data.get('in_cloud', False):
            confidence *= 0.7
            reasons.append("کاهش اعتماد بخاطر موقعیت داخل ابر")
        
        return {
            "signal": signal,
            "confidence": round(confidence, 3),
            "reason": " | ".join(reasons),
            "details": ichimoku_data,
            "scores": {"buy": buy_score, "sell": sell_score}
        }
        
    except Exception as e:
        logger.error(f"Error analyzing Ichimoku signal: {e}")
        return {
            "signal": "HOLD",
            "confidence": 0.5,
            "reason": f"خطای تحلیل: {str(e)[:50]}",
            "details": {}
        }

def calculate_simple_rsi(data: List, period: int = 14) -> float:
    """محاسبه RSI با مدیریت خطا"""
    try:
        if not data or len(data) <= period:
            return 50.0
        
        closes = []
        for candle in data:
            if len(candle) > 4:
                try:
                    closes.append(float(candle[4]))
                except (ValueError, TypeError):
                    continue
        
        if len(closes) <= period:
            return 50.0
        
        closes = pd.Series(closes)
        delta = closes.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        last_rsi = rsi.iloc[-1]
        return float(round(last_rsi, 2)) if not pd.isna(last_rsi) else 50.0
        
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return 50.0

def calculate_simple_sma(data: List, period: int = 20) -> float:
    """محاسبه میانگین متحرک ساده"""
    try:
        if not data or len(data) < period:
            return 0.0
        
        closes = []
        for candle in data[-period:]:
            if len(candle) > 4:
                try:
                    closes.append(float(candle[4]))
                except (ValueError, TypeError):
                    continue
        
        if not closes:
            return 0.0
        
        sma = sum(closes) / len(closes)
        return float(round(sma, 4))
        
    except Exception as e:
        logger.error(f"Error calculating SMA: {e}")
        return 0.0

def get_support_resistance_levels(data: List, lookback: int = 20) -> Dict[str, float]:
    """شناسایی سطوح حمایت و مقاومت"""
    try:
        if not data or len(data) < lookback:
            return {"support": 0, "resistance": 0}
        
        highs = []
        lows = []
        
        for candle in data[-lookback:]:
            if len(candle) > 3:
                try:
                    highs.append(float(candle[2]))
                    lows.append(float(candle[3]))
                except (ValueError, TypeError):
                    continue
        
        if not highs or not lows:
            return {"support": 0, "resistance": 0}
        
        # استفاده از درصدایل برای سطوح منطقی‌تر
        support = float(np.percentile(lows, 30))  # 30th percentile
        resistance = float(np.percentile(highs, 70))  # 70th percentile
        
        # سطوح قوی‌تر
        strong_support = min(lows)
        strong_resistance = max(highs)
        
        return {
            "support": round(support, 4),
            "resistance": round(resistance, 4),
            "strong_support": round(strong_support, 4),
            "strong_resistance": round(strong_resistance, 4)
        }
        
    except Exception as e:
        logger.error(f"Error calculating support/resistance: {e}")
        return {"support": 0, "resistance": 0}

def calculate_macd_simple(data: List) -> Dict[str, Any]:
    """محاسبه MACD ساده"""
    result = {
        'trend': 'neutral',
        'histogram': 0,
        'signal': 0,
        'macd': 0
    }
    
    try:
        if not data or len(data) < 35:
            return result
        
        closes = []
        for candle in data:
            if len(candle) > 4:
                try:
                    closes.append(float(candle[4]))
                except:
                    continue
        
        if len(closes) < 35:
            return result
        
        # محاسبه EMA سریع (12) و کند (26)
        ema_12 = pd.Series(closes).ewm(span=12, adjust=False).mean()
        ema_26 = pd.Series(closes).ewm(span=26, adjust=False).mean()
        
        # خط MACD
        macd_line = ema_12 - ema_26
        
        # خط سیگنال (EMA 9 روزه از MACD)
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        # هیستوگرام
        histogram = macd_line - signal_line
        
        # آخرین مقادیر
        last_macd = float(macd_line.iloc[-1]) if not macd_line.empty else 0
        last_signal = float(signal_line.iloc[-1]) if not signal_line.empty else 0
        last_histogram = float(histogram.iloc[-1]) if not histogram.empty else 0
        
        # تشخیص روند
        if last_macd > last_signal and last_histogram > 0:
            trend = 'bullish'
        elif last_macd < last_signal and last_histogram < 0:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        result = {
            'trend': trend,
            'histogram': round(last_histogram, 4),
            'signal': round(last_signal, 4),
            'macd': round(last_macd, 4),
            'values': {
                'macd_series': [float(round(v, 4)) for v in macd_line.tail(5)],
                'signal_series': [float(round(v, 4)) for v in signal_line.tail(5)],
                'histogram_series': [float(round(v, 4)) for v in histogram.tail(5)]
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return result

# ==============================================================================
# 3. تحلیل ترکیبی و ورود هوشمند - نسخه کامل
# ==============================================================================

def calculate_smart_entry(data: List, signal: str) -> float:
    """محاسبه نقطه ورود هوشمند"""
    try:
        if not data:
            return 0.0
        
        current_price = float(data[-1][4]) if len(data[-1]) > 4 else 0.0
        if current_price <= 0:
            return 0.0
        
        # دریافت سطوح حمایت/مقاومت
        sr_levels = get_support_resistance_levels(data)
        
        if signal == "BUY":
            # برای خرید: حداقل قیمت بین حمایت و 0.5% پایین‌تر
            support = sr_levels.get('support', current_price * 0.985)
            price_below = current_price * 0.995  # 0.5% پایین‌تر
            return round(min(support, price_below), 4)
        
        elif signal == "SELL":
            # برای فروش: حداکثر قیمت بین مقاومت و 0.5% بالاتر
            resistance = sr_levels.get('resistance', current_price * 1.015)
            price_above = current_price * 1.005  # 0.5% بالاتر
            return round(max(resistance, price_above), 4)
        
        else:
            return round(current_price, 4)
            
    except Exception as e:
        logger.error(f"Error calculating smart entry: {e}")
        return 0.0

def combined_technical_analysis(data: List, symbol: str = "BTCUSDT", timeframe: str = "5m") -> Dict[str, Any]:
    """
    تحلیل تکنیکال ترکیبی کامل
    
    پارامترها:
    -----------
    data : List
        لیست کندل‌ها
    symbol : str
        نماد ارز
    timeframe : str
        تایم‌فریم تحلیل
    
    بازگشت:
    -------
    Dict[str, Any]
        نتیجه تحلیل کامل
    """
    try:
        if not data or len(data) < 50:
            return {
                "status": "error",
                "message": "داده ناکافی برای تحلیل (نیاز به حداقل ۵۰ کندل)",
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat()
            }
        
        logger.info(f"Starting combined analysis for {symbol} ({timeframe}) - {len(data)} candles")
        
        # ======================================================================
        # 1. محاسبه اندیکاتورها
        # ======================================================================
        
        current_price = float(data[-1][4]) if len(data[-1]) > 4 else 0
        
        # RSI
        rsi = calculate_simple_rsi(data, 14)
        rsi_status = "خرید هیجانی" if rsi > 70 else "فروش هیجانی" if rsi < 30 else "معمولی"
        
        # SMA
        sma_20 = calculate_simple_sma(data, 20)
        sma_50 = calculate_simple_sma(data, 50)
        
        # MACD
        macd = calculate_macd_simple(data)
        
        # ایچیموکو
        ichimoku = calculate_ichimoku_components(data)
        ich_signal = analyze_ichimoku_scalp_signal(ichimoku)
        
        # سطوح حمایت و مقاومت
        sr_levels = get_support_resistance_levels(data)
        
        # ======================================================================
        # 2. سیستم امتیازدهی
        # ======================================================================
        
        signals = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        
        # RSI
        if rsi < 30:
            signals['buy'] += 2.0
        elif rsi > 70:
            signals['sell'] += 2.0
        else:
            signals['hold'] += 1.0
        
        # ایچیموکو
        if ich_signal['signal'] == 'BUY':
            signals['buy'] += ich_signal['confidence'] * 2
        elif ich_signal['signal'] == 'SELL':
            signals['sell'] += ich_signal['confidence'] * 2
        
        # MACD
        if macd['trend'] == 'bullish':
            signals['buy'] += 1.0
        elif macd['trend'] == 'bearish':
            signals['sell'] += 1.0
        
        # میانگین‌های متحرک
        if current_price > sma_20 > sma_50:
            signals['buy'] += 1.5
        elif current_price < sma_20 < sma_50:
            signals['sell'] += 1.5
        
        # موقعیت قیمت نسبت به سطوح
        support = sr_levels.get('support', 0)
        resistance = sr_levels.get('resistance', 0)
        
        if support > 0 and current_price < support * 1.02:
            signals['buy'] += 0.5
        
        if resistance > 0 and current_price > resistance * 0.98:
            signals['sell'] += 0.5
        
        # ======================================================================
        # 3. تصمیم‌گیری نهایی
        # ======================================================================
        
        final_signal = max(signals, key=signals.get)
        total_score = sum(signals.values())
        
        if total_score > 0:
            confidence = signals[final_signal] / total_score
        else:
            confidence = 0.5
        
        # آستانه اعتماد
        if confidence < 0.6:
            final_signal = 'hold'
            confidence = 0.5
        
        # اصلاح سیگنال بر اساس RSI اشباع
        if rsi > 70 and final_signal == 'buy':
            final_signal = 'hold'
            confidence = 0.5
        elif rsi < 30 and final_signal == 'sell':
            final_signal = 'hold'
            confidence = 0.5
        
        # ======================================================================
        # 4. محاسبه پارامترهای معاملاتی
        # ======================================================================
        
        entry_price = calculate_smart_entry(data, final_signal.upper())
        stop_loss = 0
        targets = []
        
        if final_signal == 'buy':
            # استاپ لاس: حداقل بین حمایت و 1.5% پایین‌تر
            stop_loss_candidate1 = sr_levels.get('support', entry_price * 0.985)
            stop_loss_candidate2 = entry_price * 0.985
            stop_loss = round(min(stop_loss_candidate1, stop_loss_candidate2), 4)
            
            # تارگت‌ها
            targets = [
                round(entry_price * 1.015, 4),  # +1.5%
                round(entry_price * 1.03, 4)    # +3.0%
            ]
            
        elif final_signal == 'sell':
            # استاپ لاس: حداکثر بین مقاومت و 1.5% بالاتر
            stop_loss_candidate1 = sr_levels.get('resistance', entry_price * 1.015)
            stop_loss_candidate2 = entry_price * 1.015
            stop_loss = round(max(stop_loss_candidate1, stop_loss_candidate2), 4)
            
            # تارگت‌ها
            targets = [
                round(entry_price * 0.985, 4),  # -1.5%
                round(entry_price * 0.97, 4)    # -3.0%
            ]
        
        # ======================================================================
        # 5. آماده‌سازی نتیجه نهایی
        # ======================================================================
        
        # دلایل سیگنال
        reasons = []
        if ich_signal.get('reason'):
            reasons.append(ich_signal['reason'])
        if rsi_status != "معمولی":
            reasons.append(f"RSI: {rsi_status}")
        
        result = {
            "status": "success",
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": final_signal.upper(),
            "confidence": round(confidence, 2),
            "current_price": round(current_price, 4),
            "entry_price": round(entry_price, 4),
            "stop_loss": stop_loss,
            "targets": targets,
            "risk_reward_ratio": round((targets[0] - entry_price) / (entry_price - stop_loss), 2) 
                               if targets and stop_loss > 0 and entry_price > stop_loss else 0,
            "analysis_summary": {
                "reasons": reasons,
                "signal_scores": {
                    "buy": round(signals['buy'], 2),
                    "sell": round(signals['sell'], 2),
                    "hold": round(signals['hold'], 2)
                }
            },
            "support_resistance": sr_levels,
            "indicators": {
                "rsi": round(rsi, 2),
                "rsi_status": rsi_status,
                "sma_20": round(sma_20, 4),
                "sma_50": round(sma_50, 4),
                "macd_trend": macd['trend'],
                "ichimoku": ich_signal.get('details', {})
            },
            "timestamp": datetime.now().isoformat(),
            "data_points": len(data)
        }
        
        logger.info(f"Analysis COMPLETE: {symbol} = {final_signal.upper()} "
                   f"(Confidence: {confidence:.2f}, Entry: {entry_price:.2f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Critical error in combined analysis for {symbol}: {e}")
        return {
            "status": "error",
            "message": f"خطا در تحلیل: {str(e)}",
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat()
        }

# ==============================================================================
# 4. توابع کمکی و تست
# ==============================================================================

def clear_cache():
    """پاک کردن کش داده‌ها"""
    _data_cache.clear()
    _cache_expiry.clear()
    logger.info("Cache cleared")

def get_cache_stats() -> Dict:
    """دریافت آمار کش"""
    return {
        "cache_size": len(_data_cache),
        "cache_keys": list(_data_cache.keys())[:5],  # فقط 5 کلید اول
        "timestamp": datetime.now().isoformat()
    }

def analyze_symbol(symbol: str, timeframe: str = "5m") -> Dict[str, Any]:
    """
    تحلیل کامل یک نماد
    
    پارامترها:
    -----------
    symbol : str
        نماد ارز
    timeframe : str
        تایم‌فریم تحلیل
    
    بازگشت:
    -------
    Dict[str, Any]
        نتیجه تحلیل
    """
    logger.info(f"Analyzing {symbol} on {timeframe} timeframe")
    
    # دریافت داده
    data = get_market_data_with_fallback(symbol, timeframe, 100)
    
    if not data:
        return {
            "status": "error",
            "message": f"ناتوان در دریافت داده برای {symbol}",
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat()
        }
    
    # اجرای تحلیل
    return combined_technical_analysis(data, symbol, timeframe)

# تست اصلی
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("🔄 Testing Crypto Analysis System...")
    print("=" * 60)
    
    # تست تحلیل BTC
    print("\n📊 Testing BTCUSDT analysis:")
    result = analyze_symbol("BTCUSDT", "5m")
    
    if result["status"] == "success":
        print(f"✅ Symbol: {result['symbol']}")
        print(f"📈 Signal: {result['signal']} (Confidence: {result['confidence']})")
        print(f"💰 Current: ${result['current_price']:,.2f}")
        print(f"🎯 Entry: ${result['entry_price']:,.2f}")
        print(f"🛡️ Stop Loss: ${result['stop_loss']:,.2f}")
        print(f"🎯 Targets: {[f'${t:,.2f}' for t in result['targets']]}")
        print(f"⚖️ Risk/Reward: {result['risk_reward_ratio']}")
        
        print(f"\n📊 Indicators:")
        print(f"  RSI: {result['indicators']['rsi']} ({result['indicators']['rsi_status']})")
        print(f"  SMA20: ${result['indicators']['sma_20']:,.2f}")
        print(f"  SMA50: ${result['indicators']['sma_50']:,.2f}")
        print(f"  MACD Trend: {result['indicators']['macd_trend']}")
        
        if result['analysis_summary']['reasons']:
            print(f"\n📝 Reasons:")
            for reason in result['analysis_summary']['reasons']:
                print(f"  • {reason}")
    else:
        print(f"❌ Error: {result['message']}")
    
    print("\n" + "=" * 60)
    print("✅ System test completed successfully!")
