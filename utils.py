import pandas as pd
import numpy as np
import yfinance as yf
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ==============================================================================
# بخش ویژه: تحلیل ایچیموکو و Smart Entry (هماهنگ‌سازی با main.py)
# ==============================================================================

defget_market_data_with_fallback(symbol: str, interval: str = "5m", limit: int = 150, return_source: bool = False):
    """نسخه نهایی با خروجی عددی (Float) برای عبور از فیلتر v8.0-PRO"""
    try:
        # ۱. تنظیم تایم‌فریم
        tf_map = {'1m':'1m', '5m':'5m', '15m':'15m', '30m':'30m', '1h':'60m', '4h':'240m', '1d':'1d'}
        interval = tf_map.get(interval, interval)
        
        # ۲. تنظیم نماد
        symbol = symbol.upper().replace("/", "")
        yf_symbol = symbol.replace('USDT', '-USD') if 'USDT' in symbol else f"{symbol}-USD"
        
        # ۳. دریافت داده از یاهو
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period="5d", interval=interval)
        
        if df.empty:
            return [] if not return_source else {"success": False, "data": []}
            
        # ۴. ساخت لیست کندل‌ها با اعداد واقعی (Float) - فرمت ۶ ستونه خالص
        candles = []
        for idx, row in df.tail(limit).iterrows():
            candles.append([
                int(idx.timestamp() * 1000),    # 0: Timestamp
                float(row['Open']),             # 1: Open
                float(row['High']),             # 2: High
                float(row['Low']),              # 3: Low
                float(row['Close']),            # 4: Close
                float(row['Volume'])            # 5: Volume
            ])
            
        logger.info(f"✅ {len(candles)} candles ready for {symbol}")

        # ۵. مهم‌ترین بخش: شبیه‌سازی خروجی برای تست main
        if return_source:
            return {
                "success": True,
                "data": candles,
                "source": "yahoo_finance",
                "last_price": candles[-1][4]
            }
            
        return candles

    except Exception as e:
        logger.error(f"Error: {e}")
        return [] if not return_source else {"success": False, "data": []}
        
        # تبدیل به فرمت کندل استاندارد
        candles = []
        for idx, row in df.iterrows():
            timestamp = int(idx.timestamp() * 1000)
            candle = [
                timestamp,
                float(row['Open']),
                float(row['High']),
                float(row['Low']),
                float(row['Close']),
                float(row['Volume']),
                timestamp + 300000,  # close_time (5 دقیقه بعد)
                str(float(row['Volume']) * float(row['Close'])),
                "0", "0", "0", "0"
            ]
            candles.append(candle)
        
        logger.info(f"Fetched {len(candles)} candles for {symbol} from Yahoo Finance")
        
        if return_source:
            return {
                "data": candles,
                "source": "yahoo_finance",
                "success": True,
                "candle_count": len(candles),
                "current_price": candles[-1][4] if candles else 0
            }
        
        return candles
        
    except Exception as e:
        logger.error(f"Error in get_market_data_with_fallback for {symbol}: {e}")
        return [] if not return_source else {"data": [], "source": "error", "success": False}

def calculate_ichimoku_components(data: List) -> Dict[str, Any]:
    """محاسبه کامل اجزای ایچیموکو - نسخه بهبود یافته"""
    try:
        if not data or len(data) < 52:
            logger.warning(f"Insufficient data for Ichimoku: {len(data)} candles")
            return {}
        
        # استخراج داده‌ها
        highs = []
        lows = []
        closes = []
        
        for candle in data:
            if len(candle) > 4:
                try:
                    highs.append(float(candle[2]))
                    lows.append(float(candle[3]))
                    closes.append(float(candle[4]))
                except (ValueError, TypeError):
                    continue
        
        if len(highs) < 52 or len(lows) < 52 or len(closes) < 52:
            return {}
        
        # تبدیل به Series برای محاسبات رولینگ
        highs_series = pd.Series(highs)
        lows_series = pd.Series(lows)
        closes_series = pd.Series(closes)
        
        # 1. Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        nine_period_high = highs_series.rolling(window=9).max()
        nine_period_low = lows_series.rolling(window=9).min()
        tenkan_sen = (nine_period_high + nine_period_low) / 2
        
        # 2. Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        twentysix_period_high = highs_series.rolling(window=26).max()
        twentysix_period_low = lows_series.rolling(window=26).min()
        kijun_sen = (twentysix_period_high + twentysix_period_low) / 2
        
        # 3. Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 (shifted 26 periods forward)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # 4. Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 (shifted 26 periods forward)
        fiftytwo_period_high = highs_series.rolling(window=52).max()
        fiftytwo_period_low = lows_series.rolling(window=52).min()
        senkou_span_b = ((fiftytwo_period_high + fiftytwo_period_low) / 2).shift(26)
        
        # 5. Chikou Span (Lagging Span): Close price shifted 26 periods backward
        chikou_span = closes_series.shift(-26)
        
        # مقادیر آخرین دوره
        last_idx = len(tenkan_sen) - 1
        current_price = closes[-1]
        tenkan_val = float(tenkan_sen.iloc[last_idx]) if not pd.isna(tenkan_sen.iloc[last_idx]) else 0
        kijun_val = float(kijun_sen.iloc[last_idx]) if not pd.isna(kijun_sen.iloc[last_idx]) else 0
        senkou_a_val = float(senkou_span_a.iloc[last_idx]) if not pd.isna(senkou_span_a.iloc[last_idx]) else 0
        senkou_b_val = float(senkou_span_b.iloc[last_idx]) if not pd.isna(senkou_span_b.iloc[last_idx]) else 0
        chikou_val = float(chikou_span.iloc[last_idx]) if not pd.isna(chikou_span.iloc[last_idx]) else 0
        
        # محاسبه ابر کومو
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # وضعیت قیمت نسبت به ابر
        above_cloud = current_price > cloud_top
        below_cloud = current_price < cloud_bottom
        in_cloud = cloud_bottom <= current_price <= cloud_top
        
        # کراس تنکان و کیجون
        tenkan_kijun_cross = "bullish" if tenkan_val > kijun_val else "bearish"
        
        # رنگ ابر (صعودی/نزولی)
        cloud_color = "green" if senkou_a_val > senkou_b_val else "red" if senkou_a_val < senkou_b_val else "neutral"
        
        # ضخامت ابر
        cloud_thickness = ((cloud_top - cloud_bottom) / cloud_bottom * 100) if cloud_bottom > 0 else 0
        
        return {
            "tenkan_sen": round(tenkan_val, 4),
            "kijun_sen": round(kijun_val, 4),
            "senkou_span_a": round(senkou_a_val, 4),
            "senkou_span_b": round(senkou_b_val, 4),
            "chikou_span": round(chikou_val, 4),
            "cloud_top": round(cloud_top, 4),
            "cloud_bottom": round(cloud_bottom, 4),
            "current_price": round(current_price, 4),
            "above_cloud": above_cloud,
            "below_cloud": below_cloud,
            "in_cloud": in_cloud,
            "tenkan_kijun_cross": tenkan_kijun_cross,
            "cloud_color": cloud_color,
            "cloud_thickness": round(cloud_thickness, 2),
            "cloud_distance": round(((current_price - cloud_bottom) / cloud_bottom * 100), 2) if cloud_bottom > 0 else 0,
            "tenkan_distance": round(((tenkan_val - kijun_val) / kijun_val * 100), 2) if kijun_val > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error in calculate_ichimoku_components: {e}")
        return {}

def analyze_ichimoku_scalp_signal(data: List) -> Dict[str, Any]:
    """تشخیص سیگنال اسکلپ بر اساس ایچیموکو - نسخه کامل"""
    try:
        if not data or len(data) < 26:
            return {
                "signal": "NEUTRAL",
                "confidence": 0.5,
                "reason": "Insufficient data",
                "details": {}
            }
        
        # محاسبه ایچیموکو
        ichimoku = calculate_ichimoku_components(data)
        if not ichimoku:
            return {
                "signal": "NEUTRAL",
                "confidence": 0.5,
                "reason": "Ichimoku calculation failed",
                "details": {}
            }
        
        current_price = ichimoku.get('current_price', 0)
        tenkan_sen = ichimoku.get('tenkan_sen', 0)
        kijun_sen = ichimoku.get('kijun_sen', 0)
        above_cloud = ichimoku.get('above_cloud', False)
        below_cloud = ichimoku.get('below_cloud', False)
        tenkan_kijun_cross = ichimoku.get('tenkan_kijun_cross', 'neutral')
        cloud_color = ichimoku.get('cloud_color', 'neutral')
        
        # سیستم امتیازدهی
        bullish_score = 0
        bearish_score = 0
        reasons = []
        
        # شرایط صعودی
        if tenkan_kijun_cross == 'bullish':
            bullish_score += 2
            reasons.append("Tenkan above Kijun (Bullish cross)")
        
        if above_cloud:
            bullish_score += 2
            reasons.append("Price above cloud")
        
        if cloud_color == 'green':
            bullish_score += 1
            reasons.append("Cloud is green (bullish)")
        
        if current_price > tenkan_sen > kijun_sen:
            bullish_score += 1
            reasons.append("Price > Tenkan > Kijun")
        
        # شرایط نزولی
        if tenkan_kijun_cross == 'bearish':
            bearish_score += 2
            reasons.append("Tenkan below Kijun (Bearish cross)")
        
        if below_cloud:
            bearish_score += 2
            reasons.append("Price below cloud")
        
        if cloud_color == 'red':
            bearish_score += 1
            reasons.append("Cloud is red (bearish)")
        
        if current_price < tenkan_sen < kijun_sen:
            bearish_score += 1
            reasons.append("Price < Tenkan < Kijun")
        
        # تصمیم‌گیری
        if bullish_score >= 3 and bullish_score > bearish_score:
            signal = "BULLISH"
            confidence = min(0.5 + (bullish_score * 0.1), 0.9)
        elif bearish_score >= 3 and bearish_score > bullish_score:
            signal = "BEARISH"
            confidence = min(0.5 + (bearish_score * 0.1), 0.9)
        else:
            signal = "NEUTRAL"
            confidence = 0.5
            reasons.append("Market in equilibrium")
        
        # کاهش اعتماد اگر داخل ابر
        if ichimoku.get('in_cloud', False):
            confidence *= 0.7
            reasons.append("Reduced confidence (price in cloud)")
        
        return {
            "signal": signal,
            "confidence": round(confidence, 3),
            "reason": " | ".join(reasons),
            "details": ichimoku,
            "scores": {"bullish": bullish_score, "bearish": bearish_score}
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_ichimoku_scalp_signal: {e}")
        return {
            "signal": "NEUTRAL",
            "confidence": 0.5,
            "reason": f"Analysis error: {str(e)[:50]}",
            "details": {}
        }

def calculate_simple_rsi(data: List, period: int = 14) -> float:
    """محاسبه RSI ساده - هماهنگ با main.py"""
    try:
        if not data or len(data) < period + 1:
            return 50.0
        
        closes = []
        for candle in data:
            if len(candle) > 4:
                try:
                    closes.append(float(candle[4]))
                except (ValueError, TypeError):
                    continue
        
        if len(closes) < period + 1:
            return 50.0
        
        # محاسبه RSI
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # ادامه محاسبات
        for i in range(period, len(gains)):
            avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
            avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
        
        return float(round(rsi, 2))
        
    except Exception as e:
        logger.error(f"Error in calculate_simple_rsi: {e}")
        return 50.0

def calculate_simple_sma(data: List, period: int = 50) -> float:
    """محاسبه SMA ساده - هماهنگ با main.py"""
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
        logger.error(f"Error in calculate_simple_sma: {e}")
        return 0.0

def calculate_smart_entry(symbol: str, signal_type: str, current_price: float, 
                         data: Optional[List] = None) -> Dict[str, float]:
    """محاسبه دقیق نقطه ورود، تارگت و حد ضرر بر اساس الگوی QM"""
    try:
        if current_price <= 0:
            return {"entry": 0, "tp": 0, "sl": 0}
        
        # تنظیمات نماد
        symbol_settings = {
            "BTCUSDT": {"volatility": 0.02, "risk_reward": 1.5},
            "ETHUSDT": {"volatility": 0.025, "risk_reward": 1.5},
            "BNBUSDT": {"volatility": 0.03, "risk_reward": 1.5},
            "SOLUSDT": {"volatility": 0.035, "risk_reward": 1.5},
            "DEFAULT": {"volatility": 0.02, "risk_reward": 1.5}
        }
        
        settings = symbol_settings.get(symbol.upper(), symbol_settings["DEFAULT"])
        volatility = settings["volatility"]
        risk_reward = settings["risk_reward"]
        
        if signal_type.upper() == "BUY":
            # برای سیگنال خرید
            if data and len(data) >= 20:
                # محاسبه حمایت از داده‌ها
                lows = []
                for candle in data[-20:]:
                    if len(candle) > 3:
                        try:
                            lows.append(float(candle[3]))
                        except:
                            continue
                
                if lows:
                    support_level = min(lows)
                    # ورود: کمی بالاتر از حمایت
                    entry = max(current_price * 0.998, support_level * 1.001)
                else:
                    entry = current_price * 0.998
            else:
                entry = current_price * 0.998
            
            # حد ضرر: بر اساس نوسان
            sl = entry * (1 - volatility)
            
            # تارگت اول: نسبت ریسک/پاداش
            tp1 = entry + (entry - sl) * risk_reward
            
            # تارگت دوم: ضریب بیشتر
            tp2 = entry * (1 + (volatility * 2))
            
            return {
                "entry": round(entry, 4),
                "tp1": round(tp1, 4),
                "tp2": round(tp2, 4),
                "sl": round(sl, 4),
                "risk_reward": round((tp1 - entry) / (entry - sl), 2)
            }
            
        elif signal_type.upper() == "SELL":
            # برای سیگنال فروش
            if data and len(data) >= 20:
                # محاسبه مقاومت از داده‌ها
                highs = []
                for candle in data[-20:]:
                    if len(candle) > 2:
                        try:
                            highs.append(float(candle[2]))
                        except:
                            continue
                
                if highs:
                    resistance_level = max(highs)
                    # ورود: کمی پایین‌تر از مقاومت
                    entry = min(current_price * 1.002, resistance_level * 0.999)
                else:
                    entry = current_price * 1.002
            else:
                entry = current_price * 1.002
            
            # حد ضرر: بر اساس نوسان
            sl = entry * (1 + volatility)
            
            # تارگت اول: نسبت ریسک/پاداش
            tp1 = entry - (sl - entry) * risk_reward
            
            # تارگت دوم: ضریب بیشتر
            tp2 = entry * (1 - (volatility * 2))
            
            return {
                "entry": round(entry, 4),
                "tp1": round(tp1, 4),
                "tp2": round(tp2, 4),
                "sl": round(sl, 4),
                "risk_reward": round((entry - tp1) / (sl - entry), 2)
            }
        
        else:
            # برای وضعیت خنثی
            return {
                "entry": round(current_price, 4),
                "tp1": round(current_price * 1.01, 4),
                "tp2": round(current_price * 1.02, 4),
                "sl": round(current_price * 0.99, 4),
                "risk_reward": 1.0
            }
            
    except Exception as e:
        logger.error(f"Error in calculate_smart_entry for {symbol}: {e}")
        return {
            "entry": round(current_price, 4),
            "tp": round(current_price * 1.02, 4) if signal_type == "BUY" else round(current_price * 0.98, 4),
            "sl": round(current_price * 0.985, 4) if signal_type == "BUY" else round(current_price * 1.015, 4),
            "risk_reward": 1.5
        }

# ==============================================================================
# توابع کمکی اضافی برای سازگاری کامل
# ==============================================================================

def calculate_rsi(closes: List[float], period: int = 14) -> float:
    """محاسبه RSI از لیست قیمت‌های بسته‌شدن"""
    return calculate_simple_rsi([ [0,0,0,0,c,0] for c in closes ], period)

def calculate_sma(closes: List[float], period: int = 50) -> float:
    """محاسبه SMA از لیست قیمت‌های بسته‌شدن"""
    return calculate_simple_sma([ [0,0,0,0,c,0] for c in closes ], period)

def get_binance_klines_enhanced(symbol: str, interval: str = "5m", limit: int = 100):
    """تابع سازگاری با نام قدیمی"""
    return get_market_data_with_fallback(symbol, interval, limit)

def get_support_resistance_levels_qm(data: List, lookback: int = 20) -> Dict[str, float]:
    """شناسایی سطوح حمایت و مقاومت برای الگوی QM"""
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
        
        # استفاده از روش پیوت برای QM
        pivot = (max(highs) + min(lows) + data[-1][4]) / 3
        r1 = (2 * pivot) - min(lows)
        s1 = (2 * pivot) - max(highs)
        r2 = pivot + (max(highs) - min(lows))
        s2 = pivot - (max(highs) - min(lows))
        
        return {
            "pivot": round(pivot, 4),
            "support1": round(s1, 4),
            "support2": round(s2, 4),
            "resistance1": round(r1, 4),
            "resistance2": round(r2, 4),
            "high": round(max(highs), 4),
            "low": round(min(lows), 4)
        }
        
    except Exception as e:
        logger.error(f"Error in get_support_resistance_levels_qm: {e}")
        return {"support": 0, "resistance": 0}

# ==============================================================================
# رفع خطای Missing functions در main.py
# ==============================================================================

def get_market_data_simple(symbol: str, interval: str = "5m", limit: int = 100):
    """
    این تابع دقیقاً همان کار get_market_data_with_fallback را انجام می‌دهد
    تا خطای Attribute Error در main.py برطرف شود.
    """
    return get_market_data_with_fallback(symbol, interval, limit)
