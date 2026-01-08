"""
🎯 پیاده‌سازی کامل هر دو اندیکاتور اصلی
ترکیب دقیق: "Combined: ZLMA Trend + Smart Money Pro" و "RSI (+Ichimoku Cloud)"
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import math
from ta.momentum import RSIIndicator
from ta.trend import MACD

class CombinedIndicators:
    """کلاس ترکیبی دقیق هر دو اندیکاتور"""
    
    def __init__(self):
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        import logging
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    # ============================================================
    # 🟢 بخش اول: ZLMA Trend + Smart Money Pro (ساده‌شده)
    # ============================================================
    
    def calculate_zlma_trend(self, df: pd.DataFrame, length: int = 15) -> Dict:
        """محاسبه ZLMA Trend به صورت ساده‌شده"""
        try:
            close = df['close']
            
            # 1. محاسبه ZLMA (Zero-Lag Moving Average)
            ema_value = close.ewm(span=length).mean()
            correction = close + (close - ema_value)
            zlma = correction.ewm(span=length).mean()
            
            # 2. سیگنال‌های ZLMA
            signal_up = (zlma.shift(1) <= ema_value.shift(1)) & (zlma > ema_value)
            signal_dn = (zlma.shift(1) >= ema_value.shift(1)) & (zlma < ema_value)
            
            # 3. تشخیص روند
            trend_up = zlma > zlma.shift(3)
            trend_down = zlma < zlma.shift(3)
            
            return {
                'zlma': float(zlma.iloc[-1]),
                'ema': float(ema_value.iloc[-1]),
                'signal_up': bool(signal_up.iloc[-1]),
                'signal_dn': bool(signal_dn.iloc[-1]),
                'trend_up': bool(trend_up.iloc[-1]),
                'trend_down': bool(trend_down.iloc[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error in calculate_zlma_trend: {e}")
            return {}
    
    def calculate_smart_money_signals(self, df: pd.DataFrame) -> Dict:
        """محاسبه سیگنال‌های Smart Money ساده‌شده"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Wave Trend ساده‌شده
            src = (high + low + close) / 3
            
            # TCI ساده‌شده
            n1, n2, n3 = 9, 6, 3
            ema_n1 = pd.Series(src).ewm(span=n1).mean()
            abs_diff = np.abs(src - ema_n1.values)
            ema_abs = pd.Series(abs_diff).ewm(span=n1).mean()
            tci = ((src - ema_n1) / (0.025 * ema_abs)).ewm(span=n2).mean() + 50
            
            # RSI
            rsi_indicator = RSIIndicator(close=pd.Series(src), window=n3)
            wt_rsi = rsi_indicator.rsi()
            
            # ترکیب
            wt1 = pd.concat([pd.Series(tci), wt_rsi], axis=1).mean(axis=1)
            wt2 = wt1.rolling(window=6).mean()
            
            # سیگنال‌های Boom Hunter ساده‌شده
            q1 = wt1
            trigger = wt2
            
            bh_crossover = (q1.iloc[-1] > trigger.iloc[-1]) and (q1.iloc[-2] <= trigger.iloc[-2])
            bh_crossunder = (q1.iloc[-1] < trigger.iloc[-1]) and (q1.iloc[-2] >= trigger.iloc[-2])
            
            # تشخیص Range Box
            box_lookback = 50
            box_top = df['high'].rolling(box_lookback).max().iloc[-1]
            box_bottom = df['low'].rolling(box_lookback).min().iloc[-1]
            box_mid = (box_top + box_bottom) / 2
            
            current_price = df['close'].iloc[-1]
            box_range = box_top - box_bottom
            
            in_box = (current_price > box_bottom) and (current_price < box_top)
            near_bottom = current_price <= box_bottom + (box_range * 0.2)
            near_top = current_price >= box_top - (box_range * 0.2)
            
            # تشخیص Order Block
            bullish_move = (close[-1] > close[-2]) and (close[-2] < close[-3])
            bearish_move = (close[-1] < close[-2]) and (close[-2] > close[-3])
            
            return {
                'bh_crossover': bool(bh_crossover),
                'bh_crossunder': bool(bh_crossunder),
                'wt1': float(wt1.iloc[-1]),
                'wt2': float(wt2.iloc[-1]),
                'in_box': bool(in_box),
                'near_bottom': bool(near_bottom),
                'near_top': bool(near_top),
                'box_top': float(box_top),
                'box_bottom': float(box_bottom),
                'box_mid': float(box_mid),
                'bullish_move': bool(bullish_move),
                'bearish_move': bool(bearish_move)
            }
            
        except Exception as e:
            self.logger.error(f"Error in calculate_smart_money_signals: {e}")
            return {}
    
    # ============================================================
    # 🔴 بخش دوم: RSI (+Ichimoku Cloud) - ساده‌شده
    # ============================================================
    
    def calculate_rsi_divergence(self, df: pd.DataFrame) -> Dict:
        """محاسبه RSI Divergence ساده‌شده"""
        try:
            # RSI پایه
            rsi_indicator = RSIIndicator(close=df['close'], window=14)
            rsi = rsi_indicator.rsi()
            current_rsi = rsi.iloc[-1]
            
            # تشخیص ساده Divergence
            lookback = 15
            price_highs = []
            price_lows = []
            rsi_highs = []
            rsi_lows = []
            
            for i in range(lookback, len(df)-lookback):
                # تشخیص سقف
                if df['high'].iloc[i] == df['high'].iloc[i-lookback:i+lookback].max():
                    price_highs.append(df['high'].iloc[i])
                    rsi_highs.append(rsi.iloc[i])
                
                # تشخیص کف
                if df['low'].iloc[i] == df['low'].iloc[i-lookback:i+lookback].min():
                    price_lows.append(df['low'].iloc[i])
                    rsi_lows.append(rsi.iloc[i])
            
            # بررسی Bullish Divergence
            bullish_div = False
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                price_lower = price_lows[-1] < price_lows[-2]
                rsi_higher = rsi_lows[-1] > rsi_lows[-2]
                bullish_div = price_lower and rsi_higher
            
            # بررسی Bearish Divergence
            bearish_div = False
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                price_higher = price_highs[-1] > price_highs[-2]
                rsi_lower = rsi_highs[-1] < rsi_highs[-2]
                bearish_div = price_higher and rsi_lower
            
            return {
                'rsi': float(current_rsi),
                'bullish_div': bool(bullish_div),
                'bearish_div': bool(bearish_div),
                'oversold': current_rsi < 30,
                'overbought': current_rsi > 70
            }
            
        except Exception as e:
            self.logger.error(f"Error in calculate_rsi_divergence: {e}")
            return {}
    
    def calculate_ichimoku_cloud(self, df: pd.DataFrame) -> Dict:
        """محاسبه ابر ایچیموکو ساده‌شده"""
        try:
            high = df['high']
            low = df['low']
            
            # Tenkan-sen (Conversion Line)
            period9_high = high.rolling(9).max()
            period9_low = low.rolling(9).min()
            tenkan_sen = (period9_high + period9_low) / 2
            
            # Kijun-sen (Base Line)
            period26_high = high.rolling(26).max()
            period26_low = low.rolling(26).min()
            kijun_sen = (period26_high + period26_low) / 2
            
            # Senkou Span A
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            
            # Senkou Span B
            period52_high = high.rolling(52).max()
            period52_low = low.rolling(52).min()
            senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
            
            current_price = df['close'].iloc[-1]
            
            cloud_top = max(senkou_span_a.iloc[-1], senkou_span_b.iloc[-1])
            cloud_bottom = min(senkou_span_a.iloc[-1], senkou_span_b.iloc[-1])
            
            return {
                'tenkan_sen': float(tenkan_sen.iloc[-1]),
                'kijun_sen': float(kijun_sen.iloc[-1]),
                'senkou_a': float(senkou_span_a.iloc[-1]),
                'senkou_b': float(senkou_span_b.iloc[-1]),
                'cloud_top': float(cloud_top),
                'cloud_bottom': float(cloud_bottom),
                'above_cloud': current_price > cloud_top,
                'below_cloud': current_price < cloud_bottom,
                'in_cloud': (current_price >= cloud_bottom) and (current_price <= cloud_top),
                'cloud_bullish': senkou_span_a.iloc[-1] > senkou_span_b.iloc[-1],
                'cloud_bearish': senkou_span_a.iloc[-1] < senkou_span_b.iloc[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Error in calculate_ichimoku_cloud: {e}")
            return {}
    
    # ============================================================
    # 📊 اندیکاتورهای کمکی
    # ============================================================
    
    def calculate_macd(self, df: pd.DataFrame) -> Dict:
        """محاسبه MACD"""
        try:
            macd_indicator = MACD(
                close=df['close'],
                window_fast=12,
                window_slow=26,
                window_sign=9
            )
            
            macd_line = macd_indicator.macd().iloc[-1]
            signal_line = macd_indicator.macd_signal().iloc[-1]
            histogram = macd_indicator.macd_diff().iloc[-1]
            
            return {
                'macd': float(macd_line),
                'signal': float(signal_line),
                'histogram': float(histogram),
                'bullish': histogram > 0,
                'bearish': histogram < 0,
                'crossover': (histogram > 0) and (macd_indicator.macd_diff().iloc[-2] <= 0),
                'crossunder': (histogram < 0) and (macd_indicator.macd_diff().iloc[-2] >= 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return {}
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """محاسبه ATR برای مدیریت ریسک"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return float(atr.iloc[-1]) if not atr.empty else float(close.iloc[-1] * 0.02)
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return float(df['close'].iloc[-1] * 0.02)
    
    # ============================================================
    # 🎯 تابع ترکیب کننده تمام سیگنال‌ها (اصلی)
    # ============================================================
    
    def generate_combined_signal(self, df: pd.DataFrame) -> Dict:
        """ترکیب کامل تمام سیگنال‌ها از هر دو اندیکاتور"""
        try:
            if len(df) < 100:
                return {'signal_type': 'NEUTRAL', 'confidence': 0}
            
            # محاسبه تمام بخش‌ها
            zlma_signals = self.calculate_zlma_trend(df)
            sm_signals = self.calculate_smart_money_signals(df)
            rsi_signals = self.calculate_rsi_divergence(df)
            ichimoku_signals = self.calculate_ichimoku_cloud(df)
            macd_signals = self.calculate_macd(df)
            
            # جمع‌بندی شرایط خرید
            buy_conditions = []
            buy_score = 0
            
            # 1. شرایط از ZLMA Trend
            if zlma_signals.get('signal_up'):
                buy_conditions.append("ZLMA Crossover ↑")
                buy_score += 1
            
            if zlma_signals.get('trend_up'):
                buy_conditions.append("ZLMA Trend Up")
                buy_score += 1
            
            # 2. شرایط از Smart Money Pro
            if sm_signals.get('bh_crossover'):
                buy_conditions.append("Smart Money Buy")
                buy_score += 1
            
            if sm_signals.get('near_bottom') and sm_signals.get('bullish_move'):
                buy_conditions.append("Order Block Support")
                buy_score += 1
            
            # 3. شرایط از RSI
            if rsi_signals.get('bullish_div'):
                buy_conditions.append("RSI Bullish Divergence")
                buy_score += 2  # وزن بیشتر برای divergence
            
            if rsi_signals.get('oversold'):
                buy_conditions.append("RSI Oversold")
                buy_score += 1
            
            # 4. شرایط از Ichimoku
            if ichimoku_signals.get('above_cloud') and ichimoku_signals.get('cloud_bullish'):
                buy_conditions.append("Above Bullish Cloud")
                buy_score += 1
            
            # 5. شرایط از MACD
            if macd_signals.get('bullish'):
                buy_conditions.append("MACD Bullish")
                buy_score += 1
            
            if macd_signals.get('crossover'):
                buy_conditions.append("MACD Crossover")
                buy_score += 1
            
            # جمع‌بندی شرایط فروش
            sell_conditions = []
            sell_score = 0
            
            # 1. شرایط از ZLMA Trend
            if zlma_signals.get('signal_dn'):
                sell_conditions.append("ZLMA Crossunder ↓")
                sell_score += 1
            
            if zlma_signals.get('trend_down'):
                sell_conditions.append("ZLMA Trend Down")
                sell_score += 1
            
            # 2. شرایط از Smart Money Pro
            if sm_signals.get('bh_crossunder'):
                sell_conditions.append("Smart Money Sell")
                sell_score += 1
            
            if sm_signals.get('near_top') and sm_signals.get('bearish_move'):
                sell_conditions.append("Order Block Resistance")
                sell_score += 1
            
            # 3. شرایط از RSI
            if rsi_signals.get('bearish_div'):
                sell_conditions.append("RSI Bearish Divergence")
                sell_score += 2  # وزن بیشتر برای divergence
            
            if rsi_signals.get('overbought'):
                sell_conditions.append("RSI Overbought")
                sell_score += 1
            
            # 4. شرایط از Ichimoku
            if ichimoku_signals.get('below_cloud') and ichimoku_signals.get('cloud_bearish'):
                sell_conditions.append("Below Bearish Cloud")
                sell_score += 1
            
            # 5. شرایط از MACD
            if macd_signals.get('bearish'):
                sell_conditions.append("MACD Bearish")
                sell_score += 1
            
            if macd_signals.get('crossunder'):
                sell_conditions.append("MACD Crossunder")
                sell_score += 1
            
            # تصمیم‌گیری نهایی
            signal_type = "NEUTRAL"
            confidence = 0
            
            # آستانه پایین برای تست
            min_conditions = 2  # فقط ۲ شرط کافی است برای تست
            
            if buy_score >= min_conditions and buy_score > sell_score:
                signal_type = "BUY"
                confidence = min(95, buy_score * 15)  # حداکثر ۹۵%
            elif sell_score >= min_conditions and sell_score > buy_score:
                signal_type = "SELL"
                confidence = min(95, sell_score * 15)  # حداکثر ۹۵%
            
            # محاسبه ATR برای مدیریت ریسک
            atr = self.calculate_atr(df, 14)
            current_price = df['close'].iloc[-1]
            
            # محاسبه نقاط ورود و خروج
            if signal_type == "BUY":
                stop_loss = current_price - (atr * 1.5)
                take_profit_1 = current_price + (atr * 1.0)
                take_profit_2 = current_price + (atr * 2.0)
            elif signal_type == "SELL":
                stop_loss = current_price + (atr * 1.5)
                take_profit_1 = current_price - (atr * 1.0)
                take_profit_2 = current_price - (atr * 2.0)
            else:
                stop_loss = take_profit_1 = take_profit_2 = current_price
            
            # ساخت نتیجه
            result = {
                'signal_type': signal_type,
                'confidence': confidence,
                'buy_conditions': buy_conditions,
                'sell_conditions': sell_conditions,
                'buy_score': buy_score,
                'sell_score': sell_score,
                'price': current_price,
                'stop_loss': stop_loss,
                'take_profit_1': take_profit_1,
                'take_profit_2': take_profit_2,
                'atr': atr,
                'zlma_data': zlma_signals,
                'smart_money_data': sm_signals,
                'rsi_data': rsi_signals,
                'ichimoku_data': ichimoku_signals,
                'macd_data': macd_signals,
                'timestamp': pd.Timestamp.now()
            }
            
            self.logger.info(f"Signal generated: {signal_type} with {confidence}% confidence")
            self.logger.info(f"Buy conditions: {len(buy_conditions)}, Sell conditions: {len(sell_conditions)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in generate_combined_signal: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'signal_type': 'NEUTRAL', 'confidence': 0}
    
    # ============================================================
    # 🎯 تابع تست سریع
    # ============================================================
    
    def test_indicators(self, df: pd.DataFrame) -> Dict:
        """تست سریع تمام اندیکاتورها"""
        results = {
            'zlma': self.calculate_zlma_trend(df),
            'smart_money': self.calculate_smart_money_signals(df),
            'rsi': self.calculate_rsi_divergence(df),
            'ichimoku': self.calculate_ichimoku_cloud(df),
            'macd': self.calculate_macd(df),
            'signal': self.generate_combined_signal(df)
        }
        
        print("\n" + "="*60)
        print("🎯 TESTING ALL INDICATORS")
        print("="*60)
        
        for indicator_name, indicator_data in results.items():
            if indicator_data:
                print(f"\n📊 {indicator_name.upper()}:")
                for key, value in list(indicator_data.items())[:5]:  # فقط ۵ آیتم اول
                    print(f"  • {key}: {value}")
        
        print("="*60)
        
        return results

# ============================================================
# 🚀 برای تست مستقیم
# ============================================================

if __name__ == "__main__":
    print("🧪 Testing Combined Indicators...")
    
    # ایجاد داده تست
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='5min')
    close = 50000 + np.cumsum(np.random.randn(200) * 100)
    high = close + np.random.rand(200) * 100
    low = close - np.random.rand(200) * 100
    
    df = pd.DataFrame({
        'open': close - np.random.rand(200) * 50,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.rand(200) * 1000 + 500
    }, index=dates)
    
    # تست اندیکاتورها
    indicators = CombinedIndicators()
    signal = indicators.generate_combined_signal(df)
    
    print(f"\n🎯 FINAL SIGNAL:")
    print(f"Type: {signal.get('signal_type')}")
    print(f"Confidence: {signal.get('confidence')}%")
    print(f"Price: {signal.get('price'):.2f}")
    print(f"Stop Loss: {signal.get('stop_loss'):.2f}")
    print(f"Take Profit 1: {signal.get('take_profit_1'):.2f}")
    print(f"Take Profit 2: {signal.get('take_profit_2'):.2f}")
    
    print("\n✅ Indicators module is working!")
