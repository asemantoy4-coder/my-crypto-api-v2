"""
📦 Utilities Module - ابزارهای کمکی کامل برای Fast Scalp Bot
🚀 Version: 2.0.0 | برای Render, Northflank, Vercel بهینه‌شده
"""

import os
import sys
import logging
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import gzip
import csv
from collections import defaultdict, deque

# ============================================
# 🎯 Logger Configuration
# ============================================

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

def setup_logger(
    name: str = "fast_scalp",
    level: LogLevel = LogLevel.INFO,
    log_to_file: bool = False,
    log_file: str = "fast_scalp.log",
    console_output: bool = True
) -> logging.Logger:
    """
    تنظیم لاگر پیشرفته با قابلیت‌های مختلف
    
    Args:
        name: نام لاگر
        level: سطح لاگ
        log_to_file: ذخیره در فایل
        log_file: نام فایل لاگ
        console_output: نمایش در کنسول
    
    Returns:
        logging.Logger: آبجکت لاگر
    """
    logger = logging.getLogger(name)
    
    # اگر قبلاً تنظیم شده، return کن
    if logger.hasHandlers():
        return logger
    
    # تنظیم سطح
    logger.setLevel(level.value)
    
    # فرمت پیشرفته
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # هندلر کنسول
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level.value)
        logger.addHandler(console_handler)
    
    # هندلر فایل
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level.value)
        logger.addHandler(file_handler)
        
        # همچنین handler برای errorها
        error_handler = logging.FileHandler(log_dir / f"error_{log_file}", encoding='utf-8')
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        logger.addHandler(error_handler)
    
    # جلوگیری از انتشار به root logger
    logger.propagate = False
    
    return logger

def log_performance(func):
    """
    دکوراتور برای لاگ کردن عملکرد توابع
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            if elapsed > 1.0:
                logger.warning(f"⏱️ {func.__name__} took {elapsed:.2f}s (Slow)")
            elif elapsed > 0.5:
                logger.info(f"⏱️ {func.__name__} took {elapsed:.2f}s")
            else:
                logger.debug(f"⏱️ {func.__name__} took {elapsed:.4f}s")
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ {func.__name__} failed after {elapsed:.2f}s: {e}", exc_info=True)
            raise
    
    return wrapper

# ============================================
# 🔧 Data Utilities
# ============================================

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str] = None,
    min_rows: int = 50,
    check_nulls: bool = True,
    check_inf: bool = True
) -> Tuple[bool, str]:
    """
    اعتبارسنجی جامع دیتافریم OHLCV
    
    Args:
        df: دیتافریم ورودی
        required_columns: ستون‌های ضروری
        min_rows: حداقل سطرها
        check_nulls: بررسی مقادیر null
        check_inf: بررسی مقادیر infinite
    
    Returns:
        Tuple[bool, str]: (معتبر است, پیام خطا)
    """
    if required_columns is None:
        required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # 1. بررسی وجود دیتافریم
    if df is None:
        return False, "DataFrame is None"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    # 2. بررسی تعداد سطرها
    if len(df) < min_rows:
        return False, f"Not enough rows: {len(df)} < {min_rows}"
    
    # 3. بررسی ستون‌های ضروری
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing columns: {missing_columns}"
    
    # 4. بررسی مقادیر NaN
    if check_nulls:
        null_counts = df[required_columns].isna().sum()
        if null_counts.any():
            problematic = null_counts[null_counts > 0].to_dict()
            return False, f"NaN values found: {problematic}"
    
    # 5. بررسی مقادیر Infinite
    if check_inf:
        for col in required_columns:
            if col in df.columns:
                if np.isinf(df[col]).any():
                    return False, f"Infinite values in column: {col}"
    
    # 6. بررسی مقادیر منفی (برای قیمت و حجم)
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            if (df[col] <= 0).any():
                return False, f"Non-positive values in {col}"
    
    if 'volume' in df.columns:
        if (df['volume'] < 0).any():
            return False, "Negative volume values"
    
    # 7. بررسی توالی زمانی (اگر index datetime است)
    if isinstance(df.index, pd.DatetimeIndex):
        time_diff = df.index.to_series().diff().dropna()
        if (time_diff < timedelta(seconds=0)).any():
            return False, "Non-chronological timestamps"
    
    return True, "DataFrame is valid"

def clean_ohlcv_data(
    df: pd.DataFrame,
    remove_outliers: bool = True,
    outlier_threshold: float = 3.0,
    fill_method: str = 'ffill',
    volume_filter: bool = True
) -> pd.DataFrame:
    """
    تمیز کردن و پیش‌پردازش داده‌های OHLCV
    
    Args:
        df: دیتافریم خام
        remove_outliers: حذف outliers
        outlier_threshold: آستانه outlier
        fill_method: روش پر کردن مقادیر خالی
        volume_filter: فیلتر حجم صفر
    
    Returns:
        pd.DataFrame: دیتافریم تمیزشده
    """
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # 1. حذف سطرهای با حجم صفر یا منفی
    if volume_filter and 'volume' in df_clean.columns:
        df_clean = df_clean[df_clean['volume'] > 0]
    
    # 2. حذف outliers قیمت (تغییرات غیرعادی)
    if remove_outliers and len(df_clean) > 10:
        for col in ['open', 'high', 'low', 'close']:
            if col in df_clean.columns:
                # محاسبه درصد تغییر
                pct_change = df_clean[col].pct_change().abs()
                
                # حذف تغییرات بیش از threshold
                outlier_mask = pct_change > outlier_threshold
                
                # حذف outliers (اما حداکثر 5% داده‌ها)
                max_outliers = int(len(df_clean) * 0.05)
                if outlier_mask.sum() > max_outliers:
                    # فقط worst outliers را حذف کن
                    outlier_indices = pct_change.nlargest(max_outliers).index
                    df_clean = df_clean.drop(outlier_indices)
                else:
                    df_clean = df_clean[~outlier_mask]
    
    # 3. پر کردن مقادیر NaN
    if fill_method == 'ffill':
        df_clean = df_clean.ffill()
    elif fill_method == 'bfill':
        df_clean = df_clean.bfill()
    elif fill_method == 'interpolate':
        df_clean = df_clean.interpolate(method='linear')
    
    # 4. حذف هر ردیف که هنوز NaN دارد
    df_clean = df_clean.dropna()
    
    # 5. اطمینان از توالی زمانی
    if isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean = df_clean.sort_index()
    
    # 6. اعتبارسنجی نهایی
    is_valid, msg = validate_dataframe(df_clean)
    if not is_valid:
        logging.warning(f"Data cleaning warning: {msg}")
    
    return df_clean

def calculate_volume_profile(
    df: pd.DataFrame,
    bins: int = 20,
    price_col: str = 'close',
    volume_col: str = 'volume'
) -> Dict[str, Any]:
    """
    محاسبه Volume Profile با جزئیات کامل
    
    Args:
        df: دیتافریم OHLCV
        bins: تعداد bins
        price_col: ستون قیمت
        volume_col: ستون حجم
    
    Returns:
        Dict: اطلاعات volume profile
    """
    if df.empty or len(df) < 10:
        return {}
    
    try:
        prices = df[price_col].values
        volumes = df[volume_col].values
        
        # محاسبه دامنه قیمت
        min_price = np.nanmin(prices)
        max_price = np.nanmax(prices)
        
        if min_price == max_price or np.isnan(min_price) or np.isnan(max_price):
            return {}
        
        # ایجاد bins
        price_bins = np.linspace(min_price, max_price, bins + 1)
        
        # محاسبه حجم در هر bin
        volume_by_bin = np.zeros(bins)
        price_midpoints = np.zeros(bins)
        
        for i in range(bins):
            bin_low = price_bins[i]
            bin_high = price_bins[i + 1]
            price_midpoints[i] = (bin_low + bin_high) / 2
            
            # پیدا کردن کندل‌هایی که در این بازه هستند
            mask = (prices >= bin_low) & (prices <= bin_high)
            volume_by_bin[i] = np.sum(volumes[mask])
        
        # یافتن نقطه کنترل حجم (POC)
        if np.sum(volume_by_bin) > 0:
            poc_index = np.argmax(volume_by_bin)
            poc_price = price_midpoints[poc_index]
            poc_volume = volume_by_bin[poc_index]
        else:
            poc_index = -1
            poc_price = np.nan
            poc_volume = 0
        
        # محاسبه Value Area (70% حجم)
        total_volume = np.sum(volume_by_bin)
        if total_volume > 0:
            # مرتب کردن bins بر اساس حجم
            sorted_indices = np.argsort(volume_by_bin)[::-1]
            cumulative_volume = 0
            value_area_indices = []
            
            for idx in sorted_indices:
                cumulative_volume += volume_by_bin[idx]
                value_area_indices.append(idx)
                
                if cumulative_volume / total_volume >= 0.7:
                    break
            
            # قیمت‌های Value Area
            value_area_prices = price_midpoints[value_area_indices]
            value_area_low = np.min(value_area_prices)
            value_area_high = np.max(value_area_prices)
        else:
            value_area_low = np.nan
            value_area_high = np.nan
        
        # تشخیص وضعیت فعلی نسبت به Volume Profile
        current_price = prices[-1]
        
        if not np.isnan(current_price):
            if current_price > value_area_high:
                price_position = "above_value_area"
            elif current_price < value_area_low:
                price_position = "below_value_area"
            else:
                price_position = "inside_value_area"
        else:
            price_position = "unknown"
        
        return {
            'price_range': {
                'min': float(min_price),
                'max': float(max_price),
                'range': float(max_price - min_price)
            },
            'poc': {
                'price': float(poc_price),
                'volume': float(poc_volume),
                'index': int(poc_index)
            },
            'value_area': {
                'low': float(value_area_low),
                'high': float(value_area_high),
                'width': float(value_area_high - value_area_low)
            },
            'current_position': price_position,
            'bins': {
                'prices': [float(p) for p in price_midpoints],
                'volumes': [float(v) for v in volume_by_bin]
            },
            'total_volume': float(total_volume),
            'current_price': float(current_price) if not np.isnan(current_price) else None
        }
        
    except Exception as e:
        logging.error(f"Error calculating volume profile: {e}")
        return {}

# ============================================
# 📈 Technical Analysis Utilities
# ============================================

def calculate_support_resistance(
    df: pd.DataFrame,
    window: int = 20,
    pivot_window: int = 5,
    strength_threshold: int = 2,
    merge_threshold: float = 0.02
) -> Dict[str, Any]:
    """
    تشخیص سطوح حمایت و مقاومت با الگوریتم پیشرفته
    
    Args:
        df: دیتافریم قیمت
        window: window برای تشخیص pivot
        pivot_window: window محلی برای pivot
        strength_threshold: حداقل قدرت سطح
        merge_threshold: آستانه ادغام سطوح نزدیک
    
    Returns:
        Dict: سطوح حمایت و مقاومت
    """
    if len(df) < window * 2:
        return {'supports': [], 'resistances': []}
    
    try:
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        supports = []
        resistances = []
        
        # تشخیص pivot points
        for i in range(window, len(df) - window):
            # مقاومت (سقف محلی)
            is_resistance = True
            
            # بررسی سمت چپ
            for j in range(1, pivot_window + 1):
                if highs[i] <= highs[i - j]:
                    is_resistance = False
                    break
            
            # بررسی سمت راست
            if is_resistance:
                for j in range(1, pivot_window + 1):
                    if highs[i] <= highs[i + j]:
                        is_resistance = False
                        break
            
            if is_resistance:
                resistances.append({
                    'price': float(highs[i]),
                    'index': i,
                    'strength': 1,
                    'timestamp': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else None
                })
            
            # حمایت (کف محلی)
            is_support = True
            
            # بررسی سمت چپ
            for j in range(1, pivot_window + 1):
                if lows[i] >= lows[i - j]:
                    is_support = False
                    break
            
            # بررسی سمت راست
            if is_support:
                for j in range(1, pivot_window + 1):
                    if lows[i] >= lows[i + j]:
                        is_support = False
                        break
            
            if is_support:
                supports.append({
                    'price': float(lows[i]),
                    'index': i,
                    'strength': 1,
                    'timestamp': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else None
                })
        
        # تقویت سطوح بر اساس تعداد برخوردها
        current_price = closes[-1]
        price_range = np.max(highs) - np.min(lows)
        merge_distance = price_range * merge_threshold
        
        def merge_and_strengthen(levels, is_support=True):
            if not levels:
                return []
            
            levels.sort(key=lambda x: x['price'])
            merged = []
            
            for level in levels:
                if not merged:
                    merged.append(level.copy())
                    continue
                
                last = merged[-1]
                price_diff = abs(level['price'] - last['price'])
                
                if price_diff <= merge_distance:
                    # ادغام سطح
                    last['price'] = (last['price'] + level['price']) / 2
                    last['strength'] += level['strength']
                    # میانگین index
                    last['index'] = (last['index'] + level['index']) // 2
                else:
                    merged.append(level.copy())
            
            # فقط سطوح قوی
            filtered = [l for l in merged if l['strength'] >= strength_threshold]
            
            # مرتب‌سازی بر اساس نزدیکی به قیمت فعلی
            filtered.sort(key=lambda x: abs(x['price'] - current_price))
            
            return filtered
        
        supports = merge_and_strengthen(supports, is_support=True)
        resistances = merge_and_strengthen(resistances, is_support=False)
        
        # یافتن نزدیک‌ترین سطوح
        nearest_support = supports[0] if supports else None
        nearest_resistance = resistances[0] if resistances else None
        
        # تشخیص وضعیت فعلی
        if nearest_support and nearest_resistance:
            distance_to_support = abs(current_price - nearest_support['price'])
            distance_to_resistance = abs(current_price - nearest_resistance['price'])
            
            if distance_to_support < distance_to_resistance:
                current_zone = "near_support"
                zone_distance = distance_to_support / price_range * 100
            else:
                current_zone = "near_resistance"
                zone_distance = distance_to_resistance / price_range * 100
        elif nearest_support:
            current_zone = "near_support"
            zone_distance = abs(current_price - nearest_support['price']) / price_range * 100
        elif nearest_resistance:
            current_zone = "near_resistance"
            zone_distance = abs(current_price - nearest_resistance['price']) / price_range * 100
        else:
            current_zone = "no_level"
            zone_distance = 100.0
        
        return {
            'supports': supports[-10:],  # آخرین 10 سطح
            'resistances': resistances[-10:],
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'current_price': float(current_price),
            'current_zone': current_zone,
            'zone_distance_percent': float(zone_distance),
            'price_range': float(price_range),
            'total_levels': len(supports) + len(resistances)
        }
        
    except Exception as e:
        logging.error(f"Error calculating support/resistance: {e}")
        return {'supports': [], 'resistances': []}

def calculate_market_structure(df: pd.DataFrame, lookback: int = 5) -> Dict[str, Any]:
    """
    تشخیص ساختار بازار (Higher Highs/Lower Lows)
    
    Args:
        df: دیتافریم قیمت
        lookback: lookback برای تشخیص swing points
    
    Returns:
        Dict: ساختار بازار
    """
    if len(df) < lookback * 4:
        return {'trend': 'neutral', 'structure': 'insufficient_data'}
    
    try:
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # تشخیص swing highs و lows
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(df) - lookback):
            # Swing High
            if highs[i] == highs[i - lookback:i + lookback + 1].max():
                swing_highs.append({
                    'index': i,
                    'price': float(highs[i]),
                    'time': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else None
                })
            
            # Swing Low
            if lows[i] == lows[i - lookback:i + lookback + 1].min():
                swing_lows.append({
                    'index': i,
                    'price': float(lows[i]),
                    'time': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else None
                })
        
        # تشخیص روند
        trend = "neutral"
        trend_strength = 0
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # آخرین دو swing
            last_high = swing_highs[-1]['price']
            prev_high = swing_highs[-2]['price']
            last_low = swing_lows[-1]['price']
            prev_low = swing_lows[-2]['price']
            
            # بررسی Higher Highs و Higher Lows
            higher_highs = last_high > prev_high
            higher_lows = last_low > prev_low
            
            # بررسی Lower Lows و Lower Highs
            lower_lows = last_low < prev_low
            lower_highs = last_high < prev_high
            
            if higher_highs and higher_lows:
                trend = "uptrend"
                trend_strength = min(
                    (last_high - prev_high) / prev_high * 100,
                    (last_low - prev_low) / prev_low * 100
                )
            elif lower_highs and lower_lows:
                trend = "downtrend"
                trend_strength = min(
                    (prev_high - last_high) / prev_high * 100,
                    (prev_low - last_low) / prev_low * 100
                )
            elif higher_highs and lower_lows:
                trend = "expansion"
                trend_strength = 0
            elif lower_highs and higher_lows:
                trend = "contraction"
                trend_strength = 0
            else:
                trend = "ranging"
                trend_strength = 0
        
        # تشخیص شکست ساختار
        structure_break = None
        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            # بررسی break of structure (BOS)
            if trend == "uptrend" and closes[-1] > swing_highs[-2]['price']:
                structure_break = "bullish_bos"
            elif trend == "downtrend" and closes[-1] < swing_lows[-2]['price']:
                structure_break = "bearish_bos"
        
        # تشخیص تغییر روند احتمالی
        potential_reversal = None
        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            if trend == "uptrend" and closes[-1] < swing_lows[-2]['price']:
                potential_reversal = "bearish_reversal"
            elif trend == "downtrend" and closes[-1] > swing_highs[-2]['price']:
                potential_reversal = "bullish_reversal"
        
        return {
            'trend': trend,
            'trend_strength': float(trend_strength),
            'swing_highs': swing_highs[-5:] if swing_highs else [],
            'swing_lows': swing_lows[-5:] if swing_lows else [],
            'structure_break': structure_break,
            'potential_reversal': potential_reversal,
            'current_price': float(closes[-1]),
            'market_phase': get_market_phase(df),
            'volatility': calculate_volatility(df, 20)
        }
        
    except Exception as e:
        logging.error(f"Error calculating market structure: {e}")
        return {'trend': 'error', 'structure': str(e)}

def get_market_phase(df: pd.DataFrame) -> str:
    """
    تشخیص فاز بازار
    """
    if len(df) < 50:
        return "unknown"
    
    # محاسبه moving averages
    ma20 = df['close'].rolling(20).mean()
    ma50 = df['close'].rolling(50).mean()
    
    current_price = df['close'].iloc[-1]
    ma20_current = ma20.iloc[-1]
    ma50_current = ma50.iloc[-1]
    
    # بررسی موقعیت قیمت نسبت به MAها
    above_ma20 = current_price > ma20_current
    above_ma50 = current_price > ma50_current
    ma20_above_ma50 = ma20_current > ma50_current
    
    # تشخیص فاز
    if above_ma20 and above_ma50 and ma20_above_ma50:
        return "bullish"
    elif not above_ma20 and not above_ma50 and not ma20_above_ma50:
        return "bearish"
    elif above_ma50 and not ma20_above_ma50:
        return "recovery"
    elif not above_ma50 and ma20_above_ma50:
        return "pullback"
    else:
        return "consolidation"

def calculate_volatility(df: pd.DataFrame, period: int = 20) -> float:
    """
    محاسبه نوسان بازار
    """
    if len(df) < period:
        return 0.0
    
    returns = df['close'].pct_change().dropna()
    if len(returns) < period:
        return 0.0
    
    volatility = returns.tail(period).std() * np.sqrt(252) * 100  # نوسان سالانه درصدی
    return float(volatility)

# ============================================
# ⚡ Performance Tracking
# ============================================

@dataclass
class SignalRecord:
    """رکورد سیگنال"""
    symbol: str
    signal_type: str  # BUY/SELL
    entry_price: float
    entry_time: datetime
    confidence: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    reasons: List[str]
    atr: float
    volume: float
    status: str = "open"  # open, closed, cancelled
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    pnl_percent: Optional[float] = None
    pnl_absolute: Optional[float] = None

class PerformanceTracker:
    """
    سیستم ردیابی عملکرد سیگنال‌ها
    """
    
    def __init__(self, cache_file: str = "performance_cache.json"):
        self.cache_file = Path(cache_file)
        self.signals = self._load_signals()
        self.metrics_cache = {}
    
    def _load_signals(self) -> List[Dict]:
        """لود سیگنال‌ها از کش"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # تبدیل string تاریخ به datetime
                    for signal in data:
                        for time_field in ['entry_time', 'exit_time']:
                            if signal.get(time_field):
                                signal[time_field] = datetime.fromisoformat(signal[time_field])
                    return data
        except Exception as e:
            logging.error(f"Error loading signals cache: {e}")
        return []
    
    def _save_signals(self):
        """ذخیره سیگنال‌ها در کش"""
        try:
            # تبدیل datetime به string
            signals_to_save = []
            for signal in self.signals:
                signal_copy = signal.copy()
                for time_field in ['entry_time', 'exit_time']:
                    if signal_copy.get(time_field):
                        signal_copy[time_field] = signal_copy[time_field].isoformat()
                signals_to_save.append(signal_copy)
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(signals_to_save[-100:], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving signals cache: {e}")
    
    def add_signal(self, signal_data: Dict) -> str:
        """
        افزودن سیگنال جدید
        
        Returns:
            str: signal_id
        """
        try:
            signal_id = hashlib.md5(
                f"{signal_data['symbol']}_{signal_data['entry_price']}_{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()[:12]
            
            signal_record = {
                'signal_id': signal_id,
                'symbol': signal_data.get('symbol'),
                'signal_type': signal_data.get('type'),
                'entry_price': signal_data.get('price'),
                'entry_time': datetime.utcnow(),
                'confidence': signal_data.get('confidence', 0),
                'stop_loss': signal_data.get('stop_loss'),
                'take_profit_1': signal_data.get('take_profit_1'),
                'take_profit_2': signal_data.get('take_profit_2'),
                'reasons': signal_data.get('reasons', []),
                'atr': signal_data.get('atr', 0),
                'volume': signal_data.get('volume', 0),
                'status': 'open',
                'metadata': signal_data.get('metadata', {})
            }
            
            self.signals.append(signal_record)
            self._save_signals()
            
            logging.info(f"📝 Signal recorded: {signal_id} | {signal_data['symbol']} {signal_data['type']}")
            return signal_id
            
        except Exception as e:
            logging.error(f"Error adding signal: {e}")
            return ""
    
    def update_signal(
        self,
        signal_id: str,
        exit_price: float,
        exit_reason: str,
        status: str = "closed"
    ) -> bool:
        """آپدیت نتیجه سیگنال"""
        for signal in self.signals:
            if signal.get('signal_id') == signal_id and signal.get('status') == 'open':
                signal['exit_price'] = exit_price
                signal['exit_time'] = datetime.utcnow()
                signal['exit_reason'] = exit_reason
                signal['status'] = status
                
                # محاسبه P&L
                if signal['signal_type'] == 'BUY':
                    signal['pnl_percent'] = ((exit_price - signal['entry_price']) / signal['entry_price']) * 100
                else:  # SELL
                    signal['pnl_percent'] = ((signal['entry_price'] - exit_price) / signal['entry_price']) * 100
                
                signal['pnl_absolute'] = abs(exit_price - signal['entry_price'])
                
                self._save_signals()
                logging.info(f"📊 Signal updated: {signal_id} | P&L: {signal['pnl_percent']:.2f}%")
                return True
        
        logging.warning(f"Signal not found or already closed: {signal_id}")
        return False
    
    def get_performance_stats(self, days: int = 30) -> Dict[str, Any]:
        """آمار عملکرد"""
        now = datetime.utcnow()
        cutoff_date = now - timedelta(days=days)
        
        # فیلتر سیگنال‌های بسته شده در بازه زمانی
        closed_signals = [
            s for s in self.signals 
            if s.get('status') == 'closed' and s.get('exit_time') and s['exit_time'] >= cutoff_date
        ]
        
        if not closed_signals:
            return {
                'total_signals': 0,
                'message': 'No closed signals in period'
            }
        
        # جدا کردن سیگنال‌های خرید و فروش
        buy_signals = [s for s in closed_signals if s['signal_type'] == 'BUY']
        sell_signals = [s for s in closed_signals if s['signal_type'] == 'SELL']
        
        # محاسبات کلی
        total_signals = len(closed_signals)
        winning_signals = [s for s in closed_signals if s.get('pnl_percent', 0) > 0]
        losing_signals = [s for s in closed_signals if s.get('pnl_percent', 0) < 0]
        breakeven_signals = [s for s in closed_signals if s.get('pnl_percent', 0) == 0]
        
        win_rate = (len(winning_signals) / total_signals * 100) if total_signals > 0 else 0
        
        # محاسبه میانگین‌ها
        avg_win = np.mean([s.get('pnl_percent', 0) for s in winning_signals]) if winning_signals else 0
        avg_loss = np.mean([abs(s.get('pnl_percent', 0)) for s in losing_signals]) if losing_signals else 0
        
        # Risk/Reward Ratio
        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Expectancy
        win_probability = len(winning_signals) / total_signals if total_signals > 0 else 0
        loss_probability = len(losing_signals) / total_signals if total_signals > 0 else 0
        expectancy = (win_probability * avg_win) - (loss_probability * avg_loss)
        
        # Sharpe Ratio (ساده شده)
        returns = [s.get('pnl_percent', 0) for s in closed_signals]
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # محاسبه drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max * 100 if len(running_max) > 0 else []
        max_drawdown = min(drawdown) if len(drawdown) > 0 else 0
        
        # بهترین و بدترین سیگنال
        best_signal = max(closed_signals, key=lambda x: x.get('pnl_percent', -999), default=None)
        worst_signal = min(closed_signals, key=lambda x: x.get('pnl_percent', 999), default=None)
        
        stats = {
            'period_days': days,
            'total_signals': total_signals,
            'win_rate': round(win_rate, 2),
            'wins': len(winning_signals),
            'losses': len(losing_signals),
            'breakevens': len(breakeven_signals),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'risk_reward': round(risk_reward, 2),
            'expectancy': round(expectancy, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(abs(max_drawdown), 2),
            'total_return': round(sum(returns), 2),
            'avg_return': round(np.mean(returns), 2),
            'std_return': round(np.std(returns), 2) if len(returns) > 1 else 0,
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'best_signal': {
                'symbol': best_signal['symbol'] if best_signal else None,
                'type': best_signal['signal_type'] if best_signal else None,
                'pnl': best_signal.get('pnl_percent') if best_signal else None,
                'date': best_signal['exit_time'].date().isoformat() if best_signal and best_signal.get('exit_time') else None
            } if best_signal else None,
            'worst_signal': {
                'symbol': worst_signal['symbol'] if worst_signal else None,
                'type': worst_signal['signal_type'] if worst_signal else None,
                'pnl': worst_signal.get('pnl_percent') if worst_signal else None,
                'date': worst_signal['exit_time'].date().isoformat() if worst_signal and worst_signal.get('exit_time') else None
            } if worst_signal else None,
            'last_updated': now.isoformat()
        }
        
        # کش کردن آمار
        self.metrics_cache[f'stats_{days}d'] = {
            'data': stats,
            'timestamp': now
        }
        
        return stats
    
    def get_recent_signals(self, limit: int = 10, signal_type: str = None) -> List[Dict]:
        """دریافت آخرین سیگنال‌ها"""
        filtered = self.signals[-limit * 2:]  # کمی بیشتر بگیر
        
        if signal_type:
            filtered = [s for s in filtered if s.get('signal_type') == signal_type]
        
        return filtered[-limit:] if filtered else []
    
    def get_active_signals(self) -> List[Dict]:
        """دریافت سیگنال‌های فعال"""
        return [s for s in self.signals if s.get('status') == 'open']
    
    def cleanup_old_signals(self, max_age_days: int = 90):
        """پاکسازی سیگنال‌های قدیمی"""
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        
        initial_count = len(self.signals)
        self.signals = [s for s in self.signals if s.get('entry_time', datetime.min) >= cutoff_date]
        
        removed = initial_count - len(self.signals)
        if removed > 0:
            logging.info(f"🧹 Cleaned up {removed} old signals")
            self._save_signals()
        
        return removed

# ============================================
# 🔐 Security & Validation
# ============================================

def validate_api_keys(config: Dict) -> Tuple[bool, List[str]]:
    """
    اعتبارسنجی کامل کلیدهای API
    
    Returns:
        Tuple[bool, List[str]]: (معتبر است, لیست خطاها)
    """
    errors = []
    
    # 1. بررسی ساختار config
    if not config or not isinstance(config, dict):
        errors.append("Config is empty or not a dictionary")
        return False, errors
    
    # 2. بررسی تلگرام (ضروری)
    telegram_token = config.get('telegram_token') or config.get('telegram', {}).get('token')
    chat_id = config.get('chat_id') or config.get('telegram', {}).get('chat_id')
    
    if not telegram_token:
        errors.append("TELEGRAM_BOT_TOKEN is required")
    else:
        # بررسی فرمت توکن تلگرام
        if not isinstance(telegram_token, str):
            errors.append("TELEGRAM_BOT_TOKEN must be a string")
        elif len(telegram_token) < 30:
            errors.append("TELEGRAM_BOT_TOKEN seems invalid (too short)")
        elif not telegram_token.startswith(''):
            errors.append("TELEGRAM_BOT_TOKEN has invalid format")
    
    if not chat_id:
        errors.append("TELEGRAM_CHAT_ID is required")
    else:
        # بررسی فرمت chat_id
        try:
            # می‌تواند عددی یا رشته باشد
            chat_id_str = str(chat_id)
            if not chat_id_str.strip():
                errors.append("TELEGRAM_CHAT_ID cannot be empty")
            elif chat_id_str.startswith('-100'):
                # بررسی گروه/کانال
                if not chat_id_str[4:].isdigit():
                    errors.append("Invalid group/channel ID format")
        except:
            errors.append("TELEGRAM_CHAT_ID must be string or number")
    
    # 3. بررسی MEXC API (اختیاری)
    mexc_api_key = config.get('mexc_api_key') or config.get('exchange', {}).get('api_key')
    mexc_secret = config.get('mexc_secret_key') or config.get('exchange', {}).get('secret')
    
    if mexc_api_key or mexc_secret:
        # اگر یکی از آن‌ها پر شده، هر دو باید پر شوند
        if not mexc_api_key:
            errors.append("MEXC_API_KEY is required when secret is provided")
        if not mexc_secret:
            errors.append("MEXC_SECRET_KEY is required when API key is provided")
        
        if mexc_api_key:
            if not isinstance(mexc_api_key, str):
                errors.append("MEXC_API_KEY must be a string")
            elif len(mexc_api_key) < 20:
                errors.append("MEXC_API_KEY seems invalid (too short)")
        
        if mexc_secret:
            if not isinstance(mexc_secret, str):
                errors.append("MEXC_SECRET_KEY must be a string")
            elif len(mexc_secret) < 20:
                errors.append("MEXC_SECRET_KEY seems invalid (too short)")
    
    # 4. بررسی تنظیمات استراتژی
    strategy_config = config.get('strategy', {})
    if strategy_config:
        required_strategy_params = ['timeframe', 'top_n', 'update_interval']
        for param in required_strategy_params:
            if param not in strategy_config:
                errors.append(f"Strategy parameter '{param}' is required")
    
    return len(errors) == 0, errors

def sanitize_output(data: Any, depth: int = 0, max_depth: int = 3) -> Any:
    """
    پاکسازی خروجی برای جلوگیری از افشای اطلاعات حساس
    
    Args:
        data: داده ورودی
        depth: عمق فعلی (برای جلوگیری از recursion infinite)
        max_depth: حداکثر عمق
    
    Returns:
        Any: داده سانتایز شده
    """
    if depth > max_depth:
        return "[Max depth reached]"
    
    sensitive_keywords = [
        'key', 'token', 'secret', 'password', 'pass', 'auth',
        'apikey', 'api_key', 'private', 'cert', 'credential'
    ]
    
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            key_lower = str(key).lower()
            
            # اگر کلید حساس است، سانسور کن
            if any(sensitive in key_lower for sensitive in sensitive_keywords):
                if isinstance(value, str) and value:
                    # نمایش جزئی از مقدار
                    if len(value) > 8:
                        sanitized[key] = f"{value[:2]}...{value[-2:]}"
                    else:
                        sanitized[key] = "***"
                else:
                    sanitized[key] = "***"
            else:
                sanitized[key] = sanitize_output(value, depth + 1, max_depth)
        
        return sanitized
    
    elif isinstance(data, list):
        return [sanitize_output(item, depth + 1, max_depth) for item in data]
    
    elif isinstance(data, tuple):
        return tuple(sanitize_output(list(data), depth + 1, max_depth))
    
    else:
        return data

# ============================================
# 📊 Formatting Utilities
# ============================================

def format_price(price: float, symbol: str = 'USDT', precision: int = None) -> str:
    """
    فرمت‌دهی قیمت بر اساس مقدار و جفت ارز
    """
    if price is None or np.isnan(price):
        return "N/A"
    
    # تعیین precision خودکار
    if precision is None:
        if price >= 1000:
            precision = 2
        elif price >= 1:
            precision = 4
        elif price >= 0.01:
            precision = 6
        else:
            precision = 8
    
    # فرمت عدد
    if price >= 1000:
        formatted = f"{price:,.{precision}f}"
    elif price >= 1:
        formatted = f"{price:.{precision}f}"
    else:
        formatted = f"{price:.{precision}f}"
    
    # اضافه کردن symbol
    if symbol:
        formatted = f"{formatted} {symbol}"
    
    return formatted

def format_percentage(value: float, show_plus: bool = True, precision: int = 2) -> str:
    """فرمت‌دهی درصد"""
    if value is None or np.isnan(value):
        return "N/A"
    
    sign = "+" if value > 0 and show_plus else ""
    return f"{sign}{value:.{precision}f}%"

def format_timestamp(
    timestamp,
    format_str: str = '%Y-%m-%d %H:%M:%S',
    timezone: str = 'UTC'
) -> str:
    """فرمت‌دهی timestamp به رشته خوانا"""
    if timestamp is None:
        return "N/A"
    
    try:
        if isinstance(timestamp, (int, float)):
            from datetime import datetime
            timestamp = datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            from datetime import datetime
            # تبدیل string به datetime
            timestamp = timestamp.replace('Z', '+00:00')
            timestamp = datetime.fromisoformat(timestamp)
        
        # اعمال timezone اگر نیاز است
        if timezone != 'UTC':
            import pytz
            tz = pytz.timezone(timezone)
            timestamp = timestamp.astimezone(tz)
        
        return timestamp.strftime(format_str)
        
    except Exception as e:
        logging.debug(f"Error formatting timestamp: {e}")
        return str(timestamp)

def format_signal_for_display(signal: Dict, include_details: bool = True) -> str:
    """
    فرمت‌دهی سیگنال برای نمایش در خروجی‌های مختلف
    """
    if not signal:
        return "No signal data"
    
    emoji = "🟢" if signal.get('type') == 'BUY' else "🔴"
    symbol = signal.get('symbol', 'N/A')
    signal_type = signal.get('type', 'N/A')
    confidence = signal.get('confidence', 0)
    price = signal.get('price', 0)
    
    lines = [
        f"{emoji} **{symbol}** - {signal_type}",
        f"Confidence: **{confidence}%**",
        f"Price: `{format_price(price)}`"
    ]
    
    if include_details:
        if 'stop_loss' in signal:
            sl = signal['stop_loss']
            sl_pct = ((price - sl) / price * 100) if signal_type == 'BUY' else ((sl - price) / price * 100)
            lines.append(f"Stop Loss: `{format_price(sl)}` ({abs(sl_pct):.2f}%)")
        
        if 'take_profit_1' in signal:
            tp1 = signal['take_profit_1']
            tp1_pct = ((tp1 - price) / price * 100) if signal_type == 'BUY' else ((price - tp1) / price * 100)
            lines.append(f"Take Profit 1: `{format_price(tp1)}` ({tp1_pct:.2f}%)")
        
        if 'take_profit_2' in signal:
            tp2 = signal['take_profit_2']
            tp2_pct = ((tp2 - price) / price * 100) if signal_type == 'BUY' else ((price - tp2) / price * 100)
            lines.append(f"Take Profit 2: `{format_price(tp2)}` ({tp2_pct:.2f}%)")
        
        if 'reasons' in signal and signal['reasons']:
            reasons = ", ".join(signal['reasons'][:3])  # فقط ۳ دلیل اول
            lines.append(f"Reasons: {reasons}")
        
        if 'atr' in signal:
            lines.append(f"ATR: `{format_price(signal['atr'])}`")
    
    return "\n".join(lines)

def format_telegram_message(signal: Dict) -> str:
    """
    فرمت مخصوص پیام تلگرام
    """
    emoji = "🟢" if signal.get('type') == 'BUY' else "🔴"
    
    message = f"""
{emoji} *FAST SCALP SIGNAL* {emoji}

*Symbol:* `{signal.get('symbol', 'N/A')}`
*Type:* {signal.get('type', 'N/A')}
*Confidence:* {signal.get('confidence', 0)}%

💰 *Entry Price:* {format_price(signal.get('price', 0))}
🛑 *Stop Loss:* {format_price(signal.get('stop_loss', 0))}
🎯 *TP1:* {format_price(signal.get('take_profit_1', 0))}
🎯 *TP2:* {format_price(signal.get('take_profit_2', 0))}

📊 *Risk/Reward:* 1:{abs(signal.get('take_profit_1', 0) - signal.get('price', 0)) / abs(signal.get('price', 0) - signal.get('stop_loss', 0)):.1f}
📈 *ATR:* {format_price(signal.get('atr', 0))}

📝 *Reasons:*
"""
    
    if 'reasons' in signal and signal['reasons']:
        for i, reason in enumerate(signal['reasons'][:5], 1):
            message += f"  {i}. {reason}\n"
    
    message += f"\n⏰ *Time:* {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    
    return message

# ============================================
# ⏰ Time Utilities
# ============================================

def is_market_hours() -> bool:
    """
    بررسی ساعات فعال بازار کریپتو
    """
    now_utc = datetime.utcnow()
    hour_utc = now_utc.hour
    
    # ساعات پرترافیک بازار کریپتو (۸ صبح تا ۸ شب UTC)
    # این ساعات حجم معاملات بیشتر و نوسانات بهتری دارد
    return 8 <= hour_utc < 20

def next_scan_time() -> Dict[str, Any]:
    """محاسبه زمان اسکن بعدی"""
    now = datetime.utcnow()
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    
    remaining = next_hour - now
    total_seconds = int(remaining.total_seconds())
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    return {
        'next_scan': next_hour,
        'remaining_seconds': total_seconds,
        'formatted': f"{minutes:02d}:{seconds:02d}",
        'detailed': f"{hours}h {minutes}m {seconds}s",
        'is_soon': total_seconds < 300  # کمتر از ۵ دقیقه
    }

def calculate_time_until(target_time: str, timezone: str = 'UTC') -> Dict[str, Any]:
    """
    محاسبه زمان باقی‌مانده تا زمان هدف
    
    Args:
        target_time: 'HH:MM' یا 'HH:MM:SS' format
        timezone: timezone هدف
    
    Returns:
        Dict: اطلاعات زمان باقی‌مانده
    """
    now = datetime.utcnow()
    
    try:
        # پارس کردن زمان هدف
        time_parts = target_time.split(':')
        if len(time_parts) == 2:
            target_hour, target_minute = map(int, time_parts)
            target_second = 0
        elif len(time_parts) == 3:
            target_hour, target_minute, target_second = map(int, time_parts)
        else:
            return {"error": "Invalid time format"}
        
        # ساخت datetime برای امروز
        target = now.replace(
            hour=target_hour,
            minute=target_minute,
            second=target_second,
            microsecond=0
        )
        
        # اگر زمان گذشته است، برای فردا حساب کن
        if target < now:
            target += timedelta(days=1)
        
        # محاسبه تفاوت
        remaining = target - now
        total_seconds = int(remaining.total_seconds())
        
        # تجزیه به اجزا
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        return {
            'target': target,
            'remaining_seconds': total_seconds,
            'days': days,
            'hours': hours,
            'minutes': minutes,
            'seconds': seconds,
            'formatted': f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            'detailed': f"{days}d {hours}h {minutes}m {seconds}s" if days > 0 else f"{hours}h {minutes}m {seconds}s",
            'is_today': days == 0,
            'is_soon': total_seconds < 3600  # کمتر از ۱ ساعت
        }
        
    except Exception as e:
        return {"error": str(e)}

def get_market_session() -> str:
    """
    تشخیص session بازار
    """
    now_utc = datetime.utcnow()
    hour_utc = now_utc.hour
    
    if 0 <= hour_utc < 4:
        return "asian_session"
    elif 4 <= hour_utc < 8:
        return "asian_europe_overlap"
    elif 8 <= hour_utc < 12:
        return "european_session"
    elif 12 <= hour_utc < 16:
        return "europe_us_overlap"
    elif 16 <= hour_utc < 20:
        return "us_session"
    else:
        return "late_us_pacific"

# ============================================
# 📈 Signal Scoring
# ============================================

class SignalScorer:
    """
    سیستم امتیازدهی پیشرفته به سیگنال‌ها
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Args:
            weights: وزن هر فاکتور
        """
        self.weights = weights or {
            'zlma_signal': 25,
            'smart_money': 20,
            'rsi_divergence': 15,
            'ichimoku': 15,
            'volume_confirmation': 10,
            'market_structure': 10,
            'support_resistance': 5
        }
        
        # تنظیمات اضافی
        self.config = {
            'volume_threshold': 1.5,  # حداقل حجم نسبت به میانگین
            'min_confidence': 65,      # حداقل اطمینان کلی
            'max_age_hours': 24,       # حداکثر عمر سیگنال
            'require_multiple_confirmations': True
        }
    
    def calculate_score(self, signal_data: Dict, df: pd.DataFrame, market_data: Dict = None) -> Dict[str, Any]:
        """
        محاسبه امتیاز سیگنال با در نظر گرفتن همه فاکتورها
        
        Args:
            signal_data: داده‌های سیگنال
            df: دیتافریم قیمت
            market_data: داده‌های بازار اضافی
        
        Returns:
            Dict: امتیاز و جزئیات
        """
        score = 0
        max_score = sum(self.weights.values())
        breakdown = {}
        reasons = []
        warnings = []
        
        signal_type = signal_data.get('type')
        current_price = df['close'].iloc[-1] if not df.empty else 0
        
        # 1. ZLMA Signal
        zlma_score = 0
        if signal_data.get('zlma_signal_up') and signal_type == 'BUY':
            zlma_score = self.weights['zlma_signal']
            reasons.append("ZLMA Bullish Crossover")
        elif signal_data.get('zlma_signal_dn') and signal_type == 'SELL':
            zlma_score = self.weights['zlma_signal']
            reasons.append("ZLMA Bearish Crossunder")
        score += zlma_score
        breakdown['zlma'] = zlma_score
        
        # 2. Smart Money Signals
        sm_score = 0
        sm_reasons = []
        
        if signal_type == 'BUY':
            if signal_data.get('smart_money_buy'):
                sm_score += self.weights['smart_money'] * 0.6
                sm_reasons.append("Smart Money Buy Signal")
            if signal_data.get('near_bottom'):
                sm_score += self.weights['smart_money'] * 0.4
                sm_reasons.append("Near Range Bottom")
        else:  # SELL
            if signal_data.get('smart_money_sell'):
                sm_score += self.weights['smart_money'] * 0.6
                sm_reasons.append("Smart Money Sell Signal")
            if signal_data.get('near_top'):
                sm_score += self.weights['smart_money'] * 0.4
                sm_reasons.append("Near Range Top")
        
        if sm_score > 0:
            reasons.extend(sm_reasons)
        score += min(sm_score, self.weights['smart_money'])
        breakdown['smart_money'] = min(sm_score, self.weights['smart_money'])
        
        # 3. RSI Divergence
        rsi_score = 0
        if signal_data.get('rsi_bull_div') and signal_type == 'BUY':
            rsi_score = self.weights['rsi_divergence']
            reasons.append("RSI Bullish Divergence")
        elif signal_data.get('rsi_bear_div') and signal_type == 'SELL':
            rsi_score = self.weights['rsi_divergence']
            reasons.append("RSI Bearish Divergence")
        score += rsi_score
        breakdown['rsi_divergence'] = rsi_score
        
        # 4. Ichimoku Cloud
        ichimoku_score = 0
        if signal_data.get('ichimoku_buy') and signal_type == 'BUY':
            ichimoku_score = self.weights['ichimoku']
            reasons.append("Ichimoku Cloud Bullish")
        elif signal_data.get('ichimoku_sell') and signal_type == 'SELL':
            ichimoku_score = self.weights['ichimoku']
            reasons.append("Ichimoku Cloud Bearish")
        score += ichimoku_score
        breakdown['ichimoku'] = ichimoku_score
        
        # 5. Volume Confirmation
        volume_score = 0
        if not df.empty:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                
                if volume_ratio > self.config['volume_threshold']:
                    volume_score = self.weights['volume_confirmation']
                    reasons.append(f"Volume Spike ({volume_ratio:.1f}x)")
                elif volume_ratio > 1.0:
                    volume_score = self.weights['volume_confirmation'] * 0.5
                    reasons.append("Above Average Volume")
                else:
                    warnings.append(f"Low volume ({volume_ratio:.1f}x avg)")
        
        score += volume_score
        breakdown['volume'] = volume_score
        
        # 6. Market Structure Alignment
        structure_score = 0
        
        if market_data:
            trend = market_data.get('trend', 'neutral')
            
            if (signal_type == 'BUY' and trend == 'uptrend') or \
               (signal_type == 'SELL' and trend == 'downtrend'):
                structure_score = self.weights['market_structure']
                reasons.append(f"Aligned with {trend} trend")
            elif trend == 'neutral':
                structure_score = self.weights['market_structure'] * 0.5
                reasons.append("Neutral market structure")
            else:
                warnings.append(f"Against {trend} trend")
        
        score += structure_score
        breakdown['market_structure'] = structure_score
        
        # 7. Support/Resistance Levels
        sr_score = 0
        
        if market_data:
            sr_levels = market_data.get('support_resistance', {})
            
            if signal_type == 'BUY':
                nearest_support = sr_levels.get('nearest_support')
                if nearest_support:
                    distance_pct = abs(current_price - nearest_support['price']) / current_price * 100
                    if distance_pct < 3:  # کمتر از ۳٪ فاصله
                        sr_score = self.weights['support_resistance']
                        reasons.append(f"Near support level ({distance_pct:.1f}%)")
            
            elif signal_type == 'SELL':
                nearest_resistance = sr_levels.get('nearest_resistance')
                if nearest_resistance:
                    distance_pct = abs(current_price - nearest_resistance['price']) / current_price * 100
                    if distance_pct < 3:  # کمتر از ۳٪ فاصله
                        sr_score = self.weights['support_resistance']
                        reasons.append(f"Near resistance level ({distance_pct:.1f}%)")
        
        score += sr_score
        breakdown['support_resistance'] = sr_score
        
        # Normalize score to percentage
        score_percent = (score / max_score) * 100 if max_score > 0 else 0
        
        # اعمال تنبیه برای warnings
        if warnings:
            warning_penalty = len(warnings) * 2  # 2% penalty per warning
            score_percent = max(0, score_percent - warning_penalty)
        
        # بررسی حداقل اطمینان
        if score_percent < self.config['min_confidence']:
            warnings.append(f"Below minimum confidence ({self.config['min_confidence']}%)")
        
        # محاسبه grade
        grade, grade_color = self._get_grade(score_percent)
        
        return {
            'score': round(score_percent, 1),
            'grade': grade,
            'grade_color': grade_color,
            'raw_score': score,
            'max_score': max_score,
            'breakdown': breakdown,
            'reasons': reasons,
            'warnings': warnings,
            'signal_type': signal_type,
            'is_valid': score_percent >= self.config['min_confidence'],
            'validation_message': f"{'Valid' if score_percent >= self.config['min_confidence'] else 'Invalid'} signal",
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _get_grade(self, score: float) -> Tuple[str, str]:
        """درجه‌بندی سیگنال"""
        if score >= 90:
            return "A+ (Excellent)", "green"
        elif score >= 80:
            return "A (Very Good)", "green"
        elif score >= 70:
            return "B (Good)", "blue"
        elif score >= 60:
            return "C (Fair)", "yellow"
        elif score >= 50:
            return "D (Weak)", "orange"
        else:
            return "F (Poor)", "red"

# ============================================
# 🔄 Cache Management
# ============================================

class DataCache:
    """
    سیستم مدیریت کش پیشرفته
    """
    
    def __init__(self, cache_dir: str = ".cache", max_size_mb: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache_index_file = self.cache_dir / "index.json"
        self.cache_index = self._load_index()
    
    def _load_index(self) -> Dict:
        """لود index کش"""
        try:
            if self.cache_index_file.exists():
                with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            pass
        return {}
    
    def _save_index(self):
        """ذخیره index کش"""
        try:
            with open(self.cache_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving cache index: {e}")
    
    def _get_cache_path(self, key: str) -> Path:
        """دریافت مسیر فایل کش"""
        # ایجاد hash از key برای نام فایل
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str, max_age_seconds: int = 300, default=None) -> Any:
        """
        دریافت از کش با بررسی عمر
        
        Args:
            key: کلید کش
            max_age_seconds: حداکثر عمر مجاز
            default: مقدار پیش‌فرض
        
        Returns:
            Any: داده کش شده یا default
        """
        try:
            cache_path = self._get_cache_path(key)
            
            if not cache_path.exists():
                return default
            
            # بررسی عمر فایل
            file_age = time.time() - cache_path.stat().st_mtime
            if file_age > max_age_seconds:
                cache_path.unlink(missing_ok=True)
                
                # حذف از index
                if key in self.cache_index:
                    del self.cache_index[key]
                    self._save_index()
                
                return default
            
            # خواندن داده
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            # آپدیت last accessed
            self.cache_index[key] = {
                'last_accessed': time.time(),
                'size': cache_path.stat().st_size,
                'created': cache_path.stat().st_ctime
            }
            self._save_index()
            
            return data
            
        except Exception as e:
            logging.debug(f"Cache get error for key {key}: {e}")
            return default
    
    def set(self, key: str, data: Any, compress: bool = False):
        """
        ذخیره در کش
        
        Args:
            key: کلید کش
            data: داده برای ذخیره
            compress: فشرده‌سازی
        """
        try:
            cache_path = self._get_cache_path(key)
            
            # بررسی اندازه کش
            self._enforce_size_limit()
            
            # ذخیره داده
            mode = 'wb'
            if compress:
                # فشرده‌سازی با gzip
                cache_path = cache_path.with_suffix('.cache.gz')
                open_func = gzip.open
            else:
                open_func = open
            
            with open_func(cache_path, mode) as f:
                pickle.dump(data, f)
            
            # آپدیت index
            self.cache_index[key] = {
                'last_accessed': time.time(),
                'size': cache_path.stat().st_size,
                'created': cache_path.stat().st_ctime,
                'compressed': compress
            }
            self._save_index()
            
        except Exception as e:
            logging.error(f"Cache set error for key {key}: {e}")
    
    def delete(self, key: str):
        """حذف از کش"""
        try:
            cache_path = self._get_cache_path(key)
            cache_path.unlink(missing_ok=True)
            
            # همچنین فایل فشرده شده را حذف کن
            gz_path = cache_path.with_suffix('.cache.gz')
            gz_path.unlink(missing_ok=True)
            
            if key in self.cache_index:
                del self.cache_index[key]
                self._save_index()
                
        except Exception as e:
            logging.error(f"Cache delete error for key {key}: {e}")
    
    def clear(self, pattern: str = None):
        """
        پاکسازی کش
        
        Args:
            pattern: pattern برای فیلتر کردن (مثل 'ohlcv_*')
        """
        try:
            if pattern:
                # پاکسازی انتخابی
                for key in list(self.cache_index.keys()):
                    if pattern in key:
                        self.delete(key)
            else:
                # پاکسازی کامل
                for cache_file in self.cache_dir.glob("*.cache*"):
                    cache_file.unlink(missing_ok=True)
                self.cache_index = {}
                self._save_index()
                logging.info("🧹 Cache cleared completely")
                
        except Exception as e:
            logging.error(f"Cache clear error: {e}")
    
    def _enforce_size_limit(self):
        """اجرای محدودیت اندازه کش"""
        try:
            # محاسبه اندازه کل
            total_size = sum(info.get('size', 0) for info in self.cache_index.values())
            
            if total_size > self.max_size_bytes:
                logging.info(f"Cache size limit reached: {total_size / (1024*1024):.1f}MB > {self.max_size_bytes / (1024*1024):.1f}MB")
                
                # مرتب کردن بر اساس last accessed (قدیمی‌ترین اول)
                sorted_keys = sorted(
                    self.cache_index.keys(),
                    key=lambda k: self.cache_index[k].get('last_accessed', 0)
                )
                
                # حذف قدیمی‌ترین‌ها تا زمانی که زیر حد باشیم
                removed = 0
                for key in sorted_keys:
                    if total_size <= self.max_size_bytes * 0.8:  # تا 80% حد
                        break
                    
                    self.delete(key)
                    removed += 1
                    
                    # محاسبه مجدد اندازه
                    total_size = sum(info.get('size', 0) for info in self.cache_index.values())
                
                if removed > 0:
                    logging.info(f"Removed {removed} old cache entries")
                    
        except Exception as e:
            logging.error(f"Error enforcing cache size limit: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """دریافت آمار کش"""
        total_size = sum(info.get('size', 0) for info in self.cache_index.values())
        avg_age = 0
        
        if self.cache_index:
            now = time.time()
            ages = [now - info.get('last_accessed', now) for info in self.cache_index.values()]
            avg_age = np.mean(ages)
        
        return {
            'total_entries': len(self.cache_index),
            'total_size_mb': total_size / (1024 * 1024),
            'avg_age_seconds': avg_age,
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'usage_percent': (total_size / self.max_size_bytes * 100) if self.max_size_bytes > 0 else 0,
            'compressed_entries': sum(1 for info in self.cache_index.values() if info.get('compressed', False))
        }

# ============================================
# 🎯 Main Utility Functions
# ============================================

def initialize_utils(config: Dict = None) -> Dict[str, Any]:
    """
    راه‌اندازی اولیه تمام utilities
    
    Args:
        config: تنظیمات
        
    Returns:
        Dict: آبجکت‌های utility
    """
    # تنظیمات پیش‌فرض
    default_config = {
        'log_level': 'INFO',
        'log_to_file': False,
        'cache_enabled': True,
        'performance_tracking': True
    }
    
    if config:
        default_config.update(config.get('system', {}))
    
    # ایجاد logger
    logger = setup_logger(
        name="fast_scalp_utils",
        level=LogLevel[default_config['log_level']],
        log_to_file=default_config['log_to_file']
    )
    
    # ایجاد performance tracker
    performance_tracker = None
    if default_config['performance_tracking']:
        try:
            performance_tracker = PerformanceTracker()
            logger.info("📊 Performance tracker initialized")
        except Exception as e:
            logger.warning(f"Could not initialize performance tracker: {e}")
    
    # ایجاد cache manager
    cache_manager = None
    if default_config['cache_enabled']:
        try:
            cache_manager = DataCache(max_size_mb=50)
            logger.info(f"💾 Cache manager initialized ({cache_manager.get_stats()['total_entries']} entries)")
        except Exception as e:
            logger.warning(f"Could not initialize cache manager: {e}")
    
    # ایجاد signal scorer
    signal_scorer = SignalScorer()
    logger.info("📈 Signal scorer initialized")
    
    return {
        'logger': logger,
        'performance_tracker': performance_tracker,
        'cache_manager': cache_manager,
        'signal_scorer': signal_scorer,
        'config': default_config,
        'initialized_at': datetime.utcnow().isoformat()
    }

def cleanup_resources(utils_dict: Dict[str, Any]):
    """
    پاکسازی منابع utilities
    """
    logger = utils_dict.get('logger')
    
    if logger:
        logger.info("🧹 Cleaning up resources...")
    
    # پاکسازی cache
    cache_manager = utils_dict.get('cache_manager')
    if cache_manager:
        try:
            cache_manager.clear('temp_')
            logger.info("💾 Temporary cache cleared")
        except:
            pass
    
    # پاکسازی performance tracker
    performance_tracker = utils_dict.get('performance_tracker')
    if performance_tracker:
        try:
            performance_tracker.cleanup_old_signals(30)  # سیگنال‌های قدیمی‌تر از ۳۰ روز
            logger.info("📊 Old signals cleaned up")
        except:
            pass

# ============================================
# 🚀 Quick Utilities
# ============================================

def quick_validate_config(config: Dict) -> bool:
    """اعتبارسنجی سریع config"""
    has_token = bool(config.get('telegram_token') or config.get('telegram', {}).get('token'))
    has_chat_id = bool(config.get('chat_id') or config.get('telegram', {}).get('chat_id'))
    return has_token and has_chat_id

def get_memory_usage() -> Dict[str, float]:
    """دریافت استفاده از حافظه"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
            'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }
    except ImportError:
        return {'error': 'psutil not installed'}

def generate_report(data: Dict, title: str = "Report") -> str:
    """ایجاد گزارش متناسب"""
    lines = [f"📋 {title}", "=" * 40]
    
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"\n{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {sub_value}")
        elif isinstance(value, list):
            lines.append(f"\n{key}:")
            for i, item in enumerate(value[:5], 1):  # فقط ۵ مورد اول
                lines.append(f"  {i}. {item}")
            if len(value) > 5:
                lines.append(f"  ... and {len(value) - 5} more")
        else:
            lines.append(f"{key}: {value}")
    
    lines.append("\n" + "=" * 40)
    return "\n".join(lines)

# ============================================
# 🎬 Export مهم‌ترین توابع
# ============================================

__all__ = [
    # Logger
    'setup_logger', 'log_performance', 'LogLevel',
    
    # Data
    'validate_dataframe', 'clean_ohlcv_data', 'calculate_volume_profile',
    
    # Technical Analysis
    'calculate_support_resistance', 'calculate_market_structure',
    'get_market_phase', 'calculate_volatility',
    
    # Performance Tracking
    'PerformanceTracker', 'SignalRecord',
    
    # Security
    'validate_api_keys', 'sanitize_output',
    
    # Formatting
    'format_price', 'format_percentage', 'format_timestamp',
    'format_signal_for_display', 'format_telegram_message',
    
    # Time
    'is_market_hours', 'next_scan_time', 'calculate_time_until',
    'get_market_session',
    
    # Signal Scoring
    'SignalScorer',
    
    # Cache
    'DataCache',
    
    # Main
    'initialize_utils', 'cleanup_resources',
    
    # Quick
    'quick_validate_config', 'get_memory_usage', 'generate_report'
]

# ============================================
# 🧪 Test
# ============================================

if __name__ == "__main__":
    print("🧪 Testing utilities...")
    
    # تست توابع اصلی
    test_data = {
        'test_price': 12345.6789,
        'test_percentage': 5.1234,
        'test_timestamp': datetime.utcnow(),
        'test_config': {
            'telegram_token': 'secret_token_123',
            'chat_id': '123456789',
            'api_key': 'test_key'
        }
    }
    
    print("\n📊 Formatting Tests:")
    print(f"Price: {format_price(test_data['test_price'])}")
    print(f"Percentage: {format_percentage(test_data['test_percentage'])}")
    print(f"Timestamp: {format_timestamp(test_data['test_timestamp'])}")
    
    print("\n🔐 Sanitization Test:")
    sanitized = sanitize_output(test_data['test_config'])
    print(f"Sanitized config: {sanitized}")
    
    print("\n✅ Utilities test completed!")
