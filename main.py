"""
Crypto AI Trading System v8.0 - Professional Version
Real Analysis with Ichimoku + QM Pattern Detection
"""

import os
import time
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Import Utils Module
# ==============================================================================

try:
    import utils
    logger.info("✅ Utils module imported successfully")
    
    # Check essential functions
    essential_functions = [
        'get_market_data_with_fallback',
        'calculate_ichimoku_components',
        'analyze_ichimoku_scalp_signal',
        'calculate_simple_rsi',
        'calculate_simple_sma',
        'calculate_smart_entry'
    ]
    
    missing_functions = []
    for func in essential_functions:
        if not hasattr(utils, func):
            missing_functions.append(func)
    
    if missing_functions:
        logger.warning(f"⚠️ Missing functions in utils: {missing_functions}")
        
except ImportError as e:
    logger.error(f"❌ Failed to import utils module: {e}")
    raise ImportError("Utils module is required. Please ensure utils.py exists in the same directory.")

# ==============================================================================
# FastAPI App Configuration
# ==============================================================================

API_VERSION = "8.0-PRO"

app = FastAPI(
    title=f"Crypto AI Trading System v{API_VERSION}",
    description="Professional Trading System with Ichimoku + QM Pattern Detection",
    version=API_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# Pydantic Models
# ==============================================================================

class AnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class ScalpRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class IchimokuRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class ScanRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"

class SignalResponse(BaseModel):
    status: str
    count: int
    last_updated: str
    signals: List[Dict[str, Any]]

# ==============================================================================
# Technical Analysis Functions
# ==============================================================================

def detect_qm_structure(data):
    """تشخیص الگوی QM (Quasimodo) و شکست ساختار (BOS)"""
    try:
        if not data or len(data) < 30:  # افزایش به 30 کندل برای تحلیل بهتر
            return "NEUTRAL"
        
        # استخراج داده‌های قیمتی
        highs = [float(c[2]) for c in data[-30:]]
        lows = [float(c[3]) for c in data[-30:]]
        closes = [float(c[4]) for c in data[-30:]]
        
        current_close = closes[-1]
        current_high = highs[-1]
        current_low = lows[-1]
        
        # ========== شناسایی الگوی QM (Quasimodo) ==========
        
        # QM خرید: ساختار High -> Low -> Higher High -> Lower Low
        # بررسی 10 کندل اخیر برای شناسایی الگو
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        
        # شرط 1: یک High قابل توجه داشته باشیم
        if len(recent_highs) >= 5:
            # پیدا کردن High اصلی
            main_high = max(recent_highs[:-3])  # High در 7 کندل قبل
            
            # شرط 2: افت قیمت پس از High
            if current_low < min(recent_lows[-5:-2]):
                # شرط 3: تشکیل Higher Low
                if len(recent_lows) >= 8:
                    first_low = min(recent_lows[-8:-5])
                    second_low = min(recent_lows[-4:-1])
                    
                    if second_low > first_low:  # Higher Low تشکیل شده
                        # شرط 4: قیمت شروع به بازگشت کند
                        if current_close > recent_lows[-2]:
                            return "QM_POTENTIAL_BUY"
        
        # QM فروش: ساختار Low -> High -> Lower Low -> Higher High
        if len(recent_lows) >= 5:
            # پیدا کردن Low اصلی
            main_low = min(recent_lows[:-3])
            
            # شرط 2: رشد قیمت پس از Low
            if current_high > max(recent_highs[-5:-2]):
                # شرط 3: تشکیل Lower High
                if len(recent_highs) >= 8:
                    first_high = max(recent_highs[-8:-5])
                    second_high = max(recent_highs[-4:-1])
                    
                    if second_high < first_high:  # Lower High تشکیل شده
                        # شرط 4: قیمت شروع به افت کند
                        if current_close < recent_highs[-2]:
                            return "QM_POTENTIAL_SELL"
        
        # ========== شناسایی شکست ساختار (BOS) ==========
        
        # BOS صعودی: شکست High قبلی
        if len(highs) >= 15:
            previous_high = max(highs[-15:-5])  # High در بازه 15-5 کندل قبل
            if current_close > previous_high and current_high > previous_high:
                # تأیید با حجم (اگر داده حجم موجود باشد)
                return "BULLISH_BOS"
        
        # BOS نزولی: شکست Low قبلی
        if len(lows) >= 15:
            previous_low = min(lows[-15:-5])  # Low در بازه 15-5 کندل قبل
            if current_close < previous_low and current_low < previous_low:
                return "BEARISH_BOS"
        
        # ========== شناسایی رنج بازار ==========
        
        # اگر بازار در رنج باشد
        high_20 = max(highs[-20:]) if len(highs) >= 20 else max(highs)
        low_20 = min(lows[-20:]) if len(lows) >= 20 else min(lows)
        range_percentage = ((high_20 - low_20) / low_20) * 100
        
        if range_percentage < 1.5:  # بازار رنج (کمتر از 1.5% نوسان)
            if current_close > (high_20 + low_20) / 2:
                return "RANGE_BREAKOUT_UP"
            elif current_close < (high_20 + low_20) / 2:
                return "RANGE_BREAKOUT_DOWN"
            else:
                return "MARKET_RANGE"
        
        return "NEUTRAL"
        
    except Exception as e:
        logger.error(f"Error in QM detection: {e}")
        return "NEUTRAL"

def calculate_targets_stoploss(entry_price: float, signal: str, confidence: float = 0.5):
    """محاسبه تارگت‌ها و استاپ لاس بر اساس سیگنال و اطمینان"""
    if entry_price <= 0:
        return [0, 0, 0], 0, [0, 0, 0], 0
    
    # تنظیم پارامترهای ریسک بر اساس اطمینان
    risk_multiplier = min(confidence * 2, 1.5)  # حداکثر 1.5 برابر
    
    if signal in ["BUY", "BULLISH_BOS", "QM_POTENTIAL_BUY", "RANGE_BREAKOUT_UP"]:
        # برای سیگنال‌های خرید
        targets = [
            round(entry_price * (1 + 0.008 * risk_multiplier), 8),  # 0.8%
            round(entry_price * (1 + 0.015 * risk_multiplier), 8),  # 1.5%
            round(entry_price * (1 + 0.025 * risk_multiplier), 8)   # 2.5%
        ]
        stop_loss = round(entry_price * (1 - 0.010 * risk_multiplier), 8)  # -1.0%
    
    elif signal in ["SELL", "BEARISH_BOS", "QM_POTENTIAL_SELL", "RANGE_BREAKOUT_DOWN"]:
        # برای سیگنال‌های فروش
        targets = [
            round(entry_price * (1 - 0.008 * risk_multiplier), 8),  # -0.8%
            round(entry_price * (1 - 0.015 * risk_multiplier), 8),  # -1.5%
            round(entry_price * (1 - 0.025 * risk_multiplier), 8)   # -2.5%
        ]
        stop_loss = round(entry_price * (1 + 0.010 * risk_multiplier), 8)  # +1.0%
    
    else:  # HOLD یا NEUTRAL
        targets = [
            round(entry_price * 1.005, 8),
            round(entry_price * 1.010, 8),
            round(entry_price * 1.015, 8)
        ]
        stop_loss = round(entry_price * 0.990, 8)
    
    # محاسبه درصدها
    targets_percent = [
        round(((target - entry_price) / entry_price) * 100, 2)
        for target in targets
    ]
    stop_loss_percent = round(((stop_loss - entry_price) / entry_price) * 100, 2)
    
    return targets, stop_loss, targets_percent, stop_loss_percent

def determine_final_signal(ichimoku_signal: str, structure_signal: str) -> str:
    """تعیین سیگنال نهایی بر اساس ترکیب ایچیموکو و ساختار"""
    
    signal_mapping = {
        "BULLISH_BOS": "BUY",
        "QM_POTENTIAL_BUY": "BUY",
        "RANGE_BREAKOUT_UP": "BUY",
        "BEARISH_BOS": "SELL",
        "QM_POTENTIAL_SELL": "SELL",
        "RANGE_BREAKOUT_DOWN": "SELL"
    }
    
    # تبدیل سیگنال ساختار به BUY/SELL
    structure_simple = signal_mapping.get(structure_signal, "NEUTRAL")
    
    # منطق ترکیب سیگنال‌ها
    if ichimoku_signal == "BUY" and structure_simple == "BUY":
        return "STRONG_BUY"
    elif ichimoku_signal == "SELL" and structure_simple == "SELL":
        return "STRONG_SELL"
    elif ichimoku_signal == "BUY" and structure_simple == "NEUTRAL":
        return "WEAK_BUY"
    elif ichimoku_signal == "SELL" and structure_simple == "NEUTRAL":
        return "WEAK_SELL"
    elif ichimoku_signal == "HOLD" and structure_simple == "BUY":
        return "BUY"
    elif ichimoku_signal == "HOLD" and structure_simple == "SELL":
        return "SELL"
    else:
        return "HOLD"

def calculate_confidence(ichimoku_signal: Dict, structure_signal: str) -> float:
    """محاسبه اطمینان سیگنال بر اساس عوامل مختلف"""
    try:
        confidence = 0.5  # مقدار پایه
        
        # اطمینان از ایچیموکو
        ichimoku_conf = ichimoku_signal.get('confidence', 0.5)
        confidence += (ichimoku_conf - 0.5) * 0.4  # 40% وزن
        
        # اطمینان از ساختار
        if structure_signal in ["BULLISH_BOS", "BEARISH_BOS"]:
            confidence += 0.3  # 30% برای BOS
        elif structure_signal in ["QM_POTENTIAL_BUY", "QM_POTENTIAL_SELL"]:
            confidence += 0.2  # 20% برای QM
        elif structure_signal in ["RANGE_BREAKOUT_UP", "RANGE_BREAKOUT_DOWN"]:
            confidence += 0.15  # 15% برای برک‌اوت رنج
        
        # محدود کردن بین 0.1 تا 0.95
        confidence = max(0.1, min(0.95, confidence))
        
        return round(confidence, 2)
        
    except Exception as e:
        logger.error(f"Error calculating confidence: {e}")
        return 0.5

# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/")
async def read_root():
    return {
        "message": f"Crypto AI Trading System v{API_VERSION}",
        "status": "Active",
        "version": API_VERSION,
        "description": "Professional Trading System with Ichimoku + QM Pattern Detection",
        "author": "Crypto AI Team",
        "endpoints": {
            "/api/health": "Health check",
            "/api/analyze": "Complete analysis (POST)",
            "/api/scalp-signal": "Scalp signal (POST)",
            "/api/ichimoku-scalp": "Ichimoku signal (POST)",
            "/api/scan-all": "Multi-timeframe scan (GET)",
            "/market/{symbol}": "Market data (GET)"
        }
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "Healthy",
        "version": API_VERSION,
        "timestamp": datetime.now().isoformat(),
        "system": {
            "python_version": sys.version,
            "platform": sys.platform,
            "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "analysis": {
            "qm_detection": True,
            "ichimoku_analysis": True,
            "real_time_data": True
        }
    }

@app.post("/api/analyze")
async def analyze_crypto(request: AnalysisRequest):
    """آنالیز کامل ارز با ترکیب ایچیموکو و الگوی QM"""
    try:
        logger.info(f"🔍 Analysis request: {request.symbol} ({request.timeframe})")
        start_time = time.time()
        
        # ۱. دریافت داده بازار
        market_data = utils.get_market_data_with_fallback(
            request.symbol, 
            request.timeframe, 
            100
        )
        
        if not market_data or len(market_data) < 30:
            raise HTTPException(
                status_code=500, 
                detail="دیتای بازار کافی نیست. حداقل 30 کندل نیاز است."
            )
        
        logger.info(f"📊 دریافت {len(market_data)} کندل برای {request.symbol}")
        
        # ۲. تحلیل ایچیموکو
        ichimoku_data = utils.calculate_ichimoku_components(market_data)
        ichimoku_signal = utils.analyze_ichimoku_scalp_signal(ichimoku_data)
        
        # ۳. تشخیص الگوی QM و ساختار
        structure_signal = detect_qm_structure(market_data)
        
        # ۴. تعیین سیگنال نهایی
        final_signal = determine_final_signal(
            ichimoku_signal['signal'], 
            structure_signal
        )
        
        # ۵. محاسبه اطمینان
        confidence = calculate_confidence(ichimoku_signal, structure_signal)
        
        # ۶. محاسبه قیمت ورود هوشمند
        smart_entry = utils.calculate_smart_entry(market_data, final_signal)
        current_price = ichimoku_data.get('current_price', 0)
        entry_price = smart_entry if smart_entry > 0 else current_price
        
        # ۷. محاسبه تارگت‌ها و استاپ لاس
        targets, stop_loss, targets_percent, stop_loss_percent = calculate_targets_stoploss(
            entry_price, final_signal, confidence
        )
        
        # ۸. محاسبه اندیکاتورهای تکمیلی
        rsi = utils.calculate_simple_rsi(market_data, 14)
        sma_20 = utils.calculate_simple_sma(market_data, 20)
        
        # ۹. تحلیل نوسان
        volatility = utils.calculate_volatility(market_data) if hasattr(utils, 'calculate_volatility') else 0
        
        # ۱۰. تشخیص واگرایی
        closes = [float(c[4]) for c in market_data]
        rsi_values = utils.calculate_rsi_series(closes, 14) if hasattr(utils, 'calculate_rsi_series') else []
        divergence = utils.detect_divergence(closes, rsi_values, 5) if hasattr(utils, 'detect_divergence') else {"detected": False}
        
        # ساخت پاسخ
        response = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "analysis": {
                "final_signal": final_signal,
                "confidence": f"{confidence * 100:.1f}%",
                "ichimoku_signal": ichimoku_signal['signal'],
                "structure_signal": structure_signal,
                "reason": f"ایچیموکو: {ichimoku_signal['reason']} | ساختار: {structure_signal}"
            },
            "price": {
                "current": round(current_price, 8),
                "entry": round(entry_price, 8),
                "smart_entry": round(smart_entry, 8) if smart_entry > 0 else None
            },
            "targets": {
                "levels": targets,
                "percentages": targets_percent,
                "stop_loss": stop_loss,
                "stop_loss_percent": stop_loss_percent
            },
            "indicators": {
                "rsi": round(rsi, 2),
                "sma_20": round(sma_20, 2),
                "volatility": round(volatility, 2) if volatility else None,
                "divergence": divergence['detected'] if divergence else False
            },
            "ichimoku_levels": {
                "tenkan_sen": round(ichimoku_data.get('tenkan_sen', 0), 8),
                "kijun_sen": round(ichimoku_data.get('kijun_sen', 0), 8),
                "cloud_top": round(ichimoku_data.get('cloud_top', 0), 8),
                "cloud_bottom": round(ichimoku_data.get('cloud_bottom', 0), 8),
                "cloud_position": "بالای ابر" if current_price > ichimoku_data.get('cloud_top', 0) 
                                 else "درون ابر" if ichimoku_data.get('cloud_bottom', 0) <= current_price <= ichimoku_data.get('cloud_top', 0)
                                 else "زیر ابر"
            },
            "metadata": {
                "strategy": "ایچیموکو + QM + ساختارشکنی",
                "data_points": len(market_data),
                "generated_at": datetime.now().isoformat(),
                "processing_time": f"{round((time.time() - start_time) * 1000, 2)}ms",
                "version": API_VERSION
            }
        }
        
        logger.info(f"✅ تحلیل کامل شد: سیگنال {final_signal} با اطمینان {confidence*100:.1f}%")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ خطا در تحلیل: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"خطا در تحلیل: {str(e)[:200]}"
        )

@app.post("/api/scalp-signal")
async def get_scalp_signal(request: ScalpRequest):
    """سیگنال اسکلپ برای معاملات کوتاه‌مدت"""
    allowed_timeframes = ["1m", "5m", "15m"]
    if request.timeframe not in allowed_timeframes:
        raise HTTPException(
            status_code=400, 
            detail=f"تایم‌فریم نامعتبر. مجاز: {allowed_timeframes}"
        )
    
    try:
        logger.info(f"⚡ Scalp request: {request.symbol} ({request.timeframe})")
        start_time = time.time()
        
        # دریافت داده برای اسکلپ
        market_data = utils.get_market_data_with_fallback(
            request.symbol,
            request.timeframe,
            50  # کمتر برای اسکلپ
        )
        
        if not market_data or len(market_data) < 20:
            raise HTTPException(
                status_code=500,
                detail="داده کافی برای اسکلپ وجود ندارد"
            )
        
        # تحلیل اسکلپ
        scalp_analysis = utils.analyze_scalp_conditions(market_data, request.timeframe)
        
        # تشخیص ساختار
        structure_signal = detect_qm_structure(market_data)
        
        # ترکیب سیگنال‌ها برای اسکلپ
        final_signal = "HOLD"
        if scalp_analysis["condition"] == "BULLISH" and structure_signal in ["QM_POTENTIAL_BUY", "BULLISH_BOS"]:
            final_signal = "BUY"
        elif scalp_analysis["condition"] == "BEARISH" and structure_signal in ["QM_POTENTIAL_SELL", "BEARISH_BOS"]:
            final_signal = "SELL"
        
        # محاسبه قیمت‌ها
        current_price = scalp_analysis.get("current_price", 0)
        smart_entry = utils.calculate_smart_entry(market_data, final_signal)
        entry_price = smart_entry if smart_entry > 0 else current_price
        
        # محاسبه تارگت‌های تنگ‌تر برای اسکلپ
        targets = []
        if final_signal == "BUY":
            targets = [
                round(entry_price * 1.003, 8),  # +0.3%
                round(entry_price * 1.006, 8),  # +0.6%
                round(entry_price * 1.009, 8)   # +0.9%
            ]
            stop_loss = round(entry_price * 0.998, 8)
        elif final_signal == "SELL":
            targets = [
                round(entry_price * 0.997, 8),  # -0.3%
                round(entry_price * 0.994, 8),  # -0.6%
                round(entry_price * 0.991, 8)   # -0.9%
            ]
            stop_loss = round(entry_price * 1.002, 8)
        else:
            targets = [current_price, current_price, current_price]
            stop_loss = current_price
        
        # محاسبه درصدها
        targets_percent = [
            round(((target - entry_price) / entry_price) * 100, 2)
            for target in targets
        ]
        stop_loss_percent = round(((stop_loss - entry_price) / entry_price) * 100, 2)
        
        response = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": final_signal,
            "entry_price": round(entry_price, 8),
            "current_price": round(current_price, 8),
            "rsi": scalp_analysis.get("rsi", 50),
            "condition": scalp_analysis.get("condition", "NEUTRAL"),
            "structure": structure_signal,
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": targets_percent,
            "stop_loss_percent": stop_loss_percent,
            "volatility": scalp_analysis.get("volatility", 0),
            "reason": f"شرایط: {scalp_analysis.get('reason', '')} | ساختار: {structure_signal}",
            "type": "SCALP",
            "generated_at": datetime.now().isoformat(),
            "processing_time": f"{round((time.time() - start_time) * 1000, 2)}ms"
        }
        
        logger.info(f"✅ سیگنال اسکلپ: {final_signal} برای {request.symbol}")
        
        return response
        
    except Exception as e:
        logger.error(f"خطا در اسکلپ: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"خطای اسکلپ: {str(e)[:200]}"
        )

@app.post("/api/ichimoku-scalp")
async def get_ichimoku_scalp(request: IchimokuRequest):
    """سیگنال اسکلپ مبتنی بر ایچیموکو"""
    try:
        logger.info(f"☁️ Ichimoku scalp: {request.symbol} ({request.timeframe})")
        start_time = time.time()
        
        market_data = utils.get_market_data_with_fallback(
            request.symbol,
            request.timeframe,
            100
        )
        
        if not market_data or len(market_data) < 52:
            raise HTTPException(
                status_code=500,
                detail="داده کافی برای ایچیموکو نیست"
            )
        
        # تحلیل ایچیموکو
        ichimoku_data = utils.calculate_ichimoku_components(market_data)
        ichimoku_signal = utils.analyze_ichimoku_scalp_signal(ichimoku_data)
        
        # سیگنال ایچیموکو مخصوص اسکلپ
        ichimoku_scalp_signal = utils.get_ichimoku_scalp_signal(market_data, request.timeframe)
        
        # تعیین سیگنال نهایی
        final_signal = ichimoku_signal['signal']
        if ichimoku_scalp_signal and 'signal' in ichimoku_scalp_signal:
            final_signal = ichimoku_scalp_signal['signal']
        
        # قیمت‌ها
        current_price = ichimoku_data.get('current_price', 0)
        smart_entry = utils.calculate_smart_entry(market_data, final_signal)
        entry_price = smart_entry if smart_entry > 0 else current_price
        
        # تارگت‌ها
        targets, stop_loss, targets_percent, stop_loss_percent = calculate_targets_stoploss(
            entry_price, final_signal, ichimoku_signal.get('confidence', 0.5)
        )
        
        # سطوح سووینگ
        swing_high, swing_low = utils.get_swing_high_low(market_data) if hasattr(utils, 'get_swing_high_low') else (0, 0)
        
        response = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "signal": final_signal,
            "confidence": ichimoku_signal.get('confidence', 0.5),
            "current_price": round(current_price, 8),
            "entry_price": round(entry_price, 8),
            "targets": targets,
            "stop_loss": stop_loss,
            "targets_percent": targets_percent,
            "stop_loss_percent": stop_loss_percent,
            "ichimoku": {
                "tenkan_sen": round(ichimoku_data.get('tenkan_sen', 0), 8),
                "kijun_sen": round(ichimoku_data.get('kijun_sen', 0), 8),
                "cloud_top": round(ichimoku_data.get('cloud_top', 0), 8),
                "cloud_bottom": round(ichimoku_data.get('cloud_bottom', 0), 8),
                "position": "درون ابر" if ichimoku_data.get('cloud_bottom', 0) <= current_price <= ichimoku_data.get('cloud_top', 0)
                            else "بالای ابر" if current_price > ichimoku_data.get('cloud_top', 0)
                            else "زیر ابر"
            },
            "swing_levels": {
                "high": round(swing_high, 8),
                "low": round(swing_low, 8)
            },
            "trend_power": ichimoku_data.get('trend_power', 50),
            "reason": ichimoku_signal.get('reason', 'تحلیل ایچیموکو'),
            "type": "ICHIMOKU_SCALP",
            "generated_at": datetime.now().isoformat(),
            "processing_time": f"{round((time.time() - start_time) * 1000, 2)}ms"
        }
        
        logger.info(f"✅ ایچیموکو اسکلپ: {final_signal}")
        
        return response
        
    except Exception as e:
        logger.error(f"خطا در ایچیموکو اسکلپ: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"خطای ایچیموکو: {str(e)[:200]}"
        )

@app.get("/api/scan-all/{symbol}")
async def scan_all_timeframes(symbol: str):
    """اسکن همه تایم‌فریم‌ها"""
    try:
        timeframes = ["1m", "5m", "15m", "1h", "4h"]
        results = []
        
        logger.info(f"🔍 Scanning all timeframes for {symbol}")
        
        for tf in timeframes:
            try:
                # انتخاب اندپوینت مناسب بر اساس تایم‌فریم
                if tf in ["1m", "5m", "15m"]:
                    request = ScalpRequest(symbol=symbol, timeframe=tf)
                    response = await get_scalp_signal(request)
                else:
                    request = IchimokuRequest(symbol=symbol, timeframe=tf)
                    response = await get_ichimoku_scalp(request)
                
                response["timeframe"] = tf
                results.append(response)
                
                logger.debug(f"  ✓ {tf}: {response.get('signal', 'ERROR')}")
                
            except Exception as e:
                logger.warning(f"  ✗ {tf}: {str(e)[:50]}")
                results.append({
                    "symbol": symbol,
                    "timeframe": tf,
                    "signal": "ERROR",
                    "error": str(e)[:100]
                })
        
        # تحلیل کلی
        signals = [r.get("signal") for r in results if r.get("signal") not in ["ERROR", "HOLD"]]
        
        overall_signal = "HOLD"
        if signals:
            buy_count = signals.count("BUY") + signals.count("STRONG_BUY") + signals.count("WEAK_BUY")
            sell_count = signals.count("SELL") + signals.count("STRONG_SELL") + signals.count("WEAK_SELL")
            
            if buy_count > sell_count:
                overall_signal = "BUY"
            elif sell_count > buy_count:
                overall_signal = "SELL"
        
        return {
            "symbol": symbol,
            "overall_signal": overall_signal,
            "scan_time": datetime.now().isoformat(),
            "total_timeframes": len(timeframes),
            "successful_scans": len([r for r in results if r.get("signal") != "ERROR"]),
            "timeframe_analysis": results,
            "summary": {
                "buy_signals": len([r for r in results if "BUY" in str(r.get("signal"))]),
                "sell_signals": len([r for r in results if "SELL" in str(r.get("signal"))]),
                "hold_signals": len([r for r in results if r.get("signal") == "HOLD"]),
                "error_signals": len([r for r in results if r.get("signal") == "ERROR"])
            }
        }
        
    except Exception as e:
        logger.error(f"خطا در اسکن: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"خطای اسکن: {str(e)[:200]}"
        )

@app.get("/market/{symbol}")
async def get_market_data(symbol: str, timeframe: str = "5m"):
    """داده‌های بازار"""
    try:
        market_data_result = utils.get_market_data_with_fallback(
            symbol, timeframe, 50, return_source=True
        )
        
        if isinstance(market_data_result, dict):
            market_data = market_data_result.get("data", [])
            source = market_data_result.get("source", "unknown")
        else:
            market_data = market_data_result
            source = "direct"
        
        if not market_data:
            raise HTTPException(status_code=404, detail="داده بازار موجود نیست")
        
        # محاسبات
        latest = market_data[-1] if market_data else []
        change_24h = utils.calculate_24h_change_from_dataframe(market_data)
        rsi = utils.calculate_simple_rsi(market_data, 14)
        sma_20 = utils.calculate_simple_sma(market_data, 20)
        
        # سطوح حمایت/مقاومت
        sr_levels = utils.get_support_resistance_levels(market_data) if hasattr(utils, 'get_support_resistance_levels') else {"support": 0, "resistance": 0}
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "source": source,
            "price": {
                "current": float(latest[4]) if len(latest) > 4 else 0,
                "high": float(latest[2]) if len(latest) > 2 else 0,
                "low": float(latest[3]) if len(latest) > 3 else 0,
                "open": float(latest[1]) if len(latest) > 1 else 0
            },
            "change_24h": change_24h,
            "indicators": {
                "rsi": round(rsi, 2),
                "sma_20": round(sma_20, 2)
            },
            "levels": {
                "support": sr_levels.get("support", 0),
                "resistance": sr_levels.get("resistance", 0)
            },
            "timestamp": datetime.now().isoformat(),
            "candles": len(market_data)
        }
        
    except Exception as e:
        logger.error(f"خطای داده بازار: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"خطای داده بازار: {str(e)[:200]}"
        )

@app.get("/api/quick-scan/{symbol}")
async def quick_scan(symbol: str):
    """اسکن سریع برای نمودارها"""
    try:
        # تحلیل 5 دقیقه برای اسکن سریع
        request = AnalysisRequest(symbol=symbol, timeframe="5m")
        analysis = await analyze_crypto(request)
        
        return {
            "symbol": symbol,
            "signal": analysis["analysis"]["final_signal"],
            "confidence": analysis["analysis"]["confidence"],
            "price": analysis["price"]["current"],
            "rsi": analysis["indicators"]["rsi"],
            "summary": analysis["analysis"]["reason"],
            "timestamp": datetime.now().isoformat(),
            "quick": True
        }
        
    except Exception as e:
        logger.error(f"خطای اسکن سریع: {e}")
        return {
            "symbol": symbol,
            "signal": "ERROR",
            "error": str(e)[:100],
            "timestamp": datetime.now().isoformat()
        }

# ==============================================================================
# Startup
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info(f"🚀 شروع سیستم معاملاتی هوش مصنوعی v{API_VERSION}")
    logger.info("📊 سیستم: ایچیموکو + الگوی QM + تشخیص ساختار")
    logger.info("✅ سیستم با موفقیت راه‌اندازی شد")

# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"🌐 سرور در حال راه‌اندازی روی {host}:{port}")
    print(f"\n{'=' * 60}")
    print(f"🚀 سیستم معاملاتی هوش مصنوعی ارز دیجیتال v{API_VERSION}")
    print(f"📡 آدرس سرور: http://{host}:{port}")
    print(f"📚 مستندات API: http://{host}:{port}/api/docs")
    print(f"❤️  وضعیت سیستم: http://{host}:{port}/api/health")
    print(f"{'=' * 60}\n")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )