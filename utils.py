# ==============================================================================
# ۱۳. توابع عملیاتی و ارتباطی (Operational & Telegram)
# ==============================================================================

import requests
import os
import time
import csv
from datetime import datetime
from typing import List
import logging

# تنظیمات logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log')
    ]
)
logger = logging.getLogger(__name__)

def send_telegram_notification(message: str, token: str, chat_id: str):
    """ارسال واقعی پیام به تلگرام"""
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info("📡 پیام با موفقیت به تلگرام ارسال شد")
            return True
        else:
            logger.error(f"❌ خطای تلگرام API: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error("⏰ زمان ارسال پیام به تلگرام به پایان رسید")
        return False
    except Exception as e:
        logger.error(f"❌ خطا در ارسال به تلگرام: {e}")
        return False

def format_telegram_message(val: dict) -> str:
    """قالب‌بندی پیام با استفاده از خروجی تابع combined_technical_analysis"""
    # ایموجی مناسب
    if val['signal'] == "BUY":
        emoji = "🟢"
        action = "خرید"
    elif val['signal'] == "SELL":
        emoji = "🔴"
        action = "فروش"
    else:
        emoji = "🟡"
        action = "نگه‌دارید"
    
    # دلایل تحلیل
    reasons = val.get('reasons', ["تحلیل تکنیکال نشان‌دهنده شرایط خاص است"])
    reasons_str = "\n".join([f"• {r}" for r in reasons[:5]])  # حداکثر 5 دلیل
    
    # قیمت‌ها
    current_price = val.get('current_price', val.get('price', 0))
    entry_price = val.get('entry_price', current_price)
    
    # اهداف و حد ضرر
    targets = val.get('targets', [])
    stop_loss = val.get('stop_loss', 0)
    
    # ساخت پیام
    message = f"""
{emoji} **سیگنال معاملاتی جدید** {emoji}

**نماد:** #{val['symbol']}
**عمل:** {action} ({val['signal']})
**اعتماد:** {val.get('confidence', 0)*100:.1f}%

**سطوح قیمتی:**
• قیمت فعلی: ${current_price:,.2f}
• نقطه ورود: ${entry_price:,.2f}
• حد ضرر: ${stop_loss:,.2f}

**اهداف قیمتی:**
{chr(10).join([f'• هدف {i+1}: ${target:,.2f}' for i, target in enumerate(targets)]) if targets else '• اهداف تعیین نشده'}

**دلایل فنی:**
{reasons_str}

**تایم‌فریم:** {val.get('interval', '5m')}
**زمان تحلیل:** {val.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}

⚠️ **تذکر:** این سیگنال توصیه مالی نیست. تحلیل شخصی خود را انجام دهید.
"""
    return message

def save_trade_to_csv(signal_data: dict):
    """ذخیره سیگنال در فایل CSV"""
    filename = "trading_signals_history.csv"
    
    # فیلدهای CSV
    fieldnames = [
        'timestamp', 'symbol', 'signal', 'confidence', 
        'current_price', 'entry_price', 'stop_loss', 
        'targets', 'reasons', 'interval', 'rsi', 'macd_signal'
    ]
    
    # آماده‌سازی داده
    row_data = {
        'timestamp': signal_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        'symbol': signal_data.get('symbol', 'UNKNOWN'),
        'signal': signal_data.get('signal', 'HOLD'),
        'confidence': signal_data.get('confidence', 0),
        'current_price': signal_data.get('current_price', signal_data.get('price', 0)),
        'entry_price': signal_data.get('entry_price', 0),
        'stop_loss': signal_data.get('stop_loss', 0),
        'targets': str(signal_data.get('targets', [])),
        'reasons': " | ".join(signal_data.get('reasons', ['No reasons provided'])),
        'interval': signal_data.get('interval', '5m'),
        'rsi': signal_data.get('rsi', 0),
        'macd_signal': signal_data.get('macd_signal', 'NEUTRAL')
    }
    
    try:
        # بررسی وجود فایل
        file_exists = os.path.isfile(filename)
        
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row_data)
            logger.info(f"💾 سیگنال {signal_data.get('symbol')} در CSV ذخیره شد")
            
    except Exception as e:
        logger.error(f"❌ خطا در ذخیره‌سازی CSV: {e}")

def initialize_system():
    """راه‌اندازی اولیه سیستم"""
    logger.info("🔄 در حال راه‌اندازی سیستم معاملاتی...")
    
    # بررسی متغیرهای محیطی
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or bot_token == "YOUR_ACTUAL_TOKEN":
        logger.warning("⚠️ توکن تلگرام تنظیم نشده است. لطفاً TELEGRAM_BOT_TOKEN را تنظیم کنید")
    
    if not chat_id or chat_id == "YOUR_ACTUAL_ID":
        logger.warning("⚠️ چت آیدی تلگرام تنظیم نشده است. لطفاً TELEGRAM_CHAT_ID را تنظیم کنید")
    
    # ایجاد دایرکتوری‌های مورد نیاز
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    logger.info("✅ سیستم با موفقیت راه‌اندازی شد")
    return True

def clear_expired_cache():
    """پاک‌سازی حافظه کش منقضی شده"""
    # این تابع می‌تواند کش‌های موقت سیستم را پاک کند
    # فعلاً یک پیاده‌سازی ساده
    pass

# ==============================================================================
# ۱۴. توابع اصلی تحلیل بازار
# ==============================================================================

def get_market_data_simple(symbol: str, interval: str = "5m", limit: int = 100):
    """
    دریافت داده‌های بازار از صرافی
    
    Args:
        symbol: نماد معاملاتی (مثال: BTCUSDT)
        interval: تایم‌فریم (1m, 5m, 15m, 1h, 4h, 1d)
        limit: تعداد کندل‌های درخواستی
    
    Returns:
        list: لیست کندل‌ها به صورت [timestamp, open, high, low, close, volume]
    """
    try:
        # استفاده از API عمومی Binance (بدون نیاز به کلید API)
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            candles = []
            
            for candle in data:
                candles.append([
                    int(candle[0]),  # timestamp
                    float(candle[1]),  # open
                    float(candle[2]),  # high
                    float(candle[3]),  # low
                    float(candle[4]),  # close
                    float(candle[5])   # volume
                ])
            
            logger.debug(f"📊 داده‌های {symbol} دریافت شد ({len(candles)} کندل)")
            return candles
            
        else:
            logger.error(f"❌ خطا در دریافت داده‌های {symbol}: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"❌ خطا در دریافت داده‌های بازار: {e}")
        return None

def combined_technical_analysis(data, symbol: str, interval: str):
    """
    تحلیل تکنیکال ترکیبی
    
    Args:
        data: داده‌های کندل‌ستیکی
        symbol: نماد معاملاتی
        interval: تایم‌فریم
    
    Returns:
        dict: نتایج تحلیل تکنیکال
    """
    if not data or len(data) < 20:
        logger.warning(f"⚠️ داده‌های ناکافی برای تحلیل {symbol}")
        return None
    
    try:
        # استخراج قیمت‌های بسته شدن
        closes = [candle[4] for candle in data]
        highs = [candle[2] for candle in data]
        lows = [candle[3] for candle in data]
        
        # قیمت فعلی
        current_price = closes[-1]
        
        # ۱. محاسبه RSI
        rsi_value = calculate_rsi(closes)
        
        # ۲. محاسبه MACD
        macd_signal = calculate_macd_signal(closes)
        
        # ۳. محاسبه میانگین‌های متحرک
        sma_20 = calculate_sma(closes, 20)
        sma_50 = calculate_sma(closes, 50)
        
        # ۴. سطوح حمایت و مقاومت
        support, resistance = calculate_support_resistance(highs, lows)
        
        # ۵. تشخیص روند
        trend = identify_trend(closes, sma_20, sma_50)
        
        # جمع‌آوری دلایل تحلیل
        reasons = []
        
        # تحلیل RSI
        if rsi_value < 30:
            reasons.append("RSI در منطقه اشباع فروش (<30)")
        elif rsi_value > 70:
            reasons.append("RSI در منطقه اشباع خرید (>70)")
        
        # تحلیل MACD
        if macd_signal == "BULLISH":
            reasons.append("MACD سیگنال صعودی")
        elif macd_signal == "BEARISH":
            reasons.append("MACD سیگنال نزولی")
        
        # تحلیل میانگین‌های متحرک
        if sma_20 > sma_50 and closes[-1] > sma_20:
            reasons.append("قیمت بالای میانگین‌های متحرک")
        elif sma_20 < sma_50 and closes[-1] < sma_20:
            reasons.append("قیمت زیر میانگین‌های متحرک")
        
        return {
            'symbol': symbol,
            'interval': interval,
            'current_price': current_price,
            'rsi': rsi_value,
            'macd_signal': macd_signal,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'support': support,
            'resistance': resistance,
            'trend': trend,
            'reasons': reasons,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        logger.error(f"❌ خطا در تحلیل تکنیکال {symbol}: {e}")
        return None

def validate_final_signal(analysis):
    """
    اعتبارسنجی نهایی سیگنال
    
    Args:
        analysis: خروجی تابع combined_technical_analysis
    
    Returns:
        dict: سیگنال اعتبارسنجی شده
    """
    if not analysis:
        return {'signal': 'HOLD', 'is_validated': False}
    
    try:
        signal = "HOLD"
        confidence = 0.0
        reasons = analysis.get('reasons', [])
        
        # استخراج مقادیر
        rsi = analysis.get('rsi', 50)
        macd = analysis.get('macd_signal', 'NEUTRAL')
        trend = analysis.get('trend', 'SIDEWAYS')
        price = analysis.get('current_price', 0)
        support = analysis.get('support', 0)
        resistance = analysis.get('resistance', 0)
        
        # منطق تولید سیگنال (قابل تنظیم)
        buy_score = 0
        sell_score = 0
        
        # شرایط خرید
        if rsi < 35:
            buy_score += 2
            reasons.append("RSI پایین (شرایط اشباع فروش)")
        if macd == "BULLISH":
            buy_score += 2
        if trend == "UPTREND":
            buy_score += 1
        if price < support * 1.02:  # نزدیک به حمایت
            buy_score += 1
            reasons.append("نزدیک به سطح حمایت")
        
        # شرایط فروش
        if rsi > 65:
            sell_score += 2
            reasons.append("RSI بالا (شرایط اشباع خرید)")
        if macd == "BEARISH":
            sell_score += 2
        if trend == "DOWNTREND":
            sell_score += 1
        if price > resistance * 0.98:  # نزدیک به مقاومت
            sell_score += 1
            reasons.append("نزدیک به سطح مقاومت")
        
        # تصمیم‌گیری نهایی
        if buy_score >= 4 and buy_score > sell_score:
            signal = "BUY"
            confidence = min(0.9, buy_score / 10)
            # محاسبه اهداف و حد ضرر برای خرید
            entry = price
            stop_loss = entry * 0.97  # 3% حد ضرر
            targets = [
                entry * 1.02,  # 2% هدف اول
                entry * 1.05,  # 5% هدف دوم
                entry * 1.08   # 8% هدف سوم
            ]
            
        elif sell_score >= 4 and sell_score > buy_score:
            signal = "SELL"
            confidence = min(0.9, sell_score / 10)
            # محاسبه اهداف و حد ضرر برای فروش
            entry = price
            stop_loss = entry * 1.03  # 3% حد ضرر
            targets = [
                entry * 0.98,  # -2% هدف اول
                entry * 0.95,  # -5% هدف دوم
                entry * 0.92   # -8% هدف سوم
            ]
        
        else:
            signal = "HOLD"
            confidence = 0.0
            entry = price
            stop_loss = 0
            targets = []
        
        # ساخت خروجی نهایی
        validated_signal = {
            'symbol': analysis.get('symbol'),
            'signal': signal,
            'confidence': confidence,
            'is_validated': signal != "HOLD",
            'current_price': price,
            'entry_price': entry,
            'stop_loss': stop_loss,
            'targets': targets,
            'reasons': list(set(reasons)),  # حذف موارد تکراری
            'rsi': rsi,
            'macd_signal': macd,
            'trend': trend,
            'interval': analysis.get('interval', '5m'),
            'timestamp': analysis.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        }
        
        return validated_signal
        
    except Exception as e:
        logger.error(f"❌ خطا در اعتبارسنجی سیگنال: {e}")
        return {'signal': 'HOLD', 'is_validated': False}

# ==============================================================================
# ۱۵. توابع کمکی تحلیل تکنیکال
# ==============================================================================

def calculate_rsi(prices, period=14):
    """محاسبه RSI"""
    if len(prices) < period + 1:
        return 50
    
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    
    gains = [delta if delta > 0 else 0 for delta in deltas]
    losses = [-delta if delta < 0 else 0 for delta in deltas]
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return round(rsi, 2)

def calculate_macd_signal(prices, fast_period=12, slow_period=26, signal_period=9):
    """محاسبه سیگنال MACD"""
    if len(prices) < slow_period + signal_period:
        return "NEUTRAL"
    
    # محاسبه EMA سریع و کند
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)
    
    # خط MACD
    macd_line = ema_fast - ema_slow
    
    # خط سیگنال
    signal_line = calculate_ema([macd_line], signal_period)
    
    # هیستوگرام MACD
    histogram = macd_line - signal_line
    
    # تعیین سیگنال
    if histogram > 0 and macd_line > signal_line:
        return "BULLISH"
    elif histogram < 0 and macd_line < signal_line:
        return "BEARISH"
    else:
        return "NEUTRAL"

def calculate_ema(prices, period):
    """محاسبه میانگین متحرک نمایی"""
    if len(prices) < period:
        return sum(prices) / len(prices) if prices else 0
    
    multiplier = 2 / (period + 1)
    ema = prices[0]
    
    for price in prices[1:]:
        ema = (price - ema) * multiplier + ema
    
    return ema

def calculate_sma(prices, period):
    """محاسبه میانگین متحرک ساده"""
    if len(prices) < period:
        return sum(prices) / len(prices) if prices else 0
    
    return sum(prices[-period:]) / period

def calculate_support_resistance(highs, lows, lookback=20):
    """محاسبه سطوح حمایت و مقاومت"""
    if len(highs) < lookback or len(lows) < lookback:
        return 0, 0
    
    support = min(lows[-lookback:])
    resistance = max(highs[-lookback:])
    
    return support, resistance

def identify_trend(prices, sma_20, sma_50):
    """تشخیص روند بازار"""
    if len(prices) < 3:
        return "SIDEWAYS"
    
    # مقایسه میانگین‌های متحرک
    if sma_20 > sma_50 * 1.02:
        return "UPTREND"
    elif sma_20 < sma_50 * 0.98:
        return "DOWNTREND"
    else:
        return "SIDEWAYS"

# ==============================================================================
# ۱۶. موتور اجرایی ربات (The Trading Engine)
# ==============================================================================

def run_trading_bot(symbols: List[str], interval: str = "5m"):
    """حلقه اصلی و عملیاتی ربات"""
    
    # تنظیمات را از محیط سیستم بخوانید
    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_ACTUAL_TOKEN")
    CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_ACTUAL_ID")
    
    # راه‌اندازی سیستم
    initialize_system()
    
    logger.info(f"🚀 ربات شروع به کار کرد. مانیتورینگ {len(symbols)} ارز در تایم‌فریم {interval}")
    print("\n" + "="*60)
    print(f"🤖 TRADING BOT ACTIVE")
    print(f"📊 نمادها: {', '.join(symbols)}")
    print(f"⏰ تایم‌فریم: {interval}")
    print(f"🔔 نوتیفیکیشن: {'فعال' if BOT_TOKEN != 'YOUR_ACTUAL_TOKEN' else 'غیرفعال'}")
    print("="*60 + "\n")
    
    # دیکشنری برای ذخیره آخرین سیگنال‌ها
    last_signals = {}
    
    while True:
        try:
            cycle_start = time.time()
            logger.info(f"🔄 شروع چرخه جدید تحلیل - {datetime.now().strftime('%H:%M:%S')}")
            
            for symbol in symbols:
                try:
                    logger.info(f"🔍 در حال تحلیل {symbol}...")
                    
                    # ۱. دریافت داده‌های بازار
                    data = get_market_data_simple(symbol, interval)
                    if not data:
                        logger.warning(f"⚠️ داده‌ای برای {symbol} دریافت نشد")
                        continue
                    
                    # ۲. تحلیل تکنیکال
                    analysis = combined_technical_analysis(data, symbol, interval)
                    if not analysis:
                        logger.warning(f"⚠️ تحلیل {symbol} ناموفق بود")
                        continue
                    
                    # ۳. اعتبارسنجی سیگنال
                    validated = validate_final_signal(analysis)
                    
                    # ۴. بررسی تغییر سیگنال نسبت به آخرین بار
                    signal_key = f"{symbol}_{interval}"
                    last_signal = last_signals.get(signal_key, {})
                    
                    if (validated['signal'] != "HOLD" and 
                        validated.get('is_validated', False) and
                        (last_signal.get('signal') != validated['signal'] or 
                         validated['confidence'] >= 0.7)):
                        
                        # ۵. نمایش پیش‌نمایش در کنسول
                        print("\n" + "="*60)
                        print("📊 سیگنال جدید شناسایی شد!")
                        print("="*60)
                        
                        # نمایش ساده در کنسول
                        console_msg = f"""
نماد: {validated['symbol']}
سیگنال: {validated['signal']} ({validated['confidence']*100:.1f}%)
قیمت: ${validated['current_price']:,.2f}
دلایل: {', '.join(validated.get('reasons', [])[:3])}
                        """
                        print(console_msg)
                        
                        # ۶. ذخیره در CSV
                        save_trade_to_csv(validated)
                        
                        # ۷. ارسال به تلگرام
                        if BOT_TOKEN != "YOUR_ACTUAL_TOKEN" and CHAT_ID != "YOUR_ACTUAL_ID":
                            msg = format_telegram_message(validated)
                            success = send_telegram_notification(msg, BOT_TOKEN, CHAT_ID)
                            
                            if success:
                                logger.info(f"✅ سیگنال {symbol} به تلگرام ارسال شد")
                            else:
                                logger.warning(f"⚠️ ارسال سیگنال {symbol} به تلگرام ناموفق بود")
                        
                        # ۸. به‌روزرسانی آخرین سیگنال
                        last_signals[signal_key] = {
                            'signal': validated['signal'],
                            'timestamp': validated['timestamp'],
                            'price': validated['current_price']
                        }
                    
                    else:
                        logger.info(f"📈 هیچ سیگنال قابل‌اقدامی برای {symbol}")
                    
                    # وقفه کوتاه بین نمادها
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"⚠️ خطا در پردازش {symbol}: {e}")
                    continue
            
            # ۹. پاکسازی حافظه
            clear_expired_cache()
            
            # ۱۰. محاسبه زمان باقی‌مانده تا چرخه بعدی
            cycle_time = time.time() - cycle_start
            wait_time = max(60 - cycle_time, 10)  # حداقل ۱۰ ثانیه
            
            logger.info(f"⏳ چرخه تحلیل پایان یافت. چرخه بعدی در {wait_time:.0f} ثانیه دیگر")
            
            # ۱۱. انتظار برای چرخه بعدی
            for i in range(int(wait_time)):
                if i % 10 == 0:  # هر ۱۰ ثانیه لاگ کن
                    remaining = int(wait_time) - i
                    logger.debug(f"⏰ منتظر چرخه بعدی... {remaining} ثانیه باقی مانده")
                time.sleep(1)
            
        except KeyboardInterrupt:
            logger.info("🛑 ربات توسط کاربر متوقف شد")
            print("\n\n" + "="*60)
            print("👋 ربات متوقف شد. خدانگهدار!")
            print("="*60)
            break
            
        except Exception as e:
            logger.error(f"💥 خطای بحرانی در چرخه اصلی: {e}")
            print(f"❌ خطا: {e}")
            time.sleep(30)

# ==============================================================================
# ۱۷. نقطه شروع (Main)
# ==============================================================================

if __name__ == "__main__":
    # نمایش بنر آغازین
    print("\n" + "="*60)
    print("🚀 Trading Bot v1.0 - Complete Edition")
    print("="*60)
    print("📊 تحلیل تکنیکال ترکیبی + نوتیفیکیشن تلگرام")
    print("👨‍💻 توسعه‌دهنده: Trading Bot Team")
    print("="*60 + "\n")
    
    # لیست نمادهای مورد نظر
    MY_WATCHLIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    
    try:
        # اجرای ربات
        run_trading_bot(symbols=MY_WATCHLIST, interval="5m")
        
    except Exception as e:
        logger.critical(f"💥 خطای فاجعه‌بار: {e}")
        print(f"\n❌ ربات با خطا متوقف شد: {e}")
        
    finally:
        print("\n" + "="*60)
        print("📁 لاگ‌ها در فایل trading_bot.log ذخیره شدند")
        print("📊 تاریخچه سیگنال‌ها در trading_signals_history.csv ذخیره شد")
        print("="*60)