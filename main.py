#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 FAST SCALP COMPLETE BOT - NORTHFLANK DEPLOYMENT
🤖 ربات فست اسکلپ کامل برای بازار کریپتو
📅 Version: 1.0.0
"""

import os
import sys
import asyncio
import logging
import traceback
import threading
from datetime import datetime
from pathlib import Path

# Flask برای health check
from flask import Flask, jsonify

# افزودن مسیر پروژه به sys.path
sys.path.append(str(Path(__file__).parent))

from bot import FastScalpCompleteBot
from utils import setup_logger, sanitize_output

# ============================================
# 🎨 Banner و نمایش اطلاعات
# ============================================

def display_banner():
    """نمایش بنر زیبا"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🤖 FAST SCALP COMPLETE TRADING BOT v1.0.0              ║
║   📊 ترکیب کامل دو اندیکاتور پیشرفته                    ║
║   ⚡ تایم‌فرم ۵ دقیقه - اسکالپینگ سریع                   ║
║   🚀 توسعه یافته برای Northflank                         ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

📋 ویژگی‌ها:
├── 🟢 ZLMA Trend + Smart Money Pro
├── 🔴 RSI Divergence + Ichimoku Cloud
├── 📊 تحلیل ارزهای برتر
├── ⏰ اسکن هر ساعت
├── 📱 ارسال ۳ سیگنال برتر به تلگرام
├── 🛡️ مدیریت ریسک با ATR
├── 🩺 Health Check اتوماتیک
└── 📈 سیستم امتیازدهی پیشرفته
"""
    print(banner)

# ============================================
# 🌐 Flask App برای Health Check
# ============================================

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "service": "fast-scalp-bot",
        "time": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "environment": "northflank"
    })

@app.route('/health')
def health():
    """Endpoint برای Health Check Northflank"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }), 200

@app.route('/metrics')
def metrics():
    """Endpoint برای monitoring"""
    return jsonify({
        "status": "operational",
        "signals_today": 0,
        "last_scan": datetime.utcnow().isoformat(),
        "uptime": "0 days 0 hours"
    })

def run_flask():
    """اجرای Flask در background"""
    print(f"[FLASK] Starting Flask server on port 8080")
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)

# ============================================
# ⚙️ Configuration Loader
# ============================================

def load_config() -> dict:
    """لود کردن و اعتبارسنجی تنظیمات"""
    
    print("\n" + "="*60)
    print("⚙️  LOADING CONFIGURATION")
    print("="*60)
    
    # ساختار config ساده‌شده
    config = {}
    
    # ======================
    # 📱 تنظیمات تلگرام (ضروری)
    # ======================
    required_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            print(f"❌ متغیر محیطی ضروری یافت نشد: {var}")
            print("لطفاً در Northflank Dashboard → Variables تنظیم کنید")
            sys.exit(1)
        
        if var == 'TELEGRAM_BOT_TOKEN':
            config['telegram_token'] = value
            # نمایش جزئی از توکن برای تایید
            token_preview = value[:8] + "..." + value[-8:] if len(value) > 16 else value
            print(f"✅ Telegram Token: {token_preview}")
        else:
            config['chat_id'] = value
            print(f"✅ Telegram Chat ID: {value}")
    
    # ======================
    # 💱 تنظیمات صرافی (اختیاری)
    # ======================
    mexc_api_key = os.getenv('MEXC_API_KEY', '')
    mexc_secret = os.getenv('MEXC_SECRET_KEY', '')
    
    if mexc_api_key and mexc_secret:
        config['mexc_api_key'] = mexc_api_key
        config['mexc_secret_key'] = mexc_secret
        print("✅ MEXC API: Enabled (با احراز هویت)")
    else:
        config['mexc_api_key'] = ''
        config['mexc_secret_key'] = ''
        print("ℹ️ MEXC API: Disabled (استفاده از داده عمومی)")
    
    # ======================
    # 📈 تنظیمات استراتژی
    # ======================
    config.update({
        'timeframe': os.getenv('TIMEFRAME', '5m'),
        'top_n': int(os.getenv('TOP_N_SIGNALS', '3')),
        'update_interval': int(os.getenv('UPDATE_INTERVAL', '3600')),
        'min_confidence': int(os.getenv('MIN_CONFIDENCE', '65')),
        'max_symbols': int(os.getenv('MAX_SYMBOLS', '20')),
        'risk_reward': float(os.getenv('RISK_REWARD_RATIO', '1.5')),
        'atr_period': int(os.getenv('ATR_PERIOD', '14'))
    })
    
    print(f"\n📊 Strategy Config:")
    print(f"   • Timeframe: {config['timeframe']}")
    print(f"   • Top Signals: {config['top_n']}")
    print(f"   • Scan Interval: {config['update_interval']}s")
    print(f"   • Min Confidence: {config['min_confidence']}%")
    print(f"   • Max Symbols: {config['max_symbols']}")
    
    # ======================
    # 🖥️ تنظیمات سیستم
    # ======================
    config.update({
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'timezone': os.getenv('TZ', 'UTC'),
        'debug_mode': os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    })
    
    # تنظیم تایم‌زون
    os.environ['TZ'] = config['timezone']
    
    # ======================
    # ✅ نمایش config سانتایز شده
    # ======================
    print("\n" + "="*60)
    print("✅ CONFIGURATION LOADED SUCCESSFULLY")
    print("="*60)
    
    safe_config = sanitize_output(config)
    print(f"Config: {safe_config}")
    
    return config

# ============================================
# 🔧 System Health Check
# ============================================

async def system_health_check() -> bool:
    """بررسی سلامت سیستم قبل از راه‌اندازی"""
    
    print("\n" + "="*60)
    print("🔧 SYSTEM HEALTH CHECK")
    print("="*60)
    
    checks = []
    
    # 1. بررسی Python version
    python_version = sys.version_info
    python_ok = python_version >= (3, 8)
    checks.append(("Python >= 3.8", python_ok, f"{python_version.major}.{python_version.minor}"))
    
    # 2. بررسی وجود فایل‌های ضروری
    required_files = ['requirements.txt', 'bot.py', 'utils.py']
    for file in required_files:
        exists = Path(file).exists()
        checks.append((f"File: {file}", exists, "Found" if exists else "Missing"))
    
    # 3. بررسی حافظه (تقریبی)
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_ok = memory.available > 100 * 1024 * 1024  # 100MB
        checks.append(("Memory", memory_ok, f"{memory.available // (1024*1024)}MB available"))
    except ImportError:
        checks.append(("Memory Check", True, "psutil not installed"))
    
    # نمایش نتایج
    all_passed = True
    for check_name, status, details in checks:
        symbol = "✅" if status else "❌"
        print(f"{symbol} {check_name}: {details}")
        if not status:
            all_passed = False
    
    if all_passed:
        print("✅ همه بررسی‌های سلامت PASSED")
        return True
    else:
        print("❌ برخی بررسی‌های سلامت FAILED")
        return False

# ============================================
# 📱 Telegram Initialization
# ============================================

async def send_startup_message(config: dict):
    """ارسال پیام شروع به تلگرام"""
    try:
        from telegram import Bot
        
        bot_token = config['telegram_token']
        chat_id = config['chat_id']
        
        bot = Bot(token=bot_token)
        
        startup_msg = f"""
🚀 *Fast Scalp Bot Started Successfully!*

📋 *Configuration:*
• Version: 1.0.0
• Timeframe: {config['timeframe']}
• Scan Interval: {config['update_interval']} seconds
• Max Symbols: {config['max_symbols']}
• Timezone: {config['timezone']}

⏰ *Startup Time:* {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
📍 *Deployment:* Northflank

🤖 *Bot will scan the market every hour and send top {config['top_n']} signals.*

✅ *Status:* Active and Running
"""
        
        await bot.send_message(
            chat_id=chat_id,
            text=startup_msg,
            parse_mode='Markdown'
        )
        
        print("📤 Startup message sent to Telegram")
        
    except Exception as e:
        print(f"⚠️ Could not send startup message: {e}")

# ============================================
# 🎯 Main Bot Function
# ============================================

async def main_bot(config: dict):
    """ربات اصلی"""
    try:
        # ایجاد ربات
        bot = FastScalpCompleteBot(config)
        
        # اجرای ربات
        await bot.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
        raise
    except Exception as e:
        print(f"❌ Bot error: {e}")
        raise

# ============================================
# 🎬 Entry Point
# ============================================

async def main():
    """تابع اصلی اجرای ربات"""
    
    # نمایش بنر
    display_banner()
    
    # تنظیم لاگر
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    logger = logging.getLogger(__name__)
    
    print(f"🚀 Starting Fast Scalp Complete Bot")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version}")
    print(f"📁 Working Dir: {os.getcwd()}")
    print(f"🌐 Port: 8080")
    
    try:
        # 1. بررسی سلامت سیستم
        if not await system_health_check():
            print("System health check failed. Exiting...")
            sys.exit(1)
        
        # 2. لود کردن تنظیمات
        config = load_config()
        
        # 3. اجرای Flask در background
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        
        print(f"\n✅ Flask server started on http://0.0.0.0:8080")
        print(f"   Health Check: http://0.0.0.0:8080/health")
        
        # کمی صبر برای راه‌اندازی Flask
        import time
        time.sleep(2)
        
        # 4. ارسال پیام شروع به تلگرام
        await send_startup_message(config)
        
        # 5. اجرای ربات اصلی
        print("\n" + "="*60)
        print("🤖 STARTING MAIN BOT LOOP")
        print("="*60)
        print("Press Ctrl+C to stop the bot\n")
        
        await main_bot(config)
        
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("👋 BOT STOPPED BY USER")
        print("="*60)
        sys.exit(0)
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ FATAL ERROR OCCURRED")
        print("="*60)
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print("\nStack Trace:")
        traceback.print_exc()
        
        # ارسال خطا به تلگرام
        try:
            from telegram import Bot
            bot_token = config['telegram_token']
            chat_id = config['chat_id']
            
            error_msg = f"""
⚠️ *Bot Crashed!*

*Error:* `{type(e).__name__}`
*Message:* {str(e)[:200]}
*Time:* {datetime.utcnow().strftime('%H:%M:%S')} UTC

Please check the logs.
"""
            
            bot = Bot(token=bot_token)
            await bot.send_message(
                chat_id=chat_id,
                text=error_msg,
                parse_mode='Markdown'
            )
        except:
            pass
        
        sys.exit(1)

# ============================================
# 🚀 Startup
# ============================================

if __name__ == "__main__":
    # تنظیمات مخصوص Northflank
    is_northflank = 'NORTHFLANK' in os.environ or 'NF_' in os.environ
    
    if is_northflank:
        print("\n" + "="*60)
        print("🌐 RUNNING ON NORTHFLANK")
        print("="*60)
        
        # تنظیمات بهینه برای Northflank
        os.environ['LOG_TO_FILE'] = 'false'  # استفاده از stdout برای لاگ
        
        # حذف handler اضافی اگر وجود دارد
        root_logger = logging.getLogger()
        if root_logger.handlers:
            for handler in root_logger.handlers:
                root_logger.removeHandler(handler)
        
        # اضافه کردن handler برای Northflank
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        root_logger.addHandler(handler)
    
    # اجرای main
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Critical error during startup: {e}")
        traceback.print_exc()
        sys.exit(1)
