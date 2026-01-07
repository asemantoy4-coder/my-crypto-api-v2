#!/usr/bin/env python3
"""
ورودی اصلی برای Vercel Deployment
"""

import os
import sys
import asyncio
import logging
import threading
from datetime import datetime
import requests

# اضافه کردن مسیر
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bot import FastScalpCompleteBot
from utils import setup_logger

# ============================================
# 🎯 تابع اصلی
# ============================================

async def main():
    """تابع اصلی اجرای ربات"""
    
    print("\n" + "="*60)
    print("🚀 FAST SCALP BOT - VERCEL DEPLOYMENT")
    print("="*60)
    print(f"Start Time: {datetime.utcnow()}")
    print(f"Python: {sys.version}")
    print(f"Environment: {os.getenv('VERCEL_ENV', 'development')}")
    print("="*60 + "\n")
    
    # تنظیم لاگر
    logger = setup_logger("fast_scalp_vercel")
    
    # بررسی متغیرهای محیطی
    required_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"❌ Missing environment variables: {missing}")
        sys.exit(1)
    
    # پیکربندی
    config = {
        'telegram_token': os.getenv('TELEGRAM_BOT_TOKEN'),
        'chat_id': os.getenv('TELEGRAM_CHAT_ID'),
        'mexc_api_key': os.getenv('MEXC_API_KEY', ''),
        'mexc_secret_key': os.getenv('MEXC_SECRET_KEY', ''),
        'timeframe': '5m',
        'top_n': 3,
        'update_interval': 3600,
        'max_symbols': 20,
        'min_confidence': 65
    }
    
    # ایجاد و اجرای ربات
    try:
        bot = FastScalpCompleteBot(config)
        
        # ارسال پیام شروع
        try:
            from telegram import Bot
            telegram_bot = Bot(token=config['telegram_token'])
            await telegram_bot.send_message(
                chat_id=config['chat_id'],
                text=f"🚀 *Fast Scalp Bot Started on Vercel*\n\nTime: {datetime.utcnow().strftime('%H:%M:%S')} UTC",
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.warning(f"Could not send startup message: {e}")
        
        # اجرای ربات
        logger.info("🤖 Starting main bot loop...")
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("👋 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Bot error: {e}", exc_info=True)
        sys.exit(1)

# ============================================
# 🔄 Keep-alive برای Vercel
# ============================================

def keep_alive():
    """ارسال درخواست‌های دوره‌ای برای جلوگیری از sleep شدن"""
    import time
    
    # دریافت آدرس پروژه از متغیرهای محیطی Vercel
    vercel_url = os.getenv('VERCEL_URL')
    if not vercel_url:
        # اگر Vercel URL وجود ندارد، از localhost استفاده کن
        vercel_url = "http://localhost:3000"
    
    while True:
        try:
            response = requests.get(f"{vercel_url}/health", timeout=10)
            print(f"✅ Keep-alive ping: {response.status_code} - {datetime.utcnow().strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"⚠️ Keep-alive failed: {e}")
        
        # هر 5 دقیقه یکبار
        time.sleep(300)

# ============================================
# 🎬 نقطه ورود
# ============================================

if __name__ == "__main__":
    # در Vercel، باید endpoint HTTP داشته باشیم
    # اما ربات ما یک background worker است
    # بنابراین دو کار همزمان انجام می‌دهیم:
    
    # 1. اجرای keep-alive در background
    keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
    keep_alive_thread.start()
    
    # 2. اجرای ربات اصلی
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
