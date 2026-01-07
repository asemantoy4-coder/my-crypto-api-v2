import os
import sys
import asyncio
import logging
from datetime import datetime

# تنظیمات لاگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

async def run_bot():
    try:
        from bot import FastScalpCompleteBot

        # تنظیمات بدون کلیدهای MEXC
        config = {
            'telegram_token': os.getenv('TELEGRAM_BOT_TOKEN', '').strip(),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID', '').strip(),
            # کلیدها را خالی می‌گذاریم چون فقط دیتای عمومی می‌خواهیم
            'mexc_api_key': '', 
            'mexc_secret_key': '',
            'timeframe': '5m',
            'top_n': 3
        }

        if not config['telegram_token'] or not config['chat_id']:
            logger.error("❌ متغیرهای تلگرام در Railway تنظیم نشده‌اند!")
            return

        bot = FastScalpCompleteBot(config)
        logger.info("🚀 ربات اسکنر (فقط قیمت) با موفقیت اجرا شد...")

        while True:
            try:
                # این متد در bot.py باید از داده‌های عمومی استفاده کند
                result = await bot.scan_market() 
                logger.info(f"📊 وضعیت بازار: {result}")
                
                # ۵ دقیقه صبر تا اسکن بعدی
                await asyncio.sleep(300) 
            except Exception as e:
                logger.error(f"⚠️ خطا در اسکن: {e}")
                await asyncio.sleep(60)

    except Exception as e:
        logger.error(f"🔥 خطای بحرانی: {e}")

if __name__ == "__main__":
    asyncio.run(run_bot())
