import os
import sys
import asyncio
import logging
import traceback

# تنظیم لاگ فقط روی کنسول
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# اصلاح مسیر برای پیدا کردن bot.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

async def run_bot():
    try:
        from bot import FastScalpCompleteBot

        # خواندن تنظیمات از Environment Variables در Railway
        config = {
            'telegram_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
            'mexc_api_key': os.getenv('MEXC_API_KEY', ''),
            'mexc_secret_key': os.getenv('MEXC_SECRET_KEY', ''),
            'timeframe': '5m',
            'top_n': 3
        }

        if not config['telegram_token'] or not config['chat_id']:
            logger.error("❌ Missing Env Vars (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")
            return

        logger.info("🚀 Bot started on Railway...")

        bot = FastScalpCompleteBot(config)
        result = await bot.scan_market()
        logger.info(f"✅ Scan Result: {result}")

    except Exception as e:
        logger.error(f"🔥 Critical Error: {str(e)}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(run_bot())
