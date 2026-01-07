import os
import sys
import asyncio
import logging
from bot import FastScalpCompleteBot

# تنظیمات لاگ برای Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

async def run_bot():
    # تنظیمات دریافتی از بخش Variables در Railway
    config = {
        'telegram_token': os.getenv('TELEGRAM_BOT_TOKEN', '').strip(),
        'chat_id': os.getenv('TELEGRAM_CHAT_ID', '').strip(),
        'timeframe': '5m'
    }

    if not config['telegram_token'] or not config['chat_id']:
        logger.error("❌ Critical: Telegram Token or Chat ID is missing in Railway Variables!")
        return

    bot = FastScalpCompleteBot(config)
    logger.info("🤖 Bot is starting the monitoring loop...")

    while True:
        try:
            # اجرای اسکن
            result = await bot.scan_market()
            logger.info(f"✅ Result: {result}")
            
            # وقفه ۵ دقیقه‌ای (۳۰۰ ثانیه) بین هر اسکن
            await asyncio.sleep(300) 
            
        except Exception as e:
            logger.error(f"⚠️ Loop Exception: {e}")
            await asyncio.sleep(60) # در صورت خطا، یک دقیقه صبر کن

if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped.")
