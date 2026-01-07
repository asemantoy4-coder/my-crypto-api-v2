import os
import sys
import asyncio
import logging
import traceback
from datetime import datetime

# تنظیمات لاگ برای نمایش دقیق در کنسول Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# اصلاح مسیر برای پیدا کردن فایل bot.py در کنار این فایل
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

async def run_bot():
    try:
        # وارد کردن کلاس ربات از فایل bot.py شما
        from bot import FastScalpCompleteBot

        # خواندن تنظیمات از بخش Variables در Railway
        config = {
            'telegram_token': os.getenv('TELEGRAM_BOT_TOKEN', '').strip(),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID', '').strip(),
            'mexc_api_key': os.getenv('MEXC_API_KEY', ''),
            'mexc_secret_key': os.getenv('MEXC_SECRET_KEY', ''),
            'timeframe': '5m',
            'top_n': 3
        }

        # بررسی وجود اطلاعات حیاتی
        if not config['telegram_token'] or not config['chat_id']:
            logger.error("❌ خطا: متغیرهای توکن یا چت‌آیدی در Railway تنظیم نشده‌اند!")
            return

        logger.info("🚀 ربات در حال آماده‌سازی روی Railway...")

        # مقداردهی اولیه ربات
        bot = FastScalpCompleteBot(config)

        # --- تست اتصال (اختیاری: می‌توانید بعد از اطمینان این بخش را حذف کنید) ---
        try:
            from telegram import Bot
            test_bot = Bot(token=config['telegram_token'])
            await test_bot.send_message(
                chat_id=config['chat_id'], 
                text=f"✅ اتصال برقرار شد!\nزمان شروع: {datetime.now().strftime('%H:%M:%S')}"
            )
            logger.info("✅ پیام تست با موفقیت به تلگرام ارسال شد.")
        except Exception as te:
            logger.error(f"❌ خطا در ارسال پیام تست: {te}")
        # -----------------------------------------------------------------------

        # حلقه دائمی برای اسکن بازار (مثلاً هر 5 دقیقه)
        while True:
            try:
                logger.info(f"🔍 در حال اسکن بازار... ({datetime.now().strftime('%H:%M:%S')})")
                result = await bot.scan_market()
                logger.info(f"📊 نتیجه اسکن: {result}")
                
                # وقفه 300 ثانیه‌ای (5 دقیقه) بین هر بار اسکن
                await asyncio.sleep(300) 
                
            except Exception as e:
                logger.error(f"⚠️ خطا در طول اسکن: {e}")
                await asyncio.sleep(60) # در صورت بروز خطا، یک دقیقه صبر کن و دوباره تلاش کن

    except Exception as e:
        logger.error(f"🔥 خطای بحرانی در اجرای اصلی: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("Stopping bot...")
