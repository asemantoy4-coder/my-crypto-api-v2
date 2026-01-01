import utils
import logging

# تنظیمات لاگ برای مشاهده خروجی
logging.basicConfig(level=logging.INFO)

# تست دریافت داده و تحلیل ایچیموکو برای بیت‌کوین
data = utils.get_market_data_simple("BTCUSDT", interval="5m", limit=100)
if data:
    analysis = utils.analyze_ichimoku_scalp_signal(data)
    print(f"Signal: {analysis['signal']} | Confidence: {analysis['confidence']}")
    print(f"Reason: {analysis['reason']}")
