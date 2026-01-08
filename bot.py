import logging
import asyncio
import ccxt.async_support as ccxt
from datetime import datetime
from telegram import Bot
# وارد کردن ابزارهای کمکی از فایل utils خودتان
from utils import (
    calculate_market_structure, 
    calculate_support_resistance, 
    calculate_volatility,
    setup_logger
)

class FastScalpCompleteBot:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger("FastScalpBot")
        self.exchange = ccxt.mexc({'enableRateLimit': True})
        
    async def scan_market(self):
        """اسکن پیشرفته بازار با استفاده از متدهای utils.py"""
        try:
            # لیست ارزها برای اسکن
            symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'UNI/USDT', 'ENA/USDT' , 'OP/USDT' , 'XAU/USDT']
            report = "🔍 *Advanced Market Analysis*\n\n"

            for symbol in symbols:
                # 1. دریافت داده‌های OHLCV (شمعی)
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe='5m', limit=100)
                import pandas as pd
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # 2. تحلیل ساختار بازار با استفاده از utils شما
                structure = calculate_market_structure(df)
                
                # 3. تشخیص سطوح حمایت و مقاومت با استفاده از utils شما
                levels = calculate_support_resistance(df)
                
                # 4. محاسبه نوسان
                vol = calculate_volatility(df)

                # ساخت گزارش برای هر ارز
                trend_icon = "📈" if structure['trend'] == "uptrend" else "📉"
                report += f"{trend_icon} *{symbol}*\n"
                report += f"• Trend: {structure['trend']}\n"
                report += f"• Volatility: {vol:.2f}%\n"
                if levels['nearest_support']:
                    report += f"• Support: ${levels['nearest_support']['price']}\n"
                report += "------------------\n"

            # ارسال گزارش نهایی به تلگرام
            if self.config.get('telegram_token'):
                tg_bot = Bot(token=self.config['telegram_token'])
                await tg_bot.send_message(
                    chat_id=self.config['chat_id'],
                    text=report,
                    parse_mode='Markdown'
                )
            
            return "Scan Completed Successfully"

        except Exception as e:
            self.logger.error(f"Error during scan: {e}")
            return f"Error: {e}"
        finally:
            await self.exchange.close()

    async def run(self):
        """متد اجرا که در main.py فراخوانی می‌شود"""
        await self.scan_market()
