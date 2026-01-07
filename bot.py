import logging
import ccxt.async_support as ccxt
from datetime import datetime
from telegram import Bot

class FastScalpCompleteBot:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        # ایجاد شیء صرافی بدون نیاز به کلید برای دیتای عمومی
        self.exchange = ccxt.mexc({
            'enableRateLimit': True,
        })
        
    async def scan_market(self):
        """اسکن واقعی قیمت‌ها از صرافی MEXC و ارسال به تلگرام"""
        try:
            # لیست ارزهای مورد نظر برای اسکن
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
            
            # دریافت اطلاعات قیمت
            tickers = await self.exchange.fetch_tickers(symbols)
            
            # ساخت متن گزارش
            report = f"🚀 *MEXC Market Update*\n"
            report += f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}\n"
            report += "----------------------------\n"
            
            for symbol in symbols:
                if symbol in tickers:
                    price = tickers[symbol]['last']
                    change = tickers[symbol]['percentage']
                    icon = "🟢" if change >= 0 else "🔴"
                    report += f"{icon} *{symbol}*: ${price:,} ({change:+.2f}%)\n"
            
            # ارسال به تلگرام
            if self.config.get('telegram_token') and self.config.get('chat_id'):
                bot = Bot(token=self.config['telegram_token'])
                await bot.send_message(
                    chat_id=self.config['chat_id'],
                    text=report,
                    parse_mode='Markdown'
                )
                return f"Success: Reported {len(symbols)} symbols"
            
            return "Error: Telegram config missing"
            
        except Exception as e:
            self.logger.error(f"Scan Error: {str(e)}")
            return f"Error: {str(e)}"
        finally:
            # بستن کانکشن صرافی برای جلوگیری از نشت حافظه
            await self.exchange.close()
