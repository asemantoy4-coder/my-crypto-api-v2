import logging
import asyncio
import ccxt.async_support as ccxt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from telegram import Bot
from telegram.constants import ParseMode
import time

# وارد کردن اندیکاتورهای ترکیبی
from indicators import CombinedIndicators
from utils import (
    calculate_market_structure, 
    calculate_support_resistance, 
    calculate_volatility,
    setup_logger,
    PerformanceTracker,
    SignalScorer
)

class FastScalpCompleteBot:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger("FastScalpBot")
        
        # تنظیمات تلگرام
        self.telegram_token = config.get('telegram_token')
        self.chat_id = config.get('chat_id')
        self.bot = Bot(token=self.telegram_token) if self.telegram_token else None
        
        # اتصال به صرافی
        self.exchange = ccxt.mexc({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        # اندیکاتورهای ترکیبی
        self.indicators = CombinedIndicators()
        
        # سیستم امتیازدهی و ردیابی
        self.signal_scorer = SignalScorer()
        self.performance_tracker = PerformanceTracker()
        
        # تنظیمات استراتژی
        self.timeframe = config.get('timeframe', '5m')
        self.top_n = config.get('top_n', 3)
        self.min_confidence = config.get('min_confidence', 65)
        self.symbols = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
            'AVAX/USDT', 'DOGE/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT'
        ]
        
        # کش برای جلوگیری از سیگنال تکراری
        self.signal_cache = {}
        
        self.logger.info("✅ Fast Scalp Bot initialized with signal generation")

    async def fetch_ohlcv_data(self, symbol: str, limit: int = 200) -> pd.DataFrame:
        """دریافت داده‌های OHLCV از صرافی"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol, 
                timeframe=self.timeframe, 
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    async def analyze_symbol(self, symbol: str) -> dict:
        """تحلیل کامل یک ارز و تولید سیگنال"""
        try:
            # دریافت داده‌ها
            df = await self.fetch_ohlcv_data(symbol)
            if df.empty or len(df) < 100:
                return None
            
            # تحلیل با اندیکاتورهای ترکیبی
            signal_data = self.indicators.generate_combined_signal(df)
            
            # اگر سیگنال NEUTRAL است یا اعتماد کافی ندارد
            if signal_data.get('signal_type') == 'NEUTRAL' or signal_data.get('confidence', 0) < self.min_confidence:
                return None
            
            # محاسبه امتیاز سیگنال
            score_result = self.signal_scorer.calculate_score(signal_data, df)
            
            # اضافه کردن اطلاعات اضافی
            signal_data.update({
                'symbol': symbol,
                'score': score_result.get('score', 0),
                'grade': score_result.get('grade', 'D'),
                'volume': float(df['volume'].iloc[-1]),
                'volume_avg': float(df['volume'].rolling(20).mean().iloc[-1]),
                'volatility': float(calculate_volatility(df)),
                'market_structure': calculate_market_structure(df),
                'support_resistance': calculate_support_resistance(df)
            })
            
            # جلوگیری از سیگنال تکراری
            cache_key = f"{symbol}_{signal_data['signal_type']}_{datetime.now().strftime('%Y%m%d%H')}"
            if cache_key in self.signal_cache:
                return None
            
            self.signal_cache[cache_key] = True
            return signal_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return None

    async def send_telegram_signal(self, symbol: str, signal: dict):
        """ارسال سیگنال معاملاتی به تلگرام"""
        try:
            if not self.bot:
                return
            
            # تعیین نوع سیگنال
            if signal['signal_type'] == "BUY":
                emoji = "🟢"
                type_fa = "خرید"
                conditions = signal.get('buy_conditions', [])
            else:
                emoji = "🔴"
                type_fa = "فروش"
                conditions = signal.get('sell_conditions', [])
            
            # محاسبه درصد تغییر
            price = signal['price']
            tp1 = signal['take_profit_1']
            tp2 = signal['take_profit_2']
            sl = signal['stop_loss']
            
            tp1_pct = ((tp1 - price) / price) * 100
            tp2_pct = ((tp2 - price) / price) * 100
            sl_pct = ((sl - price) / price) * 100
            
            # ساخت پیام
            message = f"""
{emoji} *سیگنال {type_fa} فست‌اسکلپ* {emoji}

📊 *جفت ارز:* `{symbol}`
🎯 *اعتماد:* {signal['confidence']}% (درجه: {signal.get('grade', 'B')})
💰 *قیمت فعلی:* {price:,.4f} USDT

🎯 *اهداف:*
TP1: {tp1:,.4f} ({tp1_pct:+.2f}%)
TP2: {tp2:,.4f} ({tp2_pct:+.2f}%)

🛑 *حد ضرر:* {sl:,.4f} ({sl_pct:+.2f}%)

📊 *دلایل سیگنال:*"""
            
            # اضافه کردن شرایط اصلی
            for i, condition in enumerate(conditions[:4], 1):
                message += f"\n{i}. {condition}"
            
            # اطلاعات فنی
            message += f"""
            
📈 *اطلاعات فنی:*
• حجم: {signal['volume']:,.0f} (میانگین: {signal['volume_avg']:,.0f})
• نوسان: {signal['volatility']:.2f}%
• روند: {signal['market_structure']['trend']}
• ATR: {signal.get('atr', 0):.4f}

⏰ *زمان:* {datetime.utcnow().strftime('%H:%M:%S')} UTC
📅 *تاریخ:* {datetime.utcnow().strftime('%Y/%m/%d')}

#FastScalp #{symbol.replace('/', '').replace('USDT', '')}
"""
            
            # ارسال پیام
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown',
                disable_web_page_preview=True
            )
            
            # ذخیره در ردیابی عملکرد
            self.performance_tracker.add_signal({
                'symbol': symbol,
                'type': signal['signal_type'],
                'price': price,
                'confidence': signal['confidence'],
                'timestamp': datetime.now()
            })
            
            self.logger.info(f"📤 Signal sent: {symbol} {signal['signal_type']} ({signal['confidence']}%)")
            
        except Exception as e:
            self.logger.error(f"Error sending telegram signal: {e}")

    async def send_market_report(self, signals_found: int, total_symbols: int, top_signals: list):
        """ارسال گزارش خلاصه بازار"""
        try:
            if not self.bot or signals_found == 0:
                return
            
            report = f"""
📊 *گزارش اسکن بازار*

🔍 *آمار کلی:*
• کل ارزهای اسکن شده: {total_symbols}
• سیگنال‌های یافت شده: {signals_found}
• سیگنال‌های ارسال شده: {len(top_signals)}

🏆 *سیگنال‌های برتر:*"""
            
            for i, signal in enumerate(top_signals, 1):
                emoji = "🟢" if signal['signal_type'] == "BUY" else "🔴"
                report += f"\n{i}. {emoji} `{signal['symbol']}` - اعتماد: {signal['confidence']}%"
            
            report += f"""
            
⏰ *زمان اسکن:* {datetime.utcnow().strftime('%H:%M:%S')} UTC
🔄 *اسکن بعدی:* هر ۱ ساعت

#MarketScan #{datetime.utcnow().strftime('%Y%m%d')}
"""
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=report,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            self.logger.error(f"Error sending market report: {e}")

    async def scan_market(self):
        """اسکن کامل بازار و یافتن سیگنال‌ها"""
        self.logger.info("🔄 Starting market scan...")
        
        all_signals = []
        
        # تحلیل هر ارز
        for symbol in self.symbols:
            try:
                signal = await self.analyze_symbol(symbol)
                if signal:
                    all_signals.append(signal)
                    self.logger.info(f"🎯 Signal found: {symbol} {signal['signal_type']} ({signal['confidence']}%)")
                
                await asyncio.sleep(1)  # جلوگیری از rate limit
                
            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        # انتخاب و ارسال سیگنال‌های برتر
        if all_signals:
            # مرتب‌سازی بر اساس اعتماد
            all_signals.sort(key=lambda x: x['confidence'], reverse=True)
            top_signals = all_signals[:self.top_n]
            
            # ارسال هر سیگنال
            for signal in top_signals:
                await self.send_telegram_signal(signal['symbol'], signal)
                await asyncio.sleep(2)  # فاصله بین ارسال
            
            # ارسال گزارش خلاصه
            await self.send_market_report(
                signals_found=len(all_signals),
                total_symbols=len(self.symbols),
                top_signals=top_signals
            )
            
            self.logger.info(f"✅ Scan completed: {len(all_signals)} signals found, {len(top_signals)} sent")
        else:
            self.logger.info("ℹ️ No signals found in this scan")
        
        return f"Scan completed. Found {len(all_signals)} signals."

    async def run(self):
        """اجرای اصلی ربات - اسکن هر ساعت"""
        self.logger.info("🚀 Fast Scalp Bot started")
        
        # ارسال پیام شروع
        try:
            if self.bot:
                startup_msg = f"""
🚀 *Fast Scalp Bot Started Successfully!*

📋 *Configuration:*
• Version: 2.0.0
• Timeframe: {self.timeframe}
• Scan Interval: 1 hour
• Min Confidence: {self.min_confidence}%
• Top Signals: {self.top_n}
• Symbols: {len(self.symbols)}

⏰ *Startup Time:* {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

🤖 *Bot will scan the market every hour and send top {self.top_n} signals.*

✅ *Status:* Active and Running
"""
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=startup_msg,
                    parse_mode='Markdown'
                )
        except Exception as e:
            self.logger.warning(f"Could not send startup message: {e}")
        
        try:
            # حلقه اصلی - هر 1 ساعت اسکن کن
            while True:
                try:
                    await self.scan_market()
                    self.logger.info(f"⏳ Next scan in 1 hour...")
                    await asyncio.sleep(3600)  # 1 ساعت
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(300)  # 5 دقیقه صبر و دوباره تلاش
        finally:
            # بستن اتصال صرافی
            try:
                await self.exchange.close()
                self.logger.info("✅ Exchange connection closed")
            except Exception as e:
                self.logger.error(f"Error closing exchange: {e}")
