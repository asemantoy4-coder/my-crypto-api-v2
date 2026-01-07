import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from http.server import BaseHTTPRequestHandler
import traceback

# ۱. تنظیمات لاگ مخصوص Vercel (فقط کنسول، بدون ذخیره فایل)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ۲. اصلاح مسیرها برای پیدا کردن فایل bot.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ۳. تابع اصلی اجرای اسکن
async def simple_scan():
    """اجرای منطق بات و ارسال به تلگرام"""
    try:
        # وارد کردن کلاس بات از فایل bot.py
        from bot import FastScalpCompleteBot
        
        # دریافت تنظیمات از Environment Variables در Vercel
        config = {
            'telegram_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
            'mexc_api_key': os.getenv('MEXC_API_KEY', ''),
            'mexc_secret_key': os.getenv('MEXC_SECRET_KEY', ''),
            'timeframe': '5m',
            'top_n': 3
        }
        
        # بررسی وجود توکن‌ها
        if not config['telegram_token'] or not config['chat_id']:
            return {"success": False, "error": "Missing Telegram Token or Chat ID in Vercel Variables"}

        bot = FastScalpCompleteBot(config)
        result = await bot.scan_market()
        return result
        
    except Exception as e:
        logger.error(f"Scan Error: {str(e)}")
        return {"success": False, "error": str(e), "trace": traceback.format_exc()}

# ۴. هندلر HTTP برای Vercel
class handler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        try:
            # مسیر اصلی برای تست سلامت
            if self.path in ['/', '/api', '/health']:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    "status": "online",
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": "FastScalp Bot is ready"
                }
                self.wfile.write(json.dumps(response).encode())

            # مسیر اجرای اسکن (این آدرس را در مرورگر بزنید یا کرون‌جاب ست کنید)
            elif self.path == '/scan':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                # اجرای بخش Async
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(simple_scan())
                loop.close()
                
                self.wfile.write(json.dumps(result).encode())
            
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Endpoint not found")

        except Exception as e:
            logger.error(f"Critical Error: {str(e)}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_msg = {"error": "Internal Server Error", "details": str(e)}
            self.wfile.write(json.dumps(error_msg).encode())

# ۵. تست محلی (اختیاری)
if __name__ == "__main__":
    from http.server import HTTPServer
    print("Running local server on http://localhost:3000")
    server = HTTPServer(('localhost', 3000), handler)
    server.serve_forever()
