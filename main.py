import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from http.server import BaseHTTPRequestHandler
import traceback

# ۱. تنظیمات لاگ فقط برای کنسول (جلوگیری از خطای Read-only)
# حذف FileHandler که باعث ارور ۳۰ می‌شد
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ۲. اصلاح مسیرها برای پیدا کردن فایل bot.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ۳. تابع اجرای اسکن
async def simple_scan():
    """اجرای منطق بات و ارسال به تلگرام"""
    try:
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
        
        if not config['telegram_token'] or not config['chat_id']:
            return {"success": False, "error": "Missing Env Vars (Token or Chat ID)"}

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
            # مسیر اصلی
            if self.path in ['/', '/api', '/health']:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    "status": "online",
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": "Vercel Server is Running"
                }
                self.wfile.write(json.dumps(response).encode())

            # مسیر اجرای اسکن: https://your-site.vercel.app/scan
            elif self.path == '/scan':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                # اجرای بخش Async در محیط Sync
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(simple_scan())
                loop.close()
                
                self.wfile.write(json.dumps(result).encode())
            
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not Found")

        except Exception as e:
            logger.error(f"Critical Error: {str(e)}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

# ۵. تست محلی
if __name__ == "__main__":
    from http.server import HTTPServer
    print("Local: http://localhost:3000")
    server = HTTPServer(('localhost', 3000), handler)
    server.serve_forever()
