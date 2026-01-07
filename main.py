#!/usr/bin/env python3
"""
Vercel Serverless Function - Fast Scalp Bot
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from http.server import BaseHTTPRequestHandler
import traceback

# راه‌حل برای Vercel - اضافه کردن مسیرها
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# ============================================
# 🎯 تابع ساده برای تست
# ============================================

async def simple_scan():
    """یک اسکن ساده برای تست"""
    try:
        from bot import FastScalpCompleteBot
        
        config = {
            'telegram_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
            'mexc_api_key': os.getenv('MEXC_API_KEY', ''),
            'mexc_secret_key': os.getenv('MEXC_SECRET_KEY', ''),
            'timeframe': '5m',
            'top_n': 3,
            'update_interval': 3600,
            'max_symbols': 5  # کاهش برای تست
        }
        
        bot = FastScalpCompleteBot(config)
        result = await bot.scan_market()
        return {"success": True, "result": result}
        
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

# ============================================
# 🚀 HTTP Handler برای Vercel
# ============================================

class handler(BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        """غیرفعال کردن logهای پیش‌فرض"""
        pass
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path == '/' or self.path == '/api':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                response = {
                    "status": "running",
                    "service": "fast-scalp-bot",
                    "time": datetime.utcnow().isoformat(),
                    "version": "1.0.0",
                    "endpoints": [
                        "/health",
                        "/scan",
                        "/test"
                    ]
                }
                self.wfile.write(json.dumps(response, indent=2).encode())
                
            elif self.path == '/health':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                response = {
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "environment": os.getenv('VERCEL_ENV', 'development')
                }
                self.wfile.write(json.dumps(response, indent=2).encode())
                
            elif self.path == '/test':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                # تست ساده بدون وابستگی‌های خارجی
                response = {
                    "test": "success",
                    "python": sys.version,
                    "path": sys.path,
                    "env_keys": list(os.environ.keys())[:5]  # فقط ۵ تا اول
                }
                self.wfile.write(json.dumps(response, indent=2).encode())
                
            elif self.path == '/scan':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                # اجرای اسکن در background
                try:
                    # ایجاد event loop جدید
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # اجرای اسکن
                    result = loop.run_until_complete(simple_scan())
                    loop.close()
                    
                    response = {
                        "status": "scan_completed",
                        "result": result,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                except Exception as e:
                    response = {
                        "status": "error",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                self.wfile.write(json.dumps(response, indent=2).encode())
                
            else:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {"error": "Endpoint not found", "path": self.path}
                self.wfile.write(json.dumps(response, indent=2).encode())
                
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            error_response = {
                "error": "Internal server error",
                "message": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "timestamp": datetime.utcnow().isoformat()
            }
            self.wfile.write(json.dumps(error_response, indent=2).encode())

# ============================================
# 🧪 برای تست محلی
# ============================================

if __name__ == "__main__":
    from http.server import HTTPServer
    
    print("🚀 Starting local server on http://localhost:3000")
    print("📁 Current directory:", os.getcwd())
    print("🐍 Python path:", sys.path)
    
    server = HTTPServer(('localhost', 3000), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
