#!/usr/bin/env python3
"""
Vercel Serverless Function برای Fast Scalp Bot
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from http.server import BaseHTTPRequestHandler

# اضافه کردن مسیر ماژول‌ها
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from bot import FastScalpCompleteBot
    from utils import setup_logger
except ImportError as e:
    print(f"Import error: {e}")
    # ساخت mock برای زمانی که ماژول‌ها موجود نیستند
    class FastScalpCompleteBot:
        def __init__(self, config):
            self.config = config
        async def run(self):
            return {"status": "mock bot"}

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "status": "running",
                "service": "fast-scalp-bot",
                "time": datetime.utcnow().isoformat(),
                "version": "1.0.0"
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
            self.wfile.write(json.dumps(response).encode())
            return
        
        elif self.path == '/scan':
            # اجرای اسکن بازار
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            try:
                # اجرای اسکن در background
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                config = {
                    'telegram_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
                    'chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
                    'mexc_api_key': os.getenv('MEXC_API_KEY', ''),
                    'mexc_secret_key': os.getenv('MEXC_SECRET_KEY', '')
                }
                
                bot = FastScalpCompleteBot(config)
                
                # اجرای یک اسکن
                import threading
                def run_scan():
                    try:
                        loop.run_until_complete(bot.scan_market())
                    except Exception as e:
                        print(f"Scan error: {e}")
                
                thread = threading.Thread(target=run_scan, daemon=True)
                thread.start()
                
                response = {
                    "status": "scan_started",
                    "message": "Market scan initiated in background",
                    "time": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                response = {
                    "status": "error",
                    "message": str(e),
                    "time": datetime.utcnow().isoformat()
                }
            
            self.wfile.write(json.dumps(response).encode())
            return
        
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"error": "Not found", "path": self.path}
            self.wfile.write(json.dumps(response).encode())
            return

# برای local testing
if __name__ == "__main__":
    from http.server import HTTPServer
    server = HTTPServer(('localhost', 3000), handler)
    print("Server running on http://localhost:3000")
    server.serve_forever()
