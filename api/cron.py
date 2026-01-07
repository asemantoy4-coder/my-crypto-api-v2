import os
import asyncio
import requests
from datetime import datetime

async def run_hourly_scan():
    """تابع cron برای اجرای ساعتی"""
    try:
        # دریافت آدرس پروژه
        vercel_url = os.getenv('VERCEL_URL', 'https://your-project.vercel.app')
        
        # اجرای اسکن
        response = requests.post(f"{vercel_url}/api/scan", timeout=300)
        
        print(f"Cron executed: {datetime.utcnow()} - Status: {response.status_code}")
        
    except Exception as e:
        print(f"Cron error: {e}")

# برای testing
if __name__ == "__main__":
    asyncio.run(run_hourly_scan())
