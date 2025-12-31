import requests
import time
import os

# آدرس سایت خودتان در رندر را اینجا بگذارید
# مثلا: https://my-crypto-api.onrender.com
BASE_URL = os.getenv("APP_URL", "آدرس_سایت_شما_در_رندر")

def keep_alive():
    print(f"🚀 Cron Job started for: {BASE_URL}")
    while True:
        try:
            # فراخوانی اندپوینت سلامت برای بیدار نگه داشتن سرور
            response = requests.get(f"{BASE_URL}/api/health", timeout=10)
            if response.status_code == 200:
                print(f"✅ Heartbeat sent successfully: {response.json().get('status')}")
            else:
                print(f"⚠️ Server responded with status: {response.status_code}")
        except Exception as e:
            print(f"❌ Connection error: {e}")
        
        # ۵ دقیقه انتظار (۳۰۰ ثانیه)
        time.sleep(300)

if __name__ == "__main__":
    keep_alive()