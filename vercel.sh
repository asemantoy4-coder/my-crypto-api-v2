#!/bin/bash
# اسکریپت اجرا برای Vercel

echo "🚀 Starting Fast Scalp Bot on Vercel"
echo "====================================="

# بررسی متغیرهای محیطی ضروری
REQUIRED_VARS=("TELEGRAM_BOT_TOKEN" "TELEGRAM_CHAT_ID")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo "❌ Missing environment variables: ${MISSING_VARS[*]}"
    echo "Please set in Vercel dashboard → Environment Variables"
    exit 1
fi

# پاکسازی کش پایتون
echo "🧹 Cleaning Python cache..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# ایجاد دایرکتوری‌های لازم
mkdir -p logs .cache data

# نصب وابستگی‌ها
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir

# اجرای برنامه
echo "🤖 Starting bot..."
echo "📅 Time: $(date)"
echo "🐍 Python: $(python --version)"
echo "📁 Directory: $(pwd)"
echo "====================================="

# اجرای ربات
python main.py
