FROM python:3.9-slim

# تنظیم مسیر کاری
WORKDIR /app

# نصب وابستگی‌های سیستم
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# آپدیت pip به آخرین نسخه
RUN python -m pip install --upgrade pip

# کپی و نصب وابستگی‌های Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# کپی کل کد پروژه
COPY . .

# ایجاد دایرکتوری‌های لازم
RUN mkdir -p logs .cache data

# پورت برای health check
EXPOSE 3000

# اجرای اسکریپت اصلی
CMD ["bash", "vercel.sh"]
