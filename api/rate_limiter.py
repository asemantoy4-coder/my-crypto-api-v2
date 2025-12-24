"""
Rate Limiting برای API با slowapi
نسخه سازگار با Render.com
"""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, HTTPException
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# تنظیمات rate limit بر اساس نوع endpoint
RATE_LIMIT_CONFIG = {
    'default': "150/hour",          # عمومی
    'heavy': "30/minute",           # endpointهای سنگین
    'scalp': "40/minute",           # اسکالپ
    'ichimoku': "25/minute",        # ایچیموکو
    'market_data': "200/hour",      # داده بازار
    'internal': "1000/hour",        # داخلی
    'monitoring': "10/minute",      # مانیتورینگ
}

# نگاشت endpointها به نوع rate limit
ENDPOINT_LIMIT_MAP = {
    # تحلیل
    'analyze_crypto': 'heavy',
    'get_scalp_signal': 'scalp',
    'get_ichimoku_scalp_signal': 'ichimoku',
    'get_combined_analysis': 'ichimoku',
    
    # داده
    'get_market_data': 'market_data',
    'get_all_signals_endpoint': 'default',
    'get_market_overview': 'market_data',
    
    # سیستم
    'get_performance': 'monitoring',
    'system_health': 'monitoring',
    'clear_cache': 'internal',
    'scan_all_timeframes': 'heavy',
}

# ساخت Limiter با تنظیمات Render-friendly
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[RATE_LIMIT_CONFIG['default']],
    storage_uri="memory://",      # برای Render مناسب (نیاز به Redis نیست)
    strategy="fixed-window",      # ساده و کارآمد
    headers_enabled=True,         # نمایش اطلاعات در headers
    retry_after="http-date"       # فرمت استاندارد
)

def get_rate_limit_for_endpoint(endpoint_name: str) -> str:
    """دریافت limit مناسب برای endpoint"""
    return RATE_LIMIT_CONFIG.get(
        ENDPOINT_LIMIT_MAP.get(endpoint_name, 'default'),
        RATE_LIMIT_CONFIG['default']
    )

def setup_rate_limiting(app):
    """تنظیم rate limiting روی FastAPI app"""
    
    # اضافه کردن limiter به app
    app.state.limiter = limiter
    
    # اضافه کردن handler برای خطای rate limit
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Middleware برای log کردن rate limit
    @app.middleware("http")
    async def rate_limit_logging_middleware(request: Request, call_next):
        client_ip = get_remote_address(request)
        endpoint = request.url.path
        method = request.method
        
        logger.debug(f"🌐 {method} {endpoint} from {client_ip}")
        
        try:
            response = await call_next(request)
            
            # اضافه کردن headers اطلاعاتی
            limit_info = response.headers.get('X-RateLimit-Limit', '150/hour')
            remaining = response.headers.get('X-RateLimit-Remaining', 'unknown')
            
            response.headers["X-API-Version"] = "8.0.0"
            response.headers["X-RateLimit-Policy"] = "per-ip"
            
            # log برای نزدیک شدن به limit
            if remaining.isdigit() and int(remaining) < 10:
                logger.warning(f"⚠️ Low rate limit remaining for {client_ip}: {remaining}/{limit_info}")
            
            return response
            
        except RateLimitExceeded:
            logger.warning(f"🚫 Rate limit exceeded for {client_ip} on {endpoint}")
            raise
        except Exception as e:
            logger.error(f"❌ Error in rate limit middleware: {e}")
            raise
    
    logger.info("✅ Rate Limiting راه‌اندازی شد")
    logger.info(f"📊 Config: {len(RATE_LIMIT_CONFIG)} limit profiles, {len(ENDPOINT_LIMIT_MAP)} mapped endpoints")
    
    return app

# دکوراتور ساده‌تر برای استفاده
def rate_limit(limit: Optional[str] = None):
    """دکوراتور rate limit برای endpointها"""
    def decorator(func):
        # اگر limit مشخص نشده، از نگاشت استفاده کن
        if limit is None:
            limit_str = get_rate_limit_for_endpoint(func.__name__)
        else:
            limit_str = limit
        
        # استفاده از دکوراتور slowapi
        return limiter.limit(limit_str)(func)
    return decorator

# تابع کمکی برای تست rate limit
def get_client_limits(client_ip: str) -> Dict[str, Any]:
    """دریافت وضعیت limit برای یک IP خاص"""
    # این یک پیاده‌سازی ساده است
    # در نسخه واقعی از storage limiter استفاده می‌شود
    return {
        "client_ip": client_ip,
        "limits": RATE_LIMIT_CONFIG,
        "note": "Using memory storage - limits reset on restart"
    }