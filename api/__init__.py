"""
Crypto Trading System API Package
نسخه 8.0.0 با قابلیت‌های پیشرفته
"""

import logging
import os

logger = logging.getLogger(__name__)

# ==============================================================================
# تنظیمات اولیه
# ==============================================================================

__version__ = "8.0.0"
__author__ = "Crypto AI Trading System"
__description__ = "سیستم تحلیل معاملاتی ارز دیجیتال با کش، rate limiting و مانیتورینگ"

# ==============================================================================
# Import مدیریت شده با fallback
# ==============================================================================

def safe_import(module_name, class_name=None):
    """Import امن با fallback"""
    try:
        module = __import__(f"api.{module_name}", fromlist=[''])
        if class_name:
            return getattr(module, class_name)
        return module
    except ImportError as e:
        logger.warning(f"⚠️ {module_name} import failed: {e}")
        return None

# Import اصلی: FastAPI app
try:
    from .main import app
    logger.info("✅ Main app imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import main app: {e}")
    app = None

# Import ماژول‌های اصلی با fallback
utils_module = safe_import("utils")
data_collector_module = safe_import("data_collector")
cache_module = safe_import("cache_manager")
rate_limiter_module = safe_import("rate_limiter")
performance_module = safe_import("performance_monitor")

# اختصاص توابع برای export راحت
if utils_module:
    try:
        from .utils import (
            get_market_data_with_fallback,
            analyze_with_multi_timeframe_strategy,
            get_ichimoku_scalp_signal,
            combined_analysis
        )
        UTILS_AVAILABLE = True
    except ImportError:
        UTILS_AVAILABLE = False
else:
    UTILS_AVAILABLE = False

# Import ماژول‌های جدید
CACHE_AVAILABLE = cache_module is not None
RATE_LIMIT_AVAILABLE = rate_limiter_module is not None
MONITOR_AVAILABLE = performance_module is not None

# ==============================================================================
# Export
# ==============================================================================

__all__ = ['app', '__version__', '__description__']

# اضافه کردن ماژول‌های موجود
if UTILS_AVAILABLE:
    __all__.extend([
        'get_market_data_with_fallback',
        'analyze_with_multi_timeframe_strategy',
        'get_ichimoku_scalp_signal',
        'combined_analysis'
    ])

if CACHE_AVAILABLE:
    from .cache_manager import cache
    __all__.append('cache')

if RATE_LIMIT_AVAILABLE:
    from .rate_limiter import limiter, rate_limit
    __all__.extend(['limiter', 'rate_limit'])

if MONITOR_AVAILABLE:
    from .performance_monitor import monitor, monitor_endpoint
    __all__.extend(['monitor', 'monitor_endpoint'])

# ==============================================================================
# Startup Message
# ==============================================================================

if os.getenv("DEBUG", "false").lower() == "true":
    print("=" * 60)
    print(f"🚀 Crypto Trading System API v{__version__}")
    print("📊 Features:")
    print(f"   • Technical Analysis: {'✅' if UTILS_AVAILABLE else '❌'}")
    print(f"   • Caching System: {'✅' if CACHE_AVAILABLE else '❌'}")
    print(f"   • Rate Limiting: {'✅' if RATE_LIMIT_AVAILABLE else '❌'}")
    print(f"   • Performance Monitor: {'✅' if MONITOR_AVAILABLE else '❌'}")
    print(f"   • Ichimoku Advanced: ✅")
    print(f"   • Scalp Signals: ✅")
    print(f"   • Market Overview: ✅")
    print("=" * 60)