"""
سیستم کشینگ هوشمند برای API
نسخه سازگار با Render.com
"""

import time
import threading
from typing import Any, Optional, Dict, Callable
import hashlib
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class TradingCache:
    """کش مخصوص تریدینگ - thread-safe و بهینه"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TradingCache, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """مقداردهی اولیه کش"""
        self.cache: Dict[str, Dict] = {}
        self.hits = 0
        self.misses = 0
        self.max_size = 300  # مناسب برای Render (محدودیت حافظه)
        self.default_ttl = {
            'market_data': 25,      # 25 ثانیه برای داده بازار
            'analysis': 45,         # 45 ثانیه برای تحلیل
            'ichimoku': 90,         # 1.5 دقیقه برای ایچیموکو
            'signals': 15,          # 15 ثانیه برای سیگنال‌ها
            'overview': 60,         # 1 دقیقه برای نمای کلی
        }
        self.cleanup_interval = 300  # هر 5 دقیقه تمیزکاری
        self.last_cleanup = time.time()
        logger.info("✅ سیستم کش راه‌اندازی شد (Max: %s items)", self.max_size)
    
    def _make_key(self, func_name: str, *args, **kwargs) -> str:
        """تولید کلید یکتا از پارامترها"""
        key_parts = [func_name]
        key_parts.extend(str(arg) for arg in args)
        key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
        key_string = "_".join(key_parts)
        
        # استفاده از hash برای کلیدهای کوتاه‌تر
        return f"{func_name[:10]}_{hashlib.md5(key_string.encode()).hexdigest()[:12]}"
    
    def _auto_cleanup(self):
        """تمیزکاری خودکار آیتم‌های منقضی شده"""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        keys_to_delete = []
        for key, entry in self.cache.items():
            if now > entry['expires']:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self.cache[key]
        
        if keys_to_delete:
            logger.debug(f"🧹 {len(keys_to_delete)} آیتم منقضی از کش حذف شد")
        
        self.last_cleanup = now
    
    def get(self, key: str) -> Optional[Any]:
        """دریافت از کش با تمیزکاری خودکار"""
        self._auto_cleanup()
        
        if key in self.cache:
            entry = self.cache[key]
            if time.time() < entry['expires']:
                self.hits += 1
                return entry['data']
            else:
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None, func_type: str = 'default'):
        """ذخیره در کش با مدیریت اندازه"""
        if ttl is None:
            ttl = self.default_ttl.get(func_type, 30)
        
        # اگر کش پر است، قدیمی‌ترین‌ها را حذف کن
        if len(self.cache) >= self.max_size:
            self._remove_oldest(5)  # 5 تا از قدیمی‌ترین‌ها
        
        self.cache[key] = {
            'data': data,
            'expires': time.time() + ttl,
            'created': time.time(),
            'type': func_type,
            'size': len(str(data)) if isinstance(data, (str, dict, list)) else 1
        }
    
    def _remove_oldest(self, count: int = 1):
        """حذف قدیمی‌ترین آیتم‌ها"""
        if not self.cache:
            return
        
        # مرتب کردن بر اساس زمان ایجاد
        sorted_items = sorted(self.cache.items(), key=lambda x: x[1]['created'])
        
        for i in range(min(count, len(sorted_items))):
            key, _ = sorted_items[i]
            del self.cache[key]
        
        logger.debug(f"🗑️ {count} آیتم قدیمی از کش حذف شد")
    
    def cached(self, ttl: Optional[int] = None, func_type: str = 'default'):
        """دکوراتور برای کش کردن توابع"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # تولید کلید
                cache_key = self._make_key(func.__name__, *args, **kwargs)
                
                # چک کش
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    if isinstance(cached_result, dict):
                        cached_result['_cached'] = True
                        cached_result['_cache_hit'] = True
                        cached_result['_cache_key'] = cache_key
                    return cached_result
                
                # اجرای تابع اصلی
                result = func(*args, **kwargs)
                
                # ذخیره در کش
                if result is not None:
                    self.set(cache_key, result, ttl, func_type)
                    if isinstance(result, dict):
                        result['_cached'] = False
                        result['_cache_hit'] = False
                        result['_cache_key'] = cache_key
                
                return result
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict:
        """دریافت آمار کامل کش"""
        self._auto_cleanup()
        
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        # محاسبه اندازه تقریبی کش
        total_size = sum(entry.get('size', 1) for entry in self.cache.values())
        
        # گروه‌بندی آیتم‌ها بر اساس نوع
        type_counts = {}
        type_sizes = {}
        
        for entry in self.cache.values():
            t = entry['type']
            type_counts[t] = type_counts.get(t, 0) + 1
            type_sizes[t] = type_sizes.get(t, 0) + entry.get('size', 1)
        
        # قدیمی‌ترین آیتم
        oldest_age = 0
        if self.cache:
            oldest = min(self.cache.values(), key=lambda x: x['created'])
            oldest_age = time.time() - oldest['created']
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate_percent': round(hit_rate, 2),
            'current_size': len(self.cache),
            'max_size': self.max_size,
            'total_size_bytes': total_size,
            'type_distribution': type_counts,
            'type_sizes': type_sizes,
            'oldest_item_seconds': round(oldest_age, 1),
            'cleanup_last': round(time.time() - self.last_cleanup, 1)
        }
    
    def clear(self, func_type: Optional[str] = None):
        """پاک کردن کش"""
        if func_type:
            keys_to_delete = [
                k for k, v in self.cache.items() 
                if v['type'] == func_type
            ]
            for key in keys_to_delete:
                del self.cache[key]
            logger.info(f"کش نوع '{func_type}' پاک شد ({len(keys_to_delete)} آیتم)")
        else:
            self.cache.clear()
            logger.info("تمامی کش پاک شد")
        
        return True

# Singleton instance - استفاده آسان
cache = TradingCache()

# دکوراتورهای آماده برای انواع مختلف
market_data_cached = cache.cached(ttl=25, func_type='market_data')
analysis_cached = cache.cached(ttl=45, func_type='analysis')
ichimoku_cached = cache.cached(ttl=90, func_type='ichimoku')
signal_cached = cache.cached(ttl=15, func_type='signals')
overview_cached = cache.cached(ttl=60, func_type='overview')
general_cached = cache.cached(ttl=30, func_type='default')

# تابع کمکی برای endpointها
def endpoint_cache_key(request, func_name: str) -> str:
    """تولید کلید کش برای endpointهای FastAPI"""
    from fastapi import Request
    
    if not isinstance(request, Request):
        return f"endpoint_{func_name}"
    
    # استفاده از path و query parameters برای کلید
    path = request.url.path
    params = dict(request.query_params)
    
    key_parts = [func_name, path]
    key_parts.extend(f"{k}:{v}" for k, v in sorted(params.items()))
    
    key_string = "_".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()[:16]