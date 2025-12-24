"""
سیستم مانیتورینگ performance برای API
"""

import time
import threading
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from collections import deque
import json

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """مانیتور performance سیستم"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PerformanceMonitor, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """مقداردهی اولیه مانیتور"""
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_response_time': 0,
            'start_time': time.time(),
            'last_reset': time.time()
        }
        
        # ذخیره تاریخچه (برای نمودارها)
        self.history = {
            'response_times': deque(maxlen=100),  # آخرین ۱۰۰ درخواست
            'endpoint_calls': {},                 # تعداد فراخوانی endpointها
            'hourly_stats': deque(maxlen=24),     # آمار ساعتی
        }
        
        # زمان‌های endpointها
        self.endpoint_timings = {}
        
        # cache hits/misses
        self.cache_stats = {'hits': 0, 'misses': 0}
        
        logger.info("✅ Performance Monitor راه‌اندازی شد")
    
    def record_request(self, endpoint: str, duration: float, success: bool = True, 
                      status_code: int = 200, client_ip: Optional[str] = None):
        """ثبت یک درخواست"""
        with self._lock:
            # آمار کلی
            self.metrics['total_requests'] += 1
            self.metrics['total_response_time'] += duration
            
            if success and status_code < 400:
                self.metrics['successful_requests'] += 1
            else:
                self.metrics['failed_requests'] += 1
            
            # تاریخچه زمان پاسخ
            self.history['response_times'].append({
                'timestamp': datetime.now().isoformat(),
                'duration_ms': round(duration * 1000, 2),
                'endpoint': endpoint,
                'status': status_code,
                'success': success
            })
            
            # آمار endpoint
            if endpoint not in self.history['endpoint_calls']:
                self.history['endpoint_calls'][endpoint] = {
                    'total_calls': 0,
                    'total_time': 0,
                    'success_calls': 0,
                    'failed_calls': 0
                }
            
            ep_stats = self.history['endpoint_calls'][endpoint]
            ep_stats['total_calls'] += 1
            ep_stats['total_time'] += duration
            
            if success:
                ep_stats['success_calls'] += 1
            else:
                ep_stats['failed_calls'] += 1
            
            # ذخیره زمان‌بندی برای endpoint
            if endpoint not in self.endpoint_timings:
                self.endpoint_timings[endpoint] = deque(maxlen=50)
            
            self.endpoint_timings[endpoint].append(duration)
    
    def record_cache(self, hit: bool):
        """ثبت hit/miss کش"""
        with self._lock:
            if hit:
                self.cache_stats['hits'] += 1
            else:
                self.cache_stats['misses'] += 1
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """دریافت متریک‌های سیستم"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory
            memory = psutil.virtual_memory()
            
            # Disk (اگر در Render باشد)
            try:
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                disk_free_gb = round(disk.free / (1024**3), 2)
            except:
                disk_percent = 0
                disk_free_gb = 0
            
            # Network
            net_io = psutil.net_io_counters()
            
            # Uptime
            uptime_seconds = time.time() - self.metrics['start_time']
            
            return {
                'cpu': {
                    'percent': cpu_percent,
                    'cores': psutil.cpu_count(logical=True)
                },
                'memory': {
                    'percent': memory.percent,
                    'available_gb': round(memory.available / (1024**3), 2),
                    'total_gb': round(memory.total / (1024**3), 2),
                    'used_gb': round(memory.used / (1024**3), 2)
                },
                'disk': {
                    'percent': disk_percent,
                    'free_gb': disk_free_gb
                },
                'network': {
                    'bytes_sent_mb': round(net_io.bytes_sent / (1024**2), 2),
                    'bytes_recv_mb': round(net_io.bytes_recv / (1024**2), 2),
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                },
                'uptime': {
                    'seconds': round(uptime_seconds, 1),
                    'hours': round(uptime_seconds / 3600, 2),
                    'days': round(uptime_seconds / 86400, 2)
                },
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ خطا در خواندن متریک‌های سیستم: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """گزارش کامل performance"""
        
        # محاسبه آمار
        total_requests = self.metrics['total_requests']
        avg_response_time = (
            self.metrics['total_response_time'] / total_requests 
            if total_requests > 0 else 0
        )
        
        success_rate = (
            (self.metrics['successful_requests'] / total_requests * 100)
            if total_requests > 0 else 100
        )
        
        # آمار cache
        total_cache = self.cache_stats['hits'] + self.cache_stats['misses']
        cache_hit_rate = (
            (self.cache_stats['hits'] / total_cache * 100)
            if total_cache > 0 else 0
        )
        
        # پیدا کردن کندترین endpointها
        slow_endpoints = []
        for endpoint, timings in self.endpoint_timings.items():
            if timings:
                avg_time = sum(timings) / len(timings)
                slow_endpoints.append({
                    'endpoint': endpoint,
                    'avg_time_ms': round(avg_time * 1000, 2),
                    'calls': len(timings),
                    'max_time_ms': round(max(timings) * 1000, 2)
                })
        
        # مرتب کردن از کند به سریع
        slow_endpoints.sort(key=lambda x: x['avg_time_ms'], reverse=True)
        
        # پرکاربردترین endpointها
        popular_endpoints = []
        for endpoint, stats in self.history['endpoint_calls'].items():
            if stats['total_calls'] > 0:
                success_rate_ep = (
                    stats['success_calls'] / stats['total_calls'] * 100
                    if stats['total_calls'] > 0 else 0
                )
                
                popular_endpoints.append({
                    'endpoint': endpoint,
                    'total_calls': stats['total_calls'],
                    'success_rate': round(success_rate_ep, 1),
                    'avg_time_ms': round(
                        (stats['total_time'] / stats['total_calls'] * 1000)
                        if stats['total_calls'] > 0 else 0, 2
                    )
                })
        
        popular_endpoints.sort(key=lambda x: x['total_calls'], reverse=True)
        
        # آخرین زمان‌های پاسخ
        recent_response_times = list(self.history['response_times'])[-10:]  # 10 تا آخر
        
        return {
            'requests': {
                'total': total_requests,
                'successful': self.metrics['successful_requests'],
                'failed': self.metrics['failed_requests'],
                'success_rate_percent': round(success_rate, 2),
                'avg_response_time_ms': round(avg_response_time * 1000, 2)
            },
            'cache': {
                'hits': self.cache_stats['hits'],
                'misses': self.cache_stats['misses'],
                'hit_rate_percent': round(cache_hit_rate, 2),
                'total_operations': total_cache
            },
            'system': self.get_system_metrics(),
            'performance': {
                'slow_endpoints': slow_endpoints[:5],  # 5 تا کندترین
                'popular_endpoints': popular_endpoints[:5],  # 5 تا پرکاربرد
                'recent_response_times': recent_response_times
            },
            'monitoring': {
                'start_time': datetime.fromtimestamp(self.metrics['start_time']).isoformat(),
                'last_reset': datetime.fromtimestamp(self.metrics['last_reset']).isoformat(),
                'uptime_hours': round((time.time() - self.metrics['start_time']) / 3600, 2)
            }
        }
    
    def reset(self):
        """ریست کردن همه متریک‌ها"""
        with self._lock:
            self._initialize()
            logger.info("📊 همه متریک‌های performance ریست شدند")
    
    def get_health_status(self) -> Dict[str, Any]:
        """وضعیت سلامت سیستم"""
        system_metrics = self.get_system_metrics()
        
        # بررسی سلامت
        checks = {
            'api_responding': True,
            'memory_ok': system_metrics.get('memory', {}).get('percent', 0) < 85,
            'cpu_ok': system_metrics.get('cpu', {}).get('percent', 0) < 80,
            'requests_flowing': self.metrics['total_requests'] > 0,
            'success_rate_ok': (
                (self.metrics['successful_requests'] / self.metrics['total_requests'] * 100) > 95
                if self.metrics['total_requests'] > 0 else True
            )
        }
        
        # وضعیت کلی
        all_healthy = all(checks.values())
        
        return {
            'status': 'healthy' if all_healthy else 'degraded',
            'checks': checks,
            'timestamp': datetime.now().isoformat(),
            'message': 'سیستم در حال اجرا است' if all_healthy else 'برخی مشکلات شناسایی شد'
        }

# Singleton instance
monitor = PerformanceMonitor()

# دکوراتور برای مانیتورینگ endpointها
def monitor_endpoint(func):
    """دکوراتور برای مانیتور کردن endpointهای FastAPI"""
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        success = True
        status_code = 200
        
        # پیدا کردن request object
        request = None
        for arg in args:
            if hasattr(arg, 'method'):  # احتمالاً FastAPI Request است
                request = arg
                break
        
        endpoint_name = func.__name__
        client_ip = None
        
        if request:
            client_ip = request.client.host if request.client else 'unknown'
        
        try:
            result = await func(*args, **kwargs)
            return result
            
        except Exception as e:
            success = False
            if hasattr(e, 'status_code'):
                status_code = e.status_code
            else:
                status_code = 500
            
            # اگر خطای rate limit باشد، log خاص
            if "rate limit" in str(e).lower():
                logger.warning(f"🚫 Rate limit hit for {client_ip} on {endpoint_name}")
            else:
                logger.error(f"❌ Error in {endpoint_name}: {e}")
            
            raise e
            
        finally:
            duration = time.time() - start_time
            monitor.record_request(
                endpoint=endpoint_name,
                duration=duration,
                success=success,
                status_code=status_code,
                client_ip=client_ip
            )
    
    return wrapper