#!/usr/bin/env python3
"""
Cron job برای اجرای تحلیل‌های دوره‌ای
"""

import os
import sys
import logging
from datetime import datetime

# اضافه کردن مسیر پروژه
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import analyze_symbols, generate_dashboard_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_analysis():
    """اجرای تحلیل دوره‌ای"""
    try:
        logger.info(f"Starting scheduled analysis at {datetime.now()}")
        
        # تحلیل چند نماد مهم
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
        
        results = []
        for symbol in symbols:
            try:
                result = analyze_symbols([symbol])
                if result:
                    results.extend(result)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        # ذخیره نتایج
        if results:
            logger.info(f"Analysis completed: {len(results)} signals generated")
            
            # تولید داده برای داشبورد
            dashboard_data = generate_dashboard_data()
            logger.info("Dashboard data updated")
            
        logger.info(f"Scheduled analysis completed at {datetime.now()}")
        
    except Exception as e:
        logger.error(f"Error in scheduled analysis: {e}")

if __name__ == "__main__":
    run_analysis()