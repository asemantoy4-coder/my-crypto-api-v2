import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Any, Optional, Union
import yfinance as yf

logger = logging.getLogger(__name__)

# کش برای کاهش درخواست‌ها
_data_cache = {}
_cache_expiry = {}
CACHE_DURATION = 30  # ثانیه

def convert_timeframe_to_yahoo(timeframe: str) -> str:
    """
    تبدیل تایم‌فریم‌های معاملاتی به فرمت Yahoo Finance
    """
    timeframe_map = {
        # دقیقه‌ای
        '1m': '1m',
        '2m': '2m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        # ساعتی
        '1h': '60m',
        '2h': '120m',
        '4h': '240m',
        # روزانه
        '1d': '1d',
        '5d': '5d',
        # هفتگی
        '1w': '1wk',
        '1wk': '1wk',
        # ماهانه
        '1mo': '1mo',
        '3mo': '3mo'
    }
    return timeframe_map.get(timeframe, timeframe)

def convert_symbol_to_yahoo(symbol: str) -> str:
    """
    تبدیل نمادهای ارز دیجیتال به فرمت Yahoo Finance
    """
    symbol = symbol.upper().strip()
    
    # نگاشت نمادهای محبوب
    popular_symbols = {
        'BTCUSDT': 'BTC-USD',
        'ETHUSDT': 'ETH-USD',
        'BNBUSDT': 'BNB-USD',
        'SOLUSDT': 'SOL-USD',
        'XRPUSDT': 'XRP-USD',
        'ADAUSDT': 'ADA-USD',
        'DOGEUSDT': 'DOGE-USD',
        'DOTUSDT': 'DOT-USD',
        'MATICUSDT': 'MATIC-USD',
        'SHIBUSDT': 'SHIB-USD',
        'AVAXUSDT': 'AVAX-USD',
        'LINKUSDT': 'LINK-USD',
        'LTCUSDT': 'LTC-USD',
        'UNIUSDT': 'UNI-USD',
        'ATOMUSDT': 'ATOM-USD',
        'XLMUSDT': 'XLM-USD',
        'ALGOUSDT': 'ALGO-USD',
        'VETUSDT': 'VET-USD',
        'ICPUSDT': 'ICP-USD',
        'FILUSDT': 'FIL-USD',
        'ETCUSDT': 'ETC-USD',
        'XMRUSDT': 'XMR-USD',
        'AAVEUSDT': 'AAVE-USD',
        'EOSUSDT': 'EOS-USD',
        'XTZUSDT': 'XTZ-USD',
        'NEOUSDT': 'NEO-USD',
        'MKRUSDT': 'MKR-USD',
        'COMPUSDT': 'COMP-USD',
        'YFIUSDT': 'YFI-USD',
        'SANDUSDT': 'SAND-USD',
        'MANAUSDT': 'MANA-USD',
        'GALAUSDT': 'GALA-USD',
        'AXSUSDT': 'AXS-USD',
        'FTMUSDT': 'FTM-USD',
        'CRVUSDT': 'CRV-USD',
        'ENJUSDT': 'ENJ-USD',
        'THETAUSDT': 'THETA-USD',
        'ZILUSDT': 'ZIL-USD',
        'HOTUSDT': 'HOT-USD',
        'ONEUSDT': 'ONE-USD',
        'ROSEUSDT': 'ROSE-USD',
        'IOTAUSDT': 'IOTA-USD',
        'WAVESUSDT': 'WAVES-USD',
        'RVNUSDT': 'RVN-USD',
        'CHZUSDT': 'CHZ-USD',
        'ANKRUSDT': 'ANKR-USD',
        'CELOUSDT': 'CELO-USD',
        'ARUSDT': 'AR-USD',
        'RLCUSDT': 'RLC-USD',
        'STORJUSDT': 'STORJ-USD',
        'BATUSDT': 'BAT-USD',
        'ZRXUSDT': 'ZRX-USD',
        'OMGUSDT': 'OMG-USD',
        'SNXUSDT': 'SNX-USD',
        'SUSHIUSDT': 'SUSHI-USD',
        'BALUSDT': 'BAL-USD',
        'UMAUSDT': 'UMA-USD',
        'REPUSDT': 'REP-USD',
        'NMRUSDT': 'NMR-USD',
        'CVCUSDT': 'CVC-USD',
        'BANDUSDT': 'BAND-USD',
        'OXYUSDT': 'OXY-USD',
        'SRMUSDT': 'SRM-USD',
        'SKLUSDT': 'SKL-USD',
        'SXPUSDT': 'SXP-USD',
        'AUDIOUSDT': 'AUDIO-USD',
        'CTSIUSDT': 'CTSI-USD',
        'STXUSDT': 'STX-USD',
        'DGBUSDT': 'DGB-USD',
        'RSRUSDT': 'RSR-USD',
        'SCUSDT': 'SC-USD',
        'STMXUSDT': 'STMX-USD',
        'COTIUSDT': 'COTI-USD',
        'DODOUSDT': 'DODO-USD',
        'TRBUSDT': 'TRB-USD',
        'ORNUSDT': 'ORN-USD',
        'UTKUSDT': 'UTK-USD',
        'XVSUSDT': 'XVS-USD',
        'ALPHAUSDT': 'ALPHA-USD',
        'VIDTUSDT': 'VIDT-USD',
        'ATAUSDT': 'ATA-USD',
        'GTCUSDT': 'GTC-USD',
        'OGNUSDT': 'OGN-USD',
        'REEFUSDT': 'REEF-USD',
        'BELUSDT': 'BEL-USD',
        'WINGUSDT': 'WING-USD',
        'CTKUSDT': 'CTK-USD',
        'TROYUSDT': 'TROY-USD',
        'DREPUSDT': 'DREP-USD',
        'BEAMUSDT': 'BEAM-USD',
        'WRXUSDT': 'WRX-USD',
        'LTOUSDT': 'LTO-USD',
        'COSUSDT': 'COS-USD',
        'MTLUSDT': 'MTL-USD',
        'DOCKUSDT': 'DOCK-USD',
        'WANUSDT': 'WAN-USD',
        'BLZUSDT': 'BLZ-USD',
        'VITEUSDT': 'VITE-USD',
        'CHRUSDT': 'CHR-USD',
        'MDTUSDT': 'MDT-USD',
        'PERLUSDT': 'PERL-USD',
        'HARDUSDT': 'HARD-USD',
        'FORUSDT': 'FOR-USD',
        'BZRXUSDT': 'BZRX-USD',
        'CKBUSDT': 'CKB-USD',
        'TWTUSDT': 'TWT-USD',
        'FIROUSDT': 'FIRO-USD',
        'EPSUSDT': 'EPS-USD',
        'AUTOUSDT': 'AUTO-USD',
        'CAKEUSDT': 'CAKE-USD',
        'BAKEUSDT': 'BAKE-USD',
        'BURGERUSDT': 'BURGER-USD',
        'SLPUSDT': 'SLP-USD',
        'SFPUSDT': 'SFP-USD',
        'ALICEUSDT': 'ALICE-USD',
        'DEGOUSDT': 'DEGO-USD',
        'EASYUSDT': 'EASY-USD',
        'TVKUSDT': 'TVK-USD',
        'BADGERUSDT': 'BADGER-USD',
        'FISUSDT': 'FIS-USD',
        'OMUSDT': 'OM-USD',
        'PONDUSDT': 'POND-USD',
        'DEXEUSDT': 'DEXE-USD',
        'LINAUSDT': 'LINA-USD',
        'ACMUSDT': 'ACM-USD',
        'BTCBUSD': 'BTC-USD',
        'ETHBUSD': 'ETH-USD',
        'BNBBUSD': 'BNB-USD',
        'ADAUSDC': 'ADA-USD',
        'MATICBUSD': 'MATIC-USD',
        'DOGEBUSD': 'DOGE-USD',
        'XRPBUSD': 'XRP-USD',
        'DOTBUSD': 'DOT-USD',
        'UNIBUSD': 'UNI-USD',
        'LTCBUSD': 'LTC-USD',
        'LINKBUSD': 'LINK-USD',
        'BCHBUSD': 'BCH-USD',
        'XLMBUSD': 'XLM-USD',
        'VETBUSD': 'VET-USD',
        'TRXBUSD': 'TRX-USD',
        'FILBUSD': 'FIL-USD',
        'ATOMBUSD': 'ATOM-USD',
        'ETCBUSD': 'ETC-USD',
        'XTZBUSD': 'XTZ-USD',
        'EOSBUSD': 'EOS-USD',
        'AAVEBUSD': 'AAVE-USD',
        'NEOBUSD': 'NEO-USD',
        'MKRBUSD': 'MKR-USD',
        'COMPBUSD': 'COMP-USD',
        'YFIBUSD': 'YFI-USD',
        'SNXBUSD': 'SNX-USD',
        'SUSHIBUSD': 'SUSHI-USD',
        'CRVBUSD': 'CRV-USD',
        'EGLDBUSD': 'EGLD-USD',
        'ICPBUSD': 'ICP-USD',
        'SHIBBUSD': 'SHIB-USD',
        'SANDBUSD': 'SAND-USD',
        'GALABUSD': 'GALA-USD',
        'FTMBUSD': 'FTM-USD',
        'AXSBUSD': 'AXS-USD',
        'ENJBUSD': 'ENJ-USD',
        'THETABUSD': 'THETA-USD',
        'HOTBUSD': 'HOT-USD',
        'ONEINCHBUSD': '1INCH-USD',
        'CELRBUSD': 'CELR-USD',
        'CHZBUSD': 'CHZ-USD',
        'ANKRBUSD': 'ANKR-USD',
        'RVNBUSD': 'RVN-USD',
        'DGBBUSD': 'DGB-USD',
        'SCBUSD': 'SC-USD',
        'STMXBUSD': 'STMX-USD',
        'HBARBUSD': 'HBAR-USD',
        'PERPBUSD': 'PERP-USD',
        'DENTBUSD': 'DENT-USD',
        'CELOBUSD': 'CELO-USD',
        'ARDRBUSD': 'ARDR-USD',
        'IOTABUSD': 'IOTA-USD',
        'ONGUSB': 'ONG-USD',
        'ONTBUSD': 'ONT-USD',
        'QTUMBUSD': 'QTUM-USD',
        'XEMBUSD': 'XEM-USD',
        'WAVESBUSD': 'WAVES-USD',
        'ICXBUSD': 'ICX-USD',
        'OMGBUSD': 'OMG-USD',
        'NANOBUSD': 'NANO-USD',
        'RLCBUSD': 'RLC-USD',
        'ADXBUSD': 'ADX-USD',
        'POABUSD': 'POA-USD',
        'KNCBUSD': 'KNC-USD',
        'LRCBUSD': 'LRC-USD',
        'PNTBUSD': 'PNT-USD',
        'VGXBUSD': 'VGX-USD',
        'SYSBUSD': 'SYS-USD',
        'FUNBUSD': 'FUN-USD',
        'SNTBUSD': 'SNT-USD',
        'HIVEBUSD': 'HIVE-USD',
        'KAVABUSD': 'KAVA-USD',
        'RENBUSD': 'REN-USD',
        'NEBLBUSD': 'NEBL-USD',
        'VTHOBUSD': 'VTHO-USD',
        'DUSKBUSD': 'DUSK-USD',
        'BRDBUSD': 'BRD-USD',
        'NCASHBUSD': 'NCASH-USD',
        'POEBUSD': 'POE-USD',
        'WPRBUSD': 'WPR-USD',
        'QLCBUSD': 'QLC-USD',
        'REQBUSD': 'REQ-USD',
        'OSTBUSD': 'OST-USD',
        'ELFBUSD': 'ELF-USD',
        'POWRBUSD': 'POWR-USD',
        'LUNBUSD': 'LUN-USD',
        'GVTBUSD': 'GVT-USD',
        'CDTBUSD': 'CDT-USD',
        'AGIBUSD': 'AGI-USD',
        'NXSBUSD': 'NXS-USD',
        'SALTBUSD': 'SALT-USD',
        'MCOBUSD': 'MCO-USD',
        'ADABUSD': 'ADA-USD',
        'NULSBUSD': 'NULS-USD',
        'VIBBUSD': 'VIB-USD',
        'CMTBUSD': 'CMT-USD',
        'WABIBUSD': 'WABI-USD',
        'GTOBUSD': 'GTO-USD',
        'ICNBUSD': 'ICN-USD',
        'WINGSBUSD': 'WINGS-USD',
        'MTHBUSD': 'MTH-USD',
        'DPTBUSD': 'DPT-USD',
        'QKCBUSD': 'QKC-USD',
        'BSVBUSD': 'BSV-USD',
        'BCNBUSD': 'BCN-USD',
        'PPTBUSD': 'PPT-USD',
        'BCPTBUSD': 'BCPT-USD',
        'BCCBUSD': 'BCC-USD',
        'EDOBUSD': 'EDO-USD',
        'NAVBUSD': 'NAV-USD',
        'RCNBUSD': 'RCN-USD',
        'VIBEBUSD': 'VIBE-USD',
        'RPXBUSD': 'RPX-USD',
        'MODBUSD': 'MOD-USD',
        'ENGBUSD': 'ENG-USD',
        'BNTUSD': 'BNT-USD',
        'DASHUSD': 'DASH-USD',
        'ZECUSD': 'ZEC-USD',
    }
    
    # اگر نماد در لیست نگاشت بود
    if symbol in popular_symbols:
        return popular_symbols[symbol]
    
    # منطق تبدیل پیش‌فرض
    if symbol.endswith('USDT'):
        base = symbol.replace('USDT', '')
        return f"{base}-USD"
    elif symbol.endswith('BUSD'):
        base = symbol.replace('BUSD', '')
        return f"{base}-USD"
    elif symbol.endswith('USDC'):
        base = symbol.replace('USDC', '')
        return f"{base}-USD"
    elif symbol.endswith('TUSD'):
        base = symbol.replace('TUSD', '')
        return f"{base}-USD"
    else:
        # برای سایر موارد
        return f"{symbol}-USD"

def get_period_from_limit(timeframe: str, limit: int) -> str:
    """
    محاسبه دوره زمانی بر اساس تایم‌فریم و تعداد کندل‌ها
    """
    # تخمین مدت زمان مورد نیاز
    timeframe_minutes = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '4h': 240,
        '1d': 1440,
        '1w': 10080,
    }
    
    minutes_per_candle = timeframe_minutes.get(timeframe, 5)
    total_minutes = limit * minutes_per_candle
    
    # تبدیل به دوره Yahoo Finance
    if total_minutes <= 60:  # تا 1 ساعت
        return '1d'
    elif total_minutes <= 1440:  # تا 1 روز
        return '5d'
    elif total_minutes <= 10080:  # تا 1 هفته
        return '1mo'
    elif total_minutes <= 43200:  # تا 1 ماه
        return '3mo'
    elif total_minutes <= 129600:  # تا 3 ماه
        return '6mo'
    elif total_minutes <= 525600:  # تا 1 سال
        return '1y'
    else:
        return '2y'

def get_yahoo_finance_data(symbol: str, interval: str = "5m", period: str = "1d", 
                          limit: int = 100) -> Dict[str, Any]:
    """
    دریافت داده از Yahoo Finance با فرمت استاندارد
    """
    try:
        yf_symbol = convert_symbol_to_yahoo(symbol)
        yf_interval = convert_timeframe_to_yahoo(interval)
        
        logger.info(f"Fetching Yahoo Finance data: {symbol} -> {yf_symbol}, "
                   f"Interval: {yf_interval}, Period: {period}")
        
        ticker = yf.Ticker(yf_symbol)
        data = ticker.history(period=period, interval=yf_interval)
        
        if data.empty:
            logger.warning(f"No data from Yahoo Finance for {yf_symbol}")
            return {
                "status": "error",
                "message": f"No data available for {yf_symbol}",
                "symbol": symbol,
                "yahoo_symbol": yf_symbol
            }
        
        # محدود کردن به تعداد مورد نیاز
        if limit > 0 and len(data) > limit:
            data = data.tail(limit)
        
        # تبدیل به لیست کندل‌ها در فرمت استاندارد
        candles = []
        for idx, row in data.iterrows():
            # تبدیل timestamp به میلی‌ثانیه
            if hasattr(idx, 'timestamp'):
                timestamp = int(idx.timestamp() * 1000)
            else:
                timestamp = int(pd.Timestamp(idx).timestamp() * 1000)
            
            candle = [
                timestamp,  # timestamp
                float(row['Open']),  # open
                float(row['High']),  # high
                float(row['Low']),   # low
                float(row['Close']), # close
                float(row['Volume']), # volume
                timestamp + 300000,  # close_time (فرضی)
                "0",  # quote_asset_volume
                "0",  # number_of_trades
                "0",  # taker_buy_base_asset_volume
                "0",  # taker_buy_quote_asset_volume
                "0"   # ignore
            ]
            candles.append(candle)
        
        # دریافت اطلاعات تکمیلی
        info = ticker.info
        
        return {
            "status": "success",
            "symbol": symbol,
            "yahoo_symbol": yf_symbol,
            "interval": interval,
            "period": period,
            "candles": candles,
            "candle_count": len(candles),
            "current_price": float(data['Close'].iloc[-1]),
            "info": {
                "name": info.get('shortName', ''),
                "currency": info.get('currency', 'USD'),
                "market_cap": info.get('marketCap', 0),
                "day_high": info.get('dayHigh', 0),
                "day_low": info.get('dayLow', 0),
                "volume": info.get('volume', 0),
                "avg_volume": info.get('averageVolume', 0),
                "previous_close": info.get('previousClose', 0),
                "open": info.get('open', 0),
                "bid": info.get('bid', 0),
                "ask": info.get('ask', 0),
                "fifty_two_week_high": info.get('fiftyTwoWeekHigh', 0),
                "fifty_two_week_low": info.get('fiftyTwoWeekLow', 0),
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching Yahoo Finance data for {symbol}: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "symbol": symbol
        }

def get_market_data_with_fallback(symbol: str, timeframe: str = "5m", 
                                 limit: int = 100, return_source: bool = False) -> Union[List, Dict]:
    """
    دریافت داده بازار با استفاده از Yahoo Finance به عنوان منبع اصلی
    
    پارامترها:
    -----------
    symbol : str
        نماد ارز (مثلاً BTCUSDT, ETHUSDT)
    timeframe : str
        تایم‌فریم (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
    limit : int
        تعداد کندل‌های مورد نیاز
    return_source : bool
        اگر True باشد، منبع داده نیز بازگردانده می‌شود
    
    بازگشت:
    -------
    Union[List, Dict]
        اگر return_source=False: لیست کندل‌ها
        اگر return_source=True: دیکشنری شامل داده و منبع
    """
    # ایجاد کلید کش
    cache_key = f"yahoo_{symbol}_{timeframe}_{limit}"
    current_time = time.time()
    
    # بررسی کش
    if cache_key in _data_cache:
        cache_time = _cache_expiry.get(cache_key, 0)
        if current_time - cache_time < CACHE_DURATION:
            logger.debug(f"Using cached data for {symbol} ({timeframe})")
            if return_source:
                return {
                    "data": _data_cache[cache_key],
                    "source": "yahoo_finance_cache",
                    "success": True,
                    "cached": True
                }
            return _data_cache[cache_key]
    
    try:
        logger.info(f"Fetching market data for {symbol} (timeframe: {timeframe}, limit: {limit})")
        
        # محاسبه period بر اساس limit و timeframe
        period = get_period_from_limit(timeframe, limit)
        
        # دریافت داده از Yahoo Finance
        yahoo_data = get_yahoo_finance_data(symbol, timeframe, period, limit)
        
        if yahoo_data["status"] != "success":
            logger.error(f"Yahoo Finance failed for {symbol}: {yahoo_data.get('message')}")
            
            # تلاش برای استفاده از داده‌های قبلی از کش
            if _data_cache:
                # پیدا کردن نزدیک‌ترین داده‌های موجود
                for key in list(_data_cache.keys()):
                    if symbol in key:
                        logger.warning(f"Using old cached data for {symbol}")
                        old_data = _data_cache[key]
                        
                        if return_source:
                            return {
                                "data": old_data,
                                "source": "yahoo_finance_old_cache",
                                "success": True,
                                "cached": True,
                                "warning": "Using old cached data due to API failure"
                            }
                        return old_data
            
            # اگر هیچ داده‌ای نبود
            if return_source:
                return {
                    "data": [],
                    "source": "yahoo_finance",
                    "success": False,
                    "error": yahoo_data.get("message", "Unknown error")
                }
            return []
        
        candles = yahoo_data["candles"]
        
        # بررسی کیفیت داده
        if not candles or len(candles) < 10:
            logger.warning(f"Insufficient data from Yahoo Finance for {symbol}: {len(candles)} candles")
            if return_source:
                return {
                    "data": candles if candles else [],
                    "source": "yahoo_finance",
                    "success": False,
                    "warning": f"Insufficient data: {len(candles)} candles"
                }
            return candles if candles else []
        
        # ذخیره در کش
        _data_cache[cache_key] = candles
        _cache_expiry[cache_key] = current_time
        
        logger.info(f"Successfully fetched {len(candles)} candles for {symbol} from Yahoo Finance")
        
        if return_source:
            return {
                "data": candles,
                "source": "yahoo_finance",
                "success": True,
                "candle_count": len(candles),
                "current_price": yahoo_data.get("current_price", 0),
                "info": yahoo_data.get("info", {})
            }
        
        return candles
        
    except Exception as e:
        logger.error(f"Critical error in get_market_data_with_fallback for {symbol}: {e}")
        
        # تلاش نهایی: تولید داده مصنوعی در صورت شکست کامل
        try:
            logger.warning(f"Generating emergency mock data for {symbol}")
            mock_data = generate_emergency_mock_data(symbol, timeframe, limit)
            
            if return_source:
                return {
                    "data": mock_data,
                    "source": "emergency_mock",
                    "success": False,
                    "emergency": True,
                    "warning": "Using emergency mock data due to critical error"
                }
            return mock_data
            
        except Exception as mock_error:
            logger.error(f"Even emergency mock failed: {mock_error}")
            if return_source:
                return {
                    "data": [],
                    "source": "failed",
                    "success": False,
                    "error": f"Critical error: {str(e)}"
                }
            return []

def generate_emergency_mock_data(symbol: str, timeframe: str, limit: int) -> List:
    """
    تولید داده مصنوعی اضطراری در صورت شکست کامل API
    """
    try:
        # قیمت‌های پایه برای نمادهای محبوب
        base_prices = {
            'BTCUSDT': 45000.0,
            'ETHUSDT': 2400.0,
            'BNBUSDT': 310.0,
            'SOLUSDT': 100.0,
            'XRPUSDT': 0.60,
            'ADAUSDT': 0.45,
            'DOGEUSDT': 0.08,
            'DOTUSDT': 7.0,
            'MATICUSDT': 0.80,
            'SHIBUSDT': 0.000008,
            'AVAXUSDT': 35.0,
            'LINKUSDT': 14.0,
            'LTCUSDT': 70.0,
            'UNIUSDT': 6.0,
            'ATOMUSDT': 10.0,
            'XLMUSDT': 0.12,
            'ALGOUSDT': 0.18,
            'VETUSDT': 0.03,
            'ICPUSDT': 12.0,
            'FILUSDT': 5.0,
        }
        
        base_price = base_prices.get(symbol.upper(), 100.0)
        
        # تنظیم تایم‌فریم
        timeframe_ms = {
            '1m': 60000,
            '5m': 300000,
            '15m': 900000,
            '30m': 1800000,
            '1h': 3600000,
            '4h': 14400000,
            '1d': 86400000,
            '1w': 604800000
        }
        
        interval_ms = timeframe_ms.get(timeframe, 300000)
        
        # زمان فعلی
        current_time = int(time.time() * 1000)
        
        # تولید کندل‌ها
        candles = []
        price = base_price
        
        for i in range(limit):
            timestamp = current_time - ((limit - i - 1) * interval_ms)
            
            # تغییرات قیمت تصادفی (اما منطقی)
            change_percent = np.random.uniform(-0.02, 0.02)
            price = price * (1 + change_percent)
            
            open_price = price * np.random.uniform(0.995, 1.005)
            close_price = price
            high_price = max(open_price, close_price) * np.random.uniform(1.0, 1.01)
            low_price = min(open_price, close_price) * np.random.uniform(0.99, 1.0)
            volume = np.random.uniform(1000, 10000)
            
            candle = [
                int(timestamp),           # timestamp
                float(open_price),        # open
                float(high_price),        # high
                float(low_price),         # low
                float(close_price),       # close
                float(volume),            # volume
                int(timestamp + interval_ms),  # close_time
                str(float(volume) * float(close_price)),  # quote_asset_volume
                str(np.random.randint(100, 1000)),  # number_of_trades
                str(float(volume) * 0.6),  # taker_buy_base_asset_volume
                str(float(volume) * float(close_price) * 0.6),  # taker_buy_quote_asset_volume
                "0"  # ignore
            ]
            
            candles.append(candle)
        
        logger.warning(f"Generated {len(candles)} emergency mock candles for {symbol}")
        return candles
        
    except Exception as e:
        logger.error(f"Failed to generate emergency mock data: {e}")
        return []

def clear_cache():
    """
    پاک کردن کش داده‌ها
    """
    _data_cache.clear()
    _cache_expiry.clear()
    logger.info("Market data cache cleared")

def get_cache_stats() -> Dict:
    """
    دریافت آمار کش
    """
    return {
        "cache_size": len(_data_cache),
        "cache_keys": list(_data_cache.keys()),
        "timestamp": datetime.now().isoformat()
    }

# توابع کمکی برای سازگاری
def get_binance_klines_enhanced(symbol, interval="5m", limit=100):
    """
    تابع سازگاری با کد قدیمی (با Yahoo Finance کار می‌کند)
    """
    return get_market_data_with_fallback(symbol, interval, limit, return_source=False)

def get_market_data_simple(symbol):
    """
    تابع ساده برای دریافت داده (سازگاری)
    """
    data = get_market_data_with_fallback(symbol, "5m", 100, return_source=False)
    if data and len(data) > 0:
        return {
            "symbol": symbol,
            "price": float(data[-1][4]),
            "status": "success"
        }
    return {
        "status": "error",
        "message": "No data available",
        "symbol": symbol
    }

# تست توابع
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Yahoo Finance Market Data Functions...")
    print("=" * 60)
    
    # تست 1: دریافت داده ساده
    print("\n1. Testing simple market data:")
    result = get_market_data_simple("BTCUSDT")
    print(f"   BTCUSDT: {result}")
    
    # تست 2: دریافت داده با تایم‌فریم
    print("\n2. Testing with timeframe:")
    data = get_market_data_with_fallback("ETHUSDT", "15m", 50)
    print(f"   ETHUSDT 15m: {len(data)} candles")
    if data and len(data) > 0:
        print(f"   Last candle: Open={data[-1][1]}, Close={data[-1][4]}")
    
    # تست 3: دریافت با منبع
    print("\n3. Testing with source info:")
    result = get_market_data_with_fallback("BNBUSDT", "1h", 20, return_source=True)
    print(f"   BNBUSDT 1h: Source={result.get('source')}, Success={result.get('success')}")
    
    # تست 4: دریافت چندین نماد
    print("\n4. Testing multiple symbols:")
    symbols = ["SOLUSDT", "XRPUSDT", "ADAUSDT"]
    for sym in symbols:
        data = get_market_data_with_fallback(sym, "5m", 10)
        print(f"   {sym}: {len(data)} candles")
    
    # تست 5: آمار کش
    print("\n5. Cache statistics:")
    stats = get_cache_stats()
    print(f"   Cache size: {stats['cache_size']}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")