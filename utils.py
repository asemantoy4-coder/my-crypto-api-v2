import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)

# کش برای کاهش درخواست‌ها
_price_cache = {}
_cache_timestamps = {}

def convert_symbol_to_yahoo(symbol):
    """
    تبدیل نمادهای مختلف به فرمت Yahoo Finance
    """
    symbol = symbol.upper()
    
    # لیست نگاشت نمادها
    symbol_map = {
        'BTCUSDT': 'BTC-USD',
        'ETHUSDT': 'ETH-USD', 
        'BNBUSDT': 'BNB-USD',
        'SOLUSDT': 'SOL-USD',
        'XRPUSDT': 'XRP-USD',
        'ADAUSDT': 'ADA-USD',
        'DOGEUSDT': 'DOGE-USD',
        'DOTUSDT': 'DOT-USD',
        'LINKUSDT': 'LINK-USD',
        'MATICUSDT': 'MATIC-USD',
        'SHIBUSDT': 'SHIB-USD',
        'AVAXUSDT': 'AVAX-USD',
        'ATOMUSDT': 'ATOM-USD',
        'UNIUSDT': 'UNI-USD',
        'XLMUSDT': 'XLM-USD',
        'ALGOUSDT': 'ALGO-USD',
        'VETUSDT': 'VET-USD',
        'ICPUSDT': 'ICP-USD',
        'FILUSDT': 'FIL-USD',
        'ETCUSDT': 'ETC-USD',
        'THETAUSDT': 'THETA-USD',
        'XMRUSDT': 'XMR-USD',
        'EOSUSDT': 'EOS-USD',
        'AAVEUSDT': 'AAVE-USD',
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
        'KLAYUSDT': 'KLAY-USD',
        'ONEUSDT': 'ONE-USD',
        'ROSEUSDT': 'ROSE-USD',
        'IOTAUSDT': 'IOTA-USD',
        'ZILUSDT': 'ZIL-USD',
        'IOSTUSDT': 'IOST-USD',
        'WAVESUSDT': 'WAVES-USD',
        'RVNUSDT': 'RVN-USD',
        'ENJUSDT': 'ENJ-USD',
        'BATUSDT': 'BAT-USD',
        'ZRXUSDT': 'ZRX-USD',
        'OMGUSDT': 'OMG-USD',
        'SNXUSDT': 'SNX-USD',
        'SUSHIUSDT': 'SUSHI-USD',
        'BALUSDT': 'BAL-USD',
        'UMAUSDT': 'UMA-USD',
        'RENUSDT': 'REN-USD',
        'STORJUSDT': 'STORJ-USD',
        'OXTUSDT': 'OXT-USD',
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
        'ANKRUSDT': 'ANKR-USD',
        'COTIUSDT': 'COTI-USD',
        'CHZUSDT': 'CHZ-USD',
        'HOTUSDT': 'HOT-USD',
        'ARUSDT': 'AR-USD',
        'CELOUSDT': 'CELO-USD',
        'RLCUSDT': 'RLC-USD',
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
        'STMXUSDT': 'STMX-USD',
        'PERLUSDT': 'PERL-USD',
        'HARDUSDT': 'HARD-USD',
        'FORUSDT': 'FOR-USD',
        'BZRXUSDT': 'BZRX-USD',
        'RSRUSDT': 'RSR-USD',
        'CKBUSDT': 'CKB-USD',
        'TWTUSDT': 'TWT-USD',
        'FIROUSDT': 'FIRO-USD',
        'EPSUSDT': 'EPS-USD',
        'AUTOUSDT': 'AUTO-USD',
        'CAKEUSDT': 'CAKE-USD',
        'BAKEUSDT': 'BAKE-USD',
        'BURGERUSDT': 'BURGER-USD',
        'SLPUSDT': 'SLP-USD',
        'AXSUSDT': 'AXS-USD',
        'SFPUSDT': 'SFP-USD',
        'ALICEUSDT': 'ALICE-USD',
        'DEGOUSDT': 'DEGO-USD',
        'BELUSDT': 'BEL-USD',
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
        'BALBUSD': 'BAL-USD',
        'CRVBUSD': 'CRV-USD',
        'SXPBUSD': 'SXP-USD',
        'EGLDBUSD': 'EGLD-USD',
        'ICPBUSD': 'ICP-USD',
        'SHIBBUSD': 'SHIB-USD',
        'MANABUSD': 'MANA-USD',
        'SANDBUSD': 'SAND-USD',
        'GALABUSD': 'GALA-USD',
        'FTMBUSD': 'FTM-USD',
        'AXSBUSD': 'AXS-USD',
        'ENJBUSD': 'ENJ-USD',
        'THETABUSD': 'THETA-USD',
        'ZILBUSD': 'ZIL-USD',
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
        'ZRXBUSD': 'ZRX-USD',
        'BATBUSD': 'BAT-USD',
        'RLCBUSD': 'RLC-USD',
        'ADXBUSD': 'ADX-USD',
        'POABUSD': 'POA-USD',
        'STORJBUSD': 'STORJ-USD',
        'MANABUSD': 'MANA-USD',
        'KNCBUSD': 'KNC-USD',
        'REPBUSD': 'REP-USD',
        'LRCBUSD': 'LRC-USD',
        'PNTBUSD': 'PNT-USD',
        'VGXBUSD': 'VGX-USD',
        'SYSBUSD': 'SYS-USD',
        'FUNBUSD': 'FUN-USD',
        'CVCBUSD': 'CVC-USD',
        'DATAUSD': 'DATA-USD',
        'SNTBUSD': 'SNT-USD',
        'HIVEBUSD': 'HIVE-USD',
        'CHRBUSD': 'CHR-USD',
        'ARBUSD': 'AR-USD',
        'MDTBUSD': 'MDT-USD',
        'STMXBUSD': 'STMX-USD',
        'KAVABUSD': 'KAVA-USD',
        'RENBUSD': 'REN-USD',
        'NEBLBUSD': 'NEBL-USD',
        'VTHOBUSD': 'VTHO-USD',
        'DUSKBUSD': 'DUSK-USD',
        'BLZBUSD': 'BLZ-USD',
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
        'NEBLBUSD': 'NEBL-USD',
        'BRDBUSD': 'BRD-USD',
        'EDOBUSD': 'EDO-USD',
        'WINGSBUSD': 'WINGS-USD',
        'NAVBUSD': 'NAV-USD',
        'RCNBUSD': 'RCN-USD',
        'VIBEBUSD': 'VIBE-USD',
        'RPXBUSD': 'RPX-USD',
        'MODBUSD': 'MOD-USD',
        'ENGBUSD': 'ENG-USD',
        'NEBLBUSD': 'NEBL-USD',
        'BNTUSD': 'BNT-USD',
        'DASHUSD': 'DASH-USD',
        'ZECUSD': 'ZEC-USD',
        'XMRUSD': 'XMR-USD',
    }
    
    if symbol in symbol_map:
        return symbol_map[symbol]
    
    # منطق تبدیل پیش‌فرض
    if symbol.endswith('USDT'):
        return symbol.replace('USDT', '-USD')
    elif symbol.endswith('BUSD'):
        return symbol.replace('BUSD', '-USD')
    elif symbol.endswith('USDC'):
        return symbol.replace('USDC', '-USD')
    else:
        return f"{symbol}-USD"

def get_market_data(symbol, interval="1m", period="1d", use_cache=True, cache_duration=30):
    """
    دریافت داده بازار از Yahoo Finance با قابلیت‌های پیشرفته
    
    پارامترها:
    -----------
    symbol : str
        نماد ارز (مثلاً BTCUSDT, ETHUSDT)
    interval : str
        بازه زمانی (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    period : str
        دوره زمانی (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    use_cache : bool
        استفاده از کش برای کاهش درخواست‌ها
    cache_duration : int
        مدت زمان اعتبار کش (ثانیه)
    
    بازگشت:
    -------
    dict
        دیکشنری شامل داده‌های بازار
    """
    try:
        # تبدیل نماد
        yf_symbol = convert_symbol_to_yahoo(symbol)
        
        cache_key = f"{yf_symbol}_{interval}_{period}"
        current_time = time.time()
        
        # بررسی کش
        if use_cache and cache_key in _price_cache:
            cache_time = _cache_timestamps.get(cache_key, 0)
            if current_time - cache_time < cache_duration:
                logger.debug(f"Using cached data for {symbol} ({yf_symbol})")
                return _price_cache[cache_key]
        
        logger.info(f"Fetching market data for {symbol} -> {yf_symbol} ({interval}, {period})")
        
        # دریافت داده از Yahoo Finance
        ticker = yf.Ticker(yf_symbol)
        
        # تاریخچه قیمت
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            logger.warning(f"No data found for {yf_symbol}")
            return {
                "status": "error",
                "symbol": symbol,
                "yahoo_symbol": yf_symbol,
                "message": "No data available",
                "timestamp": datetime.now().isoformat()
            }
        
        # اطلاعات تکمیلی
        info = ticker.info
        current_price = data['Close'].iloc[-1] if 'Close' in data.columns else 0
        
        # محاسبه تغییرات
        if len(data) > 1:
            prev_close = data['Close'].iloc[-2]
            price_change = current_price - prev_close
            price_change_percent = (price_change / prev_close) * 100 if prev_close != 0 else 0
        else:
            price_change = 0
            price_change_percent = 0
        
        # محاسبه حجم
        volume = data['Volume'].iloc[-1] if 'Volume' in data.columns else 0
        avg_volume = data['Volume'].mean() if 'Volume' in data.columns else 0
        
        # داده خام برای تحلیل
        candles = []
        for idx, row in data.tail(100).iterrows():  # 100 کندل آخر
            candle = {
                'timestamp': idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                'open': float(row['Open']) if 'Open' in row else 0,
                'high': float(row['High']) if 'High' in row else 0,
                'low': float(row['Low']) if 'Low' in row else 0,
                'close': float(row['Close']) if 'Close' in row else 0,
                'volume': float(row['Volume']) if 'Volume' in row else 0,
            }
            candles.append(candle)
        
        # اطلاعات روزانه
        day_high = data['High'].max() if 'High' in data.columns else current_price
        day_low = data['Low'].min() if 'Low' in data.columns else current_price
        
        # آماده‌سازی نتیجه
        result = {
            "status": "success",
            "symbol": symbol,
            "yahoo_symbol": yf_symbol,
            "current_price": float(current_price),
            "price_change": float(price_change),
            "price_change_percent": float(price_change_percent),
            "volume": float(volume),
            "avg_volume": float(avg_volume),
            "day_high": float(day_high),
            "day_low": float(day_low),
            "interval": interval,
            "period": period,
            "data_points": len(data),
            "candles": candles,
            "timestamp": datetime.now().isoformat(),
            "market_info": {
                "name": info.get('shortName', ''),
                "currency": info.get('currency', 'USD'),
                "market_cap": info.get('marketCap', 0),
                "day_range": f"{day_low:.2f} - {day_high:.2f}",
                "year_range": info.get('fiftyTwoWeekRange', ''),
                "volume_avg_10d": info.get('averageVolume10days', 0),
                "volume_avg_3m": info.get('averageVolume', 0),
            }
        }
        
        # ذخیره در کش
        if use_cache:
            _price_cache[cache_key] = result
            _cache_timestamps[cache_key] = current_time
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        return {
            "status": "error",
            "symbol": symbol,
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

def get_multiple_market_data(symbols, interval="1m", period="1d"):
    """
    دریافت داده چندین نماد به صورت همزمان
    """
    results = []
    
    for symbol in symbols:
        try:
            data = get_market_data(symbol, interval, period)
            results.append(data)
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            results.append({
                "status": "error",
                "symbol": symbol,
                "message": str(e)
            })
    
    return {
        "timestamp": datetime.now().isoformat(),
        "count": len(results),
        "results": results
    }

def get_historical_data(symbol, start_date, end_date=None, interval="1d"):
    """
    دریافت داده تاریخی
    """
    try:
        yf_symbol = convert_symbol_to_yahoo(symbol)
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        ticker = yf.Ticker(yf_symbol)
        data = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            return {
                "status": "error",
                "message": "No historical data found"
            }
        
        # تبدیل به فرمت مناسب
        historical = []
        for idx, row in data.iterrows():
            historical.append({
                'date': idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': float(row['Volume']),
                'adj_close': float(row.get('Adj Close', row['Close']))
            })
        
        return {
            "status": "success",
            "symbol": symbol,
            "yahoo_symbol": yf_symbol,
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval,
            "data_count": len(historical),
            "historical_data": historical
        }
        
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

def get_real_time_quote(symbol):
    """
    دریافت قیمت لحظه‌ای با تاخیر 15 دقیقه‌ای (Yahoo Finance رایگان)
    """
    try:
        data = get_market_data(symbol, interval="2m", period="1d", use_cache=False)
        
        if data["status"] == "success":
            return {
                "status": "success",
                "symbol": symbol,
                "price": data["current_price"],
                "change": data["price_change"],
                "change_percent": data["price_change_percent"],
                "volume": data["volume"],
                "timestamp": datetime.now().isoformat(),
                "data_delay": "15 minutes",  # Yahoo Finance رایگان تاخیر دارد
                "source": "Yahoo Finance"
            }
        else:
            return data
            
    except Exception as e:
        return {
            "status": "error",
            "symbol": symbol,
            "message": str(e)
        }

# تابع اصلی برای سازگاری با کد قدیمی
def get_market_data_simple(symbol):
    """
    نسخه ساده شده برای سازگاری با کد قبلی
    """
    return get_market_data(symbol, interval="1m", period="1d")

# مثال استفاده
if __name__ == "__main__":
    # تست توابع
    print("Testing market data functions...")
    
    # تست دریافت داده ساده
    result = get_market_data_simple("BTCUSDT")
    print(f"Simple BTCUSDT data: {result.get('current_price', 'N/A')}")
    
    # تست دریافت داده پیشرفته
    result = get_market_data("ETHUSDT", interval="5m", period="5d")
    print(f"Advanced ETHUSDT data: {len(result.get('candles', []))} candles")
    
    # تست چند نماد
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    results = get_multiple_market_data(symbols)
    print(f"Multiple symbols: {results['count']} results")
    
    # تست داده تاریخی
    hist_data = get_historical_data("BTCUSDT", "2024-01-01", "2024-01-07")
    print(f"Historical data: {hist_data.get('data_count', 0)} records")