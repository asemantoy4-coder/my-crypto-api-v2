from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import random
import uvicorn
import json
from typing import Optional, List
import asyncio
import aiohttp

app = FastAPI(
    title="Crypto Trading API v2",
    description="Advanced cryptocurrency trading signals API",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class AnalyzeRequest(BaseModel):
    symbol: str
    timeframe: str

class ScalpSignalRequest(BaseModel):
    symbol: str
    timeframe: str

class SignalResponse(BaseModel):
    symbol: str
    signal: str
    confidence: float
    rsi: Optional[float] = None
    macd: Optional[dict] = None
    bollinger_bands: Optional[dict] = None
    recommendation: Optional[str] = None
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    uptime: float
    endpoints: dict

# Global variables
app_start_time = datetime.now()
price_cache = {}
symbol_cache = {}

# External API endpoints for price data
PRICE_APIS = [
    {
        "name": "Binance",
        "url": "https://api.binance.com/api/v3/ticker/price?symbol={symbol}",
        "parser": lambda data: float(data["price"])
    },
    {
        "name": "CoinGecko",
        "url": "https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd",
        "parser": lambda data, coin_id: float(data[coin_id]["usd"]) if coin_id in data else None
    },
    {
        "name": "KuCoin",
        "url": "https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={symbol}",
        "parser": lambda data: float(data["data"]["price"]) if data["data"] else None
    }
]

# Symbol mappings
SYMBOL_MAPPINGS = {
    'BTCUSDT': {'gecko_id': 'bitcoin', 'name': 'Bitcoin'},
    'ETHUSDT': {'gecko_id': 'ethereum', 'name': 'Ethereum'},
    'BNBUSDT': {'gecko_id': 'binancecoin', 'name': 'Binance Coin'},
    'SOLUSDT': {'gecko_id': 'solana', 'name': 'Solana'},
    'XRPUSDT': {'gecko_id': 'ripple', 'name': 'Ripple'},
    'ADAUSDT': {'gecko_id': 'cardano', 'name': 'Cardano'},
    'AVAXUSDT': {'gecko_id': 'avalanche-2', 'name': 'Avalanche'},
    'DOTUSDT': {'gecko_id': 'polkadot', 'name': 'Polkadot'},
    'DOGEUSDT': {'gecko_id': 'dogecoin', 'name': 'Dogecoin'},
    'MATICUSDT': {'gecko_id': 'matic-network', 'name': 'Polygon'},
    'SHIBUSDT': {'gecko_id': 'shiba-inu', 'name': 'Shiba Inu'},
    'LTCUSDT': {'gecko_id': 'litecoin', 'name': 'Litecoin'},
    'UNIUSDT': {'gecko_id': 'uniswap', 'name': 'Uniswap'},
    'LINKUSDT': {'gecko_id': 'chainlink', 'name': 'Chainlink'},
    'ATOMUSDT': {'gecko_id': 'cosmos', 'name': 'Cosmos'},
    'XLMUSDT': {'gecko_id': 'stellar', 'name': 'Stellar'},
    'ALGOUSDT': {'gecko_id': 'algorand', 'name': 'Algorand'},
    'TRXUSDT': {'gecko_id': 'tron', 'name': 'TRON'},
    'VETUSDT': {'gecko_id': 'vechain', 'name': 'VeChain'},
    'FILUSDT': {'gecko_id': 'filecoin', 'name': 'Filecoin'},
    'AXSUSDT': {'gecko_id': 'axie-infinity', 'name': 'Axie Infinity'},
    'ETCUSDT': {'gecko_id': 'ethereum-classic', 'name': 'Ethereum Classic'},
    'FTMUSDT': {'gecko_id': 'fantom', 'name': 'Fantom'},
    'THETAUSDT': {'gecko_id': 'theta-token', 'name': 'Theta Network'},
    'EOSUSDT': {'gecko_id': 'eos', 'name': 'EOS'},
    'AAVEUSDT': {'gecko_id': 'aave', 'name': 'Aave'},
    'XTZUSDT': {'gecko_id': 'tezos', 'name': 'Tezos'},
    'SANDUSDT': {'gecko_id': 'the-sandbox', 'name': 'The Sandbox'},
    'MANAUSDT': {'gecko_id': 'decentraland', 'name': 'Decentraland'},
    'APEUSDT': {'gecko_id': 'apecoin', 'name': 'ApeCoin'},
    'GALAUSDT': {'gecko_id': 'gala', 'name': 'Gala'},
    'CHZUSDT': {'gecko_id': 'chiliz', 'name': 'Chiliz'},
    'ENJUSDT': {'gecko_id': 'enjincoin', 'name': 'Enjin'},
    'CRVUSDT': {'gecko_id': 'curve-dao-token', 'name': 'Curve DAO'},
    'ONEUSDT': {'gecko_id': 'harmony', 'name': 'Harmony'},
    'NEARUSDT': {'gecko_id': 'near', 'name': 'NEAR Protocol'},
    'ICPUSDT': {'gecko_id': 'internet-computer', 'name': 'Internet Computer'},
    'GRTUSDT': {'gecko_id': 'the-graph', 'name': 'The Graph'},
    'SNXUSDT': {'gecko_id': 'havven', 'name': 'Synthetix'},
    'COMPUSDT': {'gecko_id': 'compound', 'name': 'Compound'},
    'MKRUSDT': {'gecko_id': 'maker', 'name': 'Maker'},
    'YFIUSDT': {'gecko_id': 'yearn-finance', 'name': 'yearn.finance'},
    'SUSHIUSDT': {'gecko_id': 'sushi', 'name': 'SushiSwap'},
}

# Helper functions
async def get_price_from_api(symbol: str):
    """Get price from multiple APIs with fallback"""
    cache_key = symbol.upper()
    
    # Check cache first
    if cache_key in price_cache:
        cached = price_cache[cache_key]
        if (datetime.now() - cached['timestamp']).seconds < 30:
            return cached['price'], cached['source']
    
    # Try Binance first
    try:
        async with aiohttp.ClientSession() as session:
            # Format symbol for Binance
            binance_symbol = symbol.upper()
            if not binance_symbol.endswith('USDT'):
                binance_symbol += 'USDT'
            
            async with session.get(
                f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}",
                timeout=5
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    price = float(data['price'])
                    price_cache[cache_key] = {
                        'price': price,
                        'source': 'Binance',
                        'timestamp': datetime.now()
                    }
                    return price, 'Binance'
    except:
        pass
    
    # Try CoinGecko if symbol is mapped
    if cache_key in SYMBOL_MAPPINGS:
        try:
            async with aiohttp.ClientSession() as session:
                gecko_id = SYMBOL_MAPPINGS[cache_key]['gecko_id']
                async with session.get(
                    f"https://api.coingecko.com/api/v3/simple/price?ids={gecko_id}&vs_currencies=usd",
                    timeout=5
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if gecko_id in data:
                            price = float(data[gecko_id]['usd'])
                            price_cache[cache_key] = {
                                'price': price,
                                'source': 'CoinGecko',
                                'timestamp': datetime.now()
                            }
                            return price, 'CoinGecko'
        except:
            pass
    
    # Fallback to random price based on symbol
    base_price = 100  # Default base
    if 'BTC' in symbol:
        base_price = 50000
    elif 'ETH' in symbol:
        base_price = 3000
    elif 'SOL' in symbol:
        base_price = 100
    elif 'XRP' in symbol or 'ADA' in symbol:
        base_price = 0.5
    
    price = base_price * (0.95 + random.random() * 0.1)  # ±5% variation
    price_cache[cache_key] = {
        'price': price,
        'source': 'Fallback',
        'timestamp': datetime.now()
    }
    return price, 'Fallback'

def generate_signal(symbol: str, price: float, timeframe: str):
    """Generate trading signal based on technical analysis simulation"""
    
    # Simulate RSI (30-70 range)
    rsi = 40 + random.random() * 30
    
    # Simulate MACD
    macd = {
        'macd_line': random.uniform(-0.5, 0.5),
        'signal_line': random.uniform(-0.5, 0.5),
        'histogram': random.uniform(-0.2, 0.2)
    }
    
    # Simulate Bollinger Bands
    bb_middle = price
    bb_upper = price * (1 + random.uniform(0.02, 0.05))
    bb_lower = price * (1 - random.uniform(0.02, 0.05))
    
    bollinger_bands = {
        'upper': bb_upper,
        'middle': bb_middle,
        'lower': bb_lower,
        'width': (bb_upper - bb_lower) / bb_middle
    }
    
    # Determine signal based on simulated indicators
    signal_confidence = 0.5
    
    if rsi < 30:
        # Oversold - potential BUY
        signal = "BUY"
        signal_confidence = 0.7 + random.random() * 0.25
        recommendation = f"Oversold (RSI: {rsi:.1f}). Consider buying on dip."
    elif rsi > 70:
        # Overbought - potential SELL
        signal = "SELL"
        signal_confidence = 0.7 + random.random() * 0.25
        recommendation = f"Overbought (RSI: {rsi:.1f}). Consider taking profits."
    else:
        # Neutral zone - random signal with lower confidence
        if random.random() > 0.5:
            signal = "BUY"
        else:
            signal = "SELL"
        signal_confidence = 0.5 + random.random() * 0.2
        recommendation = f"Neutral market (RSI: {rsi:.1f}). Wait for better entry."
    
    # Adjust for timeframe
    if timeframe in ['1m', '5m']:
        # Scalp signals
        signal_confidence *= 0.9  # Slightly lower confidence for scalp
        recommendation += " ⚡ Scalp opportunity"
    elif timeframe in ['1h', '4h']:
        # Swing signals
        signal_confidence *= 1.1  # Higher confidence for swing
        recommendation += " 📊 Swing trade setup"
    
    # Add price action context
    current_time = datetime.now().strftime("%H:%M")
    recommendation += f" | Time: {current_time} | TF: {timeframe}"
    
    return {
        "signal": signal,
        "confidence": round(signal_confidence, 3),
        "rsi": round(rsi, 2),
        "macd": macd,
        "bollinger_bands": bollinger_bands,
        "recommendation": recommendation
    }

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🚀 Crypto Trading API v2.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "GET /health",
            "analyze": "POST /api/analyze",
            "scalp": "POST /api/scalp",
            "price": "GET /api/price/{symbol}",
            "symbols": "GET /api/symbols"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        timestamp=datetime.now().isoformat(),
        uptime=round(uptime, 2),
        endpoints={
            "analyze": "POST /api/analyze",
            "scalp_signal": "POST /api/scalp",
            "price": "GET /api/price/{symbol}",
            "symbols": "GET /api/symbols",
            "health": "GET /health"
        }
    )

@app.post("/api/analyze")
async def analyze_symbol(request: AnalyzeRequest):
    """Analyze a symbol for trading signals"""
    try:
        # Get current price
        price, source = await get_price_from_api(request.symbol)
        
        # Generate signal
        analysis = generate_signal(request.symbol, price, request.timeframe)
        
        return SignalResponse(
            symbol=request.symbol.upper(),
            signal=analysis["signal"],
            confidence=analysis["confidence"],
            rsi=analysis["rsi"],
            macd=analysis["macd"],
            bollinger_bands=analysis["bollinger_bands"],
            recommendation=analysis["recommendation"],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scalp")
async def scalp_signal(request: ScalpSignalRequest):
    """Generate scalp trading signals (1m, 5m timeframes)"""
    try:
        if request.timeframe not in ['1m', '5m', '15m']:
            raise HTTPException(status_code=400, detail="Invalid timeframe for scalp. Use 1m, 5m, or 15m")
        
        # Get current price
        price, source = await get_price_from_api(request.symbol)
        
        # Generate scalp-specific signal
        analysis = generate_signal(request.symbol, price, request.timeframe)
        
        # Enhance for scalp
        analysis["recommendation"] = f"⚡ SCALP SIGNAL: {analysis['recommendation']}"
        
        return SignalResponse(
            symbol=request.symbol.upper(),
            signal=analysis["signal"],
            confidence=analysis["confidence"],
            rsi=analysis["rsi"],
            recommendation=analysis["recommendation"],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/price/{symbol}")
async def get_price(symbol: str):
    """Get current price for a symbol"""
    try:
        price, source = await get_price_from_api(symbol)
        return {
            "symbol": symbol.upper(),
            "price": price,
            "source": source,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/symbols")
async def get_symbols():
    """Get list of available symbols"""
    symbols = list(SYMBOL_MAPPINGS.keys())
    
    # Add some forex pairs
    forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'XAUUSD', 'XAGUSD']
    
    return {
        "crypto": symbols,
        "forex": forex_pairs,
        "total": len(symbols) + len(forex_pairs),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/market/status")
async def market_status():
    """Get overall market status"""
    try:
        # Get BTC price as market indicator
        btc_price, source = await get_price_from_api("BTCUSDT")
        
        # Simulate market status based on BTC price movement
        if btc_price > 50000:
            status = "BULLISH"
            confidence = 0.7
        elif btc_price < 45000:
            status = "BEARISH"
            confidence = 0.6
        else:
            status = "NEUTRAL"
            confidence = 0.5
        
        return {
            "market_status": status,
            "confidence": confidence,
            "btc_price": btc_price,
            "source": source,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )