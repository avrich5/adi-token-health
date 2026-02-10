"""
ADI Token Health Dashboard - Backend API
Economic Observability Layer for Tokenomics Governance
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import json
import httpx
import os
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# ============================================================================
# ENVIRONMENT & APPLICATION SETUP
# ============================================================================

load_dotenv()

CMC_API_KEY = os.getenv("COINMARKETCAP_API_KEY", "")
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET", "")

app = FastAPI(
    title="ADI Token Health API",
    description="Economic Observability Layer for ADI Token",
    version="0.2.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176", "http://localhost:3000", "http://localhost:8003"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration paths
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"

# ============================================================================
# SIMPLE CACHE
# ============================================================================

class SimpleCache:
    """In-memory cache with TTL"""
    def __init__(self):
        self.cache = {}

    def get(self, key: str, ttl_seconds: int = 60) -> Optional[Any]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=ttl_seconds):
                return data
        return None

    def set(self, key: str, data: Any):
        self.cache[key] = (data, datetime.now())

cache = SimpleCache()


# ============================================================================
# LOAD CONFIGURATION FILES
# ============================================================================

def load_config(filename: str) -> dict:
    """Load JSON config file"""
    path = CONFIG_DIR / filename
    with open(path, 'r') as f:
        return json.load(f)

states_config = load_config('states_config.json')
forces_config = load_config('forces_mockup.json')
transitions_config = load_config('transitions_mockup.json')
vesting_schedule = load_config('vesting_schedule.json')
state_thresholds = load_config('state_thresholds.json')

print("✅ All configuration files loaded successfully")

# ============================================================================
# DATA FETCHING
# ============================================================================

async def fetch_btc_data() -> dict:
    """Fetch BTC price data from CoinMarketCap"""
    cached = cache.get("btc_data")
    if cached:
        return cached
    try:
        url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest"
        headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY, "Accept": "application/json"}
        params = {"symbol": "BTC", "convert": "USD"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
        token_data = data["data"]["BTC"][0]
        quote = token_data["quote"]["USD"]
        result = {
            "price_change_7d": quote["percent_change_7d"] / 100,
            "price_change_30d": quote["percent_change_30d"] / 100
        }
        cache.set("btc_data", result)
        return result
    except Exception as e:
        print(f"⚠️  BTC data fetch failed: {e}")
        return {"price_change_7d": 0.0, "price_change_30d": 0.0}


async def fetch_coinmarketcap_data(symbol: str = "ADI") -> dict:
    """Fetch price and volume data from CoinMarketCap Pro API"""
    cached = cache.get("coinmarketcap")
    if cached:
        return cached
    try:
        url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest"
        headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY, "Accept": "application/json"}
        params = {"symbol": symbol, "convert": "USD"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
        token_data = data["data"][symbol][0]
        quote = token_data["quote"]["USD"]
        result = {
            "price_usd": quote["price"],
            "volume_24h": quote["volume_24h"],
            "price_change_24h": quote["percent_change_24h"] / 100,
            "price_change_7d": quote["percent_change_7d"] / 100,
            "price_change_30d": quote["percent_change_30d"] / 100,
            "market_cap": quote["market_cap"]
        }
        cache.set("coinmarketcap", result)
        return result
    except Exception as e:
        print(f"⚠️  CoinMarketCap API failed, using fallback: {e}")
        return {
            "price_usd": 2.70, "volume_24h": 1968296,
            "price_change_24h": 0.02,
            "price_change_7d": 0.086, "price_change_30d": 0.935,
            "market_cap": 262696230
        }

async def calculate_historical_volume(symbol: str = "ADI") -> dict:
    """Calculate 7-day average volume and BTC correlation"""
    adi_data = await fetch_coinmarketcap_data(symbol)
    btc_data = await fetch_btc_data()
    volume_7d_avg = adi_data["volume_24h"] * 0.85
    adi_7d = adi_data["price_change_7d"]
    btc_7d = btc_data["price_change_7d"]
    adi_30d = adi_data["price_change_30d"]
    btc_30d = btc_data["price_change_30d"]
    if abs(adi_7d) < 0.01 or abs(btc_7d) < 0.01:
        correlation = 0.5
    else:
        same_7d = (adi_7d > 0 and btc_7d > 0) or (adi_7d < 0 and btc_7d < 0)
        same_30d = (adi_30d > 0 and btc_30d > 0) or (adi_30d < 0 and btc_30d < 0)
        if same_7d and same_30d:
            correlation = 0.7
        elif same_7d or same_30d:
            correlation = 0.5
        else:
            correlation = 0.3
    return {
        "volume_7d_avg": volume_7d_avg,
        "btc_correlation": correlation,
        "btc_change_7d": btc_7d
    }


async def fetch_kraken_ohlc(pair: str = "ADIUSD", interval: int = 1440) -> list:
    """Fetch daily OHLCV from Kraken (up to 720 candles)"""
    cached = cache.get("kraken_ohlc", ttl_seconds=300)
    if cached:
        return cached
    try:
        # Request 90 days ago
        since = int((datetime.now() - timedelta(days=90)).timestamp())
        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": pair, "interval": interval, "since": since}
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        if data.get("error") and len(data["error"]) > 0:
            raise Exception(f"Kraken OHLC error: {data['error']}")
        pair_key = [k for k in data["result"].keys() if k != "last"][0]
        raw = data["result"][pair_key]
        # Kraken OHLC format: [time, open, high, low, close, vwap, volume, count]
        candles = []
        for c in raw:
            candles.append({
                "timestamp": int(c[0]),
                "date": datetime.fromtimestamp(int(c[0])).strftime("%Y-%m-%d"),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "vwap": float(c[5]),
                "volume": float(c[6]),
                "count": int(c[7])
            })
        cache.set("kraken_ohlc", candles)
        print(f"✅ Kraken OHLC: {len(candles)} daily candles loaded")
        return candles
    except Exception as e:
        print(f"⚠️  Kraken OHLC failed: {e}")
        return []


async def fetch_kraken_orderbook(pair: str = "ADIUSD") -> dict:
    """Fetch order book from Kraken"""
    cached = cache.get("kraken_orderbook", ttl_seconds=30)
    if cached:
        return cached
    try:
        url = "https://api.kraken.com/0/public/Depth"
        params = {"pair": pair, "count": 50}
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        if data.get("error") and len(data["error"]) > 0:
            raise Exception(f"Kraken API error: {data['error']}")
        pair_key = list(data["result"].keys())[0]
        orderbook = data["result"][pair_key]
        result = {
            "bids": [[float(p), float(v)] for p, v, _ in orderbook["bids"]],
            "asks": [[float(p), float(v)] for p, v, _ in orderbook["asks"]]
        }
        cache.set("kraken_orderbook", result)
        return result
    except Exception as e:
        print(f"⚠️  Kraken API failed, using mockup orderbook: {e}")
        price = 2.70
        return {
            "bids": [[price * (1 - i * 0.001), 1000 + i * 100] for i in range(50)],
            "asks": [[price * (1 + i * 0.001), 1000 + i * 100] for i in range(50)]
        }


# ============================================================================
# CALCULATION FUNCTIONS
# ============================================================================

def calculate_resistance(orderbook: dict, current_price: float, volume_24h: float) -> float:
    """Calculate Market Resistance = cumulative_liquidity_5% / volume_24h"""
    target_price = current_price * 1.05
    cumulative_liquidity = 0
    for price, volume in orderbook["asks"]:
        if price <= target_price:
            cumulative_liquidity += price * volume
        else:
            break
    if volume_24h > 0:
        resistance = cumulative_liquidity / volume_24h
    else:
        resistance = 0.5
    return round(resistance, 2)

def calculate_spread(orderbook: dict, current_price: float) -> float:
    """Calculate bid-ask spread percentage"""
    best_bid = orderbook["bids"][0][0] if orderbook["bids"] else current_price
    best_ask = orderbook["asks"][0][0] if orderbook["asks"] else current_price
    if current_price > 0:
        return round(((best_ask - best_bid) / current_price) * 100, 4)
    return 0.0

def calculate_asymmetry(orderbook: dict) -> float:
    """Calculate bid/ask asymmetry. >0.5 = bid-heavy (buy pressure)"""
    bid_depth = sum(p * v for p, v in orderbook["bids"][:20])
    ask_depth = sum(p * v for p, v in orderbook["asks"][:20])
    total = bid_depth + ask_depth
    return round(bid_depth / total, 2) if total > 0 else 0.5


def calculate_market_pressure(price_data: dict) -> float:
    """Calculate Market Pressure force (-1.0 to +1.0)"""
    volume_24h = price_data.get("volume_24h", 0)
    volume_7d_avg = price_data.get("volume_7d_avg", volume_24h)
    if volume_7d_avg > 0:
        volume_ratio = volume_24h / volume_7d_avg
    else:
        volume_ratio = 1.0
    if volume_ratio > 1.5:
        volume_score = 0.3
    elif volume_ratio < 0.5:
        volume_score = -0.3
    else:
        volume_score = (volume_ratio - 1.0) * 0.3
    btc_correlation = price_data.get("btc_correlation", 0.5)
    price_change_7d = price_data.get("price_change_7d", 0)
    btc_change_7d = price_data.get("btc_change_7d", 0)
    if btc_correlation > 0.6:
        same_dir = (price_change_7d > 0 and btc_change_7d > 0) or (price_change_7d < 0 and btc_change_7d < 0)
        correlation_score = btc_correlation * 0.2 if same_dir else 0
    else:
        correlation_score = 0
    if price_change_7d > 0.10:
        momentum_score = 0.5
    elif price_change_7d < -0.10:
        momentum_score = -0.5
    else:
        momentum_score = price_change_7d * 3
    total = volume_score + correlation_score + momentum_score
    return round(max(-1.0, min(1.0, total)), 2)


def calculate_emission_pressure(vesting_data: dict) -> float:
    """Calculate Emission Pressure force (-1.0 to +1.0)"""
    today = datetime.now()
    upcoming_emissions = 0
    total_supply = vesting_data["total_supply"]
    tge_date = datetime.fromisoformat(vesting_data["tge_date"])
    for allocation in vesting_data["allocations"]:
        cliff_months = allocation["cliff_months"]
        vesting_months = allocation["vesting_months"]
        months_since_tge = (today.year - tge_date.year) * 12 + today.month - tge_date.month
        if months_since_tge >= cliff_months and vesting_months > 0:
            allocation_amount = (allocation["percentage"] / 100) * total_supply
            monthly_unlock = allocation_amount / vesting_months
            upcoming_emissions += monthly_unlock
    circulating = total_supply * 0.1
    emission_ratio = upcoming_emissions / circulating if circulating > 0 else 0
    if emission_ratio > 0.05:
        pressure = -0.8
    elif emission_ratio < 0.01:
        pressure = 0.5
    else:
        pressure = 0.3 - (emission_ratio * 10)
    return round(max(-1.0, min(1.0, pressure)), 2)

def get_emission_info(vesting_data: dict) -> dict:
    """Get emission schedule info for dashboard"""
    today = datetime.now()
    tge_date = datetime.fromisoformat(vesting_data["tge_date"])
    total_supply = vesting_data["total_supply"]
    months_since_tge = (today.year - tge_date.year) * 12 + today.month - tge_date.month
    # Calculate circulating
    circulating = 0
    for alloc in vesting_data["allocations"]:
        tge_release = (alloc["tge_release_pct"] / 100) * (alloc["percentage"] / 100) * total_supply
        circulating += tge_release
        if alloc["vesting_months"] > 0 and months_since_tge > alloc["cliff_months"]:
            vesting_elapsed = min(months_since_tge - alloc["cliff_months"], alloc["vesting_months"])
            alloc_total = (alloc["percentage"] / 100) * total_supply
            remaining_after_tge = alloc_total - tge_release
            circulating += (remaining_after_tge / alloc["vesting_months"]) * vesting_elapsed
    circulating_pct = round((circulating / total_supply) * 100, 1)
    # Next unlock info — approximate next month's unlock
    next_unlock_days = 30 - today.day
    if next_unlock_days <= 0:
        next_unlock_days = 30
    monthly_unlock_pct = 0
    for alloc in vesting_data["allocations"]:
        if alloc["vesting_months"] > 0 and months_since_tge >= alloc["cliff_months"]:
            monthly_unlock_pct += alloc["percentage"] / alloc["vesting_months"]
    return {
        "circulating_pct": circulating_pct,
        "next_unlock_days": next_unlock_days,
        "next_unlock_pct": round(monthly_unlock_pct, 2)
    }


# ============================================================================
# STATE CLASSIFICATION
# ============================================================================

def classify_state(metrics: dict, forces: dict) -> dict:
    """Classify economic state. Returns {id, name, category, description, confidence}"""
    resistance = metrics["resistance"]
    volume_ratio = metrics.get("volume_ratio", 1.0)
    price_7d = metrics.get("price_change_7d", 0)
    price_30d = metrics.get("price_change_30d", 0)
    volatility_ratio = metrics.get("volatility_ratio", 1.0)
    btc_correlation = metrics.get("btc_correlation", 0.5)

    market_val = forces.get("market_pressure", 0)
    utility_val = forces.get("utility_demand", 0)

    # Priority 1: Liquidity Stress (9) — absolute thin market OR relative volume drop
    # Absolute stress: resistance < 0.001 (approx $2.5k daily volume)
    if resistance < state_thresholds["resistance"]["critical"]:
        state_id, confidence = 9, 0.95
    # Relative stress: resistance < 0.01 AND volume dropped below 80% of average
    elif resistance < state_thresholds["resistance"]["low"] and volume_ratio < state_thresholds["volume_ratio"]["normal_low"]:
        state_id, confidence = 9, 0.85
        
    # Priority 2: Structural Transition (11)
    elif volatility_ratio > state_thresholds["volatility_ratio"]["high"] and abs(price_7d) > 0.1:
        state_id, confidence = 11, 0.75
        
    # Priority 3: Speculative Dominance (3)
    elif btc_correlation > 0.7 and volatility_ratio > 1.5 and price_7d > 0.15:
        state_id, confidence = 3, 0.82
        
    # Erosion Phase (12)
    elif price_30d < -0.15 and volume_ratio < 0.8:
        state_id, confidence = 12, 0.78
        
    # Utility Degradation (8)
    elif utility_val < -0.5 and market_val < 0:
        state_id, confidence = 8, 0.70
        
    # Healthy Utility Expansion (1) - requires High Resistance + Utility + Price Growth
    elif resistance > state_thresholds["resistance"]["high"] and utility_val > 0.3 and price_7d > 0:
        state_id, confidence = 1, 0.80
        
    # Liquidity-Driven Expansion (5) — High Resistance + Strong Growth
    elif resistance > state_thresholds["resistance"]["high"] and price_7d > 0.05:
        state_id, confidence = 5, 0.72
        
    # Utility-Driven Stability (2) — Medium Resistance + Stable Price
    elif resistance > state_thresholds["resistance"]["medium"] and abs(price_7d) < 0.05:
        state_id, confidence = 2, 0.78
        
    # Speculative with lower correlation (3)
    elif market_val > 0.5 and utility_val < 0:
        state_id, confidence = 3, 0.65
        
    # Utility-Market Divergence (4)
    elif utility_val > 0.3 and market_val < -0.3:
        state_id, confidence = 4, 0.68
        
    # Moderate expansion — resistance between low and medium, price up
    elif resistance > state_thresholds["resistance"]["low"] and price_7d > 0.03:
        state_id, confidence = 5, 0.55
        
    # Narrative-Driven Volatility (10)
    elif volatility_ratio > 1.5 and volume_ratio > 2.0:
        state_id, confidence = 10, 0.65
        
    # Low resistance fallback — Incentive-Driven Usage (6)
    # This catches everything with Low-to-Medium resistance that isn't Liquidity Stress
    elif resistance < state_thresholds["resistance"]["medium"]:
        state_id, confidence = 6, 0.60
        
    # Default — stable enough for Utility-Driven Stability (2)
    else:
        state_id, confidence = 2, 0.50

    state_info = next((s for s in states_config["states"] if s["id"] == state_id), None)
    return {
        "id": state_id,
        "name": state_info["name"] if state_info else "Unknown",
        "category": state_info["category"] if state_info else "caution",
        "description": state_info["description"] if state_info else "",
        "confidence": confidence
    }


# ============================================================================
# FORCE ASSEMBLY — normalized {items, net, interpretation}
# ============================================================================

def assemble_forces(market_pressure: float, emission_pressure: float) -> dict:
    """Assemble all 5 forces into normalized structure"""
    items = [
        {"id": "market_pressure", "name": "Market Pressure",
         "value": market_pressure, "is_mockup": False,
         "description": "Aggregate market sentiment and momentum"},
        {"id": "emission_pressure", "name": "Emission Pressure",
         "value": emission_pressure, "is_mockup": False,
         "description": "Token supply schedule impact"},
    ]
    # Mockup forces from config — use correct IDs: utility_demand, mm_activity, narrative
    for force_id in ["utility_demand", "mm_activity", "narrative"]:
        fc = next((f for f in forces_config["forces"] if f["id"] == force_id), None)
        if fc:
            items.append({
                "id": force_id,
                "name": fc["name"],
                "value": fc["current_value"],
                "is_mockup": True,
                "description": fc["description"]
            })
    net = round(sum(f["value"] for f in items), 2)
    if net > 0.3:
        interpretation = "Strong positive momentum"
    elif net > 0.1:
        interpretation = "Slight positive bias"
    elif net > -0.1:
        interpretation = "Roughly balanced"
    elif net > -0.3:
        interpretation = "Slight negative pressure"
    else:
        interpretation = "Strong negative pressure"
    return {"items": items, "net": net, "interpretation": interpretation}


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {"service": "ADI Token Health API", "version": "0.2.0",
            "endpoints": ["/api/data", "/api/scenario", "/health"]}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/history")
async def get_history():
    """Historical timeline: daily states + resistance for last 90 days"""
    cached = cache.get("history_timeline", ttl_seconds=300)
    if cached:
        return cached
    try:
        candles = await fetch_kraken_ohlc()
        if not candles:
            raise HTTPException(status_code=503, detail="No historical data available")

        # Compute 7-day rolling average volume for each day
        timeline = []
        emission_pressure = calculate_emission_pressure(vesting_schedule)
        # Utility/narrative mockup forces (constant for history)
        mock_forces = {f["id"]: f["current_value"] for f in forces_config["forces"]}

        for i, candle in enumerate(candles):
            price = candle["close"]
            vol = candle["volume"] * price  # Convert token volume to USD volume
            
            # 7-day rolling average volume (USD)
            lookback = candles[max(0, i-6):i+1]
            avg_vol = sum(c["volume"] * c["close"] for c in lookback) / len(lookback) if lookback else vol
            volume_ratio = vol / avg_vol if avg_vol > 0 else 1.0

            # Simplified resistance proxy (same scale as hero: cumulative_liq / volume)
            # Without orderbook we approximate: vol_usd / (fully_diluted_value * 0.05)
            # ADI fully_diluted ~51M tokens. Lower volume = lower resistance = more fragile
            fully_diluted_value = price * 51_000_000
            if fully_diluted_value > 0 and vol > 0:
                resistance_approx = round(vol / (fully_diluted_value * 0.05), 4)
            else:
                resistance_approx = 0.01

            # 7-day price change
            if i >= 7:
                price_7d_ago = candles[i-7]["close"]
                change_7d = (price - price_7d_ago) / price_7d_ago if price_7d_ago > 0 else 0
            else:
                change_7d = 0.0

            # 30-day price change
            if i >= 30:
                price_30d_ago = candles[i-30]["close"]
                change_30d = (price - price_30d_ago) / price_30d_ago if price_30d_ago > 0 else 0
            else:
                change_30d = 0.0

            # Market pressure simplified
            if change_7d > 0.10:
                market_pressure = 0.5
            elif change_7d < -0.10:
                market_pressure = -0.5
            else:
                market_pressure = round(change_7d * 3, 2)

            # Classify state
            metrics = {
                "resistance": resistance_approx,
                "volume_ratio": round(volume_ratio, 2),
                "price_change_7d": round(change_7d, 4),
                "price_change_30d": round(change_30d, 4),
                "btc_correlation": 0.5,  # Not available in historical without BTC data
                "volatility_ratio": 1.0
            }
            force_vals = {
                "market_pressure": market_pressure,
                "emission_pressure": emission_pressure,
                **mock_forces
            }
            state = classify_state(metrics, force_vals)

            timeline.append({
                "date": candle["date"],
                "timestamp": candle["timestamp"],
                "price": round(price, 4),
                "volume_usd": round(vol, 0),
                "resistance": resistance_approx,
                "state_id": state["id"],
                "state_name": state["name"],
                "category": state["category"],
                "confidence": state["confidence"],
                "change_7d_pct": round(change_7d * 100, 2)
            })

        result = {
            "timeline": timeline,
            "period_days": len(timeline),
            "data_source": "Kraken OHLC (daily)",
            "generated_at": datetime.now().isoformat()
        }
        cache.set("history_timeline", result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"❌ ERROR in history:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data")
async def get_dashboard_data():
    """Main endpoint: complete dashboard state"""
    try:
        # Fetch external data
        price_data = await fetch_coinmarketcap_data()
        historical = await calculate_historical_volume()
        orderbook = await fetch_kraken_orderbook()
        price_data.update(historical)

        # Metrics
        resistance = calculate_resistance(orderbook, price_data["price_usd"], price_data["volume_24h"])
        spread_pct = calculate_spread(orderbook, price_data["price_usd"])
        asymmetry = calculate_asymmetry(orderbook)
        volume_ratio = price_data["volume_24h"] / price_data["volume_7d_avg"] if price_data.get("volume_7d_avg", 0) > 0 else 1.0

        # Forces
        market_pressure = calculate_market_pressure(price_data)
        emission_pressure = calculate_emission_pressure(vesting_schedule)
        forces = assemble_forces(market_pressure, emission_pressure)

        # Emission info
        emission = get_emission_info(vesting_schedule)

        # Build force values dict for classifier
        force_vals = {f["id"]: f["value"] for f in forces["items"]}

        # Classify state
        metrics = {
            "resistance": resistance,
            "volume_ratio": volume_ratio,
            "price_change_7d": price_data["price_change_7d"],
            "price_change_30d": price_data["price_change_30d"],
            "btc_correlation": price_data.get("btc_correlation", 0.5),
            "volatility_ratio": 1.0
        }
        current_state = classify_state(metrics, force_vals)

        # Transitions for current state
        state_trans = transitions_config["transitions"].get(str(current_state["id"]), {})
        raw_transitions = state_trans.get("possible_transitions", [])
        top_transitions = sorted(raw_transitions, key=lambda x: x["probability"], reverse=True)[:3]
        enriched_transitions = []
        for t in top_transitions:
            target = next((s for s in states_config["states"] if s["id"] == t["to"]), None)
            enriched_transitions.append({
                "to_id": t["to"],
                "to_state": target["name"] if target else "Unknown",
                "probability": t["probability"],
                "trigger": t["trigger"],
                "window": t["window"],
                "category": target["category"] if target else "caution"
            })

        response = {
            "timestamp": datetime.now().isoformat(),
            "state": current_state,
            "price": {
                "current": round(price_data["price_usd"], 4),
                "change_24h_pct": round(price_data["price_change_24h"] * 100, 2),
                "change_7d_pct": round(price_data["price_change_7d"] * 100, 2),
                "change_30d_pct": round(price_data["price_change_30d"] * 100, 2),
                "volume_24h": round(price_data["volume_24h"], 0),
                "market_cap": round(price_data["market_cap"], 0)
            },
            "resistance": {
                "value": resistance,
                "asymmetry": asymmetry,
                "spread_pct": spread_pct
            },
            "emission": emission,
            "forces": forces,
            "transitions": enriched_transitions,
            "metadata": {
                "data_sources": ["CoinMarketCap", "Kraken"],
                "mockup_fields": ["utility_demand", "mm_activity", "narrative", "transitions"],
                "cache_ttl": 60
            }
        }
        return response
    except Exception as e:
        import traceback
        print(f"❌ ERROR in get_dashboard_data:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SCENARIO MODELING
# ============================================================================

class ScenarioRequest(BaseModel):
    adjusted_forces: Dict[str, float]

@app.post("/api/scenario")
async def analyze_scenario(request: ScenarioRequest):
    """Scenario modeling: recalculate state with adjusted forces"""
    try:
        price_data = await fetch_coinmarketcap_data()
        historical = await calculate_historical_volume()
        orderbook = await fetch_kraken_orderbook()
        price_data.update(historical)

        resistance = calculate_resistance(orderbook, price_data["price_usd"], price_data["volume_24h"])
        volume_ratio = price_data["volume_24h"] / price_data["volume_7d_avg"] if price_data.get("volume_7d_avg", 0) > 0 else 1.0

        # Build forces: apply user adjustments, keep defaults for rest
        base_market = calculate_market_pressure(price_data)
        base_emission = calculate_emission_pressure(vesting_schedule)
        base_forces = assemble_forces(base_market, base_emission)

        adjusted_items = []
        for f in base_forces["items"]:
            val = request.adjusted_forces.get(f["id"], f["value"])
            val = max(-1.0, min(1.0, val))
            adjusted_items.append({**f, "value": val, "is_adjusted": f["id"] in request.adjusted_forces})

        net = round(sum(f["value"] for f in adjusted_items), 2)
        if net > 0.3:
            interp = "Strong positive momentum"
        elif net > 0.1:
            interp = "Slight positive bias"
        elif net > -0.1:
            interp = "Roughly balanced"
        elif net > -0.3:
            interp = "Slight negative pressure"
        else:
            interp = "Strong negative pressure"

        adjusted_forces = {"items": adjusted_items, "net": net, "interpretation": interp}
        force_vals = {f["id"]: f["value"] for f in adjusted_items}

        metrics = {
            "resistance": resistance,
            "volume_ratio": volume_ratio,
            "price_change_7d": price_data["price_change_7d"],
            "price_change_30d": price_data["price_change_30d"],
            "btc_correlation": price_data.get("btc_correlation", 0.5),
            "volatility_ratio": 1.0
        }
        new_state = classify_state(metrics, force_vals)

        # Transitions for new state
        state_trans = transitions_config["transitions"].get(str(new_state["id"]), {})
        raw_trans = state_trans.get("possible_transitions", [])
        top_trans = sorted(raw_trans, key=lambda x: x["probability"], reverse=True)[:3]
        enriched = []
        for t in top_trans:
            target = next((s for s in states_config["states"] if s["id"] == t["to"]), None)
            enriched.append({
                "to_id": t["to"],
                "to_state": target["name"] if target else "Unknown",
                "probability": t["probability"],
                "trigger": t["trigger"],
                "window": t["window"],
                "category": target["category"] if target else "caution"
            })

        return {
            "scenario_timestamp": datetime.now().isoformat(),
            "state": new_state,
            "forces": adjusted_forces,
            "transitions": enriched,
            "metadata": {"forces_modified": list(request.adjusted_forces.keys())}
        }
    except Exception as e:
        import traceback
        print(f"❌ ERROR in scenario:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
