# ADI Token Health Dashboard — Implementation Log

**Project:** Economic Observability Layer for ADI Token Governance
**Status:** MVP in active development
**Last updated:** 2026-02-11
**Repository:** `/Users/andriy/VisualStudio/adi-token-health`

---

## 1. Project Purpose

Dashboard for institutional token governance that provides formalized economic state detection — not trading signals. Core question the system answers: **"Is the ecosystem ready for the next token unlock?"** rather than "What is the current market state?"

The product creates a **semantic layer** over raw market data — discrete, reproducible economic states with causal explanations suitable for state-level governance decisions.

**Target:** ADI Foundation — utility token backed by $240B IHC Abu Dhabi, ~4% of 1B tokens currently on market, compliance-first infrastructure requiring explainable, reproducible reasoning.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────┐
│  React Frontend (Vite)       localhost:5174       │
│  Dashboard.jsx → components/* → Recharts          │
├──────────────────────────────────────────────────┤
│  FastAPI Backend              localhost:8002       │
│  main.py (870 lines) → config/*.json              │
├──────────────────────────────────────────────────┤
│  Data Sources                                     │
│  • Kraken OHLC (daily candles, 90 days)           │
│  • CoinMarketCap Pro API (price, volume, BTC)     │
│  • Kraken Orderbook (real-time depth)             │
│  • Vesting Schedule (config/vesting_schedule.json)│
└──────────────────────────────────────────────────┘
```

### File Structure

```
adi-token-health/
├── backend/
│   ├── main.py              # All API logic (870 lines)
│   ├── .env                 # API keys (CMC, Kraken)
│   └── requirements.txt
├── config/
│   ├── states_config.json   # 12 economic states + short_names
│   ├── state_thresholds.json# Classification thresholds
│   ├── forces_mockup.json   # 3 synthetic force configs
│   ├── transitions_mockup.json
│   └── vesting_schedule.json# TGE 2025-12-15, allocations
├── frontend/
│   ├── src/
│   │   ├── Dashboard.jsx / .css
│   │   ├── components/
│   │   │   ├── Header.jsx / .css
│   │   │   ├── StateCard.jsx / .css
│   │   │   ├── MetricCard.jsx / .css
│   │   │   ├── ForcesPanel.jsx / .css
│   │   │   ├── Timeline.jsx / .css      # Economic State Timeline
│   │   │   ├── ForcesTimeline.jsx / .css # Force Decomposition chart
│   │   │   ├── TransitionsPanel.jsx / .css
│   │   │   └── Footer.jsx / .css
│   │   ├── hooks/useApiData.js
│   │   ├── api/client.js
│   │   ├── utils/formatters.js
│   │   └── tokens.css        # Design system variables
│   └── vite.config.js
├── docs/
│   ├── 01-state-classification-rules.md
│   ├── 02-ui-ux-design-guidelines.md
│   ├── 03-development-plan.md
│   ├── PROMPT-developer.md
│   ├── PROMPT-developer-ADDITION.md
│   └── PROMPT-designer.md
├── DEVELOPER-HANDOFF.md
└── implementation_log.md     ← this file
```

---

## 3. API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/data` | GET | Current state: price, resistance, forces, state classification, transitions |
| `/api/history` | GET | 65+ days timeline: daily states, resistance, force decomposition |
| `/api/scenario` | POST | Scenario modeling: adjust force sliders, get projected state |
| `/health` | GET | Service health check |

### Key Data Pipeline (`/api/history`)

```
Kraken OHLC (90 days) 
  → per-candle calculations:
    ├── Resistance Index (tanh-based, high-low range)
    ├── 7-day rolling volume ratio
    ├── 7d/30d price changes
    ├── Market Pressure (real: volume + BTC correlation + momentum)
    ├── Emission Pressure (real: F=m×a from vesting schedule)
    ├── Utility Demand (synthetic)
    ├── MM Activity (synthetic)
    └── Narrative (synthetic)
  → classify_state() → 1 of 12 states
  → JSON response with timeline[]
```

---

## 4. Core Metrics — What's Implemented

### 4.1 Resistance Index ✅ REAL DATA

**Formula:** `R = 1 - tanh(α × |High-Low|/Close / Volume_USD)`

**Properties:**
- Range: [0, 1] where 1.0 = maximum stability, 0.0 = fragile
- Uses **high-low range** (not close-to-close) — captures orderbook depth reality
- α = 200,000 (calibrated for ADI typical daily volumes)

**Why high-low, not close-to-close:**
Flash spikes (wicks) reveal real liquidity risks that close-to-close misses. Example: 2026-01-26 had High=$4.93, Low=$1.74 (164.8% range) but Close moved only 10.8%. A wick like that means the orderbook was empty — one order moved price 3x. Close-to-close would show R=0.94 (healthy), high-low correctly shows R=0.045 (Liquidity Stress).

**Verified edge cases:**
| Date | High-Low Range | Volume | Resistance | State |
|---|---|---|---|---|
| 2026-01-25 | normal | normal | 0.936 | Expansion |
| 2026-01-26 | 164.8% flash spike | low | 0.045 | **Stress** |
| 2026-01-27 | normal | normal | 0.899 | Expansion |

### 4.2 Market Pressure ✅ REAL DATA (3 components)

Three-component algorithm:
1. **Volume ratio** — current volume / 7d average. >1.5 = bullish signal, <0.5 = bearish
2. **BTC correlation** — directional agreement between ADI and BTC price changes (7d/30d)
3. **Price momentum** — 7-day price change, scaled ×3

Range: [-1.0, +1.0]

### 4.3 Emission Pressure ✅ REAL DATA (F=m×a model)

Physics-based model:
- **m (mass)** = monthly_unlock / circulating_supply (fraction of supply entering market)
- **a (acceleration)** = 1 / remaining_vesting_months (shorter remaining = faster pressure)
- **F** = -(m × a × 500), always negative, range [-1.0, 0.0]

Calculated per-day from actual vesting_schedule.json. TGE date: 2025-12-15.

### 4.4 Synthetic Forces ⚠️ MOCKUP (3 of 5)

| Force | Status | Description |
|---|---|---|
| Utility Demand | 🟡 synthetic | Approximated from price/volume dynamics |
| MM Activity | 🟡 synthetic | Approximated from volume ratio |
| Narrative | 🟡 synthetic | Approximated from absolute price change |

All synthetic forces follow F=m×a model with different mass/acceleration parameters. They produce plausible shapes but require real data sources (on-chain activity, MM reports, social sentiment) for production use.

---

## 5. State Classification — 12 Economic States

Defined in `config/states_config.json`:

| ID | Full Name | Short Name | Category | Color |
|---|---|---|---|---|
| 1 | Healthy Utility Expansion | Expansion | healthy | green |
| 2 | Utility-Driven Stability | Stability | healthy | green |
| 3 | Speculative Dominance | Speculation | caution | amber |
| 4 | Utility-Market Divergence | Divergence | caution | amber |
| 5 | Liquidity-Driven Expansion | Expansion | healthy | green |
| 6 | Incentive-Driven Usage | Incentive Use | caution | amber |
| 7 | Incentive Misalignment | Misalignment | warning | red |
| 8 | Utility Degradation | Degradation | warning | red |
| 9 | Liquidity Stress | Stress | warning | red |
| 10 | Narrative-Driven Volatility | Volatility | caution | amber |
| 11 | Structural Transition | Transition | caution | amber |
| 12 | Erosion Phase | Erosion | warning | red |

**Classification priority:**
1. Liquidity Stress (R < critical threshold OR R < low + volume drop)
2. Structural Transition (high volatility + large price move)
3. Speculative Dominance (high BTC correlation + high volatility + strong growth)
4. Erosion Phase (sustained 30d decline + low volume)
5. Utility Degradation (negative utility + negative market)
6. Healthy Utility Expansion (high R + positive utility + price growth)
7. Liquidity-Driven Expansion (high R + price growth)
8. Utility-Driven Stability (medium R + stable price)
9. Remaining classifications cascade down

**Note:** States 1 and 5 both use short_name "Expansion" — they are in the same category (healthy/green) and distinguishable by tooltip showing full name.

---

## 6. Frontend Components

### 6.1 Economic State Timeline ✅

- **State ribbon** — colored band at top showing state transitions over time
  - Short names displayed on segments >5 days wide
  - Full name in tooltip on hover
  - Categories: healthy (green), caution (amber), warning (red)
- **Price line** — white, left Y-axis ($USD)
- **Resistance Index line** — cyan dashed, right Y-axis (0-100%)
- Synchronized X-axis with Force Decomposition

### 6.2 Force Decomposition ✅

- **Butterfly chart** — stacked areas split positive (up) / negative (down)
- 5 forces: Market Pressure, Emission Pressure, Utility Demand, MM Activity, Narrative
- Badge: "3 of 5 forces estimated" (transparency indicator)
- Tooltip with per-force values and Net Force sum
- X-axis aligned with Economic State Timeline

### 6.3 Hero Screen ✅

- Current state card with ID and category
- Metric cards (Resistance Index, etc.)
- Data source badges ("mockup data", "3 of 5 forces estimated")
- Scenario Model button (leads to `/api/scenario`)

### 6.4 Design System

Based on ADI Foundation brand research:
- Dark background: `#0A0C18` / `#0D1117`
- Primary accent: cyan `#00DCFF`
- Categories: green `#00E5A0`, amber `#FFB547`, red `#FF5C5C`
- Typography: Inter (display), JetBrains Mono (data)
- Institutional feel, no "crypto dashboard" aesthetics

---

## 7. Development Timeline

### Day 1 (2026-02-10)

**Morning:**
- Project status assessment — backend functional, frontend absent
- ADI Foundation brand research (website, colors, typography)
- Hero screen HTML mockup with real API data
- Logo replacement with authentic ADI SVG

**Afternoon:**
- Full React frontend implementation (Vite + Recharts)
- Dashboard.jsx + 8 components built
- Live API integration (CoinMarketCap + Kraken)
- Economic State Timeline with state ribbon
- Force Decomposition — initial stacked area chart

**Evening:**
- Critical pivot: discovered force direction matters (positive/negative)
- Established F=m×a physics model for all forces
- Butterfly chart implementation (split positive/negative areas)
- TGE date correction (2025-12-15, not 2024)
- Emission pressure rewritten with per-day vesting calculation

### Day 2 (2026-02-11)

**Morning:**
- Force Decomposition visual refinements
- X-axis alignment between Timeline and Forces
- Per-day emission calculation (not monthly approximation)
- "3 of 5 forces estimated" badge

**Afternoon:**
- **Resistance Index redesign** — replaced linear market resistance with tanh-based formula
- Deep analysis of 2026-01-26 flash spike anomaly
- Decision: high-low range > close-to-close for orderbook depth assessment
- Verified: flash spike correctly triggers Liquidity Stress (R=0.045)
- Market maker failure pattern identified

**Evening:**
- State label simplification — short_name field added to all 12 states
- Ribbon labels: "Expansion" instead of "Liquidity-Driven Expansion"
- Label display threshold adjusted (>5 days) to prevent text overflow
- Implementation log created (this document)

---

## 8. What's Real vs Mockup

| Component | Status | Data Source |
|---|---|---|
| Price (OHLCV) | ✅ Real | Kraken API |
| Volume | ✅ Real | Kraken API |
| Orderbook | ✅ Real | Kraken API (real-time) |
| BTC price/correlation | ✅ Real | CoinMarketCap Pro API |
| Resistance Index | ✅ Real | Computed from OHLCV |
| Market Pressure force | ✅ Real | Computed (volume + BTC + momentum) |
| Emission Pressure force | ✅ Real | Computed from vesting schedule |
| Utility Demand force | 🟡 Synthetic | Approximated from price/volume |
| MM Activity force | 🟡 Synthetic | Approximated from volume ratio |
| Narrative force | 🟡 Synthetic | Approximated from price change |
| State transitions/probabilities | 🟡 Mockup | Config file, not yet data-driven |
| Scenario modeling | 🟡 Partial | Force sliders work, state reclassification works |

---

## 9. Known Issues & Technical Debt

1. **`import math` inside loop** — in `/api/history`, math is imported per-candle. Should move to module level.

2. **Alpha calibration** — Resistance Index α=200,000 was hand-calibrated for ADI's typical daily volumes ($2-5K). Needs systematic calibration if volumes change significantly.

3. **BTC correlation in history** — currently hardcoded to 0.5 for historical timeline (no daily BTC data fetched for each historical day). Should integrate BTC OHLC for accurate correlation.

4. **Volatility ratio** — hardcoded to 1.0 in historical. Needs actual computation from rolling std dev.

5. **State classification thresholds** — in `state_thresholds.json`, need periodic review as market matures. Currently calibrated for early-stage, low-liquidity ADI market.

6. **Cache invalidation** — simple TTL-based cache (60-300 seconds). No forced refresh mechanism.

7. **No database** — all data is computed on-the-fly from API calls. Historical data depends on Kraken's 720-candle limit for OHLC.

---

## 10. Next Steps (Roadmap)

### Immediate (this week)
- [ ] Scenario modeling UI — interactive force sliders on frontend
- [ ] Visual polish — responsive design, loading states
- [ ] Documentation consolidation — merge docs/* into single handoff

### Short-term (2-3 weeks)
- [ ] Replace synthetic forces with real data sources
- [ ] On-chain data integration for Utility Demand
- [ ] MM performance metrics from orderbook analysis
- [ ] Narrative detection from social/news feeds
- [ ] BTC correlation in historical timeline (daily BTC OHLC)

### Medium-term (1-2 months)
- [ ] State transition probability model (data-driven, not config)
- [ ] Economic memory — store state history in database
- [ ] Alerting — notify on state transitions to warning states
- [ ] Pattern Genome integration — mechanism dictionary for explanations
- [ ] Data-Driven Expansion — automated discovery of causal patterns

---

## 11. Running the Project

### Backend
```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
# Ensure .env has COINMARKETCAP_API_KEY
python main.py
# → http://localhost:8002
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# → http://localhost:5174
```

### Verify
```bash
# Health check
curl http://localhost:8002/health

# Current state
curl http://localhost:8002/api/data | python3 -m json.tool

# Historical timeline
curl http://localhost:8002/api/history | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Period: {d[\"period_days\"]} days')
for t in d['timeline'][-5:]:
    print(f'{t[\"date\"]}: R={t[\"resistance\"]:.3f} {t[\"short_name\"]} ({t[\"state_name\"]})')
"
```
