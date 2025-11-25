<!-- 17e3f31c-8d7c-4ab2-89be-88de3d83d16e 5218c605-6ae5-44ee-9b90-3546b9681744 -->
# Adaptive AI Trading System + Advanced Dashboard

## Part 1: Adaptive AI Engine

### 1.1 Create AI Strategy Class (`src/strategy/adaptive_ai.py`)

New file that replaces the simple RSI logic with intelligent AI:

```python
class AdaptiveAI:
    - Loads XGBoost model from models/xgboost_model.json
    - Maintains rolling window of bars (1-min data)
    - Calculates features using TA-Lib (RSI, MACD, BB, ATR)
    - Evaluates EVERY bar for opportunities
    - Tracks confidence levels and triggers
```

**Key Features:**

- Multi-timeframe analysis (aggregates 1-min into 5-min and 15-min views)
- Opportunity detection (RSI extremes, volume spikes, breakouts)
- Confidence scoring (0-100%) based on AI prediction + indicator alignment
- Only trades when confidence exceeds threshold (default 70%)

### 1.2 Update Multi-Asset Bot (`src/execution/bot_multi.py`)

Modify to use AdaptiveAI instead of simple indicators:

- Import and initialize AdaptiveAI class
- Pass each bar to AI for evaluation
- AI returns: signal, confidence, reasoning, all indicator values
- Execute trades based on AI decision
- Broadcast full AI state to dashboard

### 1.3 Enhanced Data Broadcasting

Update the WebSocket broadcast to include:

```python
"ai": {
    "prediction": 0.73,           # Raw model output
    "confidence": 78,             # Overall confidence %
    "signal": "BUY",              # Final decision
    "reasoning": ["RSI oversold", "Volume spike", "Model bullish"],
    "features": {
        "rsi": 28.5,
        "macd": 0.0023,
        "bb_position": 0.15,      # % within bands
        "atr_pct": 1.2,
        "volume_ratio": 2.3
    },
    "thresholds": {
        "buy_threshold": 0.65,
        "sell_threshold": 0.35,
        "min_confidence": 70
    },
    "multi_timeframe": {
        "1min": "BULLISH",
        "5min": "NEUTRAL",
        "15min": "BULLISH"
    }
}
```

---

## Part 2: Advanced Dashboard

### 2.1 AI Decision Flow Panel (New Component)

Visual representation of how AI makes decisions:

```
┌─────────────────────────────────────────┐
│         AI DECISION FLOW                │
├─────────────────────────────────────────┤
│                                         │
│  [RAW DATA] ──► [FEATURES] ──► [MODEL]  │
│      │              │            │      │
│   Price         RSI: 28        73%      │
│   Volume        MACD: +        ──►      │
│   OHLC          BB: low        BUY      │
│                                         │
│  ──────────────────────────────────     │
│  CONFIDENCE METER:  [████████░░] 78%    │
│  ──────────────────────────────────     │
│                                         │
│  REASONING:                             │
│  ✓ RSI oversold (28.5 < 30)            │
│  ✓ Volume spike (2.3x average)         │
│  ✓ Model prediction bullish (73%)      │
│  ○ MACD neutral                         │
│                                         │
└─────────────────────────────────────────┘
```

### 2.2 Multi-Timeframe View (New Component)

Show how signals look across timeframes:

```
┌─────────────────────────────────────────┐
│       MULTI-TIMEFRAME ANALYSIS          │
├─────────────────────────────────────────┤
│  1 MIN   [▲ BULLISH]  RSI: 28  Vol: 2x │
│  5 MIN   [─ NEUTRAL]  RSI: 45  Vol: 1x │
│  15 MIN  [▲ BULLISH]  RSI: 35  Vol: 1.5x│
├─────────────────────────────────────────┤
│  ALIGNMENT: 2/3 Bullish = TRADE OK     │
└─────────────────────────────────────────┘
```

### 2.3 Feature Importance Panel (New Component)

Show which indicators matter most to AI:

```
┌─────────────────────────────────────────┐
│       WHAT AI IS LOOKING AT             │
├─────────────────────────────────────────┤
│  RSI             [████████████] 35%     │
│  Volume          [████████░░░░] 25%     │
│  MACD            [██████░░░░░░] 18%     │
│  Bollinger       [████░░░░░░░░] 12%     │
│  Price Action    [███░░░░░░░░░] 10%     │
└─────────────────────────────────────────┘
```

### 2.4 Live Trade Reasoning (New Component)

When a trade happens, show exactly why:

```
┌─────────────────────────────────────────┐
│  LAST TRADE: BUY BTC @ $87,050          │
├─────────────────────────────────────────┤
│  WHY AI BOUGHT:                         │
│  1. Model prediction: 73% bullish       │
│  2. RSI at 28.5 (oversold < 30)        │
│  3. Volume 2.3x above average          │
│  4. 2 of 3 timeframes aligned          │
│  5. Confidence: 78% (threshold: 70%)   │
├─────────────────────────────────────────┤
│  TARGETS:                               │
│  Stop Loss:   $85,269 (-2.1%)          │
│  Take Profit: $91,402 (+5.0%)          │
└─────────────────────────────────────────┘
```

### 2.5 Update App.jsx Structure

Reorganize dashboard into clear sections:

1. **Header**: Status, connection, account
2. **Left Column**: AI Decision Flow + Reasoning
3. **Center Column**: Price chart + Positions
4. **Right Column**: Multi-Timeframe + Feature Importance + Logs

---

## Part 3: Files to Create/Modify

| File | Action | Description |

|------|--------|-------------|

| `src/strategy/adaptive_ai.py` | CREATE | New adaptive AI engine |

| `src/execution/bot_multi.py` | MODIFY | Integrate AI, broadcast full state |

| `src/api/server.py` | MODIFY | Add AI state to broadcasts |

| `dashboard/src/App.jsx` | MODIFY | Add all new visualization panels |

| `dashboard/src/components/AIDecisionFlow.jsx` | CREATE | Decision flow visualization |

| `dashboard/src/components/MultiTimeframe.jsx` | CREATE | Timeframe analysis view |

| `dashboard/src/components/FeatureImportance.jsx` | CREATE | AI feature weights |

| `dashboard/src/components/TradeReasoning.jsx` | CREATE | Trade explanation panel |

---

## Part 4: Aggressive Position Sizing (3-5%)

### Position Sizing Logic

In `bot_multi.py` `handle_bar()`, calculate position size based on AI confidence:

```python
if signal == 'BUY' and confidence >= 70:
    # Aggressive: 3-5% based on confidence
    # 70% confidence -> 3%
    # 100% confidence -> 5%
    min_pct = 0.03
    max_pct = 0.05
    
    scale = (confidence - 70) / 30  # 0 to 1
    position_pct = min_pct + (scale * (max_pct - min_pct))
    
    position_value = equity * position_pct
    qty = position_value / price
```

### Configuration (add to `config_ultra.py`)

```python
# Aggressive Position Sizing
MIN_POSITION_PCT = 0.03  # 3% at minimum confidence
MAX_POSITION_PCT = 0.05  # 5% at maximum confidence
MIN_CONFIDENCE = 70      # Don't trade below 70%
```

### Risk Limits

- Per trade: 3-5% of equity
- Max positions: 4 concurrent (unchanged)
- Stop loss: 2% per position (unchanged)
- Max drawdown: 20% of equity (unchanged)

---

## Part 5: Implementation Order

1. Create AdaptiveAI class with XGBoost integration
2. Update bot_multi.py to use AI with aggressive position sizing
3. Update config_ultra.py with new position sizing parameters
4. Enhance data broadcasting with full AI state
5. Build dashboard components one by one
6. Test end-to-end with live data

### To-dos

- [ ] Create AdaptiveAI class with XGBoost, multi-timeframe analysis, confidence scoring
- [ ] Update bot_multi.py to use AdaptiveAI instead of simple RSI rules
- [ ] Enhance WebSocket broadcast with full AI decision state
- [ ] Build AI Decision Flow panel showing data -> features -> model -> signal
- [ ] Build Multi-Timeframe Analysis panel
- [ ] Build Feature Importance panel showing what AI looks at
- [ ] Build Trade Reasoning panel explaining why trades happen
- [ ] Test complete system with live data