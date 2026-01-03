# Lessons from LACUNA for Live Trading

Observations from paper trading session (Jan 3, 2026):
- **PnL**: +$78.54 | **Trades**: 3,270 | **Win Rate**: 37% | **ROI**: 78%

---

## Key Insights

### 1. Low Win Rate Can Be Profitable
LACUNA wins only 37% of trades but profits because binary markets have asymmetric payoffs:
- Buy at 0.40 → win pays 0.60, lose costs 0.40
- Edge comes from *when* you enter, not win frequency

**Lesson for Parallax**: Don't filter signals by expected win rate alone. A 35% win rate with 2:1 payoff ratio beats 60% win rate with 1:1.

### 2. Position Sizing Matters More Than Entry
Share-based PnL (Phase 4 breakthrough):
- At entry price 0.30: get 3.33 shares per dollar
- At entry price 0.70: get 1.43 shares per dollar
- Same price move = larger return at lower entry prices

**Lesson for Parallax**: Kelly criterion should weight entry price, not just edge. Favor markets with prices far from 0.50.

### 3. Exposure Limits Prevent Blowups
LACUNA caps at $20 exposure across 4 markets ($5 each). Never fully invested.

**Lesson for Parallax**: Hard cap at 25% of bankroll in active positions. Leave room for averaging down or new opportunities.

### 4. Rate Limiting Reduces Noise Trading
10 trades/min cap forces selectivity. Without it, agent overtrades on noise.

**Lesson for Parallax**: Implement cooldown between trades on same market. Minimum 5-minute spacing.

### 5. Multi-Source Data Fusion Works
Combining fast market data (Binance futures) with slow market (Polymarket) provides edge. Information propagates with lag.

**Lesson for Parallax**:
- Use crypto spot prices as leading indicator for crypto prediction markets
- News sentiment from fast sources (Twitter, news APIs) before market pricing

### 6. Temporal Context Improves Decisions
TemporalEncoder processes last 5 states to capture momentum. Single-state decisions are noisier.

**Lesson for Parallax**:
- Track price history (already doing this with CLOB API)
- Weight recent momentum in signal generation
- Don't just look at current probability—look at trajectory

### 7. Sparse Rewards Still Work
Agent only gets feedback when positions close (every 15 min). No intermediate shaping.

**Lesson for Parallax**: Deep Research analysis quality matters more than frequency. One good 15-min analysis beats ten quick checks.

---

## Structural Differences

| Aspect | LACUNA | Parallax |
|--------|--------|----------|
| Market type | 15-min binary crypto | Days-to-weeks events |
| Data source | Real-time WebSocket | Polling + Deep Research |
| Decision speed | Seconds | Minutes to hours |
| Position duration | 15 minutes max | Days to weeks |
| Edge source | Information lag | Fundamental analysis |

---

## Actionable Changes for Parallax

### Immediate
1. **Add momentum tracking** to signal generation (price trajectory over last 24-48h)
2. **Favor extreme prices** (< 0.30 or > 0.70) for better share economics
3. **Hard exposure cap** at 25% of bankroll

### Medium-term
4. **Fast data integration** - crypto spot prices for crypto markets
5. **Cooldown periods** - minimum time between trades on same market
6. **Trajectory-aware Kelly** - weight position size by entry price distance from 0.50

### Architecture
7. Consider **temporal state encoding** for markets with sufficient price history
8. **Multi-source fusion** for categories with leading indicators (crypto, sports with live data)

---

## What NOT to Copy

- **Trade frequency**: LACUNA trades every few seconds; Parallax should stay at hours/days
- **Automated entry**: LACUNA's RL decides autonomously; Parallax should keep human confirmation for live
- **Position duration**: 15-min windows don't apply to event markets resolving in weeks

---

## Bottom Line

LACUNA proves:
1. Edge exists in prediction markets (78% ROI in one session)
2. Low win rate is fine with good payoff structure
3. Position sizing and exposure management are as important as signal quality
4. Multi-source data fusion provides actionable alpha

Apply these principles to Parallax's longer-horizon, fundamental-analysis-based approach.
