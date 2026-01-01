# Training Journal: RL on Polymarket

Training a PPO agent to trade 15-minute binary prediction markets. This documents what worked, what didn't, and what it means.

---

## The Experiment

**Question**: Can an RL agent learn profitable trading patterns from sparse PnL rewards?

**Setup**: Paper trade 4 concurrent crypto markets (BTC, ETH, SOL, XRP) on Polymarket using live data from Binance + Polymarket orderbooks.

**Result**: ~$50K PnL (2,500% ROI) in Phase 5 with temporal architecture. The path there was interesting - from reward shaping failures to sparse PnL rewards to share-based economics.

---

## Why This Market is Interesting

### Concurrent Multi-Asset Trading

Unlike typical RL trading that focuses on one asset, this agent trades 4 markets simultaneously with a single shared policy. Every 15-minute window spawns 4 independent binary markets. The agent must:
- Allocate attention across all active markets
- Learn asset-specific patterns while sharing weights
- Handle asynchronous expirations and refreshes

The same neural network decides for all assets - learning generalizable crypto patterns rather than overfitting to one market.

### Unique Market Structure

Polymarket's 15-minute crypto markets are unusual:
- **Binary resolution**: Market resolves to $1 or $0 based on price direction. Training uses share-based PnL (Phase 4+), which matches actual binary market economics.
- **Known resolution time**: You know exactly when the market closes. Changes the decision calculus.
- **Orderbook-based**: Real CLOB with bid/ask spreads, not an AMM.
- **Cross-exchange lag**: Polymarket prices lag Binance by seconds. Exploitable.

This creates arbitrage opportunities - observe Binance move, bet on Polymarket before the orderbook adjusts.

### Multi-Source Data Fusion

The agent fuses two real-time streams:

```
Binance Futures WSS  → Price returns (1m, 5m, 10m), volatility, order flow, CVD, large trades
Polymarket CLOB WSS  → Bid/ask spread, orderbook imbalance
```

**Why this matters**: Binance futures are the "fast" market (higher liquidity, institutional flow, sub-second price discovery). Polymarket is the "slow" market (lower liquidity, retail-heavy, prices lag by seconds). By observing both, the agent sees information before it's fully priced into the prediction market.

This is a general pattern: **fast signal source + slow execution venue**. The same architecture could extend to:
- Sports betting (live game data + betting exchange)
- Election markets (polling/prediction aggregators + Polymarket)
- Weather derivatives (forecast models + prediction markets)
- Any market where information propagates with measurable delay

This creates an 18-dimensional state that captures both underlying asset dynamics AND prediction market microstructure. See [README.md](README.md) for the full feature breakdown.

### The Reward Problem

RL agents need reward signals to learn. In trading, the obvious choice is profit—but how and when do you give that signal?

- **Dense rewards** (every tick): Easier credit assignment, but prone to gaming
- **Sparse rewards** (on trade close): Harder to learn, but honest signal

We started with dense rewards (Phase 1) and it backfired. The agent gamed the micro-bonuses instead of learning to trade. The fix was sparse rewards: reward only when a position closes, based on actual PnL. See Training Evolution below for the full story.

---

## Training Evolution

### Phase 1: Shaped Rewards (Updates 1-36)

**Duration**: ~52 minutes | **Trades**: 1,545 | **Entropy coef**: 0.05

Started with a reward function that tried to guide learning with micro-bonuses:

```python
# Reward given every step (not just on close)
reward = 0.0

# 1. Unrealized PnL delta - scaled DOWN by 0.1
if has_position:
    reward += (current_pnl - prev_pnl) * 0.1

# 2. Transaction cost penalty
if is_trade:
    reward -= 0.002 * size_multiplier

# 3. Spread cost on entry
if opening_position:
    reward -= spread * 0.5

# 4. Micro-bonuses (the problem)
reward += 0.002 * momentum_aligned  # Bonus for trading with momentum
reward += 0.001 * size_multiplier   # Bonus for larger positions
reward -= 0.001 * wrong_momentum    # Penalty for fighting momentum
```

**What happened**: Entropy collapsed from 1.09 → 0.36. The policy became nearly deterministic.

| Update | Entropy | PnL | Win Rate |
|--------|---------|-----|----------|
| 1 | 1.09 | $3.25 | 19.8% |
| 10 | 0.79 | $4.40 | 17.3% |
| 20 | 0.40 | $1.37 | 19.4% |
| 36 | 0.36 | $3.90 | 20.2% |

**Why it failed**: The shaping rewards were similar magnitude to actual PnL. With typical PnL deltas of $0.01-0.05, the scaled signal was 0.001-0.005 - same as the bonuses.

The agent learned to game the reward function:
- Trade with momentum → collect +0.002 bonus
- Use large sizes → collect +0.001 bonus
- Actual profitability? Optional.

Buffer win rate showed 90%+ (counting bonus-positive experiences) while actual trade win rate was 20%. The agent was optimizing the reward function, not the underlying goal.

**The tell**: Buffer win rate (% of experiences with reward > 0) diverged from trade win rate (% of profitable trades). When these diverge, the agent is learning the wrong thing.

---

### Phase 2: Pure Realized PnL (Updates 37+)

**Changes made**:
1. Reward ONLY on position close (not every step)
2. Increased entropy coefficient (0.05 → 0.10)
3. Simplified action space (7 → 3 actions)
4. Reduced buffer size (2048 → 512) and batch size (128 → 64) for faster updates
5. Reset reward normalization stats

```python
# Reward is 0 for all steps EXCEPT position close
def _compute_step_reward(self, cid, state, action, pos):
    return self.pending_rewards.pop(cid, 0.0)

# pending_rewards is set when position closes:
# pnl = (exit_prob - entry_prob) * size
self.pending_rewards[cid] = pnl
```

No more micro-bonuses. No more step-by-step unrealized PnL. Just: did you make money when you closed?

**Note on reward normalization**: Raw PnL rewards are z-score normalized before training:
```python
norm_reward = (raw_pnl - running_mean) / (running_std + 1e-8)
```

**Results**: Entropy recovered to 1.05 (near maximum for 3 actions). PnL grew steadily.

| Update | Entropy | PnL | Win Rate |
|--------|---------|-----|----------|
| 1 | 0.68 | $5.20 | 33.3% |
| 10 | 1.06 | $9.55 | 22.9% |
| 20 | 1.05 | $5.85 | 21.1% |
| 36 | 1.05 | $10.93 | 21.2% |

**Final**: $10.93 PnL on $20 max exposure = **55% ROI**

**Why 21% win rate is profitable**: Binary markets have asymmetric payoffs. Buy UP at 0.40: win pays $0.60, lose costs $0.40. You can win 40% of the time and break even. Win 21% but pick your spots at low probabilities? Still profitable.

---

### Phase 1 → 2: The Fix

| Aspect | Phase 1 | Phase 2 |
|--------|---------|---------|
| Reward | PnL delta + shaping bonuses | Sparse PnL only |
| Entropy coef | 0.02 → 0.05 | 0.10 |
| Buffer/batch | 2048/128 | 512/64 |
| Actions | 7 (variable sizing) | 3 (fixed 50%) |
| Final entropy | 0.36 (collapsed) | 1.05 (healthy) |

**What fixed it**:
1. Removed ALL shaping rewards - just `(exit_prob - entry_prob) * size` on close
2. 5x entropy coefficient (0.02 → 0.10) - stronger exploration
3. Simplified actions (7 → 3) - learn *when* before *how much*
4. Smaller buffer (2048 → 512) - faster updates
5. Reset reward normalization stats

---

### Phase 3: Scaled Up (Updates 73-108)

**Changes made**:
1. Increased trade size from $5 to $50 (10x)
2. Continued training from Phase 2 model checkpoint

**Duration**: ~50 minutes | **Trades**: 4,133 | **Size**: $50

**What happened**: The first update hit a -$64 drawdown - unlucky market timing. But the agent recovered steadily over the next 35 updates.

| Update | Entropy | PnL | Win Rate |
|--------|---------|-----|----------|
| 1 | 1.04 | -$63.75 | 29.5% |
| 10 | 1.08 | -$14.75 | 20.8% |
| 20 | 1.04 | -$8.40 | 17.6% |
| 36 | 0.97 | +$23.10 | 15.6% |

**Final**: $23.10 PnL on $200 max exposure = **12% ROI** (or **44% ROI** measuring recovery from -$64 trough)

**Key observations**:
- Entropy remained healthy (0.97) - no policy collapse despite the drawdown
- Win rate dropped to 15.6% but remained profitable (asymmetric payoffs)
- Recovery of $87 over 35 updates demonstrates robustness to adverse starts

#### Phase 3 Analysis

![Phase 3 Trading Analysis](phase3_analysis.png)

**Key findings**:
- **Equity curve** (top-left): -$75 max drawdown early, then steady recovery. Policy didn't collapse under pressure.
- **PnL by asset** (top-right): XRP carried (+$44), ETH struggled (-$45). Same policy, different results per asset.
- **PnL by side** (top-right): UP bets (+$16) outperformed DOWN (-$7) despite similar win rates. Slight long bias works.
- **Entry distribution** (bottom-left): Agent favors extreme probabilities (near 0 or 1) - hunting asymmetric payoffs.
- **Duration vs PnL** (bottom-middle): Correlation 0.02 - trade length doesn't predict outcome. Quick flips ≈ longer holds.
- **Entry timing vs PnL** (bottom-right): Correlation 0.01 - early vs late entry in 15-min window doesn't matter. Agent reacts to market state, not time.

---

### Phase 4: Share-Based PnL (Updates 1-46)

**Changes made**:
1. Switched from probability-based to share-based PnL reward signal
2. Increased trade size from $50 to $500 (10x)
3. Fresh model (no checkpoint)

**Duration**: ~2 hours | **Trades**: 4,873 | **Size**: $500

**The key change**: Reward signal now reflects actual binary market economics.

```python
# Old (Phases 1-3): probability-based
pnl = (exit_price - entry_price) * dollars

# New (Phase 4): share-based
shares = dollars / entry_price
pnl = (exit_price - entry_price) * shares
```

**Why this matters**: When you enter at probability 0.30, you get 3.33 shares per dollar vs 1.43 shares at 0.70. The same price move generates proportionally larger returns at lower entry prices. The reward signal now captures this asymmetry.

| Update | Entropy | PnL | Win Rate |
|--------|---------|-----|----------|
| 1 | 1.09 | -$197 | 18.9% |
| 10 | 1.07 | -$383 | 18.0% |
| 20 | 1.05 | -$465 | 18.5% |
| 30 | 1.04 | +$1,233 | 19.2% |
| 46 | 1.08 | +$3,392 | 19.0% |

**Final**: $3,392 PnL on $2,000 max exposure = **170% ROI**

**Comparison to Phase 3** (recalculated with share-based formula):
- Phase 3: $76 PnL, 38% ROI
- Phase 4: $3,392 PnL, 170% ROI
- **4.5x improvement** in ROI per dollar of exposure

**Key observations**:
- Entropy remained healthy (1.08) throughout - no policy collapse
- Win rate stable at ~19% - still profitable via asymmetric payoffs
- Large early drawdown (-$465 at update 20) followed by strong recovery
- Policy learned to exploit share-based dynamics: seek low-probability entries where price moves generate outsized returns

#### Phase 4 Analysis

![Phase 4 Trading Analysis](phase4_analysis.png)

The switch from probability-based to share-based PnL yielded a 4.5x improvement in ROI. The agent learned to seek low-probability entries where the same price move generates proportionally larger returns due to share mechanics.

---

### Phase 5: LACUNA (Temporal Architecture)

**[Visual writeup and story →](https://humanplane.com/lacuna)**

**Changes made**:
1. Added TemporalEncoder to process last 5 states into 32-dim momentum/trend features
2. Asymmetric actor-critic: Actor (64 hidden) vs Critic (96 hidden)
3. Feature normalization: All 18 input features clamped to [-1, 1]
4. Reduced gamma (0.99 → 0.95) and entropy coef (0.10 → 0.03)
5. Smaller buffer (512 → 256) for faster adaptation
6. Fixed market discovery bugs: persistent retry loop, fresh aiohttp sessions

**Duration**: 10+ hours | **Trades**: 34,730 | **Size**: $500/trade | **Mode**: Paper trading

**Final**: ~$50K PnL on $2,000 max exposure = **2,500% ROI** | **Win Rate**: 23.3%

**By asset performance**:
- BTC: +$40,088 (carried the session)
- ETH: +$7,648
- SOL: +$994
- XRP: +$593

---

## Takeaways

1. **Reward shaping is risky** - When shaping rewards are gameable and similar magnitude to the real signal, agents optimize the wrong thing. Sparse but honest > dense but noisy.

2. **Reward signal design matters** - Share-based PnL (Phase 4) outperformed probability-based PnL (Phases 2-3) by 4.5x ROI. The reward signal should match actual market economics.

3. **Entropy coefficient matters** - 0.05 caused policy collapse; 0.10 maintained healthy exploration. Phase 5 reduced to 0.03 successfully after the policy stabilized.

4. **Watch for buffer/trade win rate divergence** - When these diverge, the agent is optimizing the wrong objective.

5. **Robustness to drawdowns** - Phases 3, 4, and 5 showed the agent can recover from large adverse moves without policy collapse. Entropy stayed healthy throughout.

6. **Temporal context improves decisions** - Phase 5's TemporalEncoder (processing last 5 states) captures momentum and trend patterns that single-state observation misses. Asymmetric actor-critic (larger critic) helps value estimation.

7. **Feature normalization matters** - Clamping all 18 features to [-1, 1] prevents any single feature from dominating the gradient signal.

---

*December 29-31, 2025*
