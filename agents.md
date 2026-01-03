# Agents

## LACUNA (Cross-Market State Fusion)

PPO-based RL agent that paper trades Polymarket's 15-minute binary crypto markets by exploiting information lag between fast markets (Binance futures) and slow markets (Polymarket prediction markets).

### Quick Start

```bash
# Paper trading with dashboard
python run.py rl --load lacuna_model --balance 100 --size 5 --dashboard --port 5050

# Training mode
python run.py rl --train --size 50

# Inference only (no training)
python run.py rl --load lacuna_model --size 100
```

### Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--balance` | 100 | Starting paper balance ($) |
| `--size` | 5 | Position size per trade ($) |
| `--max-exposure` | 20 | Maximum concurrent exposure ($) |
| `--max-trades-per-min` | 10 | Rate limit |
| `--dashboard` | false | Enable web dashboard |
| `--port` | 5050 | Dashboard port |
| `--train` | false | Enable online learning |
| `--load` | - | Load saved model (e.g., `lacuna_model`) |
| `--live` | false | LIVE TRADING (real money) |

### Markets

Trades 4 concurrent 15-minute binary crypto markets:
- **BTC** - Bitcoin price direction
- **ETH** - Ethereum price direction
- **SOL** - Solana price direction
- **XRP** - Ripple price direction

### State Space (18 dimensions)

| Category | Features | Source |
|----------|----------|--------|
| Momentum | `returns_1m`, `returns_5m`, `returns_10m` | Binance futures |
| Order Flow | `ob_imbalance_l1`, `ob_imbalance_l5`, `trade_flow`, `cvd_accel` | Binance futures |
| Microstructure | `spread_pct`, `trade_intensity`, `large_trade_flag` | Polymarket CLOB |
| Volatility | `vol_5m`, `vol_expansion` | Polymarket + Binance |
| Position | `has_position`, `position_side`, `position_pnl`, `time_remaining` | Internal |
| Regime | `vol_regime`, `trend_regime` | Derived |

### Action Space

| Action | Description |
|--------|-------------|
| HOLD (0) | No action |
| BUY (1) | Long UP token (bet price goes up) |
| SELL (2) | Long DOWN token (bet price goes down) |

### Architecture

```
TemporalEncoder: (5 states × 18 features) → 64 → LayerNorm → tanh → 32

Actor:  [current(18) + temporal(32)] → 64 → LN → tanh → 64 → LN → tanh → 3 (softmax)
Critic: [current(18) + temporal(32)] → 96 → LN → tanh → 96 → LN → tanh → 1
```

### Performance

From training sessions:
- **Win Rate**: ~37% (low but profitable due to asymmetric payoffs)
- **ROI**: 170-2500% depending on phase
- **Key Insight**: Buy at 0.40, win pays 0.60; lose costs 0.40

### Models

| Model | Description |
|-------|-------------|
| `lacuna_model` | Production model with temporal architecture |
| `rl_model` | Earlier model without temporal features |
| `rl_model_prob_pnl` | Share-based PnL model |

### Monitoring

```bash
# Live logs
tail -f logs/paper_simulation.log

# Trade history
cat logs/trades_*.csv

# Dashboard
open http://localhost:5050
```

### Safety

- Paper trading by default (no real orders)
- `--live` flag required for real trading (requires `--confirm`)
- Max exposure limits prevent over-leveraging
- Rate limiting prevents excessive trading
