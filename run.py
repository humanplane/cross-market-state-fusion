#!/usr/bin/env python3
"""
Unified runner for Polymarket trading strategies.

Usage:
    python run.py                     # List available strategies
    python run.py random              # Run random baseline
    python run.py mean_revert         # Run mean reversion
    python run.py rl                  # Run RL strategy
    python run.py gating              # Run gating (MoE)
    python run.py rl --train          # Train RL strategy
    python run.py rl --train --dashboard  # Train with web dashboard
"""
import asyncio
import argparse
import copy
import sys
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, ".")
from helpers import get_15m_markets, BinanceStreamer, OrderbookStreamer, Market, FuturesStreamer, get_logger
from strategies import (
    Strategy, MarketState, Action,
    create_strategy, AVAILABLE_STRATEGIES,
    RLStrategy,
)

# Dashboard integration (optional)
try:
    from dashboard_cinematic import update_dashboard_state, update_rl_metrics, emit_rl_buffer, run_dashboard, emit_trade
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    def update_dashboard_state(**kwargs): pass
    def update_rl_metrics(metrics): pass
    def emit_rl_buffer(buffer_size, max_buffer=256, avg_reward=None): pass
    def emit_trade(action, asset, size=0, pnl=None): pass


@dataclass
class Position:
    """Track position for a market."""
    asset: str
    side: Optional[str] = None
    size: float = 0.0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    entry_prob: float = 0.0
    time_remaining_at_entry: float = 0.0


class TradingEngine:
    """
    Paper trading engine with strategy harness.
    Supports optional live trading via --live flag.

    Safety features:
    - max_exposure: Maximum total $ in open positions
    - max_loss: Stop trading if cumulative loss exceeds this amount
    - reserve_pct: Keep this % of capital untouchable (default 20%)
    - max_trades_per_min: Rate limit trades (default 10)
    """

    def __init__(
        self,
        strategy: Strategy,
        trade_size: float = 10.0,
        live_mode: bool = False,
        max_exposure: float = None,
        max_loss: float = None,
        reserve_pct: float = 0.20,
        max_trades_per_min: int = 10,
        starting_balance: float = None,
    ):
        self.strategy = strategy
        self.trade_size = trade_size
        self.live_mode = live_mode
        self.max_exposure = max_exposure  # Maximum total $ in open positions
        self.max_loss = max_loss  # Stop trading if losses exceed this
        self.reserve_pct = reserve_pct  # Keep 20% reserve by default
        self.max_trades_per_min = max_trades_per_min
        self.starting_balance = starting_balance

        # Trade rate limiting
        self.recent_trades: List[datetime] = []  # Timestamps of recent trades
        self.trading_halted = False  # Set True if max_loss exceeded
        self.halt_reason = ""

        # Live execution setup
        self.live_executor = None
        if live_mode:
            try:
                from execution import SyncLiveExecutor
                self.live_executor = SyncLiveExecutor(dry_run=False)
                print("=" * 60)
                print("LIVE TRADING MODE - REAL MONEY AT RISK")
                print(f"  Dry run: {self.live_executor.dry_run}")
                print(f"  Kill switch: {self.live_executor.kill_switch}")
                print("=" * 60)
            except Exception as e:
                print(f"ERROR: Failed to initialize live executor: {e}")
                print("Falling back to paper trading.")
                self.live_mode = False

        # Streamers
        self.price_streamer = BinanceStreamer(["BTC", "ETH", "SOL", "XRP"])
        self.orderbook_streamer = OrderbookStreamer()
        self.futures_streamer = FuturesStreamer(["BTC", "ETH", "SOL", "XRP"])

        # State
        self.markets: Dict[str, Market] = {}
        self.positions: Dict[str, Position] = {}
        self.states: Dict[str, MarketState] = {}
        self.prev_states: Dict[str, MarketState] = {}  # For RL transitions
        self.open_prices: Dict[str, float] = {}  # Binance price at market open
        self.running = False

        # Stats
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0

        # Pending rewards for RL (set on position close)
        self.pending_rewards: Dict[str, float] = {}

        # Logger (for RL training)
        self.logger = get_logger() if isinstance(strategy, RLStrategy) else None

        # Real balance tracking for live mode
        self.last_balance_check = None
        self.cached_usdc_balance = starting_balance
        self.initial_balance_snapshot = None  # Snapshot at start for hard stop

    def get_total_exposure(self) -> float:
        """Calculate total $ currently in open positions."""
        total = 0.0
        for pos in self.positions.values():
            if pos.size > 0:
                total += pos.size
        return total

    def get_real_usdc_balance(self) -> float:
        """Get actual USDC balance from blockchain (cached for 10 seconds)."""
        if not self.live_mode:
            return float('inf')  # Paper trading has unlimited balance

        now = datetime.now(timezone.utc)

        # Use cache if fresh (within 10 seconds)
        if self.last_balance_check and (now - self.last_balance_check).total_seconds() < 10:
            return self.cached_usdc_balance

        try:
            import os
            from web3 import Web3

            # Load env if needed
            proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS")
            if not proxy_address:
                return self.cached_usdc_balance or 0

            w3 = Web3(Web3.HTTPProvider('https://polygon-rpc.com'))
            USDC = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
            usdc_abi = [{"constant":True,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"type":"function"}]
            balance = w3.eth.contract(address=USDC, abi=usdc_abi).functions.balanceOf(Web3.to_checksum_address(proxy_address)).call() / 1e6

            self.cached_usdc_balance = balance
            self.last_balance_check = now
            return balance
        except Exception as e:
            print(f"    ⚠️ Balance check failed: {e}")
            return self.cached_usdc_balance or 0

    def can_open_position(self, amount: float) -> bool:
        """Check if we can open a new position without exceeding max exposure."""
        if self.max_exposure is None:
            return True
        current = self.get_total_exposure()
        return (current + amount) <= self.max_exposure

    def check_rate_limit(self) -> bool:
        """Check if we've exceeded the max trades per minute. Returns True if OK to trade."""
        now = datetime.now(timezone.utc)
        # Remove trades older than 1 minute
        self.recent_trades = [t for t in self.recent_trades if (now - t).total_seconds() < 60]

        # Per-minute limit
        if len(self.recent_trades) >= self.max_trades_per_min:
            return False

        # BURST PROTECTION: Max 2 trades per second
        recent_second = [t for t in self.recent_trades if (now - t).total_seconds() < 1]
        if len(recent_second) >= 2:
            return False

        return True

    def record_trade_time(self):
        """Record that a trade was made for rate limiting."""
        self.recent_trades.append(datetime.now(timezone.utc))

    def check_loss_limit(self) -> bool:
        """Check if we've exceeded max loss. Returns True if OK to trade."""
        if self.max_loss is None:
            return True
        if self.total_pnl < -abs(self.max_loss):
            self.trading_halted = True
            self.halt_reason = f"Max loss ${self.max_loss:.2f} exceeded (current: ${self.total_pnl:.2f})"
            return False
        return True

    def get_available_capital(self) -> float:
        """Get capital available for trading (after reserve)."""
        if self.starting_balance is None:
            return float('inf')  # Paper trading, no limit
        reserved = self.starting_balance * self.reserve_pct
        current_exposure = self.get_total_exposure()
        available = self.starting_balance - reserved - current_exposure + self.total_pnl
        return max(0, available)

    def can_trade(self, amount: float) -> tuple[bool, str]:
        """
        Master safety check - returns (can_trade, reason).
        Combines all safety checks into one.
        """
        # Check if trading is halted
        if self.trading_halted:
            return False, f"HALTED: {self.halt_reason}"

        # Check rate limit FIRST (cheap check)
        if not self.check_rate_limit():
            return False, f"Rate limit: {len(self.recent_trades)}/{self.max_trades_per_min} trades/min"

        # CRITICAL: Check REAL USDC balance for live trading
        if self.live_mode:
            real_balance = self.get_real_usdc_balance()

            # Take initial snapshot on first check
            if self.initial_balance_snapshot is None:
                self.initial_balance_snapshot = real_balance
                print(f"    📸 Initial balance snapshot: ${real_balance:.2f}")

            # HARD STOP #1: If real balance dropped more than max_loss from snapshot
            if self.max_loss and self.initial_balance_snapshot:
                actual_loss = self.initial_balance_snapshot - real_balance
                if actual_loss > self.max_loss:
                    self.trading_halted = True
                    self.halt_reason = f"HARD STOP: Lost ${actual_loss:.2f} (max: ${self.max_loss:.2f})"
                    print(f"\n🛑 {self.halt_reason}")
                    return False, self.halt_reason

            # HARD STOP #2: If balance below reserve
            if self.starting_balance:
                min_balance = self.starting_balance * self.reserve_pct
                if real_balance < min_balance:
                    self.trading_halted = True
                    self.halt_reason = f"Balance ${real_balance:.2f} below reserve ${min_balance:.2f}"
                    print(f"\n🛑 {self.halt_reason}")
                    return False, self.halt_reason

            # Check if we have enough for this trade
            if real_balance < amount:
                return False, f"Insufficient USDC: ${real_balance:.2f} < ${amount:.2f} needed"

        # Check loss limit (paper P&L)
        if not self.check_loss_limit():
            return False, f"Loss limit exceeded: ${self.total_pnl:.2f}"

        # Check max exposure (paper positions)
        if not self.can_open_position(amount):
            current = self.get_total_exposure()
            return False, f"Max exposure: ${current:.0f}/${self.max_exposure:.0f}"

        return True, "OK"

    def refresh_markets(self):
        """Find active 15-min markets."""
        print("\n" + "=" * 60)
        print(f"STRATEGY: {self.strategy.name.upper()}")
        print("=" * 60)

        markets = get_15m_markets(assets=["BTC", "ETH", "SOL", "XRP"])
        now = datetime.now(timezone.utc)

        # Clear stale data
        self.markets.clear()
        self.states.clear()

        for m in markets:
            mins_left = (m.end_time - now).total_seconds() / 60
            if mins_left < 0.5:
                continue

            print(f"\n{m.asset} 15m | {mins_left:.1f}m left")
            print(f"  UP: {m.price_up:.3f} | DOWN: {m.price_down:.3f}")

            self.markets[m.condition_id] = m
            self.orderbook_streamer.subscribe(m.condition_id, m.token_up, m.token_down)

            # Init state
            self.states[m.condition_id] = MarketState(
                asset=m.asset,
                prob=m.price_up,
                time_remaining=mins_left / 15.0,
            )

            # Init position
            if m.condition_id not in self.positions:
                self.positions[m.condition_id] = Position(asset=m.asset)

            # Record open price
            current_price = self.price_streamer.get_price(m.asset)
            if current_price > 0:
                self.open_prices[m.condition_id] = current_price

        if not self.markets:
            print("\nNo active markets!")
        else:
            # Clear stale orderbook subscriptions
            active_cids = set(self.markets.keys())
            self.orderbook_streamer.clear_stale(active_cids)

    def execute_action(self, cid: str, action: Action, state: MarketState):
        """Execute paper trade (and optionally live trade) with flexible sizing."""
        if action == Action.HOLD:
            return

        pos = self.positions.get(cid)
        market = self.markets.get(cid)
        if not pos or not market:
            return

        # ============================================================
        # SAFETY CHECKS FIRST - Before any calculations
        # ============================================================

        # Check if trading is halted
        if self.trading_halted:
            print(f"    🛑 HALTED: {self.halt_reason}")
            return

        # Check rate limit (includes burst protection)
        if not self.check_rate_limit():
            return  # Silent - don't spam logs

        # For OPENING new positions only - check exposure limits
        if pos.size == 0:
            # Calculate proposed trade amount with HARD CAPS
            size_multiplier = min(action.size_multiplier, 1.0)  # Cap at 1x
            trade_amount = min(self.trade_size * size_multiplier, self.trade_size)

            # Check max exposure using ACTUAL open positions
            if self.max_exposure:
                current_exposure = self.get_total_exposure()
                available_exposure = self.max_exposure - current_exposure
                if available_exposure <= 0:
                    print(f"    🛑 MAX EXPOSURE: ${current_exposure:.0f}/${self.max_exposure:.0f} (no room)")
                    return
                # Cap trade to available exposure
                trade_amount = min(trade_amount, available_exposure)

            # Check real USDC balance for live mode
            if self.live_mode:
                real_balance = self.get_real_usdc_balance()

                # Take snapshot on first trade
                if self.initial_balance_snapshot is None:
                    self.initial_balance_snapshot = real_balance
                    print(f"    📸 Initial balance: ${real_balance:.2f}")

                # Hard stop: loss exceeded
                if self.max_loss and self.initial_balance_snapshot:
                    actual_loss = self.initial_balance_snapshot - real_balance
                    if actual_loss >= self.max_loss:
                        self.trading_halted = True
                        self.halt_reason = f"Lost ${actual_loss:.2f} (limit: ${self.max_loss:.2f})"
                        print(f"    🛑 {self.halt_reason}")
                        return

                # Check sufficient balance
                if real_balance < trade_amount:
                    print(f"    🛑 INSUFFICIENT: ${real_balance:.2f} < ${trade_amount:.2f}")
                    return
        else:
            # For CLOSING positions, use existing position size
            trade_amount = pos.size

        # ============================================================
        # EXECUTION - Only reached if all safety checks pass
        # ============================================================

        price = state.prob

        # Close existing position if switching sides
        if pos.size > 0:
            if action.is_sell and pos.side == "UP":
                # Live execution: sell UP token
                if self.live_mode and self.live_executor:
                    result = self.live_executor.execute_trade(
                        condition_id=cid,
                        token_id=market.token_up,
                        outcome="Up",
                        side="sell",
                        size_usd=pos.size,
                        limit_price=price,
                    )
                    if result.success:
                        print(f"    LIVE CLOSE UP: {result.order_id}")
                    else:
                        print(f"    LIVE CLOSE UP FAILED: {result.error}")

                shares = pos.size / pos.entry_price
                pnl = (price - pos.entry_price) * shares
                self._record_trade(pos, price, pnl, "CLOSE UP", cid=cid)
                self.pending_rewards[cid] = pnl  # Pure realized PnL reward
                pos.size = 0
                pos.side = None
                return

            elif action.is_buy and pos.side == "DOWN":
                exit_down_price = 1 - price  # Current DOWN token price

                # Live execution: sell DOWN token
                if self.live_mode and self.live_executor:
                    result = self.live_executor.execute_trade(
                        condition_id=cid,
                        token_id=market.token_down,
                        outcome="Down",
                        side="sell",
                        size_usd=pos.size,
                        limit_price=exit_down_price,
                    )
                    if result.success:
                        print(f"    LIVE CLOSE DOWN: {result.order_id}")
                    else:
                        print(f"    LIVE CLOSE DOWN FAILED: {result.error}")

                shares = pos.size / pos.entry_price
                pnl = (exit_down_price - pos.entry_price) * shares  # DOWN token went up = profit
                self._record_trade(pos, price, pnl, "CLOSE DOWN", cid=cid)
                self.pending_rewards[cid] = pnl  # Pure realized PnL reward
                pos.size = 0
                pos.side = None
                return

        # Open new position (safety already checked at top of function)
        if pos.size == 0:
            size_label = {0.25: "SM", 0.5: "MD", 1.0: "LG"}.get(action.size_multiplier, "")

            if action.is_buy:
                # Live execution: buy UP token
                if self.live_mode and self.live_executor:
                    result = self.live_executor.execute_trade(
                        condition_id=cid,
                        token_id=market.token_up,
                        outcome="Up",
                        side="buy",
                        size_usd=trade_amount,
                        limit_price=price,
                    )
                    if result.success:
                        print(f"    LIVE BUY UP: {result.order_id}")
                    else:
                        print(f"    LIVE BUY UP FAILED: {result.error}")
                        return  # Don't update paper position on live failure

                pos.side = "UP"
                pos.size = trade_amount
                pos.entry_price = price
                pos.entry_time = datetime.now(timezone.utc)
                pos.entry_prob = price
                pos.time_remaining_at_entry = state.time_remaining
                self.record_trade_time()  # Rate limit tracking
                current_exposure = self.get_total_exposure()
                print(f"    OPEN {pos.asset} UP ({size_label}) ${trade_amount:.0f} @ {price:.3f} [exposure: ${current_exposure:.0f}/${self.max_exposure or 999:.0f}]")
                emit_trade(f"BUY_{size_label}", pos.asset, pos.size)

            elif action.is_sell:
                down_price = 1 - price  # DOWN token price = 1 - UP prob

                # Live execution: buy DOWN token
                if self.live_mode and self.live_executor:
                    result = self.live_executor.execute_trade(
                        condition_id=cid,
                        token_id=market.token_down,
                        outcome="Down",
                        side="buy",
                        size_usd=trade_amount,
                        limit_price=down_price,
                    )
                    if result.success:
                        print(f"    LIVE BUY DOWN: {result.order_id}")
                    else:
                        print(f"    LIVE BUY DOWN FAILED: {result.error}")
                        return  # Don't update paper position on live failure

                pos.side = "DOWN"
                pos.size = trade_amount
                pos.entry_price = down_price
                pos.entry_time = datetime.now(timezone.utc)
                pos.entry_prob = price  # Keep original UP prob for reference
                pos.time_remaining_at_entry = state.time_remaining
                self.record_trade_time()  # Rate limit tracking
                current_exposure = self.get_total_exposure()
                print(f"    OPEN {pos.asset} DOWN ({size_label}) ${trade_amount:.0f} @ {down_price:.3f} [exposure: ${current_exposure:.0f}/${self.max_exposure or 999:.0f}]")
                emit_trade(f"SELL_{size_label}", pos.asset, pos.size)

    def _record_trade(self, pos: Position, price: float, pnl: float, action: str, cid: str = None):
        """Record completed trade."""
        self.total_pnl += pnl
        self.trade_count += 1
        if pnl > 0:
            self.win_count += 1
        print(f"    {action} {pos.asset} @ {price:.3f} | PnL: ${pnl:+.2f}")
        # Emit to dashboard
        emit_trade(action, pos.asset, pos.size, pnl)

        # Log to CSV
        if self.logger and pos.entry_time:
            duration = (datetime.now(timezone.utc) - pos.entry_time).total_seconds()
            binance_change = 0.0
            if cid and cid in self.open_prices:
                current = self.price_streamer.get_price(pos.asset)
                if current > 0 and self.open_prices[cid] > 0:
                    binance_change = (current - self.open_prices[cid]) / self.open_prices[cid]

            self.logger.log_trade(
                asset=pos.asset,
                action="BUY" if "UP" in action else "SELL",
                side=pos.side or "UNKNOWN",
                entry_price=pos.entry_price,
                exit_price=price,
                size=pos.size,
                pnl=pnl,
                duration_sec=duration,
                time_remaining=pos.time_remaining_at_entry,
                prob_at_entry=pos.entry_prob,
                prob_at_exit=price,
                binance_change=binance_change,
                condition_id=cid
            )

    def _compute_step_reward(self, cid: str, state: MarketState, action: Action, pos: Position) -> float:
        """Compute reward signal for RL training - pure realized PnL."""
        # Only reward on position close - cleaner signal
        # Reward is set when trade closes in _execute_trade via self.pending_rewards
        return self.pending_rewards.pop(cid, 0.0)

    def close_all_positions(self):
        """Close all positions at current prices."""
        for cid, pos in self.positions.items():
            if pos.size > 0:
                state = self.states.get(cid)
                if state:
                    price = state.prob
                    shares = pos.size / pos.entry_price
                    if pos.side == "UP":
                        pnl = (price - pos.entry_price) * shares
                    else:
                        exit_down_price = 1 - price
                        pnl = (exit_down_price - pos.entry_price) * shares

                    self._record_trade(pos, price, pnl, f"FORCE CLOSE {pos.side}", cid=cid)
                    self.pending_rewards[cid] = pnl  # Pure realized PnL reward
                    pos.size = 0
                    pos.side = None

    async def decision_loop(self):
        """Main trading loop."""
        tick = 0
        tick_interval = 0.5  # 500ms ticks for faster decisions
        while self.running:
            await asyncio.sleep(tick_interval)
            tick += 1
            now = datetime.now(timezone.utc)

            # Check expired markets
            expired = [cid for cid, m in self.markets.items() if m.end_time <= now]
            for cid in expired:
                print(f"\n  EXPIRED: {self.markets[cid].asset}")

                # RL: Store terminal experience with final PnL
                if isinstance(self.strategy, RLStrategy) and self.strategy.training:
                    state = self.states.get(cid)
                    prev_state = self.prev_states.get(cid)
                    pos = self.positions.get(cid)
                    if state and prev_state:
                        # Terminal reward is the realized PnL
                        terminal_reward = state.position_pnl if pos and pos.size > 0 else 0.0
                        self.strategy.store(prev_state, Action.HOLD, terminal_reward, state, done=True)

                    # Clean up prev_state
                    if cid in self.prev_states:
                        del self.prev_states[cid]

                del self.markets[cid]

            if not self.markets:
                print("\nAll markets expired. Refreshing...")
                self.close_all_positions()
                self.refresh_markets()
                if not self.markets:
                    print("No new markets. Waiting...")
                    await asyncio.sleep(30)
                continue

            # Update states and make decisions
            for cid, m in self.markets.items():
                state = self.states.get(cid)
                if not state:
                    continue

                # Update state from orderbook - CRITICAL for 15-min
                ob = self.orderbook_streamer.get_orderbook(cid, "UP")
                if ob and ob.mid_price:
                    state.prob = ob.mid_price
                    state.prob_history.append(ob.mid_price)
                    if len(state.prob_history) > 100:
                        state.prob_history = state.prob_history[-100:]
                    state.best_bid = ob.best_bid or 0.0
                    state.best_ask = ob.best_ask or 0.0
                    state.spread = ob.spread or 0.0

                    # Orderbook imbalance - L1 (top of book)
                    if ob.bids and ob.asks:
                        bid_vol_l1 = ob.bids[0][1] if ob.bids else 0
                        ask_vol_l1 = ob.asks[0][1] if ob.asks else 0
                        total_l1 = bid_vol_l1 + ask_vol_l1
                        state.order_book_imbalance_l1 = (bid_vol_l1 - ask_vol_l1) / total_l1 if total_l1 > 0 else 0.0

                        # Orderbook imbalance - L5 (depth)
                        bid_vol_l5 = sum(size for _, size in ob.bids[:5])
                        ask_vol_l5 = sum(size for _, size in ob.asks[:5])
                        total_l5 = bid_vol_l5 + ask_vol_l5
                        state.order_book_imbalance_l5 = (bid_vol_l5 - ask_vol_l5) / total_l5 if total_l5 > 0 else 0.0

                # Update binance price
                binance_price = self.price_streamer.get_price(m.asset)
                state.binance_price = binance_price
                open_price = self.open_prices.get(cid, binance_price)
                if open_price > 0:
                    state.binance_change = (binance_price - open_price) / open_price

                # Update futures data (focused on fast-updating features)
                futures = self.futures_streamer.get_state(m.asset)
                if futures:
                    # Order flow - THE EDGE
                    old_cvd = state.cvd
                    state.cvd = futures.cvd
                    state.cvd_acceleration = (futures.cvd - old_cvd) / 1e6 if old_cvd != 0 else 0.0
                    state.trade_flow_imbalance = futures.trade_flow_imbalance

                    # Ultra-short momentum
                    state.returns_1m = futures.returns_1m
                    state.returns_5m = futures.returns_5m
                    state.returns_10m = futures.returns_10m  # Properly computed from klines

                    # Microstructure - CRITICAL for 15-min
                    state.trade_intensity = futures.trade_intensity
                    state.large_trade_flag = futures.large_trade_flag

                    # Volatility
                    state.realized_vol_5m = futures.realized_vol_1h / 3.5 if futures.realized_vol_1h > 0 else 0.0
                    state.vol_expansion = futures.vol_ratio - 1.0

                    # Regime context (slow but useful for context)
                    state.vol_regime = 1.0 if futures.realized_vol_1h > 0.01 else 0.0
                    state.trend_regime = 1.0 if abs(futures.returns_1h) > 0.005 else 0.0

                # Time remaining - CRITICAL
                state.time_remaining = (m.end_time - now).total_seconds() / 900

                # Update position info in state
                pos = self.positions.get(cid)
                if pos and pos.size > 0:
                    state.has_position = True
                    state.position_side = pos.side
                    shares = pos.size / pos.entry_price
                    if pos.side == "UP":
                        state.position_pnl = (state.prob - pos.entry_price) * shares
                    else:
                        current_down_price = 1 - state.prob
                        state.position_pnl = (current_down_price - pos.entry_price) * shares
                else:
                    state.has_position = False
                    state.position_side = None
                    state.position_pnl = 0.0

                # For non-RL strategies, force close near expiry as safety
                # For RL, let it learn to close on its own (gets penalty at expiry)
                if pos and pos.size > 0 and state.very_near_expiry:
                    if not isinstance(self.strategy, RLStrategy):
                        print(f"    ⏰ EARLY CLOSE: {pos.asset}")
                        close_action = Action.SELL if pos.side == "UP" else Action.BUY
                        self.execute_action(cid, close_action, state)
                        continue

                # Get action from strategy
                action = self.strategy.act(state)

                # RL: Store experience EVERY tick (dense learning signal)
                if isinstance(self.strategy, RLStrategy) and self.strategy.training:
                    prev_state = self.prev_states.get(cid)
                    if prev_state:
                        step_reward = self._compute_step_reward(cid, state, action, pos)
                        # Episode not done unless market expired
                        self.strategy.store(prev_state, action, step_reward, state, done=False)

                    # Deep copy state for next iteration
                    self.prev_states[cid] = copy.deepcopy(state)

                # Execute
                if action != Action.HOLD:
                    self.execute_action(cid, action, state)

            # Status update every 10 ticks (console), but dashboard every tick
            if tick % 10 == 0:
                self.print_status()
            else:
                # Update dashboard state every tick for responsiveness
                self._update_dashboard_only()

            # RL training: emit buffer progress every tick
            if isinstance(self.strategy, RLStrategy) and self.strategy.training:
                buffer_size = len(self.strategy.experiences)
                # Compute average reward from recent experiences
                avg_reward = None
                if buffer_size > 0:
                    recent_rewards = [exp.reward for exp in self.strategy.experiences[-50:]]  # Last 50
                    avg_reward = sum(recent_rewards) / len(recent_rewards)
                emit_rl_buffer(buffer_size, self.strategy.buffer_size, avg_reward)

                # PPO update when buffer is full
                if buffer_size >= self.strategy.buffer_size:
                    # Get buffer rewards before update clears them
                    buffer_rewards = [exp.reward for exp in self.strategy.experiences]
                    metrics = self.strategy.update()
                    if metrics:
                        print(f"  [RL] loss={metrics['policy_loss']:.4f} "
                              f"v_loss={metrics['value_loss']:.4f} "
                              f"ent={metrics['entropy']:.3f} "
                              f"kl={metrics['approx_kl']:.4f} "
                              f"ev={metrics['explained_variance']:.2f}")
                        # Send to dashboard
                        metrics['buffer_size'] = len(self.strategy.experiences)
                        update_rl_metrics(metrics)
                        # Log to CSV
                        if self.logger:
                            self.logger.log_update(
                                metrics=metrics,
                                buffer_rewards=buffer_rewards,
                                cumulative_pnl=self.total_pnl,
                                cumulative_trades=self.trade_count,
                                cumulative_wins=self.win_count
                            )

    def _update_dashboard_only(self):
        """Update dashboard state without printing to console."""
        now = datetime.now(timezone.utc)
        dashboard_markets = {}
        dashboard_positions = {}

        for cid, m in self.markets.items():
            state = self.states.get(cid)
            pos = self.positions.get(cid)
            if state:
                mins_left = (m.end_time - now).total_seconds() / 60
                vel = state._velocity()
                dashboard_markets[cid] = {
                    'asset': m.asset,
                    'prob': state.prob,
                    'time_left': mins_left,
                    'velocity': vel,
                }
                if pos:
                    dashboard_positions[cid] = {
                        'side': pos.side,
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                    }

        update_dashboard_state(
            strategy_name=self.strategy.name,
            total_pnl=self.total_pnl,
            trade_count=self.trade_count,
            win_count=self.win_count,
            positions=dashboard_positions,
            markets=dashboard_markets,
        )

    def print_status(self):
        """Print current status."""
        now = datetime.now(timezone.utc)
        win_rate = self.win_count / max(1, self.trade_count) * 100
        exposure = self.get_total_exposure()

        # Clean up old trades from rate limit tracker
        self.recent_trades = [t for t in self.recent_trades if (now - t).total_seconds() < 60]

        print(f"\n[{now.strftime('%H:%M:%S')}] {self.strategy.name.upper()}")
        print(f"  PnL: ${self.total_pnl:+.2f} | Trades: {self.trade_count} | Win: {win_rate:.0f}%")

        # Safety status line
        safety_parts = []
        if self.max_exposure:
            safety_parts.append(f"Exp: ${exposure:.0f}/${self.max_exposure:.0f}")
        if self.max_loss:
            pnl_pct = (self.total_pnl / abs(self.max_loss) * 100) if self.max_loss else 0
            safety_parts.append(f"Loss: ${self.total_pnl:.0f}/${-self.max_loss:.0f}")
        safety_parts.append(f"Rate: {len(self.recent_trades)}/{self.max_trades_per_min}/min")
        if self.trading_halted:
            safety_parts.append("⛔ HALTED")

        if safety_parts:
            print(f"  [{' | '.join(safety_parts)}]")

        # Prepare dashboard data
        dashboard_markets = {}
        dashboard_positions = {}

        for cid, m in self.markets.items():
            state = self.states.get(cid)
            pos = self.positions.get(cid)
            if state:
                mins_left = (m.end_time - now).total_seconds() / 60
                pos_str = f"{pos.side} ${pos.size:.0f}" if pos and pos.size > 0 else "FLAT"
                vel = state._velocity()
                print(f"  {m.asset}: prob={state.prob:.3f} vel={vel:+.3f} | {pos_str} | {mins_left:.1f}m")

                # Dashboard data
                dashboard_markets[cid] = {
                    'asset': m.asset,
                    'prob': state.prob,
                    'time_left': mins_left,
                    'velocity': vel,
                }
                if pos:
                    dashboard_positions[cid] = {
                        'side': pos.side,
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                    }

        # Update dashboard
        update_dashboard_state(
            strategy_name=self.strategy.name,
            total_pnl=self.total_pnl,
            trade_count=self.trade_count,
            win_count=self.win_count,
            positions=dashboard_positions,
            markets=dashboard_markets,
        )

    def print_final_stats(self):
        """Print final results."""
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Strategy: {self.strategy.name}")
        print(f"Total PnL: ${self.total_pnl:+.2f}")
        print(f"Trades: {self.trade_count}")
        print(f"Win Rate: {self.win_count / max(1, self.trade_count) * 100:.1f}%")

    async def run(self):
        """Run the trading engine."""
        self.running = True
        self.refresh_markets()

        if not self.markets:
            print("No markets to trade!")
            return

        tasks = [
            self.price_streamer.stream(),
            self.orderbook_streamer.stream(),
            self.futures_streamer.stream(),
            self.decision_loop(),
        ]

        try:
            await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass  # Handle in finally
        finally:
            print("\n\nShutting down...")
            self.running = False
            self.price_streamer.stop()
            self.orderbook_streamer.stop()
            self.futures_streamer.stop()
            self.close_all_positions()
            self.print_final_stats()

            # Save RL model if training
            if isinstance(self.strategy, RLStrategy) and self.strategy.training:
                self.strategy.save("rl_model")
                print("  [RL] Model saved to rl_model.safetensors")


async def main():
    parser = argparse.ArgumentParser(description="Polymarket Trading")
    parser.add_argument(
        "strategy",
        nargs="?",
        choices=AVAILABLE_STRATEGIES,
        help="Strategy to run"
    )
    parser.add_argument("--train", action="store_true", help="Enable training mode for RL")
    parser.add_argument("--size", type=float, default=10.0, help="Trade size in $")
    parser.add_argument("--load", type=str, help="Load RL model from file")
    parser.add_argument("--dashboard", action="store_true", help="Enable web dashboard")
    parser.add_argument("--port", type=int, default=5050, help="Dashboard port")
    parser.add_argument("--live", action="store_true", help="Enable LIVE trading on Polymarket (REAL MONEY)")
    parser.add_argument("--confirm", action="store_true", help="Skip interactive confirmation for live trading")
    parser.add_argument("--max-exposure", type=float, default=10.0, help="Maximum total $ in open positions (default: 10)")
    parser.add_argument("--max-loss", type=float, default=None, help="Stop trading if losses exceed this $ amount")
    parser.add_argument("--reserve", type=float, default=0.20, help="Keep this fraction of capital as reserve (default: 0.20 = 20%%)")
    parser.add_argument("--max-trades-per-min", type=int, default=10, help="Maximum trades per minute (default: 10)")
    parser.add_argument("--balance", type=float, default=None, help="Starting balance for capital checks (required for --live)")

    args = parser.parse_args()

    if not args.strategy:
        print("Available strategies:")
        for name in AVAILABLE_STRATEGIES:
            print(f"  - {name}")
        print("\nUsage: python run.py <strategy>")
        print("       python run.py rl --train")
        print("       python run.py rl --train --dashboard")
        return

    # Start dashboard in background if requested
    if args.dashboard:
        if DASHBOARD_AVAILABLE:
            dashboard_thread = threading.Thread(
                target=run_dashboard,
                kwargs={'port': args.port},
                daemon=True
            )
            dashboard_thread.start()
            import time
            time.sleep(1)  # Give dashboard time to start
        else:
            print("Warning: Dashboard not available. Install flask-socketio.")

    # Create strategy
    strategy = create_strategy(args.strategy)

    # RL-specific setup
    if isinstance(strategy, RLStrategy):
        if args.load:
            strategy.load(args.load)
            print(f"Loaded RL model from {args.load}")
        if args.train:
            strategy.train()
            print("RL training mode enabled")
        else:
            strategy.eval()

    # Live trading safety confirmation
    live_mode = False
    if args.live:
        # Require balance for live trading
        if args.balance is None:
            print("\nERROR: --balance is required for live trading")
            print("Example: python run.py rl --live --balance 50")
            return

        # Set sensible defaults for live trading if not specified
        max_loss = args.max_loss if args.max_loss else args.balance * 0.5  # Default: stop at 50% loss

        print("\n" + "=" * 60)
        print("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK ⚠️")
        print("=" * 60)
        print("\nSAFETY LIMITS:")
        print(f"  💰 Starting balance:    ${args.balance:.2f}")
        print(f"  📊 Max exposure:        ${args.max_exposure:.2f} (total open positions)")
        print(f"  🛑 Max loss limit:      ${max_loss:.2f} (trading halts if exceeded)")
        print(f"  🏦 Capital reserve:     {args.reserve*100:.0f}% (${args.balance * args.reserve:.2f} untouchable)")
        print(f"  ⏱️  Rate limit:          {args.max_trades_per_min} trades/minute")
        print(f"  💵 Trade size:          ${args.size:.2f} per trade")
        print("\nPRE-FLIGHT CHECKLIST:")
        print("  ✓ .env configured with Polymarket credentials")
        print("  ✓ Tested in paper trading mode")
        print("  ✓ Account has sufficient balance")
        print("  ✓ VPN connected (Oslo recommended)")
        print()

        if args.confirm:
            print("Live trading confirmed via --confirm flag")
        else:
            confirm = input("Type 'LIVE' to confirm live trading: ")
            if confirm.strip() != "LIVE":
                print("Live trading cancelled.")
                return

        print("\n🚀 Starting LIVE trading...")
        live_mode = True
        args.max_loss = max_loss  # Apply default if not set

    # Run
    engine = TradingEngine(
        strategy,
        trade_size=args.size,
        live_mode=live_mode,
        max_exposure=args.max_exposure,
        max_loss=args.max_loss,
        reserve_pct=args.reserve,
        max_trades_per_min=args.max_trades_per_min,
        starting_balance=args.balance,
    )
    await engine.run()


if __name__ == "__main__":
    asyncio.run(main())
