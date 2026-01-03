#!/usr/bin/env python3
"""
RL Trading Lab - Real-time dashboard for reinforcement learning trading experiments.

Usage:
    python dashboard.py                    # Start dashboard server only
    python dashboard.py --with-trading rl  # Start dashboard + trading
    python dashboard.py --with-trading rl --train  # Dashboard + RL training
"""
import asyncio
import argparse
import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
from collections import deque
import json

from flask import Flask, render_template_string
from flask_socketio import SocketIO

# Global state for dashboard
class DashboardState:
    def __init__(self):
        self.strategy_name = ""
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.positions: Dict[str, dict] = {}
        self.markets: Dict[str, dict] = {}
        self.rl_metrics: List[dict] = []
        self.pnl_history: List[dict] = []
        self.last_update = None

dashboard_state = DashboardState()

# Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'rl-trading-lab'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>rl trading</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --bg: #0c0c0c;
            --surface: #111;
            --border: #222;
            --fg: #e8e8e8;
            --dim: #666;
            --muted: #444;
            --green: #4ade80;
            --red: #f87171;
            --blue: #60a5fa;
            --yellow: #fbbf24;
        }
        html, body {
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
            font-size: 12px;
            background: var(--bg);
            color: var(--fg);
            height: 100vh;
            overflow: hidden;
        }
        .app {
            display: grid;
            grid-template-rows: 40px 1fr;
            height: 100vh;
        }
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 16px;
            border-bottom: 1px solid var(--border);
            background: var(--surface);
        }
        .header-left {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .logo {
            color: var(--dim);
            font-size: 11px;
        }
        .status {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--muted);
        }
        .dot.live { background: var(--green); }
        .dot.training { background: var(--yellow); animation: pulse 1.5s infinite; }
        @keyframes pulse { 50% { opacity: 0.5; } }
        .strat { color: var(--dim); }
        .metrics {
            display: flex;
            gap: 32px;
        }
        .metric {
            text-align: right;
        }
        .metric-val {
            font-size: 14px;
            font-weight: 500;
        }
        .metric-val.pos { color: var(--green); }
        .metric-val.neg { color: var(--red); }
        .metric-lbl {
            font-size: 10px;
            color: var(--muted);
            text-transform: lowercase;
        }
        .clock {
            color: var(--dim);
            font-size: 11px;
        }
        .main {
            display: grid;
            grid-template-columns: 1fr 280px;
            overflow: hidden;
        }
        .content {
            padding: 16px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .section-lbl {
            font-size: 10px;
            color: var(--muted);
            text-transform: lowercase;
            margin-bottom: 8px;
            letter-spacing: 0.5px;
        }
        .markets-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
        }
        .market {
            background: var(--surface);
            border: 1px solid var(--border);
            padding: 12px;
        }
        .market.active {
            border-color: var(--blue);
        }
        .market-head {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        .market-asset {
            color: var(--fg);
            font-size: 11px;
        }
        .market-time {
            color: var(--muted);
            font-size: 10px;
        }
        .market-time.urgent { color: var(--red); }
        .market-prob {
            font-size: 24px;
            font-weight: 500;
            margin-bottom: 2px;
        }
        .market-delta {
            font-size: 10px;
            color: var(--muted);
        }
        .market-delta.up { color: var(--green); }
        .market-delta.down { color: var(--red); }
        .market-flow {
            margin-top: 8px;
            height: 2px;
            background: var(--border);
            position: relative;
        }
        .flow-bar {
            position: absolute;
            top: 0;
            height: 100%;
        }
        .flow-bar.buy { left: 50%; background: var(--green); }
        .flow-bar.sell { right: 50%; background: var(--red); }
        .market-pos {
            margin-top: 8px;
            padding: 6px 8px;
            font-size: 10px;
            display: flex;
            justify-content: space-between;
        }
        .market-pos.long { background: rgba(74,222,128,0.1); color: var(--green); }
        .market-pos.short { background: rgba(248,113,113,0.1); color: var(--red); }
        .no-pos {
            margin-top: 8px;
            font-size: 10px;
            color: var(--muted);
            text-align: center;
            padding: 6px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
        }
        .stat {
            background: var(--surface);
            border: 1px solid var(--border);
            padding: 12px;
        }
        .stat-val {
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 2px;
        }
        .stat-val.pos { color: var(--green); }
        .stat-val.neg { color: var(--red); }
        .stat-lbl {
            font-size: 10px;
            color: var(--muted);
            text-transform: lowercase;
        }
        .rl-panel {
            background: var(--surface);
            border: 1px solid var(--border);
        }
        .rl-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
        }
        .rl-title {
            font-size: 11px;
            color: var(--fg);
        }
        .rl-badge {
            font-size: 9px;
            padding: 2px 6px;
            text-transform: lowercase;
            color: var(--muted);
            border: 1px solid var(--border);
        }
        .rl-badge.training {
            color: var(--yellow);
            border-color: var(--yellow);
        }
        .rl-body {
            padding: 12px;
        }
        .buffer-row {
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            margin-bottom: 6px;
        }
        .buffer-lbl { color: var(--muted); }
        .buffer-val { color: var(--fg); }
        .buffer-track {
            height: 3px;
            background: var(--border);
            margin-bottom: 12px;
        }
        .buffer-fill {
            height: 100%;
            background: var(--blue);
            transition: width 0.3s;
        }
        .rl-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }
        .rl-stat {
            text-align: center;
            padding: 8px;
            background: var(--bg);
        }
        .rl-stat-val {
            font-size: 12px;
            font-weight: 500;
            margin-bottom: 2px;
        }
        .rl-stat-lbl {
            font-size: 9px;
            color: var(--muted);
            text-transform: lowercase;
        }
        .sidebar {
            background: var(--surface);
            border-left: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .sidebar-section {
            padding: 12px;
            border-bottom: 1px solid var(--border);
        }
        .sidebar-section:last-child {
            flex: 1;
            overflow-y: auto;
            border-bottom: none;
        }
        .sidebar-lbl {
            font-size: 10px;
            color: var(--muted);
            text-transform: lowercase;
            margin-bottom: 8px;
        }
        .chart-wrap {
            height: 60px;
        }
        .trades {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .trade {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px;
            background: var(--bg);
        }
        .trade-side {
            font-size: 9px;
            font-weight: 500;
            padding: 2px 4px;
            text-transform: lowercase;
        }
        .trade-side.buy { color: var(--green); background: rgba(74,222,128,0.1); }
        .trade-side.sell { color: var(--red); background: rgba(248,113,113,0.1); }
        .trade-info { flex: 1; }
        .trade-asset { font-size: 11px; }
        .trade-time { font-size: 9px; color: var(--muted); }
        .trade-pnl { font-size: 11px; }
        .trade-pnl.pos { color: var(--green); }
        .trade-pnl.neg { color: var(--red); }
        .empty {
            text-align: center;
            padding: 24px;
            color: var(--muted);
            font-size: 11px;
        }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border); }
        @media (max-width: 1200px) { .markets-grid { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 900px) { .main { grid-template-columns: 1fr; } .sidebar { display: none; } }
    </style>
</head>
<body>
    <div class="app">
        <header class="header">
            <div class="header-left">
                <span class="logo">rl/trading</span>
                <div class="status">
                    <span class="dot" id="status-dot"></span>
                    <span class="strat" id="strategy-name">ppo</span>
                </div>
            </div>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-val" id="total-pnl">$0.00</div>
                    <div class="metric-lbl">pnl</div>
                </div>
                <div class="metric">
                    <div class="metric-val" id="win-rate">0%</div>
                    <div class="metric-lbl">win</div>
                </div>
                <div class="metric">
                    <div class="metric-val" id="trade-count">0</div>
                    <div class="metric-lbl">trades</div>
                </div>
            </div>
            <div class="clock" id="clock">00:00:00</div>
        </header>
        <div class="main">
            <div class="content">
                <section>
                    <div class="section-lbl">markets</div>
                    <div class="markets-grid" id="markets-grid">
                        <div class="market"><div class="empty">waiting...</div></div>
                    </div>
                </section>
                <section>
                    <div class="section-lbl">performance</div>
                    <div class="stats-grid">
                        <div class="stat">
                            <div class="stat-val" id="exposure">$0</div>
                            <div class="stat-lbl">exposure</div>
                        </div>
                        <div class="stat">
                            <div class="stat-val" id="avg-trade">$0.00</div>
                            <div class="stat-lbl">avg trade</div>
                        </div>
                        <div class="stat">
                            <div class="stat-val pos" id="best-trade">$0.00</div>
                            <div class="stat-lbl">best</div>
                        </div>
                        <div class="stat">
                            <div class="stat-val neg" id="worst-trade">$0.00</div>
                            <div class="stat-lbl">worst</div>
                        </div>
                    </div>
                </section>
                <section>
                    <div class="section-lbl">agent</div>
                    <div class="rl-panel">
                        <div class="rl-head">
                            <span class="rl-title">ppo network</span>
                            <span class="rl-badge" id="rl-status">collecting</span>
                        </div>
                        <div class="rl-body">
                            <div class="buffer-row">
                                <span class="buffer-lbl">buffer</span>
                                <span class="buffer-val" id="buffer-count">0 / 256</span>
                            </div>
                            <div class="buffer-track">
                                <div class="buffer-fill" id="buffer-progress" style="width: 0%"></div>
                            </div>
                            <div class="rl-stats">
                                <div class="rl-stat">
                                    <div class="rl-stat-val" id="updates">0</div>
                                    <div class="rl-stat-lbl">updates</div>
                                </div>
                                <div class="rl-stat">
                                    <div class="rl-stat-val" id="avg-reward">--</div>
                                    <div class="rl-stat-lbl">reward</div>
                                </div>
                                <div class="rl-stat">
                                    <div class="rl-stat-val" id="policy-loss">--</div>
                                    <div class="rl-stat-lbl">policy</div>
                                </div>
                                <div class="rl-stat">
                                    <div class="rl-stat-val" id="value-loss">--</div>
                                    <div class="rl-stat-lbl">value</div>
                                </div>
                                <div class="rl-stat">
                                    <div class="rl-stat-val" id="entropy">--</div>
                                    <div class="rl-stat-lbl">entropy</div>
                                </div>
                                <div class="rl-stat">
                                    <div class="rl-stat-val" id="kl-div">--</div>
                                    <div class="rl-stat-lbl">kl</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>
            </div>
            <aside class="sidebar">
                <div class="sidebar-section">
                    <div class="sidebar-lbl">pnl</div>
                    <div class="chart-wrap"><canvas id="pnl-chart"></canvas></div>
                </div>
                <div class="sidebar-section">
                    <div class="sidebar-lbl">loss</div>
                    <div class="chart-wrap"><canvas id="loss-chart"></canvas></div>
                </div>
                <div class="sidebar-section">
                    <div class="sidebar-lbl">trades</div>
                    <div class="trades" id="trades-list">
                        <div class="empty">no trades</div>
                    </div>
                </div>
            </aside>
        </div>
    </div>
    <script>
        const socket = io();
        let pnlChart, lossChart;
        let pnlData = { labels: [], data: [] };
        let lossData = { labels: [], policy: [], value: [] };
        let tradesList = [];
        let updateCount = 0;
        let bestTrade = 0, worstTrade = 0;
        let lastMarketsHash = '';  // Track if markets actually changed

        function updateClock() {
            document.getElementById('clock').textContent = new Date().toLocaleTimeString('en-US', {hour12: false});
        }
        setInterval(updateClock, 1000);
        updateClock();

        const chartOpts = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { enabled: false } },
            scales: { x: { display: false }, y: { display: false } },
            elements: { point: { radius: 0 }, line: { borderWidth: 1.5, tension: 0.3 } },
            animation: false
        };

        function initCharts() {
            pnlChart = new Chart(document.getElementById('pnl-chart'), {
                type: 'line',
                data: { labels: [], datasets: [{ data: [], borderColor: '#4ade80', fill: false }] },
                options: chartOpts
            });
            lossChart = new Chart(document.getElementById('loss-chart'), {
                type: 'line',
                data: { labels: [], datasets: [
                    { data: [], borderColor: '#f87171', fill: false },
                    { data: [], borderColor: '#60a5fa', fill: false }
                ]},
                options: chartOpts
            });
        }

        function fmt(v) { return (v >= 0 ? '+' : '') + '$' + v.toFixed(2); }
        function fmtTime(m) { return Math.floor(m) + ':' + String(Math.round((m % 1) * 60)).padStart(2, '0'); }

        socket.on('connect', () => {
            document.getElementById('status-dot').classList.add('live');
        });
        socket.on('disconnect', () => {
            document.getElementById('status-dot').classList.remove('live');
        });

        socket.on('state_update', (d) => {
            document.getElementById('strategy-name').textContent = (d.strategy_name || 'ppo').toLowerCase();
            const pnl = d.total_pnl || 0;
            const pnlEl = document.getElementById('total-pnl');
            pnlEl.textContent = fmt(pnl);
            pnlEl.className = 'metric-val ' + (pnl >= 0 ? 'pos' : 'neg');

            const tc = d.trade_count || 0;
            document.getElementById('trade-count').textContent = tc;
            const wr = tc > 0 ? (d.win_count || 0) / tc * 100 : 0;
            const wrEl = document.getElementById('win-rate');
            wrEl.textContent = wr.toFixed(0) + '%';
            wrEl.className = 'metric-val ' + (wr >= 50 ? 'pos' : wr > 0 ? 'neg' : '');

            document.getElementById('avg-trade').textContent = fmt(tc > 0 ? pnl / tc : 0);
            let exp = 0;
            Object.values(d.positions || {}).forEach(p => { if (p.size > 0) exp += p.size; });
            document.getElementById('exposure').textContent = '$' + exp.toFixed(0);
            document.getElementById('best-trade').textContent = fmt(bestTrade);
            document.getElementById('worst-trade').textContent = fmt(worstTrade);

            const markets = d.markets || {};
            const positions = d.positions || {};
            const grid = document.getElementById('markets-grid');
            const marketKeys = Object.keys(markets);
            // Create hash to detect structural changes (new/removed markets)
            const marketsHash = marketKeys.sort().join(',');
            const needsRebuild = marketsHash !== lastMarketsHash;

            if (marketKeys.length > 0) {
                if (needsRebuild) {
                    // Full rebuild only when markets change
                    lastMarketsHash = marketsHash;
                    grid.innerHTML = Object.entries(markets).map(([cid, m]) => {
                        return `<div class="market" data-cid="${cid}">
                            <div class="market-head">
                                <span class="market-asset">${m.asset || 'ASSET'}</span>
                                <span class="market-time"></span>
                            </div>
                            <div class="market-prob"></div>
                            <div class="market-delta"></div>
                            <div class="market-flow"><div class="flow-bar"></div></div>
                            <div class="market-pos-wrap"></div>
                        </div>`;
                    }).join('');
                }
                // Update existing elements in-place (fast)
                Object.entries(markets).forEach(([cid, m]) => {
                    const el = grid.querySelector(`[data-cid="${cid}"]`);
                    if (!el) return;
                    const pos = positions[cid];
                    const hasPos = pos?.size > 0;
                    const isLong = pos?.side === 'UP';
                    const vel = m.velocity || 0;
                    const timeLeft = m.time_left || 0;
                    let posPnl = 0;
                    if (hasPos && m.prob && pos.entry_price > 0) {
                        const shares = pos.size / pos.entry_price;
                        if (isLong) {
                            posPnl = (m.prob - pos.entry_price) * shares;
                        } else {
                            const currentDownPrice = 1 - m.prob;
                            posPnl = (currentDownPrice - pos.entry_price) * shares;
                        }
                    }
                    const flowPct = Math.min(50, Math.abs(vel) * 5000);

                    el.className = 'market' + (hasPos ? ' active' : '');
                    el.querySelector('.market-time').textContent = fmtTime(timeLeft);
                    el.querySelector('.market-time').className = 'market-time' + (timeLeft < 2 ? ' urgent' : '');
                    el.querySelector('.market-prob').textContent = (m.prob * 100).toFixed(1) + '%';
                    const deltaEl = el.querySelector('.market-delta');
                    deltaEl.textContent = (vel >= 0 ? '+' : '') + (vel * 100).toFixed(2) + '%';
                    deltaEl.className = 'market-delta' + (vel > 0.001 ? ' up' : vel < -0.001 ? ' down' : '');
                    const flowBar = el.querySelector('.flow-bar');
                    flowBar.className = 'flow-bar ' + (vel >= 0 ? 'buy' : 'sell');
                    flowBar.style.width = flowPct + '%';
                    const posWrap = el.querySelector('.market-pos-wrap');
                    if (hasPos) {
                        posWrap.innerHTML = `<div class="market-pos ${isLong ? 'long' : 'short'}"><span>${isLong ? 'long' : 'short'} $${pos.size}</span><span>${fmt(posPnl)}</span></div>`;
                    } else {
                        posWrap.innerHTML = '<div class="no-pos">-</div>';
                    }
                });
            }

            if (pnlData.data.length === 0 || pnlData.data[pnlData.data.length - 1] !== pnl) {
                pnlData.labels.push('');
                pnlData.data.push(pnl);
                if (pnlData.labels.length > 100) { pnlData.labels.shift(); pnlData.data.shift(); }
                if (pnlChart) {
                    pnlChart.data.labels = pnlData.labels;
                    pnlChart.data.datasets[0].data = pnlData.data;
                    pnlChart.data.datasets[0].borderColor = pnl >= 0 ? '#4ade80' : '#f87171';
                    pnlChart.update('none');
                }
            }
        });

        socket.on('rl_buffer', (d) => {
            const pct = Math.min(100, (d.buffer_size || 0) / (d.max_buffer || 256) * 100);
            document.getElementById('buffer-count').textContent = `${d.buffer_size || 0} / ${d.max_buffer || 256}`;
            document.getElementById('buffer-progress').style.width = pct + '%';
            const status = document.getElementById('rl-status');
            const dot = document.getElementById('status-dot');
            if (pct >= 100) {
                status.textContent = 'training';
                status.className = 'rl-badge training';
                dot.className = 'dot training';
            } else {
                status.textContent = 'collecting';
                status.className = 'rl-badge';
                dot.className = 'dot live';
            }
            if (d.avg_reward !== undefined) {
                const el = document.getElementById('avg-reward');
                el.textContent = d.avg_reward.toFixed(4);
                el.style.color = d.avg_reward >= 0 ? 'var(--green)' : 'var(--red)';
            }
        });

        socket.on('rl_update', (d) => {
            updateCount++;
            document.getElementById('updates').textContent = updateCount;
            document.getElementById('policy-loss').textContent = d.policy_loss?.toFixed(4) || '--';
            document.getElementById('value-loss').textContent = d.value_loss?.toFixed(4) || '--';
            document.getElementById('entropy').textContent = d.entropy?.toFixed(3) || '--';
            document.getElementById('kl-div').textContent = d.approx_kl?.toFixed(5) || '--';

            lossData.labels.push('');
            lossData.policy.push(d.policy_loss || 0);
            lossData.value.push(d.value_loss || 0);
            if (lossData.labels.length > 50) { lossData.labels.shift(); lossData.policy.shift(); lossData.value.shift(); }
            if (lossChart) {
                lossChart.data.labels = lossData.labels;
                lossChart.data.datasets[0].data = lossData.policy;
                lossChart.data.datasets[1].data = lossData.value;
                lossChart.update('none');
            }
        });

        socket.on('trade', (t) => {
            if (t.pnl != null) {
                if (t.pnl > bestTrade) bestTrade = t.pnl;
                if (t.pnl < worstTrade) worstTrade = t.pnl;
            }
            tradesList.unshift(t);
            if (tradesList.length > 20) tradesList.pop();
            document.getElementById('trades-list').innerHTML = tradesList.map(tr => {
                const isBuy = tr.action?.includes('BUY') || tr.action?.includes('UP');
                const hasPnl = tr.pnl != null;
                return `<div class="trade">
                    <span class="trade-side ${isBuy ? 'buy' : 'sell'}">${isBuy ? 'buy' : 'sell'}</span>
                    <div class="trade-info"><div class="trade-asset">${tr.asset || 'ASSET'}</div><div class="trade-time">${tr.time || 'now'}</div></div>
                    <span class="trade-pnl ${hasPnl ? (tr.pnl >= 0 ? 'pos' : 'neg') : ''}">${hasPnl ? fmt(tr.pnl) : '$' + (tr.size || 0).toFixed(0)}</span>
                </div>`;
            }).join('');
        });

        document.addEventListener('DOMContentLoaded', initCharts);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


def emit_state():
    """Emit current state to all clients."""
    socketio.emit('state_update', {
        'strategy_name': dashboard_state.strategy_name,
        'total_pnl': dashboard_state.total_pnl,
        'trade_count': dashboard_state.trade_count,
        'win_count': dashboard_state.win_count,
        'positions': dashboard_state.positions,
        'markets': dashboard_state.markets,
    })


def emit_rl_metrics(metrics: dict):
    """Emit RL training metrics."""
    socketio.emit('rl_update', metrics)


def emit_rl_buffer(buffer_size: int, max_buffer: int = 2048, avg_reward: float = None):
    """Emit RL buffer progress (call every tick during training)."""
    data = {'buffer_size': buffer_size, 'max_buffer': max_buffer}
    if avg_reward is not None:
        data['avg_reward'] = avg_reward
    socketio.emit('rl_buffer', data)


def emit_trade(action: str, asset: str, size: float = 0, pnl: float = None):
    """Emit trade event for activity log."""
    from datetime import datetime
    socketio.emit('trade', {
        'action': action,
        'asset': asset,
        'size': size,
        'pnl': pnl,
        'time': datetime.now().strftime('%H:%M:%S'),
    })


# Background emitter thread
def state_emitter():
    """Periodically emit state updates."""
    while True:
        time.sleep(0.25)  # 250ms for responsive UI
        emit_state()


def run_dashboard(host='0.0.0.0', port=None):
    """Run the dashboard server."""
    import os
    # Use PORT env var for Railway, fallback to 5050
    if port is None:
        port = int(os.environ.get('PORT', 5050))

    print(f"\n  RL Trading Lab")
    print(f"  http://localhost:{port}\n")

    # Start state emitter thread
    emitter_thread = threading.Thread(target=state_emitter, daemon=True)
    emitter_thread.start()

    socketio.run(app, host=host, port=port, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)


# Monkey-patch for integration with run.py
def update_dashboard_state(
    strategy_name: str = None,
    total_pnl: float = None,
    trade_count: int = None,
    win_count: int = None,
    positions: dict = None,
    markets: dict = None,
):
    """Update dashboard state from trading engine."""
    if strategy_name is not None:
        dashboard_state.strategy_name = strategy_name
    if total_pnl is not None:
        dashboard_state.total_pnl = total_pnl
    if trade_count is not None:
        dashboard_state.trade_count = trade_count
    if win_count is not None:
        dashboard_state.win_count = win_count
    if positions is not None:
        dashboard_state.positions = positions
    if markets is not None:
        dashboard_state.markets = markets
    dashboard_state.last_update = datetime.now(timezone.utc)


def update_rl_metrics(metrics: dict):
    """Update RL metrics from training."""
    dashboard_state.rl_metrics.append(metrics)
    emit_rl_metrics(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RL Trading Lab")
    parser.add_argument('--port', type=int, default=5050, help='Dashboard port')
    parser.add_argument('--with-trading', type=str, help='Run trading with specified strategy')
    parser.add_argument('--train', action='store_true', help='Enable RL training mode')
    args = parser.parse_args()

    if args.with_trading:
        print("Running dashboard with trading engine...")
        print("Use separate terminal for now - integration coming soon")

    run_dashboard(port=args.port)
