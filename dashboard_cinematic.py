#!/usr/bin/env python3
"""
Cinematic Dashboard v2 - With PnL chart and trade visualization.

Usage:
    python dashboard_cinematic.py
"""
import threading
import time
from datetime import datetime, timezone
from typing import Dict
from flask import Flask, render_template_string
from flask_socketio import SocketIO

# Global state
class DashboardState:
    def __init__(self):
        self.strategy_name = "ppo"
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.positions: Dict[str, dict] = {}
        self.markets: Dict[str, dict] = {}
        self.buffer_size = 0
        self.max_buffer = 2048
        self.updates = 0
        self.entropy = 0.0
        self.avg_reward = 0.0

dashboard_state = DashboardState()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cinematic'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>rl training</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --bg: #050505;
            --surface: #0a0a0a;
            --border: #151515;
            --text: #e0e0e0;
            --dim: #444;
            --green: #00ff88;
            --red: #ff3355;
            --blue: #3388ff;
            --amber: #ffaa00;
        }

        body {
            font-family: 'JetBrains Mono', monospace;
            background: var(--bg);
            color: var(--text);
            height: 100vh;
            overflow: hidden;
        }

        .container {
            display: grid;
            grid-template-columns: 1fr 320px;
            grid-template-rows: 80px 1fr 200px;
            height: 100vh;
            gap: 1px;
            background: var(--border);
        }

        /* Header spans full width */
        .header {
            grid-column: 1 / -1;
            background: var(--surface);
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 32px;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .logo h1 {
            font-size: 13px;
            font-weight: 500;
            color: var(--dim);
            letter-spacing: 3px;
            text-transform: uppercase;
        }

        .live-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid rgba(0, 255, 136, 0.2);
        }

        .live-dot {
            width: 6px;
            height: 6px;
            background: var(--green);
            animation: pulse 1.5s infinite;
        }

        .live-text {
            font-size: 10px;
            color: var(--green);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }

        .header-stats {
            display: flex;
            gap: 48px;
        }

        .header-stat {
            text-align: right;
        }

        .header-stat-value {
            font-size: 32px;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
        }

        .header-stat-value.positive { color: var(--green); }
        .header-stat-value.negative { color: var(--red); }

        .header-stat-label {
            font-size: 10px;
            color: var(--dim);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Main chart area */
        .chart-area {
            background: var(--surface);
            padding: 24px;
            display: flex;
            flex-direction: column;
        }

        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }

        .chart-title {
            font-size: 11px;
            color: var(--dim);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .chart-legend {
            display: flex;
            gap: 16px;
            font-size: 10px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
            color: var(--dim);
        }

        .legend-dot {
            width: 8px;
            height: 8px;
        }

        .legend-dot.pnl { background: var(--green); }
        .legend-dot.win { background: var(--green); opacity: 0.5; }
        .legend-dot.loss { background: var(--red); opacity: 0.5; }

        .chart-container {
            flex: 1;
            position: relative;
            min-height: 0;
        }

        #pnl-chart {
            width: 100%;
            height: 100%;
        }

        /* Sidebar - trades */
        .sidebar {
            background: var(--surface);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .sidebar-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            font-size: 11px;
            color: var(--dim);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .trades-list {
            flex: 1;
            overflow-y: auto;
            padding: 8px;
        }

        .trade-item {
            display: flex;
            align-items: center;
            padding: 12px;
            margin-bottom: 4px;
            background: var(--bg);
            gap: 12px;
        }

        .trade-item.win { border-left: 2px solid var(--green); }
        .trade-item.loss { border-left: 2px solid var(--red); }
        .trade-item.pending { border-left: 2px solid var(--dim); }

        .trade-side {
            font-size: 9px;
            font-weight: 600;
            padding: 4px 8px;
            text-transform: uppercase;
        }

        .trade-side.long { background: rgba(0,255,136,0.15); color: var(--green); }
        .trade-side.short { background: rgba(255,51,85,0.15); color: var(--red); }

        .trade-details {
            flex: 1;
        }

        .trade-asset {
            font-size: 12px;
            font-weight: 500;
        }

        .trade-meta {
            font-size: 10px;
            color: var(--dim);
            margin-top: 2px;
        }

        .trade-pnl {
            font-size: 14px;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
        }

        .trade-pnl.positive { color: var(--green); }
        .trade-pnl.negative { color: var(--red); }

        /* Markets strip */
        .markets-strip {
            grid-column: 1 / -1;
            background: var(--surface);
            display: flex;
            gap: 1px;
            overflow-x: auto;
        }

        .market-card {
            flex: 1;
            min-width: 200px;
            padding: 16px 20px;
            background: var(--bg);
            position: relative;
        }

        .market-card.has-position {
            background: var(--surface);
        }

        .market-card.has-position::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--blue);
        }

        .market-top {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }

        .market-asset {
            font-size: 14px;
            font-weight: 600;
        }

        .market-timer {
            font-size: 20px;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
            color: var(--text);
        }

        .market-timer.urgent {
            color: var(--red);
            animation: blink 0.5s infinite;
        }

        @keyframes blink { 50% { opacity: 0.5; } }

        .market-mid {
            display: flex;
            align-items: baseline;
            gap: 12px;
            margin-bottom: 8px;
        }

        .market-prob {
            font-size: 36px;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            line-height: 1;
        }

        .market-delta {
            font-size: 12px;
            color: var(--dim);
        }

        .market-delta.up { color: var(--green); }
        .market-delta.down { color: var(--red); }

        .market-position {
            display: flex;
            justify-content: space-between;
            padding: 8px 10px;
            font-size: 11px;
            margin-top: 8px;
        }

        .market-position.long { background: rgba(0,255,136,0.1); color: var(--green); }
        .market-position.short { background: rgba(255,51,85,0.1); color: var(--red); }

        .pos-label { font-weight: 500; text-transform: uppercase; }
        .pos-pnl { font-weight: 600; }

        .no-position {
            text-align: center;
            padding: 8px;
            color: var(--dim);
            font-size: 11px;
        }

        /* Time progress */
        .time-progress {
            position: absolute;
            bottom: 0;
            left: 0;
            height: 2px;
            background: var(--blue);
            opacity: 0.5;
            transition: width 1s linear;
        }

        /* Stats row inside markets strip */
        .stats-row {
            display: flex;
            gap: 1px;
            min-width: 300px;
            background: var(--border);
        }

        .stat-cell {
            flex: 1;
            padding: 16px;
            background: var(--bg);
            text-align: center;
        }

        .stat-value {
            font-size: 18px;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
            margin-bottom: 4px;
        }

        .stat-value.green { color: var(--green); }
        .stat-value.red { color: var(--red); }
        .stat-value.amber { color: var(--amber); }

        .stat-label {
            font-size: 9px;
            color: var(--dim);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: var(--border); }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <div class="logo">
                <h1>Cross-Market RL</h1>
                <div class="live-indicator">
                    <div class="live-dot"></div>
                    <span class="live-text">Live</span>
                </div>
            </div>
            <div class="header-stats">
                <div class="header-stat">
                    <div class="header-stat-value" id="trades">0</div>
                    <div class="header-stat-label">Trades</div>
                </div>
                <div class="header-stat">
                    <div class="header-stat-value" id="winrate">0%</div>
                    <div class="header-stat-label">Win Rate</div>
                </div>
                <div class="header-stat">
                    <div class="header-stat-value positive" id="pnl">+$0.00</div>
                    <div class="header-stat-label">Session PnL</div>
                </div>
            </div>
        </header>

        <div class="chart-area">
            <div class="chart-header">
                <span class="chart-title">Equity Curve</span>
                <div class="chart-legend">
                    <div class="legend-item"><div class="legend-dot pnl"></div> PnL</div>
                    <div class="legend-item"><div class="legend-dot win"></div> Win</div>
                    <div class="legend-item"><div class="legend-dot loss"></div> Loss</div>
                </div>
            </div>
            <div class="chart-container">
                <canvas id="pnl-chart"></canvas>
            </div>
        </div>

        <div class="sidebar">
            <div class="sidebar-header">Recent Trades</div>
            <div class="trades-list" id="trades-list">
                <div style="text-align:center;padding:40px;color:var(--dim);font-size:11px;">
                    Waiting for trades...
                </div>
            </div>
        </div>

        <div class="markets-strip">
            <div id="markets-container" style="display:flex;gap:1px;flex:1;">
                <!-- Markets populated by JS -->
            </div>
            <div class="stats-row">
                <div class="stat-cell">
                    <div class="stat-value" id="updates">0</div>
                    <div class="stat-label">Updates</div>
                </div>
                <div class="stat-cell">
                    <div class="stat-value amber" id="entropy">0.00</div>
                    <div class="stat-label">Entropy</div>
                </div>
                <div class="stat-cell">
                    <div class="stat-value" id="buffer">0</div>
                    <div class="stat-label">Buffer</div>
                </div>
                <div class="stat-cell">
                    <div class="stat-value" id="reward">0.00</div>
                    <div class="stat-label">Avg Reward</div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        const socket = io();

        // Chart setup
        let pnlChart;
        let pnlHistory = [];
        let tradeMarkers = [];
        let trades = [];
        let maxPoints = 200;

        function initChart() {
            const ctx = document.getElementById('pnl-chart').getContext('2d');
            pnlChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'PnL',
                        data: [],
                        borderColor: '#00ff88',
                        borderWidth: 2,
                        fill: true,
                        backgroundColor: (context) => {
                            const chart = context.chart;
                            const {ctx, chartArea} = chart;
                            if (!chartArea) return null;
                            const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
                            const lastValue = pnlHistory[pnlHistory.length - 1] || 0;
                            if (lastValue >= 0) {
                                gradient.addColorStop(0, 'rgba(0, 255, 136, 0.15)');
                                gradient.addColorStop(1, 'rgba(0, 255, 136, 0)');
                            } else {
                                gradient.addColorStop(0, 'rgba(255, 51, 85, 0.15)');
                                gradient.addColorStop(1, 'rgba(255, 51, 85, 0)');
                            }
                            return gradient;
                        },
                        tension: 0.3,
                        pointRadius: 0,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: { display: false },
                        tooltip: { enabled: false }
                    },
                    scales: {
                        x: {
                            display: false,
                        },
                        y: {
                            position: 'right',
                            grid: {
                                color: 'rgba(255,255,255,0.03)',
                                drawBorder: false,
                            },
                            ticks: {
                                color: '#444',
                                font: { family: 'JetBrains Mono', size: 10 },
                                callback: (v) => '$' + v.toFixed(0)
                            }
                        }
                    }
                }
            });
        }

        function updateChart(pnl) {
            pnlHistory.push(pnl);
            if (pnlHistory.length > maxPoints) pnlHistory.shift();

            pnlChart.data.labels = pnlHistory.map((_, i) => i);
            pnlChart.data.datasets[0].data = pnlHistory;
            pnlChart.data.datasets[0].borderColor = pnl >= 0 ? '#00ff88' : '#ff3355';
            pnlChart.update('none');
        }

        function formatPnl(v) {
            const sign = v >= 0 ? '+' : '';
            return sign + '$' + Math.abs(v).toFixed(2);
        }

        function formatTime(minutes) {
            const m = Math.floor(minutes);
            const s = Math.round((minutes - m) * 60);
            return m + ':' + String(s).padStart(2, '0');
        }

        socket.on('connect', () => {
            console.log('Connected');
            initChart();
        });

        socket.on('state_update', (d) => {
            // PnL
            const pnl = d.total_pnl || 0;
            const pnlEl = document.getElementById('pnl');
            pnlEl.textContent = formatPnl(pnl);
            pnlEl.className = 'header-stat-value ' + (pnl >= 0 ? 'positive' : 'negative');

            // Update chart
            if (pnlChart) updateChart(pnl);

            // Stats
            const tradeCount = d.trade_count || 0;
            const wins = d.win_count || 0;
            const wr = tradeCount > 0 ? (wins / tradeCount * 100) : 0;

            document.getElementById('trades').textContent = tradeCount;
            const wrEl = document.getElementById('winrate');
            wrEl.textContent = wr.toFixed(0) + '%';
            wrEl.className = 'header-stat-value ' + (wr >= 50 ? 'green' : wr > 0 ? 'red' : '');

            // Markets
            const markets = d.markets || {};
            const positions = d.positions || {};
            const container = document.getElementById('markets-container');

            const marketKeys = Object.keys(markets);
            if (marketKeys.length > 0) {
                container.innerHTML = marketKeys.map(cid => {
                    const m = markets[cid];
                    const pos = positions[cid];
                    const hasPos = pos?.size > 0;
                    const isLong = pos?.side === 'UP';
                    const timeLeft = m.time_left || 0;
                    const vel = m.velocity || 0;
                    const timePercent = (timeLeft / 15) * 100;

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

                    const deltaClass = vel > 0.001 ? 'up' : vel < -0.001 ? 'down' : '';
                    const deltaSign = vel >= 0 ? '+' : '';

                    let posHtml = '<div class="no-position">—</div>';
                    if (hasPos) {
                        posHtml = `
                            <div class="market-position ${isLong ? 'long' : 'short'}">
                                <span class="pos-label">${isLong ? 'long' : 'short'} $${pos.size}</span>
                                <span class="pos-pnl">${formatPnl(posPnl)}</span>
                            </div>
                        `;
                    }

                    return `
                        <div class="market-card ${hasPos ? 'has-position' : ''}">
                            <div class="market-top">
                                <span class="market-asset">${m.asset || '???'}</span>
                                <span class="market-timer ${timeLeft < 2 ? 'urgent' : ''}">${formatTime(timeLeft)}</span>
                            </div>
                            <div class="market-mid">
                                <span class="market-prob">${(m.prob * 100).toFixed(1)}</span>
                                <span class="market-delta ${deltaClass}">${deltaSign}${(vel * 100).toFixed(2)}%</span>
                            </div>
                            ${posHtml}
                            <div class="time-progress" style="width:${timePercent}%"></div>
                        </div>
                    `;
                }).join('');
            }
        });

        socket.on('rl_buffer', (d) => {
            document.getElementById('buffer').textContent = d.buffer_size || 0;

            if (d.avg_reward !== undefined) {
                const el = document.getElementById('reward');
                el.textContent = d.avg_reward.toFixed(4);
                el.className = 'stat-value ' + (d.avg_reward >= 0 ? 'green' : 'red');
            }
        });

        socket.on('rl_update', (d) => {
            const el = document.getElementById('updates');
            el.textContent = parseInt(el.textContent) + 1;

            if (d.entropy !== undefined) {
                document.getElementById('entropy').textContent = d.entropy.toFixed(2);
            }
        });

        socket.on('trade', (t) => {
            // Add to trades list
            const isLong = t.action?.includes('BUY') || t.action?.includes('UP');
            const hasPnl = t.pnl != null;
            const isWin = hasPnl && t.pnl >= 0;

            trades.unshift({
                asset: t.asset,
                side: isLong ? 'long' : 'short',
                pnl: t.pnl,
                size: t.size,
                time: t.time,
                isWin: isWin
            });

            if (trades.length > 50) trades.pop();

            // Update trades list
            document.getElementById('trades-list').innerHTML = trades.map(tr => {
                const statusClass = tr.pnl == null ? 'pending' : (tr.isWin ? 'win' : 'loss');
                const pnlClass = tr.pnl == null ? '' : (tr.pnl >= 0 ? 'positive' : 'negative');
                const pnlText = tr.pnl != null ? formatPnl(tr.pnl) : '$' + (tr.size || 0).toFixed(0);

                return `
                    <div class="trade-item ${statusClass}">
                        <span class="trade-side ${tr.side}">${tr.side}</span>
                        <div class="trade-details">
                            <div class="trade-asset">${tr.asset}</div>
                            <div class="trade-meta">${tr.time}</div>
                        </div>
                        <span class="trade-pnl ${pnlClass}">${pnlText}</span>
                    </div>
                `;
            }).join('');
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


def emit_state():
    socketio.emit('state_update', {
        'strategy_name': dashboard_state.strategy_name,
        'total_pnl': dashboard_state.total_pnl,
        'trade_count': dashboard_state.trade_count,
        'win_count': dashboard_state.win_count,
        'positions': dashboard_state.positions,
        'markets': dashboard_state.markets,
    })


def emit_rl_metrics(metrics: dict):
    socketio.emit('rl_update', metrics)


def emit_rl_buffer(buffer_size: int, max_buffer: int = 2048, avg_reward: float = None):
    data = {'buffer_size': buffer_size, 'max_buffer': max_buffer}
    if avg_reward is not None:
        data['avg_reward'] = avg_reward
    socketio.emit('rl_buffer', data)


def emit_trade(action: str, asset: str, size: float = 0, pnl: float = None):
    socketio.emit('trade', {
        'action': action,
        'asset': asset,
        'size': size,
        'pnl': pnl,
        'time': datetime.now().strftime('%H:%M:%S'),
    })


def state_emitter():
    while True:
        time.sleep(0.25)
        emit_state()


def update_dashboard_state(
    strategy_name: str = None,
    total_pnl: float = None,
    trade_count: int = None,
    win_count: int = None,
    positions: dict = None,
    markets: dict = None,
):
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


def update_rl_metrics(metrics: dict):
    emit_rl_metrics(metrics)


def run_dashboard(host='0.0.0.0', port=5051):
    import os
    port = int(os.environ.get('PORT', port))

    print(f"\n  Cinematic Dashboard v2")
    print(f"  http://localhost:{port}\n")

    emitter_thread = threading.Thread(target=state_emitter, daemon=True)
    emitter_thread.start()

    socketio.run(app, host=host, port=port, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    run_dashboard()
