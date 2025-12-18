import React, { useState, useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import TrainingDashboard from './components/TrainingDashboard';
import HistoryDashboard from './components/HistoryDashboard';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// ═══════════════════════════════════════════════════════════════════════════
// BLUEBIRD PRIVATE - Midnight Luxury Trading Suite
// ═══════════════════════════════════════════════════════════════════════════

// Visual threshold gauge for risk triggers
const TriggerGauge = ({ label, value, threshold, max, format, invert = false, direction = null }) => {
  // For momentum: value and threshold are negative, we want to show how "bad" the value is
  // For correlation/ADX: value approaches threshold from below

  let fillPercent, thresholdPercent, status;

  if (invert) {
    // Momentum: threshold is negative (e.g., -0.015), value is negative (e.g., -0.018)
    // More negative = worse. Scale from 0 to threshold*2 for visual range
    const absThreshold = Math.abs(threshold);
    const absValue = Math.abs(value || 0);
    const visualMax = absThreshold * 3; // Show up to 3x threshold
    fillPercent = Math.min((absValue / visualMax) * 100, 100);
    thresholdPercent = (absThreshold / visualMax) * 100;
    status = absValue >= absThreshold ? 'danger' : absValue >= absThreshold * 0.7 ? 'warning' : 'safe';
  } else {
    // Correlation/ADX: threshold is positive, value approaches from below
    const visualMax = max || (threshold * 1.2);
    fillPercent = Math.min(((value || 0) / visualMax) * 100, 100);
    thresholdPercent = (threshold / visualMax) * 100;
    status = (value || 0) >= threshold ? 'danger' : (value || 0) >= threshold * 0.85 ? 'warning' : 'safe';
  }

  const displayValue = format ? format(value || 0) : (value || 0).toFixed(2);
  const displayThreshold = format ? format(threshold) : threshold.toFixed(2);

  return (
    <div className="trigger-gauge">
      <div className="trigger-gauge-header">
        <span className="trigger-gauge-label">
          {label}
          {direction && direction !== 'neutral' && (
            <span className={`trigger-gauge-direction ${direction}`}>
              {direction === 'down' ? ' ↓' : direction === 'up' ? ' ↑' : ''}
            </span>
          )}
        </span>
        <span className={`trigger-gauge-value ${status}`}>{displayValue}</span>
      </div>
      <div className="trigger-gauge-bar">
        <div
          className="trigger-gauge-threshold"
          style={{ left: `${thresholdPercent}%` }}
          title={`Threshold: ${displayThreshold}`}
        />
        <div
          className={`trigger-gauge-fill ${status}`}
          style={{ width: `${fillPercent}%` }}
        />
      </div>
      <div className="trigger-gauge-scale">
        <span className="trigger-gauge-min">{invert ? '0%' : '0'}</span>
        <span className="trigger-gauge-threshold-label">{displayThreshold}</span>
      </div>
    </div>
  );
};

function App() {
  // Backend connection settings
  // NOTE: The dashboard may be opened from another machine; avoid hardcoding localhost.
  const API_HOST = import.meta.env.VITE_API_HOST || window.location.hostname;
  const API_PORT = import.meta.env.VITE_API_PORT || '8000';
  const API_BASE = `${window.location.protocol}//${API_HOST}:${API_PORT}`;
  const WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const WS_URL = `${WS_PROTOCOL}://${API_HOST}:${API_PORT}/ws`;

  const [currentView, setCurrentView] = useState('trading');
  const [status, setStatus] = useState('disconnected');
  const [price, setPrice] = useState(0.0);
  const [logs, setLogs] = useState([]);
  const [account, setAccount] = useState({ equity: 0, buying_power: 0, balance: 0 });
  const [positions, setPositions] = useState([]);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [secondsSinceUpdate, setSecondsSinceUpdate] = useState(0);
  const [autoRefreshInterval, setAutoRefreshInterval] = useState(5); // seconds
  const [data, setData] = useState({});

  // Symbol selector state
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USD');
  const [symbolPrices, setSymbolPrices] = useState({
    'BTC/USD': { price: 0, change: 0, changePercent: 0, history: [] },
    'SOL/USD': { price: 0, change: 0, changePercent: 0, history: [] },
    'LTC/USD': { price: 0, change: 0, changePercent: 0, history: [] },
    'AVAX/USD': { price: 0, change: 0, changePercent: 0, history: [] }
  });

  const [ai, setAi] = useState({
    prediction: null,
    confidence: 0,
    signal: 'HOLD',
    reasoning: [],
    features: {},
    thresholds: {},
    multi_timeframe: {},
    feature_importance: {}
  });

  const [ultra, setUltra] = useState({
    regime: 'AI_ADAPTIVE',
    strategy: 'WAIT',
    confidence: 0,
    signal: 'HOLD',
    should_trade: false,
    trade_reason: '',
    metrics: {},
    time_filter: { score: 0.5, window_name: 'NEUTRAL' },
    kelly: {}
  });

  const [grid, setGrid] = useState({
    active: false,
    total_trades: 0,
    total_profit: 0,
    summaries: {}
  });

  // Derived grid metrics (from per-symbol summaries)
  const gridTotalFills = Object.values(grid.summaries || {}).reduce(
    (sum, s) => sum + (s?.performance?.completed_trades || 0),
    0
  );
  const gridTotalCycles = Object.values(grid.summaries || {}).reduce(
    (sum, s) => sum + (s?.performance?.completed_cycles || 0),
    0
  );

  const [risk, setRisk] = useState({
    daily_pnl: 0,
    daily_pnl_pct: 0,
    drawdown_pct: 0,
    peak_equity: 0,
    daily_limit_hit: false,
    max_drawdown_hit: false,
    trading_halted: false,
    stop_losses: {},
    // All-time performance (since Nov 24)
    alltime_starting_equity: 96811.55,
    alltime_starting_date: '2025-11-24',
    alltime_pnl: 0,
    alltime_pnl_pct: 0,
    // Grid trading performance (since Dec 2)
    grid_starting_equity: 90276.26,
    grid_starting_date: '2025-12-02',
    grid_pnl: 0,
    grid_pnl_pct: 0
  });

  // New profitability tracking state
  const [smartFilters, setSmartFilters] = useState({
    time_filter: { enabled: true, should_trade: true, time_quality: 1.0, reason: '', hour: 0, is_weekend: false },
    correlations: {},
    momentum: { status: {}, allow_buy: true, allow_sell: true }
  });

  // Smart Grid Regime Detection state
  const [regime, setRegime] = useState({
    current: 'UNKNOWN',
    adx: 0,
    allow_buy: true,
    allow_sell: true,
    is_strong_trend: false,
    paused: false,
    reason: 'Initializing...',
    strategy_hint: 'WAIT',
    confidence: 0,
    all_regimes: {},
    all_paused: {}
  });

  const [performance, setPerformance] = useState({
    total_trades: 0,
    total_profit: 0,
    avg_profit_per_trade: 0,
    win_rate: 0,
    best_hour: 0,
    best_hour_profit: 0,
    worst_hour: 0,
    worst_hour_profit: 0,
    trades_by_hour: {},
    profit_by_hour: {},
    expected_vs_actual: 0
  });

  const [market, setMarket] = useState({ high: 0, low: 0, volume: 0, change: 0 });
  const [lastTrade, setLastTrade] = useState(null);

  // Order confirmation tracking state
  const [orders, setOrders] = useState({
    confirmed: [],
    stats: {
      total_confirmed: 0,
      by_symbol: {},
      by_side: { buy: 0, sell: 0 },
      total_volume: 0
    },
    reconciliation: {
      synced: true,
      last_check: null,
      matched: 0,
      total: 0,
      discrepancies: []
    }
  });

  // SMS Notifier state
  const [notifier, setNotifier] = useState({
    running: false,
    pid: null,
    last_sms_sent: null,
    sms_count_today: 0,
    quiet_hours_active: false,
    uptime: null,
    history: [],
    config: {
      quiet_hours_enabled: true,
      quiet_hours_start: 23,
      quiet_hours_end: 7,
      notify_on_trade: true,
      notify_on_alert: true,
      notify_daily_summary: true,
      notify_on_startup: true,
      phone_number: '+1••••••••••'
    }
  });
  const [showNotifierPanel, setShowNotifierPanel] = useState(false);
  const [notifierLoading, setNotifierLoading] = useState(false);
  const [notifierTab, setNotifierTab] = useState('status'); // status | history | settings | watchdog

  // Watchdog state
  const [watchdog, setWatchdog] = useState({
    enabled: true,
    status: 'initializing',
    last_check: null,
    last_restart: null,
    restart_count: 0,
    notifier_health: {
      healthy: true,
      reason: 'OK',
      trades_since_last_sms: 0,
      minutes_since_last_sms: 0
    }
  });

  // Stream health state (independent of WebSocket - detects Alpaca stream issues)
  const [streamHealth, setStreamHealth] = useState({
    status: 'unknown',
    secondsSinceBar: 0
  });

  // Comprehensive Health state (stream + order tracking from bot)
  const [health, setHealth] = useState({
    stream: {
      status: 'unknown',
      last_bar_at: null,
      seconds_since_bar: 0,
      stale_threshold_seconds: 180
    },
    orders: {
      tracked_open_limits: 0,
      alpaca_open_grid_limits: 0,
      mismatch: 0,
      healthy: true,
      last_check_at: null,
      orphan_ids: [],
      stale_ids: []
    }
  });

  // Circuit Breaker state (persists across bot restarts)
  const [circuitBreaker, setCircuitBreaker] = useState({
    from_file: {},
    from_api: {},
    loading: false,
    lastFetch: null,
    anyTriggered: false
  });
  const [circuitBreakerResetting, setCircuitBreakerResetting] = useState(false);

  // Windfall Profit-Taking state (captures profits on big unrealized gains)
  const [windfall, setWindfall] = useState({
    enabled: false,
    total_captures: 0,
    total_profit: 0,
    transactions: [],
    config: {
      soft_threshold_pct: 4.0,
      hard_threshold_pct: 6.0,
      rsi_threshold: 70,
      sell_portion: 0.70
    },
    active_cooldowns: {}
  });

  // Risk Overlay state (protects capital during adverse market conditions)
  const [riskOverlay, setRiskOverlay] = useState({
    enabled: true,
    mode: 'NORMAL',
    mode_since: null,
    mode_duration_minutes: 0,
    trigger_reasons: [],
    position_multiplier: 1.0,
    recovery: null,
    telemetry: {
      avoided_buys_count: 0,
      avoided_buys_notional: 0,
      cancelled_limits_count: 0,
      cancelled_limits_notional: 0,
      untracked_buys_count: 0,
      rebalances_blocked_count: 0
    },
    manual_override: null,
    caps: { normal_exposure: 0.70, risk_off_exposure: 0.40 }
  });
  const [riskOverlayExpanded, setRiskOverlayExpanded] = useState(false);
  const [riskOverlayFetch, setRiskOverlayFetch] = useState({
    status: 'init', // init | ok | error
    lastFetchedAt: null,
    error: null
  });

  // Per-symbol chart data storage
  const [symbolChartData, setSymbolChartData] = useState({
    'BTC/USD': { labels: [], data: [] },
    'SOL/USD': { labels: [], data: [] },
    'LTC/USD': { labels: [], data: [] },
    'AVAX/USD': { labels: [], data: [] }
  });

  // Computed chart data based on selected symbol
  const chartData = {
    labels: symbolChartData[selectedSymbol]?.labels || [],
    datasets: [{
      label: selectedSymbol,
      data: symbolChartData[selectedSymbol]?.data || [],
      borderColor: '#d4af37',
      backgroundColor: 'rgba(212, 175, 55, 0.08)',
      tension: 0.4,
      fill: true,
      pointRadius: 0,
      borderWidth: 2
    }]
  };

  const ws = useRef(null);
  const wsReconnectTimer = useRef(null);
  const wsReconnectAttempts = useRef(0);
  const logsContainerRef = useRef(null);

  useEffect(() => {
    const connectWebSocket = () => {
      // Ensure we don't accumulate reconnect timers
      if (wsReconnectTimer.current) {
        clearTimeout(wsReconnectTimer.current);
        wsReconnectTimer.current = null;
      }

      // Close any previous socket (prevents SYN flood / hanging CONNECTING sockets)
      try {
        if (ws.current && ws.current.readyState !== WebSocket.CLOSED) {
          ws.current.close();
        }
      } catch (_) {
        // ignore
      }

      setStatus('connecting');
      ws.current = new WebSocket(WS_URL);

      ws.current.onopen = () => {
        wsReconnectAttempts.current = 0;
        setStatus('connected');
        addLog('INFO', `Connected to BlueBird Private (${API_HOST}:${API_PORT})`);
      };

      ws.current.onmessage = (event) => {
        const message = JSON.parse(event.data);

        if (message.type === 'update') {
          const updateData = message.data;
          setData(updateData);
          setPrice(updateData.price);
          if (updateData.account) setAccount(updateData.account);
          if (updateData.positions) setPositions(updateData.positions);
          if (updateData.market) setMarket(updateData.market);
          if (updateData.ultra) setUltra(updateData.ultra);
          if (updateData.ai) setAi(updateData.ai);
          if (updateData.grid) setGrid(updateData.grid);
          if (updateData.risk) setRisk(updateData.risk);
          if (updateData.last_trade) setLastTrade(updateData.last_trade);
          // New profitability data
          if (updateData.time_filter || updateData.correlations || updateData.momentum) {
            setSmartFilters({
              time_filter: updateData.time_filter || smartFilters.time_filter,
              correlations: updateData.correlations || {},
              momentum: updateData.momentum || { allow_buy: true, allow_sell: true }
            });
          }
          if (updateData.performance) setPerformance(updateData.performance);
          if (updateData.orders) setOrders(updateData.orders);
          if (updateData.windfall) setWindfall(updateData.windfall);
          if (updateData.health) setHealth(updateData.health);
          if (updateData.risk_overlay) {
            setRiskOverlay(updateData.risk_overlay);
            setRiskOverlayFetch({ status: 'ok', lastFetchedAt: Date.now(), error: null });
          }

          // Track per-symbol prices from positions and grid data
          if (updateData.positions || updateData.grid?.summaries) {
            setSymbolPrices(prev => {
              const newPrices = { ...prev };
              const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });

              // Get prices from positions
              updateData.positions?.forEach(pos => {
                const sym = pos.symbol?.replace('USD', '/USD') || pos.symbol;
                if (newPrices[sym]) {
                  const currentPrice = parseFloat(pos.current_price) || 0;
                  const entryPrice = parseFloat(pos.avg_entry_price) || 0;
                  const oldPrice = newPrices[sym].price || entryPrice;
                  const change = currentPrice - oldPrice;
                  const changePercent = oldPrice > 0 ? (change / oldPrice) * 100 : 0;

                  const history = [...(newPrices[sym].history || []), { time: timestamp, price: currentPrice }];
                  if (history.length > 60) history.shift();

                  newPrices[sym] = { price: currentPrice, change, changePercent, history };
                }
              });

              // Also update from active symbol price
              const activeSymbol = updateData.symbol || updateData.multi_asset?.active_symbol;
              if (activeSymbol && newPrices[activeSymbol] && updateData.price) {
                const currentPrice = parseFloat(updateData.price);
                const oldPrice = newPrices[activeSymbol].price || currentPrice;
                const change = currentPrice - oldPrice;
                const changePercent = oldPrice > 0 ? (change / oldPrice) * 100 : 0;

                const history = [...(newPrices[activeSymbol].history || []), { time: timestamp, price: currentPrice }];
                if (history.length > 60) history.shift();

                newPrices[activeSymbol] = { price: currentPrice, change, changePercent, history };
              }

              return newPrices;
            });
          }

          setLastUpdate(new Date());
          updateChart(updateData.timestamp, updateData.price, updateData.symbol, updateData.positions || []);
        } else if (message.type === 'log') {
          const msg = message.data.message;
          const type = msg.includes('BUY') ? 'BUY' : msg.includes('SELL') ? 'SELL' : 'INFO';
          addLog(type, msg);
        }
      };

      const scheduleReconnect = (reason) => {
        // exponential backoff, capped
        wsReconnectAttempts.current += 1;
        const base = Math.min(30000, 1000 * (2 ** Math.min(wsReconnectAttempts.current, 5)));
        const jitter = Math.floor(Math.random() * 250);
        const delay = base + jitter;
        setStatus('disconnected');
        addLog('WARN', `Connection lost (${reason}). Reconnecting in ${Math.round(delay / 1000)}s...`);
        wsReconnectTimer.current = setTimeout(connectWebSocket, delay);
      };

      ws.current.onclose = () => scheduleReconnect('closed');
      ws.current.onerror = () => scheduleReconnect('error');
    };

    connectWebSocket();

    return () => {
      if (wsReconnectTimer.current) clearTimeout(wsReconnectTimer.current);
      try {
        if (ws.current) ws.current.close();
      } catch (_) {
        // ignore
      }
    };
  }, []);

  useEffect(() => {
    // Scroll within the logs container only, not the whole page
    if (logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  }, [logs]);

  // Auto-refresh countdown timer (UI only)
  useEffect(() => {
    const timer = setInterval(() => {
      if (lastUpdate) {
        const elapsed = Math.floor((Date.now() - lastUpdate.getTime()) / 1000);
        setSecondsSinceUpdate(elapsed);
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [lastUpdate]);

  const addLog = (type, msg) => {
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
    setLogs(prev => [...prev.slice(-49), { timestamp, type, message: msg }]);
  };

  const updateChart = (timestamp, price, symbol, allPositions = []) => {
    const timeLabel = new Date(timestamp).toLocaleTimeString('en-US', { hour12: false });

    setSymbolChartData(prev => {
      const updated = { ...prev };

      // Update active symbol from stream
      const activeSymbol = symbol || 'BTC/USD';
      if (updated[activeSymbol] && price) {
        const newLabels = [...updated[activeSymbol].labels, timeLabel];
        const newData = [...updated[activeSymbol].data, price];
        if (newLabels.length > 60) {
          newLabels.shift();
          newData.shift();
        }
        updated[activeSymbol] = { labels: newLabels, data: newData };
      }

      // Also update from positions data (for symbols we hold)
      allPositions.forEach(pos => {
        const sym = pos.symbol?.replace('USD', '/USD') || pos.symbol;
        const posPrice = parseFloat(pos.current_price);
        if (updated[sym] && posPrice && sym !== activeSymbol) {
          const newLabels = [...updated[sym].labels, timeLabel];
          const newData = [...updated[sym].data, posPrice];
          if (newLabels.length > 60) {
            newLabels.shift();
            newData.shift();
          }
          updated[sym] = { labels: newLabels, data: newData };
        }
      });

      return updated;
    });
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: '#151d2e',
        titleColor: '#d4af37',
        bodyColor: '#f5f0e6',
        borderColor: 'rgba(212, 175, 55, 0.2)',
        borderWidth: 1,
        titleFont: { family: 'DM Sans', weight: 600 },
        bodyFont: { family: 'JetBrains Mono' },
        padding: 12,
        cornerRadius: 8
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.03)', drawBorder: false },
        ticks: { color: '#6b7a94', font: { family: 'DM Sans', size: 10 }, maxTicksLimit: 8 }
      },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.03)', drawBorder: false },
        ticks: { color: '#6b7a94', font: { family: 'JetBrains Mono', size: 10 } }
      }
    }
  };

  const getSignalClass = (signal) => {
    if (signal === 'BUY') return 'signal signal-buy';
    if (signal === 'SELL') return 'signal signal-sell';
    return 'signal signal-hold';
  };

  // Fetch all notifier data (status, history, config)
  const fetchNotifierData = async () => {
    try {
      const [statusRes, historyRes, configRes, watchdogRes] = await Promise.all([
        fetch(`${API_BASE}/api/notifier/status`),
        fetch(`${API_BASE}/api/notifier/history`),
        fetch(`${API_BASE}/api/notifier/config`),
        fetch(`${API_BASE}/api/watchdog/status`)
      ]);

      const updates = {};

      if (statusRes.ok) {
        const status = await statusRes.json();
        Object.assign(updates, status);
      }

      if (historyRes.ok) {
        const historyData = await historyRes.json();
        updates.history = historyData.history || [];
      }

      if (configRes.ok) {
        const configData = await configRes.json();
        updates.config = configData;
      }

      setNotifier(prev => ({ ...prev, ...updates }));

      // Update watchdog state
      if (watchdogRes.ok) {
        const watchdogData = await watchdogRes.json();
        setWatchdog({
          enabled: watchdogData.watchdog?.enabled ?? true,
          status: watchdogData.watchdog?.status || 'unknown',
          last_check: watchdogData.watchdog?.last_check,
          last_restart: watchdogData.watchdog?.last_restart,
          restart_count: watchdogData.watchdog?.restart_count || 0,
          notifier_health: watchdogData.notifier_health || { healthy: true, reason: 'OK' }
        });
      }
    } catch (error) {
      console.log('Could not fetch notifier data');
    }
  };

  // Fetch stream health from /health endpoint (independent of WebSocket)
  const fetchStreamHealth = async () => {
    try {
      const response = await fetch(`${API_BASE}/health`);
      if (response.ok) {
        const data = await response.json();
        if (data.stream_health) {
          setStreamHealth(data.stream_health);
        }
      }
    } catch (error) {
      setStreamHealth({ status: 'error', secondsSinceBar: 0 });
    }
  };

  // Fetch risk overlay status
  const fetchRiskOverlay = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/risk/overlay`);
      if (response.ok) {
        const data = await response.json();
        setRiskOverlay(data);
        setRiskOverlayFetch({ status: 'ok', lastFetchedAt: Date.now(), error: null });
      } else {
        setRiskOverlayFetch(prev => ({
          status: 'error',
          lastFetchedAt: prev.lastFetchedAt,
          error: `HTTP ${response.status}`
        }));
      }
    } catch (error) {
      // Silent fail - overlay shows "unavailable" state
      console.log('Could not fetch risk overlay status');
      setRiskOverlayFetch(prev => ({
        status: 'error',
        lastFetchedAt: prev.lastFetchedAt,
        error: error?.message || 'fetch_failed'
      }));
    }
  };

  // Set risk overlay mode (manual override)
  const setRiskMode = async (mode, reason) => {
    try {
      const response = await fetch(`${API_BASE}/api/risk/overlay`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode, reason })
      });
      if (response.ok) {
        addLog('INFO', `Risk override: ${mode}`);
        await fetchRiskOverlay();
      } else {
        addLog('WARN', 'Failed to set risk mode');
      }
    } catch (error) {
      addLog('WARN', `Override failed: ${error.message}`);
    }
  };

  // Toggle watchdog on/off
  const toggleWatchdog = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/watchdog/toggle`, { method: 'POST' });
      if (response.ok) {
        const result = await response.json();
        setWatchdog(prev => ({ ...prev, enabled: result.enabled }));
      }
    } catch (error) {
      console.error('Failed to toggle watchdog:', error);
    }
  };

  // Manual restart via watchdog
  const manualRestartNotifier = async () => {
    setNotifierLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/watchdog/restart-notifier`, { method: 'POST' });
      if (response.ok) {
        setTimeout(async () => {
          await fetchNotifierData();
          setNotifierLoading(false);
        }, 2000);
      } else {
        setNotifierLoading(false);
      }
    } catch (error) {
      console.error('Failed to restart notifier:', error);
      setNotifierLoading(false);
    }
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // CIRCUIT BREAKER FUNCTIONS - Emergency Trading Halt Controls
  // ═══════════════════════════════════════════════════════════════════════════

  // Fetch circuit breaker status from persistent file
  const fetchCircuitBreakerStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/risk/status`);
      if (response.ok) {
        const data = await response.json();
        const fileData = data.from_file || {};
        const apiData = data.from_api || {};
        const anyTriggered = fileData.max_drawdown_hit ||
                            fileData.daily_limit_hit ||
                            Object.keys(fileData.stop_losses_triggered || {}).length > 0;
        setCircuitBreaker({
          from_file: fileData,
          from_api: apiData,
          loading: false,
          lastFetch: new Date().toISOString(),
          anyTriggered
        });

        // Also update risk state with API data (for when WebSocket isn't connected)
        if (apiData.daily_pnl !== undefined) {
          setRisk(prev => ({
            ...prev,
            daily_pnl: apiData.daily_pnl || 0,
            daily_pnl_pct: apiData.daily_pnl_pct || 0,
            alltime_pnl: apiData.alltime_pnl || 0,
            alltime_pnl_pct: apiData.alltime_pnl_pct || 0,
            grid_pnl: apiData.grid_pnl || 0,
            grid_pnl_pct: apiData.grid_pnl_pct || 0,
            drawdown_pct: apiData.drawdown_pct || 0,
            peak_equity: apiData.peak_equity || 0,
            daily_limit_hit: apiData.daily_limit_hit || false,
            max_drawdown_hit: apiData.max_drawdown_hit || false,
            trading_halted: apiData.trading_halted || false
          }));
        }
      }

      // Also fetch positions from API (fallback when WebSocket isn't connected)
      const posResponse = await fetch(`${API_BASE}/api/positions`);
      if (posResponse.ok) {
        const posData = await posResponse.json();
        if (posData.positions && posData.positions.length > 0) {
          setPositions(posData.positions);
        }
      }
    } catch (error) {
      console.log('Could not fetch circuit breaker status');
    }
  };

  // Reset circuit breakers (requires confirmation)
  const resetCircuitBreaker = async (resetType = 'all') => {
    if (!window.confirm(`Are you sure you want to reset circuit breakers (${resetType})?\n\nThis will allow trading to resume. Make sure the underlying issue is resolved!`)) {
      return;
    }

    setCircuitBreakerResetting(true);
    try {
      const response = await fetch(`${API_BASE}/api/risk/reset?reset_type=${resetType}`, {
        method: 'POST'
      });
      if (response.ok) {
        const result = await response.json();
        console.log('Circuit breaker reset:', result);
        // Refresh status after reset
        setTimeout(() => {
          fetchCircuitBreakerStatus();
          setCircuitBreakerResetting(false);
        }, 500);
      } else {
        setCircuitBreakerResetting(false);
      }
    } catch (error) {
      console.error('Failed to reset circuit breaker:', error);
      setCircuitBreakerResetting(false);
    }
  };

  // Fetch circuit breaker status periodically
  useEffect(() => {
    fetchCircuitBreakerStatus();
    const interval = setInterval(fetchCircuitBreakerStatus, 15000); // Every 15 seconds
    return () => clearInterval(interval);
  }, []);

  // Fetch notifier status periodically
  useEffect(() => {
    fetchNotifierData();
    const interval = setInterval(fetchNotifierData, 10000); // Every 10 seconds
    return () => clearInterval(interval);
  }, []);

  // Fetch stream health periodically (detects Alpaca stream issues)
  useEffect(() => {
    fetchStreamHealth();
    const interval = setInterval(fetchStreamHealth, 10000); // Every 10 seconds
    return () => clearInterval(interval);
  }, []);

  // Fetch risk overlay status periodically
  useEffect(() => {
    fetchRiskOverlay();
    const interval = setInterval(fetchRiskOverlay, 15000); // Every 15 seconds
    return () => clearInterval(interval);
  }, []);

  // Toggle notifier on/off
  const toggleNotifier = async () => {
    setNotifierLoading(true);
    try {
      const endpoint = notifier.running ? '/api/notifier/stop' : '/api/notifier/start';
      const response = await fetch(`${API_BASE}${endpoint}`, { method: 'POST' });
      if (response.ok) {
        // Refetch all data after toggle
        setTimeout(async () => {
          await fetchNotifierData();
          setNotifierLoading(false);
        }, 1500);
      } else {
        setNotifierLoading(false);
      }
    } catch (error) {
      console.error('Failed to toggle notifier:', error);
      setNotifierLoading(false);
    }
  };

  // Send test SMS
  const sendTestSms = async () => {
    setNotifierLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/notifier/test`, { method: 'POST' });
      const result = await response.json();
      if (result.status === 'sent') {
        addLog('INFO', 'Test SMS sent successfully');
      } else {
        addLog('WARN', `SMS test: ${result.message || result.status}`);
      }
      // Refetch all data to update count and history
      await fetchNotifierData();
    } catch (error) {
      addLog('WARN', 'Failed to send test SMS');
    }
    setNotifierLoading(false);
  };

  // Update notifier config setting
  const updateNotifierSetting = async (key, value) => {
    try {
      const response = await fetch(`${API_BASE}/api/notifier/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key, value })
      });
      if (response.ok) {
        const result = await response.json();
        if (result.config) {
          setNotifier(prev => ({ ...prev, config: result.config }));
        }
      }
    } catch (error) {
      console.error('Failed to update setting:', error);
    }
  };

  // Format time for display
  const formatNotifierTime = (isoString) => {
    if (!isoString) return 'Never';
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true });
  };

  // Get SMS type icon
  const getSmsTypeIcon = (type) => {
    const icons = {
      trade: '📈',
      alert: '⚠️',
      summary: '📊',
      test: '🧪',
      startup: '🚀'
    };
    return icons[type] || '📱';
  };

  // Get SMS type color class
  const getSmsTypeClass = (type) => {
    const classes = {
      trade: 'sms-trade',
      alert: 'sms-alert',
      summary: 'sms-summary',
      test: 'sms-test',
      startup: 'sms-startup'
    };
    return classes[type] || '';
  };

  return (
    <div className="luxury-bg">
      <div className="relative z-10 p-6 min-h-screen">
        {/* ═══════════════════════════════════════════════════════════════════
            HEADER
        ═══════════════════════════════════════════════════════════════════ */}
        <header className="mb-6">
          <div className="flex items-center justify-between">
            {/* Logo & Title */}
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-[#d4af37] to-[#a08a3c] flex items-center justify-center shadow-lg glow-gold">
                <span className="text-[#080c14] font-display text-xl font-bold">B</span>
              </div>
              <div>
                <h1 className="font-display text-2xl text-[#f5f0e6] tracking-tight">
                  BlueBird <span className="text-gold">Private</span>
                </h1>
                <p className="text-muted text-sm">Intelligent Grid Trading</p>
              </div>
            </div>

            {/* Navigation */}
            <div className="flex items-center gap-3">
              <button
                onClick={() => setCurrentView('trading')}
                className={`btn btn-secondary ${currentView === 'trading' ? 'active' : ''}`}
              >
                Trading
              </button>
              <button
                onClick={() => setCurrentView('training')}
                className={`btn btn-secondary ${currentView === 'training' ? 'active' : ''}`}
              >
                Training
              </button>
              <button
                onClick={() => setCurrentView('history')}
                className={`btn btn-secondary ${currentView === 'history' ? 'active' : ''}`}
              >
                History
              </button>
            </div>

            {/* Status */}
            {currentView === 'trading' && (
              <div className="flex items-center gap-4">
                {ai.warmup && !ai.warmup.all_ready && (
                  <div className="badge badge-gold">
                    Warming Up · {ai.warmup?.avg_bars || 0}/50
                  </div>
                )}
                <div className={`badge ${ai.confidence >= 70 ? 'badge-success' : ai.confidence >= 50 ? 'badge-gold' : 'badge-neutral'}`}>
                  AI Confidence · {ai.confidence}%
                </div>
                <div className={`badge ${status === 'connected' ? 'badge-success' : 'badge-danger'}`}>
                  <div className={`status-dot ${status === 'connected' ? 'success pulse' : 'danger'}`} />
                  {status === 'connected' ? 'Online' : 'Offline'}
                </div>
                {/* Live Update Timer */}
                <div className={`badge ${secondsSinceUpdate <= 2 ? 'badge-success' : secondsSinceUpdate <= 5 ? 'badge-gold' : 'badge-danger'}`}>
                  <svg className={`w-3 h-3 mr-1.5 ${secondsSinceUpdate <= 2 ? 'animate-spin' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  {secondsSinceUpdate <= 2 ? 'Live' : `${secondsSinceUpdate}s ago`}
                </div>

                {/* Stream Health Badge - detects Alpaca stream issues */}
                <div className={`badge ${
                  streamHealth.status === 'healthy' ? 'badge-success' :
                  streamHealth.status === 'degraded' ? 'badge-gold' :
                  'badge-danger'
                }`} title={`Alpaca stream: ${streamHealth.status} (${streamHealth.secondsSinceBar}s since last bar)`}>
                  <div className={`status-dot ${
                    streamHealth.status === 'healthy' ? 'success pulse' :
                    streamHealth.status === 'degraded' ? 'warning' :
                    'danger'
                  }`} />
                  <span className="font-mono text-xs">
                    {streamHealth.status === 'healthy' ? 'Stream' :
                     streamHealth.status === 'degraded' ? `Stream ${streamHealth.secondsSinceBar}s` :
                     streamHealth.status === 'stale' ? 'Stream DOWN' :
                     'Stream ?'}
                  </span>
                </div>

                {/* SMS Notifier Badge */}
                <div className="notifier-header-control">
                  <button
                    className={`notifier-badge ${notifier.running ? 'active' : 'inactive'}`}
                    onClick={() => { setShowNotifierPanel(!showNotifierPanel); setNotifierTab('status'); }}
                  >
                    <span className="notifier-icon">📱</span>
                    <span className="notifier-status">
                      {notifier.running ? 'SMS On' : 'SMS Off'}
                    </span>
                    <span className={`notifier-dot ${notifier.running ? 'active' : ''}`} />
                  </button>

                  {/* Expanded Dropdown Panel with Tabs */}
                  {showNotifierPanel && (
                    <div className="notifier-dropdown expanded">
                      {/* Header */}
                      <div className="notifier-dropdown-header">
                        <div className="notifier-header-left">
                          <span className="notifier-title">SMS Notifications</span>
                          <span className={`notifier-status-badge ${notifier.running ? 'running' : 'stopped'}`}>
                            <span className="status-indicator" />
                            {notifier.running ? 'Running' : 'Stopped'}
                          </span>
                        </div>
                        <span className="sms-count">{notifier.sms_count_today} today</span>
                      </div>

                      {/* Tab Navigation */}
                      <div className="notifier-tabs">
                        <button
                          className={`notifier-tab ${notifierTab === 'status' ? 'active' : ''}`}
                          onClick={() => setNotifierTab('status')}
                        >
                          Status
                        </button>
                        <button
                          className={`notifier-tab ${notifierTab === 'history' ? 'active' : ''}`}
                          onClick={() => setNotifierTab('history')}
                        >
                          History
                        </button>
                        <button
                          className={`notifier-tab ${notifierTab === 'settings' ? 'active' : ''}`}
                          onClick={() => setNotifierTab('settings')}
                        >
                          Settings
                        </button>
                        <button
                          className={`notifier-tab watchdog-tab ${notifierTab === 'watchdog' ? 'active' : ''} ${watchdog.enabled ? 'guarding' : ''}`}
                          onClick={() => setNotifierTab('watchdog')}
                        >
                          <span className="watchdog-tab-icon">🛡️</span>
                        </button>
                      </div>

                      {/* Tab Content */}
                      <div className="notifier-tab-content">
                        {/* Status Tab */}
                        {notifierTab === 'status' && (
                          <div className="notifier-status-tab">
                            <div className="status-hero">
                              <div className={`status-circle ${notifier.running ? 'active' : ''}`}>
                                <span className="status-icon">{notifier.running ? '📡' : '📴'}</span>
                              </div>
                              <div className="status-details">
                                <div className="status-label">{notifier.running ? 'Service Active' : 'Service Stopped'}</div>
                                {notifier.running && notifier.uptime && (
                                  <div className="status-uptime">Uptime: {notifier.uptime}</div>
                                )}
                              </div>
                            </div>

                            <div className="status-metrics">
                              <div className="status-metric">
                                <span className="metric-icon">📨</span>
                                <span className="metric-value">{notifier.sms_count_today}</span>
                                <span className="metric-label">Sent Today</span>
                              </div>
                              <div className="status-metric">
                                <span className="metric-icon">🕐</span>
                                <span className="metric-value">{formatNotifierTime(notifier.last_sms_sent)}</span>
                                <span className="metric-label">Last SMS</span>
                              </div>
                            </div>

                            {notifier.quiet_hours_active && (
                              <div className="quiet-hours-banner">
                                <span className="quiet-icon">🌙</span>
                                <span>Quiet Hours Active</span>
                                <span className="quiet-time">Until {notifier.config?.quiet_hours_end || 7}:00 AM</span>
                              </div>
                            )}
                          </div>
                        )}

                        {/* History Tab */}
                        {notifierTab === 'history' && (
                          <div className="notifier-history-tab">
                            {notifier.history && notifier.history.length > 0 ? (
                              <div className="sms-history-list">
                                {notifier.history.slice(0, 10).map((sms, index) => (
                                  <div key={sms.id || index} className={`sms-history-item ${getSmsTypeClass(sms.type)}`}>
                                    <div className="sms-icon">{getSmsTypeIcon(sms.type)}</div>
                                    <div className="sms-content">
                                      <div className="sms-preview">{sms.preview}</div>
                                      <div className="sms-meta">
                                        <span className="sms-time">{formatNotifierTime(sms.timestamp)}</span>
                                        <span className={`sms-status ${sms.status}`}>{sms.status}</span>
                                      </div>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="empty-history">
                                <span className="empty-icon">📭</span>
                                <span>No SMS history yet</span>
                              </div>
                            )}
                          </div>
                        )}

                        {/* Settings Tab */}
                        {notifierTab === 'settings' && (
                          <div className="notifier-settings-tab">
                            <div className="settings-group">
                              <div className="setting-item">
                                <div className="setting-info">
                                  <span className="setting-icon">🌙</span>
                                  <div className="setting-text">
                                    <span className="setting-label">Quiet Hours</span>
                                    <span className="setting-desc">{notifier.config?.quiet_hours_start || 23}:00 - {notifier.config?.quiet_hours_end || 7}:00</span>
                                  </div>
                                </div>
                                <label className="toggle-switch">
                                  <input
                                    type="checkbox"
                                    checked={notifier.config?.quiet_hours_enabled ?? true}
                                    onChange={(e) => updateNotifierSetting('quiet_hours_enabled', e.target.checked)}
                                  />
                                  <span className="toggle-slider" />
                                </label>
                              </div>

                              <div className="setting-item">
                                <div className="setting-info">
                                  <span className="setting-icon">📈</span>
                                  <div className="setting-text">
                                    <span className="setting-label">Trade Alerts</span>
                                    <span className="setting-desc">Notify on trade execution</span>
                                  </div>
                                </div>
                                <label className="toggle-switch">
                                  <input
                                    type="checkbox"
                                    checked={notifier.config?.notify_on_trade ?? true}
                                    onChange={(e) => updateNotifierSetting('notify_on_trade', e.target.checked)}
                                  />
                                  <span className="toggle-slider" />
                                </label>
                              </div>

                              <div className="setting-item">
                                <div className="setting-info">
                                  <span className="setting-icon">⚠️</span>
                                  <div className="setting-text">
                                    <span className="setting-label">Risk Alerts</span>
                                    <span className="setting-desc">Notify on risk events</span>
                                  </div>
                                </div>
                                <label className="toggle-switch">
                                  <input
                                    type="checkbox"
                                    checked={notifier.config?.notify_on_alert ?? true}
                                    onChange={(e) => updateNotifierSetting('notify_on_alert', e.target.checked)}
                                  />
                                  <span className="toggle-slider" />
                                </label>
                              </div>

                              <div className="setting-item">
                                <div className="setting-info">
                                  <span className="setting-icon">📊</span>
                                  <div className="setting-text">
                                    <span className="setting-label">Daily Summary</span>
                                    <span className="setting-desc">8 PM daily report</span>
                                  </div>
                                </div>
                                <label className="toggle-switch">
                                  <input
                                    type="checkbox"
                                    checked={notifier.config?.notify_daily_summary ?? true}
                                    onChange={(e) => updateNotifierSetting('notify_daily_summary', e.target.checked)}
                                  />
                                  <span className="toggle-slider" />
                                </label>
                              </div>
                            </div>

                            <div className="phone-display">
                              <span className="phone-icon">📱</span>
                              <span className="phone-number">{notifier.config?.phone_number || '+1••••••••••'}</span>
                            </div>
                          </div>
                        )}

                        {/* Watchdog Tab */}
                        {notifierTab === 'watchdog' && (
                          <div className="watchdog-tab">
                            {/* Guardian Shield Visual */}
                            <div className="watchdog-hero">
                              <div className={`shield-container ${watchdog.enabled ? 'active' : 'inactive'}`}>
                                <div className="shield-icon">🛡️</div>
                                <div className={`shield-pulse ${watchdog.enabled ? 'pulsing' : ''}`} />
                                <div className={`shield-ring ${watchdog.enabled ? 'rotating' : ''}`} />
                              </div>
                              <div className="watchdog-title">
                                {watchdog.enabled ? 'Guardian Active' : 'Guardian Disabled'}
                              </div>
                              <div className={`watchdog-status-text ${watchdog.notifier_health?.healthy ? 'healthy' : 'alert'}`}>
                                {watchdog.notifier_health?.healthy ? 'All Systems Nominal' : watchdog.notifier_health?.reason}
                              </div>
                            </div>

                            {/* Health Metrics */}
                            <div className="watchdog-metrics">
                              <div className={`watchdog-metric ${watchdog.notifier_health?.healthy ? 'healthy' : 'alert'}`}>
                                <span className="metric-icon">{watchdog.notifier_health?.healthy ? '✓' : '⚠'}</span>
                                <span className="metric-label">Health</span>
                                <span className="metric-value">{watchdog.notifier_health?.healthy ? 'OK' : 'Issue'}</span>
                              </div>
                              <div className="watchdog-metric">
                                <span className="metric-icon">🔄</span>
                                <span className="metric-label">Restarts</span>
                                <span className="metric-value">{watchdog.restart_count}</span>
                              </div>
                              <div className="watchdog-metric">
                                <span className="metric-icon">⏱️</span>
                                <span className="metric-label">Since SMS</span>
                                <span className="metric-value">{Math.round(watchdog.notifier_health?.minutes_since_last_sms || 0)}m</span>
                              </div>
                            </div>

                            {/* Last Events */}
                            <div className="watchdog-events">
                              <div className="event-item">
                                <span className="event-label">Last Check</span>
                                <span className="event-value">{formatNotifierTime(watchdog.last_check)}</span>
                              </div>
                              {watchdog.last_restart && (
                                <div className="event-item alert">
                                  <span className="event-label">Last Restart</span>
                                  <span className="event-value">{formatNotifierTime(watchdog.last_restart)}</span>
                                </div>
                              )}
                            </div>

                            {/* Controls */}
                            <div className="watchdog-controls">
                              <div className="watchdog-toggle-row">
                                <span className="toggle-label">Auto-Recovery</span>
                                <label className="toggle-switch">
                                  <input
                                    type="checkbox"
                                    checked={watchdog.enabled}
                                    onChange={toggleWatchdog}
                                  />
                                  <span className="toggle-slider" />
                                </label>
                              </div>
                              <button
                                className="btn-watchdog-restart"
                                onClick={manualRestartNotifier}
                                disabled={notifierLoading}
                              >
                                {notifierLoading ? 'Restarting...' : 'Force Restart'}
                              </button>
                            </div>
                          </div>
                        )}
                      </div>

                      {/* Action Buttons */}
                      <div className="notifier-actions">
                        <button
                          onClick={toggleNotifier}
                          disabled={notifierLoading}
                          className={`btn-notifier-toggle ${notifier.running ? 'stop' : 'start'}`}
                        >
                          {notifierLoading ? '...' : notifier.running ? 'Stop Service' : 'Start Service'}
                        </button>
                        <button
                          onClick={sendTestSms}
                          disabled={notifierLoading}
                          className="btn-notifier-test"
                        >
                          Send Test
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </header>

        {currentView === 'training' ? (
          <TrainingDashboard />
        ) : currentView === 'history' ? (
          <HistoryDashboard />
        ) : (
          <>
            {/* ═══════════════════════════════════════════════════════════════════
                RISK OVERLAY BANNER - Command Center Alert System
            ═══════════════════════════════════════════════════════════════════ */}
            {(() => {
              const staleMs = Math.max(15000, (autoRefreshInterval || 5) * 1000 * 3);
              const isStale = riskOverlayFetch.lastFetchedAt != null
                ? (Date.now() - riskOverlayFetch.lastFetchedAt) > staleMs
                : true;
              const overlayOk = riskOverlayFetch.status === 'ok' && !isStale;
              const displayMode = overlayOk ? (riskOverlay.mode || 'NORMAL') : 'UNKNOWN';
              const stageIdx0 = Number.isFinite(riskOverlay?.recovery?.stage) ? riskOverlay.recovery.stage : null; // 0-indexed from backend
              const displayStage = (overlayOk && displayMode === 'RECOVERY' && stageIdx0 != null) ? (stageIdx0 + 1) : null; // human-friendly 1..4

              return (
            <div
              className={`risk-banner ${overlayOk ? `risk-banner-${(displayMode || 'NORMAL').toLowerCase()}` : 'risk-banner-unavailable'}`}
              onClick={() => setRiskOverlayExpanded(!riskOverlayExpanded)}
            >
              <div className="risk-banner-content">
                {/* Mode Badge */}
                <div className="risk-mode-badge">
                  <span className="risk-mode-icon">
                    {displayMode === 'RISK_OFF' ? '⚠️' : displayMode === 'RECOVERY' ? '🔄' : displayMode === 'NORMAL' ? '🛡️' : '❓'}
                  </span>
                  <span className="risk-mode-text">{displayMode}</span>
                </div>

                {/* Duration */}
                <div className="risk-duration">
                  <span className="risk-duration-label">Since</span>
                  <span className="risk-duration-value">
                    {overlayOk && riskOverlay.mode_duration_minutes != null
                      ? riskOverlay.mode_duration_minutes < 60
                        ? `${Math.round(riskOverlay.mode_duration_minutes)}m`
                        : `${(riskOverlay.mode_duration_minutes / 60).toFixed(1)}h`
                      : '--'}
                  </span>
                </div>

                {/* Status Text */}
                <div className="risk-status-text">
                  {!overlayOk && <span className="risk-unavailable-text">API disconnected</span>}
                  {overlayOk && displayMode === 'RISK_OFF' && 'Buys BLOCKED'}
                  {overlayOk && displayMode === 'RECOVERY' && `Size: ${(riskOverlay.position_multiplier || 1).toFixed(2)}x`}
                  {overlayOk && displayMode === 'NORMAL' && 'Full trading enabled'}
                </div>

                {/* Telemetry Summary (shown when not NORMAL) */}
                {overlayOk && displayMode !== 'NORMAL' && riskOverlay.telemetry?.avoided_buys_notional > 0 && (
                  <div className="risk-protected-badge">
                    <span className="risk-protected-icon">💰</span>
                    <span className="risk-protected-value">
                      ${(riskOverlay.telemetry.avoided_buys_notional + (riskOverlay.telemetry.cancelled_limits_notional || 0)).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                    </span>
                    <span className="risk-protected-label">protected</span>
                  </div>
                )}

                {/* Manual Override Indicator */}
                {riskOverlay.manual_override && (
                  <div className="risk-override-badge">
                    <span className="risk-override-icon">👤</span>
                    <span className="risk-override-text">Manual</span>
                  </div>
                )}

                {/* Expand/Collapse Chevron */}
                <div className="risk-expand-chevron">
                  <svg
                    className={`risk-chevron-icon ${riskOverlayExpanded ? 'expanded' : ''}`}
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <polyline points="6 9 12 15 18 9"></polyline>
                  </svg>
                </div>
              </div>

              {/* Trigger Reasons (shown inline when RISK_OFF) */}
              {overlayOk && displayMode === 'RISK_OFF' && riskOverlay.trigger_reasons?.length > 0 && (
                <div className="risk-triggers-inline">
                  <span className="risk-triggers-label">Triggers:</span>
                  {riskOverlay.trigger_reasons.slice(0, 3).map((reason, idx) => (
                    <span key={idx} className="risk-trigger-chip">{reason}</span>
                  ))}
                </div>
              )}

              {/* Recovery Progress (shown inline when RECOVERY) */}
              {overlayOk && displayMode === 'RECOVERY' && riskOverlay.recovery && displayStage != null && (
                <div className="risk-recovery-inline">
                  <span className="risk-recovery-stage">
                    Stage {displayStage}/4
                  </span>
                  <div className="risk-recovery-bar-mini">
                    <div
                      className="risk-recovery-fill-mini"
                      style={{ width: `${Math.min((riskOverlay.recovery.bars_in_stage / 10) * 100, 100)}%` }}
                    />
                  </div>
                  <span className="risk-recovery-bars">
                    {riskOverlay.recovery.bars_in_stage}/10 bars
                  </span>
                </div>
              )}
            </div>
              );
            })()}

            {/* ═══════════════════════════════════════════════════════════════════
                RISK OVERLAY EXPANDED PANEL
            ═══════════════════════════════════════════════════════════════════ */}
            {riskOverlayExpanded && (
              <div className="risk-expanded-panel">
                <div className="risk-panel-grid">
                  {/* Triggers Section - Visual Threshold Gauges */}
                  <div className="risk-panel-section risk-triggers">
                    <div className="risk-section-header">
                      <span className="risk-section-icon">⚡</span>
                      <span className="risk-section-title">Trigger Levels</span>
                    </div>
                    <div className="risk-triggers-gauges">
                      {/* Momentum Gauge - negative values, threshold at -1.5% */}
                      <TriggerGauge
                        label="Momentum"
                        value={riskOverlay.current_signals?.momentum || 0}
                        threshold={riskOverlay.thresholds?.momentum || -0.015}
                        format={(v) => `${(v * 100).toFixed(2)}%`}
                        invert={true}
                      />

                      {/* Correlation Gauge - 0 to 1, threshold at 0.90 */}
                      <TriggerGauge
                        label="Correlation"
                        value={riskOverlay.current_signals?.correlation || 0}
                        threshold={riskOverlay.thresholds?.correlation || 0.90}
                        max={1.0}
                        format={(v) => v.toFixed(2)}
                      />

                      {/* ADX Gauge - 0 to 100, threshold at 35 */}
                      <TriggerGauge
                        label="ADX"
                        value={riskOverlay.current_signals?.adx || 0}
                        threshold={riskOverlay.thresholds?.adx || 35}
                        max={60}
                        format={(v) => v.toFixed(0)}
                        direction={riskOverlay.current_signals?.adx_direction}
                      />

                      {/* Active trigger reasons as chips below gauges */}
                      {riskOverlay.trigger_reasons?.length > 0 && (
                        <div className="risk-active-triggers">
                          <span className="risk-active-label">Active:</span>
                          {riskOverlay.trigger_reasons.map((reason, idx) => (
                            <span key={idx} className="risk-trigger-chip-small">{reason}</span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Telemetry Section */}
                  <div className="risk-panel-section risk-telemetry">
                    <div className="risk-section-header">
                      <span className="risk-section-icon">📊</span>
                      <span className="risk-section-title">Protection Stats</span>
                    </div>
                    <div className="risk-telemetry-grid">
                      <div className="risk-telemetry-item">
                        <span className="risk-telemetry-value">
                          ${(riskOverlay.telemetry?.avoided_buys_notional || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </span>
                        <span className="risk-telemetry-label">Buys Avoided</span>
                        <span className="risk-telemetry-count">{riskOverlay.telemetry?.avoided_buys_count || 0} orders</span>
                      </div>
                      <div className="risk-telemetry-item">
                        <span className="risk-telemetry-value">
                          ${(riskOverlay.telemetry?.cancelled_limits_notional || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </span>
                        <span className="risk-telemetry-label">Limits Cancelled</span>
                        <span className="risk-telemetry-count">{riskOverlay.telemetry?.cancelled_limits_count || 0} orders</span>
                      </div>
                      <div className="risk-telemetry-item">
                        <span className="risk-telemetry-value">{riskOverlay.telemetry?.rebalances_blocked_count || 0}</span>
                        <span className="risk-telemetry-label">Rebalances Blocked</span>
                      </div>
                    </div>
                  </div>

                  {/* Recovery Progress Section (only in RECOVERY mode) */}
                  {riskOverlay.mode === 'RECOVERY' && riskOverlay.recovery && (
                    <div className="risk-panel-section risk-recovery-section">
                      <div className="risk-section-header">
                        <span className="risk-section-icon">🔄</span>
                        <span className="risk-section-title">Recovery Progress</span>
                      </div>
                      <div className="risk-recovery-content">
                        <div className="risk-recovery-stages">
                          {[1, 2, 3, 4].map((stage) => (
                            <div
                              key={stage}
                              className={`risk-stage ${
                                stage < (riskOverlay.recovery.stage + 1) ? 'completed'
                                : stage === (riskOverlay.recovery.stage + 1) ? 'active'
                                : 'pending'
                              }`}
                            >
                              <span className="risk-stage-number">{stage}</span>
                              <span className="risk-stage-mult">
                                {stage === 1 ? '0.25x' : stage === 2 ? '0.50x' : stage === 3 ? '0.75x' : '1.0x'}
                              </span>
                            </div>
                          ))}
                        </div>
                        <div className="risk-recovery-bar">
                          <div
                            className="risk-recovery-fill"
                            style={{ width: `${Math.min((riskOverlay.recovery.bars_in_stage / 10) * 100, 100)}%` }}
                          />
                        </div>
                        <div className="risk-recovery-info">
                          <span>{riskOverlay.recovery.bars_in_stage}/10 bars in current stage</span>
                          <span>Total: {riskOverlay.recovery.total_bars_in_recovery} bars</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Exposure Section */}
                  <div className="risk-panel-section risk-exposure-section">
                    <div className="risk-section-header">
                      <span className="risk-section-icon">📈</span>
                      <span className="risk-section-title">Exposure Caps</span>
                    </div>
                    <div className="risk-exposure-content">
                      <div className="risk-exposure-row">
                        <span className="risk-exposure-label">Normal Cap</span>
                        <div className="risk-exposure-bar-container">
                          <div
                            className="risk-exposure-bar normal"
                            style={{ width: `${(riskOverlay.caps?.normal_exposure || 0.70) * 100}%` }}
                          />
                        </div>
                        <span className="risk-exposure-value">{((riskOverlay.caps?.normal_exposure || 0.70) * 100).toFixed(0)}%</span>
                      </div>
                      <div className="risk-exposure-row">
                        <span className="risk-exposure-label">Risk-Off Cap</span>
                        <div className="risk-exposure-bar-container">
                          <div
                            className="risk-exposure-bar risk-off"
                            style={{ width: `${(riskOverlay.caps?.risk_off_exposure || 0.40) * 100}%` }}
                          />
                        </div>
                        <span className="risk-exposure-value">{((riskOverlay.caps?.risk_off_exposure || 0.40) * 100).toFixed(0)}%</span>
                      </div>
                      {riskOverlay.mode !== 'NORMAL' && (
                        <div className="risk-exposure-row active">
                          <span className="risk-exposure-label">Position Mult</span>
                          <div className="risk-exposure-bar-container">
                            <div
                              className="risk-exposure-bar current"
                              style={{ width: `${(riskOverlay.position_multiplier || 1) * 100}%` }}
                            />
                          </div>
                          <span className="risk-exposure-value">{((riskOverlay.position_multiplier || 1) * 100).toFixed(0)}%</span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Manual Override Controls */}
                  <div className="risk-panel-section risk-controls">
                    <div className="risk-section-header">
                      <span className="risk-section-icon">🎛️</span>
                      <span className="risk-section-title">Manual Override</span>
                    </div>
                    <div className="risk-control-buttons">
                      <button
                        className={`risk-control-btn normal ${riskOverlay.mode === 'NORMAL' && !riskOverlay.manual_override ? 'active' : ''}`}
                        onClick={(e) => { e.stopPropagation(); setRiskMode('NORMAL', 'Dashboard manual override'); }}
                      >
                        <span className="risk-btn-icon">🛡️</span>
                        <span className="risk-btn-text">Force NORMAL</span>
                      </button>
                      <button
                        className={`risk-control-btn risk-off ${riskOverlay.mode === 'RISK_OFF' && riskOverlay.manual_override ? 'active' : ''}`}
                        onClick={(e) => { e.stopPropagation(); setRiskMode('RISK_OFF', 'Dashboard manual override'); }}
                      >
                        <span className="risk-btn-icon">⚠️</span>
                        <span className="risk-btn-text">Force RISK_OFF</span>
                      </button>
                      {riskOverlay.manual_override && (
                        <button
                          className="risk-control-btn clear"
                          onClick={(e) => { e.stopPropagation(); setRiskMode('CLEAR', 'Dashboard clear override'); }}
                        >
                          <span className="risk-btn-icon">↩️</span>
                          <span className="risk-btn-text">Clear Override</span>
                        </button>
                      )}
                    </div>
                    {riskOverlay.manual_override && (
                      <div className="risk-override-info">
                        <span className="risk-override-detail">
                          Override: {riskOverlay.manual_override.mode} by {riskOverlay.manual_override.reason}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* ═══════════════════════════════════════════════════════════════════
                HEALTH STATUS PANEL
            ═══════════════════════════════════════════════════════════════════ */}
            <div className="card mb-5" style={{
              background: 'linear-gradient(135deg, rgba(21, 29, 46, 0.95), rgba(8, 12, 20, 0.98))',
              borderColor: 'rgba(212, 175, 55, 0.12)'
            }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '1rem 1.5rem',
                flexWrap: 'wrap',
                gap: '1rem'
              }}>
                {/* Panel Label */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span style={{
                    fontSize: '0.65rem',
                    color: 'var(--gold-primary)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.15em',
                    fontWeight: '600'
                  }}>System Health</span>
                </div>

                {/* Health Indicators */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '2rem', flexWrap: 'wrap' }}>

                  {/* Stream Health */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <span style={{
                      fontSize: '0.65rem',
                      color: 'var(--text-muted)',
                      textTransform: 'uppercase',
                      letterSpacing: '0.1em'
                    }}>Stream</span>
                    <span className={`badge ${
                      (health.stream?.seconds_since_bar || 0) < 90 ? 'badge-success' :
                      (health.stream?.seconds_since_bar || 0) < (health.stream?.stale_threshold_seconds || 180) ? 'badge-gold' : 'badge-danger'
                    }`} style={{ fontSize: '0.7rem' }}>
                      <span className={`status-dot ${
                        (health.stream?.seconds_since_bar || 0) < 90 ? 'success pulse' :
                        (health.stream?.seconds_since_bar || 0) < (health.stream?.stale_threshold_seconds || 180) ? '' : 'danger'
                      }`} style={{ width: '6px', height: '6px' }}></span>
                      {(health.stream?.seconds_since_bar || 0) < 90 ? 'Connected' :
                       (health.stream?.seconds_since_bar || 0) < (health.stream?.stale_threshold_seconds || 180) ?
                         `Degraded (${health.stream?.seconds_since_bar || 0}s)` : 'Stale'}
                    </span>
                    {health.stream?.last_bar_at && (
                      <span style={{
                        fontSize: '0.65rem',
                        color: 'var(--text-muted)',
                        fontFamily: 'var(--font-mono)'
                      }}>
                        Last: {new Date(health.stream.last_bar_at).toLocaleTimeString()}
                      </span>
                    )}
                  </div>

                  {/* Divider */}
                  <div style={{
                    width: '1px',
                    height: '24px',
                    background: 'rgba(212, 175, 55, 0.15)'
                  }}></div>

                  {/* Order Tracking Health */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <span style={{
                      fontSize: '0.65rem',
                      color: 'var(--text-muted)',
                      textTransform: 'uppercase',
                      letterSpacing: '0.1em'
                    }}>Orders</span>
                    <span className={`badge ${
                      (health.orders?.mismatch || 0) === 0 ? 'badge-success' :
                      (health.orders?.mismatch || 0) <= 2 ? 'badge-gold' : 'badge-danger'
                    }`} style={{ fontSize: '0.7rem' }}>
                      <span className={`status-dot ${
                        (health.orders?.mismatch || 0) === 0 ? 'success' :
                        (health.orders?.mismatch || 0) <= 2 ? '' : 'danger'
                      }`} style={{ width: '6px', height: '6px' }}></span>
                      {(health.orders?.mismatch || 0) === 0 ? 'Synced' : `Mismatch: ${health.orders?.mismatch}`}
                    </span>
                    <span style={{
                      fontSize: '0.65rem',
                      color: 'var(--text-secondary)',
                      fontFamily: 'var(--font-mono)'
                    }}>
                      {health.orders?.tracked_open_limits || 0} tracked / {health.orders?.alpaca_open_grid_limits || 0} Alpaca
                    </span>
                    {health.orders?.last_check_at && (
                      <span style={{
                        fontSize: '0.6rem',
                        color: 'var(--text-muted)'
                      }}>
                        ({Math.round((Date.now() - new Date(health.orders.last_check_at).getTime()) / 60000)}m ago)
                      </span>
                    )}
                  </div>

                  {/* Show error if present */}
                  {health.orders?.error && (
                    <>
                      <div style={{
                        width: '1px',
                        height: '24px',
                        background: 'rgba(229, 115, 115, 0.3)'
                      }}></div>
                      <span style={{
                        fontSize: '0.65rem',
                        color: 'var(--danger)',
                        maxWidth: '200px',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap'
                      }} title={health.orders.error}>
                        {health.orders.error}
                      </span>
                    </>
                  )}
                </div>
              </div>
            </div>

            {/* ═══════════════════════════════════════════════════════════════════
                MAIN GRID
            ═══════════════════════════════════════════════════════════════════ */}
            <div className="grid grid-cols-4 gap-5">

              {/* ─────────────────────────────────────────────────────────────────
                  LEFT COLUMN
              ───────────────────────────────────────────────────────────────── */}
              <div className="space-y-5">
                {/* Grid Status */}
                <div className={`card ${grid.active ? 'card-gold' : ''}`}>
                  <div className="card-header">
                    <div className={`status-dot ${grid.active ? 'success pulse' : ''}`} />
                    <span className="card-title">Grid Status</span>
                  </div>
                  <div className="card-body">
                    <div className="grid grid-cols-2 gap-3 mb-4">
                      <div className="metric">
                        <div className="metric-label">Fills</div>
                        <div className="metric-value text-gold font-mono">{gridTotalFills}</div>
                      </div>
                      <div className="metric">
                        <div className="metric-label">Cycles</div>
                        <div className="metric-value text-gold font-mono">{gridTotalCycles}</div>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 gap-3 mb-4">
                      <div className="metric">
                        <div className="metric-label">Profit</div>
                        <div className={`metric-value font-mono ${(grid.total_profit || 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                          ${(grid.total_profit || 0).toFixed(2)}
                        </div>
                      </div>
                    </div>

                    <div className={`text-center py-3 rounded-xl ${grid.active ? 'bg-[rgba(62,207,142,0.08)] border border-[rgba(62,207,142,0.2)]' : 'bg-[rgba(255,255,255,0.02)] border border-[rgba(255,255,255,0.06)]'}`}>
                      <span className={`font-display text-lg ${grid.active ? 'text-success' : 'text-muted'}`}>
                        {grid.active ? 'Active' : 'Standby'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Risk Management */}
                <div className={`card ${risk.trading_halted ? 'border-[rgba(229,115,115,0.3)]' : ''}`}>
                  <div className="card-header">
                    <div className={`status-dot ${risk.trading_halted ? 'danger' : ''}`} style={{ background: '#d4af37' }} />
                    <span className="card-title">Risk Management</span>
                  </div>
                  <div className="card-body space-y-4">
                    {/* All-Time P/L (since Nov 24) */}
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span className="text-muted">All-Time P&L <span className="text-xs opacity-60">(Nov 24)</span></span>
                        <span className={risk.alltime_pnl >= 0 ? 'text-success' : 'text-danger'}>
                          ${risk.alltime_pnl?.toFixed(2) || '0.00'} ({risk.alltime_pnl_pct?.toFixed(2) || '0.00'}%)
                        </span>
                      </div>
                      <div className="progress">
                        <div
                          className={`progress-fill ${risk.alltime_pnl >= 0 ? 'success' : 'danger'}`}
                          style={{ width: `${Math.min(Math.abs(risk.alltime_pnl_pct || 0) * 10, 100)}%` }}
                        />
                      </div>
                    </div>

                    {/* Grid P/L (since Dec 2) */}
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span className="text-muted">Grid P&L <span className="text-xs opacity-60">(Dec 2)</span></span>
                        <span className={risk.grid_pnl >= 0 ? 'text-success' : 'text-danger'}>
                          ${risk.grid_pnl?.toFixed(2) || '0.00'} ({risk.grid_pnl_pct?.toFixed(2) || '0.00'}%)
                        </span>
                      </div>
                      <div className="progress">
                        <div
                          className={`progress-fill ${risk.grid_pnl >= 0 ? 'success' : 'danger'}`}
                          style={{ width: `${Math.min(Math.abs(risk.grid_pnl_pct || 0) * 20, 100)}%` }}
                        />
                      </div>
                    </div>

                    {/* Daily P/L */}
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span className="text-muted">Daily P&L</span>
                        <span className={risk.daily_pnl >= 0 ? 'text-success' : 'text-danger'}>
                          ${risk.daily_pnl?.toFixed(2) || '0.00'} ({risk.daily_pnl_pct?.toFixed(2) || '0.00'}%)
                        </span>
                      </div>
                      <div className="progress">
                        <div
                          className={`progress-fill ${risk.daily_pnl >= 0 ? 'success' : 'danger'}`}
                          style={{ width: `${Math.min(Math.abs(risk.daily_pnl_pct || 0) * 20, 100)}%` }}
                        />
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span className="text-muted">Drawdown</span>
                        <span className={risk.drawdown_pct > 5 ? 'text-danger' : 'text-gold'}>
                          {risk.drawdown_pct?.toFixed(2) || '0.00'}%
                        </span>
                      </div>
                      <div className="progress">
                        <div
                          className={`progress-fill ${risk.drawdown_pct > 5 ? 'danger' : ''}`}
                          style={{ width: `${Math.min((risk.drawdown_pct || 0) * 10, 100)}%` }}
                        />
                      </div>
                    </div>

                    <div className="divider" />

                    <div className="space-y-2">
                      <div className="flex justify-between items-center text-sm">
                        <span className="text-muted">Daily Limit (5%)</span>
                        <span className={risk.daily_limit_hit ? 'text-danger' : 'text-success'}>
                          {risk.daily_limit_hit ? 'Triggered' : 'OK'}
                        </span>
                      </div>
                      <div className="flex justify-between items-center text-sm">
                        <span className="text-muted">Max Drawdown (10%)</span>
                        <span className={risk.max_drawdown_hit ? 'text-danger' : 'text-success'}>
                          {risk.max_drawdown_hit ? 'Triggered' : 'OK'}
                        </span>
                      </div>
                    </div>

                    {/* Circuit Breaker Panel - Persistent Emergency Controls */}
                    {(risk.trading_halted || circuitBreaker.anyTriggered) && (
                      <div className="circuit-breaker-panel">
                        <div className="circuit-breaker-header">
                          <div className="circuit-breaker-icon">
                            <span className="circuit-flash">⚡</span>
                          </div>
                          <div className="circuit-breaker-title">
                            <span className="font-display text-danger">Circuit Breaker Active</span>
                            <span className="text-xs text-muted">Trading halted for safety</span>
                          </div>
                        </div>

                        <div className="circuit-breaker-status">
                          {circuitBreaker.from_file?.max_drawdown_hit && (
                            <div className="circuit-item triggered">
                              <span className="circuit-label">Max Drawdown</span>
                              <span className="circuit-value">TRIGGERED</span>
                            </div>
                          )}
                          {circuitBreaker.from_file?.daily_limit_hit && (
                            <div className="circuit-item triggered">
                              <span className="circuit-label">Daily Limit</span>
                              <span className="circuit-value">TRIGGERED</span>
                            </div>
                          )}
                          {Object.entries(circuitBreaker.from_file?.stop_losses_triggered || {}).map(([symbol, triggered]) => (
                            triggered && (
                              <div key={symbol} className="circuit-item triggered">
                                <span className="circuit-label">{symbol} Stop-Loss</span>
                                <span className="circuit-value">TRIGGERED</span>
                              </div>
                            )
                          ))}
                        </div>

                        <div className="circuit-breaker-note">
                          <span className="text-xs text-muted italic">
                            Persists across restarts. Use reset to resume trading.
                          </span>
                        </div>

                        <button
                          className="btn-circuit-reset"
                          onClick={() => resetCircuitBreaker('all')}
                          disabled={circuitBreakerResetting}
                        >
                          {circuitBreakerResetting ? (
                            <>Resetting...</>
                          ) : (
                            <>
                              <span className="reset-icon">↻</span>
                              Reset All Circuit Breakers
                            </>
                          )}
                        </button>
                      </div>
                    )}

                    {/* All Clear Status */}
                    {!risk.trading_halted && !circuitBreaker.anyTriggered && (
                      <div className="circuit-all-clear">
                        <span className="all-clear-icon">✓</span>
                        <span className="text-success text-sm font-medium">All Systems Operational</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Grid Levels */}
                <div className="card">
                  <div className="card-header">
                    <span className="card-title">Grid Levels</span>
                  </div>
                  <div className="card-body max-h-52 overflow-y-auto space-y-3">
                    {Object.entries(grid.summaries || {}).map(([symbol, summary]) => (
                      <div key={symbol} className="p-3 rounded-xl bg-[rgba(255,255,255,0.02)] border border-[rgba(255,255,255,0.04)]">
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-gold font-semibold">{symbol}</span>
                          <span className="text-xs text-muted">{summary.range?.spacing_pct?.toFixed(2) || '--'}% spacing</span>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <span className="text-muted">Range:</span>
                          <span className="text-right font-mono text-secondary">
                            ${summary.range?.lower?.toLocaleString() || '--'} - ${summary.range?.upper?.toLocaleString() || '--'}
                          </span>
                          <span className="text-muted">Buy Orders:</span>
                          <span className="text-right text-success">{summary.levels?.pending_buys || 0}</span>
                          <span className="text-muted">Sell Orders:</span>
                          <span className="text-right text-danger">{summary.levels?.pending_sells || 0}</span>
                          <span className="text-muted">Fills:</span>
                          <span className="text-right text-gold font-mono">{summary.performance?.completed_trades || 0}</span>
                          <span className="text-muted">Cycles:</span>
                          <span className="text-right text-gold font-mono">{summary.performance?.completed_cycles || 0}</span>
                        </div>
                      </div>
                    ))}
                    {Object.keys(grid.summaries || {}).length === 0 && (
                      <div className="text-center py-6 text-muted text-sm">
                        Initializing grid levels...
                      </div>
                    )}
                  </div>
                </div>

                {/* Windfall Profit Captures */}
                <div className="card windfall-card">
                  <div className="card-header">
                    <div className="windfall-icon-container">
                      <span className="windfall-icon">💰</span>
                    </div>
                    <span className="card-title">Windfall Captures</span>
                    <span className={`badge ${windfall.enabled ? 'badge-success' : 'badge-neutral'} ml-auto`}>
                      {windfall.enabled ? 'Active' : 'Inactive'}
                    </span>
                  </div>
                  <div className="card-body space-y-4">
                    {/* Stats Hero */}
                    <div className="windfall-stats-hero">
                      <div className="windfall-stat-main">
                        <span className="windfall-stat-value text-success">
                          ${windfall.total_profit?.toFixed(2) || '0.00'}
                        </span>
                        <span className="windfall-stat-label">Total Captured</span>
                      </div>
                      <div className="windfall-stat-secondary">
                        <div className="windfall-mini-stat">
                          <span className="windfall-mini-value">{windfall.total_captures || 0}</span>
                          <span className="windfall-mini-label">Captures</span>
                        </div>
                        <div className="windfall-mini-stat">
                          <span className="windfall-mini-value">
                            ${windfall.total_captures > 0 ? (windfall.total_profit / windfall.total_captures).toFixed(2) : '0.00'}
                          </span>
                          <span className="windfall-mini-label">Avg/Capture</span>
                        </div>
                      </div>
                    </div>

                    {/* Config Display */}
                    <div className="windfall-config">
                      <div className="windfall-config-item">
                        <span className="config-label">Soft Trigger</span>
                        <span className="config-value">{windfall.config?.soft_threshold_pct || 4}% + RSI&gt;{windfall.config?.rsi_threshold || 70}</span>
                      </div>
                      <div className="windfall-config-item">
                        <span className="config-label">Hard Trigger</span>
                        <span className="config-value">{windfall.config?.hard_threshold_pct || 6}%</span>
                      </div>
                      <div className="windfall-config-item">
                        <span className="config-label">Sell Portion</span>
                        <span className="config-value">{((windfall.config?.sell_portion || 0.70) * 100).toFixed(0)}%</span>
                      </div>
                    </div>

                    {/* Recent Transactions */}
                    <div className="windfall-transactions">
                      <div className="windfall-transactions-header">
                        <span className="text-xs text-muted uppercase tracking-wider">Recent Captures</span>
                      </div>
                      <div className="windfall-transactions-list">
                        {(windfall.transactions || []).slice(-5).reverse().map((tx, idx) => (
                          <div key={idx} className="windfall-transaction-item">
                            <div className="windfall-tx-icon">
                              <span className="windfall-tx-check">✓</span>
                            </div>
                            <div className="windfall-tx-details">
                              <span className="windfall-tx-symbol">{tx.symbol}</span>
                              <span className="windfall-tx-trigger">
                                {tx.trigger_type === 'hard_threshold' ? `${tx.unrealized_pct}%` : `RSI ${tx.rsi}`}
                              </span>
                            </div>
                            <div className="windfall-tx-profit">
                              <span className="text-success font-mono">+${tx.profit?.toFixed(2)}</span>
                            </div>
                          </div>
                        ))}
                        {(!windfall.transactions || windfall.transactions.length === 0) && (
                          <div className="windfall-empty">
                            <span className="windfall-empty-icon">🎯</span>
                            <span className="windfall-empty-text">Waiting for windfall opportunities...</span>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Active Cooldowns */}
                    {Object.keys(windfall.active_cooldowns || {}).length > 0 && (
                      <div className="windfall-cooldowns">
                        <span className="text-xs text-muted">Active Cooldowns:</span>
                        <div className="windfall-cooldown-list">
                          {Object.entries(windfall.active_cooldowns).map(([symbol, time]) => (
                            <span key={symbol} className="windfall-cooldown-badge">
                              {symbol}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Smart Filters - NEW */}
                <div className="card card-gold">
                  <div className="card-header">
                    <div className="w-2 h-2 rounded-full bg-gradient-to-r from-emerald-400 to-cyan-400 animate-pulse" />
                    <span className="card-title">Smart Filters</span>
                  </div>
                  <div className="card-body space-y-3">
                    {/* Time Filter */}
                    <div className={`p-3 rounded-xl ${smartFilters.time_filter?.should_trade ? 'bg-[rgba(62,207,142,0.06)] border border-[rgba(62,207,142,0.15)]' : 'bg-[rgba(229,115,115,0.06)] border border-[rgba(229,115,115,0.15)]'}`}>
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-lg">⏰</span>
                          <span className="text-sm font-medium text-secondary">Time Filter</span>
                        </div>
                        <span className={`text-xs font-mono px-2 py-1 rounded-full ${smartFilters.time_filter?.should_trade ? 'bg-[rgba(62,207,142,0.15)] text-success' : 'bg-[rgba(229,115,115,0.15)] text-danger'}`}>
                          {smartFilters.time_filter?.should_trade ? 'ACTIVE' : 'PAUSED'}
                        </span>
                      </div>
                      <div className="text-xs text-muted">{smartFilters.time_filter?.reason || 'Checking optimal hours...'}</div>
                      <div className="mt-2 flex items-center gap-2">
                        <div className="flex-1 h-1.5 rounded-full bg-[rgba(255,255,255,0.06)] overflow-hidden">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-gold to-amber-400 transition-all duration-500"
                            style={{ width: `${(smartFilters.time_filter?.time_quality || 0) * 100}%` }}
                          />
                        </div>
                        <span className="text-xs text-gold font-mono">{((smartFilters.time_filter?.time_quality || 0) * 100).toFixed(0)}%</span>
                      </div>
                    </div>

                    {/* Momentum Filter */}
                    <div className={`p-3 rounded-xl ${smartFilters.momentum?.allow_buy ? 'bg-[rgba(255,255,255,0.02)]' : 'bg-[rgba(251,191,36,0.06)]'} border border-[rgba(255,255,255,0.06)]`}>
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-lg">📈</span>
                          <span className="text-sm font-medium text-secondary">Momentum</span>
                        </div>
                        <span className={`text-xs font-mono ${(smartFilters.momentum?.status?.momentum || 0) > 0 ? 'text-success' : (smartFilters.momentum?.status?.momentum || 0) < 0 ? 'text-danger' : 'text-muted'}`}>
                          {(smartFilters.momentum?.status?.momentum || 0) > 0 ? '+' : ''}{(smartFilters.momentum?.status?.momentum || 0).toFixed(2)}%
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 mt-2">
                        <div className={`text-xs px-2 py-1 rounded text-center ${smartFilters.momentum?.allow_buy ? 'bg-[rgba(62,207,142,0.1)] text-success' : 'bg-[rgba(229,115,115,0.1)] text-danger'}`}>
                          BUY {smartFilters.momentum?.allow_buy ? '✓' : '✗'}
                        </div>
                        <div className={`text-xs px-2 py-1 rounded text-center ${smartFilters.momentum?.allow_sell ? 'bg-[rgba(62,207,142,0.1)] text-success' : 'bg-[rgba(251,191,36,0.1)] text-gold'}`}>
                          SELL {smartFilters.momentum?.allow_sell ? '✓' : 'HOLD'}
                        </div>
                      </div>
                    </div>

                    {/* Smart Grid Regime Detection - Safety Switch */}
                    <div className={`p-3 rounded-xl border ${
                      regime.is_strong_trend
                        ? 'bg-[rgba(229,115,115,0.08)] border-[rgba(229,115,115,0.3)] shadow-[0_0_20px_rgba(229,115,115,0.1)]'
                        : 'bg-[rgba(62,207,142,0.04)] border-[rgba(62,207,142,0.15)]'
                    }`}>
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-lg">{regime.is_strong_trend ? '🛡️' : '⚡'}</span>
                          <span className="text-sm font-medium text-secondary">Smart Grid</span>
                        </div>
                        <span className={`text-xs font-bold uppercase tracking-wide px-2 py-0.5 rounded ${
                          regime.current === 'TRENDING_DOWN' ? 'bg-[rgba(229,115,115,0.2)] text-danger' :
                          regime.current === 'TRENDING_UP' ? 'bg-[rgba(62,207,142,0.2)] text-success' :
                          regime.current === 'RANGING' ? 'bg-[rgba(212,175,55,0.2)] text-gold' :
                          regime.current === 'VOLATILE' ? 'bg-[rgba(100,181,246,0.2)] text-info' :
                          'bg-[rgba(255,255,255,0.1)] text-muted'
                        }`}>
                          {regime.current || 'UNKNOWN'}
                        </span>
                      </div>

                      {/* ADX Indicator */}
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-xs text-muted">ADX</span>
                        <div className="flex-1 h-1.5 rounded-full bg-[rgba(255,255,255,0.06)] overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              regime.adx > 40 ? 'bg-gradient-to-r from-danger to-rose-400' :
                              regime.adx > 25 ? 'bg-gradient-to-r from-gold to-amber-400' :
                              'bg-gradient-to-r from-success to-emerald-400'
                            }`}
                            style={{ width: `${Math.min(100, (regime.adx / 60) * 100)}%` }}
                          />
                        </div>
                        <span className={`text-xs font-mono ${regime.adx > 40 ? 'text-danger' : regime.adx > 25 ? 'text-gold' : 'text-success'}`}>
                          {(regime.adx || 0).toFixed(0)}
                        </span>
                      </div>

                      {/* Grid Status */}
                      <div className="grid grid-cols-2 gap-2">
                        <div className={`text-xs px-2 py-1 rounded text-center font-medium ${
                          regime.allow_buy
                            ? 'bg-[rgba(62,207,142,0.1)] text-success'
                            : 'bg-[rgba(229,115,115,0.15)] text-danger border border-[rgba(229,115,115,0.3)]'
                        }`}>
                          BUY {regime.allow_buy ? '✓' : '⛔'}
                        </div>
                        <div className={`text-xs px-2 py-1 rounded text-center font-medium ${
                          regime.allow_sell
                            ? 'bg-[rgba(62,207,142,0.1)] text-success'
                            : 'bg-[rgba(251,191,36,0.15)] text-gold border border-[rgba(251,191,36,0.3)]'
                        }`}>
                          SELL {regime.allow_sell ? '✓' : '🔒'}
                        </div>
                      </div>

                      {/* Warning when strong trend detected */}
                      {regime.is_strong_trend && (
                        <div className="mt-2 p-2 rounded-lg bg-[rgba(229,115,115,0.1)] border border-[rgba(229,115,115,0.2)]">
                          <div className="flex items-center gap-2 text-xs text-danger">
                            <span>⚠️</span>
                            <span className="font-medium">
                              {regime.current === 'TRENDING_DOWN'
                                ? 'Falling knife protection active'
                                : 'Strong uptrend - holding sells'}
                            </span>
                          </div>
                        </div>
                      )}

                      {/* Per-symbol regime status */}
                      {Object.keys(regime.all_regimes || {}).length > 0 && (
                        <div className="mt-2 pt-2 border-t border-[rgba(255,255,255,0.06)]">
                          <div className="grid grid-cols-2 gap-1">
                            {Object.entries(regime.all_regimes || {}).map(([symbol, symbolRegime]) => (
                              <div key={symbol} className="flex items-center justify-between text-xs px-1.5 py-0.5 rounded bg-[rgba(255,255,255,0.02)]">
                                <span className="text-muted truncate">{symbol.split('/')[0]}</span>
                                <span className={`font-mono ${
                                  symbolRegime === 'TRENDING_DOWN' ? 'text-danger' :
                                  symbolRegime === 'TRENDING_UP' ? 'text-success' :
                                  symbolRegime === 'RANGING' ? 'text-gold' :
                                  'text-muted'
                                }`}>
                                  {regime.all_paused?.[symbol] ? '⏸' : '▶'}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Correlations */}
                    {Object.keys(smartFilters.correlations || {}).length > 0 && (
                      <div className="p-3 rounded-xl bg-[rgba(255,255,255,0.02)] border border-[rgba(255,255,255,0.06)]">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-lg">🔗</span>
                          <span className="text-sm font-medium text-secondary">Correlations</span>
                        </div>
                        <div className="space-y-1">
                          {Object.entries(smartFilters.correlations || {}).slice(0, 3).map(([pair, corr]) => (
                            <div key={pair} className="flex justify-between text-xs">
                              <span className="text-muted">{pair.replace('-', ' ↔ ')}</span>
                              <span className={`font-mono ${Math.abs(corr) > 0.85 ? 'text-danger' : Math.abs(corr) > 0.7 ? 'text-gold' : 'text-success'}`}>
                                {(corr * 100).toFixed(0)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Performance Tracking - NEW */}
                <div className="card">
                  <div className="card-header">
                    <span className="card-title">Performance</span>
                    <div className="flex-1" />
                    <span className={`text-xs font-mono ${performance.win_rate >= 50 ? 'text-success' : 'text-danger'}`}>
                      {performance.win_rate || 0}% WR
                    </span>
                  </div>
                  <div className="card-body space-y-3">
                    <div className="grid grid-cols-2 gap-3">
                      <div className="text-center p-2 rounded-lg bg-[rgba(255,255,255,0.02)]">
                        <div className="text-xs text-muted mb-1">Trades</div>
                        <div className="text-lg font-mono text-gold">{performance.total_trades || 0}</div>
                      </div>
                      <div className="text-center p-2 rounded-lg bg-[rgba(255,255,255,0.02)]">
                        <div className="text-xs text-muted mb-1">Avg Profit</div>
                        <div className={`text-lg font-mono ${(performance.avg_profit_per_trade || 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                          ${(performance.avg_profit_per_trade || 0).toFixed(2)}
                        </div>
                      </div>
                    </div>

                    {performance.total_trades > 0 && (
                      <>
                        <div className="divider" />
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span className="text-muted">Best Hour</span>
                            <span className="text-success font-mono">{performance.best_hour}:00 (+${(performance.best_hour_profit || 0).toFixed(2)})</span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-muted">Worst Hour</span>
                            <span className="text-danger font-mono">{performance.worst_hour}:00 (${(performance.worst_hour_profit || 0).toFixed(2)})</span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-muted">vs Expected</span>
                            <span className={`font-mono ${(performance.expected_vs_actual || 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                              {(performance.expected_vs_actual || 0) >= 0 ? '+' : ''}{(performance.expected_vs_actual || 0).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </div>

              {/* ─────────────────────────────────────────────────────────────────
                  CENTER COLUMNS (spans 2)
              ───────────────────────────────────────────────────────────────── */}
              <div className="col-span-2 space-y-5">
                {/* Performance Metrics Row */}
                <div className="grid grid-cols-4 gap-4">
                  <div className="metric">
                    <div className="metric-label">Daily P/L</div>
                    <div className={`metric-value font-mono ${(risk.daily_pnl_pct || 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                      {(risk.daily_pnl_pct || 0) >= 0 ? '+' : ''}{(risk.daily_pnl_pct || 0).toFixed(2)}%
                    </div>
                  </div>
                  <div className="metric">
                    <div className="metric-label">Portfolio</div>
                    <div className="metric-value font-mono">
                      ${account.equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </div>
                  </div>
                  <div className="metric">
                    <div className="metric-label">Unrealized P/L</div>
                    <div className={`metric-value font-mono ${positions.reduce((sum, p) => sum + (parseFloat(p.unrealized_pl) || 0), 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                      {positions.reduce((sum, p) => sum + (parseFloat(p.unrealized_pl) || 0), 0) >= 0 ? '+' : ''}
                      ${Math.abs(positions.reduce((sum, p) => sum + (parseFloat(p.unrealized_pl) || 0), 0)).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </div>
                  </div>
                  <div className="metric">
                    <div className="metric-label">Positions</div>
                    <div className="metric-value font-mono text-gold">
                      {positions.length}/{data.multi_asset?.symbols?.length || 4}
                    </div>
                  </div>
                </div>

                {/* Price Action with Symbol Selector */}
                <div className="card overflow-visible">
                  <div className="card-header border-b border-[rgba(255,255,255,0.04)] pb-4">
                    <span className="card-title">Price Action</span>
                    <div className="flex-1" />
                    <span className={getSignalClass(ai.signal)}>{ai.signal}</span>
                  </div>

                  {/* Symbol Selector - Luxury Ticker Strip */}
                  <div className="px-4 py-3 border-b border-[rgba(255,255,255,0.04)]">
                    <div className="flex gap-2">
                      {['BTC/USD', 'SOL/USD', 'LTC/USD', 'AVAX/USD'].map((sym) => {
                        const isSelected = selectedSymbol === sym;
                        const symData = symbolPrices[sym] || { price: 0, changePercent: 0 };
                        const hasPosition = positions.some(p => p.symbol?.includes(sym.split('/')[0]));
                        const isPositive = symData.changePercent >= 0;
                        const cryptoIcons = {
                          'BTC/USD': '₿',
                          'SOL/USD': '◎',
                          'LTC/USD': 'Ł',
                          'AVAX/USD': 'A'
                        };

                        return (
                          <button
                            key={sym}
                            onClick={() => setSelectedSymbol(sym)}
                            className={`
                              group relative flex-1 px-3 py-2.5 rounded-xl
                              transition-all duration-300 ease-out
                              ${isSelected
                                ? 'bg-gradient-to-b from-[rgba(212,175,55,0.15)] to-[rgba(212,175,55,0.05)] border border-[rgba(212,175,55,0.4)] shadow-[0_0_20px_rgba(212,175,55,0.15)]'
                                : 'bg-[rgba(255,255,255,0.02)] border border-transparent hover:bg-[rgba(255,255,255,0.04)] hover:border-[rgba(255,255,255,0.08)]'
                              }
                            `}
                          >
                            {/* Animated gold underline for selected */}
                            {isSelected && (
                              <div className="absolute -bottom-px left-1/2 -translate-x-1/2 w-12 h-0.5 bg-gradient-to-r from-transparent via-[#d4af37] to-transparent" />
                            )}

                            {/* Position indicator dot */}
                            {hasPosition && (
                              <div className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-[#3ecf8e] shadow-[0_0_8px_rgba(62,207,142,0.6)]" />
                            )}

                            <div className="flex items-center gap-2">
                              {/* Crypto Icon */}
                              <span className={`
                                text-lg font-bold transition-all duration-300
                                ${isSelected ? 'text-gold scale-110' : 'text-muted group-hover:text-secondary'}
                              `}>
                                {cryptoIcons[sym]}
                              </span>

                              <div className="flex flex-col items-start min-w-0">
                                {/* Symbol Name */}
                                <span className={`
                                  text-xs font-semibold tracking-wide transition-colors duration-300
                                  ${isSelected ? 'text-gold' : 'text-secondary group-hover:text-primary'}
                                `}>
                                  {sym.split('/')[0]}
                                </span>

                                {/* Price */}
                                <span className={`
                                  font-mono text-sm font-medium transition-colors duration-300
                                  ${isSelected ? 'text-primary' : 'text-muted group-hover:text-secondary'}
                                `}>
                                  ${symData.price > 1000
                                    ? symData.price.toLocaleString(undefined, { maximumFractionDigits: 0 })
                                    : symData.price.toFixed(2)
                                  }
                                </span>
                              </div>

                              {/* Change Percentage */}
                              <span className={`
                                ml-auto font-mono text-xs font-semibold px-1.5 py-0.5 rounded
                                transition-all duration-300
                                ${isPositive
                                  ? 'text-[#3ecf8e] bg-[rgba(62,207,142,0.1)]'
                                  : 'text-[#e57373] bg-[rgba(229,115,115,0.1)]'
                                }
                              `}>
                                {isPositive ? '↑' : '↓'} {Math.abs(symData.changePercent).toFixed(2)}%
                              </span>
                            </div>

                            {/* Sparkline mini-chart preview */}
                            {symbolChartData[sym]?.data?.length > 5 && (
                              <div className="mt-2 h-6 opacity-60 group-hover:opacity-100 transition-opacity">
                                <svg viewBox="0 0 100 24" preserveAspectRatio="none" className="w-full h-full">
                                  <defs>
                                    <linearGradient id={`sparkline-${sym.replace('/', '')}`} x1="0" y1="0" x2="0" y2="1">
                                      <stop offset="0%" stopColor={isSelected ? '#d4af37' : (isPositive ? '#3ecf8e' : '#e57373')} stopOpacity="0.3" />
                                      <stop offset="100%" stopColor={isSelected ? '#d4af37' : (isPositive ? '#3ecf8e' : '#e57373')} stopOpacity="0" />
                                    </linearGradient>
                                  </defs>
                                  {(() => {
                                    const data = symbolChartData[sym]?.data?.slice(-20) || [];
                                    if (data.length < 2) return null;
                                    const min = Math.min(...data);
                                    const max = Math.max(...data);
                                    const range = max - min || 1;
                                    const points = data.map((d, i) => {
                                      const x = (i / (data.length - 1)) * 100;
                                      const y = 22 - ((d - min) / range) * 20;
                                      return `${x},${y}`;
                                    }).join(' ');
                                    const areaPoints = `0,24 ${points} 100,24`;
                                    return (
                                      <>
                                        <polygon points={areaPoints} fill={`url(#sparkline-${sym.replace('/', '')})`} />
                                        <polyline
                                          points={points}
                                          fill="none"
                                          stroke={isSelected ? '#d4af37' : (isPositive ? '#3ecf8e' : '#e57373')}
                                          strokeWidth="1.5"
                                          strokeLinecap="round"
                                          strokeLinejoin="round"
                                        />
                                      </>
                                    );
                                  })()}
                                </svg>
                              </div>
                            )}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Main Chart */}
                  <div className="card-body h-52 chart-container">
                    <Line data={chartData} options={chartOptions} />
                  </div>

                  {/* Chart Footer with selected symbol info */}
                  <div className="px-4 py-2 border-t border-[rgba(255,255,255,0.04)] flex items-center justify-between text-xs">
                    <span className="text-muted">
                      {selectedSymbol} · {symbolChartData[selectedSymbol]?.data?.length || 0} data points
                    </span>
                    <span className="text-muted font-mono">
                      Last: ${symbolPrices[selectedSymbol]?.price?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '—'}
                    </span>
                  </div>
                </div>

                {/* Last Trade */}
                {lastTrade && (
                  <div className={`card ${lastTrade.action === 'BUY' ? 'border-[rgba(62,207,142,0.2)]' : 'border-[rgba(229,115,115,0.2)]'}`}>
                    <div className="card-header">
                      <div className={`status-dot ${lastTrade.action === 'BUY' ? 'success' : 'danger'}`} />
                      <span className="card-title">Last Trade · {lastTrade.action} {lastTrade.symbol}</span>
                    </div>
                    <div className="card-body">
                      <div className="grid grid-cols-2 gap-6">
                        <div>
                          <div className="text-muted text-sm mb-2">Reasoning</div>
                          {lastTrade.reasoning?.slice(0, 3).map((r, i) => (
                            <div key={i} className="text-secondary text-sm mb-1 flex items-start gap-2">
                              <span className="text-gold">•</span>{r}
                            </div>
                          ))}
                        </div>
                        {lastTrade.stop_loss && (
                          <div>
                            <div className="text-muted text-sm mb-2">Targets</div>
                            <div className="text-danger text-sm mb-1">
                              Stop Loss: ${lastTrade.stop_loss?.toLocaleString()}
                            </div>
                            <div className="text-success text-sm">
                              Take Profit: ${lastTrade.take_profit?.toLocaleString()}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* Positions */}
                <div className="card">
                  <div className="card-header">
                    <span className="card-title">Open Positions</span>
                    <div className="flex-1" />
                    <span className="text-muted text-sm">
                      {positions.length > 0 ? `${positions.length} active · $${positions.reduce((sum, p) => sum + (parseFloat(p.qty || 0) * parseFloat(p.current_price || 0)), 0).toLocaleString(undefined, {maximumFractionDigits: 0})} invested` : 'None'}
                    </span>
                  </div>
                  <div className="card-body">
                    {positions.length === 0 ? (
                      <div className="text-center py-8">
                        <div className="font-display text-xl text-muted mb-2">No Active Positions</div>
                        <div className="text-sm text-muted">Waiting for grid buy triggers...</div>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        {positions.map((p, i) => {
                          const qty = parseFloat(p.qty) || 0;
                          const entryPrice = parseFloat(p.avg_entry_price) || 0;
                          const currentPrice = parseFloat(p.current_price) || 0;
                          const marketValue = parseFloat(p.market_value) || (qty * currentPrice);
                          const costBasis = parseFloat(p.cost_basis) || (qty * entryPrice);
                          const unrealizedPL = parseFloat(p.unrealized_pl) || (marketValue - costBasis);
                          const unrealizedPLPct = parseFloat(p.unrealized_plpc) || (costBasis > 0 ? ((marketValue - costBasis) / costBasis) * 100 : 0);
                          const portfolioPct = account.equity > 0 ? (marketValue / account.equity) * 100 : 0;

                          return (
                            <div key={i} className={`p-4 rounded-xl border ${unrealizedPL >= 0 ? 'bg-[rgba(62,207,142,0.04)] border-[rgba(62,207,142,0.15)]' : 'bg-[rgba(229,115,115,0.04)] border-[rgba(229,115,115,0.15)]'}`}>
                              {/* Header Row */}
                              <div className="flex items-center justify-between mb-3">
                                <div className="flex items-center gap-3">
                                  <span className="text-gold font-display text-lg">{p.symbol}</span>
                                  <span className="badge badge-neutral text-xs">LONG</span>
                                </div>
                                <div className={`text-xl font-mono font-semibold ${unrealizedPL >= 0 ? 'text-success' : 'text-danger'}`}>
                                  {unrealizedPL >= 0 ? '+' : ''}{unrealizedPLPct.toFixed(2)}%
                                </div>
                              </div>

                              {/* Details Grid */}
                              <div className="grid grid-cols-4 gap-4 text-sm">
                                <div>
                                  <div className="text-muted text-xs mb-1">Quantity</div>
                                  <div className="font-mono text-secondary">{qty < 1 ? qty.toFixed(6) : qty.toFixed(2)}</div>
                                </div>
                                <div>
                                  <div className="text-muted text-xs mb-1">Entry Price</div>
                                  <div className="font-mono text-secondary">${entryPrice.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
                                </div>
                                <div>
                                  <div className="text-muted text-xs mb-1">Current Price</div>
                                  <div className="font-mono text-secondary">${currentPrice.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
                                </div>
                                <div>
                                  <div className="text-muted text-xs mb-1">Unrealized</div>
                                  <div className={`font-mono font-medium ${unrealizedPL >= 0 ? 'text-success' : 'text-danger'}`}>
                                    {unrealizedPL >= 0 ? '+' : ''}${unrealizedPL.toFixed(2)}
                                  </div>
                                </div>
                              </div>

                              {/* Bottom Row */}
                              <div className="flex items-center justify-between mt-3 pt-3 border-t border-[rgba(255,255,255,0.06)]">
                                <div className="flex items-center gap-4 text-xs">
                                  <span><span className="text-muted">Value:</span> <span className="font-mono text-gold">${marketValue.toLocaleString(undefined, {maximumFractionDigits: 0})}</span></span>
                                  <span><span className="text-muted">Cost:</span> <span className="font-mono">${costBasis.toLocaleString(undefined, {maximumFractionDigits: 0})}</span></span>
                                  <span><span className="text-muted">Portfolio:</span> <span className="font-mono text-gold">{portfolioPct.toFixed(1)}%</span></span>
                                </div>
                                <div className="text-xs text-muted">
                                  Grid Level {p.grid_level || '?'}
                                </div>
                              </div>
                            </div>
                          );
                        })}

                        {/* Portfolio Summary */}
                        <div className="mt-4 p-3 rounded-xl bg-[rgba(212,175,55,0.06)] border border-[rgba(212,175,55,0.15)]">
                          <div className="flex items-center justify-between text-sm">
                            <span className="text-muted">Total Invested</span>
                            <span className="font-mono text-gold font-medium">
                              ${positions.reduce((sum, p) => sum + (parseFloat(p.qty || 0) * parseFloat(p.current_price || 0)), 0).toLocaleString(undefined, {maximumFractionDigits: 0})}
                              <span className="text-muted ml-2">
                                ({account.equity > 0 ? ((positions.reduce((sum, p) => sum + (parseFloat(p.qty || 0) * parseFloat(p.current_price || 0)), 0) / account.equity) * 100).toFixed(1) : 0}%)
                              </span>
                            </span>
                          </div>
                          <div className="flex items-center justify-between text-sm mt-2">
                            <span className="text-muted">Unrealized P&L</span>
                            <span className={`font-mono font-medium ${positions.reduce((sum, p) => sum + (parseFloat(p.unrealized_pl) || 0), 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                              {positions.reduce((sum, p) => sum + (parseFloat(p.unrealized_pl) || 0), 0) >= 0 ? '+' : ''}
                              ${positions.reduce((sum, p) => sum + (parseFloat(p.unrealized_pl) || 0), 0).toFixed(2)}
                            </span>
                          </div>
                          <div className="flex items-center justify-between text-sm mt-2">
                            <span className="text-muted">Cash Available</span>
                            <span className="font-mono text-secondary">
                              ${(account.balance || 0).toLocaleString(undefined, {maximumFractionDigits: 0})}
                              <span className="text-muted ml-2">
                                ({account.equity > 0 ? ((account.balance / account.equity) * 100).toFixed(1) : 0}%)
                              </span>
                            </span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* ═══════════════════════════════════════════════════════════════
                    CONFIRMED ORDERS - Alpaca Verified Transactions
                ═══════════════════════════════════════════════════════════════ */}
                <div className="card relative overflow-hidden">
                  {/* Decorative corner accent */}
                  <div className="absolute top-0 right-0 w-24 h-24 opacity-20">
                    <svg viewBox="0 0 100 100" className="w-full h-full">
                      <defs>
                        <linearGradient id="verifiedGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                          <stop offset="0%" style={{stopColor: '#3ecf8e', stopOpacity: 0.6}} />
                          <stop offset="100%" style={{stopColor: '#d4af37', stopOpacity: 0.3}} />
                        </linearGradient>
                      </defs>
                      <path d="M100 0 L100 100 L0 100 Z" fill="url(#verifiedGrad)" />
                    </svg>
                  </div>

                  <div className="card-header border-b border-[rgba(62,207,142,0.15)]">
                    <div className="flex items-center gap-3">
                      <div className="w-2 h-2 rounded-full bg-success animate-pulse" />
                      <span className="card-title">Alpaca Verified Orders</span>
                    </div>
                    <div className="flex items-center gap-4">
                      {/* Reconciliation Status Badge */}
                      <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${
                        orders.reconciliation?.synced
                          ? 'bg-[rgba(62,207,142,0.12)] text-success border border-[rgba(62,207,142,0.25)]'
                          : 'bg-[rgba(229,115,115,0.12)] text-danger border border-[rgba(229,115,115,0.25)]'
                      }`}>
                        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          {orders.reconciliation?.synced ? (
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          ) : (
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                          )}
                        </svg>
                        {orders.reconciliation?.synced ? 'SYNCED' : 'CHECK NEEDED'}
                      </div>
                      <span className="text-xs text-muted font-mono">
                        {orders.stats?.total_confirmed || 0} orders
                      </span>
                    </div>
                  </div>

                  <div className="card-body">
                    {/* Stats Row */}
                    <div className="grid grid-cols-4 gap-3 mb-4">
                      <div className="p-3 rounded-xl bg-[rgba(62,207,142,0.06)] border border-[rgba(62,207,142,0.12)]">
                        <div className="text-xs text-success uppercase tracking-wider mb-1">Buys</div>
                        <div className="text-xl font-display text-success">{orders.stats?.by_side?.buy || 0}</div>
                      </div>
                      <div className="p-3 rounded-xl bg-[rgba(229,115,115,0.06)] border border-[rgba(229,115,115,0.12)]">
                        <div className="text-xs text-danger uppercase tracking-wider mb-1">Sells</div>
                        <div className="text-xl font-display text-danger">{orders.stats?.by_side?.sell || 0}</div>
                      </div>
                      <div className="p-3 rounded-xl bg-[rgba(212,175,55,0.06)] border border-[rgba(212,175,55,0.12)]">
                        <div className="text-xs text-gold uppercase tracking-wider mb-1">Volume</div>
                        <div className="text-xl font-display text-gold">${(orders.stats?.total_volume || 0).toLocaleString(undefined, {maximumFractionDigits: 0})}</div>
                      </div>
                      <div className="p-3 rounded-xl bg-[rgba(100,181,246,0.06)] border border-[rgba(100,181,246,0.12)]">
                        <div className="text-xs text-info uppercase tracking-wider mb-1">Matched</div>
                        <div className="text-xl font-display text-info">{orders.reconciliation?.matched || 0}</div>
                      </div>
                    </div>

                    {/* Orders Table */}
                    {orders.confirmed?.length > 0 ? (
                      <div className="relative">
                        {/* Table Header */}
                        <div className="grid grid-cols-5 gap-2 px-3 py-2 text-xs text-muted uppercase tracking-wider border-b border-[rgba(255,255,255,0.06)]">
                          <div>Time</div>
                          <div>Symbol</div>
                          <div>Side</div>
                          <div className="text-right">Qty</div>
                          <div className="text-right">Fill Price</div>
                        </div>

                        {/* Table Body */}
                        <div className="space-y-1 max-h-48 overflow-y-auto custom-scrollbar">
                          {orders.confirmed.slice(0, 10).map((order, i) => {
                            const isBuy = order.side?.toLowerCase().includes('buy');
                            const fillTime = order.filled_at ? new Date(order.filled_at).toLocaleTimeString('en-US', {
                              hour: '2-digit',
                              minute: '2-digit',
                              second: '2-digit',
                              hour12: false
                            }) : '--:--:--';

                            return (
                              <div
                                key={order.order_id || i}
                                className={`grid grid-cols-5 gap-2 px-3 py-2 rounded-lg transition-all duration-200 hover:bg-[rgba(255,255,255,0.03)] ${
                                  i === 0 ? 'bg-[rgba(212,175,55,0.04)] border-l-2 border-gold' : ''
                                }`}
                              >
                                <div className="font-mono text-sm text-secondary">{fillTime}</div>
                                <div className="font-mono text-sm text-gold">{order.symbol}</div>
                                <div className={`text-sm font-medium ${isBuy ? 'text-success' : 'text-danger'}`}>
                                  {isBuy ? '● BUY' : '● SELL'}
                                </div>
                                <div className="font-mono text-sm text-secondary text-right">
                                  {(order.filled_qty || 0) < 1
                                    ? (order.filled_qty || 0).toFixed(6)
                                    : (order.filled_qty || 0).toFixed(2)}
                                </div>
                                <div className="font-mono text-sm text-right">
                                  <span className="text-primary">${(order.filled_avg_price || 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</span>
                                  {order.profit && (
                                    <span className={`ml-2 text-xs ${order.profit >= 0 ? 'text-success' : 'text-danger'}`}>
                                      {order.profit >= 0 ? '+' : ''}${order.profit.toFixed(2)}
                                    </span>
                                  )}
                                </div>
                              </div>
                            );
                          })}
                        </div>

                        {/* Fade overlay for scroll indication */}
                        {orders.confirmed.length > 5 && (
                          <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-[var(--midnight-surface)] to-transparent pointer-events-none" />
                        )}
                      </div>
                    ) : (
                      <div className="text-center py-8">
                        <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-[rgba(212,175,55,0.08)] border border-[rgba(212,175,55,0.15)] mb-3">
                          <svg className="w-6 h-6 text-gold" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                          </svg>
                        </div>
                        <div className="font-display text-xl text-muted mb-1">No Verified Orders</div>
                        <div className="text-sm text-muted">Orders will appear here once confirmed by Alpaca</div>
                      </div>
                    )}

                    {/* Last Reconciliation Time */}
                    {orders.reconciliation?.last_check && (
                      <div className="mt-4 pt-3 border-t border-[rgba(255,255,255,0.06)] flex items-center justify-between text-xs text-muted">
                        <span>Last sync check</span>
                        <span className="font-mono">
                          {orders.reconciliation.minutes_ago !== undefined
                            ? `${orders.reconciliation.minutes_ago}m ago`
                            : new Date(orders.reconciliation.last_check).toLocaleTimeString()}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* ─────────────────────────────────────────────────────────────────
                  RIGHT COLUMN
              ───────────────────────────────────────────────────────────────── */}
              <div className="space-y-5">
                {/* Multi-Asset */}
                {data.multi_asset && (
                  <div className="card card-gold">
                    <div className="card-header">
                      <span className="card-title text-gold">Multi-Asset</span>
                    </div>
                    <div className="card-body space-y-2">
                      {data.multi_asset.symbols?.map(sym => {
                        const sig = data.multi_asset.signals?.[sym] || 'HOLD';
                        const conf = data.multi_asset.confidences?.[sym] || 0;
                        const active = sym === data.multi_asset.active_symbol;
                        return (
                          <div key={sym} className={`flex items-center justify-between p-3 rounded-xl ${active ? 'bg-[rgba(212,175,55,0.08)] border border-[rgba(212,175,55,0.15)]' : 'bg-[rgba(255,255,255,0.02)]'}`}>
                            <div className="flex items-center gap-2">
                              {active && <div className="status-dot pulse" />}
                              <span className={active ? 'text-gold font-medium' : 'text-muted'}>{sym}</span>
                            </div>
                            <div className="flex items-center gap-3">
                              <span className="text-sm text-muted font-mono">{conf}%</span>
                              <span className={getSignalClass(sig)}>{sig}</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* AI Indicators */}
                <div className="card">
                  <div className="card-header">
                    <span className="card-title">AI Indicators</span>
                  </div>
                  <div className="card-body space-y-4">
                    {/* Momentum */}
                    <div>
                      <div className="text-xs text-muted uppercase tracking-wider mb-2">Momentum</div>
                      <div className="grid grid-cols-2 gap-2">
                        {[
                          { label: 'RSI', value: ai.features?.rsi?.toFixed(1), condition: (ai.features?.rsi || 50) < 30 ? 'success' : (ai.features?.rsi || 50) > 70 ? 'danger' : null },
                          { label: 'MACD', value: ai.features?.macd_hist?.toFixed(3), condition: (ai.features?.macd_hist || 0) > 0 ? 'success' : 'danger' },
                          { label: 'Stoch', value: `${ai.features?.stoch_k?.toFixed(0) || '--'}/${ai.features?.stoch_d?.toFixed(0) || '--'}`, condition: (ai.features?.stoch_k || 50) < 20 ? 'success' : (ai.features?.stoch_k || 50) > 80 ? 'danger' : null },
                          { label: 'MFI', value: ai.features?.mfi?.toFixed(0), condition: (ai.features?.mfi || 50) < 20 ? 'success' : (ai.features?.mfi || 50) > 80 ? 'danger' : null }
                        ].map(item => (
                          <div key={item.label} className="p-2 rounded-lg bg-[rgba(255,255,255,0.02)]">
                            <div className="text-xs text-muted">{item.label}</div>
                            <div className={`font-mono font-medium ${item.condition === 'success' ? 'text-success' : item.condition === 'danger' ? 'text-danger' : 'text-secondary'}`}>
                              {item.value || '--'}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Trend */}
                    <div>
                      <div className="text-xs text-muted uppercase tracking-wider mb-2">Trend</div>
                      <div className="grid grid-cols-2 gap-2">
                        {[
                          { label: 'ADX', value: ai.features?.adx?.toFixed(0), condition: (ai.features?.adx || 0) > 25 ? 'gold' : null },
                          { label: 'Trend', value: `${ai.features?.trend_alignment?.toFixed(0) || '--'}/3`, condition: (ai.features?.trend_alignment || 0) >= 2 ? 'success' : (ai.features?.trend_alignment || 0) <= 1 ? 'danger' : null },
                          { label: 'ROC 5', value: `${ai.features?.roc_5?.toFixed(2) || '--'}%`, condition: (ai.features?.roc_5 || 0) > 0 ? 'success' : 'danger' },
                          { label: 'Momentum', value: ai.features?.momentum_alignment, condition: (ai.features?.momentum_alignment || 0) > 0 ? 'success' : (ai.features?.momentum_alignment || 0) < 0 ? 'danger' : null }
                        ].map(item => (
                          <div key={item.label} className="p-2 rounded-lg bg-[rgba(255,255,255,0.02)]">
                            <div className="text-xs text-muted">{item.label}</div>
                            <div className={`font-mono font-medium ${item.condition === 'success' ? 'text-success' : item.condition === 'danger' ? 'text-danger' : item.condition === 'gold' ? 'text-gold' : 'text-secondary'}`}>
                              {item.value || '--'}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Volume */}
                    <div>
                      <div className="text-xs text-muted uppercase tracking-wider mb-2">Volume & Volatility</div>
                      <div className="grid grid-cols-2 gap-2">
                        {[
                          { label: 'Volume', value: `${ai.features?.volume_ratio?.toFixed(2) || '1.00'}x`, condition: (ai.features?.volume_ratio || 1) > 1.5 ? 'success' : null },
                          { label: 'Spike', value: ai.features?.volume_spike ? 'Yes' : 'No', condition: ai.features?.volume_spike ? 'gold' : null },
                          { label: 'BB Squeeze', value: ai.features?.bb_squeeze ? 'Yes' : 'No', condition: ai.features?.bb_squeeze ? 'gold' : null },
                          { label: 'Vol Regime', value: ai.features?.volatility_regime?.toFixed(2) || '1.00', condition: (ai.features?.volatility_regime || 1) > 1.5 ? 'gold' : null }
                        ].map(item => (
                          <div key={item.label} className="p-2 rounded-lg bg-[rgba(255,255,255,0.02)]">
                            <div className="text-xs text-muted">{item.label}</div>
                            <div className={`font-mono font-medium ${item.condition === 'success' ? 'text-success' : item.condition === 'gold' ? 'text-gold' : 'text-secondary'}`}>
                              {item.value}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Position Sizing */}
                <div className="card">
                  <div className="card-header">
                    <span className="card-title">Position Sizing</span>
                  </div>
                  <div className="card-body space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted">Kelly Fraction</span>
                      <span className="text-gold font-mono">{((ultra.kelly?.kelly_fraction || 0.25) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted">Win Rate</span>
                      <span className="font-mono">{((ultra.kelly?.win_rate || 0.5) * 100).toFixed(0)}%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted">Sample Size</span>
                      <span className="font-mono">{ultra.kelly?.sample_size || 0}</span>
                    </div>
                    <div className="divider" />
                    <div>
                      <div className="text-muted text-sm mb-1">Size @ {ai.confidence}% confidence</div>
                      <div className="font-display text-2xl text-gold">{(3 + ((ai.confidence - 70) / 30) * 2).toFixed(1)}%</div>
                    </div>
                  </div>
                </div>

                {/* Logs */}
                <div className="card" style={{ height: '240px' }}>
                  <div className="card-header">
                    <span className="card-title">Activity</span>
                  </div>
                  <div className="card-body h-full overflow-hidden">
                    <div ref={logsContainerRef} className="log-panel h-full overflow-y-auto">
                      {logs.map((log, i) => (
                        <div key={i} className={`log-entry ${log.type.toLowerCase()}`}>
                          <span className="log-timestamp">{log.timestamp}</span>
                          <span>{log.message}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* ═══════════════════════════════════════════════════════════════════
                FOOTER
            ═══════════════════════════════════════════════════════════════════ */}
            <footer className="mt-6">
              <div className="flex items-center justify-between px-6 py-4 rounded-2xl bg-[rgba(255,255,255,0.02)] border border-[rgba(255,255,255,0.04)]">
                <div className="flex items-center gap-8 text-sm">
                  <span><span className="text-muted">Mode:</span> <span className="text-success">Grid Trading</span></span>
                  <span><span className="text-muted">Daily Limit:</span> <span className="text-gold">5%</span></span>
                  <span><span className="text-muted">Max DD:</span> <span className="text-danger">10%</span></span>
                  <span><span className="text-muted">Stop Loss:</span> <span className="text-danger">10% below grid</span></span>
                </div>
                <div className="text-muted text-sm">
                  BlueBird Private v4.0 · {Object.keys(grid.summaries || {}).length || 4} Assets
                </div>
              </div>
            </footer>
          </>
        )}
      </div>
    </div>
  );
}

export default App;
