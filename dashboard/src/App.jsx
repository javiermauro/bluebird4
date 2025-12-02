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
import {
  Activity, TrendingUp, TrendingDown, Terminal, Zap, Brain,
  Target, Clock, BarChart3, Gauge, ArrowUpCircle, ArrowDownCircle,
  PauseCircle, AlertTriangle, CheckCircle2, XCircle, Layers,
  Cpu, Eye, GitBranch, Sparkles, GraduationCap, LineChart,
  Grid3X3, Shield, DollarSign, AlertOctagon, StopCircle
} from 'lucide-react';
import TrainingDashboard from './components/TrainingDashboard';

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

// Feature importance colors
const FEATURE_COLORS = {
  rsi: '#10b981',
  volume: '#6366f1',
  macd: '#f59e0b',
  bollinger: '#ec4899',
  price_action: '#8b5cf6'
};

// Timeframe signal config
const TF_SIGNAL_CONFIG = {
  BULLISH: { color: 'text-emerald-400', bg: 'bg-emerald-500/20', icon: '▲' },
  BEARISH: { color: 'text-red-400', bg: 'bg-red-500/20', icon: '▼' },
  NEUTRAL: { color: 'text-gray-400', bg: 'bg-gray-500/20', icon: '─' }
};

function App() {
  // Navigation state
  const [currentView, setCurrentView] = useState('trading'); // 'trading' or 'training'

  const [status, setStatus] = useState('disconnected');
  const [price, setPrice] = useState(0.0);
  const [logs, setLogs] = useState([]);
  const [account, setAccount] = useState({ equity: 0, buying_power: 0, balance: 0 });
  const [positions, setPositions] = useState([]);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [data, setData] = useState({});
  
  // AI state
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
  
  // Ultra state
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

  // Grid Trading state
  const [grid, setGrid] = useState({
    active: false,
    total_trades: 0,
    total_profit: 0,
    summaries: {}
  });

  // Risk Management state
  const [risk, setRisk] = useState({
    daily_pnl: 0,
    daily_pnl_pct: 0,
    drawdown_pct: 0,
    peak_equity: 0,
    daily_limit_hit: false,
    max_drawdown_hit: false,
    trading_halted: false,
    stop_losses: {}
  });

  const [market, setMarket] = useState({ high: 0, low: 0, volume: 0, change: 0 });
  const [lastTrade, setLastTrade] = useState(null);

  const [chartData, setChartData] = useState({
    labels: [],
    datasets: [{
      label: 'Price',
      data: [],
      borderColor: '#6366f1',
      backgroundColor: 'rgba(99, 102, 241, 0.1)',
      tension: 0.4,
      fill: true,
      pointRadius: 0
    }]
  });

  const ws = useRef(null);
  const logsEndRef = useRef(null);

  useEffect(() => {
    ws.current = new WebSocket('ws://localhost:8000/ws');

    ws.current.onopen = () => {
      setStatus('connected');
      addLog('Connected to BlueBird AI Core');
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
        setLastUpdate(new Date());
        updateChart(updateData.timestamp, updateData.price);
      } else if (message.type === 'log') {
        addLog(message.data.message);
      }
    };

    ws.current.onclose = () => {
      setStatus('disconnected');
      addLog('Disconnected from Core');
    };

    return () => {
      if (ws.current) ws.current.close();
    };
  }, []);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const addLog = (msg) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev.slice(-49), `[${timestamp}] ${msg}`]);
  };

  const updateChart = (timestamp, price) => {
    setChartData(prev => {
      const newLabels = [...prev.labels, new Date(timestamp).toLocaleTimeString()];
      const newData = [...prev.datasets[0].data, price];

      if (newLabels.length > 60) {
        newLabels.shift();
        newData.shift();
      }

      return {
        ...prev,
        labels: newLabels,
        datasets: [{ ...prev.datasets[0], data: newData }]
      };
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
        backgroundColor: 'rgba(0,0,0,0.9)',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255,255,255,0.03)' },
        ticks: { color: '#6b7280', maxTicksLimit: 8 }
      },
      y: {
        grid: { color: 'rgba(255,255,255,0.03)' },
        ticks: { color: '#6b7280' }
      }
    }
  };

  const totalExposure = positions.reduce((sum, p) => sum + (p.qty * p.current_price), 0);
  const exposurePct = account.equity > 0 ? (totalExposure / account.equity * 100) : 0;

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-gray-300 p-4 font-sans">
      {/* Header */}
      <header className="flex justify-between items-center mb-6">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-gradient-to-br from-indigo-500/20 to-purple-500/20 rounded-xl border border-indigo-500/30">
              <Brain className="w-7 h-7 text-indigo-400" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">
                BlueBird <span className="text-emerald-400">Grid</span>
              </h1>
              <p className="text-xs text-gray-500 font-mono">GRID TRADING SYSTEM v4.0</p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex items-center gap-1 bg-black/30 p-1 rounded-lg border border-gray-800">
            <button
              onClick={() => setCurrentView('trading')}
              className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition ${
                currentView === 'trading'
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-white/5'
              }`}
            >
              <LineChart className="w-4 h-4" />
              Trading
            </button>
            <button
              onClick={() => setCurrentView('training')}
              className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition ${
                currentView === 'training'
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-white/5'
              }`}
            >
              <GraduationCap className="w-4 h-4" />
              Training
            </button>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {currentView === 'trading' && (
            <>
              {/* Warmup Progress */}
              {ai.warmup && !ai.warmup.all_ready && (
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border bg-yellow-500/10 border-yellow-500/30">
                  <div className="w-16 h-2 bg-black/30 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-yellow-400 transition-all"
                      style={{ width: `${ai.warmup?.avg_progress || 0}%` }}
                    />
                  </div>
                  <span className="text-xs font-medium text-yellow-400">
                    {ai.warmup?.avg_bars || 0}/50 bars
                  </span>
                </div>
              )}

              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border ${
                ai.warmup?.all_ready
                  ? (ai.confidence >= 70 ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' :
                     ai.confidence >= 50 ? 'bg-yellow-500/10 border-yellow-500/30 text-yellow-400' :
                     'bg-gray-500/10 border-gray-500/30 text-gray-400')
                  : 'bg-orange-500/10 border-orange-500/30 text-orange-400'
              }`}>
                <Cpu className="w-4 h-4" />
                <span className="text-xs font-medium">
                  {ai.warmup?.all_ready ? `AI: ${ai.confidence}%` : 'Warming up...'}
                </span>
              </div>

              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border ${
                status === 'connected'
                  ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                  : 'bg-red-500/10 border-red-500/30 text-red-400'
              }`}>
                <div className={`w-2 h-2 rounded-full ${status === 'connected' ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`} />
                <span className="text-xs font-medium uppercase">{status}</span>
              </div>
            </>
          )}
        </div>
      </header>

      {/* Conditional Content */}
      {currentView === 'training' ? (
        <TrainingDashboard />
      ) : (
      <>
      {/* Main Grid - 3 Columns */}
      <div className="flex gap-4">

        {/* LEFT COLUMN */}
        <div className="w-1/4 space-y-4">
          {/* Grid Trading Status */}
          <div className={`glass-card rounded-xl p-5 border-2 ${grid.active ? 'border-emerald-500/30' : 'border-gray-500/30'}`}>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Grid3X3 className="w-4 h-4 text-emerald-400" />
                <span className="text-xs font-medium text-emerald-400 uppercase tracking-wider">Grid Trading</span>
              </div>
              <div className={`w-2 h-2 rounded-full ${grid.active ? 'bg-emerald-400 animate-pulse' : 'bg-gray-500'}`} />
            </div>

            {/* Status Summary */}
            <div className="grid grid-cols-2 gap-2 mb-4">
              <div className="bg-black/30 rounded p-2 text-center">
                <div className="text-[10px] text-gray-500">TRADES</div>
                <div className="text-lg font-bold text-white font-mono">{grid.total_trades || 0}</div>
              </div>
              <div className="bg-black/30 rounded p-2 text-center">
                <div className="text-[10px] text-gray-500">PROFIT</div>
                <div className={`text-lg font-bold font-mono ${(grid.total_profit || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  ${(grid.total_profit || 0).toFixed(2)}
                </div>
              </div>
            </div>

            {/* Mode indicator */}
            <div className={`flex items-center justify-center gap-3 p-3 rounded-lg ${
              grid.active ? 'bg-emerald-500/20 border border-emerald-500/30' : 'bg-gray-500/20 border border-gray-500/30'
            }`}>
              {grid.active ? <CheckCircle2 className="w-5 h-5 text-emerald-400" /> : <PauseCircle className="w-5 h-5 text-gray-400" />}
              <span className={`text-sm font-bold ${grid.active ? 'text-emerald-400' : 'text-gray-400'}`}>
                {grid.active ? 'ACTIVE' : 'INITIALIZING'}
              </span>
            </div>
          </div>

          {/* Risk Management Panel */}
          <div className={`glass-card rounded-xl p-5 border ${risk.trading_halted ? 'border-red-500/50' : 'border-yellow-500/30'}`}>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4 text-yellow-400" />
                <span className="text-xs font-medium text-yellow-400 uppercase">Risk Management</span>
              </div>
              {risk.trading_halted && <AlertOctagon className="w-4 h-4 text-red-400 animate-pulse" />}
            </div>

            {/* Daily P&L */}
            <div className="mb-3">
              <div className="flex justify-between text-xs mb-1">
                <span className="text-gray-500">Daily P&L</span>
                <span className={risk.daily_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                  ${risk.daily_pnl?.toFixed(2) || '0.00'} ({risk.daily_pnl_pct?.toFixed(2) || '0.00'}%)
                </span>
              </div>
              <div className="h-2 bg-black/30 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all ${risk.daily_pnl >= 0 ? 'bg-emerald-500' : 'bg-red-500'}`}
                  style={{ width: `${Math.min(Math.abs(risk.daily_pnl_pct || 0) * 20, 100)}%` }}
                />
              </div>
            </div>

            {/* Drawdown */}
            <div className="mb-3">
              <div className="flex justify-between text-xs mb-1">
                <span className="text-gray-500">Drawdown</span>
                <span className={risk.drawdown_pct > 5 ? 'text-red-400' : 'text-yellow-400'}>
                  {risk.drawdown_pct?.toFixed(2) || '0.00'}%
                </span>
              </div>
              <div className="h-2 bg-black/30 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all ${risk.drawdown_pct > 5 ? 'bg-red-500' : 'bg-yellow-500'}`}
                  style={{ width: `${Math.min((risk.drawdown_pct || 0) * 10, 100)}%` }}
                />
              </div>
            </div>

            {/* Circuit Breakers */}
            <div className="space-y-1 text-xs">
              <div className={`flex items-center justify-between p-2 rounded ${risk.daily_limit_hit ? 'bg-red-500/20' : 'bg-black/20'}`}>
                <span className="text-gray-400">Daily Limit (5%)</span>
                {risk.daily_limit_hit ?
                  <XCircle className="w-4 h-4 text-red-400" /> :
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                }
              </div>
              <div className={`flex items-center justify-between p-2 rounded ${risk.max_drawdown_hit ? 'bg-red-500/20' : 'bg-black/20'}`}>
                <span className="text-gray-400">Max Drawdown (10%)</span>
                {risk.max_drawdown_hit ?
                  <XCircle className="w-4 h-4 text-red-400" /> :
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                }
              </div>
            </div>

            {/* Trading Halted Warning */}
            {risk.trading_halted && (
              <div className="mt-3 p-2 bg-red-500/20 border border-red-500/30 rounded text-center">
                <div className="flex items-center justify-center gap-2">
                  <StopCircle className="w-4 h-4 text-red-400" />
                  <span className="text-xs font-bold text-red-400">TRADING HALTED</span>
                </div>
              </div>
            )}
          </div>

          {/* Grid Summaries */}
          <div className="glass-card rounded-xl p-5">
            <div className="flex items-center gap-2 mb-4">
              <Layers className="w-4 h-4 text-gray-400" />
              <span className="text-xs font-medium text-gray-400 uppercase">Grid Levels</span>
            </div>
            <div className="space-y-3 max-h-60 overflow-y-auto">
              {Object.entries(grid.summaries || {}).map(([symbol, summary]) => (
                <div key={symbol} className="bg-black/20 rounded p-2">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-white">{symbol}</span>
                    <span className="text-xs text-gray-500">{summary.range?.spacing_pct?.toFixed(2) || '--'}% spacing</span>
                  </div>
                  <div className="grid grid-cols-2 gap-1 text-xs">
                    <div className="text-gray-500">Range:</div>
                    <div className="text-gray-300 text-right font-mono">
                      ${summary.range?.lower?.toLocaleString() || '--'} - ${summary.range?.upper?.toLocaleString() || '--'}
                    </div>
                    <div className="text-gray-500">Pending Buys:</div>
                    <div className="text-emerald-400 text-right font-mono">{summary.levels?.pending_buys || 0}</div>
                    <div className="text-gray-500">Pending Sells:</div>
                    <div className="text-red-400 text-right font-mono">{summary.levels?.pending_sells || 0}</div>
                  </div>
                </div>
              ))}
              {Object.keys(grid.summaries || {}).length === 0 && (
                <div className="text-xs text-gray-500 text-center py-4">Initializing grids...</div>
              )}
            </div>
          </div>
        </div>

        {/* CENTER COLUMN */}
        <div className="w-1/2 space-y-4">
          {/* Stats */}
          <div className="grid grid-cols-4 gap-3">
            <div className="glass-card rounded-xl p-4">
              <div className="text-xs text-gray-500 mb-1">{data.symbol || 'BTC/USD'}</div>
              <div className="text-xl font-bold text-white font-mono">${price.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
            </div>
            <div className="glass-card rounded-xl p-4">
              <div className="text-xs text-gray-500 mb-1">Equity</div>
              <div className="text-xl font-bold text-white font-mono">${account.equity.toLocaleString()}</div>
            </div>
            <div className="glass-card rounded-xl p-4">
              <div className="text-xs text-gray-500 mb-1">AI Confidence</div>
              <div className={`text-xl font-bold font-mono ${ai.confidence >= 70 ? 'text-emerald-400' : 'text-gray-400'}`}>{ai.confidence}%</div>
            </div>
            <div className="glass-card rounded-xl p-4">
              <div className="text-xs text-gray-500 mb-1">Positions</div>
              <div className="text-xl font-bold text-white font-mono">{positions.length}/{data.multi_asset?.symbols?.length || 4}</div>
            </div>
          </div>

          {/* Chart */}
          <div className="glass-card rounded-xl p-5 h-64">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-sm font-semibold text-white">Price Action</h2>
              <div className={`px-2 py-1 rounded text-xs font-medium ${
                ai.signal === 'BUY' ? 'bg-emerald-500/20 text-emerald-400' :
                ai.signal === 'SELL' ? 'bg-red-500/20 text-red-400' :
                'bg-gray-500/20 text-gray-400'
              }`}>{ai.signal}</div>
            </div>
            <div className="h-44">
              <Line data={chartData} options={chartOptions} />
            </div>
          </div>

          {/* Last Trade */}
          {lastTrade && (
            <div className={`glass-card rounded-xl p-5 border ${lastTrade.action === 'BUY' ? 'border-emerald-500/30' : 'border-red-500/30'}`}>
              <div className="flex items-center gap-2 mb-3">
                {lastTrade.action === 'BUY' ? <ArrowUpCircle className="w-5 h-5 text-emerald-400" /> : <ArrowDownCircle className="w-5 h-5 text-red-400" />}
                <span className="text-sm font-semibold text-white">LAST TRADE: {lastTrade.action} {lastTrade.symbol}</span>
              </div>
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div>
                  <div className="text-gray-500 mb-1">Why AI {lastTrade.action === 'BUY' ? 'bought' : 'sold'}:</div>
                  {lastTrade.reasoning?.slice(0, 3).map((r, i) => (
                    <div key={i} className="text-gray-300">{i + 1}. {r}</div>
                  ))}
                </div>
                {lastTrade.stop_loss && (
                  <div>
                    <div className="text-gray-500 mb-1">Targets:</div>
                    <div className="text-red-400">SL: ${lastTrade.stop_loss?.toLocaleString()}</div>
                    <div className="text-emerald-400">TP: ${lastTrade.take_profit?.toLocaleString()}</div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Positions */}
          <div className="glass-card rounded-xl p-5">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-sm font-semibold text-white">Positions</h2>
              <span className="text-xs text-gray-500">Size: 3-5%</span>
            </div>
            {positions.length === 0 ? (
              <div className="text-center py-6 text-gray-500">
                <PauseCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No active positions</p>
              </div>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-xs text-gray-500 border-b border-white/5">
                    <th className="text-left pb-2">Symbol</th>
                    <th className="text-right pb-2">Entry</th>
                    <th className="text-right pb-2">Current</th>
                    <th className="text-right pb-2">PnL</th>
                  </tr>
                </thead>
                <tbody>
                  {positions.map((p, i) => (
                    <tr key={i} className="border-b border-white/5">
                      <td className="py-2 text-white">{p.symbol}</td>
                      <td className="py-2 text-right font-mono text-gray-300">${p.avg_entry_price?.toFixed(2)}</td>
                      <td className="py-2 text-right font-mono text-gray-300">${p.current_price?.toFixed(2)}</td>
                      <td className={`py-2 text-right font-mono ${p.unrealized_pl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        ${p.unrealized_pl?.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {/* RIGHT COLUMN */}
        <div className="w-1/4 space-y-4">
          {/* Multi-Asset Signals */}
          {data.multi_asset && (
            <div className="glass-card rounded-xl p-5 border border-indigo-500/30">
              <div className="flex items-center gap-2 mb-4">
                <Layers className="w-4 h-4 text-indigo-400" />
                <span className="text-xs font-medium text-indigo-400 uppercase">Multi-Asset</span>
              </div>
              <div className="space-y-2">
                {data.multi_asset.symbols?.map(sym => {
                  const sig = data.multi_asset.signals?.[sym] || 'HOLD';
                  const conf = data.multi_asset.confidences?.[sym] || 0;
                  const active = sym === data.multi_asset.active_symbol;
                  return (
                    <div key={sym} className={`flex items-center justify-between p-2 rounded ${active ? 'bg-indigo-500/10 border border-indigo-500/30' : 'bg-black/20'}`}>
                      <div className="flex items-center gap-2">
                        {active && <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-pulse" />}
                        <span className={active ? 'text-white' : 'text-gray-400'}>{sym}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-500">{conf}%</span>
                        <span className={`text-xs font-bold px-2 py-1 rounded ${
                          sig === 'BUY' ? 'bg-emerald-500/20 text-emerald-400' :
                          sig === 'SELL' ? 'bg-red-500/20 text-red-400' :
                          'bg-gray-500/20 text-gray-400'
                        }`}>{sig}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Live Indicators - Expanded */}
          <div className="glass-card rounded-xl p-5">
            <div className="flex items-center gap-2 mb-4">
              <Activity className="w-4 h-4 text-gray-400" />
              <span className="text-xs font-medium text-gray-400 uppercase">AI Indicators</span>
            </div>

            {/* Core Momentum */}
            <div className="mb-3">
              <div className="text-[10px] text-gray-500 mb-1 uppercase">Momentum</div>
              <div className="grid grid-cols-2 gap-1 text-xs">
                <div className="bg-black/30 rounded p-1.5">
                  <div className="text-gray-500 text-[10px]">RSI</div>
                  <div className={`font-mono font-bold ${
                    (ai.features?.rsi || 50) < 30 ? 'text-emerald-400' :
                    (ai.features?.rsi || 50) > 70 ? 'text-red-400' : 'text-gray-300'
                  }`}>{ai.features?.rsi?.toFixed(1) || '--'}</div>
                </div>
                <div className="bg-black/30 rounded p-1.5">
                  <div className="text-gray-500 text-[10px]">MACD</div>
                  <div className={`font-mono font-bold ${(ai.features?.macd_hist || 0) > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {ai.features?.macd_hist?.toFixed(3) || '--'}
                  </div>
                </div>
                <div className="bg-black/30 rounded p-1.5">
                  <div className="text-gray-500 text-[10px]">Stoch K/D</div>
                  <div className={`font-mono font-bold ${
                    (ai.features?.stoch_k || 50) < 20 ? 'text-emerald-400' :
                    (ai.features?.stoch_k || 50) > 80 ? 'text-red-400' : 'text-gray-300'
                  }`}>{ai.features?.stoch_k?.toFixed(0) || '--'}/{ai.features?.stoch_d?.toFixed(0) || '--'}</div>
                </div>
                <div className="bg-black/30 rounded p-1.5">
                  <div className="text-gray-500 text-[10px]">MFI</div>
                  <div className={`font-mono font-bold ${
                    (ai.features?.mfi || 50) < 20 ? 'text-emerald-400' :
                    (ai.features?.mfi || 50) > 80 ? 'text-red-400' : 'text-gray-300'
                  }`}>{ai.features?.mfi?.toFixed(0) || '--'}</div>
                </div>
              </div>
            </div>

            {/* Oscillators */}
            <div className="mb-3">
              <div className="text-[10px] text-gray-500 mb-1 uppercase">Oscillators</div>
              <div className="grid grid-cols-2 gap-1 text-xs">
                <div className="bg-black/30 rounded p-1.5">
                  <div className="text-gray-500 text-[10px]">CCI</div>
                  <div className={`font-mono font-bold ${
                    (ai.features?.cci || 0) < -100 ? 'text-emerald-400' :
                    (ai.features?.cci || 0) > 100 ? 'text-red-400' : 'text-gray-300'
                  }`}>{ai.features?.cci?.toFixed(0) || '--'}</div>
                </div>
                <div className="bg-black/30 rounded p-1.5">
                  <div className="text-gray-500 text-[10px]">Will %R</div>
                  <div className={`font-mono font-bold ${
                    (ai.features?.willr || -50) < -80 ? 'text-emerald-400' :
                    (ai.features?.willr || -50) > -20 ? 'text-red-400' : 'text-gray-300'
                  }`}>{ai.features?.willr?.toFixed(0) || '--'}</div>
                </div>
              </div>
            </div>

            {/* Trend */}
            <div className="mb-3">
              <div className="text-[10px] text-gray-500 mb-1 uppercase">Trend</div>
              <div className="grid grid-cols-2 gap-1 text-xs">
                <div className="bg-black/30 rounded p-1.5">
                  <div className="text-gray-500 text-[10px]">ADX</div>
                  <div className={`font-mono font-bold ${(ai.features?.adx || 0) > 25 ? 'text-indigo-400' : 'text-gray-300'}`}>
                    {ai.features?.adx?.toFixed(0) || '--'}
                  </div>
                </div>
                <div className="bg-black/30 rounded p-1.5">
                  <div className="text-gray-500 text-[10px]">Trend Align</div>
                  <div className={`font-mono font-bold ${
                    (ai.features?.trend_alignment || 0) >= 2 ? 'text-emerald-400' :
                    (ai.features?.trend_alignment || 0) <= 1 ? 'text-red-400' : 'text-gray-300'
                  }`}>{ai.features?.trend_alignment?.toFixed(0) || '--'}/3</div>
                </div>
                <div className="bg-black/30 rounded p-1.5">
                  <div className="text-gray-500 text-[10px]">ROC 5</div>
                  <div className={`font-mono font-bold ${(ai.features?.roc_5 || 0) > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {ai.features?.roc_5?.toFixed(2) || '--'}%
                  </div>
                </div>
                <div className="bg-black/30 rounded p-1.5">
                  <div className="text-gray-500 text-[10px]">Mom Align</div>
                  <div className={`font-mono font-bold ${
                    (ai.features?.momentum_alignment || 0) > 0 ? 'text-emerald-400' :
                    (ai.features?.momentum_alignment || 0) < 0 ? 'text-red-400' : 'text-gray-300'
                  }`}>{ai.features?.momentum_alignment || '--'}</div>
                </div>
              </div>
            </div>

            {/* Volume & Volatility */}
            <div className="mb-3">
              <div className="text-[10px] text-gray-500 mb-1 uppercase">Volume & Volatility</div>
              <div className="grid grid-cols-2 gap-1 text-xs">
                <div className="bg-black/30 rounded p-1.5">
                  <div className="text-gray-500 text-[10px]">Volume</div>
                  <div className={`font-mono font-bold ${(ai.features?.volume_ratio || 1) > 1.5 ? 'text-emerald-400' : 'text-gray-300'}`}>
                    {ai.features?.volume_ratio?.toFixed(2) || '1.00'}x
                  </div>
                </div>
                <div className="bg-black/30 rounded p-1.5">
                  <div className="text-gray-500 text-[10px]">Vol Spike</div>
                  <div className={`font-mono font-bold ${ai.features?.volume_spike ? 'text-yellow-400' : 'text-gray-500'}`}>
                    {ai.features?.volume_spike ? 'YES' : 'NO'}
                  </div>
                </div>
                <div className="bg-black/30 rounded p-1.5">
                  <div className="text-gray-500 text-[10px]">BB Squeeze</div>
                  <div className={`font-mono font-bold ${ai.features?.bb_squeeze ? 'text-yellow-400' : 'text-gray-500'}`}>
                    {ai.features?.bb_squeeze ? 'YES' : 'NO'}
                  </div>
                </div>
                <div className="bg-black/30 rounded p-1.5">
                  <div className="text-gray-500 text-[10px]">Vol Regime</div>
                  <div className={`font-mono font-bold ${(ai.features?.volatility_regime || 1) > 1.5 ? 'text-yellow-400' : 'text-gray-300'}`}>
                    {ai.features?.volatility_regime?.toFixed(2) || '1.00'}
                  </div>
                </div>
              </div>
            </div>

            {/* Price Action */}
            <div>
              <div className="text-[10px] text-gray-500 mb-1 uppercase">Price Action</div>
              <div className="grid grid-cols-2 gap-1 text-xs">
                <div className="bg-black/30 rounded p-1.5">
                  <div className="text-gray-500 text-[10px]">BB Pos</div>
                  <div className={`font-mono font-bold ${
                    (ai.features?.bb_position || 0.5) < 0.2 ? 'text-emerald-400' :
                    (ai.features?.bb_position || 0.5) > 0.8 ? 'text-red-400' : 'text-gray-300'
                  }`}>{((ai.features?.bb_position || 0.5) * 100).toFixed(0)}%</div>
                </div>
                <div className="bg-black/30 rounded p-1.5">
                  <div className="text-gray-500 text-[10px]">Price Pos</div>
                  <div className="font-mono font-bold text-gray-300">
                    {((ai.features?.price_position || 0.5) * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="bg-black/30 rounded p-1.5">
                  <div className="text-gray-500 text-[10px]">Breakout ↑</div>
                  <div className={`font-mono font-bold ${ai.features?.breakout_up ? 'text-emerald-400' : 'text-gray-500'}`}>
                    {ai.features?.breakout_up ? 'YES' : 'NO'}
                  </div>
                </div>
                <div className="bg-black/30 rounded p-1.5">
                  <div className="text-gray-500 text-[10px]">Breakout ↓</div>
                  <div className={`font-mono font-bold ${ai.features?.breakout_down ? 'text-red-400' : 'text-gray-500'}`}>
                    {ai.features?.breakout_down ? 'YES' : 'NO'}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Kelly Sizing */}
          <div className="glass-card rounded-xl p-5">
            <div className="flex items-center gap-2 mb-4">
              <Gauge className="w-4 h-4 text-gray-400" />
              <span className="text-xs font-medium text-gray-400 uppercase">Position Sizing</span>
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-500">Kelly</span>
                <span className="text-indigo-400">{((ultra.kelly?.kelly_fraction || 0.25) * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Win Rate</span>
                <span className="text-white">{((ultra.kelly?.win_rate || 0.5) * 100).toFixed(0)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Trades</span>
                <span className="text-white">{ultra.kelly?.sample_size || 0}</span>
              </div>
              <div className="pt-2 border-t border-white/5">
                <div className="text-gray-500 mb-1">Size (based on {ai.confidence}% conf)</div>
                <div className="text-indigo-400 font-mono">{(3 + ((ai.confidence - 70) / 30) * 2).toFixed(1)}%</div>
              </div>
            </div>
          </div>

          {/* Logs */}
          <div className="glass-card rounded-xl p-5 flex flex-col h-64">
            <div className="flex items-center gap-2 mb-3">
              <Terminal className="w-4 h-4 text-gray-400" />
              <span className="text-xs font-medium text-gray-400 uppercase">Logs</span>
            </div>
            <div className="flex-1 overflow-y-auto space-y-1 pr-2 custom-scrollbar">
              {logs.map((log, i) => (
                <div key={i} className="text-[11px] font-mono py-1 px-2 rounded bg-black/30 break-all">
                  <span className="text-gray-600">{log.split(']')[0]}]</span>
                  <span className={
                    log.includes('BUY') ? 'text-emerald-400' :
                    log.includes('SELL') ? 'text-red-400' :
                    log.includes('Error') ? 'text-red-400' :
                    'text-gray-400'
                  }>{log.split(']')[1]}</span>
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-4 flex items-center justify-between px-4 py-2 glass-card rounded-lg text-xs">
        <div className="flex items-center gap-6">
          <span className="text-gray-500">Mode: <span className="text-emerald-400">Grid Trading</span></span>
          <span className="text-gray-500">Daily Limit: <span className="text-yellow-400">5%</span></span>
          <span className="text-gray-500">Max Drawdown: <span className="text-red-400">10%</span></span>
          <span className="text-gray-500">Stop Loss: <span className="text-red-400">10% below grid</span></span>
        </div>
        <div className="text-gray-500">
          BlueBird Grid v4.0 • {Object.keys(grid.summaries || {}).length || 4} Assets • Buy Dips, Sell Rips
        </div>
      </footer>
      </>
      )}
    </div>
  );
}

export default App;
