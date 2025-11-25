import React, { useState, useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Database, Cpu, Target, TrendingDown, BarChart3,
  Play, Settings, CheckCircle2, Circle, Loader2,
  Brain, Zap, GitBranch, Activity, Coins, Shield, AlertTriangle
} from 'lucide-react';

const TRAINING_WS_URL = 'ws://localhost:8001/ws';
const TRAINING_API_URL = 'http://localhost:8001/api/state';

function TrainingDashboard() {
  const [connected, setConnected] = useState(false);
  const [state, setState] = useState({
    status: 'idle',
    progress: 0,
    current_horizon: null,
    horizons: [5, 15, 30],
    current_symbol: null,
    symbols: [],
    symbol_idx: 0,
    symbol_metrics: {},
    data: { rows: 0, days: 0, features: 0 },
    tuning: { iteration: 0, total: 0, best_score: null, history: [] },
    training: { round: 0, total: 200, train_loss: [], val_loss: [] },
    walk_forward: { fold: 0, total: 5, results: [] },
    metrics: {},
    holdout_metrics: {},
    feature_importance: {},
    logs: [],
    error: null
  });

  // Form state
  const [days, setDays] = useState(90);
  const [holdoutDays, setHoldoutDays] = useState(14);
  const [tuneIterations, setTuneIterations] = useState(15);
  const [enableTuning, setEnableTuning] = useState(true);

  const ws = useRef(null);
  const logsEndRef = useRef(null);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (ws.current) ws.current.close();
    };
  }, []);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [state.logs]);

  const connectWebSocket = () => {
    ws.current = new WebSocket(TRAINING_WS_URL);

    ws.current.onopen = async () => {
      setConnected(true);
      // Fetch current state via REST as fallback (in case we missed WebSocket updates)
      try {
        const response = await fetch(TRAINING_API_URL);
        const msg = await response.json();
        if (msg.type === 'update' && msg.data) {
          setState(msg.data);
        }
      } catch (err) {
        console.log('Could not fetch initial state via REST:', err);
      }
    };

    ws.current.onclose = () => {
      setConnected(false);
      setTimeout(connectWebSocket, 3000);
    };

    ws.current.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === 'update') {
        setState(msg.data);
      }
    };

    ws.current.onerror = () => {
      setConnected(false);
    };
  };

  const startTraining = () => {
    if (holdoutDays >= days) {
      alert('Holdout days must be less than total days!');
      return;
    }
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({
        action: 'start_training',
        days: days,
        holdout_days: holdoutDays,
        tune: enableTuning,
        tune_iterations: tuneIterations
      }));
    }
  };

  const isTraining = ['fetching_data', 'calculating_features', 'tuning', 'training'].includes(state.status);

  // Chart data
  const lossChartData = {
    labels: state.training.train_loss.map((_, i) => i + 1),
    datasets: [
      {
        label: 'Train Loss',
        data: state.training.train_loss,
        borderColor: '#6366f1',
        backgroundColor: 'rgba(99, 102, 241, 0.1)',
        tension: 0.4,
        pointRadius: 0,
        fill: true
      },
      {
        label: 'Val Loss',
        data: state.training.val_loss,
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
        tension: 0.4,
        pointRadius: 0,
        fill: true
      }
    ]
  };

  const tuneChartData = {
    labels: state.tuning.history.map(h => h.iteration),
    datasets: [{
      label: 'Score',
      data: state.tuning.history.map(h => h.score),
      borderColor: '#10b981',
      backgroundColor: 'rgba(16, 185, 129, 0.1)',
      tension: 0.4,
      pointRadius: 2,
      fill: true
    }]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#9ca3af', font: { size: 10 } },
        position: 'top'
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255,255,255,0.03)' },
        ticks: { color: '#6b7280', font: { size: 10 } }
      },
      y: {
        grid: { color: 'rgba(255,255,255,0.03)' },
        ticks: { color: '#6b7280', font: { size: 10 } }
      }
    }
  };

  const getStatusText = () => {
    const symbol = state.current_symbol || '';
    const symbolInfo = symbol ? ` [${symbol}]` : '';
    const statusMap = {
      'idle': 'Ready to train',
      'fetching_data': `Fetching data for ${symbol || 'assets'}...`,
      'calculating_features': `Calculating features${symbolInfo}...`,
      'tuning': `Tuning ${state.current_horizon}-min model${symbolInfo}...`,
      'training': `Training ${state.current_horizon}-min model${symbolInfo}...`,
      'complete': `Training complete! Trained ${Object.keys(state.symbol_metrics).length} assets`,
      'error': `Error: ${state.error}`
    };
    return statusMap[state.status] || state.status;
  };

  return (
    <div className="space-y-4">
      {/* Connection Status + Controls */}
      <div className="glass-card rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-lg border border-purple-500/30">
              <Brain className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">Model Training</h2>
              <p className="text-xs text-gray-500">LightGBM Multi-Horizon Training</p>
            </div>
          </div>

          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border ${
            connected
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
              : 'bg-red-500/10 border-red-500/30 text-red-400'
          }`}>
            <div className={`w-2 h-2 rounded-full ${connected ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`} />
            <span className="text-xs font-medium">{connected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>

        {!connected && (
          <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4 mb-4">
            <p className="text-yellow-400 text-sm">
              Training server not running. Start it with:
            </p>
            <code className="text-xs text-gray-400 bg-black/30 px-2 py-1 rounded mt-2 block">
              python train_with_dashboard.py
            </code>
          </div>
        )}

        {/* Controls */}
        <div className="flex items-end gap-4 flex-wrap">
          <div>
            <label className="text-xs text-gray-500 block mb-1">Total Days</label>
            <input
              type="number"
              min="1"
              value={days}
              onChange={(e) => setDays(e.target.value === '' ? '' : parseInt(e.target.value))}
              onBlur={(e) => setDays(parseInt(e.target.value) || 90)}
              disabled={isTraining}
              className="bg-black/30 border border-gray-700 rounded px-3 py-2 w-24 text-white text-sm disabled:opacity-50"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">Holdout Days</label>
            <input
              type="number"
              min="1"
              value={holdoutDays}
              onChange={(e) => setHoldoutDays(e.target.value === '' ? '' : parseInt(e.target.value))}
              onBlur={(e) => setHoldoutDays(parseInt(e.target.value) || 14)}
              disabled={isTraining}
              className="bg-black/30 border border-amber-500/30 rounded px-3 py-2 w-20 text-amber-300 text-sm disabled:opacity-50"
              title="Days reserved for true out-of-sample testing"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">Tune Iterations</label>
            <input
              type="number"
              min="1"
              value={tuneIterations}
              onChange={(e) => setTuneIterations(e.target.value === '' ? '' : parseInt(e.target.value))}
              onBlur={(e) => setTuneIterations(parseInt(e.target.value) || 15)}
              disabled={isTraining}
              className="bg-black/30 border border-gray-700 rounded px-3 py-2 w-24 text-white text-sm disabled:opacity-50"
            />
          </div>
          <div className="flex items-center gap-2 pb-2">
            <input
              type="checkbox"
              checked={enableTuning}
              onChange={(e) => setEnableTuning(e.target.checked)}
              disabled={isTraining}
              className="w-4 h-4 rounded"
            />
            <label className="text-sm text-gray-400">Hyperparameter Tuning</label>
          </div>
          <button
            onClick={startTraining}
            disabled={!connected || isTraining}
            className={`flex items-center gap-2 px-6 py-2 rounded-lg font-medium transition ${
              !connected || isTraining
                ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                : 'bg-indigo-600 hover:bg-indigo-700 text-white'
            }`}
          >
            {isTraining ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Training...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Start Training
              </>
            )}
          </button>
        </div>

        {/* Data Split Info */}
        <div className="mt-3 flex items-center gap-4 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-emerald-500"></div>
            <span className="text-gray-400">Training: <span className="text-emerald-400 font-mono">{days - holdoutDays}</span> days</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-amber-500"></div>
            <span className="text-gray-400">Holdout: <span className="text-amber-400 font-mono">{holdoutDays}</span> days</span>
          </div>
          <span className="text-gray-600">|</span>
          <span className="text-gray-500 italic">Holdout = true out-of-sample test (never seen during training)</span>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="glass-card rounded-xl p-5">
        <div className="flex justify-between items-center mb-3">
          <span className={`text-sm ${state.status === 'error' ? 'text-red-400' : 'text-gray-400'}`}>
            {getStatusText()}
          </span>
          <span className="text-sm text-indigo-400 font-mono">{state.progress}%</span>
        </div>
        <div className="h-3 bg-black/30 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-300 ${
              state.status === 'complete' ? 'bg-emerald-500' :
              state.status === 'error' ? 'bg-red-500' :
              'bg-gradient-to-r from-indigo-500 to-purple-500'
            }`}
            style={{ width: `${state.progress}%` }}
          />
        </div>

        {/* Symbol badges */}
        {state.symbols && state.symbols.length > 0 && (
          <div className="mt-4">
            <div className="flex items-center gap-2 mb-2">
              <Coins className="w-4 h-4 text-yellow-400" />
              <span className="text-xs text-gray-500 uppercase">Assets</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {state.symbols.map((sym, idx) => {
                const done = state.symbol_metrics && state.symbol_metrics[sym];
                const current = sym === state.current_symbol;
                return (
                  <span
                    key={sym}
                    className={`flex items-center gap-1 px-3 py-1.5 rounded-lg border text-xs font-medium ${
                      done ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-400' :
                      current ? 'bg-indigo-500/20 border-indigo-500/30 text-indigo-400 animate-pulse' :
                      'bg-gray-500/10 border-gray-500/30 text-gray-500'
                    }`}
                  >
                    {done ? <CheckCircle2 className="w-3 h-3" /> :
                     current ? <Loader2 className="w-3 h-3 animate-spin" /> :
                     <Circle className="w-3 h-3" />}
                    {sym}
                  </span>
                );
              })}
            </div>
          </div>
        )}

        {/* Horizon badges */}
        <div className="flex gap-2 mt-4">
          <span className="text-xs text-gray-500 mr-2">Horizons:</span>
          {state.horizons.map(h => {
            const done = state.metrics[h];
            const current = h === state.current_horizon;
            return (
              <span
                key={h}
                className={`flex items-center gap-1 px-3 py-1 rounded border text-xs ${
                  done ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-400' :
                  current ? 'bg-indigo-500/20 border-indigo-500/30 text-indigo-400' :
                  'bg-gray-500/20 border-gray-500/30 text-gray-400'
                }`}
              >
                {done ? <CheckCircle2 className="w-3 h-3" /> :
                 current ? <Loader2 className="w-3 h-3 animate-spin" /> :
                 <Circle className="w-3 h-3" />}
                {h}min
              </span>
            );
          })}
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-3 gap-4">
        {/* Left Column */}
        <div className="space-y-4">
          {/* Data Info */}
          <div className="glass-card rounded-xl p-5">
            <div className="flex items-center gap-2 mb-4">
              <Database className="w-4 h-4 text-blue-400" />
              <span className="text-xs font-medium text-blue-400 uppercase">Training Data</span>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-500 text-sm">Samples</span>
                <span className="text-white font-mono">{state.data.rows.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500 text-sm">Days</span>
                <span className="text-white font-mono">{state.data.days}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500 text-sm">Features</span>
                <span className="text-white font-mono">{state.data.features}</span>
              </div>
            </div>
          </div>

          {/* Tuning Progress */}
          <div className="glass-card rounded-xl p-5">
            <div className="flex items-center gap-2 mb-4">
              <Settings className="w-4 h-4 text-emerald-400" />
              <span className="text-xs font-medium text-emerald-400 uppercase">Hyperparameter Tuning</span>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-500 text-sm">Iteration</span>
                <span className="text-white font-mono">{state.tuning.iteration}/{state.tuning.total}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500 text-sm">Best Score</span>
                <span className="text-emerald-400 font-mono">{state.tuning.best_score || '-'}</span>
              </div>
            </div>
            <div className="mt-4 h-2 bg-black/30 rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-500 transition-all"
                style={{ width: `${state.tuning.total > 0 ? (state.tuning.iteration / state.tuning.total * 100) : 0}%` }}
              />
            </div>
          </div>

          {/* Walk-Forward Validation */}
          <div className="glass-card rounded-xl p-5">
            <div className="flex items-center gap-2 mb-4">
              <GitBranch className="w-4 h-4 text-yellow-400" />
              <span className="text-xs font-medium text-yellow-400 uppercase">Walk-Forward Validation</span>
            </div>
            <div className="space-y-2">
              {state.walk_forward.results.length > 0 ? (
                state.walk_forward.results.map(r => (
                  <div key={r.fold} className="flex justify-between items-center p-2 bg-black/20 rounded text-sm">
                    <span className="text-gray-400">Fold {r.fold}</span>
                    <span className="text-emerald-400 font-mono">{r.accuracy}%</span>
                  </div>
                ))
              ) : (
                <div className="text-gray-500 text-sm text-center py-4">No results yet</div>
              )}
            </div>
          </div>
        </div>

        {/* Center Column */}
        <div className="space-y-4">
          {/* Loss Chart */}
          <div className="glass-card rounded-xl p-5">
            <div className="flex items-center gap-2 mb-4">
              <TrendingDown className="w-4 h-4 text-indigo-400" />
              <span className="text-xs font-medium text-indigo-400 uppercase">Training Loss</span>
            </div>
            <div className="h-48">
              {state.training.train_loss.length > 0 ? (
                <Line data={lossChartData} options={chartOptions} />
              ) : (
                <div className="h-full flex items-center justify-center text-gray-500 text-sm">
                  Training will show loss curves here
                </div>
              )}
            </div>
          </div>

          {/* Tuning History Chart */}
          <div className="glass-card rounded-xl p-5">
            <div className="flex items-center gap-2 mb-4">
              <Activity className="w-4 h-4 text-emerald-400" />
              <span className="text-xs font-medium text-emerald-400 uppercase">Tuning History</span>
            </div>
            <div className="h-40">
              {state.tuning.history.length > 0 ? (
                <Line data={tuneChartData} options={chartOptions} />
              ) : (
                <div className="h-full flex items-center justify-center text-gray-500 text-sm">
                  Tuning scores will appear here
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="space-y-4">
          {/* Model Metrics (Training) */}
          <div className="glass-card rounded-xl p-5 max-h-64 overflow-y-auto">
            <div className="flex items-center gap-2 mb-4">
              <Target className="w-4 h-4 text-purple-400" />
              <span className="text-xs font-medium text-purple-400 uppercase">Training Metrics</span>
              <span className="text-xs text-gray-500">(may be optimistic)</span>
            </div>
            <div className="space-y-3">
              {/* Show per-symbol metrics if training complete */}
              {Object.keys(state.symbol_metrics || {}).length > 0 ? (
                Object.entries(state.symbol_metrics).map(([symbol, symMetrics]) => (
                  <div key={symbol} className="p-3 bg-black/20 rounded border border-gray-700/50">
                    <div className="flex items-center gap-2 mb-2">
                      <Coins className="w-3 h-3 text-yellow-400" />
                      <span className="text-yellow-400 font-semibold text-sm">{symbol}</span>
                    </div>
                    {Object.entries(symMetrics).map(([horizon, m]) => (
                      <div key={horizon} className="ml-4 mb-2 p-2 bg-black/20 rounded">
                        <div className="text-indigo-400 text-xs mb-1">{horizon}-min</div>
                        <div className="flex gap-3 text-xs">
                          <span className="text-gray-400">Acc: <span className="text-white">{m.accuracy}%</span></span>
                          <span className="text-gray-400">Buy: <span className="text-emerald-400">{m.buy_precision}%</span></span>
                        </div>
                      </div>
                    ))}
                  </div>
                ))
              ) : Object.keys(state.metrics).length > 0 ? (
                // Show current training metrics
                Object.entries(state.metrics).map(([horizon, m]) => (
                  <div key={horizon} className="p-3 bg-black/20 rounded">
                    <div className="text-indigo-400 font-semibold text-sm mb-2">
                      {state.current_symbol && `${state.current_symbol} - `}{horizon}-min Model
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      <div>
                        <span className="text-gray-500 block">Accuracy</span>
                        <span className="text-white font-mono">{m.accuracy}%</span>
                      </div>
                      <div>
                        <span className="text-gray-500 block">Confident</span>
                        <span className="text-emerald-400 font-mono">{m.confident_accuracy}%</span>
                      </div>
                      <div>
                        <span className="text-gray-500 block">Buy Prec</span>
                        <span className="text-yellow-400 font-mono">{m.buy_precision}%</span>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-gray-500 text-sm text-center py-4">
                  Metrics will appear after training
                </div>
              )}
            </div>
          </div>

          {/* HOLDOUT Metrics (TRUE Performance) */}
          <div className="glass-card rounded-xl p-5 max-h-64 overflow-y-auto border border-amber-500/30">
            <div className="flex items-center gap-2 mb-4">
              <Shield className="w-4 h-4 text-amber-400" />
              <span className="text-xs font-medium text-amber-400 uppercase">Holdout Metrics</span>
              <span className="text-xs text-gray-500">(TRUE performance)</span>
            </div>
            <div className="space-y-3">
              {Object.keys(state.holdout_metrics || {}).length > 0 ? (
                Object.entries(state.holdout_metrics).map(([symbol, symMetrics]) => (
                  <div key={symbol} className="p-3 bg-black/20 rounded border border-amber-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Coins className="w-3 h-3 text-amber-400" />
                      <span className="text-amber-400 font-semibold text-sm">{symbol}</span>
                    </div>
                    {Object.entries(symMetrics).map(([horizon, m]) => (
                      <div key={horizon} className="ml-4 mb-2 p-2 bg-black/20 rounded">
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-amber-400 text-xs font-semibold">{horizon}-min</span>
                          {m.optimal_threshold && (
                            <span className="text-xs text-gray-500">@{m.optimal_threshold}</span>
                          )}
                        </div>
                        <div className="grid grid-cols-2 gap-1 text-xs">
                          <span className="text-gray-400">Precision: <span className={`font-mono ${m.buy_precision > 50 ? 'text-emerald-400' : m.buy_precision > 30 ? 'text-amber-300' : 'text-red-400'}`}>{m.buy_precision}%</span></span>
                          <span className="text-gray-400">Recall: <span className="text-amber-300 font-mono">{m.recall || 0}%</span></span>
                          <span className="text-gray-400">Signals/day: <span className="text-cyan-400 font-mono">{m.signals_per_day || 0}</span></span>
                          <span className="text-gray-400">Acc: <span className="text-gray-300 font-mono">{m.accuracy}%</span></span>
                        </div>
                      </div>
                    ))}
                  </div>
                ))
              ) : (
                <div className="text-gray-500 text-sm text-center py-4 flex flex-col items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-gray-600" />
                  <span>Holdout metrics appear after training</span>
                  <span className="text-xs text-gray-600">This shows TRUE out-of-sample performance</span>
                </div>
              )}
            </div>
          </div>

          {/* Feature Importance */}
          <div className="glass-card rounded-xl p-5">
            <div className="flex items-center gap-2 mb-4">
              <BarChart3 className="w-4 h-4 text-pink-400" />
              <span className="text-xs font-medium text-pink-400 uppercase">Feature Importance</span>
            </div>
            <div className="space-y-2">
              {(() => {
                const horizon = state.current_horizon || Object.keys(state.feature_importance)[0];
                const importance = state.feature_importance[horizon] || {};
                const sorted = Object.entries(importance).sort((a, b) => b[1] - a[1]).slice(0, 6);

                return sorted.length > 0 ? sorted.map(([feature, weight]) => (
                  <div key={feature}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-gray-400">{feature}</span>
                      <span className="text-gray-300">{(weight * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-2 bg-black/30 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-pink-500 to-purple-500"
                        style={{ width: `${weight * 100}%` }}
                      />
                    </div>
                  </div>
                )) : (
                  <div className="text-gray-500 text-sm text-center py-4">
                    Feature weights will appear after training
                  </div>
                );
              })()}
            </div>
          </div>

          {/* Logs */}
          <div className="glass-card rounded-xl p-5 max-h-48 flex flex-col">
            <div className="flex items-center gap-2 mb-3">
              <Cpu className="w-4 h-4 text-gray-400" />
              <span className="text-xs font-medium text-gray-400 uppercase">Logs</span>
            </div>
            <div className="flex-1 overflow-y-auto space-y-1 text-xs font-mono">
              {state.logs.slice(-15).map((log, i) => (
                <div key={i} className="text-gray-400 py-0.5">{log}</div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default TrainingDashboard;
