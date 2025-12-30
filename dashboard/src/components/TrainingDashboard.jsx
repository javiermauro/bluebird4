import React, { useState, useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';

// ═══════════════════════════════════════════════════════════════════════════
// TRAINING SUITE - Model Training Interface
// ═══════════════════════════════════════════════════════════════════════════

// Training backend connection settings
const TRAINING_API_HOST = import.meta.env.VITE_TRAINING_API_HOST || window.location.hostname;
const TRAINING_API_PORT = import.meta.env.VITE_TRAINING_API_PORT || '8001';
const WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss' : 'ws';
const TRAINING_WS_URL = `${WS_PROTOCOL}://${TRAINING_API_HOST}:${TRAINING_API_PORT}/ws`;
const TRAINING_API_URL = `${window.location.protocol}//${TRAINING_API_HOST}:${TRAINING_API_PORT}/api/state`;

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

  // Chart configurations
  const lossChartData = {
    labels: state.training.train_loss.map((_, i) => i + 1),
    datasets: [
      {
        label: 'Train Loss',
        data: state.training.train_loss,
        borderColor: '#d4af37',
        backgroundColor: 'rgba(212, 175, 55, 0.08)',
        tension: 0.4,
        pointRadius: 0,
        fill: true,
        borderWidth: 2
      },
      {
        label: 'Val Loss',
        data: state.training.val_loss,
        borderColor: '#3ecf8e',
        backgroundColor: 'rgba(62, 207, 142, 0.08)',
        tension: 0.4,
        pointRadius: 0,
        fill: true,
        borderWidth: 2
      }
    ]
  };

  const tuneChartData = {
    labels: state.tuning.history.map(h => h.iteration),
    datasets: [{
      label: 'Score',
      data: state.tuning.history.map(h => h.score),
      borderColor: '#64b5f6',
      backgroundColor: 'rgba(100, 181, 246, 0.08)',
      tension: 0.4,
      pointRadius: 2,
      fill: true,
      borderWidth: 2
    }]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: '#6b7a94',
          font: { family: 'DM Sans', size: 11 }
        },
        position: 'top'
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.03)', drawBorder: false },
        ticks: { color: '#6b7a94', font: { family: 'DM Sans', size: 10 } }
      },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.03)', drawBorder: false },
        ticks: { color: '#6b7a94', font: { family: 'JetBrains Mono', size: 10 } }
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
      'complete': `Training complete! ${Object.keys(state.symbol_metrics).length} assets trained`,
      'error': `Error: ${state.error}`
    };
    return statusMap[state.status] || state.status;
  };

  return (
    <div className="space-y-5">
      {/* ═══════════════════════════════════════════════════════════════════
          CONTROL PANEL
      ═══════════════════════════════════════════════════════════════════ */}
      <div className="card card-gold">
        <div className="card-header">
          <div className={`status-dot ${connected ? 'success pulse' : 'danger'}`} />
          <span className="card-title text-gold">Model Training</span>
          <span className="text-muted text-sm ml-2">LightGBM Multi-Horizon</span>
          <div className="flex-1" />
          <div className={`badge ${connected ? 'badge-success' : 'badge-danger'}`}>
            {connected ? 'Connected' : 'Disconnected'}
          </div>
        </div>

        <div className="card-body">
          {!connected && (
            <div className="mb-5 p-4 rounded-xl bg-[rgba(212,175,55,0.05)] border border-[rgba(212,175,55,0.15)]">
              <div className="text-gold font-medium mb-1">Training Server Offline</div>
              <div className="text-muted text-sm font-mono">python train_with_dashboard.py</div>
            </div>
          )}

          {/* Controls */}
          <div className="flex items-end gap-5 flex-wrap mb-5">
            <div>
              <label className="text-muted text-xs uppercase tracking-wider block mb-2">Total Days</label>
              <input
                type="number"
                min="1"
                value={days}
                onChange={(e) => setDays(e.target.value === '' ? '' : parseInt(e.target.value))}
                onBlur={(e) => setDays(parseInt(e.target.value) || 90)}
                disabled={isTraining}
                className="input w-28"
              />
            </div>
            <div>
              <label className="text-gold text-xs uppercase tracking-wider block mb-2">Holdout Days</label>
              <input
                type="number"
                min="1"
                value={holdoutDays}
                onChange={(e) => setHoldoutDays(e.target.value === '' ? '' : parseInt(e.target.value))}
                onBlur={(e) => setHoldoutDays(parseInt(e.target.value) || 14)}
                disabled={isTraining}
                className="input input-gold w-24"
              />
            </div>
            <div>
              <label className="text-muted text-xs uppercase tracking-wider block mb-2">Tune Iterations</label>
              <input
                type="number"
                min="1"
                value={tuneIterations}
                onChange={(e) => setTuneIterations(e.target.value === '' ? '' : parseInt(e.target.value))}
                onBlur={(e) => setTuneIterations(parseInt(e.target.value) || 15)}
                disabled={isTraining}
                className="input w-28"
              />
            </div>
            <div className="flex items-center gap-3 pb-2">
              <input
                type="checkbox"
                checked={enableTuning}
                onChange={(e) => setEnableTuning(e.target.checked)}
                disabled={isTraining}
                className="w-4 h-4 rounded"
                style={{ accentColor: '#d4af37' }}
              />
              <label className="text-secondary text-sm">Hyperparameter Tuning</label>
            </div>
            <button
              onClick={startTraining}
              disabled={!connected || isTraining}
              className="btn btn-primary"
            >
              {isTraining ? 'Training...' : 'Start Training'}
            </button>
          </div>

          {/* Data Split Info */}
          <div className="flex items-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-[#d4af37]" />
              <span className="text-muted">Training:</span>
              <span className="text-gold font-mono">{days - holdoutDays} days</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-[#3ecf8e]" />
              <span className="text-muted">Holdout:</span>
              <span className="text-success font-mono">{holdoutDays} days</span>
            </div>
            <span className="text-muted">•</span>
            <span className="text-muted italic">Holdout = true out-of-sample test</span>
          </div>
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════
          PROGRESS
      ═══════════════════════════════════════════════════════════════════ */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">Progress</span>
          <div className="flex-1" />
          <span className={`font-display text-2xl ${state.status === 'error' ? 'text-danger' : state.status === 'complete' ? 'text-success' : 'text-gold'}`}>
            {state.progress}%
          </span>
        </div>
        <div className="card-body">
          <div className={`text-sm mb-4 ${state.status === 'error' ? 'text-danger' : 'text-secondary'}`}>
            {getStatusText()}
          </div>

          <div className="progress mb-5">
            <div
              className={`progress-fill ${
                state.status === 'complete' ? 'success' :
                state.status === 'error' ? 'danger' : ''
              }`}
              style={{ width: `${state.progress}%` }}
            />
          </div>

          {/* Symbol badges */}
          {state.symbols && state.symbols.length > 0 && (
            <div className="mb-4">
              <div className="text-xs text-muted uppercase tracking-wider mb-3">Assets</div>
              <div className="flex flex-wrap gap-2">
                {state.symbols.map((sym) => {
                  const done = state.symbol_metrics && state.symbol_metrics[sym];
                  const current = sym === state.current_symbol;
                  return (
                    <span
                      key={sym}
                      className={`px-3 py-1.5 rounded-lg text-sm font-medium ${
                        done ? 'bg-[rgba(62,207,142,0.1)] text-success border border-[rgba(62,207,142,0.2)]' :
                        current ? 'bg-[rgba(212,175,55,0.1)] text-gold border border-[rgba(212,175,55,0.2)]' :
                        'bg-[rgba(255,255,255,0.02)] text-muted border border-[rgba(255,255,255,0.06)]'
                      }`}
                    >
                      {sym}
                    </span>
                  );
                })}
              </div>
            </div>
          )}

          {/* Horizon badges */}
          <div className="flex gap-2 items-center">
            <span className="text-xs text-muted uppercase tracking-wider mr-2">Horizons:</span>
            {state.horizons.map(h => {
              const done = state.metrics[h];
              const current = h === state.current_horizon;
              return (
                <span
                  key={h}
                  className={`px-3 py-1.5 rounded-lg text-sm font-mono ${
                    done ? 'bg-[rgba(62,207,142,0.1)] text-success' :
                    current ? 'bg-[rgba(212,175,55,0.1)] text-gold' :
                    'bg-[rgba(255,255,255,0.02)] text-muted'
                  }`}
                >
                  {h}min
                </span>
              );
            })}
          </div>
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════
          MAIN GRID
      ═══════════════════════════════════════════════════════════════════ */}
      <div className="grid grid-cols-3 gap-5">
        {/* LEFT COLUMN */}
        <div className="space-y-5">
          {/* Data Info */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Training Data</span>
            </div>
            <div className="card-body space-y-3">
              <div className="flex justify-between">
                <span className="text-muted">Samples</span>
                <span className="font-mono">{state.data.rows.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted">Days</span>
                <span className="font-mono">{state.data.days}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted">Features</span>
                <span className="font-mono">{state.data.features}</span>
              </div>
            </div>
          </div>

          {/* Tuning Progress */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Hyperparameter Tuning</span>
            </div>
            <div className="card-body space-y-3">
              <div className="flex justify-between">
                <span className="text-muted">Iteration</span>
                <span className="font-mono">{state.tuning.iteration}/{state.tuning.total}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted">Best Score</span>
                <span className="font-mono text-success">{state.tuning.best_score || '--'}</span>
              </div>
              <div className="progress">
                <div
                  className="progress-fill"
                  style={{ width: `${state.tuning.total > 0 ? (state.tuning.iteration / state.tuning.total * 100) : 0}%` }}
                />
              </div>
            </div>
          </div>

          {/* Walk-Forward Validation */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Walk-Forward Validation</span>
            </div>
            <div className="card-body">
              {state.walk_forward.results.length > 0 ? (
                <div className="space-y-2">
                  {state.walk_forward.results.map(r => (
                    <div key={r.fold} className="flex justify-between p-3 rounded-xl bg-[rgba(255,255,255,0.02)]">
                      <span className="text-muted">Fold {r.fold}</span>
                      <span className="font-mono text-success">{r.accuracy}%</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-6 text-muted text-sm">
                  No results yet
                </div>
              )}
            </div>
          </div>
        </div>

        {/* CENTER COLUMN */}
        <div className="space-y-5">
          {/* Loss Chart */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Training Loss</span>
            </div>
            <div className="card-body h-52 chart-container">
              {state.training.train_loss.length > 0 ? (
                <Line data={lossChartData} options={chartOptions} />
              ) : (
                <div className="h-full flex items-center justify-center text-muted text-sm">
                  Loss curves will appear here
                </div>
              )}
            </div>
          </div>

          {/* Tuning History Chart */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Tuning History</span>
            </div>
            <div className="card-body h-44 chart-container">
              {state.tuning.history.length > 0 ? (
                <Line data={tuneChartData} options={chartOptions} />
              ) : (
                <div className="h-full flex items-center justify-center text-muted text-sm">
                  Tuning scores will appear here
                </div>
              )}
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN */}
        <div className="space-y-5">
          {/* Training Metrics */}
          <div className="card max-h-72 overflow-y-auto">
            <div className="card-header">
              <span className="card-title">Training Metrics</span>
              <span className="text-xs text-muted ml-2">(may be optimistic)</span>
            </div>
            <div className="card-body">
              {Object.keys(state.symbol_metrics || {}).length > 0 ? (
                Object.entries(state.symbol_metrics).map(([symbol, symMetrics]) => (
                  <div key={symbol} className="mb-4 p-3 rounded-xl bg-[rgba(255,255,255,0.02)] border border-[rgba(255,255,255,0.04)]">
                    <div className="text-gold font-semibold mb-2">{symbol}</div>
                    {Object.entries(symMetrics).map(([horizon, m]) => (
                      <div key={horizon} className="ml-2 mb-2 p-2 bg-[rgba(255,255,255,0.02)] rounded-lg">
                        <div className="text-info text-xs mb-1 font-medium">{horizon}-min</div>
                        <div className="flex gap-4 text-xs">
                          <span className="text-muted">Acc: <span className="text-secondary">{m.accuracy}%</span></span>
                          <span className="text-muted">Buy: <span className="text-success">{m.buy_precision}%</span></span>
                        </div>
                      </div>
                    ))}
                  </div>
                ))
              ) : Object.keys(state.metrics).length > 0 ? (
                Object.entries(state.metrics).map(([horizon, m]) => (
                  <div key={horizon} className="p-3 rounded-xl bg-[rgba(255,255,255,0.02)] mb-3">
                    <div className="text-info font-medium mb-2">
                      {state.current_symbol && `${state.current_symbol} · `}{horizon}-min model
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-sm">
                      <div>
                        <span className="text-muted block text-xs">Accuracy</span>
                        <span className="font-mono">{m.accuracy}%</span>
                      </div>
                      <div>
                        <span className="text-muted block text-xs">Confident</span>
                        <span className="font-mono text-success">{m.confident_accuracy}%</span>
                      </div>
                      <div>
                        <span className="text-muted block text-xs">Buy Prec</span>
                        <span className="font-mono text-gold">{m.buy_precision}%</span>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-6 text-muted text-sm">
                  Metrics appear after training
                </div>
              )}
            </div>
          </div>

          {/* Holdout Metrics */}
          <div className="card card-gold max-h-72 overflow-y-auto">
            <div className="card-header">
              <span className="card-title text-gold">Holdout Metrics</span>
              <span className="text-xs text-muted ml-2">(true performance)</span>
            </div>
            <div className="card-body">
              {Object.keys(state.holdout_metrics || {}).length > 0 ? (
                Object.entries(state.holdout_metrics).map(([symbol, symMetrics]) => (
                  <div key={symbol} className="mb-4 p-3 rounded-xl bg-[rgba(212,175,55,0.03)] border border-[rgba(212,175,55,0.1)]">
                    <div className="text-gold font-semibold mb-2">{symbol}</div>
                    {Object.entries(symMetrics).map(([horizon, m]) => (
                      <div key={horizon} className="ml-2 mb-2 p-2 bg-[rgba(255,255,255,0.02)] rounded-lg">
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-gold text-xs font-medium">{horizon}-min</span>
                          {m.optimal_threshold && (
                            <span className="text-xs text-muted">@{m.optimal_threshold}</span>
                          )}
                        </div>
                        <div className="grid grid-cols-2 gap-1 text-xs">
                          <span className="text-muted">Prec: <span className={`font-mono ${m.buy_precision > 50 ? 'text-success' : m.buy_precision > 30 ? 'text-gold' : 'text-danger'}`}>{m.buy_precision}%</span></span>
                          <span className="text-muted">Rec: <span className="text-gold font-mono">{m.recall || 0}%</span></span>
                          <span className="text-muted">Sig/Day: <span className="text-info font-mono">{m.signals_per_day || 0}</span></span>
                          <span className="text-muted">Acc: <span className="font-mono">{m.accuracy}%</span></span>
                        </div>
                      </div>
                    ))}
                  </div>
                ))
              ) : (
                <div className="text-center py-6 text-muted text-sm">
                  <p>Holdout metrics appear</p>
                  <p>after training complete</p>
                  <p className="text-xs italic mt-2">True out-of-sample performance</p>
                </div>
              )}
            </div>
          </div>

          {/* Feature Importance */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Feature Importance</span>
            </div>
            <div className="card-body">
              {(() => {
                const horizon = state.current_horizon || Object.keys(state.feature_importance)[0];
                const importance = state.feature_importance[horizon] || {};
                const sorted = Object.entries(importance).sort((a, b) => b[1] - a[1]).slice(0, 6);

                return sorted.length > 0 ? sorted.map(([feature, weight]) => (
                  <div key={feature} className="mb-3">
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-muted">{feature}</span>
                      <span className="font-mono">{(weight * 100).toFixed(1)}%</span>
                    </div>
                    <div className="progress h-1.5">
                      <div
                        className="progress-fill"
                        style={{ width: `${weight * 100}%` }}
                      />
                    </div>
                  </div>
                )) : (
                  <div className="text-center py-6 text-muted text-sm">
                    Feature weights appear after training
                  </div>
                );
              })()}
            </div>
          </div>

          {/* Logs */}
          <div className="card" style={{ maxHeight: '200px' }}>
            <div className="card-header">
              <span className="card-title">Activity</span>
            </div>
            <div className="card-body h-full overflow-hidden">
              <div className="log-panel h-full overflow-y-auto text-xs">
                {state.logs.slice(-15).map((log, i) => (
                  <div key={i} className="log-entry py-1">
                    <span className="text-muted">{log}</span>
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default TrainingDashboard;
