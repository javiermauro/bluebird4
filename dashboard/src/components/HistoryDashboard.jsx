import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';

// Backend connection settings - LIVE INSTANCE (Port 8001)
const API_HOST = import.meta.env.VITE_API_HOST || window.location.hostname;
const API_PORT = import.meta.env.VITE_API_PORT || '8001';  // LIVE default
const API_BASE = `${window.location.protocol}//${API_HOST}:${API_PORT}`;

function HistoryDashboard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  // State for equity curve
  const [equity, setEquity] = useState({
    dates: [],
    equity: [],
    peak: 0,
    peak_date: null,
    trough: 0,
    trough_date: null,
    current: 0,
    starting: 100000,
    recovery_pct: 0,
    total_return_pct: 0,
    current_fee_tier: null,
  });

  // State for profitability report (fees, net P&L)
  const [profitability, setProfitability] = useState({
    fees: { expected: 0, conservative: 0, uncertain_count: 0 },
    pnl: { gross: 0, net_expected: 0, net_conservative: 0 },
    current_tier: 'Tier 1',
    current_rates: { maker: 0.0015, taker: 0.0025 },
    rolling_30d_volume: 0,
    tier_progression: null,
  });

  // Paper trading warning dismissal
  const [showWarning, setShowWarning] = useState(true);

  // State for realized P/L
  const [pnl, setPnl] = useState({
    total: 0,
    by_day: [],
    by_symbol: {},
    metrics: {
      win_rate: 0,
      total_trades: 0,
      total_orders: 0,
      avg_per_symbol: 0,
      best_day: { date: null, pnl: 0 },
      worst_day: { date: null, pnl: 0 }
    }
  });

  // State for trade history
  const [trades, setTrades] = useState({
    trades: [],
    total: 0
  });

  // Filter state
  const [symbolFilter, setSymbolFilter] = useState('all');
  const [periodFilter, setPeriodFilter] = useState('1M');

  // Fetch all history data
  const fetchHistoryData = async () => {
    try {
      setLoading(true);

      const [equityRes, pnlRes, tradesRes, profitRes] = await Promise.all([
        fetch(`${API_BASE}/api/history/equity?period=${periodFilter}`),
        fetch(`${API_BASE}/api/history/realized-pnl?days=30`),
        fetch(`${API_BASE}/api/history/trades?days=7&limit=100`),
        fetch(`${API_BASE}/api/profitability-report`)
      ]);

      const equityData = await equityRes.json();
      const pnlData = await pnlRes.json();
      const tradesData = await tradesRes.json();
      const profitData = await profitRes.json();

      if (!equityData.error) setEquity(equityData);
      if (!pnlData.error) setPnl(pnlData);
      if (!tradesData.error) setTrades(tradesData);
      if (!profitData.error) setProfitability(profitData);

      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistoryData();
    const interval = setInterval(fetchHistoryData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, [periodFilter]);

  // Format currency
  const formatCurrency = (value, decimals = 2) => {
    if (value === null || value === undefined) return '$0.00';
    const sign = value >= 0 ? '' : '-';
    return `${sign}$${Math.abs(value).toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`;
  };

  // Format percentage
  const formatPercent = (value) => {
    if (value === null || value === undefined) return '0.0%';
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(1)}%`;
  };

  // Chart configuration for equity curve (3 series: Gross, Net Expected, Net Conservative)
  const equityChartData = {
    labels: equity.dates,
    datasets: [
      {
        label: 'Gross (Alpaca)',
        data: equity.equity,
        borderColor: '#d4af37',
        backgroundColor: 'rgba(212, 175, 55, 0.1)',
        tension: 0.4,
        pointRadius: 0,
        fill: false,
        borderWidth: 2.5
      },
      {
        label: 'Net (Est.)',
        data: equity.equity_fee_adjusted || equity.equity,
        borderColor: '#50c878',
        backgroundColor: 'transparent',
        tension: 0.4,
        pointRadius: 0,
        fill: false,
        borderWidth: 1.5,
        borderDash: [5, 5]
      },
      {
        label: 'Net (Worst)',
        data: equity.equity_fee_adjusted_conservative || equity.equity,
        borderColor: '#e8a87c',
        backgroundColor: 'transparent',
        tension: 0.4,
        pointRadius: 0,
        fill: false,
        borderWidth: 1.5,
        borderDash: [2, 3]
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top',
        align: 'end',
        labels: {
          color: '#6b7a94',
          font: { size: 11 },
          boxWidth: 20,
          padding: 15,
          usePointStyle: true,
          pointStyle: 'line'
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(26, 26, 46, 0.95)',
        titleColor: '#d4af37',
        bodyColor: '#e8e6e3',
        borderColor: 'rgba(212, 175, 55, 0.3)',
        borderWidth: 1,
        callbacks: {
          label: function(context) {
            const label = context.dataset.label || '';
            const value = context.parsed.y;
            return `${label}: $${value.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
          }
        }
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(212, 175, 55, 0.05)' },
        ticks: {
          color: '#6b7a94',
          maxRotation: 0,
          maxTicksLimit: 8
        }
      },
      y: {
        grid: { color: 'rgba(212, 175, 55, 0.05)' },
        ticks: {
          color: '#6b7a94',
          callback: (value) => '$' + (value / 1000).toFixed(0) + 'k'
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  };

  // Get symbols for filter
  const symbols = ['all', ...Object.keys(pnl.by_symbol)];

  // Filter trades by symbol
  const filteredTrades = symbolFilter === 'all'
    ? trades.trades
    : trades.trades.filter(t => t.symbol === symbolFilter);

  return (
    <div className="history-dashboard">
      {/* Header */}
      <div className="history-header">
        <div className="history-title">
          <h2>Account History</h2>
          <span className="history-subtitle">Realized P/L & Performance Tracking</span>
        </div>
        <div className="history-controls">
          <select
            value={periodFilter}
            onChange={(e) => setPeriodFilter(e.target.value)}
            className="history-select"
          >
            <option value="1W">1 Week</option>
            <option value="1M">1 Month</option>
            <option value="3M">3 Months</option>
            <option value="1A">1 Year</option>
          </select>
          <button onClick={fetchHistoryData} className="btn btn-secondary btn-sm">
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>

      {error && (
        <div className="history-error">
          <span>Error loading data: {error}</span>
        </div>
      )}

      {/* Paper Trading Warning Banner */}
      {showWarning && (
        <div className="paper-trading-warning">
          <div className="warning-content">
            <span className="warning-icon">&#9888;</span>
            <span className="warning-text">
              <strong>Paper Trading:</strong> Equity shown does not reflect real trading costs.
              Actual results may differ due to fees, slippage, and order queue effects.
            </span>
            <a
              href="https://docs.alpaca.markets/docs/paper-trading"
              target="_blank"
              rel="noopener noreferrer"
              className="warning-link"
            >
              Learn more
            </a>
          </div>
          <button
            className="warning-dismiss"
            onClick={() => setShowWarning(false)}
            title="Dismiss"
          >
            &times;
          </button>
        </div>
      )}

      {/* Fee Tier Info Card */}
      <div className="fee-tier-card">
        <div className="fee-tier-header">
          <span className="tier-badge">{profitability.current_tier || 'Tier 1'}</span>
          <div className="fee-rates">
            <span>Maker: {((profitability.current_rates?.maker || 0.0015) * 100).toFixed(2)}%</span>
            <span className="rate-divider">|</span>
            <span>Taker: {((profitability.current_rates?.taker || 0.0025) * 100).toFixed(2)}%</span>
          </div>
        </div>
        <div className="fee-tier-details">
          <div className="volume-info">
            <span className="volume-label">30d Volume:</span>
            <span className="volume-value">{formatCurrency(profitability.rolling_30d_volume || 0, 0)}</span>
          </div>
          {profitability.tier_progression && (
            <div className="tier-progress">
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${profitability.tier_progression.progress_pct || 0}%` }}
                />
              </div>
              <span className="progress-label">
                {formatCurrency(profitability.tier_progression.volume_needed || 0, 0)} to next tier
              </span>
            </div>
          )}
        </div>
        <div className="estimated-fees">
          <div className="fee-row">
            <span className="fee-label">Est. Fees (since Dec 1):</span>
            <span className="fee-value negative">-{formatCurrency(profitability.fees?.expected || 0)}</span>
          </div>
          <div className="fee-row conservative">
            <span className="fee-label">Conservative:</span>
            <span className="fee-value negative">-{formatCurrency(profitability.fees?.conservative || 0)}</span>
          </div>
          {profitability.fees?.uncertain_count > 0 && (
            <div className="fee-uncertain">
              <span className="uncertain-icon">?</span>
              <span>{profitability.fees.uncertain_count} orders with uncertain maker/taker classification</span>
            </div>
          )}
        </div>
      </div>

      {/* Key Metrics Row */}
      <div className="history-metrics-grid">
        <div className="history-metric-card">
          <div className="metric-label">GROSS P/L</div>
          <div className={`metric-value ${profitability.pnl?.gross >= 0 ? 'positive' : 'negative'}`}>
            {formatCurrency(profitability.pnl?.gross || 0)}
          </div>
          <div className="metric-sublabel">Alpaca (before fees)</div>
        </div>

        <div className="history-metric-card">
          <div className="metric-label">NET P/L (EST.)</div>
          <div className={`metric-value ${profitability.pnl?.net_expected >= 0 ? 'positive' : 'negative'}`}>
            {formatCurrency(profitability.pnl?.net_expected || 0)}
          </div>
          <div className="metric-sublabel">After {formatCurrency(profitability.fees?.expected || 0)} fees</div>
        </div>

        <div className="history-metric-card net-conservative">
          <div className="metric-label">NET (CONSERVATIVE)</div>
          <div className={`metric-value ${profitability.pnl?.net_conservative >= 0 ? 'positive' : 'negative'}`}>
            {formatCurrency(profitability.pnl?.net_conservative || 0)}
          </div>
          <div className="metric-sublabel">Worst-case fees</div>
        </div>

        <div className="history-metric-card">
          <div className="metric-label">RECOVERY</div>
          <div className="metric-value recovery">
            {equity.recovery_pct.toFixed(1)}%
          </div>
          <div className="metric-sublabel">
            ${(equity.starting - equity.current).toLocaleString()} to break-even
          </div>
        </div>
      </div>

      {/* Equity Curve Chart */}
      <div className="history-section">
        <div className="section-header">
          <h3>Equity Curve</h3>
          <div className="equity-stats">
            <span className="stat">
              Peak: {formatCurrency(equity.peak)}
              {equity.peak_date && <small> ({equity.peak_date.slice(0,10)})</small>}
            </span>
            <span className="stat">
              Trough: {formatCurrency(equity.trough)}
              {equity.trough_date && <small> ({equity.trough_date.slice(0,10)})</small>}
            </span>
            <span className="stat">
              Current: {formatCurrency(equity.current)}
            </span>
          </div>
        </div>
        <div className="equity-chart-container">
          {equity.equity.length > 0 ? (
            <Line data={equityChartData} options={chartOptions} />
          ) : (
            <div className="no-data">No equity data available</div>
          )}
        </div>
      </div>

      {/* Recovery Progress Bar */}
      <div className="history-section recovery-section">
        <div className="section-header">
          <h3>Recovery Progress</h3>
          <span className="recovery-target">Target: $100,000</span>
        </div>
        <div className="recovery-bar-container">
          <div className="recovery-bar">
            <div
              className="recovery-fill"
              style={{ width: `${Math.min(100, Math.max(0, equity.recovery_pct))}%` }}
            />
            <div className="recovery-markers">
              <span className="marker trough">${(equity.trough / 1000).toFixed(1)}k</span>
              <span className="marker current">${(equity.current / 1000).toFixed(1)}k</span>
              <span className="marker target">$100k</span>
            </div>
          </div>
          <div className="recovery-labels">
            <span>Lowest: {equity.trough_date ? equity.trough_date.slice(0,10) : '-'}</span>
            <span className="recovery-pct">{equity.recovery_pct.toFixed(1)}% Recovered</span>
            <span>Break-even</span>
          </div>
        </div>
      </div>

      {/* Two Column Layout: Daily P/L and Symbol Performance */}
      <div className="history-two-column">
        {/* Daily Activity */}
        <div className="history-section">
          <div className="section-header">
            <h3>Daily Activity</h3>
            <span className="section-subtitle">{pnl.by_day.length} days</span>
          </div>
          <div className="daily-pnl-table">
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Trades</th>
                  <th>Buys</th>
                  <th>Sells</th>
                </tr>
              </thead>
              <tbody>
                {pnl.by_day.slice(0, 10).map((day, idx) => (
                  <tr key={idx}>
                    <td className="date">{day.date}</td>
                    <td className="trades">{day.trades}</td>
                    <td className="buys">{day.buys}</td>
                    <td className="sells">{day.sells}</td>
                  </tr>
                ))}
                {pnl.by_day.length === 0 && (
                  <tr>
                    <td colSpan="4" className="no-data">No trading activity</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Symbol Performance */}
        <div className="history-section">
          <div className="section-header">
            <h3>By Symbol</h3>
            <span className="section-subtitle">Realized P/L</span>
          </div>
          <div className="symbol-performance">
            {Object.entries(pnl.by_symbol)
              .sort((a, b) => b[1].pnl - a[1].pnl)
              .map(([symbol, data]) => (
                <div key={symbol} className="symbol-row">
                  <div className="symbol-info">
                    <span className="symbol-name">{symbol}</span>
                    <span className="symbol-trades">{data.trades} trades</span>
                  </div>
                  <div className="symbol-pnl-bar">
                    <div
                      className={`pnl-fill ${data.pnl >= 0 ? 'positive' : 'negative'}`}
                      style={{
                        width: `${Math.min(100, Math.abs(data.pnl) / Math.max(...Object.values(pnl.by_symbol).map(s => Math.abs(s.pnl))) * 100)}%`
                      }}
                    />
                  </div>
                  <div className={`symbol-pnl ${data.pnl >= 0 ? 'positive' : 'negative'}`}>
                    {formatCurrency(data.pnl)}
                  </div>
                </div>
              ))}
            {Object.keys(pnl.by_symbol).length === 0 && (
              <div className="no-data">No symbol data</div>
            )}
          </div>
        </div>
      </div>

      {/* Trade History Table */}
      <div className="history-section trade-history-section">
        <div className="section-header">
          <h3>Trade History</h3>
          <div className="trade-filters">
            <select
              value={symbolFilter}
              onChange={(e) => setSymbolFilter(e.target.value)}
              className="history-select"
            >
              {symbols.map(s => (
                <option key={s} value={s}>{s === 'all' ? 'All Symbols' : s}</option>
              ))}
            </select>
            <span className="trade-count">{filteredTrades.length} trades</span>
          </div>
        </div>
        <div className="trade-history-table">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Symbol</th>
                <th>Side</th>
                <th>Quantity</th>
                <th>Price</th>
                <th>Value</th>
                <th>P/L</th>
              </tr>
            </thead>
            <tbody>
              {filteredTrades.slice(0, 50).map((trade, idx) => (
                <tr key={idx} className={trade.side}>
                  <td className="time">
                    <span className="date">{trade.date}</span>
                    <span className="clock">{trade.time}</span>
                  </td>
                  <td className="symbol">{trade.symbol}</td>
                  <td className={`side ${trade.side}`}>
                    {trade.side.toUpperCase()}
                  </td>
                  <td className="qty">{trade.qty.toFixed(6)}</td>
                  <td className="price">{formatCurrency(trade.price)}</td>
                  <td className="value">{formatCurrency(trade.value)}</td>
                  <td className={`pnl ${trade.pnl !== null ? (trade.pnl >= 0 ? 'positive' : 'negative') : ''}`}>
                    {trade.pnl !== null ? formatCurrency(trade.pnl) : '-'}
                  </td>
                </tr>
              ))}
              {filteredTrades.length === 0 && (
                <tr>
                  <td colSpan="7" className="no-data">No trades found</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Footer */}
      <div className="history-footer">
        <span>Last updated: {lastUpdate ? lastUpdate.toLocaleTimeString() : '-'}</span>
        <span>Data from Alpaca API</span>
      </div>
    </div>
  );
}

export default HistoryDashboard;
